"""
Nigerian ANPR pipeline, following the NLPDRS approach
(https://github.com/esssyjr/NLPDRS-Nierian-License-Plate-Detection-and-Recognition-System-):

  YOLOv8 segmentation model trained on Nigerian plates (annotated to segment
  just the plate-number portion) -> crop the first detected box -> EasyOCR
  on the crop -> concatenate the recognized text and strip spaces.

The model runs on the full camera frame directly — no vehicle-ROI staging,
no crop-quality gating and no preprocessing variants. On top of NLPDRS's
single-image technique this pipeline adds only a thin multi-frame majority
vote (the gate captures a short burst, and the UI reports how many frames
agreed) and the storage normalization the rest of the app has always used.
"""
from __future__ import annotations

import os
import time
from collections import Counter
from typing import Callable, List, Optional, Tuple

from .models import PlateDetection, PlateOCRCandidate, PlateRecognitionResult, RecognitionStatus
from .detector import PlateDetector, PlateModelError
from .recognizer import OCRBackend, get_ocr_backend, PLATE_ALLOWLIST
from . import postprocessing as post

RegionProvider = Callable[["object"], List[Tuple["object", Tuple[int, int], str]]]


def default_region_provider(frame_bgr):
    return [(frame_bgr, (0, 0), "full_frame")]


def _upscale_for_ocr(crop, min_width: int = 250, scale: float = 2.0):
    """
    EasyOCR drops or confuses characters on small plate crops (a plate at
    gate distance is ~100-150px wide even at 720p). A 2x cubic upscale
    measurably recovers dropped trailing characters; larger factors start
    splitting the plate into out-of-order boxes, so this is deliberately
    capped at 2x and skipped for crops that are already big.
    """
    try:
        import cv2
        if crop is not None and crop.size > 0 and crop.shape[1] < min_width:
            return cv2.resize(crop, None, fx=scale, fy=scale,
                              interpolation=cv2.INTER_CUBIC)
    except Exception:
        pass
    return crop


def extract_plate_number(ocr_results: List[Tuple[str, float]]) -> str:
    """NLPDRS text assembly: concatenate every OCR hit, then strip spaces."""
    plate_number = ""
    for text, _conf in ocr_results:
        plate_number += text + " "
    return plate_number.strip().replace(" ", "")


class PlateRecognitionPipeline:
    """
    Loaded once per process and reused — the YOLO model and OCR backend are
    each singletons behind their own lock, so concurrent gate requests never
    trigger a reload.
    """

    def __init__(self, detector: PlateDetector, ocr: OCRBackend, config: dict,
                 debug_root: str = "gate_photos/debug",
                 fallback_detector: Optional[PlateDetector] = None):
        self.detector = detector
        self.ocr = ocr
        self.config = config
        self.debug_root = debug_root
        # Second detector tried per-frame only when the primary's crop yields
        # no plausible text. Measured on gate photos: the general-purpose
        # yolov8n model's taller crops OCR better at production resolution
        # (it recovers the M in SMK that the tight crop loses), while the
        # Nigerian-trained segmentation model's tight number-line crop reads
        # better on small/distant plates — so the two cover for each other.
        self.fallback_detector = fallback_detector

    # ── config helpers ────────────────────────────────────────────────────
    def _cfg(self, key, default):
        return self.config.get(key, default)

    def status(self) -> dict:
        d = self.detector.status()
        d["ocr_backend"] = self.ocr.name
        d["ocr_available"] = self.ocr.is_available()
        return d

    # ── main entry point ─────────────────────────────────────────────────
    def process_frames(
        self,
        frames: List["object"],
        region_provider: Optional[RegionProvider] = None,
        debug_session_id: Optional[str] = None,
    ) -> PlateRecognitionResult:
        # region_provider is accepted for API compatibility but unused — the
        # NLPDRS segmentation model runs on the full frame directly.
        if not frames:
            return PlateRecognitionResult(
                status=RecognitionStatus.NOT_DETECTED.value,
                total_frames=0,
                rejection_reason="No frames captured",
            )

        try:
            self.detector.validate()
        except PlateModelError as e:
            return PlateRecognitionResult(
                status=RecognitionStatus.NOT_DETECTED.value,
                total_frames=len(frames),
                rejection_reason=str(e),
            )

        consensus_frames = self._cfg("plate_consensus_frames", 3)
        min_confirm_conf = self._cfg("plate_min_confirm_confidence", 0.55)
        debug = bool(self._cfg("plate_debug_mode", False)) and debug_session_id

        debug_dir = None
        if debug:
            debug_dir = os.path.join(self.debug_root, f"{int(time.time())}_{debug_session_id}")
            os.makedirs(debug_dir, exist_ok=True)

        # (normalized_text, ocr_confidence, PlateDetection) per frame that read
        frame_reads: List[Tuple[str, float, PlateDetection]] = []
        any_detection = False
        best_det: Optional[PlateDetection] = None

        detectors = [self.detector]
        if self.fallback_detector is not None and self.fallback_detector.is_ready():
            detectors.append(self.fallback_detector)

        def read_with(detector, frame, idx):
            """One detector's read of one frame. Returns
            (normalized, ocr_conf, det, structure_matched) or None."""
            nonlocal any_detection, best_det
            detections = detector.detect(frame, frame_index=idx, source="full_frame")
            if not detections:
                return None
            any_detection = True
            det = detections[0]  # NLPDRS: the first (highest-confidence) box
            if best_det is None or det.detector_confidence > best_det.detector_confidence:
                best_det = det
            if debug_dir is not None:
                self._save_debug_crop(debug_dir, idx, f"crop_{os.path.basename(detector.model_path)}", det.crop)

            # Each OCR backend declares the crop context it was trained on:
            # plate-specific models need surround padding, scene-text OCR
            # needs upscaling of small crops (see OCRBackend attributes).
            crop = det.crop
            pad_frac = getattr(self.ocr, "crop_pad_fraction", 0.0)
            if pad_frac > 0:
                x1, y1, x2, y2 = det.bounding_box
                pad = int((y2 - y1) * pad_frac)
                h, w = frame.shape[:2]
                crop = frame[max(0, y1 - pad):min(h, y2 + pad),
                             max(0, x1 - pad):min(w, x2 + pad)]
            if getattr(self.ocr, "wants_upscale", True):
                crop = _upscale_for_ocr(crop)

            ocr_hits = self.ocr.recognize(crop, allowlist=None)
            raw_text = extract_plate_number(ocr_hits)
            normalized = post.normalize_plate_for_storage(raw_text)
            # Plausibility guard: Nigerian plates are 7-8 characters mixing
            # letters and digits; UI overlay words and partial reads must
            # never reach the trip database as a plate.
            if (not normalized or len(normalized) < 5 or len(normalized) > 10
                    or post.is_overlay_word(normalized)
                    or not post.has_letters_and_digits(normalized)):
                return None
            # Strip junk read from outside the plate number (e.g. state-strip
            # text appended after the real plate: "SMK322EAON" -> "SMK322EA").
            normalized = post.trim_to_structure(normalized)
            confs = [c for _, c in ocr_hits]
            ocr_conf = sum(confs) / len(confs) if confs else 0.0
            # Backends with calibrated confidence (fast_plate_ocr) set a
            # floor; a read below it is noise, not a candidate.
            if ocr_conf < getattr(self.ocr, "min_candidate_confidence", 0.0):
                return None
            # Structure-guided cleanup (e.g. "ABC3928L" -> "ABC392BL"): only
            # applied when the corrected text exactly matches a known plate
            # structure, and each correction costs confidence so a heavily
            # corrected read can't outrank a clean one.
            corr = post.safe_correct(normalized)
            if not corr.rejected and corr.corrections:
                normalized = corr.text
                ocr_conf = max(0.0, ocr_conf - post.CORRECTION_CONFIDENCE_PENALTY
                               * len(corr.corrections))
            return (normalized, ocr_conf, det,
                    post.matches_any_structure(normalized) is not None)

        for idx, frame in enumerate(frames):
            # First structure-valid read wins; a plausible but structure-less
            # read is kept only if no later detector produces a valid one.
            # The fallback detector's cost is only paid on frames where the
            # primary's crop failed to yield a structure-valid plate.
            chosen = None
            for detector in detectors:
                read = read_with(detector, frame, idx)
                if read is None:
                    continue
                if read[3]:
                    chosen = read
                    break
                if chosen is None:
                    chosen = read
            if chosen is not None:
                frame_reads.append(chosen[:3])

        if not any_detection:
            return PlateRecognitionResult(
                status=RecognitionStatus.NOT_DETECTED.value,
                total_frames=len(frames),
                rejection_reason="No plate detected in any frame",
            )
        if not frame_reads:
            return PlateRecognitionResult(
                status=RecognitionStatus.NOT_DETECTED.value,
                total_frames=len(frames),
                detector_confidence=best_det.detector_confidence if best_det else 0.0,
                bounding_box=best_det.bounding_box if best_det else None,
                rejection_reason="Plate detected but no plausible plate text was read",
            )

        # Majority vote across the burst.
        counts = Counter(text for text, _, _ in frame_reads)
        winner, agree_count = counts.most_common(1)[0]
        winning = [(t, c, d) for (t, c, d) in frame_reads if t == winner]
        ocr_confidence = sum(c for _, c, _ in winning) / len(winning)
        winner_best_det = max((d for _, _, d in winning), key=lambda d: d.detector_confidence)
        detector_confidence = winner_best_det.detector_confidence
        overall = (detector_confidence + ocr_confidence) / 2.0
        # EasyOCR's confidence is poorly calibrated on plate crops (correct
        # reads routinely score ~0.1-0.2), so an exact match against a known
        # Nigerian plate structure earns the same small bonus used elsewhere
        # in postprocessing — evidence, not proof.
        if post.matches_any_structure(winner):
            overall = min(1.0, overall + post.STRUCTURE_MATCH_BONUS)

        # consensus_frames is an absolute requirement: a burst that captured
        # fewer frames than it (e.g. the single-still-photo path) can auto-fill
        # the plate but still requires human confirmation. The exit page's
        # quick check explicitly runs with plate_consensus_frames=1.
        if agree_count >= consensus_frames and overall >= min_confirm_conf:
            status = RecognitionStatus.CONFIRMED.value
        else:
            status = RecognitionStatus.LOW_CONFIDENCE.value

        result = PlateRecognitionResult(
            plate_number=winner,
            display_plate=post.format_plate_for_display(winner),
            status=status,
            overall_confidence=overall,
            detector_confidence=detector_confidence,
            ocr_confidence=ocr_confidence,
            consensus_count=agree_count,
            total_frames=len(frames),
            bounding_box=winner_best_det.bounding_box,
        )
        if debug_dir is not None:
            result.debug_information["debug_dir"] = debug_dir
            result.debug_information["frame_reads"] = [
                {"text": t, "ocr_confidence": c, "frame_index": d.frame_index}
                for (t, c, d) in frame_reads
            ]
        return result

    @staticmethod
    def _save_debug_crop(debug_dir, frame_idx, tag, image):
        try:
            import cv2
            path = os.path.join(debug_dir, f"frame{frame_idx}_{tag}.jpg")
            cv2.imwrite(path, image)
        except Exception:
            pass


# ── Compatibility wrapper ────────────────────────────────────────────────
def detect_plate_string(result: PlateRecognitionResult) -> str:
    """For call sites that only ever wanted a bare string."""
    if result.status in (RecognitionStatus.CONFIRMED.value, RecognitionStatus.MANUAL.value):
        return result.plate_number
    return ""


def build_pipeline(config: dict) -> PlateRecognitionPipeline:
    detector = PlateDetector(
        model_path=config.get("plate_model_path", os.path.join("models", "nlpdrs_plate_segment.pt")),
        confidence=config.get("plate_detection_confidence", 0.35),
        iou=config.get("plate_detection_iou", 0.45),
    )
    ocr = get_ocr_backend(config.get("plate_ocr_backend", "easyocr"))
    fallback = None
    fallback_path = config.get("plate_fallback_model_path", "")
    if fallback_path and os.path.exists(fallback_path):
        fallback = PlateDetector(
            model_path=fallback_path,
            confidence=config.get("plate_detection_confidence", 0.35),
            iou=config.get("plate_detection_iou", 0.45),
        )
    return PlateRecognitionPipeline(detector, ocr, config, fallback_detector=fallback)
