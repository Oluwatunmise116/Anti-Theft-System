"""
Tests for the end-to-end anpr.PlateRecognitionPipeline (NLPDRS-style:
full-frame detect -> crop -> OCR -> concat/strip -> majority vote) using
fake detector/OCR components — no real model weights, camera, or network
access required. Only hardware/model interfaces are mocked; the
postprocessing and consensus logic run for real.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from anpr import PlateRecognitionPipeline
from anpr.detector import PlateModelError
from anpr.models import PlateDetection, RecognitionStatus
from anpr.recognizer import OCRBackend


def make_crop(width=160, height=50, fill=None):
    """A crop with real texture (so sharpness/contrast checks behave like a real photo)."""
    if fill is not None:
        img = np.full((height, width, 3), fill, dtype=np.uint8)
        return img
    rng = np.random.default_rng(42)
    return rng.integers(0, 255, size=(height, width, 3), dtype=np.uint8)


class FakeDetector:
    """Stands in for anpr.detector.PlateDetector."""

    def __init__(self, detections_by_frame=None, ready=True, error="model missing"):
        self.detections_by_frame = detections_by_frame or {}
        self._ready = ready
        self._error = error

    def validate(self):
        if not self._ready:
            raise PlateModelError(self._error)

    def is_ready(self):
        return self._ready

    def status(self):
        return {"model_present": self._ready}

    def detect(self, frame_bgr, frame_index=0, source="full_frame", offset=(0, 0)):
        if source == "vehicle_roi":
            return self.detections_by_frame.get(("roi", frame_index), [])
        return self.detections_by_frame.get(frame_index, [])


class FakeOCR(OCRBackend):
    name = "fake"

    def __init__(self, fixed_response=None):
        self.fixed_response = fixed_response if fixed_response is not None else []

    def is_available(self):
        return True

    def recognize(self, image, allowlist=None):
        return list(self.fixed_response)


def build_pipeline(detector, ocr, config=None):
    cfg = {
        "plate_min_width_pixels": 60,
        "plate_max_corrections": 2,
        "plate_consensus_frames": 3,
        "plate_min_confirm_confidence": 0.5,
        "plate_debug_mode": False,
    }
    cfg.update(config or {})
    return PlateRecognitionPipeline(detector, ocr, cfg)


def test_missing_model_returns_not_detected_with_clear_reason():
    pipeline = build_pipeline(
        FakeDetector(ready=False, error="Nigerian plate model not found at models/x.pt"),
        FakeOCR(),
    )
    frames = [make_crop(640, 480)]
    result = pipeline.process_frames(frames)
    assert result.status == RecognitionStatus.NOT_DETECTED.value
    assert "not found" in result.rejection_reason
    # Never leaks a filesystem traceback — just the clear message we raised.
    assert "Traceback" not in result.rejection_reason


def test_no_vehicle_detected_all_frames_empty():
    pipeline = build_pipeline(FakeDetector(detections_by_frame={}), FakeOCR())
    frames = [make_crop(640, 480) for _ in range(4)]
    result = pipeline.process_frames(frames)
    assert result.status == RecognitionStatus.NOT_DETECTED.value
    assert result.plate_number == ""


def test_implausible_ocr_text_is_rejected():
    crop = make_crop()
    dets = {i: [PlateDetection((0, 0, 160, 50), 0.9, 0, crop, i)] for i in range(3)}
    # A partial read ("201X") is too short to be a real plate and must not
    # be stored as one.
    pipeline = build_pipeline(FakeDetector(detections_by_frame=dets),
                               FakeOCR([("201X.", 0.9)]))
    result = pipeline.process_frames([make_crop() for _ in range(3)])
    assert result.status == RecognitionStatus.NOT_DETECTED.value
    assert result.plate_number == ""


def test_empty_ocr_output_is_not_detected():
    crop = make_crop()
    dets = {i: [PlateDetection((0, 0, 160, 50), 0.9, 0, crop, i)] for i in range(3)}
    pipeline = build_pipeline(FakeDetector(detections_by_frame=dets), FakeOCR([]))
    result = pipeline.process_frames([make_crop() for _ in range(3)])
    assert result.status == RecognitionStatus.NOT_DETECTED.value


def test_overlay_word_never_returned_as_plate():
    crop = make_crop()
    dets = {i: [PlateDetection((0, 0, 160, 50), 0.9, 0, crop, i)] for i in range(3)}
    pipeline = build_pipeline(
        FakeDetector(detections_by_frame=dets),
        FakeOCR([("GRANTED", 0.95), ("CAMERA", 0.9)]),
    )
    result = pipeline.process_frames([make_crop() for _ in range(3)])
    assert result.status == RecognitionStatus.NOT_DETECTED.value
    assert result.plate_number != "GRANTED"


def test_multiple_detections_in_one_frame_do_not_inflate_consensus_count():
    crop = make_crop()
    # Two plate-like boxes detected in the SAME frame, both reading the
    # same plate — this must still only count as one agreeing frame.
    dets = {
        i: [
            PlateDetection((0, 0, 160, 50), 0.9, 0, crop, i),
            PlateDetection((200, 0, 360, 50), 0.4, 0, crop, i),
        ]
        for i in range(3)
    }
    pipeline = build_pipeline(FakeDetector(detections_by_frame=dets),
                               FakeOCR([("KJA456GH", 0.9)]))
    result = pipeline.process_frames([make_crop() for _ in range(3)])
    assert result.status == RecognitionStatus.CONFIRMED.value
    assert result.consensus_count == 3   # not 6


def test_confirmed_requires_consensus_threshold():
    crop = make_crop()
    dets = {i: [PlateDetection((0, 0, 160, 50), 0.9, 0, crop, i)] for i in range(2)}
    pipeline = build_pipeline(
        FakeDetector(detections_by_frame=dets),
        FakeOCR([("KJA456GH", 0.9)]),
        config={"plate_consensus_frames": 3},
    )
    result = pipeline.process_frames([make_crop() for _ in range(2)])
    assert result.status == RecognitionStatus.LOW_CONFIDENCE.value


def test_vehicle_roi_used_first_then_falls_back_to_full_frame():
    crop = make_crop()
    # ROI detection finds nothing; full-frame fallback finds the plate.
    dets = {0: [PlateDetection((0, 0, 160, 50), 0.9, 0, crop, 0)],
            1: [PlateDetection((0, 0, 160, 50), 0.9, 0, crop, 1)],
            2: [PlateDetection((0, 0, 160, 50), 0.9, 0, crop, 2)]}
    pipeline = build_pipeline(FakeDetector(detections_by_frame=dets),
                               FakeOCR([("KJA456GH", 0.9)]))

    def region_provider(frame):
        return [(frame, (0, 0), "vehicle_roi"), (frame, (0, 0), "full_frame")]

    result = pipeline.process_frames([make_crop() for _ in range(3)],
                                      region_provider=region_provider)
    assert result.status == RecognitionStatus.CONFIRMED.value
    assert result.plate_number == "KJA456GH"


def test_no_frames_captured_is_not_detected():
    pipeline = build_pipeline(FakeDetector(), FakeOCR())
    result = pipeline.process_frames([])
    assert result.status == RecognitionStatus.NOT_DETECTED.value
    assert result.total_frames == 0


# ── Camera resolution negotiation (face_manager) ─────────────────────────────

class _FakeCapExactResolution:
    """Camera driver that honours whatever resolution is requested."""
    def set(self, prop, value):
        self._w = value if prop == 3 else getattr(self, "_w", 0)
        self._h = value if prop == 4 else getattr(self, "_h", 0)

    def get(self, prop):
        import cv2
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return getattr(self, "_w", 0)
        if prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return getattr(self, "_h", 0)
        return 0


class _FakeCapFixedResolution:
    """Camera driver that ignores the request and always reports 640x480."""
    def set(self, prop, value):
        pass

    def get(self, prop):
        import cv2
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return 640
        if prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return 480
        return 0


def test_camera_resolution_negotiation_honours_supported_request():
    import cv2
    import face_manager as fm
    cap = _FakeCapExactResolution()
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    w, h = fm.negotiate_camera_resolution(cap, 1280, 720)
    assert (w, h) == (1280, 720)


def test_camera_resolution_negotiation_falls_back_when_unsupported():
    import face_manager as fm
    cap = _FakeCapFixedResolution()
    w, h = fm.negotiate_camera_resolution(cap, 1280, 720)
    # The camera only supports 640x480 — the negotiated value must reflect
    # what the camera actually reported, not the request.
    assert (w, h) == (640, 480)
    assert (w, h) != (1280, 720)
