"""
Vehicle attribute inference: colour, body type and brand.

    best frames (already chosen by anpr.preprocessing.select_best_frames)
      + the plate box from the existing PlateDetector
        -> COCO vehicle boxes, linked to the plate by CONTAINMENT
        -> vehicle crop -> MobileNetV3-Large backbone -> colour head, type head
        -> vehicle crop -> single-class logo detector
             -> badge crop -> same backbone -> brand head
        -> per-attribute temporal voting
        -> VehicleAttributeResult

ADVISORY ONLY. Nothing in this package may influence an allow/deny
decision. README_ANPR.md's invariant is that entry and exit are decided by
an exact match on the normalised plate; attributes extend that invariant by
routing disagreements to the existing operator-confirmation dialog, never
to a relay. A missing or broken attribute model must never block the gate,
exactly as a missing plate model does not.

Attributes are computed AFTER the plate result has been returned, so the
barrier never waits on a brand classifier.
"""
from __future__ import annotations

import logging
import time
from typing import Callable, List, Optional, Sequence

from .models import (ATTRIBUTES, BRANDS, COLOURS, TYPES, AttributeStatus,
                     AttributeValue, BoundingBox, ESCAPE_LABEL, LABEL_SPACES,
                     LogoDetection, ScoredLabel, SOURCE_AUTO,
                     SOURCE_MANUAL_CORRECTION, SOURCE_MANUAL_ENTRY,
                     VehicleAttributeResult, VehicleLocalisation)
from . import consensus as attr_consensus
from . import settings as attr_settings
from .classifiers import AttributeClassifier, AttributeModelError, is_valid_crop
from .colour_baseline import (BODY_REGION, ColourEstimate, HSVColourBackend,
                              estimate_colour)
from .logo import LogoDetector, LogoModelError
from .vehicle_detector import (AUTHORITATIVE_TYPES, VehicleDetector,
                               contains_fraction, select_vehicle_for_plate)

log = logging.getLogger("attributes")

__all__ = [
    "COLOURS", "TYPES", "BRANDS", "LABEL_SPACES", "ESCAPE_LABEL", "ATTRIBUTES",
    "AttributeStatus", "AttributeValue", "VehicleAttributeResult",
    "VehicleLocalisation", "LogoDetection", "ScoredLabel",
    "SOURCE_AUTO", "SOURCE_MANUAL_CORRECTION", "SOURCE_MANUAL_ENTRY",
    "VehicleAttributePipeline", "build_pipeline",
    "AttributeClassifier", "LogoDetector", "VehicleDetector",
    "HSVColourBackend", "ColourEstimate", "estimate_colour", "BODY_REGION",
    "select_vehicle_for_plate", "contains_fraction",
]

#: Minimum badge crop size, in pixels, before upscaling to the head input.
#: Below this the crop carries no marque signal and classifying it would
#: manufacture a confident answer from noise.
MIN_BADGE_PIXELS = 8


class VehicleAttributePipeline:
    """
    Built once per process and reused. The classifier, logo detector and
    COCO detector are each loaded lazily behind their own lock and are
    never reloaded per request.
    """

    def __init__(self, classifier: AttributeClassifier, logo_detector: LogoDetector,
                 vehicle_detector: VehicleDetector, config: dict,
                 colour_baseline: Optional[HSVColourBackend] = None):
        self.classifier = classifier
        self.logo = logo_detector
        self.vehicles = vehicle_detector
        self.config = config
        self.colour_baseline = colour_baseline or HSVColourBackend(
            min_support=float(config.get("attr_colour_hsv_min_support", 0.28)))

    def colour_method(self) -> str:
        """
        Which colour method this run will use: "head" or "hsv".

        `auto` prefers a trained head and falls back to the baseline, so
        colour works whether or not a checkpoint exists.
        """
        configured = self.config.get("attr_colour_backend", "auto")
        if configured == "hsv":
            return "hsv"
        if configured == "head":
            return "head"
        return "head" if getattr(self.classifier, "trained", False) else "hsv"

    # ── config helpers ────────────────────────────────────────────────────
    def _cfg(self, key, default):
        return self.config.get(key, default)

    def status(self) -> dict:
        """Readiness report. Loads the classifier only if already loaded."""
        return {
            "enabled": self._cfg("attr_enabled", True),
            "heads_enabled": {
                a: attr_settings.enabled(self.config, a) for a in ATTRIBUTES
            },
            "classifier": self.classifier.status(),
            "colour_backend": {
                "configured": self.config.get("attr_colour_backend", "auto"),
                "active": self.colour_method(),
                "version": self.colour_baseline.version,
            },
            "logo_detector": self.logo.status(),
            "vehicle_detector": self.vehicles.status(),
            "latency_budget_ms": self._cfg("attr_latency_budget_ms", 150.0),
            "label_space_sizes": {k: len(v) for k, v in LABEL_SPACES.items()},
        }

    def warm_up(self):
        """
        Build the backbone outside a gate transaction. The first forward
        pass through a freshly constructed torch module is several times
        slower than the steady state, so paying it here keeps the first
        vehicle of the day within the latency budget.
        """
        try:
            if self.classifier.is_ready():
                import numpy as np
                self.classifier.predict(
                    np.zeros((64, 64, 3), dtype=np.uint8), heads=("colour",))
        except Exception as exc:
            log.debug("Attribute warm-up skipped: %s", exc)

    # ── main entry point ─────────────────────────────────────────────────
    def process_frames(
        self,
        frames: Sequence,
        plate_box: Optional[BoundingBox] = None,
        plate_box_for_frame: Optional[Callable[[int], Optional[BoundingBox]]] = None,
        vehicle_boxes_for_frame: Optional[Callable[[int], Optional[List[dict]]]] = None,
        progress: Optional[Callable[[str], None]] = None,
    ) -> VehicleAttributeResult:
        """
        Run the attribute pass over the frames the plate stage already used.

        `plate_box` is the box the plate pipeline reported. Supply
        `plate_box_for_frame` instead when per-frame plate boxes are
        available — the vehicle moves slightly between burst frames, and a
        frame whose containment test fails simply does not vote rather than
        being linked to the wrong body.

        `vehicle_boxes_for_frame` supplies COCO boxes that the caller has
        ALREADY computed. The gate runs yolov8n on these same frames during
        capture and ANPR region selection, and that pass is measured at
        ~297 ms on a 1280x720 frame on the target Pi — an order of
        magnitude more than everything else here. Passing the existing
        boxes in is the difference between fitting the latency budget and
        missing it by 3x. When omitted, this falls back to running the
        detector itself.

        Never raises. Every failure path returns a structured result.
        """
        started = time.perf_counter()
        if not self._cfg("attr_enabled", True):
            return VehicleAttributeResult.disabled()
        if not any(attr_settings.enabled(self.config, a) for a in ATTRIBUTES):
            return VehicleAttributeResult.disabled(
                "Every attribute head is disabled in configuration")

        usable = [f for f in (frames or []) if is_valid_crop(f)]
        if not usable:
            return VehicleAttributeResult.failed(
                0, "No usable frames were supplied for attribute inference")

        max_frames = int(self._cfg("attr_capture_frame_count", 3))
        budget_ms = float(self._cfg("attr_latency_budget_ms", 150.0))
        containment = float(self._cfg("attr_plate_containment_fraction", 0.90))
        selected = usable[:max_frames]

        if progress:
            progress(f"Reading vehicle attributes from {len(selected)} frame(s)…")

        votes = {a: [] for a in ATTRIBUTES}
        stage_ms = {"vehicle": 0.0, "classify": 0.0, "logo": 0.0, "brand": 0.0}
        localisations: List[VehicleLocalisation] = []
        frames_with_vehicle = 0
        frames_with_logo = 0
        frames_done = 0
        budget_exceeded = False
        classifier_error: Optional[str] = None
        last_hsv_reason: Optional[str] = None
        last_top_k = {}

        for index, frame in enumerate(selected):
            if (time.perf_counter() - started) * 1000.0 > budget_ms:
                budget_exceeded = True
                break
            frames_done += 1

            box = (plate_box_for_frame(index) if plate_box_for_frame else plate_box)

            t0 = time.perf_counter()
            supplied = vehicle_boxes_for_frame(index) if vehicle_boxes_for_frame else None
            if supplied is not None:
                # Reuse the caller's boxes: link only, no second YOLO pass.
                localisation = select_vehicle_for_plate(supplied, box, containment, index)
                if localisation.found and frame is not None:
                    x1, y1, x2, y2 = localisation.box
                    localisation.crop = frame[y1:y2, x1:x2]
            else:
                localisation = self.vehicles.localise(frame, box, containment, index)
            stage_ms["vehicle"] += (time.perf_counter() - t0) * 1000.0
            localisations.append(localisation)
            if not localisation.found:
                continue
            frames_with_vehicle += 1

            # Colour: the HSV baseline runs instead of the head whenever the
            # head is not the active backend. It needs no checkpoint, costs
            # ~2 ms, and its confidence is the fraction of body pixels
            # supporting the winner — a real quantity, not a softmax.
            use_hsv_colour = (attr_settings.enabled(self.config, "colour")
                              and self.colour_method() == "hsv")
            if use_hsv_colour:
                t0 = time.perf_counter()
                try:
                    label, confidence, top_k, reason = \
                        self.colour_baseline.predict(localisation.crop)
                    last_top_k["colour"] = top_k
                    if label:
                        votes["colour"].append({
                            "label": label, "confidence": confidence,
                            "frame_index": index,
                            # Not a stubbed head: a working method with no
                            # weights, valid with or without a checkpoint.
                            "authoritative": True})
                    else:
                        last_hsv_reason = reason
                except Exception as exc:
                    log.warning("HSV colour baseline failed on frame %d: %s: %s",
                                index, type(exc).__name__, exc)
                stage_ms["colour_hsv"] = stage_ms.get("colour_hsv", 0.0) + \
                    (time.perf_counter() - t0) * 1000.0

            # Type, and colour when the learned head is the active backend.
            heads = tuple(h for h in ("colour", "type")
                          if attr_settings.enabled(self.config, h)
                          and not (h == "colour" and use_hsv_colour))
            # A coarse COCO class settles the type outright, so the type
            # head is not consulted for buses, trucks or motorcycles.
            coarse_type = AUTHORITATIVE_TYPES.get(localisation.coco_class or "")
            if coarse_type and "type" in heads:
                heads = tuple(h for h in heads if h != "type")
                votes["type"].append({"label": coarse_type,
                                      "confidence": localisation.detector_confidence,
                                      "frame_index": index,
                                      # From COCO, not the learned head:
                                      # valid even with an untrained backbone.
                                      "authoritative": True})

            if heads:
                t0 = time.perf_counter()
                try:
                    predictions = self.classifier.predict(localisation.crop, heads=heads)
                    for head, scored in predictions.items():
                        last_top_k[head] = scored
                        votes[head].append({"label": scored[0].label,
                                            "confidence": scored[0].confidence,
                                            "frame_index": index,
                                            "authoritative": False})
                except (AttributeModelError, ValueError) as exc:
                    classifier_error = f"{type(exc).__name__}: {exc}"
                    log.warning("Attribute classification failed on frame %d: %s",
                                index, classifier_error)
                except Exception as exc:
                    classifier_error = f"{type(exc).__name__}: {exc}"
                    log.warning("Unexpected attribute failure on frame %d: %s",
                                index, classifier_error)
                stage_ms["classify"] += (time.perf_counter() - t0) * 1000.0

            # Brand: badge presence, then the badge crop into the brand head.
            if attr_settings.enabled(self.config, "brand"):
                t0 = time.perf_counter()
                badges = self.logo.detect(localisation.crop, frame_index=index)
                stage_ms["logo"] += (time.perf_counter() - t0) * 1000.0
                if badges:
                    badge = badges[0]
                    bh, bw = badge.crop.shape[:2] if badge.crop is not None else (0, 0)
                    if bh >= MIN_BADGE_PIXELS and bw >= MIN_BADGE_PIXELS:
                        frames_with_logo += 1
                        t0 = time.perf_counter()
                        try:
                            predictions = self.classifier.predict(badge.crop,
                                                                  heads=("brand",))
                            scored = predictions["brand"]
                            last_top_k["brand"] = scored
                            votes["brand"].append({"label": scored[0].label,
                                                   "confidence": scored[0].confidence,
                                                   "frame_index": index,
                                                   "authoritative": False})
                        except Exception as exc:
                            classifier_error = f"{type(exc).__name__}: {exc}"
                            log.warning("Brand classification failed on frame %d: %s",
                                        index, classifier_error)
                        stage_ms["brand"] += (time.perf_counter() - t0) * 1000.0

        elapsed_ms = (time.perf_counter() - started) * 1000.0

        if frames_with_vehicle == 0:
            reason = self._not_localised_reason(localisations)
            result = VehicleAttributeResult.vehicle_not_localised(frames_done, reason)
            result.processing_time_ms = elapsed_ms
            result.stage_ms = stage_ms
            result.debug_information = {
                "localisations": [loc.to_dict() for loc in localisations]}
            return result

        result = self._vote(votes, frames_done, frames_with_vehicle,
                            frames_with_logo, last_top_k, classifier_error,
                            last_hsv_reason)
        winner = next((loc for loc in localisations if loc.found), None)
        if winner is not None:
            result.vehicle_box = winner.box
            result.coco_class = winner.coco_class
        result.frames_processed = frames_done
        result.frames_with_vehicle = frames_with_vehicle
        result.frames_with_logo = frames_with_logo
        result.processing_time_ms = elapsed_ms
        result.stage_ms = stage_ms
        result.model_versions = {
            "attributes": self.classifier.version,
            "logo": (self.logo.model_path if self.logo.is_ready() else None),
        }
        result.status = result.roll_up_status()
        if budget_exceeded:
            result.rejection_reason = (
                f"Latency budget of {budget_ms:.0f} ms reached after "
                f"{frames_done}/{len(selected)} frame(s); attributes never delay "
                "the gate decision")
        if self._cfg("attr_debug_mode", False):
            result.debug_information = {
                "localisations": [loc.to_dict() for loc in localisations],
                "votes": votes,
            }
        return result

    # ── helpers ───────────────────────────────────────────────────────────
    @staticmethod
    def _not_localised_reason(localisations: List[VehicleLocalisation]) -> str:
        for loc in localisations:
            if loc.rejection_reason:
                return loc.rejection_reason
        return "No vehicle box contained the plate box on any frame"

    def _vote(self, votes, frames_done, frames_with_vehicle, frames_with_logo,
              last_top_k, classifier_error, hsv_reason=None) -> VehicleAttributeResult:
        consensus_frames = int(self._cfg("attr_consensus_frames", 2))
        version = self.classifier.version
        untrained = not self.classifier.trained
        colour_method = self.colour_method()

        def value_for(attribute: str, total_frames: int) -> AttributeValue:
            if not attr_settings.enabled(self.config, attribute):
                return AttributeValue.unavailable(
                    attribute, f"{attribute} head is disabled in configuration")

            attribute_votes = votes[attribute]
            if untrained:
                # The stub backbone produces real tensors and real timings,
                # but its labels are noise. Only votes that did NOT come
                # from a learned head survive — in practice the coarse COCO
                # type for a bus, truck or motorcycle. Anything else would
                # be fabricating a reading.
                attribute_votes = [v for v in attribute_votes
                                   if v.get("authoritative")]
                if not attribute_votes:
                    return AttributeValue.unavailable(
                        attribute,
                        "Attribute model is untrained (stub weights) — no "
                        "prediction is reported. Train and promote a checkpoint "
                        "with scripts/train_vehicle_attributes.py.")

            if not attribute_votes:
                if attribute == "colour" and colour_method == "hsv":
                    return AttributeValue.unknown(
                        attribute, hsv_reason or
                        "the HSV baseline found no colour holding enough of "
                        "the vehicle body")
                if classifier_error:
                    return AttributeValue.failed(attribute, classifier_error)
                if attribute == "brand":
                    # The defining rule of the two-stage design: no badge
                    # means no brand. Never guessed from the body crop.
                    return AttributeValue.unknown(
                        attribute,
                        "No vehicle logo was detected — brand is not inferred "
                        "from the body crop")
                return AttributeValue.unknown(
                    attribute, f"No frame produced a {attribute} value")
            return attr_consensus.decide(
                attribute, attribute_votes, total_frames,
                consensus_frames=consensus_frames,
                min_confirm_confidence=float(
                    self._cfg(f"attr_{attribute}_min_confidence", 0.55)),
                top_k=last_top_k.get(attribute),
                model_version=(self.colour_baseline.version
                               if attribute == "colour" and colour_method == "hsv"
                               else version))

        # Brand votes only across frames where a badge was found, so its
        # vote count is honest about its own denominator.
        return VehicleAttributeResult(
            colour=value_for("colour", frames_with_vehicle),
            type=value_for("type", frames_with_vehicle),
            brand=value_for("brand", frames_with_logo),
        )


def build_pipeline(config: dict) -> VehicleAttributePipeline:
    """Construct the pipeline from a validated settings mapping."""
    classifier = AttributeClassifier(
        model_path=attr_settings.resolve_path(config.get("attr_model_path", "")),
        input_size=int(config.get("attr_input_size", 224)),
        model_format=config.get("attr_model_format", "pytorch"),
        threads=int(config.get("attr_torch_threads", 4)),
    )
    logo_detector = LogoDetector(
        model_path=attr_settings.resolve_path(config.get("attr_logo_model_path", "")),
        confidence=float(config.get("attr_logo_detection_confidence", 0.30)),
        iou=float(config.get("attr_logo_detection_iou", 0.45)),
        model_format=config.get("attr_logo_model_format", "pytorch"),
    )
    vehicle_detector = VehicleDetector(
        confidence=float(config.get("attr_vehicle_detection_confidence", 0.25)),
        iou=float(config.get("attr_vehicle_detection_iou", 0.45)),
        imgsz=int(config.get("attr_vehicle_detection_imgsz", 320)),
    )
    return VehicleAttributePipeline(
        classifier, logo_detector, vehicle_detector, config,
        colour_baseline=HSVColourBackend(
            min_support=float(config.get("attr_colour_hsv_min_support", 0.28))))
