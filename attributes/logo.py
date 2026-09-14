"""
Vehicle badge detector: stage one of the two-stage brand pipeline.

WHY TWO STAGES
--------------
A badge occupies a few dozen pixels at gate distance. End-to-end brand
classifiers collapse on it, because the marque signal is a tiny fraction of
the body crop's pixels and the network latches onto body shape instead —
which is how a Corolla gets confidently called a Hyundai. Published
surveillance-logo pipelines restrict the detector to PRESENCE ONLY (a
single `vehicle_logo` class) and hand the crop to a separate classifier.
This module is that detector.

It runs on the VEHICLE CROP, not the full frame: the badge is always on the
vehicle, and searching the whole frame wastes the latency budget and
invites false positives on signage.

If no logo is detected, brand is UNKNOWN with zero confidence. There is
deliberately no fallback to classifying the whole body — that is precisely
the failure mode this design exists to avoid.

Nothing here downloads a model. Mirrors anpr.detector.PlateDetector.
"""
from __future__ import annotations

import logging
import os
import threading
from typing import List, Optional

from .models import LogoDetection

log = logging.getLogger("attributes.logo")

#: The detector is trained with exactly one class. Presence, not identity.
LOGO_CLASS_NAME = "vehicle_logo"


class LogoModelError(RuntimeError):
    """Raised when the configured logo model is missing or invalid."""


class LogoDetector:
    """Thin, thread-safe wrapper around a single-class YOLOv8n model."""

    def __init__(self, model_path: str, confidence: float = 0.30, iou: float = 0.45,
                 model_format: str = "pytorch"):
        self.model_path = model_path
        self.confidence = confidence
        self.iou = iou
        self.model_format = model_format
        self._model = None
        self._lock = threading.Lock()
        self._load_error: Optional[str] = None

    # ── model access ──────────────────────────────────────────────────────
    def validate(self) -> None:
        try:
            import ultralytics  # noqa: F401
        except ImportError as exc:
            raise LogoModelError(
                "ultralytics is not installed. Run: pip install ultralytics") from exc
        if not self.model_path or not os.path.exists(self.model_path):
            raise LogoModelError(
                f"Vehicle logo detector not found at '{self.model_path}'. Train it "
                "with scripts/train_vehicle_attributes.py --stage logo, or point "
                "attr_logo_model_path at an existing weights file. Brand "
                "recognition stays UNKNOWN until then; the gate is unaffected.")

    def is_ready(self) -> bool:
        try:
            self.validate()
        except LogoModelError:
            return False
        return True

    def _load(self):
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is not None:
                return self._model
            self.validate()
            import warnings

            from ultralytics import YOLO
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    self._model = YOLO(self.model_path)
                except Exception as exc:
                    self._load_error = str(exc)
                    raise LogoModelError(
                        f"Failed to load logo model '{self.model_path}': {exc}") from exc
        return self._model

    def status(self) -> dict:
        return {
            "model_path": self.model_path,
            "model_format": self.model_format,
            "model_present": bool(self.model_path) and os.path.exists(self.model_path),
            "model_loaded": self._model is not None,
            "load_error": self._load_error,
        }

    # ── detection ─────────────────────────────────────────────────────────
    def detect(self, vehicle_crop_bgr, frame_index: int = 0,
               max_detections: int = 1) -> List[LogoDetection]:
        """
        Find badges on a vehicle crop. Boxes are in crop coordinates.

        Returns the highest-confidence detections first, capped at
        `max_detections` — a vehicle has one marque, and classifying three
        boxes would triple the brand cost for no gain.

        Never raises: a missing or broken model yields an empty list, and
        the brand attribute becomes UNKNOWN.
        """
        if vehicle_crop_bgr is None or getattr(vehicle_crop_bgr, "size", 0) == 0:
            return []
        try:
            model = self._load()
        except LogoModelError:
            return []

        try:
            results = model(vehicle_crop_bgr, conf=self.confidence, iou=self.iou,
                            verbose=False)[0]
        except Exception as exc:
            log.warning("Logo detection failed on frame %d: %s: %s",
                        frame_index, type(exc).__name__, exc)
            return []

        boxes = results.boxes
        if boxes is None or len(boxes) == 0:
            return []

        h, w = vehicle_crop_bgr.shape[:2]
        detections = []
        xyxy = boxes.xyxy.cpu().numpy()
        for i in range(len(boxes)):
            x1, y1, x2, y2 = (int(v) for v in xyxy[i])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            try:
                confidence = float(boxes.conf[i].cpu().numpy())
            except Exception:
                confidence = 0.0
            detections.append(LogoDetection(
                box=(x1, y1, x2, y2), confidence=confidence, frame_index=frame_index,
                crop=vehicle_crop_bgr[y1:y2, x1:x2]))

        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections[:max_detections]
