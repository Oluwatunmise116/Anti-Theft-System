"""
Nigerian license-plate detector wrapper.

Loads exactly one, explicitly configured model file. It does NOT download
anything at runtime — model acquisition (training + export) is an explicit
offline deployment step (see scripts/train_plate_detector.py and
scripts/export_plate_model.py). If the configured model is missing or
fails to load, detection fails clearly and the caller falls back to manual
plate entry — it never silently substitutes an unrelated model.
"""
from __future__ import annotations

import os
import threading
from typing import List, Optional

from .models import PlateDetection

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    from ultralytics import YOLO as _YOLO_CLASS
    YOLO_AVAILABLE = True
except ImportError:
    _YOLO_CLASS = None
    YOLO_AVAILABLE = False


class PlateModelError(RuntimeError):
    """Raised when the configured Nigerian plate model is missing or invalid."""


class PlateDetector:
    """
    Thin, thread-safe wrapper around a single Ultralytics YOLO model. Loaded
    once and reused for the lifetime of the process — never reloaded per
    request.
    """

    def __init__(self, model_path: str, confidence: float = 0.35, iou: float = 0.45):
        self.model_path = model_path
        self.confidence = confidence
        self.iou = iou
        self._model = None
        self._lock = threading.Lock()
        self._load_error: Optional[str] = None

    def validate(self) -> None:
        """Raise PlateModelError with a clear reason if the model can't be used."""
        if not YOLO_AVAILABLE:
            raise PlateModelError(
                "ultralytics is not installed. Run: pip install ultralytics"
            )
        if not self.model_path or not os.path.exists(self.model_path):
            raise PlateModelError(
                f"Nigerian plate model not found at '{self.model_path}'. "
                "Train it with scripts/train_plate_detector.py and export it "
                "with scripts/export_plate_model.py, or point "
                "plate_model_path at an existing weights file. The "
                "application will not substitute an unrelated model."
            )

    def is_ready(self) -> bool:
        try:
            self.validate()
        except PlateModelError:
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
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    self._model = _YOLO_CLASS(self.model_path)
                except Exception as e:
                    self._load_error = str(e)
                    raise PlateModelError(
                        f"Failed to load plate model '{self.model_path}': {e}"
                    ) from e
        return self._model

    def status(self) -> dict:
        return {
            "yolo_available": YOLO_AVAILABLE,
            "model_present": bool(self.model_path and os.path.exists(self.model_path)),
            "model_loaded": self._model is not None,
            "model_path": self.model_path,
            "load_error": self._load_error,
        }

    def detect(self, frame_bgr, frame_index: int = 0, source: str = "full_frame",
               offset: tuple = (0, 0)) -> List[PlateDetection]:
        """
        Run detection on frame_bgr (which may itself be a vehicle ROI crop —
        `offset` is added back to bounding boxes to map into full-frame
        coordinates).
        """
        model = self._load()
        if not NUMPY_AVAILABLE or frame_bgr is None or frame_bgr.size == 0:
            return []

        results = model(frame_bgr, conf=self.confidence, iou=self.iou, verbose=False)[0]
        boxes = results.boxes
        if boxes is None or len(boxes) == 0:
            return []

        ox, oy = offset
        h, w = frame_bgr.shape[:2]
        detections = []
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)
            conf = float(boxes.conf[i])
            cls_id = int(boxes.cls[i]) if boxes.cls is not None else 0
            x1c, y1c = max(0, x1), max(0, y1)
            x2c, y2c = min(w, x2), min(h, y2)
            if x2c <= x1c or y2c <= y1c:
                continue
            crop = frame_bgr[y1c:y2c, x1c:x2c]
            detections.append(PlateDetection(
                bounding_box=(x1c + ox, y1c + oy, x2c + ox, y2c + oy),
                detector_confidence=conf,
                class_id=cls_id,
                crop=crop,
                frame_index=frame_index,
                source=source,
            ))
        detections.sort(key=lambda d: d.detector_confidence, reverse=True)
        return detections
