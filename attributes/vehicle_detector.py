"""
Vehicle localisation by plate-box containment.

THE RULE THIS MODULE EXISTS TO ENFORCE
--------------------------------------
The vehicle whose attributes we report must be the vehicle that carries the
plate we just read. With two vehicles in frame — routine at a gate, where a
car queues behind another — picking the largest or the most confident box
attaches the wrong body to the right plate. That produces a colour and a
brand that are confidently, silently wrong: the trip record looks complete,
and the error never shows up in aggregate accuracy because the plate was
right.

So the vehicle box is chosen ONLY by containment of the plate box. If no
vehicle box contains the plate box, this returns VEHICLE_NOT_LOCALISED and
the attribute stage reports that status. There is deliberately no fallback
to the largest box, the most confident box, or the nearest box.

Reuses the yolov8n.pt COCO model the gate already loads (via gate_manager)
rather than loading a second copy. Nothing here downloads a model.
"""
from __future__ import annotations

import logging
import threading
from typing import List, Optional, Sequence

from .models import (AttributeStatus, BoundingBox, VehicleLocalisation)

log = logging.getLogger("attributes.vehicle_detector")

# COCO ids for the four vehicle classes, and their names. `bus`, `truck` and
# `motorcycle` are authoritative for the coarse body type — the learned type
# head only resolves fine-grained style within `car`.
COCO_VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
COCO_VEHICLE_IDS = sorted(COCO_VEHICLE_CLASSES)

#: Coarse COCO classes that settle the `type` attribute without the head.
AUTHORITATIVE_TYPES = {"bus": "bus", "truck": "truck", "motorcycle": "motorcycle"}


def box_area(box: Sequence[int]) -> int:
    x1, y1, x2, y2 = box
    return max(0, int(x2) - int(x1)) * max(0, int(y2) - int(y1))


def intersection_area(a: Sequence[int], b: Sequence[int]) -> int:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    x1, y1 = max(ax1, bx1), max(ay1, by1)
    x2, y2 = min(ax2, bx2), min(ay2, by2)
    return max(0, x2 - x1) * max(0, y2 - y1)


def contains_fraction(outer: Sequence[int], inner: Sequence[int]) -> float:
    """
    Fraction of `inner`'s area that lies inside `outer`.

    Not IoU: a plate is a tiny fraction of a car, so IoU between the two is
    always near zero and would reject every correct pairing. What matters
    is whether the plate sits inside the body.
    """
    inner_area = box_area(inner)
    if inner_area <= 0:
        return 0.0
    return intersection_area(outer, inner) / float(inner_area)


def select_vehicle_for_plate(vehicle_boxes: List[dict], plate_box: Optional[BoundingBox],
                             min_contains_fraction: float = 0.90,
                             frame_index: int = 0) -> VehicleLocalisation:
    """
    Pick the vehicle box that contains `plate_box`.

    `vehicle_boxes` is a list of {"box", "class_name", "confidence"}.

    When several boxes contain the plate (a car detected inside a bus box,
    or nested duplicate detections), the SMALLEST containing box wins: it is
    the tightest body that still accounts for the plate. That is a
    containment-based tie-break, not a size heuristic — every candidate has
    already passed the containment test.
    """
    if plate_box is None:
        return VehicleLocalisation(
            frame_index=frame_index, candidate_count=len(vehicle_boxes),
            status=AttributeStatus.VEHICLE_NOT_LOCALISED.value,
            rejection_reason="No plate box was detected on this frame, so no "
                             "vehicle can be linked to a plate")

    if not vehicle_boxes:
        return VehicleLocalisation(
            frame_index=frame_index, candidate_count=0,
            status=AttributeStatus.VEHICLE_NOT_LOCALISED.value,
            rejection_reason="No vehicle boxes detected on this frame")

    containing = []
    best_fraction = 0.0
    for candidate in vehicle_boxes:
        fraction = contains_fraction(candidate["box"], plate_box)
        best_fraction = max(best_fraction, fraction)
        if fraction >= min_contains_fraction:
            containing.append((candidate, fraction))

    if not containing:
        return VehicleLocalisation(
            frame_index=frame_index, candidate_count=len(vehicle_boxes),
            contains_fraction=best_fraction,
            status=AttributeStatus.VEHICLE_NOT_LOCALISED.value,
            rejection_reason=(
                f"No vehicle box contains the plate box "
                f"(best containment {best_fraction:.2f} < {min_contains_fraction:.2f} "
                f"across {len(vehicle_boxes)} vehicle box(es)). Refusing to guess "
                "which vehicle carries this plate."))

    # Tightest containing body.
    winner, fraction = min(containing, key=lambda cf: box_area(cf[0]["box"]))
    return VehicleLocalisation(
        box=tuple(int(v) for v in winner["box"]),
        coco_class=winner.get("class_name"),
        detector_confidence=float(winner.get("confidence", 0.0) or 0.0),
        contains_fraction=fraction,
        frame_index=frame_index,
        candidate_count=len(vehicle_boxes),
        status=AttributeStatus.CONFIRMED.value)


class VehicleDetector:
    """
    Thin, thread-safe wrapper around the COCO YOLOv8n model.

    Mirrors anpr.detector.PlateDetector: loaded once, reused for the life of
    the process, never downloaded at runtime. By default it borrows the
    instance gate_manager already holds so the gate runs one COCO model, not
    two.
    """

    def __init__(self, model_path: str = "yolov8n.pt", confidence: float = 0.25,
                 iou: float = 0.45, shared_model=None, imgsz: int = 320):
        self.model_path = model_path
        self.confidence = confidence
        self.iou = iou
        self.imgsz = imgsz
        self._model = shared_model
        self._lock = threading.Lock()
        self._load_error: Optional[str] = None

    # ── model access ──────────────────────────────────────────────────────
    def _load(self):
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is not None:
                return self._model
            # Prefer the already-resident instance in gate_manager. Imported
            # lazily so this package stays importable on its own.
            try:
                import gate_manager
                shared = gate_manager._get_vehicle_yolo()
                if shared is not None:
                    self._model = shared
                    return self._model
            except Exception as exc:
                log.debug("Shared COCO model unavailable: %s", exc)
            self._load_error = (
                "COCO vehicle model (yolov8n.pt) is not loaded. Vehicle "
                "attributes are unavailable; the gate is unaffected.")
        return self._model

    def is_ready(self) -> bool:
        return self._load() is not None

    def status(self) -> dict:
        return {
            "model_path": self.model_path,
            "model_loaded": self._model is not None,
            "imgsz": self.imgsz,
            "load_error": self._load_error,
        }

    # ── detection ─────────────────────────────────────────────────────────
    def detect_vehicles(self, frame_bgr, frame_index: int = 0) -> List[dict]:
        """All vehicle boxes on a frame, as {"box", "class_name", "confidence"}."""
        model = self._load()
        if model is None or frame_bgr is None or getattr(frame_bgr, "size", 0) == 0:
            return []
        try:
            results = model(frame_bgr, classes=COCO_VEHICLE_IDS,
                            conf=self.confidence, iou=self.iou,
                            imgsz=self.imgsz, verbose=False)[0]
        except Exception as exc:
            log.warning("Vehicle detection failed on frame %d: %s: %s",
                        frame_index, type(exc).__name__, exc)
            return []
        boxes = results.boxes
        if boxes is None or len(boxes) == 0:
            return []

        h, w = frame_bgr.shape[:2]
        out = []
        xyxy = boxes.xyxy.cpu().numpy()
        for i in range(len(boxes)):
            x1, y1, x2, y2 = (int(v) for v in xyxy[i])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            try:
                class_id = int(boxes.cls[i].cpu().numpy())
                confidence = float(boxes.conf[i].cpu().numpy())
            except Exception:
                class_id, confidence = -1, 0.0
            out.append({
                "box": (x1, y1, x2, y2),
                "class_name": COCO_VEHICLE_CLASSES.get(class_id),
                "confidence": confidence,
            })
        return out

    def localise(self, frame_bgr, plate_box: Optional[BoundingBox],
                 min_contains_fraction: float = 0.90,
                 frame_index: int = 0) -> VehicleLocalisation:
        """Detect vehicles on the frame and link one to the plate box."""
        if not self.is_ready():
            return VehicleLocalisation(
                frame_index=frame_index,
                status=AttributeStatus.MODEL_UNAVAILABLE.value,
                rejection_reason=self._load_error or "vehicle detector unavailable")

        candidates = self.detect_vehicles(frame_bgr, frame_index=frame_index)
        localisation = select_vehicle_for_plate(
            candidates, plate_box, min_contains_fraction, frame_index)

        if localisation.found and frame_bgr is not None:
            x1, y1, x2, y2 = localisation.box
            localisation.crop = frame_bgr[y1:y2, x1:x2]
        return localisation
