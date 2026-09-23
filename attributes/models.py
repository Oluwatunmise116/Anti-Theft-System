"""
Typed data structures for vehicle attribute inference.

Mirrors anpr/models.py: no stage returns a bare string. Every attribute
carries its value, the confidence behind it, the number of frames that
voted for it, the status that decided it, and a `source` field with the
same vocabulary as the existing `plate_source` column, so an operator edit
is always distinguishable from an automatic reading.

These attributes are ADVISORY. Nothing in this package may participate in
an allow/deny decision — see README_ANPR.md's invariant that entry and exit
are decided by an exact match on the normalised plate.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, List, Optional, Tuple

# ── Label spaces ─────────────────────────────────────────────────────────
# Closed lists with explicit escape hatches. `unknown` and `other` are real
# classes with training examples, not post-hoc confidence thresholds: a
# night frame whose colour genuinely cannot be determined is labelled
# `unknown` during labelling, and the colour head is expected to predict it.
#
# ADJUST THESE to the vehicle mix at the installation before training. A
# label added here that has no training examples will never be predicted;
# a label removed here after training invalidates the checkpoint.

COLOURS = ["white", "black", "silver", "grey", "red", "blue",
           "green", "gold", "brown", "orange", "yellow", "unknown"]

TYPES = ["sedan", "suv", "hatchback", "minivan", "pickup",
         "bus", "truck", "motorcycle", "tricycle", "other"]

# mazda, bmw, audi, chevrolet and suzuki were added because the whole-vehicle
# brand model (attributes/brand_classifier.py) predicts them; collapsing
# them into `other` would throw away a correct answer.
BRANDS = ["toyota", "honda", "mercedes-benz", "lexus", "nissan",
          "hyundai", "kia", "ford", "volkswagen", "mitsubishi",
          "peugeot", "innoson", "mazda", "bmw", "audi", "chevrolet",
          "suzuki", "other"]

LABEL_SPACES = {"colour": COLOURS, "type": TYPES, "brand": BRANDS}

#: The escape-hatch class for each head — what the head says when it cannot
#: commit to a specific label. Never treated as a correct answer by the
#: evaluation script.
ESCAPE_LABEL = {"colour": "unknown", "type": "other", "brand": "other"}

ATTRIBUTES = ("colour", "type", "brand")


class AttributeStatus(str, Enum):
    """
    CONFIRMED             enough agreeing frames AND enough confidence.
    LOW_CONFIDENCE        a value was produced but one of those bars was
                          missed; shown to the operator, never trusted.
    UNKNOWN               deliberately no value (no logo for brand, or the
                          head predicted its escape class).
    VEHICLE_NOT_LOCALISED no vehicle box contained the plate box. Reported
                          explicitly rather than guessed at — attaching the
                          wrong body to the right plate is a silent
                          correctness bug.
    MODEL_UNAVAILABLE     feature disabled, or weights absent.
    FAILED                inference raised, or the frame was unusable.
    """
    CONFIRMED = "CONFIRMED"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    UNKNOWN = "UNKNOWN"
    VEHICLE_NOT_LOCALISED = "VEHICLE_NOT_LOCALISED"
    MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"
    FAILED = "FAILED"


#: Statuses whose value may be written to a trip record as an observation.
USABLE_STATUSES = (AttributeStatus.CONFIRMED.value, AttributeStatus.LOW_CONFIDENCE.value)

#: Provenance vocabulary, identical to the existing plate_source column.
SOURCE_AUTO = "auto"
SOURCE_MANUAL_CORRECTION = "manual_correction"
SOURCE_MANUAL_ENTRY = "manual_entry"

BoundingBox = Tuple[int, int, int, int]   # (x1, y1, x2, y2), full-frame pixels


@dataclass
class ScoredLabel:
    label: str
    confidence: float

    def to_dict(self) -> dict:
        return {"label": self.label, "confidence": round(float(self.confidence), 4)}


@dataclass
class AttributeValue:
    """One attribute's result after voting across frames."""
    attribute: str = ""
    value: Optional[str] = None
    confidence: float = 0.0
    status: str = AttributeStatus.UNKNOWN.value
    votes: int = 0                 # frames that produced the winning value
    total_frames: int = 0          # frames this attribute could vote on
    source: str = SOURCE_AUTO
    top_k: List[ScoredLabel] = field(default_factory=list)
    rejection_reason: Optional[str] = None
    model_version: Optional[str] = None

    @property
    def is_usable(self) -> bool:
        return self.status in USABLE_STATUSES and bool(self.value)

    @property
    def needs_operator_review(self) -> bool:
        """True when the UI must not present this as settled fact."""
        return self.status != AttributeStatus.CONFIRMED.value

    @property
    def is_escape_class(self) -> bool:
        """True when the head deliberately declined (`unknown` / `other`)."""
        return bool(self.value) and self.value == ESCAPE_LABEL.get(self.attribute)

    # ── constructors for the non-happy paths ─────────────────────────────
    @classmethod
    def unavailable(cls, attribute: str, reason: str) -> "AttributeValue":
        return cls(attribute=attribute, status=AttributeStatus.MODEL_UNAVAILABLE.value,
                   rejection_reason=reason)

    @classmethod
    def failed(cls, attribute: str, reason: str) -> "AttributeValue":
        return cls(attribute=attribute, status=AttributeStatus.FAILED.value,
                   rejection_reason=reason)

    @classmethod
    def unknown(cls, attribute: str, reason: str,
                top_k: Optional[List[ScoredLabel]] = None) -> "AttributeValue":
        return cls(attribute=attribute, status=AttributeStatus.UNKNOWN.value,
                   rejection_reason=reason, top_k=top_k or [])

    @classmethod
    def not_localised(cls, attribute: str, reason: str) -> "AttributeValue":
        return cls(attribute=attribute,
                   status=AttributeStatus.VEHICLE_NOT_LOCALISED.value,
                   rejection_reason=reason)

    @classmethod
    def operator(cls, attribute: str, value: str,
                 previous: Optional["AttributeValue"] = None) -> "AttributeValue":
        """
        An operator-supplied value. `source` is what distinguishes it from
        an automatic reading in the audit trail; confidence is left at 0.0
        because a human assertion has no model probability.
        """
        source = (SOURCE_MANUAL_CORRECTION if previous is not None and previous.is_usable
                  else SOURCE_MANUAL_ENTRY)
        return cls(attribute=attribute, value=value, confidence=0.0,
                   status=AttributeStatus.CONFIRMED.value, source=source)

    # ── serialization ─────────────────────────────────────────────────────
    def to_dict(self) -> dict:
        d = asdict(self)
        d["top_k"] = [t.to_dict() if isinstance(t, ScoredLabel) else dict(t)
                      for t in self.top_k]
        return d

    def to_api_dict(self) -> dict:
        """Shape sent to the UI. No filesystem paths, no tensors."""
        return {
            "attribute": self.attribute,
            "value": self.value,
            "confidence": round(float(self.confidence), 4),
            "confidence_pct": round(float(self.confidence) * 100, 1),
            "status": self.status,
            "votes": self.votes,
            "total_frames": self.total_frames,
            "source": self.source,
            "needs_operator_review": self.needs_operator_review,
            "rejection_reason": self.rejection_reason,
            "top_k": [t.to_dict() if isinstance(t, ScoredLabel) else dict(t)
                      for t in self.top_k],
        }

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "AttributeValue":
        if not d:
            return cls()
        top_k = []
        for t in d.get("top_k") or []:
            if isinstance(t, ScoredLabel):
                top_k.append(t)
            elif isinstance(t, dict) and t.get("label"):
                top_k.append(ScoredLabel(str(t["label"]),
                                         float(t.get("confidence", 0.0) or 0.0)))
        return cls(
            attribute=str(d.get("attribute", "") or ""),
            value=d.get("value"),
            confidence=float(d.get("confidence", 0.0) or 0.0),
            status=str(d.get("status", AttributeStatus.UNKNOWN.value)),
            votes=int(d.get("votes", 0) or 0),
            total_frames=int(d.get("total_frames", 0) or 0),
            source=str(d.get("source", SOURCE_AUTO)),
            top_k=top_k,
            rejection_reason=d.get("rejection_reason"),
            model_version=d.get("model_version"),
        )


@dataclass
class VehicleLocalisation:
    """
    The vehicle box linked to a plate box on one frame.

    `contains_fraction` is the fraction of the plate box's area that falls
    inside the chosen vehicle box — the evidence for the linkage, kept so a
    wrong pairing is diagnosable after the fact.
    """
    box: Optional[BoundingBox] = None
    coco_class: Optional[str] = None          # car | motorcycle | bus | truck
    detector_confidence: float = 0.0
    contains_fraction: float = 0.0
    frame_index: int = 0
    status: str = AttributeStatus.VEHICLE_NOT_LOCALISED.value
    rejection_reason: Optional[str] = None
    candidate_count: int = 0                  # vehicle boxes seen on this frame
    crop: Any = None                          # np.ndarray, kept out of to_dict()

    @property
    def found(self) -> bool:
        return self.box is not None and self.status == AttributeStatus.CONFIRMED.value

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("crop", None)
        d["box"] = list(self.box) if self.box else None
        return d


@dataclass
class LogoDetection:
    """One `vehicle_logo` box, in coordinates relative to the vehicle crop."""
    box: BoundingBox
    confidence: float
    frame_index: int = 0
    crop: Any = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("crop", None)
        d["box"] = list(self.box)
        return d


@dataclass
class VehicleAttributeResult:
    """The full output of one capture session's attribute pass."""
    colour: AttributeValue = field(default_factory=lambda: AttributeValue(attribute="colour"))
    type: AttributeValue = field(default_factory=lambda: AttributeValue(attribute="type"))
    brand: AttributeValue = field(default_factory=lambda: AttributeValue(attribute="brand"))

    vehicle_box: Optional[BoundingBox] = None
    coco_class: Optional[str] = None
    frames_processed: int = 0
    frames_with_vehicle: int = 0
    frames_with_logo: int = 0
    status: str = AttributeStatus.UNKNOWN.value
    rejection_reason: Optional[str] = None
    processing_time_ms: float = 0.0
    stage_ms: dict = field(default_factory=dict)
    model_versions: dict = field(default_factory=dict)
    debug_information: dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    # ── constructors ──────────────────────────────────────────────────────
    @classmethod
    def disabled(cls, reason: str = "Vehicle attributes are disabled in configuration"
                 ) -> "VehicleAttributeResult":
        return cls(
            colour=AttributeValue.unavailable("colour", reason),
            type=AttributeValue.unavailable("type", reason),
            brand=AttributeValue.unavailable("brand", reason),
            status=AttributeStatus.MODEL_UNAVAILABLE.value, rejection_reason=reason)

    @classmethod
    def vehicle_not_localised(cls, frames_processed: int, reason: str
                              ) -> "VehicleAttributeResult":
        """
        No vehicle box contained the plate box on any frame. Deliberately
        NOT a fallback to the largest or most confident box: with two
        vehicles in frame that silently attaches the wrong body to the
        right plate, and the error never surfaces in aggregate accuracy.
        """
        return cls(
            colour=AttributeValue.not_localised("colour", reason),
            type=AttributeValue.not_localised("type", reason),
            brand=AttributeValue.not_localised("brand", reason),
            frames_processed=frames_processed,
            status=AttributeStatus.VEHICLE_NOT_LOCALISED.value, rejection_reason=reason)

    @classmethod
    def failed(cls, frames_processed: int, reason: str) -> "VehicleAttributeResult":
        return cls(
            colour=AttributeValue.failed("colour", reason),
            type=AttributeValue.failed("type", reason),
            brand=AttributeValue.failed("brand", reason),
            frames_processed=frames_processed,
            status=AttributeStatus.FAILED.value, rejection_reason=reason)

    # ── queries ───────────────────────────────────────────────────────────
    def values(self) -> dict:
        return {"colour": self.colour, "type": self.type, "brand": self.brand}

    def roll_up_status(self) -> str:
        statuses = [v.status for v in self.values().values()]
        if all(s == AttributeStatus.VEHICLE_NOT_LOCALISED.value for s in statuses):
            return AttributeStatus.VEHICLE_NOT_LOCALISED.value
        if all(s == AttributeStatus.MODEL_UNAVAILABLE.value for s in statuses):
            return AttributeStatus.MODEL_UNAVAILABLE.value
        if all(s == AttributeStatus.FAILED.value for s in statuses):
            return AttributeStatus.FAILED.value
        if any(s == AttributeStatus.CONFIRMED.value for s in statuses):
            return AttributeStatus.CONFIRMED.value
        if any(s == AttributeStatus.LOW_CONFIDENCE.value for s in statuses):
            return AttributeStatus.LOW_CONFIDENCE.value
        return AttributeStatus.UNKNOWN.value

    # ── serialization ─────────────────────────────────────────────────────
    def to_dict(self) -> dict:
        return {
            "colour": self.colour.to_dict(),
            "type": self.type.to_dict(),
            "brand": self.brand.to_dict(),
            "vehicle_box": list(self.vehicle_box) if self.vehicle_box else None,
            "coco_class": self.coco_class,
            "frames_processed": self.frames_processed,
            "frames_with_vehicle": self.frames_with_vehicle,
            "frames_with_logo": self.frames_with_logo,
            "status": self.status,
            "rejection_reason": self.rejection_reason,
            "processing_time_ms": round(self.processing_time_ms, 2),
            "stage_ms": {k: round(v, 2) for k, v in self.stage_ms.items()},
            "model_versions": dict(self.model_versions),
        }

    def to_api_dict(self) -> dict:
        return {
            "colour": self.colour.to_api_dict(),
            "type": self.type.to_api_dict(),
            "brand": self.brand.to_api_dict(),
            "coco_class": self.coco_class,
            "frames_processed": self.frames_processed,
            "frames_with_vehicle": self.frames_with_vehicle,
            "frames_with_logo": self.frames_with_logo,
            "status": self.status,
            "rejection_reason": self.rejection_reason,
            "processing_time_ms": round(self.processing_time_ms, 1),
            # Advisory only. Repeated in the payload so a UI author cannot
            # miss it, and asserted by the route tests.
            "advisory_only": True,
        }

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "VehicleAttributeResult":
        if not d:
            return cls()
        box = d.get("vehicle_box")
        return cls(
            colour=AttributeValue.from_dict(d.get("colour")),
            type=AttributeValue.from_dict(d.get("type")),
            brand=AttributeValue.from_dict(d.get("brand")),
            vehicle_box=tuple(box) if box else None,
            coco_class=d.get("coco_class"),
            frames_processed=int(d.get("frames_processed", 0) or 0),
            frames_with_vehicle=int(d.get("frames_with_vehicle", 0) or 0),
            frames_with_logo=int(d.get("frames_with_logo", 0) or 0),
            status=str(d.get("status", AttributeStatus.UNKNOWN.value)),
            rejection_reason=d.get("rejection_reason"),
            processing_time_ms=float(d.get("processing_time_ms", 0.0) or 0.0),
            stage_ms=dict(d.get("stage_ms") or {}),
            model_versions=dict(d.get("model_versions") or {}),
        )
