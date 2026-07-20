"""
Typed data structures for the ANPR pipeline.

The pipeline never returns a bare string from its core stages — detection,
OCR, and consensus each produce a structured result so confidence, the
evidence behind it, and rejection reasons survive all the way to the UI.
A thin compatibility wrapper (anpr.detect_plate_string) is provided for
call sites that still just want a string.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Optional


class RecognitionStatus(str, Enum):
    CONFIRMED = "CONFIRMED"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    NOT_DETECTED = "NOT_DETECTED"
    MANUAL = "MANUAL"


BoundingBox = tuple  # (x1, y1, x2, y2) in full-frame pixel coordinates


@dataclass
class PlateDetection:
    """One raw detector hit on one frame, before any OCR is attempted."""
    bounding_box: BoundingBox
    detector_confidence: float
    class_id: int
    crop: Any = None            # np.ndarray BGR crop, kept out of asdict()
    frame_index: int = 0
    source: str = "full_frame"  # "full_frame" or "vehicle_roi"

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("crop", None)
        return d


@dataclass
class PlateOCRCandidate:
    raw_text: str
    normalized_text: str
    ocr_confidence: float
    preprocessing_method: str
    detector_confidence: float
    combined_confidence: float
    frame_index: int = 0
    corrections: list = field(default_factory=list)   # [{"pos":i,"from":c,"to":c}]
    structure_matched: Optional[str] = None
    crop_sharpness: float = 0.0
    crop_area: int = 0
    rejection_reason: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PlateRecognitionResult:
    plate_number: str = ""              # normalized, DB lookup key ("" if none)
    display_plate: str = ""             # human-formatted, may include dashes
    status: str = RecognitionStatus.NOT_DETECTED.value
    overall_confidence: float = 0.0
    detector_confidence: float = 0.0
    ocr_confidence: float = 0.0
    consensus_count: int = 0
    total_frames: int = 0
    bounding_box: Optional[BoundingBox] = None
    rejection_reason: Optional[str] = None
    automatic: bool = True              # False once a human edits/enters it
    source: str = "auto"                # "auto" | "manual_correction" | "manual_entry"
    crop_path: Optional[str] = None
    debug_information: dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    def to_api_dict(self) -> dict:
        """Shape used by SSE/JSON responses to the frontend — no filesystem paths, no exceptions."""
        return {
            "plate": self.plate_number,
            "display_plate": self.display_plate,
            "status": self.status,
            "detector_confidence": round(self.detector_confidence, 3),
            "ocr_confidence": round(self.ocr_confidence, 3),
            "overall_confidence": round(self.overall_confidence, 3),
            "consensus_count": self.consensus_count,
            "total_frames": self.total_frames,
            "plate_crop_url": self.crop_path,
            "manual_confirmation_required": self.status != RecognitionStatus.CONFIRMED.value,
            "rejection_reason": self.rejection_reason,
        }

    @classmethod
    def manual(cls, normalized: str, display: str) -> "PlateRecognitionResult":
        return cls(
            plate_number=normalized,
            display_plate=display,
            status=RecognitionStatus.MANUAL.value,
            overall_confidence=1.0,
            automatic=False,
            source="manual_entry",
        )
