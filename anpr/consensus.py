"""
Temporal voting across multiple frames.

Security principle (see README_ANPR.md / gate security requirements): one
weak occurrence must never outrank several consistent, good-quality
occurrences, and a low-confidence reading must never be silently reported
as CONFIRMED. Similar-but-different readings are not merged — they are kept
as separate groups and the runner-up is recorded as an ambiguous
alternative so an operator can see it, rather than the system silently
picking one.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .models import PlateOCRCandidate, PlateRecognitionResult, RecognitionStatus


@dataclass
class CandidateGroup:
    normalized_text: str
    candidates: List[PlateOCRCandidate] = field(default_factory=list)

    @property
    def count(self) -> int:
        # Count distinct frames that produced this reading, not distinct
        # preprocessing variants of the same frame.
        return len({c.frame_index for c in self.candidates})

    @property
    def avg_ocr_confidence(self) -> float:
        return sum(c.ocr_confidence for c in self.candidates) / len(self.candidates)

    @property
    def avg_detector_confidence(self) -> float:
        return sum(c.detector_confidence for c in self.candidates) / len(self.candidates)

    @property
    def avg_combined_confidence(self) -> float:
        return sum(c.combined_confidence for c in self.candidates) / len(self.candidates)

    @property
    def max_crop_area(self) -> int:
        return max((c.crop_area for c in self.candidates), default=0)

    @property
    def avg_corrections(self) -> float:
        return sum(len(c.corrections) for c in self.candidates) / len(self.candidates)

    @property
    def display_plate(self) -> str:
        from .postprocessing import format_plate_for_display
        return format_plate_for_display(self.normalized_text)

    def score(self) -> float:
        """
        Ranking score for choosing the winning candidate group. Frame
        agreement dominates — a candidate seen 3+ times beats a candidate
        seen once even with a slightly higher single-frame confidence.
        """
        return (
            self.count * 10.0
            + self.avg_combined_confidence * 3.0
            + min(self.max_crop_area / 5000.0, 2.0)
            - self.avg_corrections * 0.5
        )


def group_candidates(candidates: List[PlateOCRCandidate]) -> Dict[str, CandidateGroup]:
    groups: Dict[str, CandidateGroup] = {}
    for c in candidates:
        if not c.normalized_text or c.rejection_reason:
            continue
        g = groups.setdefault(c.normalized_text, CandidateGroup(c.normalized_text))
        g.candidates.append(c)
    return groups


def decide(
    candidates: List[PlateOCRCandidate],
    total_frames: int,
    consensus_frames: int = 3,
    min_confirm_confidence: float = 0.55,
    bounding_box=None,
) -> PlateRecognitionResult:
    """
    Score grouped candidates and decide CONFIRMED vs LOW_CONFIDENCE vs
    NOT_DETECTED. Never returns CONFIRMED unless both the frame-agreement
    threshold and the confidence threshold are met.
    """
    groups = group_candidates(candidates)

    if not groups:
        return PlateRecognitionResult(
            status=RecognitionStatus.NOT_DETECTED.value,
            total_frames=total_frames,
            rejection_reason="No plausible plate candidates in any frame",
            bounding_box=bounding_box,
        )

    ranked = sorted(groups.values(), key=lambda g: g.score(), reverse=True)
    winner = ranked[0]
    runner_up = ranked[1] if len(ranked) > 1 else None

    debug_info = {
        "candidate_groups": [
            {
                "normalized_text": g.normalized_text,
                "display_plate": g.display_plate,
                "frame_count": g.count,
                "avg_ocr_confidence": round(g.avg_ocr_confidence, 3),
                "avg_detector_confidence": round(g.avg_detector_confidence, 3),
                "avg_combined_confidence": round(g.avg_combined_confidence, 3),
                "avg_corrections": round(g.avg_corrections, 2),
                "score": round(g.score(), 3),
            }
            for g in ranked[:5]
        ],
    }
    if runner_up is not None:
        debug_info["ambiguous_alternative"] = {
            "normalized_text": runner_up.normalized_text,
            "frame_count": runner_up.count,
            "avg_combined_confidence": round(runner_up.avg_combined_confidence, 3),
        }

    confirmed = (
        winner.count >= consensus_frames
        and winner.avg_combined_confidence >= min_confirm_confidence
    )

    status = RecognitionStatus.CONFIRMED if confirmed else RecognitionStatus.LOW_CONFIDENCE
    rejection_reason = None
    if not confirmed:
        reasons = []
        if winner.count < consensus_frames:
            reasons.append(
                f"only {winner.count}/{consensus_frames} agreeing frames"
            )
        if winner.avg_combined_confidence < min_confirm_confidence:
            reasons.append(
                f"confidence {winner.avg_combined_confidence:.2f} < {min_confirm_confidence:.2f}"
            )
        rejection_reason = "; ".join(reasons)

    return PlateRecognitionResult(
        plate_number=winner.normalized_text,
        display_plate=winner.display_plate,
        status=status.value,
        overall_confidence=winner.avg_combined_confidence,
        detector_confidence=winner.avg_detector_confidence,
        ocr_confidence=winner.avg_ocr_confidence,
        consensus_count=winner.count,
        total_frames=total_frames,
        bounding_box=bounding_box,
        rejection_reason=rejection_reason,
        automatic=True,
        source="auto",
        debug_information=debug_info,
    )
