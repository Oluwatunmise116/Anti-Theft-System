"""
Temporal voting for vehicle attributes.

Mirrors anpr/consensus.py. That module could not be called directly — it is
typed to PlateOCRCandidate/PlateRecognitionResult and scores plate-specific
evidence (crop area, character corrections) — so its DECISION RULE is
reproduced here with the same weights and the same security principle:

  * frame agreement dominates confidence, so one lucky high-confidence
    frame never outranks several consistent ones;
  * a value is never reported CONFIRMED unless BOTH the frame-agreement bar
    and the confidence bar are met;
  * the runner-up is preserved as an ambiguous alternative rather than
    silently discarded.

Each attribute votes independently. Colour and type vote across the same
best frames the plate used; brand votes only across frames where a logo was
actually detected, so a brand seen once in five frames is reported as one
vote out of one, not one out of five.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .models import (ATTRIBUTES, AttributeStatus, AttributeValue, ESCAPE_LABEL,
                     ScoredLabel, SOURCE_AUTO)

#: Same weighting as anpr.consensus.CandidateGroup.score(): agreement
#: dominates. The plate-specific crop-area and correction terms have no
#: attribute equivalent and are omitted rather than invented.
FRAME_AGREEMENT_WEIGHT = 10.0
CONFIDENCE_WEIGHT = 3.0


@dataclass
class AttributeVoteGroup:
    """All frames that produced one particular value for one attribute."""
    label: str
    confidences: List[float] = field(default_factory=list)
    frame_indices: List[int] = field(default_factory=list)

    @property
    def count(self) -> int:
        # Distinct frames, not distinct forward passes.
        return len(set(self.frame_indices))

    @property
    def avg_confidence(self) -> float:
        return sum(self.confidences) / len(self.confidences) if self.confidences else 0.0

    def score(self) -> float:
        return self.count * FRAME_AGREEMENT_WEIGHT + self.avg_confidence * CONFIDENCE_WEIGHT


def group_votes(per_frame: List[dict]) -> Dict[str, AttributeVoteGroup]:
    """
    `per_frame` is [{"label": str, "confidence": float, "frame_index": int}, ...]
    for ONE attribute across frames.
    """
    groups: Dict[str, AttributeVoteGroup] = {}
    for vote in per_frame:
        label = vote.get("label")
        if not label:
            continue
        group = groups.setdefault(label, AttributeVoteGroup(label))
        group.confidences.append(float(vote.get("confidence", 0.0) or 0.0))
        group.frame_indices.append(int(vote.get("frame_index", 0) or 0))
    return groups


def decide(attribute: str, per_frame: List[dict], total_frames: int,
           consensus_frames: int = 2, min_confirm_confidence: float = 0.55,
           top_k: Optional[List[ScoredLabel]] = None,
           model_version: Optional[str] = None) -> AttributeValue:
    """
    Rank grouped votes and decide CONFIRMED vs LOW_CONFIDENCE vs UNKNOWN.

    An escape-class win (`unknown` for colour, `other` for type and brand)
    is reported as UNKNOWN, never CONFIRMED: the head declining to commit is
    an honest non-answer, not a confident finding about the vehicle.
    """
    if not per_frame:
        return AttributeValue(
            attribute=attribute, status=AttributeStatus.UNKNOWN.value,
            total_frames=total_frames, model_version=model_version,
            rejection_reason="No frame produced a value for this attribute")

    groups = group_votes(per_frame)
    if not groups:
        return AttributeValue(
            attribute=attribute, status=AttributeStatus.UNKNOWN.value,
            total_frames=total_frames, model_version=model_version,
            rejection_reason="No usable votes for this attribute")

    ranked = sorted(groups.values(), key=lambda g: g.score(), reverse=True)
    winner = ranked[0]
    runner_up = ranked[1] if len(ranked) > 1 else None

    alternatives = list(top_k or [])
    if runner_up is not None and not any(a.label == runner_up.label for a in alternatives):
        alternatives.append(ScoredLabel(runner_up.label, runner_up.avg_confidence))

    if winner.label == ESCAPE_LABEL.get(attribute):
        return AttributeValue(
            attribute=attribute, value=winner.label,
            confidence=winner.avg_confidence,
            status=AttributeStatus.UNKNOWN.value,
            votes=winner.count, total_frames=total_frames, source=SOURCE_AUTO,
            top_k=alternatives[:3], model_version=model_version,
            rejection_reason=(
                f"The model predicted '{winner.label}' — it declined to commit to a "
                f"specific {attribute} on {winner.count}/{total_frames} frame(s)"))

    enough_frames = winner.count >= consensus_frames
    enough_confidence = winner.avg_confidence >= min_confirm_confidence

    if enough_frames and enough_confidence:
        status = AttributeStatus.CONFIRMED.value
        reason = None
    else:
        status = AttributeStatus.LOW_CONFIDENCE.value
        reasons = []
        if not enough_frames:
            reasons.append(f"only {winner.count}/{consensus_frames} agreeing frames")
        if not enough_confidence:
            reasons.append(f"confidence {winner.avg_confidence:.2f} < "
                           f"{min_confirm_confidence:.2f}")
        reason = "; ".join(reasons)

    return AttributeValue(
        attribute=attribute, value=winner.label, confidence=winner.avg_confidence,
        status=status, votes=winner.count, total_frames=total_frames,
        source=SOURCE_AUTO, top_k=alternatives[:3], rejection_reason=reason,
        model_version=model_version)
