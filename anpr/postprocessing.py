"""
Nigerian plate text normalization, formatting, structure validation and
conservative ambiguous-character correction.

Design rules (see README_ANPR.md section "Safe character correction"):
  * The raw OCR text is always preserved alongside the corrected text.
  * A character is only corrected when the position it sits in is strongly
    implied by a *matched* plate structure (letter-zone vs digit-zone).
  * Every correction is recorded; candidates needing more than
    MAX_CORRECTIONS_DEFAULT edits are rejected outright rather than forced
    into a plausible-looking plate.
  * Matching a structure regex is evidence, not proof — it nudges
    confidence, it never manufactures a CONFIRMED result by itself.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

# ── Overlay / UI words that must never be treated as a plate reading ─────────
OVERLAY_WORDS = frozenset([
    "FACE", "DETECTED", "CAPTURING", "POSITION", "HOLD", "STILL",
    "FRAME", "CAMERA", "WAITING", "INSIDE", "DENIED", "GRANTED",
    "NIGERIA", "FEDERAL", "REPUBLIC", "EXPIRES", "EXPIRY",
])

_ALNUM_RE = re.compile(r"[^A-Z0-9]")

# ── Known Nigerian plate structures: (letters, digits, letters), ordered by
#    real-world frequency. Kept as an explicit list (not one broad regex) so
#    a valid but less-common category is never rejected just because it
#    doesn't fit the most common shape.
@dataclass(frozen=True)
class PlateStructure:
    name: str
    l1: int
    digits: int
    l2: int

    @property
    def length(self) -> int:
        return self.l1 + self.digits + self.l2

    def regex(self) -> re.Pattern:
        return re.compile(rf"^[A-Z]{{{self.l1}}}[0-9]{{{self.digits}}}[A-Z]{{{self.l2}}}$")


PLATE_STRUCTURES = [
    PlateStructure("standard-3-3-2", 3, 3, 2),   # AAA123BC  (most common — Lagos/Abuja private)
    PlateStructure("short-2-3-2",    2, 3, 2),    # AA123BC
    PlateStructure("long-3-4-2",     3, 4, 2),    # AAA1234BC
    PlateStructure("wide-2-4-2",     2, 4, 2),    # AA1234BC
    PlateStructure("gov-3-2-2",      3, 2, 2),    # AAA12BC
    PlateStructure("suffix3-3-3-3",  3, 3, 3),    # AAA123BCD
    PlateStructure("min-2-2-2",      2, 2, 2),    # AA12BC
    PlateStructure("single-1-3-2",   1, 3, 2),    # A123BC
    PlateStructure("suffix3-3-2-3",  3, 2, 3),    # AAA12BCD
    PlateStructure("suffix3-2-3-3",  2, 3, 3),    # AA123BCD
    PlateStructure("suffix3-2-2-3",  2, 2, 3),    # AA12BCD
]

_INT_TO_CHAR = {"0": "O", "1": "I", "5": "S", "6": "G", "8": "B", "3": "J", "4": "A", "2": "Z"}
_CHAR_TO_INT = {"O": "0", "I": "1", "S": "5", "G": "6", "B": "8", "J": "3", "A": "4", "Z": "2"}

MAX_CORRECTIONS_DEFAULT = 2
CORRECTION_CONFIDENCE_PENALTY = 0.08   # per corrected character
STRUCTURE_MATCH_BONUS = 0.06           # small — a regex match is evidence, not proof


def normalize_plate_for_storage(text: str) -> str:
    """Uppercase, strip everything except A-Z0-9. This is the DB lookup key."""
    if not text:
        return ""
    return _ALNUM_RE.sub("", text.upper())


def format_plate_for_display(normalized: str) -> str:
    """
    Human-readable form, e.g. 'ABC123DE' -> 'ABC-123-DE'.
    Falls back to the raw normalized string when no known structure matches.
    """
    if not normalized:
        return ""
    for s in PLATE_STRUCTURES:
        if len(normalized) == s.length and s.regex().match(normalized):
            return f"{normalized[:s.l1]}-{normalized[s.l1:s.l1+s.digits]}-{normalized[s.l1+s.digits:]}"
    return normalized


def is_overlay_word(clean: str) -> bool:
    return clean in OVERLAY_WORDS


def has_letters_and_digits(clean: str) -> bool:
    return any(c.isalpha() for c in clean) and any(c.isdigit() for c in clean)


def matches_any_structure(clean: str) -> Optional[PlateStructure]:
    for s in PLATE_STRUCTURES:
        if len(clean) == s.length and s.regex().match(clean):
            return s
    return None


def trim_to_structure(clean: str) -> str:
    """
    Strip OCR junk read from outside the plate number (state-strip text,
    screw covers): if the whole string matches no known structure but a
    prefix or suffix of it matches one *exactly* (zero corrections), return
    that window. Middle windows are never tried, and a string that already
    matches a structure is never shortened — so a valid 9-char plate can't
    be trimmed down to its 8-char prefix.
    """
    if not clean or matches_any_structure(clean) is not None:
        return clean
    for s in PLATE_STRUCTURES:
        if len(clean) <= s.length:
            continue
        for start in (0, len(clean) - s.length):
            window = clean[start:start + s.length]
            if s.regex().match(window):
                return window
    return clean


@dataclass
class CorrectionResult:
    text: str
    corrections: list          # [{"pos": int, "from": str, "to": str}]
    structure: Optional[str]   # matched structure name, or None
    rejected: bool
    rejection_reason: Optional[str] = None


def safe_correct(raw_text: str, max_corrections: int = MAX_CORRECTIONS_DEFAULT) -> CorrectionResult:
    """
    Conservative structure-guided correction.

    Only proposes a correction when the resulting string exactly matches a
    known Nigerian plate structure AND the number of characters changed is
    <= max_corrections. Otherwise the original (uncorrected) text is
    returned untouched — we never guess our way to a plausible plate.
    """
    clean = normalize_plate_for_storage(raw_text)
    if len(clean) < 5 or len(clean) > 10:
        return CorrectionResult(clean, [], None, rejected=False)

    # Already valid — no correction needed.
    exact = matches_any_structure(clean)
    if exact is not None:
        return CorrectionResult(clean, [], exact.name, rejected=False)

    best: Optional[CorrectionResult] = None
    for s in PLATE_STRUCTURES:
        if s.length != len(clean):
            continue
        corrected_chars = []
        corrections = []
        for j, c in enumerate(clean):
            in_letter_zone = j < s.l1 or j >= s.l1 + s.digits
            if in_letter_zone:
                new_c = _INT_TO_CHAR.get(c, c) if c.isdigit() else c
            else:
                new_c = _CHAR_TO_INT.get(c, c) if c.isalpha() else c
            corrected_chars.append(new_c)
            if new_c != c:
                corrections.append({"pos": j, "from": c, "to": new_c})
        corrected = "".join(corrected_chars)
        if not s.regex().match(corrected):
            continue   # correction still doesn't produce a valid structure — discard
        if len(corrections) > max_corrections:
            continue
        if best is None or len(corrections) < len(best.corrections):
            best = CorrectionResult(corrected, corrections, s.name, rejected=False)

    if best is not None:
        return best

    # Nothing correctable within budget — reject rather than force a guess.
    return CorrectionResult(
        clean, [], None, rejected=True,
        rejection_reason="No valid Nigerian plate structure within correction budget",
    )


def score_candidate(ocr_confidence: float, detector_confidence: float,
                     num_corrections: int, structure_matched: bool,
                     crop_sharpness_norm: float = 1.0) -> float:
    """
    Combined per-candidate confidence in [0, 1]. Corrections always reduce
    confidence; matching a structure gives only a small bonus so a
    low-confidence OCR read can never be laundered into a confident result
    purely by matching a regex.
    """
    base = 0.55 * ocr_confidence + 0.30 * detector_confidence + 0.15 * crop_sharpness_norm
    base -= num_corrections * CORRECTION_CONFIDENCE_PENALTY
    if structure_matched:
        base += STRUCTURE_MATCH_BONUS
    return max(0.0, min(1.0, base))
