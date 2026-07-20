"""
Unit tests for anpr.postprocessing — normalization, display formatting,
conservative character correction, and overlay-word rejection.

Pure logic, no camera/model mocking required.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from anpr.postprocessing import (
    normalize_plate_for_storage,
    format_plate_for_display,
    safe_correct,
    is_overlay_word,
    has_letters_and_digits,
    matches_any_structure,
    score_candidate,
)


def test_normalize_strips_punctuation_and_uppercases():
    assert normalize_plate_for_storage("abc-123-de") == "ABC123DE"
    assert normalize_plate_for_storage("AB C 12 3 D E") == "ABC123DE"
    assert normalize_plate_for_storage("") == ""
    assert normalize_plate_for_storage(None) == ""


def test_format_display_inserts_dashes_for_known_structure():
    assert format_plate_for_display("ABC123DE") == "ABC-123-DE"


def test_format_display_falls_back_when_no_structure_matches():
    assert format_plate_for_display("ZZ") == "ZZ"


def test_safe_correct_leaves_already_valid_plate_untouched():
    result = safe_correct("KJA456GH")
    assert result.text == "KJA456GH"
    assert result.corrections == []
    assert result.structure == "standard-3-3-2"
    assert not result.rejected


def test_safe_correct_fixes_single_ambiguous_character():
    # 'O' at a digit position (index 3) inside an otherwise valid structure
    result = safe_correct("KJAO56GH")
    assert result.text == "KJA056GH"
    assert len(result.corrections) == 1
    assert result.corrections[0] == {"pos": 3, "from": "O", "to": "0"}
    assert not result.rejected


def test_safe_correct_records_every_correction():
    # Two ambiguous chars: 'I' at digit pos -> '1', and 'S' at digit pos -> '5'
    result = safe_correct("KJAIS6GH")
    assert result.text == "KJA156GH"
    assert len(result.corrections) == 2
    positions = {c["pos"] for c in result.corrections}
    assert positions == {3, 4}


def test_safe_correct_rejects_when_too_many_corrections_needed():
    # Garbage that cannot be coerced into any structure within budget=2
    result = safe_correct("QXQXQXQXQX", max_corrections=2)
    assert result.rejected
    assert result.rejection_reason


def test_safe_correct_respects_configurable_max_corrections():
    # Needs 2 corrections — should succeed with budget 2, fail with budget 0
    ok = safe_correct("KJAIS6GH", max_corrections=2)
    assert not ok.rejected

    strict = safe_correct("KJAIS6GH", max_corrections=0)
    assert strict.rejected


def test_safe_correct_does_not_reject_less_common_valid_structure():
    # AA-12-BC (min-2-2-2), a valid but less common Nigerian structure
    result = safe_correct("AA12BC")
    assert result.text == "AA12BC"
    assert not result.rejected
    assert result.structure == "min-2-2-2"


def test_overlay_words_are_flagged():
    assert is_overlay_word("GRANTED")
    assert is_overlay_word("CAMERA")
    assert not is_overlay_word("KJA456GH")


def test_has_letters_and_digits():
    assert has_letters_and_digits("ABC123DE")
    assert not has_letters_and_digits("ABCDEFGH")
    assert not has_letters_and_digits("12345678")


def test_matches_any_structure_none_for_invalid_length():
    assert matches_any_structure("A") is None
    assert matches_any_structure("KJA456GH") is not None


def test_score_candidate_penalizes_corrections():
    clean = score_candidate(ocr_confidence=0.9, detector_confidence=0.9,
                             num_corrections=0, structure_matched=True)
    corrected = score_candidate(ocr_confidence=0.9, detector_confidence=0.9,
                                 num_corrections=2, structure_matched=True)
    assert corrected < clean


def test_score_candidate_structure_bonus_is_small_not_decisive():
    # A weak OCR read that happens to match a structure must not jump to
    # "confident" territory purely from the regex bonus.
    weak_matched = score_candidate(ocr_confidence=0.2, detector_confidence=0.2,
                                    num_corrections=0, structure_matched=True)
    assert weak_matched < 0.55


def test_score_candidate_bounded_0_to_1():
    lo = score_candidate(0.0, 0.0, 5, False)
    hi = score_candidate(1.0, 1.0, 0, True)
    assert 0.0 <= lo <= 1.0
    assert 0.0 <= hi <= 1.0
