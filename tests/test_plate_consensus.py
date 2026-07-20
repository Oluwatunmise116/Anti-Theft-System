"""
Unit tests for anpr.consensus — temporal voting across frames.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from anpr.consensus import decide, group_candidates
from anpr.models import PlateOCRCandidate, RecognitionStatus


def _candidate(text, frame_index, ocr_conf=0.8, det_conf=0.8, combined=None, corrections=None):
    combined = combined if combined is not None else (0.55 * ocr_conf + 0.30 * det_conf + 0.15)
    return PlateOCRCandidate(
        raw_text=text,
        normalized_text=text,
        ocr_confidence=ocr_conf,
        preprocessing_method="clahe_gray",
        detector_confidence=det_conf,
        combined_confidence=combined,
        frame_index=frame_index,
        corrections=corrections or [],
        crop_sharpness=100.0,
        crop_area=5000,
    )


def test_no_candidates_is_not_detected():
    result = decide([], total_frames=5)
    assert result.status == RecognitionStatus.NOT_DETECTED.value
    assert result.plate_number == ""


def test_three_agreeing_frames_confirms():
    candidates = [
        _candidate("KJA456GH", 0),
        _candidate("KJA456GH", 1),
        _candidate("KJA456GH", 2),
    ]
    result = decide(candidates, total_frames=5, consensus_frames=3)
    assert result.status == RecognitionStatus.CONFIRMED.value
    assert result.plate_number == "KJA456GH"
    assert result.consensus_count == 3


def test_single_weak_occurrence_does_not_beat_three_consistent_ones():
    candidates = [
        _candidate("KJA456GH", 0),
        _candidate("KJA456GH", 1),
        _candidate("KJA456GH", 2),
        _candidate("ZZZ999ZZ", 3, ocr_conf=0.95, det_conf=0.95),  # one strong but lone reading
    ]
    result = decide(candidates, total_frames=5, consensus_frames=3)
    assert result.plate_number == "KJA456GH"
    assert result.status == RecognitionStatus.CONFIRMED.value


def test_below_consensus_threshold_is_low_confidence_not_confirmed():
    candidates = [
        _candidate("KJA456GH", 0),
        _candidate("KJA456GH", 1),
    ]
    result = decide(candidates, total_frames=5, consensus_frames=3)
    assert result.status == RecognitionStatus.LOW_CONFIDENCE.value
    assert result.rejection_reason is not None


def test_low_confidence_never_silently_reported_as_confirmed():
    candidates = [_candidate("KJA456GH", i, ocr_conf=0.1, det_conf=0.1) for i in range(4)]
    result = decide(candidates, total_frames=4, consensus_frames=3, min_confirm_confidence=0.55)
    assert result.status == RecognitionStatus.LOW_CONFIDENCE.value


def test_similar_but_different_readings_are_not_silently_merged():
    candidates = [
        _candidate("KJA456GH", 0),
        _candidate("KJA456GH", 1),
        _candidate("KJA456GH", 2),
        _candidate("KJA456GK", 3),   # off by one char — distinct group
        _candidate("KJA456GK", 4),
    ]
    result = decide(candidates, total_frames=5, consensus_frames=3)
    assert result.plate_number == "KJA456GH"
    assert "ambiguous_alternative" in result.debug_information
    assert result.debug_information["ambiguous_alternative"]["normalized_text"] == "KJA456GK"


def test_multiple_frames_same_reading_multiple_variants_counts_frames_not_variants():
    # Same frame producing the same reading from 3 different preprocessing
    # variants must count as ONE agreeing frame, not three.
    candidates = [
        _candidate("KJA456GH", 0, ocr_conf=0.7),
        _candidate("KJA456GH", 0, ocr_conf=0.8),
        _candidate("KJA456GH", 0, ocr_conf=0.9),
    ]
    groups = group_candidates(candidates)
    assert groups["KJA456GH"].count == 1


def test_bounding_box_propagated_into_result():
    candidates = [_candidate("KJA456GH", i) for i in range(3)]
    result = decide(candidates, total_frames=3, consensus_frames=3, bounding_box=(1, 2, 3, 4))
    assert result.bounding_box == (1, 2, 3, 4)
