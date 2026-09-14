"""
The HSV colour baseline.

This is the number a learned colour head has to beat, so its behaviour is
pinned here: it must be exactly right on unambiguous input, and it must
decline rather than guess when the evidence is bad.
"""
import numpy as np
import pytest

from attributes.colour_baseline import (BODY_REGION, CLIPPED_CHANNEL,
                                        HSVColourBackend, MAX_CLIPPED_FRACTION,
                                        body_region, estimate_colour)
from attributes.models import COLOURS


def patch(bgr, size=300):
    return np.full((size, size, 3), bgr, dtype=np.uint8)


# ── unambiguous input ─────────────────────────────────────────────────────

@pytest.mark.parametrize("bgr,expected", [
    ((30, 30, 220), "red"),
    ((220, 60, 40), "blue"),
    ((50, 180, 50), "green"),
    ((40, 220, 230), "yellow"),
    ((245, 245, 245), "white"),
    ((15, 15, 15), "black"),
    ((190, 190, 190), "silver"),
    ((128, 128, 128), "grey"),
    ((40, 90, 210), "orange"),
    ((30, 60, 120), "brown"),
])
def test_solid_colours_are_identified_exactly(bgr, expected):
    estimate = estimate_colour(patch(bgr))
    assert estimate.label == expected
    assert estimate.confidence == pytest.approx(1.0, abs=0.02)


def test_every_output_is_in_the_application_label_space():
    rng = np.random.default_rng(5)
    for _ in range(25):
        bgr = tuple(int(v) for v in rng.integers(0, 255, 3))
        assert estimate_colour(patch(bgr)).label in COLOURS


def test_a_very_dark_saturated_colour_reads_as_black():
    """A near-black navy is black at a gate, not blue."""
    assert estimate_colour(patch((40, 14, 10))).label == "black"


def test_the_palette_never_leaves_the_application_label_space():
    """
    Emitting a name outside COLOURS silently turns a correct answer into
    `unknown`. This caught exactly that: purple and pink had hue bands but
    no place in the label space.
    """
    from attributes.colour_baseline import HUE_BANDS, PIXEL_CLASSES

    assert set(PIXEL_CLASSES) <= set(COLOURS)
    assert {name for _low, _high, name in HUE_BANDS} <= set(PIXEL_CLASSES)
    # Every non-escape colour should be reachable by some rule.
    assert set(COLOURS) - {"unknown"} == set(PIXEL_CLASSES)


# ── declining rather than guessing ────────────────────────────────────────

def test_random_noise_is_rejected():
    noise = np.random.default_rng(0).integers(0, 255, (300, 400, 3), dtype=np.uint8)
    estimate = estimate_colour(noise)
    assert estimate.label == "unknown"
    assert "holds" in estimate.reason


def test_a_two_way_split_below_support_is_unknown():
    half = np.zeros((300, 300, 3), np.uint8)
    half[:, :150] = (30, 30, 220)          # red
    half[:, 150:] = (220, 60, 40)          # blue
    # Each colour holds ~50%, which clears the default support bar.
    assert estimate_colour(half, min_support=0.28).label in ("red", "blue")
    # Demanding a supermajority makes it decline instead.
    assert estimate_colour(half, min_support=0.75).label == "unknown"


def test_a_blown_out_body_is_unjudgeable():
    """
    Sensor-clipped pixels carry the colour of the clipping, not the paint.
    A red bonnet in direct sun reads white; saying so would be wrong.
    """
    blown = patch((252, 252, 255))
    estimate = estimate_colour(blown)
    assert estimate.label == "unknown"
    assert estimate.clipped_fraction > MAX_CLIPPED_FRACTION
    assert "blown out" in estimate.reason


def test_clipped_pixels_do_not_vote():
    """A mostly-red car with a clipped highlight must not become white."""
    image = patch((30, 30, 220))
    # Inside the body region, so it is not simply cropped away.
    top, bottom, _left, _right = BODY_REGION
    band = int(300 * top) + 5
    image[band:band + 60, :] = (255, 255, 255)
    estimate = estimate_colour(image)
    assert estimate.label == "red"
    assert estimate.clipped_fraction > 0.1
    # A thin edge remains from resize interpolation at the highlight
    # boundary; what matters is that the highlight cannot outvote the paint.
    white_share = estimate.distribution.get("white", 0) / estimate.body_pixels
    assert white_share < 0.05
    assert estimate.confidence > 0.9


def test_empty_and_malformed_input_is_handled():
    assert estimate_colour(None).label == "unknown"
    assert estimate_colour(np.zeros((0, 0, 3), np.uint8)).label == "unknown"
    assert estimate_colour(np.zeros((10, 10), np.uint8)).label == "unknown"
    assert estimate_colour(np.zeros((2, 2, 3), np.uint8)).label == "unknown"


# ── body region ───────────────────────────────────────────────────────────

def test_the_body_region_excludes_roof_and_wheels():
    image = np.zeros((400, 400, 3), np.uint8)
    image[:, :] = (30, 30, 220)            # red body
    image[:80, :] = (250, 250, 250)        # sky / windscreen band
    image[320:, :] = (20, 20, 20)          # wheels / shadow band
    body = body_region(image)
    assert body.shape[0] < image.shape[0]
    assert estimate_colour(image).label == "red"


def test_the_body_region_fractions_are_sane():
    top, bottom, left, right = BODY_REGION
    assert 0.0 < top < bottom < 1.0
    assert 0.0 < left < right < 1.0


# ── backend contract ──────────────────────────────────────────────────────

def test_the_backend_returns_the_pipeline_contract():
    label, confidence, top_k, reason = HSVColourBackend().predict(patch((30, 30, 220)))
    assert label == "red"
    assert 0.0 <= confidence <= 1.0
    assert top_k and top_k[0].label == "red"
    assert reason


@pytest.mark.parametrize("seed", range(6))
def test_noise_never_reaches_a_confirmable_confidence(seed):
    """
    Uniform noise is not a car. The baseline may name a weak favourite —
    hue bands are not equal-width, so noise leans green — but it must never
    reach the confidence the pipeline requires to CONFIRM a colour.
    """
    noise = np.random.default_rng(seed).integers(0, 255, (300, 300, 3), dtype=np.uint8)
    _label, confidence, _top_k, reason = HSVColourBackend().predict(noise)
    assert confidence < 0.55
    assert reason


def test_the_backend_needs_no_checkpoint():
    backend = HSVColourBackend()
    assert backend.trained is True
    assert backend.version.startswith("hsv-baseline")


def test_the_baseline_is_fast_enough_for_the_budget():
    """It must be negligible against the 150 ms attribute budget."""
    import time

    backend = HSVColourBackend()
    crop = np.random.default_rng(2).integers(0, 255, (400, 600, 3), dtype=np.uint8)
    for _ in range(3):
        backend.predict(crop)
    timings = []
    for _ in range(30):
        start = time.perf_counter()
        backend.predict(crop)
        timings.append((time.perf_counter() - start) * 1000)
    assert sorted(timings)[len(timings) // 2] < 20.0


def test_clipping_threshold_is_below_full_saturation():
    assert 200 < CLIPPED_CHANNEL <= 255
