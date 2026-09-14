"""
Attribute pipeline orchestration, voting, and the failure paths.

Only the model interfaces are faked. The containment linkage, the voting
rule, the escape-class handling and the untrained-stub guard all run for
real — those rules are the feature.
"""
import numpy as np
import pytest

import attributes as attr
from attributes.consensus import decide
from attributes.models import (AttributeStatus, AttributeValue, ScoredLabel,
                               VehicleAttributeResult, VehicleLocalisation,
                               SOURCE_AUTO, SOURCE_MANUAL_CORRECTION,
                               SOURCE_MANUAL_ENTRY, ESCAPE_LABEL)


# ── doubles ───────────────────────────────────────────────────────────────

class FakeClassifier:
    """Stands in for AttributeClassifier. Only the forward pass is faked."""

    def __init__(self, predictions=None, trained=True, error=None,
                 version="fake-attr@test"):
        self.predictions = predictions or {}
        self.trained = trained
        self.error = error
        self.version = version
        self.calls = []

    def is_ready(self):
        return True

    def status(self):
        return {"trained": self.trained, "version": self.version}

    def predict(self, crop_bgr, heads=("colour", "type")):
        self.calls.append(tuple(heads))
        if self.error:
            raise RuntimeError(self.error)
        return {h: self.predictions[h] for h in heads if h in self.predictions}


class FakeLogoDetector:
    def __init__(self, detections=None, ready=True, model_path="fake-logo.pt"):
        self.detections = detections if detections is not None else []
        self._ready = ready
        self.model_path = model_path

    def is_ready(self):
        return self._ready

    def status(self):
        return {"model_present": self._ready}

    def detect(self, crop, frame_index=0, max_detections=1):
        return list(self.detections)


class FakeVehicleDetector:
    def __init__(self, boxes=None):
        self.boxes = boxes if boxes is not None else []

    def is_ready(self):
        return True

    def status(self):
        return {}

    def detect_vehicles(self, frame, frame_index=0):
        return list(self.boxes)

    def localise(self, frame, plate_box, min_contains_fraction=0.90, frame_index=0):
        from attributes.vehicle_detector import select_vehicle_for_plate
        loc = select_vehicle_for_plate(self.boxes, plate_box,
                                       min_contains_fraction, frame_index)
        if loc.found and frame is not None:
            x1, y1, x2, y2 = loc.box
            loc.crop = frame[y1:y2, x1:x2]
        return loc


def scored(*pairs):
    return [ScoredLabel(label, confidence) for label, confidence in pairs]


@pytest.fixture()
def settings():
    validated, _ = attr.settings.validate({
        "attr_capture_frame_count": 3,
        "attr_consensus_frames": 2,
        # Generous budget so timing never makes these tests flaky.
        "attr_latency_budget_ms": 5000.0,
    })
    return validated


@pytest.fixture()
def frame():
    return np.random.default_rng(7).integers(0, 255, (400, 600, 3), dtype=np.uint8)


CAR = {"box": (100, 100, 400, 320), "class_name": "car", "confidence": 0.9}
PLATE = (200, 260, 280, 290)


def build(settings, classifier=None, logo=None, vehicles=None):
    return attr.VehicleAttributePipeline(
        classifier or FakeClassifier(), logo or FakeLogoDetector(),
        vehicles or FakeVehicleDetector([CAR]), settings)


# ── happy path ────────────────────────────────────────────────────────────

def test_colour_and_type_are_read_from_the_vehicle_crop(settings, frame):
    classifier = FakeClassifier({"colour": scored(("red", 0.9), ("brown", 0.05)),
                                 "type": scored(("sedan", 0.8), ("suv", 0.1))})
    result = build(settings, classifier).process_frames(
        [frame, frame], plate_box=PLATE)

    assert result.colour.value == "red"
    assert result.colour.status == AttributeStatus.CONFIRMED.value
    assert result.colour.votes == 2
    assert result.type.value == "sedan"
    assert result.coco_class == "car"
    assert result.vehicle_box == CAR["box"]
    assert result.frames_with_vehicle == 2


def test_every_automatic_value_carries_the_auto_source(settings, frame):
    classifier = FakeClassifier({"colour": scored(("red", 0.9)),
                                 "type": scored(("sedan", 0.8))})
    result = build(settings, classifier).process_frames([frame, frame], plate_box=PLATE)
    assert result.colour.source == SOURCE_AUTO
    assert result.type.source == SOURCE_AUTO


def test_result_is_marked_advisory_in_the_api_payload(settings, frame):
    result = build(settings).process_frames([frame], plate_box=PLATE)
    assert result.to_api_dict()["advisory_only"] is True


# ── the coarse COCO class is authoritative ────────────────────────────────

def test_coco_bus_settles_the_type_without_consulting_the_head(settings, frame):
    bus = {"box": (100, 100, 400, 320), "class_name": "bus", "confidence": 0.93}
    classifier = FakeClassifier({"colour": scored(("white", 0.9)),
                                 "type": scored(("sedan", 0.99))})
    result = build(settings, classifier,
                   vehicles=FakeVehicleDetector([bus])).process_frames(
        [frame, frame], plate_box=PLATE)

    assert result.type.value == "bus"
    # The type head was never asked.
    assert all("type" not in call for call in classifier.calls)


def test_coco_car_defers_to_the_learned_type_head(settings, frame):
    classifier = FakeClassifier({"colour": scored(("red", 0.9)),
                                 "type": scored(("hatchback", 0.7))})
    result = build(settings, classifier).process_frames([frame, frame], plate_box=PLATE)
    assert result.type.value == "hatchback"
    assert any("type" in call for call in classifier.calls)


# ── brand is two-stage ────────────────────────────────────────────────────

def test_brand_is_read_from_the_badge_crop_not_the_body(settings, frame):
    badge = attr.LogoDetection(box=(10, 10, 60, 50), confidence=0.8,
                               crop=np.full((40, 50, 3), 200, np.uint8))
    classifier = FakeClassifier({"colour": scored(("red", 0.9)),
                                 "type": scored(("sedan", 0.8)),
                                 "brand": scored(("toyota", 0.9), ("lexus", 0.05))})
    result = build(settings, classifier,
                   logo=FakeLogoDetector([badge])).process_frames(
        [frame, frame], plate_box=PLATE)

    assert result.brand.value == "toyota"
    assert result.frames_with_logo == 2
    # The brand head was invoked on its own pass, separate from colour/type.
    assert ("brand",) in classifier.calls


def test_no_logo_means_unknown_brand_never_a_body_guess(settings, frame):
    classifier = FakeClassifier({"colour": scored(("red", 0.9)),
                                 "type": scored(("sedan", 0.8)),
                                 "brand": scored(("toyota", 0.99))})
    result = build(settings, classifier,
                   logo=FakeLogoDetector([])).process_frames(
        [frame, frame], plate_box=PLATE)

    assert result.brand.value is None
    assert result.brand.status == AttributeStatus.UNKNOWN.value
    assert result.brand.confidence == 0.0
    assert "not inferred from the body crop" in result.brand.rejection_reason
    assert ("brand",) not in classifier.calls


def test_a_badge_crop_too_small_to_carry_signal_is_skipped(settings, frame):
    tiny = attr.LogoDetection(box=(10, 10, 13, 13), confidence=0.9,
                              crop=np.full((3, 3, 3), 128, np.uint8))
    classifier = FakeClassifier({"colour": scored(("red", 0.9)),
                                 "brand": scored(("toyota", 0.99))})
    result = build(settings, classifier,
                   logo=FakeLogoDetector([tiny])).process_frames(
        [frame], plate_box=PLATE)
    assert result.brand.status == AttributeStatus.UNKNOWN.value
    assert result.frames_with_logo == 0


def test_brand_votes_only_across_frames_that_had_a_badge(settings, frame):
    """Brand's denominator is badge frames, not all frames."""
    badge = attr.LogoDetection(box=(10, 10, 60, 50), confidence=0.8,
                               crop=np.full((40, 50, 3), 200, np.uint8))

    class OneBadgeOnly(FakeLogoDetector):
        def detect(self, crop, frame_index=0, max_detections=1):
            return [badge] if frame_index == 0 else []

    settings["attr_consensus_frames"] = 1
    classifier = FakeClassifier({"colour": scored(("red", 0.9)),
                                 "type": scored(("sedan", 0.8)),
                                 "brand": scored(("kia", 0.8))})
    result = build(settings, classifier, logo=OneBadgeOnly()).process_frames(
        [frame, frame, frame], plate_box=PLATE)

    assert result.frames_with_vehicle == 3
    assert result.frames_with_logo == 1
    assert result.brand.votes == 1
    assert result.brand.total_frames == 1        # not 3
    assert result.colour.total_frames == 3


# ── failure paths ─────────────────────────────────────────────────────────

def test_no_vehicle_contains_the_plate_box(settings, frame):
    result = build(settings).process_frames([frame], plate_box=(500, 380, 560, 398))
    assert result.status == AttributeStatus.VEHICLE_NOT_LOCALISED.value
    for value in result.values().values():
        assert value.status == AttributeStatus.VEHICLE_NOT_LOCALISED.value
        assert value.value is None


def test_missing_plate_box_does_not_crash(settings, frame):
    result = build(settings).process_frames([frame], plate_box=None)
    assert result.status == AttributeStatus.VEHICLE_NOT_LOCALISED.value


def test_corrupt_frames_return_a_structured_failure(settings):
    result = build(settings).process_frames(
        [None, np.zeros((0, 0, 3), np.uint8)], plate_box=PLATE)
    assert result.status == AttributeStatus.FAILED.value


def test_classifier_error_never_propagates(settings, frame):
    classifier = FakeClassifier(error="backbone exploded")
    result = build(settings, classifier).process_frames([frame], plate_box=PLATE)
    assert result.colour.status == AttributeStatus.FAILED.value
    assert "backbone exploded" in result.colour.rejection_reason


def test_untrained_stub_reports_no_learned_prediction(settings, frame):
    """
    A random-weight backbone must never emit a label from a learned head.
    Colour is exempt only because it falls back to the HSV baseline, which
    is a real method with no weights — see the colour-backend tests below.
    """
    settings["attr_colour_backend"] = "head"        # no HSV fallback
    classifier = FakeClassifier({"colour": scored(("gold", 0.09)),
                                 "type": scored(("minivan", 0.11))},
                                trained=False)
    result = build(settings, classifier).process_frames([frame, frame], plate_box=PLATE)
    assert result.colour.status == AttributeStatus.MODEL_UNAVAILABLE.value
    assert result.colour.value is None
    assert "untrained" in result.colour.rejection_reason
    assert result.type.value is None


def test_an_untrained_head_falls_back_to_the_hsv_baseline(settings):
    """
    `auto` is the shipped default: colour works with or without a
    checkpoint, because the baseline needs none.
    """
    import numpy as np

    red = np.full((400, 600, 3), (30, 30, 220), np.uint8)
    classifier = FakeClassifier({"colour": scored(("gold", 0.09))}, trained=False)
    pipeline = build(settings, classifier)
    assert pipeline.colour_method() == "hsv"

    result = pipeline.process_frames([red, red], plate_box=PLATE)
    assert result.colour.value == "red"
    assert result.colour.status == AttributeStatus.CONFIRMED.value
    assert result.colour.model_version.startswith("hsv-baseline")
    # The stubbed head was never consulted for colour.
    assert all("colour" not in call for call in classifier.calls)


def test_a_trained_head_takes_precedence_over_the_baseline(settings):
    import numpy as np

    red = np.full((400, 600, 3), (30, 30, 220), np.uint8)
    classifier = FakeClassifier({"colour": scored(("blue", 0.9)),
                                 "type": scored(("suv", 0.8))}, trained=True)
    pipeline = build(settings, classifier)
    assert pipeline.colour_method() == "head"
    result = pipeline.process_frames([red, red], plate_box=PLATE)
    assert result.colour.value == "blue"


def test_the_colour_backend_can_be_forced_to_hsv(settings):
    import numpy as np

    blue_car = np.full((400, 600, 3), (220, 60, 40), np.uint8)
    settings["attr_colour_backend"] = "hsv"
    classifier = FakeClassifier({"colour": scored(("gold", 0.99))}, trained=True)
    result = build(settings, classifier).process_frames([blue_car], plate_box=PLATE)
    assert result.colour.value == "blue"


def test_untrained_stub_still_reports_the_authoritative_coco_type(settings, frame):
    """COCO is a trained model; only the learned heads are stubbed."""
    settings["attr_colour_backend"] = "head"        # isolate the type path
    truck = {"box": (100, 100, 400, 320), "class_name": "truck", "confidence": 0.9}
    classifier = FakeClassifier({"colour": scored(("red", 0.1))}, trained=False)
    result = build(settings, classifier,
                   vehicles=FakeVehicleDetector([truck])).process_frames(
        [frame, frame], plate_box=PLATE)
    assert result.type.value == "truck"
    assert result.type.status == AttributeStatus.CONFIRMED.value
    assert result.colour.value is None


def test_disabled_heads_are_not_invoked(settings, frame):
    settings["attr_brand_enabled"] = False
    settings["attr_type_enabled"] = False
    classifier = FakeClassifier({"colour": scored(("red", 0.9))})
    result = build(settings, classifier).process_frames([frame, frame], plate_box=PLATE)
    assert result.brand.status == AttributeStatus.MODEL_UNAVAILABLE.value
    assert result.type.status == AttributeStatus.MODEL_UNAVAILABLE.value
    assert result.colour.value == "red"
    assert all("type" not in call for call in classifier.calls)


def test_master_switch_disables_everything(settings, frame):
    settings["attr_enabled"] = False
    assert build(settings).process_frames(
        [frame], plate_box=PLATE).status == AttributeStatus.MODEL_UNAVAILABLE.value


def test_latency_budget_stops_the_pass(settings, frame):
    import time

    class SlowClassifier(FakeClassifier):
        def predict(self, crop_bgr, heads=("colour", "type")):
            time.sleep(0.08)
            return super().predict(crop_bgr, heads=heads)

    settings["attr_latency_budget_ms"] = 100.0
    settings["attr_capture_frame_count"] = 5
    classifier = SlowClassifier({"colour": scored(("red", 0.9))})
    result = build(settings, classifier).process_frames([frame] * 5, plate_box=PLATE)
    assert result.frames_processed < 5
    assert "budget" in (result.rejection_reason or "")


def test_reused_vehicle_boxes_skip_the_detector(settings, frame):
    """The gate already runs COCO on these frames; running it twice is waste."""
    vehicles = FakeVehicleDetector([])            # would find nothing itself
    classifier = FakeClassifier({"colour": scored(("blue", 0.9)),
                                 "type": scored(("suv", 0.8))})
    result = build(settings, classifier, vehicles=vehicles).process_frames(
        [frame, frame], plate_box=PLATE, vehicle_boxes_for_frame=lambda i: [CAR])
    assert result.colour.value == "blue"
    assert result.frames_with_vehicle == 2


# ── voting ────────────────────────────────────────────────────────────────

def vote(label, confidence, frame_index):
    return {"label": label, "confidence": confidence, "frame_index": frame_index}


def test_frame_agreement_dominates_a_single_confident_frame():
    result = decide("colour", [vote("red", 0.99, 0), vote("blue", 0.60, 1),
                               vote("blue", 0.61, 2)], 3, consensus_frames=2)
    assert result.value == "blue"
    assert result.votes == 2


def test_confirmation_needs_both_bars():
    enough = decide("colour", [vote("red", 0.9, 0), vote("red", 0.9, 1)], 2,
                    consensus_frames=2, min_confirm_confidence=0.55)
    assert enough.status == AttributeStatus.CONFIRMED.value

    thin = decide("colour", [vote("red", 0.9, 0)], 2, consensus_frames=2)
    assert thin.status == AttributeStatus.LOW_CONFIDENCE.value
    assert "agreeing frames" in thin.rejection_reason

    weak = decide("colour", [vote("red", 0.2, 0), vote("red", 0.2, 1)], 2,
                  consensus_frames=2, min_confirm_confidence=0.55)
    assert weak.status == AttributeStatus.LOW_CONFIDENCE.value
    assert "confidence" in weak.rejection_reason


def test_the_runner_up_is_preserved_as_an_alternative():
    result = decide("colour", [vote("red", 0.6, 0), vote("red", 0.6, 1),
                               vote("blue", 0.5, 2)], 3, consensus_frames=2)
    assert any(alt.label == "blue" for alt in result.top_k)


@pytest.mark.parametrize("attribute", ["colour", "type", "brand"])
def test_an_escape_class_win_is_reported_unknown_not_confirmed(attribute):
    escape = ESCAPE_LABEL[attribute]
    result = decide(attribute, [vote(escape, 0.95, 0), vote(escape, 0.95, 1)], 2,
                    consensus_frames=2, min_confirm_confidence=0.55)
    assert result.value == escape
    assert result.status == AttributeStatus.UNKNOWN.value
    assert result.is_escape_class is True
    assert "declined to commit" in result.rejection_reason


def test_repeated_variants_of_one_frame_count_as_one_vote():
    result = decide("colour", [vote("red", 0.9, 0), vote("red", 0.9, 0),
                               vote("red", 0.9, 0)], 3, consensus_frames=2)
    assert result.votes == 1
    assert result.status == AttributeStatus.LOW_CONFIDENCE.value


# ── provenance ────────────────────────────────────────────────────────────

def test_operator_correction_is_distinguishable_from_an_automatic_reading():
    automatic = AttributeValue(attribute="colour", value="silver", confidence=0.7,
                               status=AttributeStatus.CONFIRMED.value)
    corrected = AttributeValue.operator("colour", "white", automatic)
    assert corrected.source == SOURCE_MANUAL_CORRECTION
    assert corrected.value == "white"
    assert corrected.confidence == 0.0        # a human assertion has no probability
    assert automatic.source == SOURCE_AUTO


def test_operator_entry_with_no_prior_reading_is_manual_entry():
    assert AttributeValue.operator("brand", "innoson").source == SOURCE_MANUAL_ENTRY
    nothing = AttributeValue.unknown("brand", "no logo")
    assert AttributeValue.operator("brand", "innoson", nothing).source == SOURCE_MANUAL_ENTRY


def test_result_round_trips_through_json():
    import json
    original = VehicleAttributeResult(
        colour=AttributeValue(attribute="colour", value="red", confidence=0.8,
                              status=AttributeStatus.CONFIRMED.value, votes=2,
                              total_frames=3, top_k=scored(("red", 0.8))),
        type=AttributeValue.unknown("type", "declined"),
        brand=AttributeValue.unavailable("brand", "no model"),
        vehicle_box=(1, 2, 3, 4), coco_class="car", frames_processed=3)
    restored = VehicleAttributeResult.from_dict(json.loads(json.dumps(original.to_dict())))
    assert restored.colour.value == "red"
    assert restored.colour.votes == 2
    assert restored.vehicle_box == (1, 2, 3, 4)
    assert restored.brand.status == AttributeStatus.MODEL_UNAVAILABLE.value
