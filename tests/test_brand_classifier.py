"""
Whole-vehicle brand backend (lamnt2008/car_brands_classification).

The BEiT forward pass is faked; the make->brand aggregation, the backend
selection, the vote denominator and the failure paths run for real.
"""
import numpy as np
import pytest

import attributes as attr
from attributes.brand_classifier import (MAKE_PREFIXES, VehicleBrandClassifier,
                                         aggregate_to_brands, brand_for_model_label)
from attributes.models import BRANDS, AttributeStatus, ScoredLabel

from test_attribute_pipeline import (CAR, PLATE, FakeClassifier, FakeLogoDetector,
                                     FakeVehicleDetector, scored)


class FakeBrandModel:
    def __init__(self, prediction=None, present=True, error=None):
        self.prediction = prediction or scored(("toyota", 0.9), ("lexus", 0.05))
        self.present = present
        self.error = error
        self.version = "fake-brand@test"
        self.crops = []

    def model_present(self):
        return self.present

    def is_ready(self):
        return self.present

    def status(self):
        return {"model_present": self.present}

    def predict(self, crop, k=3):
        self.crops.append(crop.shape)
        if self.error:
            raise RuntimeError(self.error)
        return list(self.prediction)


@pytest.fixture()
def settings():
    validated, _ = attr.settings.validate({
        "attr_capture_frame_count": 3,
        "attr_consensus_frames": 2,
        "attr_latency_budget_ms": 5000.0,
    })
    return validated


@pytest.fixture()
def frame():
    return np.random.default_rng(7).integers(0, 255, (400, 600, 3), dtype=np.uint8)


def build(settings, brand_model, classifier=None, logo=None):
    return attr.VehicleAttributePipeline(
        classifier or FakeClassifier(), logo or FakeLogoDetector(),
        FakeVehicleDetector([CAR]), settings, brand_model=brand_model)


# ── label aggregation ─────────────────────────────────────────────────────

def test_make_model_labels_map_onto_the_closed_brand_space():
    assert brand_for_model_label("ToyotaCorollaAltis") == "toyota"
    assert brand_for_model_label("MercedesBenzGLC") == "mercedes-benz"
    assert brand_for_model_label("BMWX1,2,3,5,7") == "bmw"
    assert brand_for_model_label("VinfastFadil") == "other"
    assert brand_for_model_label("SomethingNew") == "other"


def test_every_mapped_brand_is_in_the_label_space():
    assert {brand for _, brand in MAKE_PREFIXES} <= set(BRANDS)


def test_probabilities_are_summed_per_make():
    id2label = {0: "ToyotaCamry", 1: "ToyotaVios", 2: "LexusRX", 3: "KiaMorning"}
    top = aggregate_to_brands([0.3, 0.3, 0.35, 0.05], id2label)
    # Lexus has the single best model class, Toyota the best make.
    assert top[0].label == "toyota"
    assert top[0].confidence == pytest.approx(0.6)
    assert [t.label for t in top] == ["toyota", "lexus", "kia"]


# ── backend selection ─────────────────────────────────────────────────────

def test_auto_uses_the_vehicle_model_when_its_weights_are_present(settings):
    assert build(settings, FakeBrandModel(present=True)).brand_method() == "vehicle"
    assert build(settings, FakeBrandModel(present=False)).brand_method() == "logo"
    assert build(settings, None).brand_method() == "logo"


def test_the_backend_can_be_forced_to_logo(settings, frame):
    settings["attr_brand_backend"] = "logo"
    model = FakeBrandModel()
    result = build(settings, model).process_frames([frame], plate_box=PLATE)
    assert model.crops == []
    assert result.brand.status == AttributeStatus.UNKNOWN.value


def test_an_invalid_backend_falls_back_to_auto():
    validated, warnings = attr.settings.validate({"attr_brand_backend": "magic"})
    assert validated["attr_brand_backend"] == "auto"
    assert any("attr_brand_backend" in w for w in warnings)


# ── pipeline ──────────────────────────────────────────────────────────────

def test_brand_is_read_from_the_vehicle_crop_without_a_logo(settings, frame):
    model = FakeBrandModel()
    result = build(settings, model, logo=FakeLogoDetector([])).process_frames(
        [frame, frame], plate_box=PLATE)

    x1, y1, x2, y2 = CAR["box"]
    assert model.crops == [(y2 - y1, x2 - x1, 3)] * 2
    assert result.brand.value == "toyota"
    assert result.brand.status == AttributeStatus.CONFIRMED.value
    assert result.brand.votes == 2
    assert result.brand.total_frames == 2
    assert result.model_versions["brand"] == "fake-brand@test"


def test_an_untrained_backbone_does_not_suppress_the_brand_model(settings, frame):
    result = build(settings, FakeBrandModel(),
                   classifier=FakeClassifier(trained=False)).process_frames(
        [frame, frame], plate_box=PLATE)
    assert result.brand.value == "toyota"
    # ...while the stub's own heads still report nothing.
    assert result.type.value is None


def test_low_confidence_brand_is_not_confirmed(settings, frame):
    model = FakeBrandModel(scored(("mazda", 0.40), ("toyota", 0.35)))
    result = build(settings, model).process_frames([frame, frame], plate_box=PLATE)
    assert result.brand.value == "mazda"
    assert result.brand.status == AttributeStatus.LOW_CONFIDENCE.value


def test_a_brand_model_error_never_propagates(settings, frame):
    result = build(settings, FakeBrandModel(error="boom")).process_frames(
        [frame], plate_box=PLATE)
    assert result.brand.status == AttributeStatus.FAILED.value


def test_missing_weights_raise_a_structured_error(tmp_path):
    model = VehicleBrandClassifier(str(tmp_path / "nowhere"))
    assert model.is_ready() is False
    assert "prepare_brand_model.py" in model.status()["load_error"]


# ── real weights ──────────────────────────────────────────────────────────

@pytest.mark.real_models
def test_real_model_labels_all_map_to_a_brand():
    model = VehicleBrandClassifier(attr.settings.resolve_path(
        attr.settings.DEFAULTS["attr_brand_model_path"]))
    if not model.model_present():
        pytest.skip("run scripts/prepare_brand_model.py first")
    assert model.is_ready()
    assert len(model.id2label) == 107
    unmapped = {l for l in model.id2label.values()
                if brand_for_model_label(l) == "other"}
    assert all(l.startswith(("Ferrari", "Vinfast")) for l in unmapped)

    top = model.predict(np.full((200, 300, 3), 127, np.uint8))
    assert len(top) == 3 and all(isinstance(t, ScoredLabel) for t in top)
    assert sum(t.confidence for t in top) <= 1.0 + 1e-5
