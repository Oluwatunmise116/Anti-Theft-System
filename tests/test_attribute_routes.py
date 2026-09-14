"""
Route-level behaviour for vehicle attributes.

The load-bearing assertions here are the integration invariants: an
attribute mismatch routes to operator confirmation and NEVER to allow/deny,
a missing model never blocks the workflow, and an operator edit is always
distinguishable from an automatic reading.

Only hardware/model interfaces are faked; the database is real temporary
SQLite, matching the existing test style.
"""
import json

import pytest

import app as application
import attributes as va
import database as db
import gate_manager as gm
from attributes.models import (AttributeStatus, AttributeValue, ScoredLabel,
                               SOURCE_AUTO, SOURCE_MANUAL_CORRECTION,
                               VehicleAttributeResult)


from werkzeug.security import generate_password_hash

from gate_helpers import staff_client


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "routes.db"))
    db.init_db()
    application.app.config["TESTING"] = True
    with staff_client(application.app) as c:
        yield c


def result(colour="red", body_type="sedan", brand="toyota",
           status=AttributeStatus.CONFIRMED.value):
    def value(attribute, label, confidence):
        return AttributeValue(attribute=attribute, value=label, confidence=confidence,
                              status=status, votes=2, total_frames=3, source=SOURCE_AUTO)
    out = VehicleAttributeResult(
        colour=value("colour", colour, 0.8), type=value("type", body_type, 0.75),
        brand=value("brand", brand, 0.7), vehicle_box=(1, 2, 3, 4), coco_class="car",
        frames_processed=3, frames_with_vehicle=3, frames_with_logo=2,
        processing_time_ms=88.0, model_versions={"attributes": "mnv3@arch1"})
    out.brand.top_k = [ScoredLabel(brand, 0.7), ScoredLabel("lexus", 0.1)]
    out.status = out.roll_up_status()
    return out


def seed(cid, plate="KJA456GH", plate_status="CONFIRMED", attribute_result=None):
    gm.temp_set(cid, "plate_number", plate)
    gm.temp_set(cid, "plate_result", {"plate_number": plate, "status": plate_status,
                                      "overall_confidence": 0.9})
    gm.temp_set(cid, "plate_confidence", 0.9)
    if attribute_result is not None:
        gm.temp_set(cid, "vehicle_attributes", attribute_result.to_dict())


def open_trip(**kwargs):
    return db.create_trip("KJA456GH", None, None, None,
                          attributes=db.build_attribute_columns(result(**kwargs)))


# ── model status and vocabularies ─────────────────────────────────────────

def test_model_status_reports_attribute_readiness(client):
    payload = client.get("/gate/model/status").get_json()
    attrs = payload["vehicle_attributes"]
    assert "classifier" in attrs and "logo_detector" in attrs
    assert set(attrs["heads_enabled"]) == {"colour", "type", "brand"}
    # Absolute filesystem paths must not leak into an API response body.
    assert "/home/" not in json.dumps(attrs.get("heads_enabled"))


def test_options_route_serves_the_closed_label_spaces(client):
    payload = client.get("/gate/attributes/options").get_json()
    assert [c["label"] for c in payload["colours"]] == va.COLOURS
    assert [t["label"] for t in payload["types"]] == va.TYPES
    assert [b["label"] for b in payload["brands"]] == va.BRANDS


# ── entry ─────────────────────────────────────────────────────────────────

def test_entry_stores_attributes_with_provenance(client):
    cid = gm.new_capture_id()
    seed(cid, attribute_result=result())
    response = client.post("/gate/entry/confirm",
                           json={"capture_id": cid, "plate": "KJA456GH", "accept_unresolved": True})
    assert response.status_code == 200
    trip = db.get_trip(response.get_json()["trip_id"])
    assert trip["vehicle_color"] == "red"
    assert trip["vehicle_color_source"] == SOURCE_AUTO
    assert trip["vehicle_color_votes"] == 2
    assert trip["attr_status"] == "CONFIRMED"


def test_entry_completes_with_no_attribute_model(client):
    """A missing attribute model must not block the gate workflow."""
    cid = gm.new_capture_id()
    seed(cid, attribute_result=VehicleAttributeResult.disabled())
    response = client.post("/gate/entry/confirm",
                           json={"capture_id": cid, "plate": "KJA456GH", "accept_unresolved": True})
    assert response.status_code == 200
    trip = db.get_trip(response.get_json()["trip_id"])
    assert trip["vehicle_color"] is None
    assert trip["plate_number"] == "KJA456GH"


def test_entry_completes_when_the_vehicle_could_not_be_localised(client):
    cid = gm.new_capture_id()
    seed(cid, attribute_result=VehicleAttributeResult.vehicle_not_localised(
        3, "No vehicle box contains the plate box"))
    response = client.post("/gate/entry/confirm",
                           json={"capture_id": cid, "plate": "KJA456GH", "accept_unresolved": True})
    assert response.status_code == 200
    trip = db.get_trip(response.get_json()["trip_id"])
    assert trip["attr_status"] == "VEHICLE_NOT_LOCALISED"
    assert trip["vehicle_color"] is None


def test_entry_with_no_attributes_at_all(client):
    cid = gm.new_capture_id()
    seed(cid)
    assert client.post("/gate/entry/confirm",
                       json={"capture_id": cid, "plate": "KJA456GH", "accept_unresolved": True}).status_code == 200


def test_operator_correction_at_entry_is_marked_as_such(client):
    cid = gm.new_capture_id()
    seed(cid, attribute_result=result())
    response = client.post("/gate/entry/confirm", json={
        "capture_id": cid, "plate": "KJA456GH", "accept_unresolved": True,
        "attribute_corrections": {"colour": "white"}})
    trip = db.get_trip(response.get_json()["trip_id"])
    assert trip["vehicle_color"] == "white"
    assert trip["vehicle_color_source"] == SOURCE_MANUAL_CORRECTION
    assert trip["vehicle_color_confidence"] is None
    assert trip["vehicle_type_source"] == SOURCE_AUTO


def test_a_correction_outside_the_label_space_is_rejected(client):
    """An operator cannot silently widen a closed label space."""
    cid = gm.new_capture_id()
    seed(cid, attribute_result=result())
    response = client.post("/gate/entry/confirm", json={
        "capture_id": cid, "plate": "KJA456GH", "accept_unresolved": True,
        "attribute_corrections": {"brand": "delorean"}})
    assert response.status_code == 400
    assert "not a valid brand" in response.get_json()["error"]


def test_attributes_do_not_weaken_plate_confirmation(client):
    """A low-confidence plate still requires explicit operator confirmation."""
    cid = gm.new_capture_id()
    seed(cid, plate_status="LOW_CONFIDENCE", attribute_result=result())
    response = client.post("/gate/entry/confirm",
                           json={"capture_id": cid, "plate": "KJA456GH", "accept_unresolved": True})
    assert response.status_code == 409
    assert response.get_json()["requires_confirmation"] is True


# ── exit comparison ───────────────────────────────────────────────────────

def test_exit_lookup_returns_stored_entry_attributes(client):
    open_trip()
    payload = client.get("/gate/exit/lookup?plate=KJA456GH").get_json()
    assert payload["entry_attributes"]["colour"] == "red"
    assert payload["entry_attributes"]["colour_source"] == SOURCE_AUTO


def test_matching_attributes_need_no_confirmation(client):
    open_trip()
    cid = gm.new_capture_id()
    gm.temp_set(cid, "vehicle_attributes", result().to_dict())
    comparison = client.get(
        f"/gate/exit/lookup?plate=KJA456GH&capture_id={cid}").get_json()["attribute_comparison"]
    assert comparison["mismatched"] == []
    assert comparison["requires_operator_confirmation"] is False


def test_a_mismatch_routes_to_operator_confirmation(client):
    trip_id = open_trip()
    cid = gm.new_capture_id()
    gm.temp_set(cid, "vehicle_attributes", result(colour="blue").to_dict())
    payload = client.get(
        f"/gate/exit/lookup?plate=KJA456GH&capture_id={cid}").get_json()

    comparison = payload["attribute_comparison"]
    assert comparison["mismatched"] == ["colour"]
    assert comparison["requires_operator_confirmation"] is True
    assert comparison["advisory_only"] is True

    trip = db.get_trip(trip_id)
    assert trip["color_match"] == 0
    assert trip["type_match"] == 1
    assert trip["attr_mismatch_flags_json"] == ["colour"]
    # A mismatch changes nothing about the trip's state.
    assert trip["status"] == "INSIDE"


def test_an_unread_attribute_is_not_a_mismatch(client):
    """An absent measurement is not evidence of a swapped vehicle."""
    open_trip()
    cid = gm.new_capture_id()
    no_brand = result()
    no_brand.brand = AttributeValue.unknown("brand", "no logo detected")
    gm.temp_set(cid, "vehicle_attributes", no_brand.to_dict())
    comparison = client.get(
        f"/gate/exit/lookup?plate=KJA456GH&capture_id={cid}").get_json()["attribute_comparison"]
    brand_row = [c for c in comparison["comparisons"] if c["attribute"] == "brand"][0]
    assert brand_row["match"] is None
    assert "brand" not in comparison["mismatched"]
    assert comparison["requires_operator_confirmation"] is False


def test_an_attribute_mismatch_never_grants_or_denies_an_exit(client):
    """
    The load-bearing integration invariant: attributes are advisory. A total
    mismatch must not deny an otherwise-verified exit, and must not grant an
    unverified one.
    """
    trip_id = db.create_trip(
        "KJA456GH", None, None, None, attributes=db.build_attribute_columns(result()),
        identity={"identity_status": db.IDENTITY_GUEST,
                  "verification_mode": db.MODE_PASSCODE,
                  "passcode_hash": generate_password_hash("4821")})
    cid = gm.new_capture_id()
    gm.temp_set(cid, "vehicle_attributes",
                result(colour="blue", body_type="bus", brand="ford").to_dict())
    payload = client.get(
        f"/gate/exit/lookup?plate=KJA456GH&capture_id={cid}").get_json()
    assert len(payload["attribute_comparison"]["mismatched"]) == 3

    # Total mismatch, no verification -> still refused, on the exit rule,
    # not on the attributes.
    assert client.post("/gate/exit/confirm", json={"trip_id": trip_id}).status_code == 403

    # A verified trip passcode still grants the exit DESPITE the total mismatch.
    assert client.post("/gate/exit/passcode/verify",
                       json={"trip_id": trip_id, "passcode": "4821"}).get_json()["valid"] is True
    granted = client.post("/gate/exit/confirm", json={"trip_id": trip_id})
    assert granted.status_code == 200
    assert granted.get_json()["decision"] == "GRANTED"


def test_exit_lookup_without_a_capture_skips_comparison(client):
    open_trip()
    assert "attribute_comparison" not in client.get(
        "/gate/exit/lookup?plate=KJA456GH").get_json()


# ── correction after entry ────────────────────────────────────────────────

def test_a_trip_attribute_can_be_corrected_later(client):
    trip_id = open_trip()
    response = client.post(f"/gate/trips/{trip_id}/attributes",
                           json={"colour": "white"})
    assert response.status_code == 200
    trip = db.get_trip(trip_id)
    assert trip["vehicle_color"] == "white"
    assert trip["vehicle_color_source"] == SOURCE_MANUAL_CORRECTION
    assert trip["vehicle_color_confidence"] is None
    assert trip["vehicle_color_votes"] is None


def test_the_correction_route_cannot_touch_the_plate_or_status(client):
    trip_id = open_trip()
    client.post(f"/gate/trips/{trip_id}/attributes",
                json={"colour": "white", "plate_number": "HACKED", "status": "EXITED"})
    trip = db.get_trip(trip_id)
    assert trip["plate_number"] == "KJA456GH"
    assert trip["status"] == "INSIDE"


def test_the_correction_route_enforces_the_label_space(client):
    trip_id = open_trip()
    response = client.post(f"/gate/trips/{trip_id}/attributes",
                           json={"type": "spaceship"})
    assert response.status_code == 400
    assert "not a valid type" in response.get_json()["error"]


def test_the_correction_route_rejects_an_empty_body(client):
    assert client.post(f"/gate/trips/{open_trip()}/attributes",
                       json={}).status_code == 400


def test_the_correction_route_404s_for_a_missing_trip(client):
    assert client.post("/gate/trips/9999/attributes",
                       json={"colour": "white"}).status_code == 404


# ── SSE payload ───────────────────────────────────────────────────────────

def test_the_sse_payload_is_json_safe_and_marked_advisory():
    payload = result().to_api_dict()
    encoded = json.dumps(payload)
    assert "/home/" not in encoded and "gate_photos" not in encoded
    assert payload["advisory_only"] is True
    for head in ("colour", "type", "brand"):
        assert set(payload[head]) >= {"value", "confidence", "status", "votes",
                                      "source", "needs_operator_review", "top_k"}


def test_the_capture_stage_pushes_a_structured_payload(monkeypatch):
    import numpy as np

    messages = []
    monkeypatch.setattr(gm, "analyse_vehicle_attributes",
                        lambda *a, **k: result())
    cid = gm.new_capture_id()
    api = gm._run_vehicle_attributes(
        cid, lambda t, txt, **kw: messages.append({"type": t, "text": txt, **kw}),
        [np.zeros((40, 40, 3), np.uint8)])

    assert api["colour"]["value"] == "red"
    pushed = [m for m in messages if "vehicle_attributes" in m]
    assert pushed and pushed[0]["vehicle_attributes"]["brand"]["value"] == "toyota"
    assert gm.temp_get(cid)["vehicle_attributes"]["colour"]["value"] == "red"


def test_a_pipeline_crash_never_reaches_the_gate(monkeypatch):
    import numpy as np

    class Exploding:
        def process_frames(self, *a, **k):
            raise RuntimeError("backbone exploded")

    monkeypatch.setattr(gm, "get_attribute_pipeline", lambda: Exploding())
    out = gm.analyse_vehicle_attributes([np.zeros((40, 40, 3), np.uint8)])
    assert out.status == AttributeStatus.FAILED.value
    assert "pipeline error" in out.rejection_reason


# ── rendering ─────────────────────────────────────────────────────────────

def test_the_trip_log_renders_attributes_and_provenance(client):
    trip_id = open_trip()
    client.post(f"/gate/trips/{trip_id}/attributes", json={"colour": "white"})
    body = client.get("/gate/trips").get_data(as_text=True)
    assert "White" in body
    assert "corrected" in body        # provenance badge


def test_rendered_attribute_values_are_escaped(client):
    trip_id = open_trip()
    db.update_trip_attributes(trip_id, {"vehicle_brand": "<script>alert(1)</script>"})
    body = client.get("/gate/trips").get_data(as_text=True)
    assert "<script>alert(1)</script>" not in body
    # The display filter title-cases, so compare case-insensitively; what
    # matters is that the angle brackets were escaped, not their casing.
    assert "&lt;script&gt;" in body.lower()


def test_settings_shows_attribute_model_state(client):
    body = client.get("/settings").get_data(as_text=True)
    assert "Vehicle Attributes" in body
    assert "never authorise or deny" in body


def test_reports_does_not_present_coverage_as_accuracy(client):
    open_trip()
    body = client.get("/reports").get_data(as_text=True)
    assert "not accuracy" in body
