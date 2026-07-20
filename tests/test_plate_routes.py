"""
Route-level tests for the ANPR-related Flask endpoints.

Only hardware interfaces (camera, sensor) are mocked — the database uses a
real temporary SQLite file (schema created via database.init_db()) so
create_trip()/get_trip() behaviour, including the new plate_source /
plate_confidence provenance columns, is exercised for real.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import database as db
import config as cfg
import gate_manager as gm
import app as application


@pytest.fixture()
def client(tmp_path, monkeypatch):
    db_path = str(tmp_path / "test_license.db")
    monkeypatch.setattr(db, "DB_PATH", db_path)
    db.init_db()
    application.app.config["TESTING"] = True
    with application.app.test_client() as c:
        yield c


def _set_plate_session(cid, plate_number, status, overall_confidence=0.9):
    gm.temp_set(cid, "plate_number", plate_number)
    gm.temp_set(cid, "plate_result", {
        "plate_number": plate_number,
        "status": status,
        "overall_confidence": overall_confidence,
    })
    gm.temp_set(cid, "plate_source", "auto" if plate_number else "")
    gm.temp_set(cid, "plate_confidence", overall_confidence)


def test_gate_entry_page_loads(client):
    r = client.get("/gate/entry")
    assert r.status_code == 200


def test_model_status_reports_plate_and_face_sections(client):
    r = client.get("/gate/model/status")
    assert r.status_code == 200
    data = r.get_json()
    assert "plate" in data and "face" in data
    assert "model_path" in data["plate"]


def test_plate_model_cannot_be_downloaded_at_runtime(client):
    r = client.post("/gate/model/download", json={"model": "plate"})
    assert r.status_code == 400
    data = r.get_json()
    assert data["ok"] is False
    assert "train" in data["error"].lower()
    # No internal path/traceback leakage
    assert "Traceback" not in data["error"]


def test_confirm_requires_plate_number(client):
    r = client.post("/gate/entry/confirm", json={"capture_id": "nosuch", "plate": ""})
    assert r.status_code == 400


def test_confirm_manual_entry_when_no_auto_suggestion_exists(client):
    cid = "cid_manual"
    r = client.post("/gate/entry/confirm", json={"capture_id": cid, "plate": "kja456gh"})
    assert r.status_code == 200
    trip_id = r.get_json()["trip_id"]
    trip = db.get_trip(trip_id)
    assert trip["plate_number"] == "KJA456GH"
    assert trip["plate_source"] == "manual_entry"


def test_confirm_accepts_unchanged_confirmed_auto_reading(client):
    cid = "cid_auto_confirmed"
    _set_plate_session(cid, "KJA456GH", "CONFIRMED", overall_confidence=0.91)
    r = client.post("/gate/entry/confirm", json={"capture_id": cid, "plate": "KJA456GH"})
    assert r.status_code == 200
    trip = db.get_trip(r.get_json()["trip_id"])
    assert trip["plate_source"] == "auto"
    assert trip["plate_confidence"] == pytest.approx(0.91)


def test_confirm_rejects_unconfirmed_low_confidence_without_explicit_confirmation(client):
    cid = "cid_low_conf"
    _set_plate_session(cid, "KJA456GH", "LOW_CONFIDENCE", overall_confidence=0.4)
    r = client.post("/gate/entry/confirm", json={"capture_id": cid, "plate": "KJA456GH"})
    assert r.status_code == 409
    data = r.get_json()
    assert data["requires_confirmation"] is True


def test_confirm_accepts_low_confidence_once_explicitly_confirmed(client):
    cid = "cid_low_conf_ok"
    _set_plate_session(cid, "KJA456GH", "LOW_CONFIDENCE", overall_confidence=0.4)
    r = client.post("/gate/entry/confirm", json={
        "capture_id": cid, "plate": "KJA456GH", "confirm_low_confidence": True,
    })
    assert r.status_code == 200
    trip = db.get_trip(r.get_json()["trip_id"])
    assert trip["plate_source"] == "auto"


def test_confirm_edited_plate_is_manual_correction(client):
    cid = "cid_corrected"
    _set_plate_session(cid, "KJA456GH", "LOW_CONFIDENCE", overall_confidence=0.4)
    # Operator disagrees with the auto suggestion and types a different plate —
    # no low-confidence confirmation gate applies to a genuine correction.
    r = client.post("/gate/entry/confirm", json={"capture_id": cid, "plate": "ABC123DE"})
    assert r.status_code == 200
    trip = db.get_trip(r.get_json()["trip_id"])
    assert trip["plate_number"] == "ABC123DE"
    assert trip["plate_source"] == "manual_correction"


def test_exit_lookup_not_found_for_unknown_plate(client):
    r = client.get("/gate/exit/lookup?plate=ZZ000ZZ")
    assert r.status_code == 200
    assert r.get_json()["found"] is False


def test_vehicle_snap_requires_capture_id(client):
    r = client.post("/gate/entry/vehicle/snap", json={})
    assert r.status_code == 400


def test_vehicle_auto_start_returns_started_without_crashing_when_no_camera(client):
    r = client.post("/gate/entry/vehicle/auto-start", json={"capture_id": "cid_nocam"})
    assert r.status_code == 200
    assert r.get_json().get("started") is True
    gm.end_vehicle_session("cid_nocam")


def test_debug_photo_route_disabled_by_default(client, monkeypatch):
    monkeypatch.setattr(cfg, "get", lambda k, d=None: False if k == "plate_debug_mode" else d)
    r = client.get("/gate/debug/somefile.jpg")
    assert r.status_code == 404


def test_debug_photo_route_404s_cleanly_for_missing_file_when_enabled(client, monkeypatch):
    monkeypatch.setattr(cfg, "get", lambda k, d=None: True if k == "plate_debug_mode" else d)
    r = client.get("/gate/debug/does_not_exist.jpg")
    assert r.status_code == 404
    assert b"Traceback" not in r.data


def test_debug_photo_route_blocks_path_traversal(client, monkeypatch):
    monkeypatch.setattr(cfg, "get", lambda k, d=None: True if k == "plate_debug_mode" else d)
    r = client.get("/gate/debug/..%2F..%2Fapp.py")
    assert r.status_code in (400, 404)
