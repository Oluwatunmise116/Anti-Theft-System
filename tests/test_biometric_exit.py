"""
Face and fingerprint — the primary exit methods.

The capture routes and gate_manager's real comparison code run; only the
camera, the face models and the fingerprint sensor are faked.
"""
import glob
import os
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

import app as application
import database as db
import face_manager as fm
import gate_manager as gm
import gate_verification as gv
from gate_helpers import (FACE, FINGERPRINT, add_holder, guest_trip, registered_trip, rows,
                          staff_client, staff_context, trip_row, unresolved_trip)


@pytest.fixture()
def gate(gate_db, sms_mock):
    application.app.config["TESTING"] = True
    return staff_client(application.app, "op", "operator")


@pytest.fixture()
def fake_camera(monkeypatch):
    """Camera and face models faked; state["distance"] is the next comparison."""
    state = {"distance": 0.05}
    monkeypatch.setattr(fm, "ensure_camera_running", lambda: True)
    monkeypatch.setattr(fm, "request_capture", lambda: None)
    monkeypatch.setattr(fm, "get_captured_frame",
                        lambda timeout=12: np.zeros((48, 64, 3), dtype=np.uint8))
    monkeypatch.setattr(fm, "get_face_embedding", lambda frame: (np.ones(128), [0, 0, 10, 10]))
    monkeypatch.setattr(fm, "face_distance", lambda a, b: state["distance"])
    monkeypatch.setattr(fm, "stop_camera", lambda: None)
    monkeypatch.setattr(gm.time, "sleep", lambda seconds: None)
    before = set(glob.glob(os.path.join(gm.GATE_PHOTOS_DIR, "exit_*.jpg")))
    yield state
    for path in set(glob.glob(os.path.join(gm.GATE_PHOTOS_DIR, "exit_*.jpg"))) - before:
        os.remove(path)


class FakeSensor:
    def __init__(self, score):
        self.score = score
        self.confidence = None

    def get_image(self):
        return gm.adafruit_fingerprint.OK

    def image_2_tz(self, slot):
        return gm.adafruit_fingerprint.OK

    def send_fpdata(self, data, sensorbuffer="char", slot=2):
        return True

    def compare_templates(self):
        self.confidence = self.score
        return gm.adafruit_fingerprint.OK


@pytest.fixture()
def fake_sensor(monkeypatch):
    """Fingerprint sensor faked; state["score"] is the next comparison score."""
    state = {"score": 120}
    constants = getattr(gm, "adafruit_fingerprint", None) or SimpleNamespace(OK=0, NOFINGER=2)
    monkeypatch.setattr(gm, "adafruit_fingerprint", constants, raising=False)
    monkeypatch.setattr(gm.fp_mgr, "_get_sensor", lambda: (FakeSensor(state["score"]), None))
    monkeypatch.setattr(gm.time, "sleep", lambda seconds: None)
    return state


def _wait_until(condition, timeout=10.0):
    waiter = threading.Event()
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "the capture thread did not finish"
        waiter.wait(0.02)


def _drain(q):
    messages = []
    while q is not None and not q.empty():
        messages.append(q.get_nowait())
    return messages


def face_scan(client, trip_id):
    response = client.post("/gate/exit/verify/start", json={"trip_id": trip_id})
    assert response.status_code == 200, response.get_json()
    q = gm.get_exit_queue(trip_id)
    _wait_until(lambda: gm.get_exit_queue(trip_id) is None)
    return _drain(q)[-1]


def fingerprint_scan(client, trip_id):
    response = client.post("/gate/exit/fp/start", json={"trip_id": trip_id})
    assert response.get_json().get("started"), response.get_json()
    q = gm.get_fp_exit_queue(trip_id)
    _wait_until(lambda: gm.get_fp_exit_queue(trip_id) is None)
    return _drain(q)[-1]


def confirm(client, trip_id):
    return client.post("/gate/exit/confirm", json={"trip_id": trip_id})


# ── primary exit without any code ────────────────────────────────────────────

def test_a_face_match_authorizes_the_exit_without_an_otp(gate, fake_camera, sms_mock):
    trip_id = registered_trip(add_holder(), face=FACE)
    final = face_scan(gate, trip_id)
    assert final["type"] == "success" and final["authorized"] is True
    closed = confirm(gate, trip_id)
    assert closed.status_code == 200 and closed.get_json()["method"] == "face"
    trip = trip_row(trip_id)
    assert trip["status"] == "EXITED" and trip["exit_verification_method"] == "face"
    assert sms_mock.sent == []                     # no SMS was needed or sent


def test_a_face_mismatch_authorizes_nothing(gate, fake_camera):
    trip_id = registered_trip(add_holder(), face=FACE)
    fake_camera["distance"] = 0.9
    final = face_scan(gate, trip_id)
    assert final["type"] == "denied" and final["authorized"] is False
    assert confirm(gate, trip_id).status_code == 403
    trip = trip_row(trip_id)
    assert trip["status"] == "INSIDE" and trip["exit_biometric_match"] == 0
    assert rows("SELECT * FROM trip_exit_authorizations") == []


def test_a_fingerprint_match_authorizes_the_exit(gate, fake_sensor):
    trip_id = registered_trip(add_holder(), fingerprint=FINGERPRINT)
    final = fingerprint_scan(gate, trip_id)
    assert final["type"] == "success" and final["authorized"] is True
    closed = confirm(gate, trip_id)
    assert closed.status_code == 200 and closed.get_json()["method"] == "fingerprint"


def test_a_fingerprint_mismatch_authorizes_nothing(gate, fake_sensor):
    trip_id = registered_trip(add_holder(), fingerprint=FINGERPRINT)
    fake_sensor["score"] = 5
    assert fingerprint_scan(gate, trip_id)["type"] == "notfound"
    assert confirm(gate, trip_id).status_code == 403


def test_a_guest_can_exit_by_face_without_the_passcode(gate, fake_camera):
    trip_id = guest_trip("4821", face=FACE)
    assert face_scan(gate, trip_id)["authorized"] is True
    assert confirm(gate, trip_id).get_json()["method"] == "face"


@pytest.mark.parametrize("identity_status", ["UNRESOLVED", "LEGACY_UNRESOLVED"])
def test_biometrics_work_for_unresolved_and_legacy_trips(gate, fake_camera, identity_status):
    trip_id = unresolved_trip(face=FACE, identity_status=identity_status)
    assert face_scan(gate, trip_id)["authorized"] is True
    assert confirm(gate, trip_id).status_code == 200


def test_biometrics_work_when_the_fallback_is_locked(gate, fake_camera):
    trip_id = registered_trip(add_holder(), face=FACE)
    with db.transaction() as conn:
        conn.execute("UPDATE trips SET verification_locked=1 WHERE id=?", (trip_id,))
    assert face_scan(gate, trip_id)["authorized"] is True
    assert confirm(gate, trip_id).status_code == 200


# ── binding and limits ───────────────────────────────────────────────────────

def test_a_biometric_authorization_belongs_to_the_scanning_session(gate, fake_camera):
    trip_id = registered_trip(add_holder(), face=FACE)
    other = staff_client(application.app, "op2", "operator")
    face_scan(gate, trip_id)
    assert confirm(other, trip_id).status_code == 403
    assert confirm(gate, trip_id).status_code == 200


def test_a_biometric_authorization_expires(gate, fake_camera, monkeypatch):
    trip_id = registered_trip(add_holder(), face=FACE)
    face_scan(gate, trip_id)
    later = time.time() + 121
    monkeypatch.setattr(gv, "_clock", lambda: later)
    assert confirm(gate, trip_id).status_code == 403


def test_a_match_needs_a_template_stored_on_this_trip(gate):
    trip_id = registered_trip(add_holder())               # no face, no fingerprint
    staff = staff_context(gate)
    assert gate.post("/gate/exit/verify/start", json={"trip_id": trip_id}).status_code == 400
    assert gv.record_biometric_result(trip_id, staff, "face", True, 0.01) is False
    assert gv.record_biometric_result(trip_id, staff, "fingerprint", True, 200) is False
    assert confirm(gate, trip_id).status_code == 403


def test_a_closed_trip_cannot_be_authorized_by_a_late_match(gate, fake_camera):
    trip_id = guest_trip("4821", face=FACE)
    staff = staff_context(gate)
    assert face_scan(gate, trip_id)["authorized"] is True
    assert confirm(gate, trip_id).status_code == 200
    assert gv.record_biometric_result(trip_id, staff, "face", True, 0.01) is False
    assert confirm(gate, trip_id).get_json()["code"] == "trip_closed"


def test_only_biometric_methods_are_recorded_as_biometrics(gate):
    with pytest.raises(ValueError):
        gv.record_biometric_result(1, staff_context(gate), "sms_otp", True)
