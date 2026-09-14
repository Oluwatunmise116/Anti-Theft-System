"""Guest passcodes at entry and exit, unresolved trips and administrator reconciliation."""
import pytest
from werkzeug.security import check_password_hash

import app as application
import gate_manager as gm
from gate_helpers import (add_holder, guest_trip, rows, staff_client, staff_context, trip_row,
                          unresolved_trip)

NO_MATCH = {"result": "no_match", "compared": 2}


@pytest.fixture()
def gate(gate_db, sms_mock):
    application.app.config["TESTING"] = True
    return staff_client(application.app, "op", "operator")


@pytest.fixture()
def admin(gate):
    return staff_client(application.app, "boss", "admin")


def capture(client, face=NO_MATCH, fp=NO_MATCH):
    cid = gm.new_capture_id(owner=staff_context(client).session_id)
    gm.temp_set(cid, "identity_face", face)
    gm.temp_set(cid, "identity_fp", fp)
    return cid


def entry(client, cid, **extra):
    return client.post("/gate/entry/confirm", json={"capture_id": cid, "plate": "GST123AA",
                                                    **extra})


def passcode(client, trip_id, code):
    return client.post("/gate/exit/passcode/verify", json={"trip_id": trip_id, "passcode": code})


def confirm(client, trip_id):
    return client.post("/gate/exit/confirm", json={"trip_id": trip_id})


def reconcile(client, trip_id, **form):
    return client.post(f"/admin/trips/{trip_id}/reconcile", data=form)


def trip_count():
    return rows("SELECT COUNT(*) AS n FROM trips")[0]["n"]


# ── entry ────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("bad", [None, "", "   ", "123", "123456789", "12a4", "١٢٣٤",
                                 1234, ["1234"]])
def test_a_confirmed_guest_must_set_a_valid_passcode(gate, bad):
    response = entry(gate, capture(gate), passcode=bad)
    assert response.status_code == 400
    assert trip_count() == 0                     # nothing half-saved


def test_a_guest_passcode_is_stored_only_as_a_hash(gate):
    response = entry(gate, capture(gate), passcode="4821")
    assert response.status_code == 200 and response.get_json()["verification_mode"] == "PASSCODE"
    trip = trip_row(response.get_json()["trip_id"])
    assert trip["identity_status"] == "GUEST" and trip["holder_id"] is None
    assert trip["passcode"] is None and "4821" not in trip["passcode_hash"]
    assert check_password_hash(trip["passcode_hash"], "4821")


@pytest.mark.parametrize("face,fp", [(None, None), (NO_MATCH, None), (None, NO_MATCH),
                                     ({"result": "error"}, NO_MATCH)])
def test_incomplete_identity_checks_are_unresolved_not_guest(gate, face, fp):
    cid = capture(gate, face=face, fp=fp)
    refused = entry(gate, cid, passcode="4821")
    assert refused.status_code == 409 and refused.get_json()["code"] == "identity_unresolved"
    assert trip_count() == 0
    accepted = entry(gate, cid, passcode="4821", accept_unresolved=True)
    trip = trip_row(accepted.get_json()["trip_id"])
    assert trip["identity_status"] == "UNRESOLVED" and trip["verification_mode"] is None
    assert check_password_hash(trip["passcode_hash"], "4821")


def test_the_acknowledgement_must_be_a_real_boolean(gate):
    assert entry(gate, capture(gate, face=None, fp=None),
                 accept_unresolved="yes").status_code == 409


def test_entry_page_passcode_limits_match_the_server(gate):
    page = gate.get("/gate/entry").get_data(as_text=True)
    assert 'minlength="4"' in page and 'maxlength="8"' in page
    assert "const PASSCODE_MIN = 4;" in page and "const PASSCODE_MAX = 8;" in page


# ── exit ─────────────────────────────────────────────────────────────────────

def test_a_guest_exits_with_the_passcode_fallback(gate, sms_mock):
    trip_id = guest_trip("4821")
    view = gate.get("/gate/exit/lookup?plate=GST123AA").get_json()["verification"]
    assert view["mode"] == "PASSCODE" and view["passcode"]["attempts_remaining"] == 5
    assert "sms" not in view

    wrong = passcode(gate, trip_id, "1111")
    assert wrong.status_code == 400 and wrong.get_json()["attempts_remaining"] == 4
    assert confirm(gate, trip_id).status_code == 403
    assert passcode(gate, trip_id, "4821").get_json() == {
        "valid": True, "method": "passcode", "authorization_expires_in": 120}
    assert confirm(gate, trip_id).status_code == 200
    trip = trip_row(trip_id)
    assert trip["status"] == "EXITED" and trip["exit_verification_method"] == "passcode"
    assert trip["passcode_hash"] is None
    assert sms_mock.sent == []


def test_a_passcode_only_opens_its_own_trip(gate):
    first = guest_trip("4821", plate="GST111AA")
    second = guest_trip("9999", plate="GST222BB")
    assert passcode(gate, second, "4821").status_code == 400
    assert passcode(gate, first, "4821").status_code == 200
    assert confirm(gate, second).status_code == 403


def test_passcode_attempts_are_limited(gate):
    trip_id = guest_trip("4821")
    for _ in range(4):
        assert passcode(gate, trip_id, "1111").status_code == 400
    assert passcode(gate, trip_id, "1111").status_code == 423
    assert passcode(gate, trip_id, "4821").get_json()["code"] == "verification_locked"
    assert confirm(gate, trip_id).status_code == 403


@pytest.mark.parametrize("bad", [None, "", "12", "abcd", 4821])
def test_malformed_passcodes_do_not_count_as_attempts(gate, bad):
    trip_id = guest_trip("4821")
    assert passcode(gate, trip_id, bad).status_code == 400
    assert trip_row(trip_id)["passcode_failed_attempts"] == 0


def test_a_guest_trip_cannot_use_the_sms_routes(gate, sms_mock):
    trip_id = guest_trip()
    for path, body in (("/gate/exit/sms/send", {"trip_id": trip_id}),
                       ("/gate/exit/sms/verify", {"trip_id": trip_id, "code": "123456"})):
        response = gate.post(path, json=body)
        assert response.status_code == 409
        assert response.get_json()["code"] == "wrong_verification_mode"
    assert sms_mock.sent == []


# ── unresolved trips and reconciliation ──────────────────────────────────────

def test_an_unresolved_trip_has_no_fallback_until_an_admin_confirms_the_guest(gate, admin):
    trip_id = unresolved_trip(passcode="4821")
    assert passcode(gate, trip_id, "4821").get_json()["code"] == "reconciliation_required"
    assert gate.post("/gate/exit/sms/send", json={"trip_id": trip_id}) \
        .get_json()["code"] == "reconciliation_required"
    assert confirm(gate, trip_id).status_code == 403
    assert gate.get("/gate/exit/lookup?plate=UNR111AA").get_json()["verification"][
        "reconciliation_required"]

    assert reconcile(gate, trip_id, action="confirm_guest", note="x").status_code == 403
    reconcile(admin, trip_id, action="confirm_guest", note="")          # a note is required
    assert trip_row(trip_id)["verification_mode"] is None
    reconcile(admin, trip_id, action="confirm_guest", note="guest known to security")
    trip = trip_row(trip_id)
    assert trip["verification_mode"] == "PASSCODE"
    assert trip["identity_method"] == "staff_reconciliation"
    assert passcode(gate, trip_id, "4821").get_json()["valid"] is True
    assert confirm(gate, trip_id).status_code == 200


def test_confirming_a_guest_needs_a_stored_passcode(gate, admin):
    trip_id = unresolved_trip()
    reconcile(admin, trip_id, action="confirm_guest", note="no code was set")
    assert trip_row(trip_id)["verification_mode"] is None


def test_an_unresolved_trip_can_only_be_assigned_to_a_driver_sms_can_reach(gate, admin):
    trip_id = unresolved_trip()
    no_phone = add_holder("EZE", "Obi", phone="12345")
    reconcile(admin, trip_id, action="assign_holder", holder_id=no_phone, note="ID checked")
    assert trip_row(trip_id)["verification_mode"] is None               # invalid number
    holder_id = add_holder()
    reconcile(admin, trip_id, action="assign_holder", holder_id=holder_id, note="ID checked")
    trip = trip_row(trip_id)
    assert (trip["verification_mode"], trip["holder_id"], trip["identity_status"]) \
        == ("SMS_OTP", holder_id, "REGISTERED")


def test_a_resolved_trip_cannot_be_switched_to_another_mode(gate, admin):
    trip_id = guest_trip()
    holder_id = add_holder()
    reconcile(admin, trip_id, action="assign_holder", holder_id=holder_id, note="try switch")
    assert trip_row(trip_id)["verification_mode"] == "PASSCODE"
