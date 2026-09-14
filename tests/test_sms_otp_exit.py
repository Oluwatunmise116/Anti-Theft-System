"""
Robase SMS OTP — the fallback for registered drivers when face and
fingerprint fail.

Robase is replaced by robase_client.MockRobaseClient, which behaves like the
documented API (it makes up the code, as Robase would); the tests read the
code from the mocked SMS exactly as a driver would read it from the phone.
"""
import importlib
import json
import logging
import os
import threading

import pytest
from werkzeug.security import generate_password_hash

import app as application
import database as db
import gate_manager as gm
import gate_verification as gv
import robase_client as robase
from gate_helpers import (add_holder, change_phone, edit_form, holder_row, last_sms_code, login,
                          registered_trip, rows, staff_client, staff_context, trip_row)

PHONE_E164 = "+2348031234567"


@pytest.fixture()
def env(gate_db, sms_mock):
    application.app.config["TESTING"] = True
    holder_id = add_holder(phone="0803 123 4567")
    return {"holder_id": holder_id, "trip_id": registered_trip(holder_id),
            "client": staff_client(application.app, "op", "operator"), "sms": sms_mock}


@pytest.fixture()
def clock(monkeypatch):
    now = [1_900_000_000.0]
    monkeypatch.setattr(gv, "_clock", lambda: now[0])
    return now


def send(client, trip_id, path="/gate/exit/sms/send"):
    return client.post(path, json={"trip_id": trip_id})


def verify(client, trip_id, code):
    return client.post("/gate/exit/sms/verify", json={"trip_id": trip_id, "code": code})


def confirm(client, trip_id):
    return client.post("/gate/exit/confirm", json={"trip_id": trip_id})


def wrong(code):
    return "000000" if code != "000000" else "111111"


def challenge_statuses():
    return [r["status"] for r in rows("SELECT status FROM sms_otp_challenges ORDER BY id")]


# ── lookup ───────────────────────────────────────────────────────────────────

def test_lookup_offers_the_sms_fallback_without_secrets(env):
    payload = env["client"].get("/gate/exit/lookup?plate=KJA456GH").get_json()
    view = payload["verification"]
    assert view["mode"] == "SMS_OTP" and view["driver_name"] == "OKAFOR, Ada"
    assert view["sms"]["available"] and view["sms"]["destination_hint"].endswith("67")
    assert "passcode" not in view
    text = json.dumps(payload)
    for secret in (PHONE_E164, "0803 123 4567", "holder_uid", "provider_otp_id", "passcode_hash"):
        assert secret not in text


# ── the fallback working ─────────────────────────────────────────────────────

def test_a_registered_driver_exits_with_the_sms_code(env):
    client, sms, trip_id = env["client"], env["sms"], env["trip_id"]
    response = send(client, trip_id)
    assert response.status_code == 200
    body = response.get_json()
    assert body["sent"] is True and body["expires_in"] == 300
    assert body["resend_available_in"] == 60 and body["code_length"] == 6

    (message,) = sms.sent
    # The number comes from the linked holder record, normalised to E.164;
    # Robase is asked for a 6-digit code valid for 5 minutes.
    assert message["phone_number"] == PHONE_E164
    assert (message["code_length"], message["ttl_seconds"]) == (6, 300)
    assert message["metadata"] == {"purpose": "gate_exit", "trip_id": str(trip_id)}
    code = last_sms_code(sms)
    assert code not in json.dumps(body)

    (challenge,) = rows("SELECT * FROM sms_otp_challenges")
    assert challenge["status"] == "ACTIVE" and challenge["provider_otp_id"] == message["otp_id"]
    assert (challenge["trip_id"], challenge["holder_id"]) == (trip_id, env["holder_id"])
    assert challenge["gate_session_id"] == staff_context(client).session_id
    assert challenge["idempotency_key"] == message["idempotency_key"]
    assert all(code not in value for value in challenge.values() if isinstance(value, str))

    assert verify(client, trip_id, code).get_json() == {
        "valid": True, "method": "sms_otp", "authorization_expires_in": 120}
    closed = confirm(client, trip_id)
    assert closed.status_code == 200 and closed.get_json()["method"] == "sms_otp"
    trip = trip_row(trip_id)
    assert trip["status"] == "EXITED" and trip["exit_verification_method"] == "sms_otp"
    assert rows("SELECT method, status, challenge_id FROM trip_exit_authorizations") == [
        {"method": "sms_otp", "status": "CONSUMED", "challenge_id": challenge["id"]}]


def test_every_send_has_its_own_idempotency_key(env, clock):
    client, sms, trip_id = env["client"], env["sms"], env["trip_id"]
    send(client, trip_id)
    clock[0] += 61
    send(client, trip_id, "/gate/exit/sms/resend")
    keys = [m["idempotency_key"] for m in sms.sent]
    assert len(keys) == 2 and len(set(keys)) == 2 and all(keys)


def test_codes_and_authorizations_are_single_use(env):
    client, trip_id = env["client"], env["trip_id"]
    send(client, trip_id)
    code = last_sms_code(env["sms"])
    assert verify(client, trip_id, code).status_code == 200
    assert verify(client, trip_id, code).get_json()["code"] == "sms_no_active_challenge"
    assert confirm(client, trip_id).status_code == 200
    assert confirm(client, trip_id).get_json()["code"] == "trip_closed"
    assert verify(client, trip_id, code).get_json()["code"] == "trip_closed"
    assert send(client, trip_id).get_json()["code"] == "trip_closed"


# ── wrong, expired, superseded ───────────────────────────────────────────────

def test_three_wrong_codes_exhaust_the_challenge(env):
    client, trip_id = env["client"], env["trip_id"]
    send(client, trip_id)
    code = last_sms_code(env["sms"])
    first = verify(client, trip_id, wrong(code))
    assert first.status_code == 400 and first.get_json()["attempts_remaining"] == 2
    assert verify(client, trip_id, wrong(code)).get_json()["attempts_remaining"] == 1
    assert verify(client, trip_id, wrong(code)).get_json()["code"] == "otp_attempts_exhausted"
    assert challenge_statuses() == ["EXHAUSTED"]
    assert verify(client, trip_id, code).get_json()["code"] == "sms_no_active_challenge"
    assert confirm(client, trip_id).status_code == 403


def test_an_expired_code_is_refused(env, clock):
    client, trip_id = env["client"], env["trip_id"]
    send(client, trip_id)
    code = last_sms_code(env["sms"])
    clock[0] += 301
    response = verify(client, trip_id, code)
    assert response.status_code == 410 and response.get_json()["code"] == "otp_expired"
    assert challenge_statuses() == ["EXPIRED"]
    assert env["sms"].verify_calls == []           # refused locally, Robase not asked


def test_a_replaced_code_is_refused_even_if_robase_would_accept_it(env, clock):
    client, sms, trip_id = env["client"], env["sms"], env["trip_id"]
    send(client, trip_id)
    first_code, first_otp = last_sms_code(sms), sms.sent[-1]["otp_id"]
    clock[0] += 61
    send(client, trip_id, "/gate/exit/sms/resend")
    second_code, second_otp = last_sms_code(sms), sms.sent[-1]["otp_id"]
    assert challenge_statuses() == ["SUPERSEDED", "ACTIVE"]
    if first_code != second_code:
        assert verify(client, trip_id, first_code).status_code == 400
        # Only the latest challenge's OTP id is ever sent to Robase.
        assert sms.verify_calls[-1]["otp_id"] == second_otp != first_otp
    assert verify(client, trip_id, second_code).status_code == 200


def test_a_delivered_resend_withdraws_an_unused_sms_authorization(env, clock):
    client, sms, trip_id = env["client"], env["sms"], env["trip_id"]
    send(client, trip_id)
    assert verify(client, trip_id, last_sms_code(sms)).status_code == 200
    clock[0] += 61
    assert send(client, trip_id, "/gate/exit/sms/resend").status_code == 200
    assert confirm(client, trip_id).status_code == 403
    assert verify(client, trip_id, last_sms_code(sms)).status_code == 200
    assert confirm(client, trip_id).status_code == 200


def test_a_failed_resend_keeps_the_earlier_code(env, clock):
    client, sms, trip_id = env["client"], env["sms"], env["trip_id"]
    send(client, trip_id)
    first = last_sms_code(sms)
    clock[0] += 61
    sms.fail_next_send = robase.RobaseError("no route", kind="network")
    assert send(client, trip_id, "/gate/exit/sms/resend").status_code == 504
    assert verify(client, trip_id, first).status_code == 200


# ── resend limits ────────────────────────────────────────────────────────────

def test_resend_has_a_60_second_cooldown(env, clock):
    client, sms, trip_id = env["client"], env["sms"], env["trip_id"]
    assert send(client, trip_id).status_code == 200
    clock[0] += 30
    early = send(client, trip_id, "/gate/exit/sms/resend")
    assert early.status_code == 429 and early.get_json()["retry_after"] == 30
    assert len(sms.sent) == 1
    clock[0] += 31
    assert send(client, trip_id, "/gate/exit/sms/resend").status_code == 200
    assert len(sms.sent) == 2


def test_the_per_trip_send_limit_is_not_reset_by_resending(env, clock, monkeypatch):
    monkeypatch.setenv("OTP_MAX_SENDS_PER_TRIP", "2")
    client, trip_id = env["client"], env["trip_id"]
    for _ in range(2):
        assert send(client, trip_id, "/gate/exit/sms/resend").status_code == 200
        clock[0] += 61
    limited = send(client, trip_id, "/gate/exit/sms/resend")
    assert limited.status_code == 429 and limited.get_json()["code"] == "sms_trip_send_limit"


def test_failed_attempts_across_codes_lock_the_fallback(env, clock):
    client, sms, trip_id = env["client"], env["sms"], env["trip_id"]
    for _ in range(2):                     # 3 wrong per code x 2 codes = 6 = trip limit
        send(client, trip_id)
        code = last_sms_code(sms)
        for _ in range(3):
            last = verify(client, trip_id, wrong(code))
        clock[0] += 61
    assert last.status_code == 423
    assert send(client, trip_id).status_code == 423
    assert client.get("/gate/exit/lookup?plate=KJA456GH").get_json()["verification"]["locked"]

    admin = staff_client(application.app, "boss", "admin")
    assert admin.post(f"/admin/trips/{trip_id}/reconcile",
                      data={"action": "unlock", "note": "driver present, ID checked"}
                      ).status_code == 302
    assert trip_row(trip_id)["verification_locked"] == 0
    assert send(client, trip_id).status_code == 200


def test_the_per_driver_limit_spans_trips(env, clock, monkeypatch):
    monkeypatch.setenv("OTP_MAX_SENDS_PER_HOLDER_WINDOW", "2")
    client, trip_id = env["client"], env["trip_id"]
    second_trip = registered_trip(env["holder_id"], plate="KJA999ZZ")
    assert send(client, trip_id).status_code == 200
    assert send(client, second_trip).status_code == 200
    clock[0] += 61
    blocked = send(client, trip_id)
    assert blocked.status_code == 429 and blocked.get_json()["code"] == "sms_driver_limit"


# ── provider failures leave the trip open and in the same category ───────────

@pytest.mark.parametrize("kind,error_type,status,code,challenge_status", [
    ("insufficient_credits", "insufficient_credits", 503, "sms_insufficient_credits", "FAILED"),
    ("rejected", "invalid_phone", 502, "sms_send_failed", "FAILED"),
    ("rejected", "country_not_supported", 502, "sms_send_failed", "FAILED"),
    ("unauthorized", "unauthorized", 503, "sms_send_failed", "FAILED"),
    ("timeout", None, 504, "sms_send_uncertain", "UNCERTAIN"),
    ("network", None, 504, "sms_send_uncertain", "UNCERTAIN"),
    ("server", "internal_error", 504, "sms_send_uncertain", "UNCERTAIN"),
    ("malformed", None, 504, "sms_send_uncertain", "UNCERTAIN"),
])
def test_a_failed_send_grants_nothing_and_never_falls_back_to_a_passcode(
        env, kind, error_type, status, code, challenge_status):
    client, sms, trip_id = env["client"], env["sms"], env["trip_id"]
    sms.fail_next_send = robase.RobaseError("provider said no", kind=kind, error_type=error_type)
    response = send(client, trip_id)
    assert response.status_code == status and response.get_json()["code"] == code
    assert challenge_statuses() == [challenge_status]
    assert verify(client, trip_id, "123456").get_json()["code"] == "sms_no_active_challenge"
    assert client.post("/gate/exit/passcode/verify",
                       json={"trip_id": trip_id, "passcode": "1234"}).get_json()["code"] \
        == "wrong_verification_mode"
    assert confirm(client, trip_id).status_code == 403
    trip = trip_row(trip_id)
    assert trip["status"] == "INSIDE" and trip["verification_mode"] == "SMS_OTP"


def test_a_robase_rate_limit_is_respected(env, clock):
    client, sms, trip_id = env["client"], env["sms"], env["trip_id"]
    sms.fail_next_send = robase.RobaseError("too many send attempts", kind="rate_limited",
                                            error_type="rate_limited", retry_after=90)
    first = send(client, trip_id)
    assert first.status_code == 429 and first.get_json()["retry_after"] == 90
    clock[0] += 61
    held = send(client, trip_id)
    assert held.status_code == 429 and held.get_json()["retry_after"] == 29
    clock[0] += 30
    assert send(client, trip_id).status_code == 200


def test_an_uncertain_verification_authorizes_nothing_and_can_be_retried(env):
    client, sms, trip_id = env["client"], env["sms"], env["trip_id"]
    send(client, trip_id)
    code = last_sms_code(sms)
    sms.fail_next_verify = robase.RobaseError("read timed out", kind="timeout")
    response = verify(client, trip_id, code)
    assert response.status_code == 504 and response.get_json()["code"] == "sms_verify_uncertain"
    assert confirm(client, trip_id).status_code == 403
    assert rows("SELECT status, attempts FROM sms_otp_challenges") == [
        {"status": "ACTIVE", "attempts": 1}]      # counted: Robase may have checked it
    assert verify(client, trip_id, code).status_code == 200


@pytest.mark.parametrize("kind", ["rate_limited", "insufficient_credits", "unauthorized"])
def test_a_verification_robase_refused_to_run_does_not_use_an_attempt(env, kind):
    client, sms, trip_id = env["client"], env["sms"], env["trip_id"]
    send(client, trip_id)
    sms.fail_next_verify = robase.RobaseError("refused", kind=kind, retry_after=30)
    assert verify(client, trip_id, last_sms_code(sms)).status_code in (429, 503)
    assert rows("SELECT status, attempts FROM sms_otp_challenges") == [
        {"status": "ACTIVE", "attempts": 0}]


@pytest.mark.parametrize("kind,status,code", [("expired", 410, "otp_expired"),
                                              ("attempts_exhausted", 400,
                                               "otp_attempts_exhausted"),
                                              ("already_verified", 409, "sms_code_unusable"),
                                              ("not_found", 409, "sms_code_unusable")])
def test_robase_terminal_answers_authorize_nothing(env, kind, status, code):
    client, sms, trip_id = env["client"], env["sms"], env["trip_id"]
    send(client, trip_id)
    sms.fail_next_verify = robase.RobaseError("terminal", kind=kind)
    response = verify(client, trip_id, last_sms_code(sms))
    assert response.status_code == status and response.get_json()["code"] == code
    assert confirm(client, trip_id).status_code == 403


def test_sms_not_configured_fails_closed(gate_db, monkeypatch):
    application.app.config["TESTING"] = True
    trip_id = registered_trip(add_holder())
    client = staff_client(application.app)
    monkeypatch.setenv("SMS_DELIVERY_MODE", "live")
    monkeypatch.setenv("ROBASE_API_KEY", "")
    response = send(client, trip_id)
    assert response.status_code == 503 and response.get_json()["code"] == "sms_not_configured"
    assert rows("SELECT * FROM sms_otp_challenges") == []


# ── the recipient is the trip's own driver ───────────────────────────────────

@pytest.mark.parametrize("phone", ["12345", "", None])
def test_a_missing_or_invalid_number_means_no_sms_and_no_passcode(gate_db, sms_mock, phone):
    application.app.config["TESTING"] = True
    trip_id = registered_trip(add_holder(phone=phone))
    client = staff_client(application.app)
    view = client.get("/gate/exit/lookup?plate=KJA456GH").get_json()["verification"]
    assert view["mode"] == "SMS_OTP" and view["sms"]["available"] is False
    response = send(client, trip_id)
    assert response.status_code == 409 and response.get_json()["code"] == "phone_missing"
    assert client.post("/gate/exit/passcode/verify",
                       json={"trip_id": trip_id, "passcode": "1234"}).get_json()["code"] \
        == "wrong_verification_mode"
    assert sms_mock.sent == [] and trip_row(trip_id)["verification_mode"] == "SMS_OTP"


def test_the_destination_cannot_be_chosen_by_the_client(env):
    client, sms, trip_id = env["client"], env["sms"], env["trip_id"]
    other = add_holder("EZE", "Obi", phone="08037654321")
    client.post("/gate/exit/sms/send", json={"trip_id": trip_id, "phone_number": "+2348099999999",
                                             "holder_id": other})
    assert [m["phone_number"] for m in sms.sent] == [PHONE_E164]


def test_operators_cannot_change_where_codes_go(env):
    client, holder_id = env["client"], env["holder_id"]
    response = client.post(f"/holders/{holder_id}/edit", data=edit_form(holder_id, "08037654321"))
    assert response.status_code == 302
    assert holder_row(holder_id)["phone_e164"] == PHONE_E164


def test_an_admin_phone_change_cancels_codes_sent_to_the_old_number(env):
    client, sms, trip_id, holder_id = env["client"], env["sms"], env["trip_id"], env["holder_id"]
    send(client, trip_id)
    old_code = last_sms_code(sms)
    admin = staff_client(application.app, "boss", "admin")
    assert admin.post(f"/holders/{holder_id}/edit",
                      data=edit_form(holder_id, "08037654321")).status_code == 302
    assert holder_row(holder_id)["phone_e164"] == "+2348037654321"
    assert challenge_statuses() == ["CANCELLED"]
    assert verify(client, trip_id, old_code).get_json()["code"] == "sms_no_active_challenge"


def test_a_number_changed_behind_an_active_code_is_detected(env):
    client, sms, trip_id, holder_id = env["client"], env["sms"], env["trip_id"], env["holder_id"]
    send(client, trip_id)
    code = last_sms_code(sms)
    with db.transaction() as conn:      # changed without going through update_holder
        conn.execute("UPDATE holders SET phone_e164='+2348037654321' WHERE id=?", (holder_id,))
    assert verify(client, trip_id, code).get_json()["code"] == "sms_recipient_changed"


def test_a_deleted_driver_gets_no_codes(env):
    client, trip_id, holder_id = env["client"], env["trip_id"], env["holder_id"]
    db.delete_holder(holder_id)
    add_holder("NEW", "Person", phone="0803 123 4567")        # reuses the id and number
    response = send(client, trip_id)
    assert response.status_code == 409 and response.get_json()["code"] == "holder_changed"


def test_a_registered_driver_cannot_use_the_guest_passcode_route(env):
    client, trip_id = env["client"], env["trip_id"]
    with db.transaction() as conn:           # even if a passcode were somehow stored
        conn.execute("UPDATE trips SET passcode_hash=? WHERE id=?",
                     (generate_password_hash("4821"), trip_id))
    response = client.post("/gate/exit/passcode/verify",
                           json={"trip_id": trip_id, "passcode": "4821"})
    assert response.status_code == 409 and response.get_json()["code"] == "wrong_verification_mode"
    assert confirm(client, trip_id).status_code == 403


# ── session and trip binding ─────────────────────────────────────────────────

def test_a_code_only_works_in_the_gate_session_that_requested_it(env):
    client, trip_id = env["client"], env["trip_id"]
    other = staff_client(application.app, "op2", "operator")
    same_user_other_browser = login(application.app.test_client(), "op")
    send(client, trip_id)
    code = last_sms_code(env["sms"])
    for intruder in (other, same_user_other_browser):
        response = verify(intruder, trip_id, code)
        assert response.status_code == 403 and response.get_json()["code"] == "sms_other_session"
    assert rows("SELECT attempts FROM sms_otp_challenges") == [{"attempts": 0}]
    assert env["sms"].verify_calls == []

    assert verify(client, trip_id, code).status_code == 200
    assert confirm(other, trip_id).status_code == 403      # the authorization is bound too
    assert confirm(client, trip_id).status_code == 200


def test_a_code_for_one_trip_cannot_verify_another(env):
    client, sms, trip_id = env["client"], env["sms"], env["trip_id"]
    other_holder = add_holder("EZE", "Obi", phone="08037654321")
    other_trip = registered_trip(other_holder, plate="LAG777BB")

    send(client, trip_id)
    code_one = last_sms_code(sms)
    assert verify(client, other_trip, code_one).get_json()["code"] == "sms_no_active_challenge"
    send(client, other_trip)
    assert sms.sent[-1]["phone_number"] == "+2348037654321"
    code_two = last_sms_code(sms)
    if code_one != code_two:
        assert verify(client, other_trip, code_one).status_code == 400
    assert verify(client, trip_id, code_one).status_code == 200
    assert confirm(client, other_trip).status_code == 403


def test_the_client_cannot_force_an_exit(env):
    client, trip_id = env["client"], env["trip_id"]
    for body in ({"trip_id": trip_id},
                 {"trip_id": trip_id, "valid": True, "method": "sms_otp"},
                 {"trip_id": trip_id, "decision": "GRANTED", "authorized": True},
                 {"trip_id": trip_id, "exit_photo_path": application.__file__}):
        assert client.post("/gate/exit/confirm", json=body).status_code == 403
    assert os.path.exists(application.__file__)          # no client-chosen file deletion
    assert trip_row(trip_id)["status"] == "INSIDE"


def test_the_retired_telegram_routes_are_gone(env):
    for path in ("/gate/exit/otp/request", "/gate/exit/otp/resend", "/gate/exit/otp/verify"):
        assert env["client"].post(path, json={"trip_id": env["trip_id"]}).status_code == 404


# ── concurrency and restarts ─────────────────────────────────────────────────

def _race(target, count=6):
    barrier, results = threading.Barrier(count), []

    def run():
        barrier.wait()
        try:
            results.append(target())
        except gv.VerificationError as exc:
            results.append(exc.code)

    threads = [threading.Thread(target=run) for _ in range(count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return results


def test_concurrent_verification_authorizes_exactly_once(env):
    client, trip_id = env["client"], env["trip_id"]
    send(client, trip_id)
    code, staff = last_sms_code(env["sms"]), staff_context(client)
    results = _race(lambda: gv.verify_sms_otp(trip_id, code, staff)["valid"])
    assert results.count(True) == 1
    assert rows("SELECT COUNT(*) AS n FROM trip_exit_authorizations") == [{"n": 1}]
    assert len(env["sms"].verify_calls) <= 3                # never beyond the attempt budget


def test_concurrent_confirmation_closes_the_trip_once(env):
    client, trip_id = env["client"], env["trip_id"]
    send(client, trip_id)
    verify(client, trip_id, last_sms_code(env["sms"]))
    staff = staff_context(client)
    results = _race(lambda: gv.confirm_exit(trip_id, staff)["ok"])
    assert results.count(True) == 1
    assert set(results) - {True} <= {"trip_closed", "exit_not_authorized", "exit_conflict"}


def test_codes_and_authorizations_survive_an_application_restart(env):
    client, trip_id = env["client"], env["trip_id"]
    send(client, trip_id)
    code = last_sms_code(env["sms"])
    cookie = client.get_cookie("session").value
    csrf = client.environ_base["HTTP_X_CSRF_TOKEN"]

    def restarted_client():
        # Process memory is gone; only the database and the browser cookie remain.
        importlib.reload(gv)
        gm._temp.clear()
        fresh = application.app.test_client()
        fresh.set_cookie("session", cookie)
        fresh.environ_base["HTTP_X_CSRF_TOKEN"] = csrf
        return fresh

    assert verify(restarted_client(), trip_id, code).status_code == 200
    assert confirm(restarted_client(), trip_id).status_code == 200


# ── delivery status ──────────────────────────────────────────────────────────

def test_delivery_status_is_reported_and_a_failed_delivery_retires_the_code(env):
    client, sms, trip_id = env["client"], env["sms"], env["trip_id"]
    send(client, trip_id)
    status = lambda c: c.get(f"/gate/exit/sms/status?trip_id={trip_id}").get_json()
    assert status(client) == {"delivery_status": "delivered"}
    assert status(staff_client(application.app, "op2")) == {"delivery_status": None}
    sms.delivery_status = "failed"
    assert status(client) == {"delivery_status": "failed"}
    assert challenge_statuses() == ["FAILED"]
    assert verify(client, trip_id, last_sms_code(sms)).get_json()["code"] \
        == "sms_no_active_challenge"


# ── entry ────────────────────────────────────────────────────────────────────

NO_MATCH = {"result": "no_match", "compared": 1}


def match(holder_id):
    return {"result": "match", "holder_id": holder_id,
            "holder_uid": holder_row(holder_id)["holder_uid"]}


def capture(client, face=None, fp=None):
    """A capture whose server-side face/fingerprint search results are set,
    as gate_manager records them. The browser has no way to set these."""
    cid = gm.new_capture_id(owner=staff_context(client).session_id)
    gm.temp_set(cid, "identity_face", face)
    gm.temp_set(cid, "identity_fp", fp)
    return cid


def trip_count():
    return rows("SELECT COUNT(*) AS n FROM trips")[0]["n"]


def test_a_registered_driver_is_logged_with_the_sms_fallback(env):
    client, holder_id = env["client"], env["holder_id"]
    cid = capture(client, face=match(holder_id))
    identity = client.get(f"/gate/entry/identity?capture_id={cid}").get_json()
    assert identity["identity_status"] == "REGISTERED" and identity["verification_mode"] == "SMS_OTP"
    assert identity["sms_fallback_available"] and identity["phone_hint"].endswith("67")
    assert PHONE_E164 not in json.dumps(identity)

    response = client.post("/gate/entry/confirm", json={"capture_id": cid, "plate": "ABC123DE"})
    assert response.status_code == 200
    assert response.get_json()["verification_mode"] == "SMS_OTP"
    assert response.get_json()["sms_fallback_available"] is True
    trip = trip_row(response.get_json()["trip_id"])
    assert trip["holder_id"] == holder_id
    assert trip["holder_uid"] == holder_row(holder_id)["holder_uid"]
    assert trip["identity_status"] == "REGISTERED" and trip["identity_method"] == "face"
    assert trip["entry_session_id"] == staff_context(client).session_id


def test_a_registered_driver_without_a_phone_stays_registered(env):
    client = env["client"]
    no_phone = add_holder("EZE", "Obi", phone="")
    response = client.post("/gate/entry/confirm", json={
        "capture_id": capture(client, fp=match(no_phone)), "plate": "ABC123DE"})
    assert response.status_code == 200
    assert response.get_json()["sms_fallback_available"] is False
    trip = trip_row(response.get_json()["trip_id"])
    assert trip["verification_mode"] == "SMS_OTP" and trip["holder_id"] == no_phone
    assert trip["passcode_hash"] is None


def test_a_registered_driver_cannot_be_given_a_guest_passcode(env):
    client = env["client"]
    before = trip_count()
    response = client.post("/gate/entry/confirm", json={
        "capture_id": capture(client, face=match(env["holder_id"]), fp=NO_MATCH),
        "plate": "ABC123DE", "passcode": "4821"})
    assert response.status_code == 409
    assert response.get_json()["code"] == "registered_driver_passcode_rejected"
    assert trip_count() == before


def test_a_browser_supplied_identity_or_mode_is_ignored(env):
    client = env["client"]
    response = client.post("/gate/entry/confirm", json={
        "capture_id": capture(client, face=NO_MATCH, fp=NO_MATCH), "plate": "ABC123DE",
        "passcode": "4821", "holder_id": env["holder_id"], "identity_status": "REGISTERED",
        "verification_mode": "SMS_OTP"})
    trip = trip_row(response.get_json()["trip_id"])
    assert trip["verification_mode"] == "PASSCODE" and trip["holder_id"] is None


def test_conflicting_biometric_matches_are_unresolved(env):
    client = env["client"]
    other = add_holder("EZE", "Obi", phone="08037654321")
    response = client.post("/gate/entry/confirm", json={
        "capture_id": capture(client, face=match(env["holder_id"]), fp=match(other)),
        "plate": "ABC123DE"})
    assert response.status_code == 409 and response.get_json()["code"] == "identity_unresolved"


def test_another_gate_sessions_capture_cannot_be_used(env):
    client = env["client"]
    other = staff_client(application.app, "op2", "operator")
    cid = capture(other, face=match(env["holder_id"]))
    assert client.post("/gate/entry/confirm",
                       json={"capture_id": cid, "plate": "ABC123DE"}).status_code == 403
    assert client.get(f"/gate/entry/identity?capture_id={cid}").status_code == 403


# ── malformed input and leakage ──────────────────────────────────────────────

@pytest.mark.parametrize("body", [None, {}, {"trip_id": None}, {"trip_id": "abc"},
                                  {"trip_id": True}, {"trip_id": -3}, {"trip_id": [1]}])
def test_missing_or_malformed_trip_ids_are_rejected(env, body):
    client = env["client"]
    for path in ("/gate/exit/sms/send", "/gate/exit/sms/resend", "/gate/exit/sms/verify",
                 "/gate/exit/passcode/verify", "/gate/exit/confirm"):
        if body is None:
            response = client.post(path, data="not json", content_type="text/plain")
        else:
            response = client.post(path, json=body)
        assert response.status_code == 400, path
    assert client.get("/gate/exit/sms/status?trip_id=abc").status_code == 400
    assert env["sms"].sent == []


@pytest.mark.parametrize("code", [None, "", "12345", "1234567", "12a456", 123456,
                                  ["123456"], "١٢٣٤٥٦"])
def test_malformed_codes_are_rejected_without_using_an_attempt(env, code):
    client, trip_id = env["client"], env["trip_id"]
    send(client, trip_id)
    response = verify(client, trip_id, code)
    assert response.status_code == 400 and response.get_json()["valid"] is False
    assert rows("SELECT attempts FROM sms_otp_challenges") == [{"attempts": 0}]
    assert env["sms"].verify_calls == []


def test_codes_and_numbers_never_appear_in_responses_logs_or_the_audit_trail(
        env, caplog, capsys):
    client, trip_id, sms = env["client"], env["trip_id"], env["sms"]
    caplog.set_level(logging.DEBUG)
    bodies = [send(client, trip_id).get_data(as_text=True)]
    code = last_sms_code(sms)
    bodies.append(verify(client, trip_id, wrong(code)).get_data(as_text=True))
    bodies.append(client.get("/gate/exit/lookup?plate=KJA456GH").get_data(as_text=True))
    bodies.append(verify(client, trip_id, code).get_data(as_text=True))
    bodies.append(confirm(client, trip_id).get_data(as_text=True))
    captured = capsys.readouterr()
    haystack = "\n".join(bodies) + caplog.text + captured.out + captured.err
    assert code not in haystack and PHONE_E164 not in haystack
    audit = json.dumps([(e["event"], e["detail"]) for e in rows("SELECT * FROM security_events")])
    assert code not in audit and PHONE_E164 not in audit
