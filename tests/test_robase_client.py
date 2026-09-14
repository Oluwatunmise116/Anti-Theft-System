"""
The Robase REST client against the documented contract (docs.robase.dev),
configuration safety and phone normalisation. No request leaves the process:
the HTTP session is a fake.
"""
import dataclasses
import os
import stat
from datetime import datetime, timezone

import pytest
import requests

import app as application
import phone_utils
import robase_client as robase
import verification_settings as vs

KEY = "robe_" + "ab" * 32
OTP_ID = "550e8400-e29b-41d4-a716-446655440000"
SEND_OK = {"id": OTP_ID, "phone_number": "+2348031234567", "country_code": "NG",
           "credit_cost": 1, "status": "pending", "code_length": 6,
           "expires_at": "2026-09-14T10:05:00Z", "created_at": "2026-09-14T10:00:00Z"}


class FakeResponse:
    def __init__(self, status_code, body, headers=None):
        self.status_code = status_code
        self._body = body
        self.headers = headers or {}

    def json(self):
        if isinstance(self._body, Exception):
            raise self._body
        return self._body


class FakeSession:
    """Plays back responses (or raises errors) in order and records each call."""

    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def request(self, method, url, json=None, headers=None, timeout=None):
        self.calls.append({"method": method, "url": url, "json": json,
                           "headers": dict(headers or {}), "timeout": timeout})
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def client_with(*outcomes):
    session = FakeSession(*outcomes)
    return robase.RobaseClient(KEY, session=session, retry_delay=0), session


def error(status, error_type, headers=None):
    return FakeResponse(status, {"error": {"type": error_type, "message": "prose"}}, headers)


# ── send ─────────────────────────────────────────────────────────────────────

def test_send_posts_the_documented_request():
    client, session = client_with(FakeResponse(200, SEND_OK))
    sent = client.send_otp("+2348031234567", code_length=6, ttl_seconds=300,
                           idempotency_key="key-1", metadata={"trip_id": "7"})
    (call,) = session.calls
    assert (call["method"], call["url"]) == ("POST", "https://api.robase.dev/v1/otp/send")
    assert call["json"] == {"phone_number": "+2348031234567", "code_length": 6,
                            "ttl_seconds": 300, "metadata": {"trip_id": "7"}}
    assert call["headers"]["Authorization"] == f"Bearer {KEY}"
    assert call["headers"]["Idempotency-Key"] == "key-1"
    assert call["timeout"] == (5.0, 10.0)
    # The send response calls the identifier "id"; verify takes it as "otp_id".
    assert sent.otp_id == OTP_ID and sent.credit_cost == 1 and sent.status == "pending"
    assert sent.expires_at == datetime(2026, 9, 14, 10, 5, tzinfo=timezone.utc).timestamp()


def test_the_sms_language_is_sent_only_when_configured():
    client, session = client_with(FakeResponse(200, SEND_OK), FakeResponse(200, SEND_OK))
    client.send_otp("+2348031234567", code_length=6, ttl_seconds=300, idempotency_key="a")
    client.send_otp("+2348031234567", code_length=6, ttl_seconds=300, idempotency_key="b",
                    language="fr")
    assert "language" not in session.calls[0]["json"]
    assert session.calls[1]["json"]["language"] == "fr"


@pytest.mark.parametrize("body", [{**SEND_OK, "id": None}, {k: v for k, v in SEND_OK.items()
                                                            if k != "id"},
                                  {**SEND_OK, "id": "../../v1/sms"}, ["not", "an", "object"],
                                  ValueError("not json")])
def test_a_send_response_without_a_usable_id_is_malformed(body):
    client, _ = client_with(FakeResponse(200, body))
    with pytest.raises(robase.RobaseError) as info:
        client.send_otp("+2348031234567", code_length=6, ttl_seconds=300, idempotency_key="k")
    assert info.value.kind == "malformed" and info.value.uncertain


# ── verify ───────────────────────────────────────────────────────────────────

def test_verify_posts_the_otp_id_and_code():
    client, session = client_with(FakeResponse(200, {"valid": True, "status": "verified",
                                                     "attempts_used": 1}))
    result = client.verify_otp(OTP_ID, "123456", idempotency_key="v-1")
    (call,) = session.calls
    assert call["url"] == "https://api.robase.dev/v1/otp/verify"
    assert call["json"] == {"otp_id": OTP_ID, "code": "123456"}
    assert call["headers"]["Idempotency-Key"] == "v-1"
    assert result == robase.VerifyResult(True, "verified", 1, None)


def test_a_wrong_code_is_a_200_with_valid_false():
    client, _ = client_with(FakeResponse(200, {"valid": False, "status": "sent",
                                               "attempts_used": 2, "attempts_remaining": 3}))
    assert client.verify_otp(OTP_ID, "000000", idempotency_key="v") == \
        robase.VerifyResult(False, "sent", 2, 3)


@pytest.mark.parametrize("body", [{"status": "verified"}, {"valid": "true", "status": "verified"},
                                  {"valid": True, "status": "sent"}, {"valid": True},
                                  {"valid": 1, "status": "verified"}])
def test_an_ambiguous_verification_is_malformed_never_valid(body):
    client, _ = client_with(FakeResponse(200, body))
    with pytest.raises(robase.RobaseError) as info:
        client.verify_otp(OTP_ID, "123456", idempotency_key="v")
    assert info.value.kind == "malformed"


# ── errors ───────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("status,error_type,kind", [
    (400, "validation_error", "rejected"), (400, "invalid_phone", "rejected"),
    (400, "country_not_supported", "rejected"), (401, "unauthorized", "unauthorized"),
    (402, "insufficient_credits", "insufficient_credits"), (404, "otp_not_found", "not_found"),
    (409, "otp_expired", "expired"), (409, "otp_already_verified", "already_verified"),
    (409, "max_attempts_exceeded", "attempts_exhausted"),
])
def test_documented_errors_are_classified_by_type_and_not_retried(status, error_type, kind):
    client, session = client_with(error(status, error_type))
    with pytest.raises(robase.RobaseError) as info:
        client.send_otp("+2348031234567", code_length=6, ttl_seconds=300, idempotency_key="k")
    assert (info.value.kind, info.value.error_type) == (kind, error_type)
    assert not info.value.uncertain and len(session.calls) == 1


def test_rate_limits_carry_retry_after():
    client, _ = client_with(error(429, "rate_limited", {"Retry-After": "42"}))
    with pytest.raises(robase.RobaseError) as info:
        client.send_otp("+2348031234567", code_length=6, ttl_seconds=300, idempotency_key="k")
    assert info.value.kind == "rate_limited" and info.value.retry_after == 42


def test_a_timeout_is_retried_once_with_the_same_idempotency_key():
    client, session = client_with(requests.Timeout("read timed out"), FakeResponse(200, SEND_OK))
    sent = client.send_otp("+2348031234567", code_length=6, ttl_seconds=300,
                           idempotency_key="same-key")
    assert sent.otp_id == OTP_ID
    assert [c["headers"]["Idempotency-Key"] for c in session.calls] == ["same-key", "same-key"]


@pytest.mark.parametrize("failure,kind", [
    (requests.Timeout("read timed out"), "timeout"),
    (requests.ConnectionError("reset"), "network"),
    (error(500, "internal_error"), "server"),
])
def test_retries_are_bounded_and_the_outcome_is_uncertain(failure, kind):
    client, session = client_with(failure, failure)
    with pytest.raises(robase.RobaseError) as info:
        client.send_otp("+2348031234567", code_length=6, ttl_seconds=300, idempotency_key="k")
    assert info.value.kind == kind and info.value.uncertain
    assert len(session.calls) == 2


def test_the_api_key_never_leaks_into_errors_or_repr():
    leak = requests.ConnectionError(f"failed with header Authorization: Bearer {KEY}")
    client, _ = client_with(leak, leak)
    with pytest.raises(robase.RobaseError) as info:
        client.send_otp("+2348031234567", code_length=6, ttl_seconds=300, idempotency_key="k")
    assert KEY not in str(info.value) and info.value.__cause__ is None
    assert KEY not in repr(client)
    assert KEY not in robase.redact(f"oops {KEY}")


def test_get_otp_reads_status_and_refuses_odd_ids_without_a_request():
    client, session = client_with(FakeResponse(200, {"id": OTP_ID, "status": "sent",
                                                     "delivered_at": "2026-09-14T10:00:02Z"}))
    assert client.get_otp(OTP_ID) == {"status": "sent", "delivered": True,
                                      "failure_reason": None}
    assert session.calls[0]["url"] == f"https://api.robase.dev/v1/otp/{OTP_ID}"
    with pytest.raises(robase.RobaseError):
        client.get_otp("../sms/1")
    assert len(session.calls) == 1


# ── configuration ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("key", ["", "not-a-robase-key", "robe_replace-with-your-robase-api-key"])
def test_live_delivery_needs_a_real_robase_key(monkeypatch, key):
    monkeypatch.setenv("SMS_DELIVERY_MODE", "live")
    monkeypatch.setenv("ROBASE_API_KEY", key)
    with pytest.raises(vs.ConfigurationError):
        robase.get_client()


def test_mock_delivery_is_refused_in_production(monkeypatch):
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("SMS_DELIVERY_MODE", "mock")
    with pytest.raises(vs.ConfigurationError):
        robase.get_client()


@pytest.mark.parametrize("url,env,ok", [("http://evil.example", "production", False),
                                        ("http://localhost:8080", "production", False),
                                        ("http://localhost:8080", "development", True),
                                        ("https://api.robase.dev", "production", True)])
def test_the_base_url_must_be_https_outside_development(monkeypatch, url, env, ok):
    monkeypatch.setenv("APP_ENV", env)
    monkeypatch.setenv("SMS_DELIVERY_MODE", "live")
    monkeypatch.setenv("ROBASE_API_KEY", KEY)
    monkeypatch.setenv("ROBASE_BASE_URL", url)
    if ok:
        assert isinstance(robase.get_client(), robase.RobaseClient)
    else:
        with pytest.raises(vs.ConfigurationError):
            robase.get_client()


def test_the_mock_replays_an_idempotency_key_without_a_second_sms(tmp_path):
    path = tmp_path / "outbox.jsonl"
    mock = robase.MockRobaseClient(str(path))
    first = mock.send_otp("+2348031234567", code_length=6, ttl_seconds=300, idempotency_key="k")
    again = mock.send_otp("+2348031234567", code_length=6, ttl_seconds=300, idempotency_key="k")
    assert first == again and len(mock.sent) == 1
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
    assert "MOCK_DEVELOPMENT_SMS" in path.read_text()


def test_settings_are_clamped_to_safe_ranges(monkeypatch):
    monkeypatch.setenv("OTP_LENGTH", "4")
    monkeypatch.setenv("OTP_TTL_SECONDS", "999999")
    monkeypatch.setenv("OTP_MAX_ATTEMPTS", "9")
    monkeypatch.setenv("PASSCODE_MIN_LENGTH", "2")
    monkeypatch.setenv("ROBASE_SMS_LANGUAGE", "de")
    settings = vs.load()
    assert settings.otp_length == 6 and settings.otp_ttl_seconds == 900
    assert settings.otp_max_attempts == 5 and settings.passcode_min_length == 4
    assert settings.sms_language == ""


def test_env_example_placeholders_are_never_accepted_as_secrets(monkeypatch):
    monkeypatch.setenv("FLASK_SECRET_KEY", "replace-with-64-hex-characters-padding-padding")
    with pytest.raises(vs.ConfigurationError):
        vs.flask_secret_key()


def test_production_refuses_to_start_without_a_flask_secret(monkeypatch):
    import auth
    monkeypatch.setenv("FLASK_SECRET_KEY", "")
    with pytest.raises(vs.ConfigurationError):
        auth._secret_key(dataclasses.replace(vs.load(), app_env="production"))
    assert len(auth._secret_key(dataclasses.replace(vs.load(), app_env="development"))) >= 32


def test_startup_detects_a_port_that_is_already_in_use():
    import socket
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        busy_port = listener.getsockname()[1]
        assert application._port_available("127.0.0.1", busy_port) is False
    assert application._port_available("127.0.0.1", busy_port) is True


# ── phone numbers ────────────────────────────────────────────────────────────

def test_nigerian_numbers_are_normalised_to_e164():
    for raw in ("0803 123 4567", "08031234567", "+234 803 123 4567", "2348031234567",
                "803-123-4567"):
        assert phone_utils.normalize_phone(raw) == "+2348031234567", raw
    for bad in ("12345", None, "", "not a number"):
        assert phone_utils.normalize_phone(bad) is None
    masked = phone_utils.mask_phone("+2348031234567")
    assert masked.endswith("67") and "803" not in masked


def test_the_default_region_is_configurable():
    assert phone_utils.normalize_phone("07911 123456", "GB") == "+447911123456"
