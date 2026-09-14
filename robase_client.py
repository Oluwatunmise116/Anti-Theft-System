"""
Client for the Robase SMS OTP API (https://docs.robase.dev).

Robase generates the code, texts it and checks what the driver types. This
app stores only the OTP id Robase returns — never the code.

Contract, checked against the published API reference (docs.robase.dev,
September 2026):

  POST /v1/otp/send    {"phone_number": E.164, "code_length": 4-8,
                        "ttl_seconds": 1-3600, "language"?: "en"|"fr",
                        "metadata"?: {...}}
                       -> 200 {"id": uuid, "phone_number", "country_code",
                               "credit_cost", "status": "pending",
                               "code_length", "expires_at", "created_at"}
  POST /v1/otp/verify  {"otp_id": <id from send>, "code": "123456"}
                       -> 200 {"valid": bool, "status", "attempts_used",
                               "attempts_remaining" (only when valid is false)}
                       -> 409 otp_expired | otp_already_verified | max_attempts_exceeded
  GET  /v1/otp/{id}    -> 200 {"status": pending|sent|verified|expired|failed,
                               "delivered_at"?, "failure_reason"?, ...}
  Errors               {"error": {"type": ..., "message": ...}} — match on type.
                       402 insufficient_credits; 429 rate_limited + Retry-After.
  Idempotency-Key      accepted on POST; a replay within 24 hours returns the
                       original response, so a timed-out send is retried with
                       the same key without a second SMS or charge.

There is no test mode: every live send spends real credits. Tests and
development use MockRobaseClient, which is refused in production.

Safety rules:
  * Every request has a bounded (connect, read) timeout.
  * At most one automatic retry, only for outcomes that are safe to replay
    with the same Idempotency-Key (timeout, connection failure, 5xx).
  * A 2xx body that does not match the documented schema is an error.
  * The API key never appears in an exception, a log line or repr().
"""
import json
import os
import re
import secrets
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from urllib.parse import quote

import requests

import verification_settings as vs

_KEY_PATTERN = re.compile(r"robe_[A-Za-z0-9_\-]+")
_OTP_ID_PATTERN = re.compile(r"^[A-Za-z0-9\-]{1,64}$")


def redact(text) -> str:
    return _KEY_PATTERN.sub("robe_<redacted>", str(text or ""))


class RobaseError(Exception):
    """
    kind is one of:
      rejected              400 validation_error / invalid_phone / country_not_supported
      unauthorized          401 missing, malformed or revoked API key
      insufficient_credits  402 the Robase balance cannot pay for the SMS
      not_found             404 otp_not_found
      expired               409 otp_expired
      already_verified      409 otp_already_verified
      attempts_exhausted    409 max_attempts_exceeded
      rate_limited          429 — retry_after seconds is set when Robase sent it
      server                5xx
      timeout / network     no reliable answer
      malformed             a success response that does not match the schema
    """

    UNCERTAIN = frozenset({"server", "timeout", "network", "malformed"})

    def __init__(self, message, *, kind, error_type=None, status_code=None, retry_after=None):
        super().__init__(redact(message))
        self.kind = kind
        self.error_type = error_type
        self.status_code = status_code
        self.retry_after = retry_after

    @property
    def uncertain(self) -> bool:
        """The request may or may not have taken effect at Robase."""
        return self.kind in self.UNCERTAIN


_KIND_BY_TYPE = {
    "validation_error": "rejected", "invalid_phone": "rejected",
    "country_not_supported": "rejected", "unauthorized": "unauthorized",
    "insufficient_credits": "insufficient_credits", "otp_not_found": "not_found",
    "otp_expired": "expired", "otp_already_verified": "already_verified",
    "max_attempts_exceeded": "attempts_exhausted", "rate_limited": "rate_limited",
    "internal_error": "server",
}
_KIND_BY_STATUS = {400: "rejected", 401: "unauthorized", 402: "insufficient_credits",
                   404: "not_found", 429: "rate_limited"}
_RETRYABLE = frozenset({"server", "timeout", "network"})


@dataclass(frozen=True)
class SentOTP:
    otp_id: str
    expires_at: float = None      # Unix time from Robase's expires_at, if parseable
    status: str = ""
    credit_cost: int = None


@dataclass(frozen=True)
class VerifyResult:
    valid: bool
    status: str
    attempts_used: int = None
    attempts_remaining: int = None


def _parse_time(value):
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def _int_or_none(value):
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _retry_after(header):
    try:
        return max(1, int(float(header)))
    except (TypeError, ValueError):
        return None


def _check_base_url(base_url: str, settings) -> str:
    local = base_url.startswith(("http://localhost", "http://127.0.0.1"))
    if base_url.startswith("https://") or (local and settings.is_development):
        return base_url
    raise vs.ConfigurationError(
        "ROBASE_BASE_URL must be an https:// URL (http://localhost is allowed only in "
        "development).")


class RobaseClient:
    is_mock = False

    def __init__(self, api_key: str, base_url: str = vs.DEFAULT_ROBASE_BASE_URL,
                 connect_timeout: float = 5.0, read_timeout: float = 10.0,
                 session=None, retry_delay: float = 0.5):
        if not api_key or vs.is_placeholder(api_key) or not api_key.startswith("robe_"):
            raise vs.ConfigurationError(
                "ROBASE_API_KEY is missing or is not a Robase key (keys start with robe_). "
                "Create one at https://robase.dev/app/api-keys and put it in .env.")
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._timeout = (connect_timeout, read_timeout)
        self._session = session or requests.Session()
        self._retry_delay = retry_delay

    def __repr__(self):
        return f"<RobaseClient {self._base_url} key=<redacted>>"

    def _once(self, method, path, payload, idempotency_key):
        headers = {"Authorization": f"Bearer {self._api_key}", "Accept": "application/json",
                   "Accept-Language": "en"}
        if idempotency_key:
            headers["Idempotency-Key"] = idempotency_key
        try:
            response = self._session.request(method, self._base_url + path, json=payload,
                                             headers=headers, timeout=self._timeout)
        except requests.Timeout:
            raise RobaseError("Robase did not respond in time", kind="timeout") from None
        except requests.RequestException as exc:
            raise RobaseError(f"Could not reach Robase ({type(exc).__name__})",
                              kind="network") from None

        code = response.status_code
        try:
            body = response.json()
        except ValueError:
            body = None
        if 200 <= code < 300:
            if not isinstance(body, dict):
                raise RobaseError(f"Unexpected response from Robase (HTTP {code})",
                                  kind="malformed", status_code=code)
            return body

        error = body.get("error") if isinstance(body, dict) else None
        error_type = error.get("type") if isinstance(error, dict) else None
        message = error.get("message") if isinstance(error, dict) else None
        kind = (_KIND_BY_TYPE.get(error_type) or _KIND_BY_STATUS.get(code)
                or ("server" if code >= 500 else "rejected"))
        retry_after = _retry_after(response.headers.get("Retry-After")) \
            if kind == "rate_limited" else None
        raise RobaseError(str(message)[:200] if isinstance(message, str) and message
                          else f"Robase error (HTTP {code})",
                          kind=kind, error_type=error_type, status_code=code,
                          retry_after=retry_after)

    def _call(self, method, path, payload=None, idempotency_key=None):
        # A POST is retried only when it carries an Idempotency-Key, so the
        # retry cannot send a second SMS or be charged twice.
        attempts = 2 if (idempotency_key or method == "GET") else 1
        for attempt in range(attempts):
            try:
                return self._once(method, path, payload, idempotency_key)
            except RobaseError as exc:
                if attempt + 1 < attempts and exc.kind in _RETRYABLE:
                    time.sleep(self._retry_delay)
                    continue
                raise

    def send_otp(self, phone_number: str, *, code_length: int, ttl_seconds: int,
                 idempotency_key: str, language: str = None, metadata: dict = None) -> SentOTP:
        payload = {"phone_number": phone_number, "code_length": int(code_length),
                   "ttl_seconds": int(ttl_seconds)}
        if language:
            payload["language"] = language
        if metadata:
            payload["metadata"] = metadata
        body = self._call("POST", "/v1/otp/send", payload, idempotency_key)
        otp_id = body.get("id")
        if not isinstance(otp_id, str) or not _OTP_ID_PATTERN.match(otp_id.strip()):
            raise RobaseError("Robase accepted the send but returned no usable OTP id",
                              kind="malformed")
        return SentOTP(otp_id=otp_id.strip(), expires_at=_parse_time(body.get("expires_at")),
                       status=str(body.get("status") or ""),
                       credit_cost=_int_or_none(body.get("credit_cost")))

    def verify_otp(self, otp_id: str, code: str, *, idempotency_key: str) -> VerifyResult:
        body = self._call("POST", "/v1/otp/verify", {"otp_id": otp_id, "code": code},
                          idempotency_key)
        valid, status = body.get("valid"), body.get("status")
        # A success must say so twice: valid is true AND status is "verified".
        if not isinstance(valid, bool) or (valid and status != "verified"):
            raise RobaseError("Robase returned a verification result that does not match "
                              "the documented schema", kind="malformed")
        return VerifyResult(valid=valid, status=str(status or ""),
                            attempts_used=_int_or_none(body.get("attempts_used")),
                            attempts_remaining=_int_or_none(body.get("attempts_remaining")))

    def get_otp(self, otp_id: str) -> dict:
        if not _OTP_ID_PATTERN.match(otp_id or ""):
            raise RobaseError("Invalid OTP id", kind="rejected")
        body = self._call("GET", f"/v1/otp/{quote(otp_id, safe='')}")
        return {"status": str(body.get("status") or ""),
                "delivered": bool(body.get("delivered_at")),
                "failure_reason": body.get("failure_reason")
                if isinstance(body.get("failure_reason"), str) else None}


class MockRobaseClient:
    """
    DEVELOPMENT/TEST ONLY — a stand-in for Robase. No SMS is sent.

    It imitates the provider (it makes up a code, as Robase would) so the
    gate can be exercised without credits. Production code never generates
    codes: get_client() refuses this class outside APP_ENV=development/test.
    Codes land in `self.sent` and, when an outbox path is given, in that
    file (mode 0600). `fail_next_send` / `fail_next_verify` make the next
    call raise, and `delivery_status` is what get_otp() reports.
    """
    is_mock = True
    MAX_ATTEMPTS = 5            # Robase's documented example budget

    def __init__(self, outbox_path: str = None):
        self.sent = []
        self.verify_calls = []
        self.fail_next_send = None
        self.fail_next_verify = None
        self.delivery_status = "sent"
        self._otps = {}
        self._replies = {}
        self._outbox_path = outbox_path
        self._lock = threading.Lock()

    def __repr__(self):
        return "<MockRobaseClient DEVELOPMENT>"

    def send_otp(self, phone_number, *, code_length, ttl_seconds, idempotency_key,
                 language=None, metadata=None) -> SentOTP:
        with self._lock:
            if idempotency_key in self._replies:          # replay: no second SMS
                return self._replies[idempotency_key]
            if self.fail_next_send is not None:
                error, self.fail_next_send = self.fail_next_send, None
                raise error
            if not str(phone_number).startswith("+"):
                raise RobaseError("phone must be in E.164 format", kind="rejected",
                                  error_type="invalid_phone", status_code=400)
            code = f"{secrets.randbelow(10 ** code_length):0{code_length}d}"
            otp_id = str(uuid.uuid4())
            expires_at = time.time() + ttl_seconds
            self._otps[otp_id] = {"code": code, "expires_at": expires_at, "attempts": 0,
                                  "status": "pending"}
            record = {"otp_id": otp_id, "phone_number": phone_number, "code": code,
                      "code_length": code_length, "ttl_seconds": ttl_seconds,
                      "language": language, "metadata": metadata,
                      "idempotency_key": idempotency_key}
            self.sent.append(record)
            if self._outbox_path:
                os.makedirs(os.path.dirname(os.path.abspath(self._outbox_path)), exist_ok=True)
                fd = os.open(self._outbox_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
                with os.fdopen(fd, "a") as fh:
                    fh.write(json.dumps({"MOCK_DEVELOPMENT_SMS": True, **record}) + "\n")
            result = SentOTP(otp_id=otp_id, expires_at=expires_at, status="pending",
                             credit_cost=1)
            self._replies[idempotency_key] = result
            return result

    def verify_otp(self, otp_id, code, *, idempotency_key) -> VerifyResult:
        with self._lock:
            self.verify_calls.append({"otp_id": otp_id, "idempotency_key": idempotency_key})
            if idempotency_key in self._replies:
                return self._replies[idempotency_key]
            if self.fail_next_verify is not None:
                error, self.fail_next_verify = self.fail_next_verify, None
                raise error
            otp = self._otps.get(otp_id)
            if otp is None:
                raise RobaseError("OTP not found", kind="not_found",
                                  error_type="otp_not_found", status_code=404)
            if otp["status"] == "verified":
                raise RobaseError("OTP already verified", kind="already_verified",
                                  error_type="otp_already_verified", status_code=409)
            if time.time() >= otp["expires_at"]:
                raise RobaseError("OTP has expired", kind="expired",
                                  error_type="otp_expired", status_code=409)
            if otp["attempts"] >= self.MAX_ATTEMPTS:
                raise RobaseError("maximum verification attempts exceeded",
                                  kind="attempts_exhausted",
                                  error_type="max_attempts_exceeded", status_code=409)
            otp["attempts"] += 1
            if secrets.compare_digest(otp["code"], str(code)):
                otp["status"] = "verified"
                result = VerifyResult(True, "verified", otp["attempts"])
            else:
                left = self.MAX_ATTEMPTS - otp["attempts"]
                otp["status"] = "failed" if left == 0 else "sent"
                result = VerifyResult(False, otp["status"], otp["attempts"], left)
            self._replies[idempotency_key] = result
            return result

    def get_otp(self, otp_id) -> dict:
        with self._lock:
            otp = self._otps.get(otp_id)
            if otp is None:
                raise RobaseError("OTP not found", kind="not_found",
                                  error_type="otp_not_found", status_code=404)
            status = otp["status"] if otp["status"] == "verified" else self.delivery_status
            return {"status": status, "delivered": status in ("sent", "verified"),
                    "failure_reason": "delivery_failed" if status == "failed" else None}


_override = None
_cache = {}
_cache_lock = threading.Lock()


def set_client_override(client):
    """Inject a client (tests). Pass None to clear."""
    global _override
    _override = client


def get_client(settings: vs.Settings = None):
    """The SMS client for the current configuration.

    Raises ConfigurationError when live delivery has no usable API key, or
    when mock delivery is requested outside development.
    """
    if _override is not None:
        return _override
    settings = settings or vs.load()
    if settings.mock_delivery:
        if not settings.is_development:
            raise vs.ConfigurationError(
                "SMS_DELIVERY_MODE=mock is only allowed with APP_ENV=development. "
                "Refusing to pretend codes were sent in production.")
        outbox = os.environ.get("SMS_MOCK_OUTBOX", "").strip() or \
            os.path.join("instance", "sms_mock_outbox.jsonl")
        with _cache_lock:
            key = ("mock", outbox)
            if key not in _cache:
                _cache[key] = MockRobaseClient(outbox_path=outbox)
            return _cache[key]

    base_url = _check_base_url(settings.robase_base_url, settings)
    api_key = vs.robase_api_key()
    with _cache_lock:
        key = ("live", api_key, base_url, settings.robase_connect_timeout,
               settings.robase_read_timeout)
        if key not in _cache:
            _cache[key] = RobaseClient(api_key, base_url, settings.robase_connect_timeout,
                                       settings.robase_read_timeout)
        return _cache[key]
