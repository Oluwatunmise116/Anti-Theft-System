"""
Configuration for staff authentication, the Robase SMS OTP fallback and
guest passcodes.

Secrets come only from the environment (optionally a local, git-ignored
.env loaded by app.py) — never from config.json, which is committed.
Tunables have safe defaults and are clamped on read, so a typo in the
environment can shorten or lengthen a limit but can never switch it off.

Settings are re-read on every call. Reading a handful of environment
variables is cheap, and it lets tests change a value with monkeypatch.
"""
import os
from dataclasses import dataclass


class ConfigurationError(RuntimeError):
    """A required secret or setting is missing or unsafe."""


APP_ENVS = ("production", "development", "test")
DELIVERY_MODES = ("live", "mock")
SMS_LANGUAGES = ("en", "fr")
DEFAULT_ROBASE_BASE_URL = "https://api.robase.dev"

#: Shortest acceptable value for FLASK_SECRET_KEY.
MIN_SECRET_LENGTH = 32


def _int(name: str, default: int, low: int, high: int) -> int:
    raw = os.environ.get(name, "").strip()
    try:
        value = int(raw) if raw else default
    except ValueError:
        value = default
    return max(low, min(high, value))


def _float(name: str, default: float, low: float, high: float) -> float:
    raw = os.environ.get(name, "").strip()
    try:
        value = float(raw) if raw else default
    except ValueError:
        value = default
    return max(low, min(high, value))


@dataclass(frozen=True)
class Settings:
    app_env: str
    sms_delivery_mode: str
    robase_base_url: str
    robase_connect_timeout: float
    robase_read_timeout: float
    sms_language: str
    phone_default_region: str
    otp_length: int
    otp_ttl_seconds: int
    otp_max_attempts: int
    otp_resend_cooldown_seconds: int
    otp_max_sends_per_trip: int
    otp_max_failed_per_trip: int
    otp_holder_window_seconds: int
    otp_max_sends_per_holder_window: int
    otp_max_failed_per_holder_window: int
    exit_authorization_ttl_seconds: int
    passcode_min_length: int
    passcode_max_length: int
    passcode_max_attempts: int
    staff_session_hours: int

    @property
    def is_development(self) -> bool:
        return self.app_env in ("development", "test")

    @property
    def mock_delivery(self) -> bool:
        return self.sms_delivery_mode == "mock"


def load() -> Settings:
    app_env = (os.environ.get("APP_ENV", "") or "production").strip().lower()
    if app_env not in APP_ENVS:
        raise ConfigurationError(
            f"APP_ENV must be one of {', '.join(APP_ENVS)} (got {app_env!r})")

    mode = (os.environ.get("SMS_DELIVERY_MODE", "") or "live").strip().lower()
    if mode not in DELIVERY_MODES:
        raise ConfigurationError(
            f"SMS_DELIVERY_MODE must be 'live' or 'mock' (got {mode!r})")

    region = (os.environ.get("PHONE_DEFAULT_REGION", "") or "NG").strip().upper()
    if len(region) != 2 or not region.isalpha():
        region = "NG"

    language = os.environ.get("ROBASE_SMS_LANGUAGE", "").strip().lower()
    if language not in SMS_LANGUAGES:
        language = ""            # the Robase workspace default

    # Existing trips accept 4-8 digit passcodes (the previous backend rule);
    # the range can be narrowed but never below 4 digits.
    passcode_min = _int("PASSCODE_MIN_LENGTH", 4, 4, 12)
    passcode_max = _int("PASSCODE_MAX_LENGTH", 8, passcode_min, 12)

    return Settings(
        app_env=app_env,
        sms_delivery_mode=mode,
        robase_base_url=(os.environ.get("ROBASE_BASE_URL", "").strip().rstrip("/")
                         or DEFAULT_ROBASE_BASE_URL),
        robase_connect_timeout=_float("ROBASE_CONNECT_TIMEOUT", 5.0, 1.0, 30.0),
        robase_read_timeout=_float("ROBASE_READ_TIMEOUT", 10.0, 2.0, 60.0),
        sms_language=language,
        phone_default_region=region,
        # Robase accepts 4-8 digit codes and TTLs up to an hour; this gate
        # asks for 6 digits and 5 minutes, and never less than 6 digits.
        otp_length=_int("OTP_LENGTH", 6, 6, 8),
        otp_ttl_seconds=_int("OTP_TTL_SECONDS", 300, 60, 900),
        # Robase's own budget is larger (5); the local limit is the one enforced.
        otp_max_attempts=_int("OTP_MAX_ATTEMPTS", 3, 1, 5),
        otp_resend_cooldown_seconds=_int("OTP_RESEND_COOLDOWN_SECONDS", 60, 30, 600),
        otp_max_sends_per_trip=_int("OTP_MAX_SENDS_PER_TRIP", 5, 1, 20),
        otp_max_failed_per_trip=_int("OTP_MAX_FAILED_PER_TRIP", 6, 1, 30),
        otp_holder_window_seconds=_int("OTP_HOLDER_WINDOW_SECONDS", 3600, 300, 86400),
        otp_max_sends_per_holder_window=_int("OTP_MAX_SENDS_PER_HOLDER_WINDOW", 10, 1, 100),
        otp_max_failed_per_holder_window=_int("OTP_MAX_FAILED_PER_HOLDER_WINDOW", 10, 1, 100),
        exit_authorization_ttl_seconds=_int("EXIT_AUTHORIZATION_TTL_SECONDS", 120, 30, 600),
        passcode_min_length=passcode_min,
        passcode_max_length=passcode_max,
        passcode_max_attempts=_int("PASSCODE_MAX_ATTEMPTS", 5, 1, 20),
        staff_session_hours=_int("STAFF_SESSION_HOURS", 12, 1, 72),
    )


def is_placeholder(value) -> bool:
    """True for the example values shipped in .env.example."""
    return "replace-with" in str(value or "").lower()


def _secret(name: str) -> str:
    value = os.environ.get(name, "")
    if len(value) < MIN_SECRET_LENGTH or is_placeholder(value):
        raise ConfigurationError(
            f"{name} is missing, still the .env.example placeholder, or shorter than "
            f"{MIN_SECRET_LENGTH} characters. "
            f"Generate one with: python -c \"import secrets; print(secrets.token_hex(32))\" "
            f"and put it in .env (see .env.example).")
    return value


def flask_secret_key() -> str:
    return _secret("FLASK_SECRET_KEY")


def robase_api_key() -> str:
    """The Robase API key (``robe_…``). Read only by robase_client."""
    return os.environ.get("ROBASE_API_KEY", "").strip()
