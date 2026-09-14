"""
Shared fixtures.

Nothing here loads a real checkpoint or a real detector. The heavyweight
pieces are faked; the containment linkage, the voting rule and the
provenance logic all run for real, because those rules ARE the feature.

Follows the existing test style: real SQLite, faked hardware and models only.

Isolation (below) runs before any test module is imported: database.py runs
init_db() at import time against LICENSE_DB_PATH, so without it a test run
would open — and migrate — the real license.db. Secrets are fixed test
values, and SMS delivery is mocked; no test can reach the real Robase API.
"""
import os
import sys
import tempfile

import pytest

_TEST_ROOT = tempfile.mkdtemp(prefix="gate-tests-")
os.environ.update({
    "LICENSE_DB_PATH": os.path.join(_TEST_ROOT, "import-time.db"),
    "APP_ENV": "test",
    "FLASK_SECRET_KEY": "test-only-flask-secret-" + "0" * 32,
    "SMS_DELIVERY_MODE": "mock",
    "ROBASE_API_KEY": "",
    "ROBASE_BASE_URL": "",
    "ROBASE_SMS_LANGUAGE": "",
    "SMS_MOCK_OUTBOX": os.path.join(_TEST_ROOT, "sms_mock_outbox.jsonl"),
    "PHONE_DEFAULT_REGION": "NG",
    # Pinned to the defaults so a local .env can never change test behaviour
    # (app.py loads .env without overriding variables that are already set).
    "OTP_LENGTH": "6", "OTP_TTL_SECONDS": "300", "OTP_MAX_ATTEMPTS": "3",
    "OTP_RESEND_COOLDOWN_SECONDS": "60", "OTP_MAX_SENDS_PER_TRIP": "5",
    "OTP_MAX_FAILED_PER_TRIP": "6", "OTP_HOLDER_WINDOW_SECONDS": "3600",
    "OTP_MAX_SENDS_PER_HOLDER_WINDOW": "10", "OTP_MAX_FAILED_PER_HOLDER_WINDOW": "10",
    "EXIT_AUTHORIZATION_TTL_SECONDS": "120", "PASSCODE_MIN_LENGTH": "4",
    "PASSCODE_MAX_LENGTH": "8", "PASSCODE_MAX_ATTEMPTS": "5", "STAFF_SESSION_HOURS": "12",
})

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#: Hosts no test may ever contact.
_BLOCKED_HOSTS = ("robase.dev", "api.telegram.org")


def pytest_addoption(parser):
    parser.addoption(
        "--run-real-models", action="store_true", default=False,
        help="run tests that load real trained weights, when they exist")


def pytest_collection_modifyitems(config, items):
    """
    Skip real-model tests unless explicitly requested, so a normal pytest
    run never depends on a checkpoint being present or spends minutes in
    CPU forward passes.
    """
    if config.getoption("--run-real-models"):
        return
    skip = pytest.mark.skip(reason="needs --run-real-models (trained weights required)")
    for item in items:
        if "real_models" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(autouse=True)
def _no_real_sms(monkeypatch):
    """Any HTTP request to Robase fails the test — no real SMS, no credits."""
    import requests

    real_request = requests.Session.request

    def guarded(self, method, url, *args, **kwargs):
        if any(host in str(url) for host in _BLOCKED_HOSTS):
            raise AssertionError("tests must never call the real Robase API")
        return real_request(self, method, url, *args, **kwargs)

    monkeypatch.setattr(requests.Session, "request", guarded)
    yield
    import robase_client
    robase_client.set_client_override(None)


@pytest.fixture(autouse=True)
def _no_live_network(monkeypatch):
    """No test may run the privileged Wi-Fi helper, probe the internet or read
    live routes: the Internet Wi-Fi tests use fake backends only."""
    import wifi_uplink

    def refuse(*args, **kwargs):
        raise AssertionError("tests must never touch the live network")

    monkeypatch.setattr(wifi_uplink, "_run_helper_process", refuse)
    monkeypatch.setattr(wifi_uplink, "_https_get", refuse)
    monkeypatch.setattr(wifi_uplink, "_route_interface", refuse)
    yield
    wifi_uplink.set_service_override(None)


@pytest.fixture()
def gate_db(tmp_path, monkeypatch):
    """A fresh, fully migrated database for one test."""
    import database as db
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "gate.db"))
    db.init_db()
    return db.DB_PATH


@pytest.fixture()
def sms_mock():
    """An in-memory stand-in for Robase; texts land in sms_mock.sent."""
    import robase_client
    mock = robase_client.MockRobaseClient()
    robase_client.set_client_override(mock)
    yield mock
    robase_client.set_client_override(None)
