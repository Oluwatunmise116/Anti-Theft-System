"""
Settings → Internet Wi-Fi: the admin-only endpoints, the connect jobs, the
sudo bridge to the helper and the reachability checks.

The helper is replaced by FakeBackend and the internet by FakeProber (or a
fake _https_get), so nothing here changes a live network setting, runs sudo,
or sends traffic. conftest.py fails any test that tries.
"""
import importlib.util
import json
import logging
import subprocess
import threading
from types import SimpleNamespace

import pytest

import app as application
import config as cfg
import face_manager as fm
import robase_client as robase
import wifi_uplink
from gate_helpers import add_holder, registered_trip, rows, staff_client

SECRET = "s3cr3t-Pa55 $(reboot) `id` \"'\\"
SAVED_UUID = "0f0e0d0c-0b0a-4908-8706-050403020100"
HOTSPOT_UUID = "84444be0-0b56-4f03-86d2-ba90fd229a67"
USB_IP = [{"address": "192.168.43.20", "prefix": 24}]


def status_reply(connected=True):
    connection = {"uuid": SAVED_UUID, "name": "SecureDrive Uplink: Home", "ssid": "Home",
                  "ssid_hex": b"Home".hex(), "managed_here": True, "protected": False,
                  "state": "connected", "ipv4": USB_IP, "gateway": "192.168.43.1",
                  "dns": ["192.168.43.1"], "signal": 70, "subnet_conflict": None}
    return {"adapter": {"present": True, "interface": "wlan1", "driver": "rtl8xxxu",
                        "mac": "5c:62:8b:d8:29:b3", "usb_id": "0bda:8179", "port_path": "usb-0:2",
                        "managed": True, "state_code": 100 if connected else 30,
                        "state": "connected" if connected else "disconnected", "reason_code": 0,
                        "problem": None},
            "connection": connection if connected else None,
            "saved": [{"uuid": SAVED_UUID, "ssid": "Home", "ssid_hex": b"Home".hex(), "hidden": False,
                       "security": "wpa-psk", "autoconnect": True, "active": connected}],
            "hotspot": [{"name": "PiHotspot", "ssid": "Secure_Drive", "active": True,
                         "interface": "wlan0", "on_usb_adapter": False}],
            "unmanaged_wifi_profiles": 7, "nm_version": "1.52.1"}


CONNECT_REPLY = {"uuid": "12345678-1234-4234-8234-123456789abc", "ssid": "Cafe", "interface": "wlan1",
                 "ipv4": USB_IP, "gateway": "192.168.43.1", "dns": ["192.168.43.1"]}


class FakeBackend:
    def __init__(self):
        self.calls = []
        self.status = status_reply()
        self.errors = {}
        self.gate = None

    def call(self, op, request=None, password=None):
        self.calls.append({"op": op, "request": dict(request or {}), "password": password})
        if op in ("connect", "reconnect") and self.gate is not None:
            self.gate.wait(5)
        if op in self.errors:
            raise self.errors[op]
        return json.loads(json.dumps({
            "status": self.status, "connect": CONNECT_REPLY, "reconnect": {**CONNECT_REPLY, "ssid": "Home"},
            "scan": {"networks": [{"ssid": "Cafe", "ssid_hex": b"Cafe".hex(), "security": "wpa-psk",
                                   "security_label": "WPA2 Personal", "supported": True, "signal": 80},
                                  {"ssid": "Free", "ssid_hex": b"Free".hex(), "security": "open",
                                   "security_label": "Open", "supported": True, "signal": 40}],
                     "hidden_networks": 0, "fresh": True},
            "disconnect": {"disconnected": True, "was_connected": True, "ssid": "Home"},
            "forget": {"forgotten": True, "ssid": "Home"},
        }[op]))

    def ops(self, name):
        return [c for c in self.calls if c["op"] == name]


def probe(state, detail="", peer="203.0.113.9"):
    return {"state": state, "detail": detail or state, "peer": peer}


class FakeProber:
    def __init__(self):
        self.calls = []
        self.results = {("internet", None): probe("reachable"), ("robase", None): probe("reachable"),
                        ("internet", "wlan1"): probe("reachable"), ("robase", "wlan1"): probe("reachable")}

    def internet(self, interface=None):
        self.calls.append(("internet", interface))
        return dict(self.results[("internet", interface)])

    def robase(self, interface=None):
        self.calls.append(("robase", interface))
        return dict(self.results[("robase", interface)])

    def route_interface(self, ip):
        return "eth0"


@pytest.fixture()
def wifi(gate_db):
    application.app.config["TESTING"] = True
    backend, prober, clock = FakeBackend(), FakeProber(), [100.0]
    service = wifi_uplink.WifiUplink(backend=backend, prober=prober, clock=lambda: clock[0])
    wifi_uplink.set_service_override(service)
    return SimpleNamespace(service=service, backend=backend, prober=prober, clock=clock)


@pytest.fixture()
def admin(wifi):
    return staff_client(application.app, "admin1", "admin")


@pytest.fixture()
def operator(wifi):
    return staff_client(application.app, "op1", "operator")


ENDPOINTS = [
    ("get", "/settings/wifi/status", None), ("post", "/settings/wifi/scan", {}),
    ("post", "/settings/wifi/connect", {"ssid_hex": "6162", "password": "12345678"}),
    ("post", "/settings/wifi/reconnect", {"uuid": SAVED_UUID}), ("get", "/settings/wifi/jobs/abc", None),
    ("post", "/settings/wifi/disconnect", {}), ("post", "/settings/wifi/forget", {"uuid": SAVED_UUID}),
    ("get", "/settings/wifi/connectivity", None),
]


def call(client, method, path, body=None):
    return getattr(client, method)(path, json=body) if body is not None else getattr(client, method)(path)


def connect(client, **body):
    return client.post("/settings/wifi/connect", json=body)


def finished(wifi, response):
    assert response.status_code == 202, response.get_json()
    return wifi.service.wait_job(response.get_json()["job"]["id"])


def wifi_events():
    return rows("SELECT event, detail FROM security_events WHERE event LIKE 'wifi_uplink_%' ORDER BY id")


# ── access control ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("method,path,body", ENDPOINTS)
def test_operators_cannot_use_internet_wifi(wifi, operator, method, path, body):
    assert call(operator, method, path, body).status_code == 403
    assert wifi.backend.calls == []


@pytest.mark.parametrize("method,path,body", ENDPOINTS)
def test_signed_out_requests_are_refused(wifi, method, path, body):
    assert call(application.app.test_client(), method, path, body).status_code in (400, 401)
    assert wifi.backend.calls == []


@pytest.mark.parametrize("method,path,body", ENDPOINTS)
def test_every_endpoint_needs_the_csrf_token_even_reads(wifi, admin, method, path, body):
    admin.environ_base.pop("HTTP_X_CSRF_TOKEN")
    assert call(admin, method, path, body).status_code == 400
    admin.environ_base["HTTP_X_CSRF_TOKEN"] = "forged"
    assert call(admin, method, path, body).status_code == 400
    assert wifi.backend.calls == []


def test_the_settings_page_shows_internet_wifi_to_administrators_only(wifi, admin, operator, monkeypatch):
    monkeypatch.setattr(fm, "list_cameras", lambda: [])
    page = admin.get("/settings").get_data(as_text=True)
    assert 'id="wifi-card"' in page and "Internet Wi-Fi" in page and "Camera Device" in page
    assert "'X-CSRF-Token': CSRF" in page and "innerHTML" not in page.split('id="wifi-card"')[1].split("</script>")[0]
    assert 'id="wifi-card"' not in operator.get("/settings").get_data(as_text=True)
    assert wifi.backend.calls == []  # the page itself never runs the helper


# ── status ───────────────────────────────────────────────────────────────────

def test_status_is_cached_briefly(wifi, admin):
    first = admin.get("/settings/wifi/status").get_json()
    assert first["available"] and first["connection"]["ssid"] == "Home" and first["busy"] is False
    admin.get("/settings/wifi/status")
    assert len(wifi.backend.ops("status")) == 1
    admin.get("/settings/wifi/status?refresh=1")
    assert len(wifi.backend.ops("status")) == 2


def test_adapter_problems_are_explained_in_plain_language(wifi, admin):
    wifi.backend.status["adapter"]["problem"] = "adapter_unmanaged"
    adapter = admin.get("/settings/wifi/status").get_json()["adapter"]
    assert "NetworkManager is not managing the USB Wi-Fi adapter" in adapter["problem_message"]


def test_a_missing_helper_is_explained(gate_db, monkeypatch, tmp_path):
    application.app.config["TESTING"] = True
    monkeypatch.setattr(wifi_uplink, "HELPER_PATH", str(tmp_path / "not-installed"))
    wifi_uplink.set_service_override(wifi_uplink.WifiUplink(prober=FakeProber()))
    client = staff_client(application.app, "admin2", "admin")
    view = client.get("/settings/wifi/status").get_json()
    assert view["available"] is False and view["error"]["code"] == "helper_missing"
    assert "README_WIFI_UPLINK.md" in view["error"]["message"]
    response = client.post("/settings/wifi/scan", json={})
    assert response.status_code == 503 and response.get_json()["code"] == "helper_missing"


# ── connecting ───────────────────────────────────────────────────────────────

def test_a_successful_connection_reports_the_address_and_internet(wifi, admin):
    job = finished(wifi, connect(admin, ssid_hex=b"Cafe".hex(), password=SECRET))
    assert job["state"] == "succeeded" and job["warning"] is None
    assert job["message"] == "Connected to Cafe (192.168.43.20). Internet and Robase are reachable."
    assert job["result"]["interface"] == "wlan1"
    sent = wifi.backend.ops("connect")[0]
    assert sent["password"] == SECRET and "password" not in sent["request"]
    assert sent["request"] == {"ssid_hex": b"Cafe".hex(), "hidden": False, "security": None}
    assert ("internet", "wlan1") in wifi.prober.calls and ("robase", "wlan1") in wifi.prober.calls
    assert [e["event"] for e in wifi_events()] == ["wifi_uplink_connect"]
    assert json.loads(wifi_events()[0]["detail"])["outcome"] == "succeeded"


def test_wifi_without_internet_is_a_warning_not_a_success(wifi, admin):
    wifi.prober.results[("internet", "wlan1")] = probe("unreachable", "No connection could be made.")
    wifi.prober.results[("robase", "wlan1")] = probe("unreachable")
    job = finished(wifi, connect(admin, ssid_hex=b"Cafe".hex(), password="correct horse"))
    assert job["state"] == "succeeded" and job["warning"] == "no_internet"
    assert "the internet is not reachable through it" in job["message"]


def test_robase_unreachable_despite_internet_is_reported(wifi, admin):
    wifi.prober.results[("robase", "wlan1")] = probe("unreachable")
    job = finished(wifi, connect(admin, ssid_hex=b"Cafe".hex(), password="correct horse"))
    assert job["warning"] == "robase_unreachable" and "Robase could not be reached" in job["message"]


@pytest.mark.parametrize("code,text", [
    ("wrong_password", "The Wi-Fi password was not accepted."),
    ("timeout", "The Wi-Fi operation timed out."),
    ("subnet_conflict", "same addresses as the Secure_Drive hotspot"),
    ("adapter_missing", "The USB Wi-Fi adapter was not found."),
    ("network_not_found", "isn't in range"),
])
def test_failures_are_reported_in_plain_language(wifi, admin, code, text):
    wifi.backend.errors["connect"] = wifi_uplink.WifiError(code, "helper detail", restored=True)
    job = finished(wifi, connect(admin, ssid_hex=b"Cafe".hex(), password=SECRET))
    assert job["state"] == "failed" and job["error"]["code"] == code and text in job["message"]
    assert job["restored"] is True and "helper detail" not in job["message"]
    detail = json.loads(wifi_events()[0]["detail"])
    assert detail == {"ssid": "Cafe", "hidden": False, "outcome": "failed", "error": code, "restored": True}


def test_the_password_never_leaves_the_helper_call(wifi, admin, caplog):
    caplog.set_level(logging.DEBUG)
    ok = finished(wifi, connect(admin, ssid_hex=b"Cafe".hex(), password=SECRET))
    wifi.backend.errors["connect"] = wifi_uplink.WifiError("wrong_password")
    failed = finished(wifi, connect(admin, ssid_hex=b"Cafe".hex(), password=SECRET))
    wifi.backend.errors["connect"] = RuntimeError(f"boom {SECRET}")
    crashed = finished(wifi, connect(admin, ssid_hex=b"Cafe".hex(), password=SECRET))
    seen = [json.dumps(j) for j in (ok, failed, crashed)]
    seen += [admin.get(f"/settings/wifi/jobs/{j['id']}").get_data(as_text=True) for j in (ok, failed)]
    seen += [admin.get("/settings/wifi/status?refresh=1").get_data(as_text=True), caplog.text]
    seen += [json.dumps(r) for r in rows("SELECT * FROM security_events")]
    seen += [json.dumps(c["request"]) for c in wifi.backend.calls]
    assert crashed["error"]["code"] == "helper_error"
    assert not any(SECRET in text for text in seen)


def test_only_one_wifi_operation_runs_at_a_time(wifi, admin):
    wifi.backend.gate = threading.Event()
    first = connect(admin, ssid_hex=b"Cafe".hex(), password="correct horse")
    assert first.status_code == 202
    busy = connect(admin, ssid_hex=b"Cafe".hex(), password="correct horse")
    assert busy.status_code == 409 and "still running" in busy.get_json()["error"]
    for method, path, body in (("post", "/settings/wifi/scan", {}), ("post", "/settings/wifi/disconnect", {}),
                               ("post", "/settings/wifi/forget", {"uuid": SAVED_UUID}),
                               ("post", "/settings/wifi/reconnect", {"uuid": SAVED_UUID})):
        assert call(admin, method, path, body).status_code == 409
    view = admin.get("/settings/wifi/status?refresh=1").get_json()
    assert view["busy"] and view["job"]["state"] == "running" and view["job"]["ssid"] == "Cafe"
    wifi.backend.gate.set()
    assert wifi.service.wait_job(first.get_json()["job"]["id"])["state"] == "succeeded"
    assert finished(wifi, connect(admin, ssid_hex=b"Cafe".hex(), password="correct horse"))["state"] == "succeeded"
    assert len(wifi.backend.ops("connect")) == 2 and len(wifi.backend.ops("disconnect")) == 0


@pytest.mark.parametrize("body", [
    {"ssid_hex": "zz", "password": "12345678"}, {"ssid_hex": "61" * 33, "password": "12345678"},
    {"ssid_hex": "6162", "password": 12345678}, {"ssid_hex": "6162", "password": "x" * 129},
    {"ssid_hex": "6162", "hidden": "yes"}, {"hidden": True, "ssid": "x" * 33, "security": "wpa-psk"},
    {"hidden": True, "ssid": "", "security": "wpa-psk"}, {"hidden": True, "ssid": "ok", "security": "wep"},
    {"hidden": True, "ssid": "ok", "security": "wpa-psk"}, {},
])
def test_invalid_requests_are_refused_before_the_helper_runs(wifi, admin, body):
    assert connect(admin, **body).status_code == 400
    assert wifi.backend.ops("connect") == []


def test_a_known_secured_network_needs_a_password_and_an_open_one_drops_it(wifi, admin):
    admin.post("/settings/wifi/scan", json={})
    response = connect(admin, ssid_hex=b"Cafe".hex())
    assert response.status_code == 400 and response.get_json()["error"] == "Enter the Wi-Fi password."
    finished(wifi, connect(admin, ssid_hex=b"Free".hex(), password="ignored"))
    assert wifi.backend.ops("connect")[0]["password"] is None


def test_hidden_network_names_are_sent_as_bytes(wifi, admin):
    name = "a:b\\c \"q\" $(x) -o ☕"
    finished(wifi, connect(admin, hidden=True, ssid=name, security="sae", password="wpa3 password"))
    sent = wifi.backend.ops("connect")[0]["request"]
    assert sent == {"ssid_hex": name.encode().hex(), "hidden": True, "security": "sae"}


# ── saved networks ───────────────────────────────────────────────────────────

def test_reconnect_disconnect_and_forget(wifi, admin):
    job = finished(wifi, admin.post("/settings/wifi/reconnect", json={"uuid": SAVED_UUID}))
    assert job["state"] == "succeeded" and wifi.backend.ops("reconnect")[0]["request"] == {"uuid": SAVED_UUID}
    assert admin.post("/settings/wifi/disconnect", json={}).get_json()["disconnected"]
    assert admin.post("/settings/wifi/forget", json={"uuid": SAVED_UUID}).get_json()["forgotten"]
    assert wifi.backend.ops("forget")[0]["request"] == {"uuid": SAVED_UUID}
    assert [e["event"] for e in wifi_events()] == ["wifi_uplink_reconnect", "wifi_uplink_disconnect",
                                                   "wifi_uplink_forget"]


@pytest.mark.parametrize("uuid", [HOTSPOT_UUID, "not-a-uuid", None, 42])
def test_only_networks_in_the_helpers_saved_list_can_be_targeted(wifi, admin, uuid):
    for path in ("/settings/wifi/forget", "/settings/wifi/reconnect"):
        response = admin.post(path, json={"uuid": uuid})
        assert response.status_code == 403 and response.get_json()["code"] == "not_permitted"
    assert wifi.backend.ops("forget") == [] and wifi.backend.ops("reconnect") == []


def test_a_helper_refusal_is_passed_on(wifi, admin):
    wifi.backend.errors["disconnect"] = wifi_uplink.WifiError("adapter_is_hotspot")
    response = admin.post("/settings/wifi/disconnect", json={})
    assert response.status_code == 409 and "carrying the Secure_Drive hotspot" in response.get_json()["error"]


# ── the sudo bridge ──────────────────────────────────────────────────────────

@pytest.fixture()
def bridge(monkeypatch, tmp_path):
    helper = tmp_path / "wifi-uplink-helper"
    helper.write_text("#!/bin/false\n")
    monkeypatch.setattr(wifi_uplink, "HELPER_PATH", str(helper))
    monkeypatch.setattr(wifi_uplink.HelperBackend, "argv", ("/usr/bin/sudo", "-n", str(helper)))
    seen = SimpleNamespace(calls=[], reply=None)

    def run(argv, data, timeout):
        seen.calls.append((argv, data, timeout))
        if isinstance(seen.reply, BaseException):
            raise seen.reply
        return seen.reply

    monkeypatch.setattr(wifi_uplink, "_run_helper_process", run)
    seen.helper = str(helper)
    return seen


def done(stdout=b"", stderr=b"", code=0):
    return subprocess.CompletedProcess([], code, stdout=stdout, stderr=stderr)


def test_the_helper_gets_the_password_on_stdin_only(bridge):
    bridge.reply = done(b'{"ok": true, "result": {"uuid": "x"}}')
    odd = "$(reboot); -o ☕".encode().hex()
    assert wifi_uplink.HelperBackend().call("connect", {"ssid_hex": odd}, password=SECRET) == {"uuid": "x"}
    argv, data, timeout = bridge.calls[0]
    assert argv == ["/usr/bin/sudo", "-n", bridge.helper] and timeout == 110
    assert json.loads(data) == {"op": "connect", "ssid_hex": odd, "password": SECRET}
    assert not any(SECRET in part for part in argv)


def test_the_process_is_started_without_a_shell(monkeypatch):
    spec = importlib.util.spec_from_file_location("wifi_uplink_copy", wifi_uplink.__file__)
    fresh = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fresh)
    captured = {}
    monkeypatch.setattr(subprocess, "run", lambda argv, **kw: captured.update(argv=argv, **kw))
    fresh._run_helper_process(["/usr/bin/sudo", "-n", "/x"], b"{}", 5)
    assert captured["shell"] is False and captured["input"] == b"{}" and captured["timeout"] == 5
    assert captured["env"] == {"PATH": "/usr/sbin:/usr/bin:/sbin:/bin", "LANG": "C.UTF-8"}


@pytest.mark.parametrize("reply,code", [
    (subprocess.TimeoutExpired(["sudo"], 110), "timeout"),
    (done(stderr=b"sudo: a password is required\n", code=1), "permission_denied"),
    (done(stderr=b"Sorry, user x is not allowed to execute ...", code=1), "permission_denied"),
    (done(b"garbage"), "helper_error"),
    (done(b'{"ok": false, "error": "wrong_password", "detail": "reason 7", "restored": true}'),
     "wrong_password"),
    (done(b'{"ok": false, "error": "rm -rf /"}'), "helper_error"),
])
def test_helper_failures_are_mapped(bridge, reply, code):
    bridge.reply = reply
    with pytest.raises(wifi_uplink.WifiError) as err:
        wifi_uplink.HelperBackend().call("connect", {"ssid_hex": "6162"}, password=SECRET)
    assert err.value.code == code and SECRET not in str(err.value) and SECRET not in err.value.message
    if code == "wrong_password":
        assert err.value.as_dict()["restored"] is True


# ── reachability ─────────────────────────────────────────────────────────────

def test_checks_are_labelled_by_the_path_they_used(wifi, admin):
    wifi.prober.results[("internet", "wlan1")] = probe("unreachable", "No connection could be made.")
    view = admin.get("/settings/wifi/connectivity").get_json()
    assert view["usb"]["interface"] == "wlan1" and view["usb"]["ipv4"] == "192.168.43.20"
    assert view["usb"]["internet"]["state"] == "unreachable"
    assert view["system"]["route_interface"] == "eth0" and view["system"]["internet"]["state"] == "reachable"
    assert view["robase_host"] == "api.robase.dev"


def test_nothing_is_attributed_to_a_disconnected_usb_adapter(wifi, admin):
    wifi.backend.status = status_reply(connected=False)
    view = admin.get("/settings/wifi/connectivity").get_json()
    assert view["usb"]["internet"]["state"] == "not_tested" and view["usb"]["robase"]["state"] == "not_tested"
    assert all(interface is None for _, interface in wifi.prober.calls)


def test_checks_are_cached_and_forced_checks_are_rate_limited(wifi, admin):
    admin.get("/settings/wifi/connectivity")
    assert len(wifi.prober.calls) == 4
    admin.get("/settings/wifi/connectivity")
    admin.get("/settings/wifi/connectivity?refresh=1")
    assert len(wifi.prober.calls) == 4
    wifi.clock[0] += 6
    admin.get("/settings/wifi/connectivity?refresh=1")
    assert len(wifi.prober.calls) == 8
    wifi.clock[0] += 6
    admin.get("/settings/wifi/connectivity")
    assert len(wifi.prober.calls) == 8


def test_the_browser_cannot_choose_what_is_probed(wifi, admin, monkeypatch, sms_mock):
    fetched = []

    def fake_get(url, interface, timeout):
        fetched.append((url, interface))
        return (200, b'{"status": "ok"}', "198.51.100.7") if url.endswith("/health") else (204, b"", "142.250.1.1")

    monkeypatch.setattr(wifi_uplink, "_https_get", fake_get)
    monkeypatch.setattr(wifi_uplink, "_route_interface", lambda ip: "eth0")
    wifi.service.prober = wifi_uplink.Prober()
    view = admin.get("/settings/wifi/connectivity?refresh=1&url=https://evil.example/&target=x").get_json()
    assert {url for url, _ in fetched} == {wifi_uplink.DEFAULT_INTERNET_PROBE, "https://api.robase.dev/health"}
    assert {iface for _, iface in fetched} == {None, "wlan1"}
    assert view["usb"]["robase"]["state"] == "reachable" and view["system"]["internet"]["state"] == "reachable"
    assert sms_mock.sent == []  # a reachability check never sends an OTP


@pytest.mark.parametrize("which,reply,state", [
    ("internet", (204, b"", "1.1.1.1"), "reachable"),
    ("internet", (200, b"<html>sign in</html>", "1.1.1.1"), "unexpected"),
    ("internet", wifi_uplink.ProbeError("tls_error", "TLS failed"), "tls_error"),
    ("internet", wifi_uplink.ProbeError("dns_error", "DNS failed"), "dns_error"),
    ("robase", (200, b'{"status": "ok"}', "1.1.1.1"), "reachable"),
    ("robase", (200, b"<html>portal</html>", "1.1.1.1"), "unexpected"),
    ("robase", (503, b"", "1.1.1.1"), "degraded"),
])
def test_probe_answers_are_interpreted(monkeypatch, which, reply, state):
    def fake_get(url, interface, timeout):
        assert url.startswith("https://") and timeout == wifi_uplink.PROBE_TIMEOUT
        if isinstance(reply, Exception):
            raise reply
        return reply

    monkeypatch.setattr(wifi_uplink, "_https_get", fake_get)
    assert getattr(wifi_uplink.Prober(), which)("wlan1")["state"] == state


def test_a_plain_http_probe_url_is_never_used(monkeypatch):
    real_load = cfg.load
    monkeypatch.setattr(cfg, "load", lambda: {**real_load(), "wifi_internet_probe_url": "http://example.com/"})
    assert wifi_uplink.Prober().internet()["state"] == "not_tested"  # _https_get would fail the test


# ── the gate is unaffected ───────────────────────────────────────────────────

def test_the_sms_fallback_stays_unauthorised_without_internet(gate_db, sms_mock):
    application.app.config["TESTING"] = True
    trip_id = registered_trip(add_holder())
    client = staff_client(application.app, "gate-op", "operator")
    sms_mock.fail_next_send = robase.RobaseError("no route to host", kind="network")
    response = client.post("/gate/exit/sms/send", json={"trip_id": trip_id})
    assert response.status_code == 504 and response.get_json()["error"]
    assert client.post("/gate/exit/confirm", json={"trip_id": trip_id}).status_code == 403
