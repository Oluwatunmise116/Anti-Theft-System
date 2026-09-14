"""
Internet Wi-Fi: lets an administrator put the USB Wi-Fi adapter on a router or
a phone hotspot so Robase SMS has internet, without touching the Secure_Drive
hotspot that runs on the Pi's built-in adapter.

This module is the unprivileged half and never talks to NetworkManager
itself. Every change goes through the root-owned helper
(deploy/wifi_uplink/wifi_uplink_helper.py, installed at HELPER_PATH), which
re-checks the USB adapter's hardware identity and the profile rules on every
call. This side adds input validation, one network operation at a time,
background connect jobs the page polls, a short status cache, and bounded,
cached internet / Robase reachability checks.

Wi-Fi passwords travel only on the helper's stdin. They are never logged,
stored, put on a command line, or sent back to the browser; NetworkManager
keeps them in its root-only profile store.
"""
import concurrent.futures
import http.client
import ipaddress
import json
import logging
import os
import re
import socket
import ssl
import subprocess
import threading
import time
import uuid as uuidlib
from collections import OrderedDict
from urllib.parse import urlsplit

import config as cfg
import database as db
import verification_settings as vs

log = logging.getLogger(__name__)

#: The only command the sudoers rule allows (with no arguments).
HELPER_PATH = "/usr/local/libexec/secure-drive/wifi-uplink-helper"
SUDO_PATH = "/usr/bin/sudo"
IP_PATH = "/usr/sbin/ip" if os.path.exists("/usr/sbin/ip") else "/sbin/ip"
_CHILD_ENV = {"PATH": "/usr/sbin:/usr/bin:/sbin:/bin", "LANG": "C.UTF-8"}

#: Seconds before the helper is abandoned. Longer than the helper's own worst
#: case (45 s activation + 20 s restoring the previous network).
TIMEOUTS = {"status": 15, "scan": 30, "connect": 110, "reconnect": 90,
            "disconnect": 25, "forget": 25}

STATUS_TTL = 4.0
CONNECTIVITY_TTL = 30.0
CONNECTIVITY_MIN_INTERVAL = 5.0
PROBE_TIMEOUT = 4.0
PROBE_DEADLINE = 10.0
MAX_JOBS = 20
MAX_PASSWORD_CHARS = 128

DEFAULT_INTERNET_PROBE = "https://connectivitycheck.gstatic.com/generate_204"

SECURITY_CHOICES = ("open", "wpa-psk", "sae")
_SSID_HEX = re.compile(r"^(?:[0-9a-f]{2}){1,32}$")
_UUID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
_IFACE = re.compile(r"^[A-Za-z0-9_.-]{1,15}$")

GUIDE = "README_WIFI_UPLINK.md"

MESSAGES = {
    "helper_missing": f"Wi-Fi control is not installed on this Pi yet. An administrator must "
                      f"run the install step in {GUIDE}.",
    "permission_denied": f"The web service is not allowed to manage the USB Wi-Fi adapter. "
                         f"Re-run the install step that adds the sudoers rule ({GUIDE}).",
    "config_missing": f"The USB Wi-Fi adapter has not been set up yet. See {GUIDE}.",
    "config_invalid": f"The USB Wi-Fi adapter settings file is incomplete. See {GUIDE}.",
    "config_insecure": "The USB Wi-Fi settings file must be owned by root and not writable "
                       "by others, so it was ignored.",
    "registry_invalid": f"The list of saved uplink networks could not be read. See {GUIDE}.",
    "nm_unavailable": "NetworkManager is not responding, so the USB Wi-Fi adapter can't be "
                      "controlled right now.",
    "adapter_missing": "The USB Wi-Fi adapter was not found. Check that it is plugged in, "
                       "then press Refresh.",
    "adapter_ambiguous": "More than one adapter matches the configured USB Wi-Fi adapter, so "
                         "nothing was changed. Unplug the extra adapter, or pin the USB port "
                         "in the config.",
    "adapter_unmanaged": f"NetworkManager is not managing the USB Wi-Fi adapter, so it can't be "
                         f"used from here. See '{GUIDE}' → Adapter is unmanaged.",
    "adapter_unavailable": "The USB Wi-Fi adapter is not ready (it may be switched off or still "
                           "starting). Try again in a moment.",
    "adapter_is_hotspot": "The configured adapter is carrying the Secure_Drive hotspot, so it "
                          "was not changed.",
    "protected_profile": "That connection belongs to the Secure_Drive hotspot and can't be "
                         "changed here.",
    "not_permitted": "That saved network isn't managed by this page, so it was left alone.",
    "profile_missing": "That saved network no longer exists. Press Refresh.",
    "invalid_ssid": "The network name must be 1-32 bytes long.",
    "invalid_password": "That password can't be used for this network.",
    "invalid_request": "The request was not valid. Reload the page and try again.",
    "unsupported_security": "This network's security type isn't supported here. Only open, "
                            "WPA/WPA2 Personal and WPA3 Personal networks can be used.",
    "network_not_found": "That network isn't in range right now. Press Scan again, or check "
                         "the name of a hidden network.",
    "wrong_password": "The Wi-Fi password was not accepted. Check it and try again.",
    "auth_timeout": "The network didn't accept the connection in time — usually a wrong "
                    "password or a weak signal.",
    "no_ip": "Joined the Wi-Fi network, but it didn't give the Pi an IP address.",
    "subnet_conflict": "That network uses the same addresses as the Secure_Drive hotspot, which "
                       "would cut hotspot devices off from this app, so it was disconnected. "
                       "Use another network, or change the router's or phone's address range.",
    "timeout": "The Wi-Fi operation timed out. The network may be out of range or slow to "
               "respond.",
    "activation_failed": "The connection failed. Check the network and try again.",
    "busy": "Another Wi-Fi operation is still running. Wait for it to finish.",
    "helper_error": "The Wi-Fi helper gave an unexpected answer. Check the service log.",
}

#: Codes whose helper-written detail is more useful than the generic text.
_DETAIL_REPLACES = {"invalid_password", "invalid_ssid", "unsupported_security"}
_DETAIL_APPENDS = {"config_invalid", "config_insecure"}

_HTTP_STATUS = {
    "busy": 409, "invalid_ssid": 400, "invalid_password": 400, "invalid_request": 400,
    "unsupported_security": 400, "not_permitted": 403, "protected_profile": 403,
    "helper_missing": 503, "permission_denied": 503, "config_missing": 503,
    "config_invalid": 503, "config_insecure": 503, "registry_invalid": 503,
    "nm_unavailable": 503, "timeout": 504, "helper_error": 502,
}


class WifiError(Exception):
    """A failure with a stable code and a plain-language message.

    str(exc) is only the code, so an exception can never carry a secret into
    a log line or a traceback."""

    def __init__(self, code, detail="", **extra):
        super().__init__(code)
        self.code = code if code in MESSAGES else "helper_error"
        self.detail = detail or ""
        self.extra = {k: v for k, v in extra.items() if v is not None}

    @property
    def message(self):
        base = MESSAGES[self.code]
        if self.detail and self.code in _DETAIL_REPLACES:
            return self.detail
        if self.detail and self.code in _DETAIL_APPENDS:
            return f"{base} ({self.detail})"
        return base

    @property
    def http_status(self):
        return _HTTP_STATUS.get(self.code, 409)

    def as_dict(self):
        view = {"code": self.code, "message": self.message}
        for key in ("restored", "hotspot_subnet"):
            if key in self.extra:
                view[key] = self.extra[key]
        return view


# ── input validation ─────────────────────────────────────────────────────────

def parse_connect_request(data):
    """ssid_hex / hidden / security / password from the browser's JSON."""
    if not isinstance(data, dict):
        raise WifiError("invalid_request")
    hidden = data.get("hidden", False)
    if not isinstance(hidden, bool):
        raise WifiError("invalid_request")
    password = data.get("password")
    if password is not None and not isinstance(password, str):
        raise WifiError("invalid_password", "The password must be text.")
    if password is not None and len(password) > MAX_PASSWORD_CHARS:
        raise WifiError("invalid_password", "That password is too long.")
    if hidden:
        ssid = data.get("ssid")
        if not isinstance(ssid, str) or "\x00" in ssid or not 1 <= len(ssid.encode("utf-8")) <= 32:
            raise WifiError("invalid_ssid", "Enter the network name (1-32 bytes).")
        ssid_hex = ssid.encode("utf-8").hex()
        security = data.get("security")
        if security not in SECURITY_CHOICES:
            raise WifiError("unsupported_security", "Choose the network's security type.")
    else:
        ssid_hex = data.get("ssid_hex")
        if not isinstance(ssid_hex, str) or not _SSID_HEX.match(ssid_hex):
            raise WifiError("invalid_ssid", "Pick a network from the list, or press Scan again.")
        security = None
    return {"ssid_hex": ssid_hex, "hidden": hidden, "security": security,
            "password": password or None}


def parse_uuid(value):
    if not isinstance(value, str) or not _UUID.match(value):
        raise WifiError("not_permitted")
    return value


def ssid_label(ssid_hex):
    return bytes.fromhex(ssid_hex).decode("utf-8", errors="replace")


# ── the privileged helper ────────────────────────────────────────────────────

def _run_helper_process(argv, data, timeout):
    """The only place a process is started for Wi-Fi changes (tests replace it)."""
    return subprocess.run(argv, input=data, capture_output=True, timeout=timeout,
                          shell=False, check=False, env=_CHILD_ENV, close_fds=True)


class HelperBackend:
    """Runs the root-owned helper through `sudo -n` with a JSON request on stdin."""

    argv = (SUDO_PATH, "-n", HELPER_PATH)

    def call(self, op, request=None, password=None):
        if not os.path.exists(HELPER_PATH):
            raise WifiError("helper_missing")
        payload = {"op": op, **(request or {})}
        if password is not None:
            payload["password"] = password
        data = json.dumps(payload).encode("utf-8")
        payload = None
        try:
            proc = _run_helper_process(list(self.argv), data, TIMEOUTS[op])
        except subprocess.TimeoutExpired:
            raise WifiError("timeout") from None
        except OSError:
            raise WifiError("helper_missing") from None
        finally:
            data = None
        return self._parse(op, proc)

    @staticmethod
    def _parse(op, proc):
        try:
            reply = json.loads((proc.stdout or b"")[:1_000_000].decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            reply = None
        if isinstance(reply, dict) and reply.get("ok") is True and isinstance(reply.get("result"), dict):
            return reply["result"]
        if isinstance(reply, dict) and reply.get("ok") is False:
            raise WifiError(str(reply.get("error") or "helper_error")[:40],
                            str(reply.get("detail") or "")[:300],
                            restored=reply.get("restored"),
                            hotspot_subnet=reply.get("hotspot_subnet"))
        stderr = (proc.stderr or b"")[:2000].decode("utf-8", errors="replace")
        if any(s in stderr for s in ("password is required", "not allowed to execute",
                                     "may not run sudo", "a terminal is required")):
            raise WifiError("permission_denied")
        log.warning("Wi-Fi helper '%s' exited with %s and no valid reply", op, proc.returncode)
        raise WifiError("helper_error")


# ── reachability checks ──────────────────────────────────────────────────────

class ProbeError(Exception):
    def __init__(self, state, detail):
        super().__init__(state)
        self.state = state
        self.detail = detail


def _https_get(url, interface, timeout):
    """
    One TLS-validated HTTPS GET. With `interface`, the socket is pinned to
    that network interface (SO_BINDTODEVICE), so a reply proves that path
    works — it cannot silently leave through Ethernet instead. Name lookup
    uses the Pi's normal resolver. Returns (status, body, peer_ip).
    """
    parts = urlsplit(url)
    host, port = parts.hostname, parts.port or 443
    path = (parts.path or "/") + (f"?{parts.query}" if parts.query else "")
    try:
        infos = socket.getaddrinfo(host, port, socket.AF_INET, socket.SOCK_STREAM)
    except (socket.gaierror, UnicodeError):
        raise ProbeError("dns_error", "The server name could not be looked up (DNS).")
    peer = infos[0][4]
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    try:
        if interface:
            try:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_BINDTODEVICE,
                                interface.encode("ascii") + b"\0")
            except OSError:
                raise ProbeError("not_tested", "The check could not be pinned to the USB adapter.")
        sock.connect(peer)
        context = ssl.create_default_context()
        sock = context.wrap_socket(sock, server_hostname=host)
        conn = http.client.HTTPSConnection(host, port, timeout=timeout, context=context)
        conn.sock = sock
        conn.request("GET", path, headers={"User-Agent": "SecureDrive-connectivity-check/1",
                                           "Accept": "*/*", "Connection": "close"})
        response = conn.getresponse()
        return response.status, response.read(4096), peer[0]
    except ssl.SSLCertVerificationError:
        raise ProbeError("tls_error", "The secure (TLS) check failed — the network may need a "
                                      "sign-in page, or traffic is being intercepted.")
    except ssl.SSLError:
        raise ProbeError("tls_error", "The secure (TLS) connection could not be set up.")
    except (TimeoutError, socket.timeout):
        raise ProbeError("unreachable", "No answer in time.")
    except (OSError, http.client.HTTPException):
        raise ProbeError("unreachable", "No connection could be made.")
    finally:
        sock.close()


def _route_interface(ip):
    """The interface the kernel would use for `ip` (read-only `ip route get`)."""
    try:
        ipaddress.ip_address(ip)
        proc = subprocess.run([IP_PATH, "-j", "route", "get", ip], capture_output=True,
                              timeout=2, shell=False, check=False, env=_CHILD_ENV)
        routes = json.loads(proc.stdout or b"[]")
        return str(routes[0].get("dev") or "") or None
    except (ValueError, OSError, subprocess.TimeoutExpired, IndexError, AttributeError):
        return None


def internet_probe_url():
    url = str(cfg.get("wifi_internet_probe_url", DEFAULT_INTERNET_PROBE) or "")
    return url if _https_url(url) else None


def robase_health_url():
    try:
        base = vs.load().robase_base_url
    except vs.ConfigurationError:
        base = vs.DEFAULT_ROBASE_BASE_URL
    url = f"{base.rstrip('/')}/health"
    return url if _https_url(url) else None


def _https_url(url):
    try:
        parts = urlsplit(url)
    except ValueError:
        return False
    return parts.scheme == "https" and bool(parts.hostname) and not parts.username


class Prober:
    """Fixed, server-side targets only. Nothing here sends an SMS or uses credits."""

    def _fetch(self, url, interface):
        return _https_get(url, interface, PROBE_TIMEOUT)

    def internet(self, interface=None):
        url = internet_probe_url()
        if not url:
            return _result("not_tested", "The internet check URL must be an https:// address.")
        expect = cfg.get("wifi_internet_probe_expect_status", 204)
        try:
            status, _, peer = self._fetch(url, interface)
        except ProbeError as exc:
            return _result(exc.state, exc.detail)
        if status == expect:
            return _result("reachable", "Internet is reachable.", peer=peer)
        return _result("unexpected", f"Unexpected answer (HTTP {status}) — the network may need "
                                     f"a sign-in page.", peer=peer)

    def robase(self, interface=None):
        url = robase_health_url()
        if not url:
            return _result("not_tested", "The Robase address is not an https:// URL.")
        try:
            status, body, peer = self._fetch(url, interface)
        except ProbeError as exc:
            return _result(exc.state, exc.detail)
        if status == 200:
            try:
                healthy = json.loads(body.decode("utf-8")).get("status") == "ok"
            except (UnicodeDecodeError, ValueError, AttributeError):
                healthy = False
            if healthy:
                return _result("reachable", "Robase is reachable (health check — no SMS sent).",
                               peer=peer)
        if status >= 500:
            return _result("degraded", f"Robase answered but reports a problem (HTTP {status}).",
                           peer=peer)
        return _result("unexpected", f"Unexpected answer from Robase (HTTP {status}).", peer=peer)

    def route_interface(self, ip):
        return _route_interface(ip)


def _result(state, detail, peer=None):
    return {"state": state, "detail": detail, "peer": peer}


# ── the service ──────────────────────────────────────────────────────────────

STAGE_LABELS = {"connecting": "Connecting…", "checking_internet": "Checking internet access…"}


class WifiUplink:
    def __init__(self, backend=None, prober=None, clock=time.monotonic):
        self.backend = backend or HelperBackend()
        self.prober = prober or Prober()
        self._clock = clock
        self._op_lock = threading.Lock()      # one network-changing operation at a time
        self._state = threading.Lock()        # guards the caches and jobs below
        self._probe_lock = threading.Lock()   # one reachability check at a time
        self._jobs = OrderedDict()
        self._active_job = None
        self._status, self._status_at = None, float("-inf")
        self._reach, self._reach_at, self._reach_key = None, float("-inf"), None
        self._scan_security = {}

    # status ------------------------------------------------------------------
    def status(self, force=False):
        with self._state:
            if not force and self._status and self._clock() - self._status_at < STATUS_TTL:
                return self._decorate(self._status)
        try:
            view = {"available": True, "error": None, **self.backend.call("status")}
            adapter = view.get("adapter")
            if isinstance(adapter, dict) and adapter.get("problem"):
                view["adapter"] = {**adapter,
                                   "problem_message": WifiError(adapter["problem"]).message}
        except WifiError as exc:
            view = {"available": False, "error": exc.as_dict()}
        view["checked_at"] = time.time()
        with self._state:
            self._status, self._status_at = view, self._clock()
            return self._decorate(view)

    def _decorate(self, view):
        view = dict(view)
        view["busy"] = self._op_lock.locked()
        job = self._jobs.get(self._active_job) if self._active_job else None
        view["job"] = dict(job) if job else None
        return view

    def _invalidate(self):
        with self._state:
            self._status_at = float("-inf")
            self._reach_at = float("-inf")

    def _saved(self, uuid):
        """Resolve a saved-network UUID against the helper's own list."""
        for attempt in (False, True):
            for item in self.status(force=attempt).get("saved") or []:
                if item.get("uuid") == uuid:
                    return item
        raise WifiError("not_permitted")

    # operations --------------------------------------------------------------
    def _begin(self):
        if not self._op_lock.acquire(blocking=False):
            raise WifiError("busy")

    def scan(self):
        self._begin()
        try:
            result = self.backend.call("scan")
        finally:
            self._op_lock.release()
            self._invalidate()
        with self._state:
            self._scan_security = {n.get("ssid_hex"): n.get("security")
                                   for n in result.get("networks") or []}
        return result

    def start_connect(self, data, actor):
        params = parse_connect_request(data)
        secret = [params.pop("password")]
        known = params["security"] or self._scan_security.get(params["ssid_hex"])
        if known == "open":
            secret[0] = None
        elif known and not secret[0]:
            raise WifiError("invalid_password", "Enter the Wi-Fi password.")
        label = ssid_label(params["ssid_hex"])
        self._begin()

        def work():
            try:
                return self.backend.call("connect", params, password=secret[0])
            finally:
                secret[0] = None

        return self._start_job("connect", label, work, actor, "wifi_uplink_connect",
                               {"ssid": label, "hidden": params["hidden"]})

    def start_reconnect(self, uuid, actor):
        uuid = parse_uuid(uuid)
        saved = self._saved(uuid)
        self._begin()
        return self._start_job("reconnect", saved["ssid"],
                               lambda: self.backend.call("reconnect", {"uuid": uuid}),
                               actor, "wifi_uplink_reconnect", {"ssid": saved["ssid"]})

    def disconnect(self, actor):
        self._begin()
        try:
            result = self.backend.call("disconnect")
            self._audit("wifi_uplink_disconnect", actor,
                        {"outcome": "succeeded", "ssid": result.get("ssid")})
            return result
        except WifiError as exc:
            self._audit("wifi_uplink_disconnect", actor, {"outcome": "failed", "error": exc.code})
            raise
        finally:
            self._op_lock.release()
            self._invalidate()

    def forget(self, uuid, actor):
        uuid = parse_uuid(uuid)
        saved = self._saved(uuid)
        self._begin()
        try:
            result = self.backend.call("forget", {"uuid": uuid})
            self._audit("wifi_uplink_forget", actor, {"outcome": "succeeded", "ssid": saved["ssid"]})
            return result
        except WifiError as exc:
            self._audit("wifi_uplink_forget", actor,
                        {"outcome": "failed", "ssid": saved["ssid"], "error": exc.code})
            raise
        finally:
            self._op_lock.release()
            self._invalidate()

    # background jobs -----------------------------------------------------------
    def _start_job(self, op, label, work, actor, event, detail):
        job = {"id": uuidlib.uuid4().hex, "op": op, "ssid": label, "state": "running",
               "stage": "connecting", "stage_label": STAGE_LABELS["connecting"],
               "message": f"Connecting to {label}…", "error": None, "warning": None,
               "result": None, "started_at": time.time(), "finished_at": None}
        with self._state:
            self._jobs[job["id"]] = job
            self._active_job = job["id"]
            while len(self._jobs) > MAX_JOBS:
                self._jobs.popitem(last=False)
        log.info("Wi-Fi uplink %s to %r started by %s", op, label, getattr(actor, "username", "?"))
        thread = threading.Thread(target=self._run_job, args=(job, work, actor, event, detail),
                                  name=f"wifi-{op}", daemon=True)
        try:
            thread.start()
        except RuntimeError:
            self._finish(job, error=WifiError("helper_error"))
            self._op_lock.release()
            raise WifiError("helper_error")
        return dict(job)

    def _update(self, job, **changes):
        with self._state:
            job.update(changes)

    def _finish(self, job, error=None, **changes):
        changes.update(state="failed" if error else "succeeded", finished_at=time.time())
        if error:
            changes.update(error=error.as_dict(), message=error.message)
        with self._state:
            job.update(changes)
            if self._active_job == job["id"]:
                self._active_job = None

    def _run_job(self, job, work, actor, event, detail):
        error, changes = None, {}
        try:
            result = work()
            self._invalidate()
            self._update(job, stage="checking_internet",
                         stage_label=STAGE_LABELS["checking_internet"])
            reach = self.connectivity(force=True, usb=(result.get("interface"),
                                                       _first_ip(result.get("ipv4"))))
            usb = reach["usb"]
            ip = _first_ip(result.get("ipv4")) or "no address"
            message = f"Connected to {result.get('ssid') or job['ssid']} ({ip})."
            warning = None
            if usb["internet"]["state"] != "reachable":
                warning = "no_internet"
                message += " The Wi-Fi works, but the internet is not reachable through it."
            elif usb["robase"]["state"] != "reachable":
                warning = "robase_unreachable"
                message += " Internet works, but Robase could not be reached through it."
            else:
                message += " Internet and Robase are reachable."
            changes = {"message": message, "warning": warning,
                       "result": {k: result.get(k) for k in ("uuid", "ssid", "interface", "ipv4", "gateway")}}
            self._audit(event, actor, {**detail, "outcome": "succeeded", "warning": warning})
        except WifiError as exc:
            error, changes = exc, {"restored": exc.extra.get("restored")}
            self._audit(event, actor, {**detail, "outcome": "failed", "error": exc.code,
                                       "restored": exc.extra.get("restored")})
        except Exception as exc:  # never let a secret-bearing traceback reach the log
            log.error("Wi-Fi uplink %s job failed unexpectedly (%s)", job["op"], type(exc).__name__)
            error = WifiError("helper_error")
            self._audit(event, actor, {**detail, "outcome": "failed", "error": "helper_error"})
        finally:
            # Unlock before publishing the outcome, so whoever sees the job
            # finish can start the next operation straight away.
            self._invalidate()
            self._op_lock.release()
            self._finish(job, error=error, **changes)

    def job(self, job_id):
        with self._state:
            job = self._jobs.get(job_id) if isinstance(job_id, str) else None
            return dict(job) if job else None

    def wait_job(self, job_id, timeout=10.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            job = self.job(job_id)
            if job and job["state"] != "running":
                return job
            time.sleep(0.01)
        return self.job(job_id)

    # reachability --------------------------------------------------------------
    def _usb_path(self):
        view = self.status()
        conn = view.get("connection") if view.get("available") else None
        adapter = view.get("adapter") or {}
        if not adapter.get("present"):
            return None, None, "The USB Wi-Fi adapter was not found."
        if not conn or conn.get("state") != "connected" or conn.get("protected"):
            return adapter.get("interface"), None, "The USB adapter is not connected to a network."
        ip = _first_ip(conn.get("ipv4"))
        if not ip:
            return adapter.get("interface"), None, "The USB adapter has no IP address."
        return adapter.get("interface"), ip, None

    def connectivity(self, force=False, usb=None):
        if usb and usb[0] and usb[1]:
            iface, ip, why_not = usb[0], usb[1], None
        else:
            iface, ip, why_not = self._usb_path()
        key = (iface, ip)
        with self._state:
            age = self._clock() - self._reach_at
            if self._reach and self._reach_key == key and (
                    (not force and age < CONNECTIVITY_TTL) or age < CONNECTIVITY_MIN_INTERVAL):
                return dict(self._reach)
        if not self._probe_lock.acquire(timeout=PROBE_DEADLINE + 5):
            with self._state:
                return dict(self._reach) if self._reach else _pending_reach()
        try:
            view = self._run_checks(iface, ip, why_not)
            with self._state:
                self._reach, self._reach_at, self._reach_key = view, self._clock(), key
            return dict(view)
        finally:
            self._probe_lock.release()

    def _run_checks(self, iface, ip, why_not):
        pinned = iface if ip and iface and _IFACE.match(iface) else None
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=4, thread_name_prefix="wifi-probe")
        try:
            jobs = {"system_internet": pool.submit(self.prober.internet, None),
                    "system_robase": pool.submit(self.prober.robase, None)}
            if pinned:
                jobs["usb_internet"] = pool.submit(self.prober.internet, pinned)
                jobs["usb_robase"] = pool.submit(self.prober.robase, pinned)
            concurrent.futures.wait(jobs.values(), timeout=PROBE_DEADLINE)
            out = {}
            for name, future in jobs.items():
                if future.done() and not future.exception():
                    out[name] = future.result()
                else:
                    out[name] = _result("unreachable" if future.done() else "timeout",
                                        "The check did not finish in time." if not future.done()
                                        else "The check failed.")
        finally:
            pool.shutdown(wait=False, cancel_futures=True)
        skipped = _result("not_tested", why_not or "The USB adapter is not connected.")
        peer = out["system_internet"].get("peer") or out["system_robase"].get("peer")
        return {
            "checked_at": time.time(),
            "usb": {"interface": iface, "ipv4": ip,
                    "internet": out.get("usb_internet", skipped),
                    "robase": out.get("usb_robase", skipped)},
            "system": {"route_interface": self.prober.route_interface(peer) if peer else None,
                       "internet": out["system_internet"], "robase": out["system_robase"]},
            "robase_host": urlsplit(robase_health_url() or "").hostname,
        }

    # audit ---------------------------------------------------------------------
    @staticmethod
    def _audit(event, actor, detail):
        detail = {k: v for k, v in detail.items() if v is not None}
        try:
            with db.transaction() as conn:
                db.log_security_event(conn, event, staff_user_id=getattr(actor, "user_id", None),
                                      session_id=getattr(actor, "session_id", None), detail=detail)
        except Exception as exc:
            log.warning("Could not record %s in the security log (%s)", event, type(exc).__name__)


def _first_ip(addresses):
    for item in addresses or []:
        if isinstance(item, dict) and item.get("address"):
            return str(item["address"])
    return None


def _pending_reach():
    pending = _result("timeout", "A check is already running — try again shortly.")
    return {"checked_at": None, "usb": {"interface": None, "ipv4": None, "internet": pending,
                                        "robase": pending},
            "system": {"route_interface": None, "internet": pending, "robase": pending},
            "robase_host": None}


_service = None
_override = None
_service_lock = threading.Lock()


def get_service():
    global _service
    if _override is not None:
        return _override
    with _service_lock:
        if _service is None:
            _service = WifiUplink()
        return _service


def set_service_override(service):
    """Tests install a WifiUplink with fake backends here."""
    global _override
    _override = service
