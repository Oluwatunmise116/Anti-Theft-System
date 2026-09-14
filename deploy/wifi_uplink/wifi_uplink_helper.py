#!/usr/bin/python3 -I
"""
SecureDrive USB Wi-Fi uplink helper — the only privileged part of the
"Internet Wi-Fi" settings feature.

Installed root-owned as /usr/local/libexec/secure-drive/wifi-uplink-helper and
run by the web service through ONE sudoers rule that allows exactly that file
with no arguments. A single JSON request arrives on stdin and a single JSON
reply is written to stdout. Wi-Fi passwords arrive only on stdin and are
handed to NetworkManager over D-Bus; they never appear in argv, in logs, in
error text or in a reply.

Every rule that protects the Secure_Drive hotspot is enforced here, inside
the privileged boundary, not in the web application:

  * The adapter is chosen by hardware identity from a root-owned config
    (USB bus + vendor/product id + permanent MAC, optionally the USB port
    path) — never by interface name. No match or several matches: refuse.
    There is no fallback to any other adapter.
  * Profiles listed in `protected_connection_uuids`, and any access-point or
    shared (hotspot-style) profile, are never modified, deactivated or
    deleted, and the adapter is refused if one of them is running on it.
  * Only profiles this helper created may be activated or forgotten: the
    UUID must be in the root-owned registry, the name must carry the
    configured prefix, and the profile must be a Wi-Fi client profile bound
    to the adapter's MAC address.
  * A connection whose IPv4 subnet overlaps the hotspot's is rolled back.
  * On any failure a newly created profile is deleted and whatever was
    running on the USB adapter before is brought back.

The file is self-contained on purpose: it imports nothing from the
application directory (which the service account can write) and needs only
the system Python and python3-dbus.

Read-only diagnostics for installation (no root needed, never change
anything):
    /usr/bin/python3 wifi_uplink_helper.py --suggest-config
    /usr/bin/python3 wifi_uplink_helper.py --check CONFIG.json
"""
import contextlib
import errno
import fcntl
import ipaddress
import json
import os
import re
import stat
import sys
import syslog
import tempfile
import time
import uuid as uuidlib

CONFIG_PATH = "/etc/secure-drive/wifi-uplink.json"
STATE_DIR = "/var/lib/secure-drive"
REGISTRY_PATH = STATE_DIR + "/wifi-uplink-profiles.json"
LOCK_DIR = "/run/secure-drive"
LOCK_PATH = LOCK_DIR + "/wifi-uplink.lock"

DEFAULT_PREFIX = "SecureDrive Uplink"
#: The hotspot's subnet route uses NetworkManager's Wi-Fi default metric (600).
#: A larger metric on the uplink means that, even for the moment before an
#: overlapping subnet is rolled back, hotspot traffic keeps using the hotspot.
DEFAULT_ROUTE_METRIC = 700
AUTOCONNECT_PRIORITY = 10
ACTIVATION_TIMEOUT = 45.0
RESTORE_TIMEOUT = 20.0
DISCONNECT_TIMEOUT = 10.0
SCAN_TIMEOUT = 12.0
POLL_INTERVAL = 0.25
MAX_REQUEST_BYTES = 16 * 1024
MAX_CONFIG_BYTES = 64 * 1024
#: NetworkManager's default address for `ipv4.method shared` without an address.
NM_SHARED_DEFAULT = "10.42.0.0/24"

# NetworkManager D-Bus enums (https://networkmanager.dev/docs/api/latest/nm-dbus-types.html)
DEVICE_TYPE_WIFI = 2
MODE_AP = 3
DEV_UNMANAGED, DEV_UNAVAILABLE, DEV_DISCONNECTED = 10, 20, 30
DEV_ACTIVATED, DEV_FAILED = 100, 120
AC_ACTIVATED, AC_DEACTIVATED = 2, 4
AP_FLAG_PRIVACY = 0x1
SEC_KEY_PSK, SEC_KEY_8021X, SEC_KEY_SAE = 0x100, 0x200, 0x400
SEC_KEY_OWE, SEC_KEY_SUITE_B = 0x800, 0x2000

DEVICE_STATES = {
    0: "unknown", 10: "unmanaged", 20: "unavailable", 30: "disconnected",
    40: "preparing", 50: "configuring", 60: "waiting for password",
    70: "getting an IP address", 80: "checking the connection", 90: "starting",
    100: "connected", 110: "disconnecting", 120: "failed",
}

#: NMDeviceStateReason → the error code reported to the web application.
REASON_CODES = {
    5: "no_ip", 6: "no_ip", 15: "no_ip", 16: "no_ip", 17: "no_ip",
    7: "wrong_password", 8: "wrong_password", 11: "auth_timeout",
    36: "adapter_missing", 53: "network_not_found",
}

UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
SSID_HEX_RE = re.compile(r"^(?:[0-9a-f]{2}){1,32}$")
MAC_RE = re.compile(r"^[0-9a-f]{2}(?::[0-9a-f]{2}){5}$")
USB_ID_RE = re.compile(r"^[0-9a-f]{4}$")
ID_PATH_RE = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")
PREFIX_RE = re.compile(r"^[A-Za-z0-9 ._-]{1,40}$")
PSK_HEX_RE = re.compile(r"^[0-9a-fA-F]{64}$")

SECURITY_CHOICES = ("open", "wpa-psk", "sae")
OPS = ("status", "scan", "connect", "reconnect", "disconnect", "forget")


class HelperError(Exception):
    """A refusal or failure with a stable code. `detail` is always safe to show."""

    def __init__(self, code, detail="", **extra):
        super().__init__(code)
        self.code = code
        self.detail = detail
        self.extra = extra


class NMError(Exception):
    """A NetworkManager D-Bus error, reduced to its D-Bus error name."""

    def __init__(self, name):
        super().__init__(name)
        self.name = name or "unknown"

    @property
    def gone(self):
        return self.name.endswith(("UnknownObject", "UnknownMethod"))


def nm_error_code(exc):
    name = exc.name
    if name.endswith("PermissionDenied") or name.endswith("AccessDenied"):
        return "permission_denied"
    if name.endswith(("ServiceUnknown", "NoReply", "Disconnected", "NameHasNoOwner")):
        return "nm_unavailable"
    if exc.gone:
        return "adapter_missing"
    return "activation_failed"


# ── validation ───────────────────────────────────────────────────────────────

def parse_ssid_hex(value):
    if not isinstance(value, str) or not SSID_HEX_RE.match(value):
        raise HelperError("invalid_ssid", "The network name must be 1-32 bytes.")
    return bytes.fromhex(value)


def validate_uuid(value):
    if not isinstance(value, str) or not UUID_RE.match(value):
        raise HelperError("invalid_request", "Unknown saved network.")
    return value


def validate_password(security, password):
    """Returns the password to store (None for an open network)."""
    if security == "open":
        if password not in (None, ""):
            raise HelperError("invalid_password", "This network is open; leave the password empty.")
        return None
    if not isinstance(password, str) or not password:
        raise HelperError("invalid_password", "Enter the Wi-Fi password.")
    if security == "wpa-psk":
        if PSK_HEX_RE.match(password):
            return password
        if not 8 <= len(password) <= 63 or any(not 0x20 <= ord(c) <= 0x7E for c in password):
            raise HelperError("invalid_password",
                              "WPA passwords are 8-63 ordinary characters (letters, digits, "
                              "spaces and punctuation).")
        return password
    if security == "sae":
        if not 8 <= len(password) <= 128 or any(ord(c) < 0x20 or ord(c) == 0x7F for c in password):
            raise HelperError("invalid_password", "WPA3 passwords are 8-128 characters.")
        return password
    raise HelperError("unsupported_security", "That security type is not supported.")


def classify_security(flags, wpa_flags, rsn_flags):
    """(key, label, supported) for an access point's advertised security."""
    km = int(wpa_flags) | int(rsn_flags)
    if km & (SEC_KEY_8021X | SEC_KEY_SUITE_B):
        return "enterprise", "WPA Enterprise (802.1X)", False
    if km & SEC_KEY_SAE and km & SEC_KEY_PSK:
        return "wpa-psk", "WPA2/WPA3 Personal", True
    if km & SEC_KEY_SAE:
        return "sae", "WPA3 Personal", True
    if km & SEC_KEY_PSK:
        return "wpa-psk", "WPA2 Personal" if int(rsn_flags) else "WPA Personal", True
    if km & SEC_KEY_OWE:
        return "owe", "Enhanced Open (OWE)", False
    if int(flags) & AP_FLAG_PRIVACY:
        return "wep", "WEP (insecure)", False
    return "open", "Open (no password)", True


def display_ssid(ssid):
    """Readable SSID text. Identity always uses the raw bytes (hex), never this."""
    text = ssid.decode("utf-8", errors="replace")
    return "".join("�" if (ord(c) < 0x20 or 0x7F <= ord(c) < 0xA0) else c for c in text)


def band(frequency):
    frequency = int(frequency or 0)
    if 2400 <= frequency < 2500:
        return "2.4 GHz"
    if 4900 <= frequency < 5925:
        return "5 GHz"
    if frequency >= 5925:
        return "6 GHz"
    return ""


def mac_bytes(mac):
    return bytes(int(part, 16) for part in mac.split(":"))


def format_mac(raw):
    if isinstance(raw, (bytes, bytearray)) and len(raw) == 6:
        return ":".join(f"{b:02x}" for b in raw)
    return str(raw or "").lower()


# ── root-owned files ─────────────────────────────────────────────────────────

def _check_root_owned(path, st):
    if stat.S_ISLNK(st.st_mode) or not stat.S_ISREG(st.st_mode):
        raise HelperError("config_insecure", f"{path} must be a regular file.")
    if st.st_uid != 0 or st.st_mode & 0o022:
        raise HelperError("config_insecure",
                          f"{path} must be owned by root and not writable by others.")
    parent = os.stat(os.path.dirname(path) or "/")
    if parent.st_uid != 0 or parent.st_mode & 0o022:
        raise HelperError("config_insecure",
                          f"{os.path.dirname(path)} must be owned by root and not writable by others.")


class Config:
    def __init__(self, raw):
        if not isinstance(raw, dict):
            raise HelperError("config_invalid", "The config must be a JSON object.")
        adapter = raw.get("usb_adapter")
        if not isinstance(adapter, dict):
            raise HelperError("config_invalid", "usb_adapter is missing.")
        self.vendor_id = self._match(adapter, "vendor_id", USB_ID_RE, "usb_adapter.vendor_id")
        self.product_id = self._match(adapter, "product_id", USB_ID_RE, "usb_adapter.product_id")
        self.permanent_mac = self._match(adapter, "permanent_mac", MAC_RE,
                                         "usb_adapter.permanent_mac")
        id_path = adapter.get("id_path")
        if id_path in (None, ""):
            self.id_path = None
        elif isinstance(id_path, str) and ID_PATH_RE.match(id_path):
            self.id_path = id_path
        else:
            raise HelperError("config_invalid", "usb_adapter.id_path is not valid.")

        protected = raw.get("protected_connection_uuids")
        if (not isinstance(protected, list) or not protected
                or not all(isinstance(u, str) and UUID_RE.match(u.lower()) for u in protected)):
            raise HelperError("config_invalid",
                              "protected_connection_uuids must list the hotspot profile UUID.")
        self.protected = frozenset(u.lower() for u in protected)

        self.reserved_subnets = []
        for item in raw.get("reserved_subnets") or []:
            try:
                self.reserved_subnets.append(ipaddress.ip_network(str(item), strict=False))
            except ValueError:
                raise HelperError("config_invalid", "reserved_subnets has an invalid entry.")

        prefix = raw.get("profile_prefix", DEFAULT_PREFIX)
        if not isinstance(prefix, str) or not PREFIX_RE.match(prefix):
            raise HelperError("config_invalid", "profile_prefix is not valid.")
        self.prefix = prefix

        metric = raw.get("route_metric", DEFAULT_ROUTE_METRIC)
        if isinstance(metric, bool) or not isinstance(metric, int) or not 1 <= metric <= 9999:
            raise HelperError("config_invalid", "route_metric must be 1-9999.")
        self.route_metric = metric

    @staticmethod
    def _match(section, key, pattern, label):
        value = section.get(key)
        value = value.lower() if isinstance(value, str) else value
        if not isinstance(value, str) or not pattern.match(value):
            raise HelperError("config_invalid", f"{label} is missing or still a placeholder.")
        return value

    @classmethod
    def load(cls, path=CONFIG_PATH, require_root=True):
        try:
            st = os.lstat(path)
        except FileNotFoundError:
            raise HelperError("config_missing", f"{path} does not exist.")
        if require_root:
            _check_root_owned(path, st)
        with open(path, "rb") as handle:
            data = handle.read(MAX_CONFIG_BYTES + 1)
        if len(data) > MAX_CONFIG_BYTES:
            raise HelperError("config_invalid", "The config file is too large.")
        try:
            return cls(json.loads(data.decode("utf-8")))
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise HelperError("config_invalid", "The config file is not valid JSON.")


class Registry:
    """UUIDs of the profiles this helper created. Root-only, written atomically."""

    def __init__(self, path=REGISTRY_PATH, secure=True):
        self.path = path
        self.secure = secure

    def load(self):
        try:
            if self.secure:
                _check_root_owned(self.path, os.lstat(self.path))
            with open(self.path, "rb") as handle:
                data = json.loads(handle.read(MAX_CONFIG_BYTES).decode("utf-8"))
        except FileNotFoundError:
            return {}
        except HelperError:
            raise
        except (OSError, ValueError):
            raise HelperError("registry_invalid", "The uplink profile registry is unreadable.")
        profiles = data.get("profiles") if isinstance(data, dict) else None
        if not isinstance(profiles, dict):
            return {}
        return {k: v for k, v in profiles.items()
                if isinstance(k, str) and UUID_RE.match(k) and isinstance(v, dict)}

    def save(self, profiles):
        directory = os.path.dirname(self.path)
        os.makedirs(directory, mode=0o700, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=".profiles-", dir=directory)
        try:
            with os.fdopen(fd, "w") as handle:
                json.dump({"version": 1, "profiles": profiles}, handle, indent=2, sort_keys=True)
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(tmp, 0o600)
            os.replace(tmp, self.path)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp)
            raise

    def put(self, uuid, **entry):
        profiles = self.load()
        profiles[uuid] = {**profiles.get(uuid, {}), **entry}
        self.save(profiles)

    def remove(self, uuid):
        profiles = self.load()
        if profiles.pop(uuid, None) is not None:
            self.save(profiles)


@contextlib.contextmanager
def operation_lock(path=LOCK_PATH):
    """One network-changing operation at a time, across processes."""
    os.makedirs(os.path.dirname(path), mode=0o700, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno in (errno.EAGAIN, errno.EACCES, errno.EWOULDBLOCK):
                raise HelperError("busy", "Another Wi-Fi operation is still running.")
            raise
        yield
    finally:
        os.close(fd)


# ── hardware identity ────────────────────────────────────────────────────────

def _read_text(path):
    try:
        with open(path) as handle:
            return handle.read(256).strip()
    except OSError:
        return None


class Sysfs:
    """Bus and USB ids for a NetworkManager device, read from sysfs."""

    def __init__(self, root="/sys"):
        self.root = root.rstrip("/")

    def identity(self, udi):
        devices = self.root + "/devices/"
        if not isinstance(udi, str) or not udi.startswith(devices):
            return {"bus": None}
        net_dir = os.path.realpath(udi)
        if not net_dir.startswith(devices):
            return {"bus": None}
        device_dir = os.path.realpath(os.path.join(net_dir, "device"))
        bus = os.path.basename(os.path.realpath(os.path.join(device_dir, "subsystem")))
        ident = {"bus": bus}
        if bus == "usb":
            # net/wlanX/device is the USB *interface* (3-2:1.0); the ids live
            # on its parent USB device (3-2).
            current = device_dir
            for _ in range(3):
                vendor = _read_text(os.path.join(current, "idVendor"))
                product = _read_text(os.path.join(current, "idProduct"))
                if vendor and product:
                    ident.update(vendor_id=vendor.lower(), product_id=product.lower())
                    break
                current = os.path.dirname(current)
        return ident


# ── NetworkManager over D-Bus ────────────────────────────────────────────────

class NMDBus:
    """The few NetworkManager D-Bus calls the helper needs. Secrets are only
    ever sent inside settings dictionaries, never read back."""

    NM = "org.freedesktop.NetworkManager"
    NM_PATH = "/org/freedesktop/NetworkManager"
    SETTINGS_PATH = "/org/freedesktop/NetworkManager/Settings"
    _INT_TYPES = {
        ("connection", "autoconnect-priority"): "Int32",
        ("connection", "autoconnect-retries"): "Int32",
        ("ipv4", "route-metric"): "Int64",
        ("ipv6", "route-metric"): "Int64",
        ("802-11-wireless-security", "psk-flags"): "UInt32",
    }

    def __init__(self):
        import dbus  # system python3-dbus
        self.dbus = dbus
        try:
            self.bus = dbus.SystemBus()
            self.manager = dbus.Interface(self._obj(self.NM_PATH), self.NM)
            self.settings = dbus.Interface(self._obj(self.SETTINGS_PATH), self.NM + ".Settings")
            self.version = str(self._prop(self.NM_PATH, self.NM, "Version"))
        except dbus.exceptions.DBusException as exc:
            raise NMError(exc.get_dbus_name()) from None

    # plumbing
    def _obj(self, path):
        return self.bus.get_object(self.NM, path)

    def _call(self, fn, *args):
        try:
            return fn(*args)
        except self.dbus.exceptions.DBusException as exc:
            raise NMError(exc.get_dbus_name()) from None

    def _props(self, path, iface):
        props = self.dbus.Interface(self._obj(path), "org.freedesktop.DBus.Properties")
        return self._py(self._call(props.GetAll, iface))

    def _prop(self, path, iface, name):
        props = self.dbus.Interface(self._obj(path), "org.freedesktop.DBus.Properties")
        return self._py(self._call(props.Get, iface, name))

    def _py(self, value):
        d = self.dbus
        if isinstance(value, d.Array) and value.signature == "y":
            return bytes(bytearray(int(b) for b in value))
        if isinstance(value, (d.ByteArray, bytes)):
            return bytes(value)
        if isinstance(value, d.Boolean):
            return bool(value)
        if isinstance(value, dict):
            return {str(k): self._py(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._py(v) for v in value]
        if isinstance(value, str):
            return str(value)
        if isinstance(value, int):
            return int(value)
        if isinstance(value, float):
            return float(value)
        return value

    def _to_dbus(self, settings):
        d = self.dbus
        out = d.Dictionary({}, signature="sa{sv}")
        for group, values in settings.items():
            inner = d.Dictionary({}, signature="sv")
            for key, value in values.items():
                if isinstance(value, bool):
                    inner[key] = d.Boolean(value)
                elif isinstance(value, bytes):
                    inner[key] = d.ByteArray(value)
                elif isinstance(value, int):
                    inner[key] = getattr(d, self._INT_TYPES.get((group, key), "Int32"))(value)
                elif isinstance(value, str):
                    inner[key] = d.String(value)
                else:
                    raise TypeError(f"unsupported setting type for {group}.{key}")
            out[group] = inner
        return out

    @staticmethod
    def _path_or_none(path):
        path = str(path or "/")
        return None if path == "/" else path

    # devices
    def devices(self):
        return [self.device(str(p)) for p in self._call(self.manager.GetAllDevices)]

    def device(self, path):
        p = self._props(path, self.NM + ".Device")
        item = {
            "path": path, "interface": p.get("Interface", ""), "udi": p.get("Udi", ""),
            "id_path": p.get("Path", ""), "driver": p.get("Driver", ""),
            "device_type": int(p.get("DeviceType", 0)), "state": int(p.get("State", 0)),
            "reason": int((p.get("StateReason") or [0, 0])[1]), "managed": bool(p.get("Managed")),
            "active_connection": self._path_or_none(p.get("ActiveConnection")),
            "ip4config": self._path_or_none(p.get("Ip4Config")),
            "perm_hw": "", "mode": 0, "active_ap": None, "last_scan": -1,
        }
        if item["device_type"] == DEVICE_TYPE_WIFI:
            w = self._props(path, self.NM + ".Device.Wireless")
            item.update(perm_hw=str(w.get("PermHwAddress", "")).lower(), mode=int(w.get("Mode", 0)),
                        active_ap=self._path_or_none(w.get("ActiveAccessPoint")),
                        last_scan=int(w.get("LastScan", -1)))
        return item

    def request_scan(self, device_path):
        wireless = self.dbus.Interface(self._obj(device_path), self.NM + ".Device.Wireless")
        self._call(wireless.RequestScan, self.dbus.Dictionary({}, signature="sv"))

    def access_points(self, device_path):
        wireless = self.dbus.Interface(self._obj(device_path), self.NM + ".Device.Wireless")
        aps = []
        for path in self._call(wireless.GetAllAccessPoints):
            try:
                p = self._props(str(path), self.NM + ".AccessPoint")
            except NMError as exc:
                if exc.gone:
                    continue
                raise
            aps.append({
                "path": str(path), "ssid": p.get("Ssid", b"") or b"", "bssid": p.get("HwAddress", ""),
                "strength": int(p.get("Strength", 0)), "flags": int(p.get("Flags", 0)),
                "wpa": int(p.get("WpaFlags", 0)), "rsn": int(p.get("RsnFlags", 0)),
                "frequency": int(p.get("Frequency", 0)), "mode": int(p.get("Mode", 0)),
            })
        return aps

    def access_point_strength(self, ap_path):
        try:
            return int(self._prop(ap_path, self.NM + ".AccessPoint", "Strength"))
        except NMError:
            return None

    # connection profiles (GetSettings never returns secrets)
    def connections(self):
        out = []
        for path in self._call(self.settings.ListConnections):
            obj = self.dbus.Interface(self._obj(str(path)), self.NM + ".Settings.Connection")
            try:
                out.append({"path": str(path), "settings": self._py(self._call(obj.GetSettings))})
            except NMError as exc:
                if not exc.gone:
                    raise
        return out

    def add_connection(self, settings):
        flags = 0x1 | 0x20  # TO_DISK | BLOCK_AUTOCONNECT
        path, _ = self._call(self.settings.AddConnection2, self._to_dbus(settings),
                             self.dbus.UInt32(flags), self.dbus.Dictionary({}, signature="sv"))
        return str(path)

    def update_connection(self, path, settings):
        obj = self.dbus.Interface(self._obj(path), self.NM + ".Settings.Connection")
        self._call(obj.Update2, self._to_dbus(settings), self.dbus.UInt32(0x1),
                   self.dbus.Dictionary({}, signature="sv"))

    def delete_connection(self, path):
        obj = self.dbus.Interface(self._obj(path), self.NM + ".Settings.Connection")
        self._call(obj.Delete)

    # activation
    def activate(self, connection_path, device_path):
        d = self.dbus
        return str(self._call(self.manager.ActivateConnection, d.ObjectPath(connection_path),
                              d.ObjectPath(device_path), d.ObjectPath("/")))

    def active_connection(self, path):
        try:
            p = self._props(path, self.NM + ".Connection.Active")
        except NMError as exc:
            if exc.gone:
                return None
            raise
        return {"path": path, "state": int(p.get("State", 0)), "uuid": str(p.get("Uuid", "")),
                "id": str(p.get("Id", "")), "connection": str(p.get("Connection", "/")),
                "devices": [str(x) for x in p.get("Devices", [])]}

    def disconnect_device(self, device_path):
        obj = self.dbus.Interface(self._obj(device_path), self.NM + ".Device")
        self._call(obj.Disconnect)

    def ip4(self, path):
        try:
            p = self._props(path, self.NM + ".IP4Config")
        except NMError as exc:
            if exc.gone:
                return {"addresses": [], "gateway": "", "dns": []}
            raise
        return {
            "addresses": [{"address": str(a.get("address")), "prefix": int(a.get("prefix", 0))}
                          for a in p.get("AddressData", [])],
            "gateway": str(p.get("Gateway", "") or ""),
            "dns": [str(n.get("address")) for n in p.get("NameserverData", []) if n.get("address")],
        }


# ── the policy ───────────────────────────────────────────────────────────────

class UplinkHelper:
    def __init__(self, nm, config, registry, sysfs=None, clock=time.monotonic, sleep=time.sleep,
                 lock=operation_lock):
        self.nm = nm
        self.config = config
        self.registry = registry
        self.sysfs = sysfs or Sysfs()
        self.clock = clock
        self.sleep = sleep
        self.lock = lock

    # identity ----------------------------------------------------------------
    def find_adapter(self):
        """The one NetworkManager device matching the configured USB identity."""
        matches = []
        for dev in self.nm.devices():
            if dev["device_type"] != DEVICE_TYPE_WIFI:
                continue
            ident = self.sysfs.identity(dev["udi"])
            if ident.get("bus") != "usb":
                continue
            if (ident.get("vendor_id"), ident.get("product_id")) != (self.config.vendor_id,
                                                                     self.config.product_id):
                continue
            if dev["perm_hw"] != self.config.permanent_mac:
                continue
            if self.config.id_path and dev["id_path"] != self.config.id_path:
                continue
            matches.append(dev)
        if not matches:
            raise HelperError("adapter_missing", "The configured USB Wi-Fi adapter was not found.")
        if len(matches) > 1:
            raise HelperError("adapter_ambiguous",
                              "More than one adapter matches the configured USB identity.")
        return matches[0]

    def _usable(self, dev, conns):
        """Refuse an adapter that must not be changed from here."""
        if dev["mode"] == MODE_AP:
            raise HelperError("adapter_is_hotspot", "The USB adapter is running an access point.")
        active = self._active_on(dev, conns)
        if active and active["protected"]:
            raise HelperError("adapter_is_hotspot", "The hotspot profile is running on the USB adapter.")
        if not dev["managed"] or dev["state"] == DEV_UNMANAGED:
            raise HelperError("adapter_unmanaged", "NetworkManager does not manage the USB adapter.")
        if dev["state"] == DEV_UNAVAILABLE:
            raise HelperError("adapter_unavailable", "The USB adapter is not ready (unavailable).")
        return active

    # profiles ----------------------------------------------------------------
    def is_protected(self, uuid, settings):
        if uuid.lower() in self.config.protected:
            return True
        wifi = settings.get("802-11-wireless", {})
        if wifi.get("mode") in ("ap", "adhoc", "mesh"):
            return True
        return any(settings.get(f, {}).get("method") == "shared" for f in ("ipv4", "ipv6"))

    def _binding_ok(self, settings):
        conn = settings.get("connection", {})
        wifi = settings.get("802-11-wireless", {})
        return (conn.get("type") == "802-11-wireless"
                and wifi.get("mode", "infrastructure") in ("infrastructure", "")
                and str(conn.get("id", "")).startswith(self.config.prefix)
                and format_mac(wifi.get("mac-address")) == self.config.permanent_mac)

    @staticmethod
    def _find(conns, uuid):
        for c in conns:
            if c["settings"].get("connection", {}).get("uuid") == uuid:
                return c
        return None

    def owned_profile(self, uuid, conns, registry=None, allow_pending=False):
        """The profile, if and only if this helper may activate or delete it."""
        registry = self.registry.load() if registry is None else registry
        conn = self._find(conns, uuid)
        if conn and self.is_protected(uuid, conn["settings"]):
            raise HelperError("protected_profile", "That profile belongs to the hotspot.")
        entry = registry.get(uuid)
        if entry is None or (entry.get("state") != "ready" and not allow_pending):
            raise HelperError("not_permitted", "That profile is not managed by SecureDrive.")
        if conn is None:
            raise HelperError("profile_missing", "That saved network no longer exists.")
        if not self._binding_ok(conn["settings"]):
            raise HelperError("not_permitted", "That profile is not bound to the USB adapter.")
        return conn

    def _owned_list(self, conns):
        registry = self.registry.load()
        owned = []
        for uuid, entry in registry.items():
            if entry.get("state") != "ready":
                continue
            try:
                owned.append(self.owned_profile(uuid, conns, registry))
            except HelperError:
                continue
        return owned

    def _active_on(self, dev, conns):
        if not dev["active_connection"]:
            return None
        ac = self.nm.active_connection(dev["active_connection"])
        if ac is None:
            return None
        conn = self._find(conns, ac["uuid"])
        settings = conn["settings"] if conn else {}
        return {**ac, "settings": settings, "conn_path": conn["path"] if conn else None,
                "protected": self.is_protected(ac["uuid"], settings)}

    def _cleanup_pending(self, conns):
        """Delete profiles left half-made by an interrupted connect."""
        registry = self.registry.load()
        for uuid, entry in list(registry.items()):
            if entry.get("state") == "ready":
                continue
            conn = self._find(conns, uuid)
            if conn and not self.is_protected(uuid, conn["settings"]) and self._binding_ok(conn["settings"]):
                with contextlib.suppress(NMError):
                    self.nm.delete_connection(conn["path"])
            self.registry.remove(uuid)

    # addresses ---------------------------------------------------------------
    def hotspot_subnets(self, conns, devices):
        nets = set(self.config.reserved_subnets)
        protected_uuids = set()
        for c in conns:
            s = c["settings"]
            uuid = s.get("connection", {}).get("uuid", "")
            if not self.is_protected(uuid, s):
                continue
            protected_uuids.add(uuid)
            ipv4 = s.get("ipv4", {})
            added = False
            for a in ipv4.get("address-data", []) or []:
                with contextlib.suppress(ValueError, TypeError):
                    nets.add(ipaddress.ip_network(f"{a['address']}/{a['prefix']}", strict=False))
                    added = True
            if ipv4.get("method") == "shared" and not added:
                nets.add(ipaddress.ip_network(NM_SHARED_DEFAULT))
        for dev in devices:
            if not dev["active_connection"] or not dev["ip4config"]:
                continue
            ac = self.nm.active_connection(dev["active_connection"])
            if ac and ac["uuid"] in protected_uuids:
                for a in self.nm.ip4(dev["ip4config"])["addresses"]:
                    with contextlib.suppress(ValueError):
                        nets.add(ipaddress.ip_network(f"{a['address']}/{a['prefix']}", strict=False))
        return nets

    @staticmethod
    def overlapping(addresses, subnets):
        for a in addresses:
            try:
                net = ipaddress.ip_network(f"{a['address']}/{a['prefix']}", strict=False)
            except ValueError:
                continue
            for other in subnets:
                if net.version == other.version and net.overlaps(other):
                    return str(other)
        return None

    # waiting -----------------------------------------------------------------
    def _wait_activated(self, ac_path, dev_path, timeout):
        deadline = self.clock() + timeout
        reason = 0
        while True:
            try:
                dev = self.nm.device(dev_path)
            except NMError as exc:
                if exc.gone:
                    raise HelperError("adapter_missing", "The USB adapter was removed.")
                raise
            if dev["reason"] not in (0, 60):  # 60 = new activation, not a failure
                reason = dev["reason"]
            ac = self.nm.active_connection(ac_path)
            if ac is not None and ac["state"] == AC_ACTIVATED:
                return
            ours = dev["active_connection"] == ac_path
            if ac is None or ac["state"] == AC_DEACTIVATED or (ours and dev["state"] == DEV_FAILED):
                code = REASON_CODES.get(reason, "activation_failed")
                raise HelperError(code, f"NetworkManager reason {reason}.", reason=reason)
            if self.clock() >= deadline:
                raise HelperError("timeout", "The connection did not finish in time.")
            self.sleep(POLL_INTERVAL)

    def _restore(self, previous, dev_path):
        """Bring back what the USB adapter was running before. USB only."""
        if not previous or previous["protected"] or not previous["conn_path"]:
            return None
        try:
            ac = self.nm.activate(previous["conn_path"], dev_path)
            self._wait_activated(ac, dev_path, RESTORE_TIMEOUT)
            return True
        except (HelperError, NMError):
            return False

    def _activation_result(self, dev_path, conns):
        dev = self.nm.device(dev_path)
        ip4 = self.nm.ip4(dev["ip4config"]) if dev["ip4config"] else {"addresses": [], "gateway": "", "dns": []}
        if not ip4["addresses"]:
            raise HelperError("no_ip", "No IPv4 address was assigned.")
        conflict = self.overlapping(ip4["addresses"], self.hotspot_subnets(conns, self.nm.devices()))
        if conflict:
            raise HelperError("subnet_conflict",
                              f"The network's addresses overlap the hotspot subnet {conflict}.",
                              hotspot_subnet=conflict)
        return dev, ip4

    # operations --------------------------------------------------------------
    def op_status(self):
        conns = self.nm.connections()
        devices = self.nm.devices()
        result = {"adapter": None, "connection": None, "saved": [],
                  "hotspot": self._hotspot_view(conns, devices, usb_path=None),
                  "unmanaged_wifi_profiles": 0,
                  "nm_version": getattr(self.nm, "version", "")}
        try:
            dev = self.find_adapter()
        except HelperError as exc:
            result["adapter"] = {"present": False, "problem": exc.code, "detail": exc.detail}
            return result
        result["hotspot"] = self._hotspot_view(conns, devices, usb_path=dev["path"])
        problem = None
        try:
            active = self._usable(dev, conns)
        except HelperError as exc:
            problem = exc.code
            active = self._active_on(dev, conns)
        result["adapter"] = {
            "present": True, "interface": dev["interface"], "driver": dev["driver"],
            "mac": dev["perm_hw"], "usb_id": f"{self.config.vendor_id}:{self.config.product_id}",
            "port_path": dev["id_path"], "managed": dev["managed"], "state_code": dev["state"],
            "state": DEVICE_STATES.get(dev["state"], "unknown"), "reason_code": dev["reason"],
            "problem": problem,
        }
        owned = {c["settings"]["connection"]["uuid"]: c for c in self._owned_list(conns)}
        if active:
            wifi = active["settings"].get("802-11-wireless", {})
            ssid = wifi.get("ssid", b"") or b""
            ip4 = self.nm.ip4(dev["ip4config"]) if dev["ip4config"] else {"addresses": [], "gateway": "", "dns": []}
            subnets = self.hotspot_subnets(conns, devices)
            result["connection"] = {
                "uuid": active["uuid"], "name": active["id"], "ssid": display_ssid(ssid),
                "ssid_hex": ssid.hex(), "managed_here": active["uuid"] in owned,
                "protected": active["protected"],
                "state": "connected" if active["state"] == AC_ACTIVATED else "connecting",
                "ipv4": ip4["addresses"], "gateway": ip4["gateway"], "dns": ip4["dns"],
                "signal": self.nm.access_point_strength(dev["active_ap"]) if dev["active_ap"] else None,
                "subnet_conflict": self.overlapping(ip4["addresses"], subnets),
            }
        for uuid, c in owned.items():
            s = c["settings"]
            wifi = s.get("802-11-wireless", {})
            ssid = wifi.get("ssid", b"") or b""
            result["saved"].append({
                "uuid": uuid, "ssid": display_ssid(ssid), "ssid_hex": ssid.hex(),
                "hidden": bool(wifi.get("hidden", False)),
                "security": s.get("802-11-wireless-security", {}).get("key-mgmt", "open"),
                "autoconnect": bool(s.get("connection", {}).get("autoconnect", True)),
                "active": bool(active and active["uuid"] == uuid),
            })
        result["saved"].sort(key=lambda item: item["ssid"].lower())
        result["unmanaged_wifi_profiles"] = sum(
            1 for c in conns if self._could_autojoin_usb(c["settings"], owned, dev["interface"]))
        return result

    def _could_autojoin_usb(self, settings, owned, usb_interface):
        """Another client profile NetworkManager may start on the USB adapter by itself."""
        conn = settings.get("connection", {})
        wifi = settings.get("802-11-wireless", {})
        uuid = conn.get("uuid", "")
        if (conn.get("type") != "802-11-wireless" or uuid in owned
                or self.is_protected(uuid, settings) or not conn.get("autoconnect", True)):
            return False
        mac = wifi.get("mac-address")
        if mac and format_mac(mac) != self.config.permanent_mac:
            return False
        interface = conn.get("interface-name")
        return not interface or interface == usb_interface

    def _hotspot_view(self, conns, devices, usb_path):
        by_uuid = {}
        for dev in devices:
            if dev["active_connection"]:
                ac = self.nm.active_connection(dev["active_connection"])
                if ac:
                    by_uuid[ac["uuid"]] = dev
        view = []
        for c in conns:
            s = c["settings"]
            uuid = s.get("connection", {}).get("uuid", "")
            if uuid.lower() not in self.config.protected:
                continue
            dev = by_uuid.get(uuid)
            view.append({"name": s["connection"].get("id", ""),
                         "ssid": display_ssid(s.get("802-11-wireless", {}).get("ssid", b"") or b""),
                         "active": dev is not None, "interface": dev["interface"] if dev else "",
                         "on_usb_adapter": bool(dev and usb_path and dev["path"] == usb_path)})
        return view

    def op_scan(self, request=None):
        conns = self.nm.connections()
        dev = self.find_adapter()
        self._usable(dev, conns)
        before = dev["last_scan"]
        fresh = True
        try:
            self.nm.request_scan(dev["path"])
        except NMError as exc:
            if nm_error_code(exc) != "activation_failed":
                raise
            fresh = False  # NetworkManager throttles scans; use what it already has
        if fresh:
            deadline = self.clock() + SCAN_TIMEOUT
            while self.nm.device(dev["path"])["last_scan"] == before and self.clock() < deadline:
                self.sleep(POLL_INTERVAL)
            dev = self.nm.device(dev["path"])
            fresh = dev["last_scan"] != before
        saved = {c["settings"]["802-11-wireless"].get("ssid", b"").hex()
                 for c in self._owned_list(conns)}
        networks, hidden = {}, 0
        for ap in self.nm.access_points(dev["path"]):
            if ap["mode"] not in (0, 2):  # infrastructure (or unknown) only
                continue
            if not ap["ssid"] or not ap["ssid"].strip(b"\x00"):
                hidden += 1
                continue
            key = ap["ssid"].hex()
            sec, label, supported = classify_security(ap["flags"], ap["wpa"], ap["rsn"])
            current = networks.get(key)
            if current is None or ap["strength"] > current["signal"]:
                networks[key] = {
                    "ssid": display_ssid(ap["ssid"]), "ssid_hex": key, "signal": ap["strength"],
                    "security": sec, "security_label": label, "supported": supported,
                    "band": band(ap["frequency"]), "in_use": False, "saved": key in saved,
                    "access_points": (current or {}).get("access_points", 0),
                }
            networks[key]["access_points"] += 1
            if ap["path"] == dev["active_ap"]:
                networks[key]["in_use"] = True
        ordered = sorted(networks.values(), key=lambda n: (not n["in_use"], -n["signal"], n["ssid"]))
        return {"networks": ordered, "hidden_networks": hidden, "fresh": fresh}

    def build_settings(self, uuid, ssid, security, password, hidden):
        settings = {
            "connection": {"id": f"{self.config.prefix}: {display_ssid(ssid)}", "uuid": uuid,
                           "type": "802-11-wireless", "autoconnect": False,
                           "autoconnect-priority": AUTOCONNECT_PRIORITY,
                           "autoconnect-retries": 0},  # 0 = keep retrying after signal loss
            "802-11-wireless": {"ssid": ssid, "mode": "infrastructure",
                                "mac-address": mac_bytes(self.config.permanent_mac),
                                "hidden": bool(hidden)},
            "ipv4": {"method": "auto", "may-fail": False, "route-metric": self.config.route_metric},
            "ipv6": {"method": "auto", "route-metric": self.config.route_metric},
        }
        if security != "open":
            settings["802-11-wireless-security"] = {"key-mgmt": security, "psk": password,
                                                    "psk-flags": 0}
        return settings

    def op_connect(self, request):
        ssid = parse_ssid_hex(request.get("ssid_hex"))
        hidden = request.get("hidden", False)
        if not isinstance(hidden, bool):
            raise HelperError("invalid_request", "hidden must be true or false.")
        password = request.get("password")

        conns = self.nm.connections()
        dev = self.find_adapter()
        previous = self._usable(dev, conns)
        self._cleanup_pending(conns)
        conns = self.nm.connections()

        visible = [ap for ap in self.nm.access_points(dev["path"]) if ap["ssid"] == ssid]
        if visible:
            best = max(visible, key=lambda ap: ap["strength"])
            security, label, supported = classify_security(best["flags"], best["wpa"], best["rsn"])
            if not supported:
                raise HelperError("unsupported_security", f"{label} networks are not supported.")
        elif hidden:
            security = request.get("security")
            if security not in SECURITY_CHOICES:
                raise HelperError("unsupported_security", "Choose Open, WPA/WPA2 or WPA3 Personal.")
        else:
            raise HelperError("network_not_found", "That network is not in range.")
        password = validate_password(security, password)

        new_uuid = str(uuidlib.uuid4())
        self.registry.put(new_uuid, state="pending", ssid_hex=ssid.hex(), created=int(time.time()))
        settings = self.build_settings(new_uuid, ssid, security, password, hidden)
        password = None
        try:
            try:
                path = self.nm.add_connection(settings)
            except NMError as exc:
                self.registry.remove(new_uuid)
                code = nm_error_code(exc)
                raise HelperError(code, "NetworkManager refused the new profile.")
            try:
                ac = self.nm.activate(path, dev["path"])
                self._wait_activated(ac, dev["path"], ACTIVATION_TIMEOUT)
                dev_now, ip4 = self._activation_result(dev["path"], self.nm.connections())
                settings["connection"]["autoconnect"] = True
                self.nm.update_connection(path, settings)
            except (HelperError, NMError) as exc:
                failure = exc if isinstance(exc, HelperError) else HelperError(nm_error_code(exc))
                with contextlib.suppress(NMError):
                    self.nm.delete_connection(path)
                self.registry.remove(new_uuid)
                failure.extra["restored"] = self._restore(previous, dev["path"])
                raise failure
        finally:
            settings = None  # drop the last reference to the password

        self.registry.put(new_uuid, state="ready")
        # A new password for a network that was already saved replaces the old profile.
        for other in self._owned_list(self.nm.connections()):
            s = other["settings"]
            if s["connection"]["uuid"] != new_uuid and s["802-11-wireless"].get("ssid") == ssid:
                with contextlib.suppress(NMError):
                    self.nm.delete_connection(other["path"])
                self.registry.remove(s["connection"]["uuid"])
        return {"uuid": new_uuid, "ssid": display_ssid(ssid), "interface": dev_now["interface"],
                "ipv4": ip4["addresses"], "gateway": ip4["gateway"], "dns": ip4["dns"]}

    def op_reconnect(self, request):
        uuid = validate_uuid(request.get("uuid"))
        conns = self.nm.connections()
        dev = self.find_adapter()
        previous = self._usable(dev, conns)
        self._cleanup_pending(conns)
        conns = self.nm.connections()
        conn = self.owned_profile(uuid, conns)
        try:
            ac = self.nm.activate(conn["path"], dev["path"])
            self._wait_activated(ac, dev["path"], ACTIVATION_TIMEOUT)
            dev_now, ip4 = self._activation_result(dev["path"], self.nm.connections())
        except (HelperError, NMError) as exc:
            failure = exc if isinstance(exc, HelperError) else HelperError(nm_error_code(exc))
            if failure.code == "subnet_conflict":
                with contextlib.suppress(NMError):
                    self.nm.disconnect_device(dev["path"])
            if previous and previous["uuid"] != uuid:
                failure.extra["restored"] = self._restore(previous, dev["path"])
            raise failure
        ssid = conn["settings"]["802-11-wireless"].get("ssid", b"")
        return {"uuid": uuid, "ssid": display_ssid(ssid), "interface": dev_now["interface"],
                "ipv4": ip4["addresses"], "gateway": ip4["gateway"], "dns": ip4["dns"]}

    def op_disconnect(self, request):
        conns = self.nm.connections()
        dev = self.find_adapter()
        active = self._active_on(dev, conns)
        if dev["mode"] == MODE_AP or (active and active["protected"]):
            raise HelperError("adapter_is_hotspot", "The hotspot is running on that adapter.")
        if not active:
            return {"disconnected": True, "was_connected": False}
        # Device.Disconnect keeps the adapter down until an explicit connect,
        # a reboot, or the adapter being plugged in again.
        self.nm.disconnect_device(dev["path"])
        deadline = self.clock() + DISCONNECT_TIMEOUT
        while self.nm.device(dev["path"])["state"] not in (DEV_DISCONNECTED, DEV_UNAVAILABLE):
            if self.clock() >= deadline:
                raise HelperError("timeout", "The adapter did not disconnect in time.")
            self.sleep(POLL_INTERVAL)
        return {"disconnected": True, "was_connected": True, "ssid": display_ssid(
            active["settings"].get("802-11-wireless", {}).get("ssid", b"") or b"")}

    def op_forget(self, request):
        uuid = validate_uuid(request.get("uuid"))
        conns = self.nm.connections()
        self._cleanup_pending(conns)
        conns = self.nm.connections()
        try:
            conn = self.owned_profile(uuid, conns)
        except HelperError as exc:
            if exc.code == "profile_missing":
                self.registry.remove(uuid)
            raise
        self.nm.delete_connection(conn["path"])
        self.registry.remove(uuid)
        return {"forgotten": True, "ssid": display_ssid(
            conn["settings"]["802-11-wireless"].get("ssid", b"") or b"")}

    def run(self, op, request):
        if op not in OPS:
            raise HelperError("invalid_request", "Unknown operation.")
        if op == "status":
            return self.op_status()  # read-only: never waits for the lock
        with self.lock():
            return getattr(self, "op_" + op)(request)


# ── entry points ─────────────────────────────────────────────────────────────

def _reply(stream, payload):
    stream.write(json.dumps(payload, sort_keys=True))
    stream.write("\n")
    stream.flush()


def _audit(message):
    with contextlib.suppress(Exception):
        syslog.openlog("secure-drive-wifi-uplink", 0, syslog.LOG_AUTH)
        syslog.syslog(syslog.LOG_INFO, message)


def handle_request(raw, make_helper, audit=None):
    """Parse one request and run it. Never lets the password reach a reply."""
    audit = audit or _audit
    try:
        request = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return {"ok": False, "error": "invalid_request", "detail": "The request is not JSON."}
    if not isinstance(request, dict):
        return {"ok": False, "error": "invalid_request", "detail": "The request is not an object."}
    op = request.get("op")
    try:
        helper = make_helper()
        result = helper.run(op, request)
        audit(f"op={op} result=ok")
        return {"ok": True, "result": result}
    except HelperError as exc:
        audit(f"op={op} result={exc.code}")
        extra = {k: v for k, v in exc.extra.items() if k in ("restored", "reason", "hotspot_subnet")}
        return {"ok": False, "error": exc.code, "detail": exc.detail, **extra}
    except NMError as exc:
        code = nm_error_code(exc)
        audit(f"op={op} result={code}")
        return {"ok": False, "error": code, "detail": "NetworkManager reported an error."}
    finally:
        request.clear()


def _real_helper(config_path=CONFIG_PATH, require_root=True, registry=None):
    config = Config.load(config_path, require_root=require_root)
    try:
        nm = NMDBus()
    except ImportError:
        raise HelperError("nm_unavailable", "python3-dbus is not installed.")
    return UplinkHelper(nm, config, registry or Registry())


def _suggest_config():
    nm = NMDBus()
    sysfs = Sysfs()
    adapters = []
    for dev in nm.devices():
        if dev["device_type"] != DEVICE_TYPE_WIFI:
            continue
        ident = sysfs.identity(dev["udi"])
        adapters.append({"interface": dev["interface"], "bus": ident.get("bus"),
                         "driver": dev["driver"], "vendor_id": ident.get("vendor_id"),
                         "product_id": ident.get("product_id"), "permanent_mac": dev["perm_hw"],
                         "id_path": dev["id_path"], "managed": dev["managed"],
                         "state": DEVICE_STATES.get(dev["state"], "unknown")})
    hotspots = []
    for c in nm.connections():
        s = c["settings"]
        wifi = s.get("802-11-wireless", {})
        if wifi.get("mode") == "ap" or s.get("ipv4", {}).get("method") == "shared":
            hotspots.append({"uuid": s["connection"]["uuid"], "name": s["connection"].get("id"),
                             "ssid": display_ssid(wifi.get("ssid", b"") or b""),
                             "interface_name": s["connection"].get("interface-name")})
    usb = [a for a in adapters if a["bus"] == "usb"]
    suggestion = None
    if len(usb) == 1:
        suggestion = {
            "usb_adapter": {k: usb[0][k] for k in ("vendor_id", "product_id", "permanent_mac")},
            "protected_connection_uuids": [h["uuid"] for h in hotspots],
            "profile_prefix": DEFAULT_PREFIX, "route_metric": DEFAULT_ROUTE_METRIC,
        }
        suggestion["usb_adapter"]["id_path"] = None
    return {"wifi_adapters": adapters, "hotspot_profiles": hotspots, "suggested_config": suggestion,
            "note": ("Review before installing. Exactly one USB adapter must be listed; set "
                     "usb_adapter.id_path to pin it to one USB port.")
            if suggestion else "Could not pick exactly one USB Wi-Fi adapter; fill the config by hand."}


def _check(config_path):
    """Read-only status with a given config (installation check, no root)."""
    class ReadOnlyRegistry(Registry):
        def load(self):
            try:
                return super().load()
            except (HelperError, PermissionError):
                return {}

        def save(self, profiles):
            raise HelperError("permission_denied", "--check never writes.")

    helper = _real_helper(config_path, require_root=False,
                          registry=ReadOnlyRegistry(REGISTRY_PATH, secure=False))
    return helper.op_status()


def main(argv=None, stdin=None, stdout=None):
    argv = sys.argv[1:] if argv is None else argv
    stdout = stdout or sys.stdout
    if argv:
        try:
            if argv == ["--suggest-config"]:
                _reply(stdout, _suggest_config())
            elif len(argv) == 2 and argv[0] == "--check":
                _reply(stdout, _check(argv[1]))
            else:
                _reply(stdout, {"ok": False, "error": "invalid_request",
                                "detail": "Usage: --suggest-config | --check CONFIG"})
                return 2
        except HelperError as exc:
            _reply(stdout, {"ok": False, "error": exc.code, "detail": exc.detail})
            return 1
        except NMError as exc:
            _reply(stdout, {"ok": False, "error": nm_error_code(exc), "detail": exc.name})
            return 1
        return 0

    if os.geteuid() != 0:
        _reply(stdout, {"ok": False, "error": "permission_denied",
                        "detail": "The helper must be run through sudo."})
        return 1
    os.umask(0o077)
    stdin = stdin or sys.stdin.buffer
    raw = stdin.read(MAX_REQUEST_BYTES + 1)
    if len(raw) > MAX_REQUEST_BYTES:
        _reply(stdout, {"ok": False, "error": "invalid_request", "detail": "Request too large."})
        return 1
    reply = handle_request(raw, _real_helper)
    raw = None
    _reply(stdout, reply)
    return 0


if __name__ == "__main__":
    sys.exit(main())
