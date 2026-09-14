"""
Fakes for the Internet Wi-Fi tests: an in-memory NetworkManager and sysfs
modelled on the real Pi (built-in SDIO adapter running the Secure_Drive
hotspot, USB RTL8188EUS adapter for the uplink). Nothing here uses D-Bus,
starts a process or touches the network.
"""
import copy
import functools
import importlib.util
import itertools
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HELPER_FILE = os.path.join(ROOT, "deploy", "wifi_uplink", "wifi_uplink_helper.py")


def load_helper():
    spec = importlib.util.spec_from_file_location("wifi_uplink_helper", HELPER_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


H = load_helper()

HOTSPOT_UUID = "84444be0-0b56-4f03-86d2-ba90fd229a67"
UNRELATED_UUID = "11111111-2222-4333-8444-555555555555"
USB_MAC = "5c:62:8b:d8:29:b3"
INTERNAL_MAC = "2c:cf:67:ca:e3:4e"
USB_ID_PATH = "platform-xhci-hcd.1-usb-0:2:1.0"
USB_BASE = "/sys/devices/platform/axi/1000120000.pcie/1f00300000.usb/xhci-hcd.1/usb3/3-2/3-2:1.0/net/"
SDIO_BASE = "/sys/devices/platform/axi/1001100000.mmc/mmc_host/mmc1/mmc1:0001/mmc1:0001:1/net/"
USB_DEV = "/org/freedesktop/NetworkManager/Devices/4"
INTERNAL_DEV = "/org/freedesktop/NetworkManager/Devices/3"

# Access-point security flags (NM80211ApSecurityFlags)
WPA2 = 0x100 | 0x8 | 0x80
WPA3 = 0x400 | 0x8 | 0x80
WPA2_WPA3 = 0x100 | 0x400 | 0x8 | 0x80
ENTERPRISE = 0x200 | 0x8 | 0x80
OWE = 0x800 | 0x8


def base_config(**adapter):
    return {"usb_adapter": {"vendor_id": "0bda", "product_id": "8179",
                            "permanent_mac": USB_MAC, "id_path": None, **adapter},
            "protected_connection_uuids": [HOTSPOT_UUID]}


def hotspot_settings(interface="wlan0"):
    return {"connection": {"id": "PiHotspot", "uuid": HOTSPOT_UUID, "type": "802-11-wireless",
                           "interface-name": interface, "autoconnect": True,
                           "autoconnect-priority": 100},
            "802-11-wireless": {"ssid": b"Secure_Drive", "mode": "ap"},
            "802-11-wireless-security": {"key-mgmt": "wpa-psk", "psk": "hotspot-secret"},
            "ipv4": {"method": "shared", "address-data": [{"address": "192.168.50.1", "prefix": 24}]},
            "ipv6": {"method": "disabled"}}


def client_settings(name, uuid, ssid=None, mac=None, prefix=""):
    wifi = {"ssid": (ssid or name.encode()), "mode": "infrastructure"}
    if mac:
        wifi["mac-address"] = bytes(int(p, 16) for p in mac.split(":"))
    return {"connection": {"id": prefix + name, "uuid": uuid, "type": "802-11-wireless",
                           "autoconnect": True},
            "802-11-wireless": wifi,
            "802-11-wireless-security": {"key-mgmt": "wpa-psk", "psk": "someone-elses"},
            "ipv4": {"method": "auto"}, "ipv6": {"method": "auto"}}


class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class FakeSysfs:
    def __init__(self, identities):
        self.identities = identities

    def identity(self, udi):
        return dict(self.identities.get(udi, {"bus": None}))


class FakeNM:
    """Just enough NetworkManager behaviour for the helper's policy."""

    version = "1.52.1-test"

    def __init__(self):
        self.devs, self.conns, self.acs, self.aps, self.ip4s = {}, {}, {}, {}, {}
        self.calls, self.added, self.updated = [], [], []
        #: ssid -> ("ok", address, prefix) | ("fail", reason) | ("hang",)
        self.outcomes = {}
        self.scan_throttled = False
        self._ids = itertools.count(100)

    # ── world building ──
    def add_device(self, path, iface, udi, mac, id_path="", driver="drv", managed=True,
                   state=30, mode=2):
        self.devs[path] = {"path": path, "interface": iface, "udi": udi, "id_path": id_path,
                           "driver": driver, "device_type": 2, "state": state, "reason": 0,
                           "managed": managed, "active_connection": None, "ip4config": None,
                           "perm_hw": mac, "mode": mode, "active_ap": None, "last_scan": 1000}
        self.aps[path] = []

    def add_ap(self, dev, ssid, strength=60, flags=0, wpa=0, rsn=0, freq=2437, mode=2):
        if rsn or wpa:
            flags |= 1
        path = f"/org/freedesktop/NetworkManager/AccessPoint/{next(self._ids)}"
        self.aps[dev].append({"path": path, "ssid": ssid, "bssid": "aa:bb:cc:dd:ee:ff",
                              "strength": strength, "flags": flags, "wpa": wpa, "rsn": rsn,
                              "frequency": freq, "mode": mode})
        return path

    def add_profile(self, settings):
        path = f"/org/freedesktop/NetworkManager/Settings/{next(self._ids)}"
        self.conns[path] = copy.deepcopy(settings)
        return path

    def bring_up(self, conn_path, dev_path, address, prefix=24):
        self._up(conn_path, dev_path, address, prefix)

    # ── the NetworkManager API used by the helper ──
    def devices(self):
        return [copy.deepcopy(d) for d in self.devs.values()]

    def device(self, path):
        if path not in self.devs:
            raise H.NMError("org.freedesktop.DBus.Error.UnknownObject")
        return copy.deepcopy(self.devs[path])

    def request_scan(self, path):
        self.calls.append(("scan", path))
        if self.scan_throttled:
            raise H.NMError("org.freedesktop.NetworkManager.Device.NotAllowed")
        self.devs[path]["last_scan"] += 1000

    def access_points(self, path):
        return copy.deepcopy(self.aps.get(path, []))

    def access_point_strength(self, ap_path):
        for aps in self.aps.values():
            for ap in aps:
                if ap["path"] == ap_path:
                    return ap["strength"]
        return None

    def connections(self):
        out = []
        for path, settings in self.conns.items():
            settings = copy.deepcopy(settings)
            settings.get("802-11-wireless-security", {}).pop("psk", None)  # NM never returns secrets
            out.append({"path": path, "settings": settings})
        return out

    def add_connection(self, settings):
        self.added.append(copy.deepcopy(settings))
        path = self.add_profile(settings)
        self.calls.append(("add", path))
        return path

    def update_connection(self, path, settings):
        self.calls.append(("update", path))
        self.updated.append(copy.deepcopy(settings))
        self.conns[path] = copy.deepcopy(settings)

    def delete_connection(self, path):
        self.calls.append(("delete", path))
        if path not in self.conns:
            raise H.NMError("org.freedesktop.DBus.Error.UnknownObject")
        uuid = self.conns.pop(path)["connection"]["uuid"]
        for dev_path, dev in self.devs.items():
            ac = self.acs.get(dev["active_connection"])
            if ac and ac["uuid"] == uuid:
                self._down(dev_path, reason=38)

    def activate(self, conn_path, dev_path):
        self.calls.append(("activate", conn_path, dev_path))
        settings = self.conns[conn_path]
        ssid = settings["802-11-wireless"]["ssid"]
        outcome = self.outcomes.get(ssid, ("ok", "192.168.43.20", 24))
        self._down(dev_path, reason=60)
        if outcome[0] == "fail":
            self.devs[dev_path].update(state=30, reason=outcome[1])
            return f"/org/freedesktop/NetworkManager/ActiveConnection/{next(self._ids)}"
        if outcome[0] == "hang":
            ac = self._new_ac(conn_path, dev_path, state=1)
            self.devs[dev_path].update(state=50, reason=0, active_connection=ac)
            return ac
        return self._up(conn_path, dev_path, outcome[1], outcome[2])

    def active_connection(self, path):
        return copy.deepcopy(self.acs[path]) if path in self.acs else None

    def disconnect_device(self, path):
        self.calls.append(("disconnect", path))
        self._down(path, reason=39)

    def ip4(self, path):
        return copy.deepcopy(self.ip4s.get(path, {"addresses": [], "gateway": "", "dns": []}))

    # ── internals ──
    def _new_ac(self, conn_path, dev_path, state):
        settings = self.conns[conn_path]
        ac = f"/org/freedesktop/NetworkManager/ActiveConnection/{next(self._ids)}"
        self.acs[ac] = {"path": ac, "state": state, "uuid": settings["connection"]["uuid"],
                        "id": settings["connection"]["id"], "connection": conn_path,
                        "devices": [dev_path]}
        return ac

    def _up(self, conn_path, dev_path, address, prefix):
        ac = self._new_ac(conn_path, dev_path, state=2)
        ip4 = f"/org/freedesktop/NetworkManager/IP4Config/{next(self._ids)}"
        gateway = address.rsplit(".", 1)[0] + ".1"
        self.ip4s[ip4] = {"addresses": [{"address": address, "prefix": prefix}],
                          "gateway": gateway, "dns": [gateway]}
        ssid = self.conns[conn_path]["802-11-wireless"]["ssid"]
        ap = next((a["path"] for a in self.aps.get(dev_path, []) if a["ssid"] == ssid), None)
        self.devs[dev_path].update(state=100, reason=0, active_connection=ac, ip4config=ip4,
                                   active_ap=ap)
        return ac

    def _down(self, dev_path, reason):
        dev = self.devs[dev_path]
        if dev["active_connection"]:
            self.acs.pop(dev["active_connection"], None)
        dev.update(active_connection=None, ip4config=None, active_ap=None, state=30, reason=reason)


class World:
    """The Pi as inspected: hotspot on the built-in adapter, USB adapter idle,
    plus one unrelated saved Wi-Fi profile the feature must never touch."""

    def __init__(self, tmp_path, usb_iface="wlan1", internal_iface="wlan0", usb=True,
                 config=None):
        self.nm = FakeNM()
        self.clock = FakeClock()
        self.internal_udi = SDIO_BASE + internal_iface
        self.usb_udi = USB_BASE + usb_iface
        self.nm.add_device(INTERNAL_DEV, internal_iface, self.internal_udi, INTERNAL_MAC,
                           "platform-1001100000.mmc", "brcmfmac", state=100, mode=3)
        self.hotspot = self.nm.add_profile(hotspot_settings(internal_iface))
        self.nm.bring_up(self.hotspot, INTERNAL_DEV, "192.168.50.1", 24)
        if usb:
            self.nm.add_device(USB_DEV, usb_iface, self.usb_udi, USB_MAC, USB_ID_PATH, "rtl8xxxu")
        self.unrelated = self.nm.add_profile(client_settings("MTN-2.4G-mAf8", UNRELATED_UUID))
        self.sysfs = FakeSysfs({
            self.internal_udi: {"bus": "sdio"},
            self.usb_udi: {"bus": "usb", "vendor_id": "0bda", "product_id": "8179"},
        })
        self.registry = H.Registry(str(tmp_path / "state" / "profiles.json"), secure=False)
        self.lock = functools.partial(H.operation_lock, str(tmp_path / "run" / "uplink.lock"))
        self.helper = H.UplinkHelper(self.nm, H.Config(config or base_config()), self.registry,
                                     self.sysfs, clock=self.clock, sleep=self.clock.sleep,
                                     lock=self.lock)

    def run(self, op, **request):
        return self.helper.run(op, {"op": op, **request})

    def connect(self, ssid, password="correct horse", rsn=WPA2, **request):
        if not any(ap["ssid"] == ssid for ap in self.nm.aps[USB_DEV]):
            self.nm.add_ap(USB_DEV, ssid, rsn=rsn)
        return self.run("connect", ssid_hex=ssid.hex(), password=password, **request)

    def usb_active_uuid(self):
        ac = self.nm.devs[USB_DEV]["active_connection"]
        return self.nm.acs[ac]["uuid"] if ac else None

    def hotspot_intact(self):
        ac = self.nm.devs[INTERNAL_DEV]["active_connection"]
        return (self.hotspot in self.nm.conns and ac is not None
                and self.nm.acs[ac]["uuid"] == HOTSPOT_UUID
                and self.nm.conns[self.hotspot] == hotspot_settings(self.nm.devs[INTERNAL_DEV]["interface"]))

    def touched(self):
        return {arg for call in self.nm.calls for arg in call[1:]}

    def ssids(self):
        return sorted(s["802-11-wireless"]["ssid"] for s in self.nm.conns.values())
