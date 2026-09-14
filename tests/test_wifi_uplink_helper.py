"""
The privileged Wi-Fi uplink helper, against a fake NetworkManager.

These are the rules that protect the Secure_Drive hotspot, so they are
tested where they are enforced: inside the helper. No test here uses D-Bus,
sudo, or the real network.
"""
import ast
import io
import json
import os
import threading

import pytest

from wifi_fakes import (ENTERPRISE, H, HELPER_FILE, HOTSPOT_UUID, INTERNAL_DEV, INTERNAL_MAC,
                        OWE, UNRELATED_UUID, USB_DEV, USB_ID_PATH, USB_MAC, WPA2, WPA2_WPA3, WPA3,
                        World, base_config, client_settings)

SECRET = "p@ss; $(reboot) `id` \"'\\ -o x"
ODD_SSIDS = [b"Cafe Wi-Fi", b"a:b\\c", b'"quoted" \'name\'', "Ünïcødé ☕".encode(),
             b"$(reboot); rm -rf / #", b"-o --help", b"\xff\xfe bad utf8", b"line\nbreak"]


@pytest.fixture(autouse=True)
def _quiet_syslog(monkeypatch):
    monkeypatch.setattr(H, "_audit", lambda message: None)


def code_of(excinfo):
    return excinfo.value.code


# ── finding the USB adapter ──────────────────────────────────────────────────

def test_the_usb_adapter_is_found_by_hardware_identity_not_by_name(tmp_path):
    w = World(tmp_path, usb_iface="wlan0", internal_iface="wlan1")
    dev = w.helper.find_adapter()
    assert dev["path"] == USB_DEV and dev["interface"] == "wlan0"


def test_a_missing_adapter_is_refused_and_nothing_else_is_used(tmp_path):
    w = World(tmp_path, usb=False)
    w.nm.add_ap(INTERNAL_DEV, b"Cafe", rsn=WPA2)
    for op, request in (("connect", {"ssid_hex": b"Cafe".hex(), "password": "correct horse"}),
                        ("scan", {}), ("disconnect", {})):
        with pytest.raises(H.HelperError) as err:
            w.run(op, **request)
        assert code_of(err) == "adapter_missing"
    assert w.nm.calls == [] and w.hotspot_intact()
    assert w.run("status")["adapter"] == {"present": False, "problem": "adapter_missing",
                                          "detail": "The configured USB Wi-Fi adapter was not found."}


def test_the_internal_adapter_is_never_chosen_even_if_the_config_names_its_mac(tmp_path):
    w = World(tmp_path, config=base_config(permanent_mac=INTERNAL_MAC))
    with pytest.raises(H.HelperError) as err:
        w.helper.find_adapter()
    assert code_of(err) == "adapter_missing"


def test_two_adapters_with_the_same_identity_are_ambiguous(tmp_path):
    w = World(tmp_path)
    clone_udi = "/sys/devices/platform/usb4/4-1/4-1:1.0/net/wlan2"
    w.nm.add_device("/org/freedesktop/NetworkManager/Devices/9", "wlan2", clone_udi, USB_MAC,
                    "platform-xhci-hcd.0-usb-0:1:1.0", "rtl8xxxu")
    w.sysfs.identities[clone_udi] = {"bus": "usb", "vendor_id": "0bda", "product_id": "8179"}
    w.nm.add_ap(USB_DEV, b"Cafe", rsn=WPA2)
    with pytest.raises(H.HelperError) as err:
        w.run("connect", ssid_hex=b"Cafe".hex(), password="correct horse")
    assert code_of(err) == "adapter_ambiguous" and w.nm.calls == []


def test_the_usb_port_can_be_pinned(tmp_path):
    assert World(tmp_path, config=base_config(id_path=USB_ID_PATH)).helper.find_adapter()["path"] == USB_DEV
    with pytest.raises(H.HelperError) as err:
        World(tmp_path, config=base_config(id_path="platform-xhci-hcd.0-usb-0:1:1.0")).helper.find_adapter()
    assert code_of(err) == "adapter_missing"


def test_the_sysfs_reader_reads_usb_ids_and_the_bus(tmp_path):
    root = tmp_path.resolve() / "sys"
    usb_device = root / "devices/platform/usb3/3-2"
    usb_iface = usb_device / "3-2:1.0"
    usb_net = usb_iface / "net/wlan1"
    usb_net.mkdir(parents=True)
    (usb_device / "idVendor").write_text("0BDA\n")
    (usb_device / "idProduct").write_text("8179\n")
    (root / "bus/usb").mkdir(parents=True)
    (usb_iface / "subsystem").symlink_to(root / "bus/usb")
    (usb_net / "device").symlink_to(usb_iface)
    sdio = root / "devices/platform/mmc/mmc1:0001:1"
    sdio_net = sdio / "net/wlan0"
    sdio_net.mkdir(parents=True)
    (root / "bus/sdio").mkdir()
    (sdio / "subsystem").symlink_to(root / "bus/sdio")
    (sdio_net / "device").symlink_to(sdio)

    reader = H.Sysfs(str(root))
    assert reader.identity(str(usb_net)) == {"bus": "usb", "vendor_id": "0bda", "product_id": "8179"}
    assert reader.identity(str(sdio_net)) == {"bus": "sdio"}
    assert reader.identity("/etc/passwd") == {"bus": None}
    assert reader.identity(str(root / "devices/../../etc")) == {"bus": None}


# ── status ───────────────────────────────────────────────────────────────────

def test_status_describes_the_adapter_hotspot_and_other_profiles(tmp_path):
    w = World(tmp_path)
    on_internal = client_settings("Ahmad1", "22222222-2222-4333-8444-555555555555")
    on_internal["connection"]["interface-name"] = "wlan0"
    w.nm.add_profile(on_internal)  # can only start on the built-in adapter: not counted
    manual = client_settings("Manual", "33333333-2222-4333-8444-555555555555")
    manual["connection"]["autoconnect"] = False
    w.nm.add_profile(manual)  # never starts by itself: not counted
    status = w.run("status")
    assert status["adapter"]["present"] and status["adapter"]["interface"] == "wlan1"
    assert status["adapter"]["mac"] == USB_MAC and status["adapter"]["problem"] is None
    assert status["hotspot"] == [{"name": "PiHotspot", "ssid": "Secure_Drive", "active": True,
                                  "interface": "wlan0", "on_usb_adapter": False}]
    assert status["connection"] is None and status["saved"] == []
    assert status["unmanaged_wifi_profiles"] == 1  # the unbound MTN profile could auto-join USB
    bound = client_settings("MTN-5G", "44444444-2222-4333-8444-555555555555")
    bound["connection"]["interface-name"] = "wlan1"  # the USB adapter's current name
    w.nm.add_profile(bound)
    assert w.run("status")["unmanaged_wifi_profiles"] == 2


def test_status_shows_the_connection_and_saved_networks(tmp_path):
    w = World(tmp_path)
    w.connect(b"Home")
    status = w.run("status")
    conn = status["connection"]
    assert conn["ssid"] == "Home" and conn["managed_here"] and conn["state"] == "connected"
    assert conn["ipv4"] == [{"address": "192.168.43.20", "prefix": 24}] and conn["subnet_conflict"] is None
    assert [s["ssid"] for s in status["saved"]] == ["Home"] and status["saved"][0]["active"]


# ── connect ──────────────────────────────────────────────────────────────────

def test_connect_creates_a_profile_bound_to_the_usb_adapter_only(tmp_path):
    w = World(tmp_path)
    result = w.connect(b"Cafe", password="correct horse")
    added, final = w.nm.added[0], w.nm.updated[-1]
    assert added["802-11-wireless"]["mac-address"] == bytes.fromhex(USB_MAC.replace(":", ""))
    assert added["connection"]["id"] == "SecureDrive Uplink: Cafe"
    assert added["connection"]["autoconnect"] is False and final["connection"]["autoconnect"] is True
    assert added["connection"]["autoconnect-retries"] == 0 and "interface-name" not in added["connection"]
    assert added["ipv4"]["route-metric"] == 700 and added["ipv4"]["method"] == "auto"
    assert added["802-11-wireless-security"] == {"key-mgmt": "wpa-psk", "psk": "correct horse",
                                                 "psk-flags": 0}
    assert [c for c in w.nm.calls if c[0] == "activate"][0][2] == USB_DEV
    assert INTERNAL_DEV not in w.touched() and w.hotspot not in w.touched() and w.hotspot_intact()
    assert w.registry.load()[result["uuid"]]["state"] == "ready"
    assert result["ipv4"] == [{"address": "192.168.43.20", "prefix": 24}]
    assert "correct horse" not in json.dumps(result)


def test_odd_names_and_passwords_reach_networkmanager_verbatim(tmp_path):
    w = World(tmp_path)
    ssid = b"$(reboot);`id`|-rf \"x\"\\:\xe2\x98\x95"  # 27 bytes: within the 32-byte limit
    w.connect(ssid, password=SECRET)
    assert w.nm.added[0]["802-11-wireless"]["ssid"] == ssid
    assert w.nm.added[0]["802-11-wireless-security"]["psk"] == SECRET


def test_a_wrong_password_removes_the_new_profile_and_restores_the_previous_network(tmp_path):
    w = World(tmp_path)
    home = w.connect(b"Home")["uuid"]
    w.nm.outcomes[b"Cafe"] = ("fail", 7)  # NO_SECRETS: the password was rejected
    with pytest.raises(H.HelperError) as err:
        w.connect(b"Cafe", password="wrong password")
    assert code_of(err) == "wrong_password" and err.value.extra["restored"] is True
    assert w.ssids() == [b"Home", b"MTN-2.4G-mAf8", b"Secure_Drive"]
    assert list(w.registry.load()) == [home] and w.usb_active_uuid() == home
    assert w.hotspot_intact()


def test_the_previous_connection_is_restored_even_if_it_was_not_created_here(tmp_path):
    w = World(tmp_path)
    w.nm.bring_up(w.unrelated, USB_DEV, "10.1.1.5")
    w.nm.outcomes[b"Cafe"] = ("fail", 53)
    with pytest.raises(H.HelperError) as err:
        w.connect(b"Cafe")
    assert code_of(err) == "network_not_found" and err.value.extra["restored"] is True
    assert w.usb_active_uuid() == UNRELATED_UUID
    assert not any(c[0] in ("update", "delete") and c[1] == w.unrelated for c in w.nm.calls)


def test_a_connection_that_never_finishes_times_out_and_is_cleaned_up(tmp_path):
    w = World(tmp_path)
    w.nm.outcomes[b"Slow"] = ("hang",)
    with pytest.raises(H.HelperError) as err:
        w.connect(b"Slow")
    assert code_of(err) == "timeout"
    assert b"Slow" not in w.ssids() and w.registry.load() == {} and w.hotspot_intact()


@pytest.mark.parametrize("reason,code", [(17, "no_ip"), (11, "auth_timeout"), (36, "adapter_missing"),
                                         (9, "activation_failed")])
def test_failure_reasons_become_plain_codes(tmp_path, reason, code):
    w = World(tmp_path)
    w.nm.outcomes[b"Cafe"] = ("fail", reason)
    with pytest.raises(H.HelperError) as err:
        w.connect(b"Cafe")
    assert code_of(err) == code and w.registry.load() == {}


def test_a_network_that_overlaps_the_hotspot_subnet_is_rolled_back(tmp_path):
    w = World(tmp_path)
    home = w.connect(b"Home")["uuid"]
    w.nm.outcomes[b"Clash"] = ("ok", "192.168.50.77", 24)
    with pytest.raises(H.HelperError) as err:
        w.connect(b"Clash")
    assert code_of(err) == "subnet_conflict"
    assert err.value.extra["hotspot_subnet"] == "192.168.50.0/24" and err.value.extra["restored"]
    assert b"Clash" not in w.ssids() and w.usb_active_uuid() == home and w.hotspot_intact()


def test_a_reserved_subnet_from_the_config_is_also_protected(tmp_path):
    w = World(tmp_path, config={**base_config(), "reserved_subnets": ["10.0.0.0/8"]})
    w.nm.outcomes[b"Office"] = ("ok", "10.20.30.40", 16)
    with pytest.raises(H.HelperError) as err:
        w.connect(b"Office")
    assert err.value.extra["hotspot_subnet"] == "10.0.0.0/8"


@pytest.mark.parametrize("flags", [dict(rsn=ENTERPRISE), dict(flags=1), dict(rsn=OWE)])
def test_unsupported_security_is_refused_before_anything_changes(tmp_path, flags):
    w = World(tmp_path)
    w.nm.add_ap(USB_DEV, b"Corp", **flags)
    with pytest.raises(H.HelperError) as err:
        w.run("connect", ssid_hex=b"Corp".hex(), password="correct horse")
    assert code_of(err) == "unsupported_security" and w.nm.calls == []


def test_a_network_out_of_range_is_not_attempted(tmp_path):
    w = World(tmp_path)
    with pytest.raises(H.HelperError) as err:
        w.run("connect", ssid_hex=b"Faraway".hex(), password="correct horse")
    assert code_of(err) == "network_not_found" and w.nm.calls == []


def test_a_hidden_network_uses_the_chosen_security(tmp_path):
    w = World(tmp_path)
    with pytest.raises(H.HelperError) as err:
        w.run("connect", ssid_hex=b"Ghost".hex(), hidden=True, security="wep", password="x" * 10)
    assert code_of(err) == "unsupported_security"
    w.run("connect", ssid_hex=b"Ghost".hex(), hidden=True, security="sae", password="wpa3 password")
    added = w.nm.added[0]
    assert added["802-11-wireless"]["hidden"] is True
    assert added["802-11-wireless-security"]["key-mgmt"] == "sae"


def test_an_open_network_needs_no_password(tmp_path):
    w = World(tmp_path)
    w.nm.add_ap(USB_DEV, b"Free")
    w.run("connect", ssid_hex=b"Free".hex())
    assert "802-11-wireless-security" not in w.nm.added[0]
    with pytest.raises(H.HelperError) as err:
        w.run("connect", ssid_hex=b"Free".hex(), password="unexpected")
    assert code_of(err) == "invalid_password"


def test_a_new_password_for_a_saved_network_replaces_its_profile(tmp_path):
    w = World(tmp_path)
    first = w.connect(b"Home", password="old password")["uuid"]
    second = w.connect(b"Home", password="new password")["uuid"]
    assert first != second and list(w.registry.load()) == [second]
    assert w.ssids().count(b"Home") == 1


def test_a_half_made_profile_is_removed_by_the_next_operation(tmp_path):
    w = World(tmp_path)
    orphan = "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee"
    w.registry.put(orphan, state="pending")
    w.nm.add_profile(client_settings("Half", orphan, mac=USB_MAC, prefix="SecureDrive Uplink: "))
    stray = "99999999-bbbb-4ccc-8ddd-eeeeeeeeeeee"
    w.registry.put(stray, state="pending")  # points at the unrelated MTN profile's lookalike
    w.nm.add_profile(client_settings("NotOurs", stray))
    w.connect(b"Home")
    assert b"Half" not in w.ssids() and b"NotOurs" in w.ssids()
    assert orphan not in w.registry.load() and stray not in w.registry.load()


# ── the hotspot and other profiles are off limits ────────────────────────────

def test_a_hotspot_running_on_the_usb_adapter_blocks_every_change(tmp_path):
    w = World(tmp_path, usb_iface="wlan0", internal_iface="wlan1")
    w.nm.devs[INTERNAL_DEV].update(state=30, mode=2, active_connection=None)
    w.nm.bring_up(w.hotspot, USB_DEV, "192.168.50.1")
    w.nm.devs[USB_DEV]["mode"] = 3
    w.nm.add_ap(USB_DEV, b"Cafe", rsn=WPA2)
    for op, request in (("connect", {"ssid_hex": b"Cafe".hex(), "password": "correct horse"}),
                        ("scan", {}), ("disconnect", {})):
        with pytest.raises(H.HelperError) as err:
            w.run(op, **request)
        assert code_of(err) == "adapter_is_hotspot"
    assert w.nm.calls == [] and w.usb_active_uuid() == HOTSPOT_UUID
    status = w.run("status")
    assert status["hotspot"][0]["on_usb_adapter"] and status["adapter"]["problem"] == "adapter_is_hotspot"


def test_an_unmanaged_adapter_is_reported_not_reconfigured(tmp_path):
    w = World(tmp_path)
    w.nm.devs[USB_DEV].update(managed=False, state=10)
    w.nm.add_ap(USB_DEV, b"Cafe", rsn=WPA2)
    with pytest.raises(H.HelperError) as err:
        w.run("connect", ssid_hex=b"Cafe".hex(), password="correct horse")
    assert code_of(err) == "adapter_unmanaged" and w.nm.calls == []
    assert w.run("status")["adapter"]["problem"] == "adapter_unmanaged"


def test_forget_only_removes_profiles_this_helper_created(tmp_path):
    w = World(tmp_path)
    ours = w.connect(b"Home")["uuid"]
    with pytest.raises(H.HelperError) as err:
        w.run("forget", uuid=HOTSPOT_UUID)
    assert code_of(err) == "protected_profile"
    with pytest.raises(H.HelperError) as err:
        w.run("forget", uuid=UNRELATED_UUID)
    assert code_of(err) == "not_permitted"
    assert w.run("forget", uuid=ours) == {"forgotten": True, "ssid": "Home"}
    assert w.ssids() == [b"MTN-2.4G-mAf8", b"Secure_Drive"] and w.registry.load() == {}
    assert w.hotspot_intact() and w.usb_active_uuid() is None


@pytest.mark.parametrize("tamper", [
    lambda s: s["802-11-wireless"].update({"mac-address": bytes.fromhex(INTERNAL_MAC.replace(":", ""))}),
    lambda s: s["connection"].update({"id": "Renamed"}),
    lambda s: s["802-11-wireless"].update({"mode": "ap"}),
])
def test_a_registered_profile_that_was_edited_elsewhere_is_not_touched(tmp_path, tamper):
    w = World(tmp_path)
    ours = w.connect(b"Home")["uuid"]
    path = next(p for p, s in w.nm.conns.items() if s["connection"]["uuid"] == ours)
    tamper(w.nm.conns[path])
    before = len(w.nm.calls)
    for op in ("forget", "reconnect"):
        with pytest.raises(H.HelperError) as err:
            w.run(op, uuid=ours)
        # An access-point-looking profile is treated as a hotspot either way.
        assert code_of(err) in ("not_permitted", "protected_profile", "adapter_is_hotspot")
    assert path in w.nm.conns and w.nm.calls[before:] == []


def test_reconnect_activates_only_our_profiles_on_the_usb_adapter(tmp_path):
    w = World(tmp_path)
    ours = w.connect(b"Home")["uuid"]
    w.run("disconnect")
    for uuid, code in ((HOTSPOT_UUID, "protected_profile"), (UNRELATED_UUID, "not_permitted")):
        with pytest.raises(H.HelperError) as err:
            w.run("reconnect", uuid=uuid)
        assert code_of(err) == code
    result = w.run("reconnect", uuid=ours)
    assert result["ssid"] == "Home" and w.usb_active_uuid() == ours
    assert all(c[2] == USB_DEV for c in w.nm.calls if c[0] == "activate")


def test_disconnect_takes_down_only_the_usb_adapter(tmp_path):
    w = World(tmp_path)
    w.connect(b"Home")
    assert w.run("disconnect") == {"disconnected": True, "was_connected": True, "ssid": "Home"}
    assert ("disconnect", USB_DEV) in w.nm.calls
    assert not any(c[0] == "delete" for c in w.nm.calls) and w.hotspot_intact()
    assert w.run("disconnect") == {"disconnected": True, "was_connected": False}


# ── scanning ─────────────────────────────────────────────────────────────────

def test_scan_lists_odd_network_names_safely(tmp_path):
    w = World(tmp_path)
    for i, ssid in enumerate(ODD_SSIDS):
        w.nm.add_ap(USB_DEV, ssid, strength=10 + i, rsn=WPA2)
    w.nm.add_ap(USB_DEV, b"")
    w.nm.add_ap(USB_DEV, b"\x00\x00\x00")
    result = w.run("scan")
    by_hex = {n["ssid_hex"]: n for n in result["networks"]}
    assert set(by_hex) == {s.hex() for s in ODD_SSIDS}
    assert by_hex[b"a:b\\c".hex()]["ssid"] == "a:b\\c"
    assert by_hex["Ünïcødé ☕".encode().hex()]["ssid"] == "Ünïcødé ☕"
    assert by_hex[b"\xff\xfe bad utf8".hex()]["ssid"] == "�� bad utf8"
    assert by_hex[b"line\nbreak".hex()]["ssid"] == "line�break"
    assert result["hidden_networks"] == 2 and result["fresh"]
    assert w.nm.calls == [("scan", USB_DEV)]


def test_scan_merges_access_points_and_flags_what_is_supported(tmp_path):
    w = World(tmp_path)
    w.nm.add_ap(USB_DEV, b"Home", strength=30, rsn=WPA2)
    w.nm.add_ap(USB_DEV, b"Home", strength=80, rsn=WPA2)
    w.nm.add_ap(USB_DEV, b"Corp", strength=90, rsn=ENTERPRISE)
    w.nm.add_ap(USB_DEV, b"Mesh", mode=1)  # ad-hoc: not offered
    nets = {n["ssid"]: n for n in w.run("scan")["networks"]}
    assert set(nets) == {"Home", "Corp"}
    assert nets["Home"]["signal"] == 80 and nets["Home"]["access_points"] == 2 and nets["Home"]["supported"]
    assert nets["Corp"]["supported"] is False and nets["Corp"]["security"] == "enterprise"


def test_a_throttled_scan_returns_the_latest_list(tmp_path):
    w = World(tmp_path)
    w.nm.scan_throttled = True
    w.nm.add_ap(USB_DEV, b"Home", rsn=WPA2)
    result = w.run("scan")
    assert result["fresh"] is False and result["networks"][0]["ssid"] == "Home"


@pytest.mark.parametrize("flags,wpa,rsn,expected", [
    (0, 0, 0, ("open", True)), (1, 0, 0, ("wep", False)), (1, 0x100, 0, ("wpa-psk", True)),
    (1, 0, WPA2, ("wpa-psk", True)), (1, 0, WPA3, ("sae", True)), (1, 0, WPA2_WPA3, ("wpa-psk", True)),
    (1, 0, ENTERPRISE, ("enterprise", False)), (1, 0, OWE, ("owe", False)), (0, 0, 0x1000, ("open", True)),
])
def test_security_classification(flags, wpa, rsn, expected):
    key, _, supported = H.classify_security(flags, wpa, rsn)
    assert (key, supported) == expected


# ── input rules and the stdin protocol ───────────────────────────────────────

@pytest.mark.parametrize("security,password,ok", [
    ("wpa-psk", "1234567", False), ("wpa-psk", "12345678", True), ("wpa-psk", "x" * 63, True),
    ("wpa-psk", "x" * 64, False), ("wpa-psk", "a" * 64, True), ("wpa-psk", "pässwörd", False),
    ("wpa-psk", "tab\there!", False), ("wpa-psk", None, False), ("sae", "ünïcode pass", True),
    ("sae", "short", False), ("open", None, True), ("open", "", True), ("wep", "whatever1", False),
])
def test_password_rules(security, password, ok):
    if ok:
        H.validate_password(security, password)
    else:
        with pytest.raises(H.HelperError):
            H.validate_password(security, password)


@pytest.mark.parametrize("request_body,code", [
    ({"op": "connect", "ssid_hex": "abc"}, "invalid_ssid"),
    ({"op": "connect", "ssid_hex": "AB"}, "invalid_ssid"),
    ({"op": "connect", "ssid_hex": "61" * 33}, "invalid_ssid"),
    ({"op": "connect", "ssid_hex": "6162", "hidden": "yes"}, "invalid_request"),
    ({"op": "forget", "uuid": "../../etc/passwd"}, "invalid_request"),
    ({"op": "shell", "cmd": "reboot"}, "invalid_request"),
])
def test_bad_requests_are_refused(tmp_path, request_body, code):
    w = World(tmp_path)
    reply = H.handle_request(json.dumps(request_body).encode(), lambda: w.helper, audit=lambda m: None)
    assert reply["ok"] is False and reply["error"] == code and w.nm.calls == []


@pytest.mark.parametrize("raw", [b"not json", b"[1, 2]", b"\xff\xfe"])
def test_malformed_input_is_refused(tmp_path, raw):
    reply = H.handle_request(raw, lambda: World(tmp_path).helper, audit=lambda m: None)
    assert reply["error"] == "invalid_request"


@pytest.mark.parametrize("outcome", [None, ("fail", 7), ("hang",), "short"])
def test_the_password_never_appears_in_a_reply_or_the_audit_log(tmp_path, outcome):
    w = World(tmp_path)
    w.nm.add_ap(USB_DEV, b"Cafe", rsn=WPA2)
    password = SECRET
    if outcome == "short":
        password = "$(x)"
    elif outcome:
        w.nm.outcomes[b"Cafe"] = outcome
    audit = []
    reply = H.handle_request(json.dumps({"op": "connect", "ssid_hex": b"Cafe".hex(),
                                         "password": password}).encode(),
                             lambda: w.helper, audit=audit.append)
    assert password not in json.dumps(reply) and password not in " ".join(audit)


def test_main_reads_one_request_from_stdin_as_root(tmp_path, monkeypatch):
    w = World(tmp_path)
    w.nm.add_ap(USB_DEV, b"Cafe", rsn=WPA2)
    monkeypatch.setattr(H.os, "geteuid", lambda: 0)
    monkeypatch.setattr(H.os, "umask", lambda mask: 0o022)
    monkeypatch.setattr(H, "_real_helper", lambda: w.helper)
    out = io.StringIO()
    stdin = io.BytesIO(json.dumps({"op": "connect", "ssid_hex": b"Cafe".hex(),
                                   "password": SECRET}).encode())
    assert H.main([], stdin=stdin, stdout=out) == 0
    reply = json.loads(out.getvalue())
    assert reply["ok"] and reply["result"]["ssid"] == "Cafe" and SECRET not in out.getvalue()


def test_main_refuses_requests_unless_run_through_sudo(monkeypatch):
    monkeypatch.setattr(H.os, "geteuid", lambda: 1000)
    out = io.StringIO()
    assert H.main([], stdin=io.BytesIO(b'{"op": "status"}'), stdout=out) == 1
    assert json.loads(out.getvalue())["error"] == "permission_denied"
    out = io.StringIO()
    assert H.main(["--connect"], stdout=out) == 2


def test_only_one_operation_runs_at_a_time(tmp_path):
    lock_path = str(tmp_path / "run" / "uplink.lock")
    held, release = threading.Event(), threading.Event()

    def holder():
        with H.operation_lock(lock_path):
            held.set()
            release.wait(5)

    thread = threading.Thread(target=holder)
    thread.start()
    held.wait(5)
    try:
        # flock is per open file, so a second open in this process conflicts too.
        with pytest.raises(H.HelperError) as err:
            with H.operation_lock(lock_path):
                pass
        assert code_of(err) == "busy"
    finally:
        release.set()
        thread.join()
    with H.operation_lock(lock_path):
        pass


# ── config and registry files ────────────────────────────────────────────────

def write_json(path, data):
    path.write_text(json.dumps(data))
    return str(path)


def test_the_shipped_example_config_is_rejected_until_filled_in():
    example = os.path.join(os.path.dirname(HELPER_FILE), "wifi-uplink.example.json")
    with pytest.raises(H.HelperError) as err:
        H.Config.load(example, require_root=False)
    assert code_of(err) == "config_invalid"


@pytest.mark.parametrize("mutate", [
    lambda c: c.pop("protected_connection_uuids"),
    lambda c: c.update(protected_connection_uuids=[]),
    lambda c: c["usb_adapter"].update(permanent_mac="REPLACE_WITH_USB_ADAPTER_PERMANENT_MAC"),
    lambda c: c["usb_adapter"].update(vendor_id="0bd"),
    lambda c: c["usb_adapter"].update(id_path="../../x y"),
    lambda c: c.update(route_metric=True),
    lambda c: c.update(reserved_subnets=["not a subnet"]),
])
def test_incomplete_configs_are_rejected(tmp_path, mutate):
    config = base_config()
    mutate(config)
    with pytest.raises(H.HelperError) as err:
        H.Config.load(write_json(tmp_path / "c.json", config), require_root=False)
    assert code_of(err) == "config_invalid"


def test_a_config_not_owned_by_root_is_ignored(tmp_path):
    path = write_json(tmp_path / "c.json", base_config())
    assert H.Config.load(path, require_root=False).permanent_mac == USB_MAC
    with pytest.raises(H.HelperError) as err:
        H.Config.load(path)  # owned by the test user, not root
    assert code_of(err) == "config_insecure"
    link = tmp_path / "link.json"
    link.symlink_to(path)
    with pytest.raises(H.HelperError) as err:
        H.Config.load(str(link))
    assert code_of(err) == "config_insecure"
    with pytest.raises(H.HelperError) as err:
        H.Config.load(str(tmp_path / "missing.json"))
    assert code_of(err) == "config_missing"


def test_the_registry_is_private_and_written_atomically(tmp_path):
    registry = H.Registry(str(tmp_path / "state" / "profiles.json"), secure=False)
    registry.put(HOTSPOT_UUID.replace("8", "7"), state="ready")
    assert oct(os.stat(registry.path).st_mode & 0o777) == "0o600"
    assert os.listdir(tmp_path / "state") == ["profiles.json"]
    with pytest.raises(H.HelperError) as err:
        H.Registry(registry.path).load()  # not root-owned
    assert code_of(err) == "config_insecure"


def test_the_helper_is_self_contained():
    source = open(HELPER_FILE).read()
    assert source.startswith("#!/usr/bin/python3 -I\n")
    imported = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module.split(".")[0])
    assert imported <= {"contextlib", "errno", "fcntl", "ipaddress", "json", "os", "re", "stat",
                        "sys", "syslog", "tempfile", "time", "uuid", "dbus"}
    assert "subprocess" not in imported and "shell" not in source.replace("shell-", "")
