# Internet Wi-Fi (USB adapter uplink)

**Settings → Internet Wi-Fi** lets an administrator connect the Pi's **USB Wi-Fi adapter** to a
router or a phone hotspot. This gives the Pi internet for Robase SMS codes. The
**Secure_Drive hotspot** on the Pi's built-in Wi-Fi stays exactly as it is, and phones and
tablets stay connected to it throughout.

Face and fingerprint exits and guest passcodes do not need internet. Only the SMS fallback for
registered drivers does. When the internet is down, that fallback shows an error and the gate
stays closed.

---

## How it keeps away from the hotspot

The web app runs as an ordinary user and cannot change network settings itself. Every change goes
through one small **root-owned helper**. A sudoers rule lets the web service run that helper and
nothing else. The helper enforces the rules itself, on every request:

| Rule | How |
|---|---|
| Only the USB adapter is used | The helper finds the adapter by **hardware identity** in `/etc/secure-drive/wifi-uplink.json`: USB bus, vendor/product id and permanent MAC, optionally the USB port. It never uses the names `wlan0`/`wlan1`. If no adapter matches, or more than one does, it refuses. It never falls back to the built-in adapter. |
| The hotspot is never touched | The hotspot profiles listed in `protected_connection_uuids` are never modified, disconnected or deleted. Any access-point or shared profile is also protected. If a hotspot is running on the USB adapter, the helper refuses to change that adapter. |
| Only its own profiles | The helper can only reconnect or forget profiles it created. The UUID must be in its root-only registry, the name must start with `SecureDrive Uplink:`, and the profile must be locked to the USB adapter's MAC. Your other saved Wi-Fi profiles are left alone. |
| The hotspot addresses stay free | If a network hands the USB adapter an address that overlaps the hotspot subnet (192.168.50.0/24), the connection is undone. The uplink also uses route metric 700, above the hotspot's 600, so hotspot traffic stays on the hotspot even during the second before that rollback. |
| Failures are undone | A failed connect deletes the new profile and brings back whatever the USB adapter was connected to before. Only the USB adapter is involved. |
| Passwords stay private | The password travels on the helper's stdin and goes to NetworkManager over D-Bus. NetworkManager keeps it in its root-only profile store. It is never put on a command line, in the app database, in the browser, in logs, or in any reply. |

**No sharing is enabled.** This feature does not enable internet sharing, bridging or forwarding.
Devices on Secure_Drive still have no internet after the USB Wi-Fi connects; only the Pi does.

---

## What was found on this Pi (read-only inspection, 14 Sep 2026)

| | |
|---|---|
| NetworkManager | 1.52.1. It manages every interface; hostapd and dnsmasq are not in use. |
| Built-in Wi-Fi | `wlan0`, SDIO, `brcmfmac`, MAC `2c:cf:67:ca:e3:4e`. It runs the hotspot. |
| Hotspot profile | `PiHotspot`, UUID `84444be0-0b56-4f03-86d2-ba90fd229a67`, SSID `Secure_Drive`, `ipv4.method shared`, `192.168.50.1/24`, bound to `wlan0` by name. |
| Second hotspot profile | `Hotspot`, UUID `20320094-1305-4f12-8207-41d585b8a358`, SSID `Hotspot-pi`, autoconnect off, bound to `wlan0`. It is protected as well. |
| USB Wi-Fi | `wlan1`, Realtek RTL8188EUS `0bda:8179`, driver `rtl8xxxu`, MAC `5c:62:8b:d8:29:b3`, port `platform-xhci-hcd.1-usb-0:2:1.0`. Managed by NetworkManager, currently disconnected. 2.4 GHz only. |
| Ethernet | `eth0`, `192.168.18.19/24`. It is the default route (metric 100) and supplies DNS. |
| Web service | `licensedb.service` runs as `user`, without `NoNewPrivileges`. |

`--suggest-config` and `--check` (see below) were run read-only against the real NetworkManager.
They identified the USB adapter correctly and reported the hotspot active on `wlan0`.

## Existing settings you should know about (reported, not changed)

1. **`user` already has passwordless sudo for everything** (`/etc/sudoers.d/010_pi-nopasswd`, the
   Raspberry Pi OS default). While that rule exists, the web service account is effectively root,
   so the helper's limits only fully protect you once it is removed. To remove it, first make sure
   you know `user`'s password (set one with `passwd`). Then run
   `sudo visudo -f /etc/sudoers.d/010_pi-nopasswd` and put `#` in front of the rule. The Wi-Fi
   feature keeps working, because it has its own rule.
2. **Hotspot devices have no internet, but their traffic is forwarded without NAT.** IP
   forwarding is on (`net.ipv4.ip_forward = 1`), yet the firewall ruleset is empty and
   `iptables` is not installed. So NetworkManager's shared mode never added its NAT rule.
   Internet-bound packets from hotspot devices leave through Ethernet with their
   192.168.50.x addresses, and no reply ever comes back.
   - Browsers therefore wait for timeouts on anything hosted on the internet (21 s per request on
     Windows). That is why the app now serves its fonts and Chart.js from the Pi (`static/`).
   - This feature doesn't change any of it. Turning forwarding off, or adding NAT to give devices
     internet, is a separate network change: ask before making one.
3. **The hotspot profiles are bound by interface name (`wlan0`).** If the kernel ever named the
   USB adapter `wlan0` at boot, the hotspot would start on the USB adapter. The helper would then
   refuse to touch that adapter, and the page shows a red warning. Binding the hotspot to the
   built-in adapter's MAC would remove the risk, but that is a change to the hotspot profile, so
   it is left to you.
4. **`MTN-2.4G-mAf8` is bound to `wlan1` with autoconnect on.** NetworkManager may join it on the
   USB adapter by itself whenever that network is in range. Profiles created here have a higher
   autoconnect priority (10 against 0), so they win when both are in range. To stop MTN from
   joining by itself: `sudo nmcli connection modify MTN-2.4G-mAf8 connection.autoconnect no`.
   The other saved client profiles (`Ahmad1`, `Hertz6@Airtel-5G-D0FD`, `MTN-5G-mAf8`, `user`,
   `Zubi_Technologies`, `Zubi_Technologies_5G`) are bound to `wlan0`. They cannot join the USB
   adapter. The page counts profiles like MTN under "other saved Wi-Fi profiles".

---

## Install

Run on the Pi, as the admin, from the project folder. The code is already in place.

```bash
cd ~/Desktop/database
sudo deploy/wifi_uplink/install.sh     # shows the detected adapter and asks before writing the config
sudo systemctl restart licensedb       # loads the new code
```

`install.sh` changes no network setting and restarts nothing. It does exactly this:

| Path | Owner / mode | What |
|---|---|---|
| `/usr/local/libexec/secure-drive/wifi-uplink-helper` | root, 0755 | the helper (copied from `deploy/wifi_uplink/wifi_uplink_helper.py`) |
| `/etc/sudoers.d/secure-drive-wifi-uplink` | root, 0440 | `user ALL=(root) NOPASSWD: /usr/local/libexec/secure-drive/wifi-uplink-helper ""`. The `""` means no arguments are allowed. It is checked with `visudo -c` before install. |
| `/etc/secure-drive/wifi-uplink.json` | root, 0644 | adapter identity and protected hotspot UUIDs, created from `--suggest-config` after you confirm |
| `/var/lib/secure-drive/` | root, 0700 | the registry of profiles the helper created (written on first connect) |

At the end it runs one read-only status request as `user`, the same way the web app will.

**Doing it by hand** does the same thing. Use `deploy/wifi_uplink/wifi-uplink.example.json` (placeholders only;
the helper refuses to run until every placeholder is replaced) and `deploy/wifi_uplink/sudoers.example`:

```bash
sudo install -d -o root -g root -m 0755 /usr/local/libexec/secure-drive /etc/secure-drive
sudo install -o root -g root -m 0755 deploy/wifi_uplink/wifi_uplink_helper.py /usr/local/libexec/secure-drive/wifi-uplink-helper
sed 's/@SERVICE_USER@/user/' deploy/wifi_uplink/sudoers.example > /tmp/sd-wifi && sudo visudo -cf /tmp/sd-wifi \
  && sudo install -o root -g root -m 0440 /tmp/sd-wifi /etc/sudoers.d/secure-drive-wifi-uplink
/usr/bin/python3 -I deploy/wifi_uplink/wifi_uplink_helper.py --suggest-config   # copy "suggested_config"
sudo nano /etc/secure-drive/wifi-uplink.json && sudo chmod 0644 /etc/secure-drive/wifi-uplink.json
```

Config fields:

- `usb_adapter.vendor_id` / `product_id` / `permanent_mac` identify the adapter.
- `usb_adapter.id_path` (optional) limits the adapter to one USB port; unplugging it into another port then counts as "missing".
- `protected_connection_uuids` lists the hotspot profiles.
- `reserved_subnets` (optional) lists extra address ranges a network must not use.
- `profile_prefix` and `route_metric` rarely need changing.

**New USB adapter?** Run `--suggest-config` again and update `usb_adapter`. Until you do, the page
says the adapter is not found, by design.

Keep `NoNewPrivileges=` out of `licensedb.service`. With it set, sudo, and therefore this feature,
stop working. The page then says "not allowed to manage the USB Wi-Fi adapter".

---

## Using it

1. Sign in as an administrator. Operators don't see this section, and every endpoint refuses
   them.
2. **Settings → Internet Wi-Fi → Scan for networks.** Networks show signal, band and security.
   WEP, Enhanced Open (OWE) and WPA Enterprise networks are marked "Not supported".
3. Pick a network, type its password, and press **Connect**. The page shows progress, then the
   network name and IP address, or a plain-language reason for the failure.
4. **Connection checks** shows each step: adapter present → joined Wi-Fi → IP address →
   internet via USB → Robase via USB.
   - The two internet checks are pinned to the USB adapter, so Ethernet can't answer for them.
   - The "Whole Pi" line is the normal route, which may be Ethernet, and is labelled as such.
   - Robase is checked with its free `/health` endpoint. No SMS is sent and no credit is used.
   - At the gate, Robase's reply to a real code request is still what counts.
5. **Hidden network:** open "Connect to a hidden network" and enter the name, security type and password.
6. **Saved networks** rejoin automatically after a reboot, a lost signal, or re-plugging the
   adapter.
   - **Connect** switches to one.
   - **Forget** deletes the profile and its saved password.
   - **Disconnect** keeps the USB adapter off until you connect again, re-plug it, or reboot.

Only one Wi-Fi operation runs at a time. A second Connect while one is running is refused.

**Routing:** the uplink uses route metric 700. While Ethernet is plugged in (metric 100), the Pi
keeps using Ethernet for internet and the USB Wi-Fi is the backup. Without Ethernet, the USB Wi-Fi
is the route. NetworkManager merges the DNS servers. Common phone hotspot ranges
(Android 192.168.43.x, iPhone 172.20.10.x) don't clash with the hotspot.

## If the USB adapter is "unmanaged"

The page reports this and changes nothing. Check with
`nmcli -f GENERAL.DEVICE,GENERAL.STATE device show wlan1`. Usual causes are an entry for that
interface in `/etc/network/interfaces` (ifupdown is set `managed=false` here), an
`unmanaged-devices=` line under `/etc/NetworkManager/`, or a udev `NM_UNMANAGED` rule. These fixes
target only the USB adapter:

```bash
sudo nmcli device set wlan1 managed yes        # now, until reboot
# permanently, matched by the adapter's MAC (not its name):
printf '[device-securedrive-usb-wifi]\nmatch-device=mac:5c:62:8b:d8:29:b3\nmanaged=1\n' | \
  sudo tee /etc/NetworkManager/conf.d/90-securedrive-usb-wifi.conf
sudo nmcli general reload conf                 # re-reads the config; does not restart NetworkManager
```

If the cause is an `/etc/network/interfaces` stanza, remove only that interface's stanza. Don't
restart NetworkManager or networking while people are using the hotspot.

---

## Pi acceptance checklist (manual, on the real hardware)

Keep a phone on **Secure_Drive** with the app open for the whole run.

1. **Hotspot unaffected.** Before and after every step,
   `nmcli -f GENERAL.STATE,GENERAL.CONNECTION device show wlan0` shows `PiHotspot` connected, and
   the phone never drops off Secure_Drive.
2. **Login.** An admin signs in on the existing login page. An operator account doesn't see
   Internet Wi-Fi.
3. **Connect.** Scan, choose your router or phone hotspot, and connect. The page shows the SSID
   and IP. On the Pi, `nmcli -f GENERAL.CONNECTION,IP4.ADDRESS,IP4.GATEWAY device show wlan1`
   agrees.
4. **The app stays reachable** throughout:
   - While connecting.
   - After a deliberate wrong password (expect "password was not accepted"; the previous network comes back).
   - After Disconnect.
   - With the adapter unplugged (expect "adapter was not found") and after plugging it back in (it rejoins).
5. **Reboot** (`sudo reboot`). The hotspot is up (`nmcli connection show --active`), the USB
   adapter rejoins the saved network, and the page shows Connected. Also switch the phone hotspot
   off for a minute and on again: the Pi rejoins.
6. **Offline gate.** Disconnect the USB Wi-Fi and unplug Ethernet, then:
   - A face or fingerprint exit works.
   - A guest passcode exit works.
   - For a registered driver, **Send SMS OTP** shows a clear error and the gate stays closed.
7. **Robase status is accurate.**
   - With Ethernet unplugged and the USB Wi-Fi connected, "Robase via USB" is ✓ and a real SMS
     arrives.
   - Turn off mobile data on the phone that provides the Wi-Fi: "Internet via USB" turns ✗ after
     **Check again**, and an SMS send fails with an error.
8. **Narrow permission.** `sudo -l -U user` lists the helper rule.
   `printf '{"op":"status"}' | sudo -u user sudo -n /usr/local/libexec/secure-drive/wifi-uplink-helper`
   prints a status. Any argument (`… wifi-uplink-helper --x`) is refused by sudo once rule 1 in
   "Existing settings" has been removed.

## Read-only inspection commands

```bash
nmcli -f DEVICE,TYPE,STATE,CONNECTION device
nmcli -f NAME,UUID,TYPE,DEVICE,AUTOCONNECT connection show
nmcli -f GENERAL.DEVICE,GENERAL.DRIVER,GENERAL.HWADDR,GENERAL.STATE,GENERAL.CONNECTION,IP4.ADDRESS device show wlan1
ip route; ip -4 addr; cat /proc/sys/net/ipv4/ip_forward
udevadm info -q property /sys/class/net/wlan1 | grep -E '^ID_(BUS|VENDOR_ID|MODEL_ID|PATH)='
/usr/bin/python3 -I deploy/wifi_uplink/wifi_uplink_helper.py --suggest-config
/usr/bin/python3 -I deploy/wifi_uplink/wifi_uplink_helper.py --check /etc/secure-drive/wifi-uplink.json
journalctl -t secure-drive-wifi-uplink --since today      # the helper's own log: operation + result, no secrets
sudo cat /var/lib/secure-drive/wifi-uplink-profiles.json   # profiles this feature created
```

The app's audit log (Verification Review) records `wifi_uplink_connect`, `_reconnect`,
`_disconnect` and `_forget` with the admin, the SSID and the outcome. Passwords are never recorded.

---

## Rollback

- **Emergency, USB only:** `sudo nmcli device disconnect wlan1` (check the name first with
  `nmcli device`). Never run it on the hotspot's adapter.
- **Remove the feature's powers:** `sudo deploy/wifi_uplink/uninstall.sh`.
  - It removes the sudoers rule and the helper.
  - It offers to delete the profiles this feature created (never the hotspot profiles) and the config.
  - The Settings page then says Wi-Fi control is not installed. Everything else in the app works as before.
- **Remove the code** (optional). The feature consists of:
  - `wifi_uplink.py`
  - `deploy/wifi_uplink/`
  - `tests/test_wifi_uplink_*.py` and `tests/wifi_fakes.py`
  - the "Internet Wi-Fi" block in `templates/settings.html`
  - the `/settings/wifi/*` routes in `app.py`
  - the two `wifi_internet_probe_*` keys in `config.py`
  - the `_no_live_network` fixture in `tests/conftest.py`

  Then `sudo systemctl restart licensedb`.

## Error messages

| Code | Meaning / what to do |
|---|---|
| `helper_missing` | Not installed: run `install.sh`. |
| `permission_denied` | The sudoers rule is missing, or `NoNewPrivileges` is set on the service. |
| `config_missing` / `config_invalid` / `config_insecure` | `/etc/secure-drive/wifi-uplink.json` is missing, still has placeholders, or isn't root-owned. |
| `adapter_missing` | The USB adapter is unplugged, the wrong adapter is plugged in, or it's in a different port than `id_path`. |
| `adapter_ambiguous` | Two adapters match (for example cheap clones sharing a MAC). Unplug one or set `id_path`. |
| `adapter_unmanaged` / `adapter_unavailable` | See "If the USB adapter is unmanaged"; or the adapter is still starting. |
| `adapter_is_hotspot` | A hotspot profile is running on the USB adapter; nothing is changed. |
| `wrong_password` / `auth_timeout` | The password was rejected, or the handshake timed out (usually a wrong password or weak signal). |
| `no_ip` | Joined, but the network gave no address (DHCP). |
| `subnet_conflict` | The network uses 192.168.50.x like the hotspot; it was undone. Change the router or phone range. |
| `network_not_found` / `timeout` | Out of range, or too slow. |
| `busy` | Another Wi-Fi operation is running. |

## Tests

```bash
env/bin/python -m pytest tests/test_wifi_uplink_helper.py tests/test_wifi_uplink_routes.py -q
```

They use a fake NetworkManager and fake probes only. `tests/conftest.py` fails any test that
tries to run the real helper, open a network connection for a check, or read live routes.

## Not yet verified on the hardware

The automated tests and the read-only checks above cover the code paths, but these have **not**
been run on the Pi:

- Any real connect, disconnect, reconnect or forget.
- A real scan request, since NetworkManager scans were not triggered.
- The installed sudoers rule and root-owned helper.
- A wrong-password failure reason and a DHCP address as the RTL8188EUS reports them.
- Autoconnect after reboot, and after unplugging and re-plugging the adapter.
- A subnet-overlap rollback, and whether the hotspot stays reachable during it.
- The pinned internet and Robase checks over the USB adapter, and the live `/health` response.
- WPA3 (SAE) with the `rtl8xxxu` driver.

Walk through the acceptance checklist before relying on it.
