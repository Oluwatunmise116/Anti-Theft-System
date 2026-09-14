#!/bin/bash
# Installs the SecureDrive "Internet Wi-Fi" helper. Run with sudo, from anywhere:
#     sudo deploy/wifi_uplink/install.sh
#
# What it does (and nothing else):
#   * copies the helper to /usr/local/libexec/secure-drive/wifi-uplink-helper (root, 0755)
#   * adds /etc/sudoers.d/secure-drive-wifi-uplink allowing ONLY that helper (validated by visudo)
#   * creates /etc/secure-drive/wifi-uplink.json from the detected USB adapter, after you confirm
#   * runs one read-only status check as the service account
#
# It does not change any network setting, restart any service, or touch the hotspot.
set -euo pipefail

HELPER=/usr/local/libexec/secure-drive/wifi-uplink-helper
SUDOERS=/etc/sudoers.d/secure-drive-wifi-uplink
CONFIG=/etc/secure-drive/wifi-uplink.json
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ $EUID -ne 0 ]]; then
    echo "Run this with sudo." >&2
    exit 1
fi

SERVICE_USER="${SERVICE_USER:-$(systemctl show -p User --value licensedb.service 2>/dev/null || true)}"
if [[ ! "$SERVICE_USER" =~ ^[a-z_][a-z0-9_-]{0,31}$ || "$SERVICE_USER" == root ]]; then
    echo "Could not read a non-root User= from licensedb.service." >&2
    echo "Re-run as:  sudo SERVICE_USER=<account that runs the web app> $0" >&2
    exit 1
fi
id "$SERVICE_USER" >/dev/null

if ! /usr/bin/python3 -I -c 'import dbus' 2>/dev/null; then
    echo "python3-dbus is missing. Install it with:  sudo apt install python3-dbus" >&2
    exit 1
fi

echo "==> Installing the helper at $HELPER"
install -d -o root -g root -m 0755 /usr/local/libexec/secure-drive
install -o root -g root -m 0755 "$HERE/wifi_uplink_helper.py" "$HELPER"
install -d -o root -g root -m 0755 /etc/secure-drive
install -d -o root -g root -m 0700 /var/lib/secure-drive

echo "==> Installing the sudoers rule for '$SERVICE_USER' at $SUDOERS"
tmp="$(mktemp)"
trap 'rm -f "$tmp"' EXIT
sed "s/@SERVICE_USER@/$SERVICE_USER/" "$HERE/sudoers.example" > "$tmp"
visudo -cf "$tmp" >/dev/null
install -o root -g root -m 0440 "$tmp" "$SUDOERS"

if [[ -e "$CONFIG" ]]; then
    echo "==> Keeping the existing $CONFIG"
else
    echo "==> Detecting the USB Wi-Fi adapter and the hotspot profile (read-only)"
    suggestion="$(/usr/bin/python3 -I "$HELPER" --suggest-config)"
    printf '%s\n' "$suggestion"
    if ! config="$(printf '%s' "$suggestion" | /usr/bin/python3 -I -c '
import json, sys
s = json.load(sys.stdin).get("suggested_config")
if not s or not s["protected_connection_uuids"]:
    sys.exit(1)
print(json.dumps(s, indent=2))')"; then
        echo "Could not pick exactly one USB adapter and a hotspot profile." >&2
        echo "Copy $HERE/wifi-uplink.example.json to $CONFIG and fill it in by hand." >&2
        exit 1
    fi
    echo
    echo "Proposed $CONFIG:"
    printf '%s\n' "$config"
    read -r -p "Write it? [y/N] " answer
    if [[ "$answer" != [yY] ]]; then
        echo "Not written. Nothing else was changed apart from the helper and sudoers rule."
        exit 1
    fi
    ( umask 022; printf '%s\n' "$config" > "$CONFIG" )
    chown root:root "$CONFIG"
    chmod 0644 "$CONFIG"
fi

echo "==> Read-only status check, run exactly as the web service will run it"
printf '{"op": "status"}' | sudo -u "$SERVICE_USER" /usr/bin/sudo -n "$HELPER"
echo
echo "Done. Restart the web app to load the new code:  sudo systemctl restart licensedb"
