#!/bin/bash
# Removes the SecureDrive "Internet Wi-Fi" helper:  sudo deploy/wifi_uplink/uninstall.sh
#
# Always removes the sudoers rule and the helper, so the web service can no
# longer change any network setting. It then OFFERS to delete the Wi-Fi
# profiles this feature created (listed in its root-only registry). Profiles
# listed as protected in the config — the hotspot — are never deleted.
set -euo pipefail

HELPER=/usr/local/libexec/secure-drive/wifi-uplink-helper
SUDOERS=/etc/sudoers.d/secure-drive-wifi-uplink
CONFIG=/etc/secure-drive/wifi-uplink.json
REGISTRY=/var/lib/secure-drive/wifi-uplink-profiles.json

if [[ $EUID -ne 0 ]]; then
    echo "Run this with sudo." >&2
    exit 1
fi

rm -f "$SUDOERS" "$HELPER"
rmdir /usr/local/libexec/secure-drive 2>/dev/null || true
echo "Removed the sudoers rule and the helper."

if [[ -f "$REGISTRY" ]]; then
    mapfile -t uuids < <(/usr/bin/python3 -I - "$REGISTRY" "$CONFIG" <<'EOF'
import json, re, sys
profiles = json.load(open(sys.argv[1])).get("profiles", {})
try:
    protected = {u.lower() for u in json.load(open(sys.argv[2])).get("protected_connection_uuids", [])}
except OSError:
    protected = set()
for uuid in profiles:
    if re.fullmatch(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", uuid) and uuid not in protected:
        print(uuid)
EOF
)
    if (( ${#uuids[@]} )); then
        echo "Wi-Fi profiles created by this feature:"
        for uuid in "${uuids[@]}"; do
            nmcli -g connection.id connection show uuid "$uuid" 2>/dev/null | sed "s/^/  $uuid  /" || echo "  $uuid  (already gone)"
        done
        read -r -p "Delete them and their saved passwords? The USB Wi-Fi disconnects if one is in use. [y/N] " answer
        if [[ "$answer" == [yY] ]]; then
            for uuid in "${uuids[@]}"; do
                nmcli connection delete uuid "$uuid" 2>/dev/null || true
            done
            echo "Deleted."
        else
            echo "Kept. Delete one later with:  sudo nmcli connection delete uuid <UUID>"
        fi
    fi
    rm -f "$REGISTRY"
fi

read -r -p "Also delete $CONFIG? [y/N] " answer
if [[ "$answer" == [yY] ]]; then
    rm -f "$CONFIG"
    rmdir /etc/secure-drive 2>/dev/null || true
fi
echo "Uninstalled. The Settings page will now say that Wi-Fi control is not installed."
