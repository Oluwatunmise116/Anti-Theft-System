"""
Persistent JSON config store.
Reads/writes config.json in the project root.
"""
import json
import os

CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")

DEFAULTS = {
    "camera_index": 0,
}


def load() -> dict:
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE) as f:
                return {**DEFAULTS, **json.load(f)}
        except Exception:
            pass
    return dict(DEFAULTS)


def save(updates: dict):
    data = load()
    data.update(updates)
    with open(CONFIG_FILE, "w") as f:
        json.dump(data, f, indent=2)


def get(key: str, default=None):
    return load().get(key, default)
