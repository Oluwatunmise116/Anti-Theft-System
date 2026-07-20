"""
Persistent JSON config store.
Reads/writes config.json in the project root.
"""
import json
import os

CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")

DEFAULTS = {
    "camera_index": 0,
    "camera_width": 1280,
    "camera_height": 720,
    "plate_model_path": "models/nlpdrs_plate_segment.pt",
    "plate_model_format": "pytorch",
    "plate_detection_confidence": 0.35,
    "plate_detection_iou": 0.45,
    "plate_min_width_pixels": 65,
    "plate_consensus_frames": 3,
    "plate_capture_frame_count": 10,
    "plate_max_corrections": 2,
    "plate_min_confirm_confidence": 0.55,
    "plate_ocr_backend": "easyocr",
    "plate_debug_mode": False,
    "plate_crop_padding": 6,
    "plate_debug_max_age_hours": 24,
    "plate_debug_max_storage_mb": 200,
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
