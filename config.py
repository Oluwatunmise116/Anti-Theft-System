"""
Persistent JSON config store.
Reads/writes config.json in the project root.
"""
import json
import os

from attributes.settings import DEFAULTS as _ATTRIBUTE_DEFAULTS

CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")

DEFAULTS = {
    "camera_index": 0,
    "camera_width": 1280,
    "camera_height": 720,
    # The live preview is only for the operator's eyes: it is downscaled and
    # rate-limited so JPEG-encoding it does not starve plate and face
    # detection of CPU. Captures always use the full-resolution frames.
    "preview_max_width": 640,
    "preview_max_fps": 15,
    # Input size for the COCO vehicle detector that Auto-Detect runs on every
    # scan (multiple of 32). 320 is ~3.7x faster than 640 on the Pi 5. Plate
    # reading does not use it: the plate detector searches the full frame.
    "vehicle_detection_imgsz": 320,
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
    # Still-image upload as an alternative to the live camera. An uploaded
    # image is analysed identically but is NOT evidence the vehicle was
    # present — see vehicle_image_source on the trip record.
    "gate_upload_enabled": True,
    "gate_upload_max_mb": 12,
    "gate_upload_max_pixels": 40000000,
    # Uploaded photos are shrunk so their longer side is at most this many
    # pixels (the browser does it before sending, the server again on
    # arrival). Phone photos are 12-48 MP; the detectors never need that.
    "gate_upload_max_side": 1920,
    # Settings → Internet Wi-Fi reachability check: a fixed, non-billable
    # https:// URL and the HTTP status it answers with. Never taken from the
    # browser. (Robase is checked with its own unauthenticated /health.)
    "wifi_internet_probe_url": "https://connectivitycheck.gstatic.com/generate_204",
    "wifi_internet_probe_expect_status": 204,
    # Vehicle attribute recognition (colour / type / brand). Defined in
    # attributes/settings.py so the package and the app share one source of
    # truth; values are validated and clamped on read there.
    **_ATTRIBUTE_DEFAULTS,
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
