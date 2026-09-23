"""
Configuration for vehicle attribute inference.

Mirrors the existing `plate_*` convention in config.json: flat keys, an
`attr_` prefix, validated and clamped on read so a bad edit degrades one
setting instead of taking the gate down. No threshold is hard-coded
elsewhere in this package.

Imports nothing at module scope: config.py merges DEFAULTS into its own,
and importing config here would be a cycle.
"""
from __future__ import annotations

import os

DEFAULTS = {
    # ── feature switches ──────────────────────────────────────────────────
    "attr_enabled": True,
    "attr_colour_enabled": True,
    "attr_type_enabled": True,
    "attr_brand_enabled": True,

    # ── models (never downloaded at runtime; see scripts/) ────────────────
    "attr_model_path": os.path.join("models", "vehicle_attributes_mnv3.pt"),
    "attr_model_format": "pytorch",              # pytorch | onnx | ncnn
    "attr_logo_model_path": os.path.join("models", "vehicle_logo_detector.pt"),
    "attr_logo_model_format": "pytorch",
    "attr_input_size": 224,
    # Which brand method runs:
    #   auto     the whole-vehicle model when its weights are present, else
    #            the two-stage logo pipeline
    #   vehicle  always lamnt2008/car_brands_classification on the vehicle
    #            crop (attributes/brand_classifier.py)
    #   logo     always badge detector -> badge crop -> brand head
    "attr_brand_backend": "auto",
    "attr_brand_model_path": os.path.join("models", "car_brands_beit"),

    # ── vehicle localisation ──────────────────────────────────────────────
    # COCO detector input size for the attribute path. MEASURED on the target
    # Pi 5 for a near vehicle at 1280x720: 366 ms at 640, 109 ms at 320, with
    # an identical box to within 25 px. 320 is the default for that reason.
    # NOT YET MEASURED: recall on small/distant vehicles at reduced input
    # size. Raise this to 640 if the localisation rate in
    # 'evaluate-attributes' drops in the `distant` bucket.
    "attr_vehicle_detection_imgsz": 320,
    "attr_vehicle_detection_confidence": 0.25,
    "attr_vehicle_detection_iou": 0.45,
    # Fraction of the PLATE box's area that must fall inside a vehicle box
    # for that vehicle to be accepted as the plate's vehicle. Not 1.0: both
    # detectors jitter by a few pixels and a plate on the bumper edge can
    # poke marginally outside the body box.
    "attr_plate_containment_fraction": 0.90,

    # ── logo detection ────────────────────────────────────────────────────
    "attr_logo_detection_confidence": 0.30,
    "attr_logo_detection_iou": 0.45,

    # ── voting and thresholds ─────────────────────────────────────────────
    # MEASURED on the target Pi 5: the full attribute path costs ~204 ms per
    # frame (COCO localisation ~113 ms at imgsz 320 + backbone ~91 ms), so
    # the 150 ms budget below admits ONE frame. These defaults are therefore
    # self-consistent: with one frame, one agreeing frame can confirm.
    #
    # To run genuine multi-frame consensus, take the measured ONNX backbone
    # (51 ms vs 91 ms) and/or a smaller attr_vehicle_detection_imgsz, then
    # raise both numbers together. Setting attr_consensus_frames above
    # attr_capture_frame_count is corrected on read, but setting it above
    # what the BUDGET actually admits would silently mean nothing is ever
    # confirmed — hence these defaults.
    "attr_capture_frame_count": 1,        # frames given attribute inference
    "attr_consensus_frames": 1,           # agreeing frames needed to confirm
    # Which colour method runs:
    #   auto  use the trained head when a checkpoint exists, else the HSV
    #         baseline. The honest default: colour works today either way.
    #   hsv   always the HSV baseline (attributes/colour_baseline.py)
    #   head  always the learned head, reporting UNKNOWN when untrained
    # The baseline is also the bar: a head that cannot beat it on
    # `evaluate-attributes` has not earned its place on the Pi.
    "attr_colour_backend": "auto",
    # Fraction of body-region pixels the winning colour must hold before the
    # HSV baseline commits. Below it the answer is `unknown`.
    "attr_colour_hsv_min_support": 0.28,
    "attr_colour_min_confidence": 0.55,
    "attr_type_min_confidence": 0.55,
    "attr_brand_min_confidence": 0.55,

    # ── operation ─────────────────────────────────────────────────────────
    # Latency budget for the whole attribute path. Exceeding it does not
    # fail the capture: remaining frames are skipped and whatever has been
    # inferred is returned, because attributes must never delay the barrier.
    "attr_latency_budget_ms": 150.0,
    # MEASURED on the target Pi 5: the MobileNetV3 backbone at 224px runs in
    # ~87 ms on one thread and ~165 ms on four. Small convolutions do not
    # amortise the thread-synchronisation cost, and extra threads also steal
    # cores from the camera and ANPR workers. One is the measured optimum,
    # not a conservative guess.
    "attr_torch_threads": 1,
    "attr_debug_mode": False,
}

_NUMERIC_BOUNDS = {
    "attr_input_size": (96, 384),
    "attr_vehicle_detection_imgsz": (192, 1280),
    "attr_vehicle_detection_confidence": (0.01, 0.99),
    "attr_vehicle_detection_iou": (0.01, 0.99),
    "attr_plate_containment_fraction": (0.50, 1.0),
    "attr_logo_detection_confidence": (0.01, 0.99),
    "attr_logo_detection_iou": (0.01, 0.99),
    "attr_capture_frame_count": (1, 10),
    "attr_consensus_frames": (1, 10),
    "attr_colour_min_confidence": (0.0, 1.0),
    "attr_colour_hsv_min_support": (0.05, 1.0),
    "attr_type_min_confidence": (0.0, 1.0),
    "attr_brand_min_confidence": (0.0, 1.0),
    "attr_latency_budget_ms": (10.0, 5000.0),
    "attr_torch_threads": (1, 8),
}

_INT_KEYS = {"attr_input_size", "attr_vehicle_detection_imgsz", "attr_capture_frame_count",
             "attr_consensus_frames", "attr_torch_threads"}

_BOOL_KEYS = {k for k, v in DEFAULTS.items() if isinstance(v, bool)}

_VALID_FORMATS = ("pytorch", "onnx", "ncnn")

_VALID_BRAND_BACKENDS = ("auto", "vehicle", "logo")


def validate(raw: dict) -> tuple:
    """
    Coerce a raw config mapping into valid attribute settings.
    Returns (settings, warnings). Never raises.
    """
    raw = raw or {}
    settings = dict(DEFAULTS)
    warnings = []

    for key, default in DEFAULTS.items():
        if key not in raw:
            continue
        value = raw[key]

        if key in _BOOL_KEYS:
            if isinstance(value, bool):
                settings[key] = value
            elif isinstance(value, str) and value.strip().lower() in (
                    "true", "false", "1", "0", "yes", "no"):
                settings[key] = value.strip().lower() in ("true", "1", "yes")
            else:
                warnings.append(f"{key}: expected true/false, using {default!r}")
            continue

        if key in _NUMERIC_BOUNDS:
            lo, hi = _NUMERIC_BOUNDS[key]
            try:
                num = int(value) if key in _INT_KEYS else float(value)
            except (TypeError, ValueError):
                warnings.append(f"{key}: {value!r} is not a number, using {default!r}")
                continue
            if num < lo or num > hi:
                clamped = min(max(num, lo), hi)
                warnings.append(f"{key}: {num} out of range [{lo}, {hi}], clamped to {clamped}")
                num = int(clamped) if key in _INT_KEYS else float(clamped)
            settings[key] = num
            continue

        if key == "attr_colour_backend":
            backend = str(value).strip().lower()
            if backend not in ("auto", "hsv", "head"):
                warnings.append(f"attr_colour_backend: {value!r} is not one of "
                                f"('auto', 'hsv', 'head'), using {default!r}")
            else:
                settings[key] = backend
            continue

        if key == "attr_brand_backend":
            backend = str(value).strip().lower()
            if backend not in _VALID_BRAND_BACKENDS:
                warnings.append(f"attr_brand_backend: {value!r} is not one of "
                                f"{_VALID_BRAND_BACKENDS}, using {default!r}")
            else:
                settings[key] = backend
            continue

        if key in ("attr_model_format", "attr_logo_model_format"):
            fmt = str(value).strip().lower()
            if fmt not in _VALID_FORMATS:
                warnings.append(f"{key}: {value!r} is not one of {_VALID_FORMATS}, "
                                f"using {default!r}")
            else:
                settings[key] = fmt
            continue

        text = str(value).strip()
        if not text:
            warnings.append(f"{key}: empty value, using {default!r}")
            continue
        settings[key] = text

    if settings["attr_consensus_frames"] > settings["attr_capture_frame_count"]:
        warnings.append(
            "attr_consensus_frames cannot exceed attr_capture_frame_count; lowered to "
            f"{settings['attr_capture_frame_count']}")
        settings["attr_consensus_frames"] = settings["attr_capture_frame_count"]

    return settings, warnings


def load(overrides: dict = None) -> dict:
    """Validated settings from the app config store. `overrides` wins."""
    raw = {}
    try:
        import config as _cfg          # lazy: avoids an import cycle
        raw = _cfg.load()
    except Exception:
        raw = {}
    if overrides:
        raw = {**raw, **overrides}
    settings, _warnings = validate(raw)
    return settings


def load_with_warnings(overrides: dict = None) -> tuple:
    raw = {}
    try:
        import config as _cfg
        raw = _cfg.load()
    except Exception:
        raw = {}
    if overrides:
        raw = {**raw, **overrides}
    return validate(raw)


def enabled(settings: dict, attribute: str) -> bool:
    """A head runs only when the master switch AND its own switch are on."""
    if not settings.get("attr_enabled", True):
        return False
    return bool(settings.get(f"attr_{attribute}_enabled", True))


def resolve_path(path: str) -> str:
    """Config paths are project-relative; resolve against the repo root."""
    if not path:
        return ""
    if os.path.isabs(path):
        return path
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(root, path)
