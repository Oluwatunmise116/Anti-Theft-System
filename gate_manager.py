"""
Gate Manager — ANPR, vehicle color detection, face capture/verify, fingerprint.
"""
import threading
import queue
import time
import os
import uuid
import re

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    np = None
    NUMPY_AVAILABLE = False

try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False

try:
    import adafruit_fingerprint
    SENSOR_AVAILABLE = True
except ImportError:
    SENSOR_AVAILABLE = False

try:
    from ultralytics import YOLO as _YOLO_CLASS
    YOLO_AVAILABLE = True
except ImportError:
    _YOLO_CLASS = None
    YOLO_AVAILABLE = False

import face_manager as fm
import fingerprint_manager as fp_mgr
import config as cfg
import anpr as _anpr
from anpr import preprocessing as _anpr_pp
# Vehicle colour / type / brand. Importing this package loads no model and
# touches no network — every model is lazy (see attributes/classifiers.py).
import attributes as va

GATE_PHOTOS_DIR   = "gate_photos"
DEBUG_PHOTOS_DIR  = os.path.join(GATE_PHOTOS_DIR, "debug")
os.makedirs(GATE_PHOTOS_DIR, exist_ok=True)

FACE_TOLERANCE  = 0.60   # SFace cosine distance; same-person threshold ~0.637
FP_THRESHOLD    = 30
COLOR_TOLERANCE = 0.35


def cleanup_debug_artifacts():
    """
    ANPR debug sessions can contain full vehicle/plate photos, which may be
    personal information — they are only ever written when plate_debug_mode
    is on, and are pruned here by age and total size so they don't
    accumulate indefinitely.
    """
    if not os.path.isdir(DEBUG_PHOTOS_DIR):
        return
    c = cfg.load()
    max_age_s = c.get("plate_debug_max_age_hours", 24) * 3600
    max_bytes = c.get("plate_debug_max_storage_mb", 200) * 1024 * 1024
    now = time.time()

    sessions = []
    total_size = 0
    for name in os.listdir(DEBUG_PHOTOS_DIR):
        path = os.path.join(DEBUG_PHOTOS_DIR, name)
        if not os.path.isdir(path):
            continue
        mtime = os.path.getmtime(path)
        size = sum(
            os.path.getsize(os.path.join(dp, f))
            for dp, _, files in os.walk(path) for f in files
        )
        if now - mtime > max_age_s:
            _rmtree(path)
            continue
        sessions.append((mtime, size, path))
        total_size += size

    sessions.sort()   # oldest first
    while total_size > max_bytes and sessions:
        mtime, size, path = sessions.pop(0)
        _rmtree(path)
        total_size -= size


def _rmtree(path):
    import shutil
    try:
        shutil.rmtree(path, ignore_errors=True)
    except Exception:
        pass

# ── NIGERIAN PLATE DETECTOR (anpr package) ────────────────────────────────────
# The plate model is the NLPDRS YOLOv8 segmentation model (trained on
# Nigerian plates, annotated to segment just the plate-number portion) from
# https://github.com/esssyjr/NLPDRS-Nierian-License-Plate-Detection-and-Recognition-System-
# placed at models/nlpdrs_plate_segment.pt. Nothing in this module downloads
# a model at runtime; if the configured weights file is missing or fails to
# load, detection fails clearly (PlateModelError) and manual plate entry
# remains available.

_pipeline_lock = threading.Lock()
_pipeline = None


def _anpr_config() -> dict:
    c = cfg.load()
    return {
        "plate_model_path": c.get(
            "plate_model_path", os.path.join("models", "nlpdrs_plate_segment.pt")
        ),
        "plate_detection_confidence": c.get("plate_detection_confidence", 0.35),
        "plate_detection_iou": c.get("plate_detection_iou", 0.45),
        "plate_min_width_pixels": c.get("plate_min_width_pixels", 100),
        "plate_consensus_frames": c.get("plate_consensus_frames", 3),
        "plate_capture_frame_count": c.get("plate_capture_frame_count", 10),
        "plate_max_corrections": c.get("plate_max_corrections", 2),
        "plate_min_confirm_confidence": c.get("plate_min_confirm_confidence", 0.55),
        "plate_ocr_backend": c.get("plate_ocr_backend", "easyocr"),
        "plate_debug_mode": c.get("plate_debug_mode", False),
        "plate_crop_padding": c.get("plate_crop_padding", 6),
    }


def get_pipeline():
    """
    Build the ANPR pipeline once and reuse it for the life of the process —
    the YOLO detector and OCR backend are each loaded once behind their own
    lock (see anpr.detector.PlateDetector / anpr.recognizer), never
    reloaded per gate request.
    """
    global _pipeline
    if _pipeline is not None:
        return _pipeline
    with _pipeline_lock:
        if _pipeline is None:
            _pipeline = _anpr.build_pipeline(_anpr_config())
    return _pipeline


def reset_pipeline():
    """Force a rebuild on next use — call after plate_* settings change."""
    global _pipeline
    with _pipeline_lock:
        _pipeline = None


def plate_model_status() -> dict:
    return get_pipeline().status()


def warm_up_ocr():
    """Pre-load the OCR backend in the background so the first plate scan is instant."""
    pipeline = get_pipeline()
    warm = getattr(pipeline.ocr, "warm_up", None)
    if warm:
        threading.Thread(target=warm, daemon=True).start()


def warm_up_vehicle_detector():
    """
    Pre-load the COCO vehicle model in the background.

    run_vehicle_auto_capture loads it BEFORE its first SSE push, so on a cold
    process the operator clicks "Auto-Detect Vehicle" and sees nothing at all
    for the duration of the load — indistinguishable from a dead button.
    Loading it at start-up, like the OCR backend, removes that silence.
    """
    threading.Thread(target=_get_vehicle_yolo, daemon=True,
                     name="VehicleYoloWarmup").start()


# ── YOLO VEHICLE MODEL (for body crop → accurate colour) ──────────────────────
# Uses YOLOv8n pre-trained on COCO.  Auto-downloaded by ultralytics on first use.
# Vehicle classes: car=2, motorcycle=3, bus=5, truck=7
_VEHICLE_CLASSES  = [2, 3, 5, 7]
_vehicle_yolo     = None
_vehicle_yolo_lock = threading.Lock()


def _get_vehicle_yolo():
    global _vehicle_yolo
    if _vehicle_yolo is not None:
        return _vehicle_yolo
    if not YOLO_AVAILABLE:
        return None
    with _vehicle_yolo_lock:
        if _vehicle_yolo is not None:
            return _vehicle_yolo
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _vehicle_yolo = _YOLO_CLASS("yolov8n.pt")   # auto-downloads ~6 MB
        except Exception as e:
            print(f"[gate_manager] Vehicle YOLO load error: {e}")
    return _vehicle_yolo


# COCO class ids -> the coarse class names the attribute fusion uses.
_VEHICLE_CLASS_NAMES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


def vehicle_detection_imgsz() -> int:
    """
    Input size for the COCO vehicle detector that Auto-Detect runs on every
    scan, and for detect_vehicle_crop() callers (config:
    vehicle_detection_imgsz). Measured on this Pi 5 over 15 public plate
    images: 114 ms per frame at 320 against 425 ms at Ultralytics' default
    640, with the same largest-vehicle box (median IoU 0.97). The plate
    pipeline does not use it — it searches the full frame. Rounded down to
    a multiple of 32, clamped 192-1280.
    """
    try:
        value = int(cfg.get("vehicle_detection_imgsz", 320))
    except (TypeError, ValueError):
        value = 320
    return max(192, min(1280, (value // 32) * 32))


def detect_vehicle_crop_detailed(frame_bgr) -> dict:
    """
    Run YOLOv8n once and return everything downstream stages need about the
    largest vehicle: the crop, its box, the coarse COCO class and the
    detection confidence.

    This is the single vehicle-detection call per frame. ANPR, colour, type
    and brand all reuse this one crop — YOLO is never re-run per attribute.
    Returns {"crop": frame, "bbox": None, ...} when nothing is detected.
    """
    empty = {"crop": frame_bgr, "bbox": None, "class_name": None, "confidence": 0.0}
    model = _get_vehicle_yolo()
    if model is None or not NUMPY_AVAILABLE or frame_bgr is None:
        return empty
    h, w = frame_bgr.shape[:2]
    try:
        results = model(frame_bgr, classes=_VEHICLE_CLASSES, conf=0.25,
                        imgsz=vehicle_detection_imgsz(), verbose=False)[0]
        if results.boxes is None or len(results.boxes) == 0:
            return empty
        boxes = results.boxes
        xyxy  = boxes.xyxy.cpu().numpy()
        areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in xyxy]
        best  = int(np.argmax(areas))
        x1, y1, x2, y2 = xyxy[best].astype(int)
        pad = 4
        x1c = max(0, x1 - pad); y1c = max(0, y1 - pad)
        x2c = min(w, x2 + pad); y2c = min(h, y2 + pad)
        try:
            class_id = int(boxes.cls[best].cpu().numpy())
            confidence = float(boxes.conf[best].cpu().numpy())
        except Exception:
            class_id, confidence = -1, 0.0
        return {
            "crop": frame_bgr[y1c:y2c, x1c:x2c],
            "bbox": (int(x1), int(y1), int(x2), int(y2)),
            "class_name": _VEHICLE_CLASS_NAMES.get(class_id),
            "confidence": confidence,
        }
    except Exception:
        return empty


def detect_vehicle_crop(frame_bgr):
    """
    Run YOLOv8n to find the largest vehicle in frame.
    Returns (crop_bgr, (x1,y1,x2,y2)) or (frame_bgr, None) if nothing detected.

    Unchanged signature — existing ANPR call sites depend on it. It now
    delegates to detect_vehicle_crop_detailed so there is one detection
    implementation.
    """
    detail = detect_vehicle_crop_detailed(frame_bgr)
    return detail["crop"], detail["bbox"]


# ── ANPR ──────────────────────────────────────────────────────────────────────
# Detection (finding the plate box) and OCR (reading the plate text) are two
# separate stages with two separate accuracy numbers — see
# reports/plate_detector_metrics.json for detector-only metrics and
# README_ANPR.md for why OCR accuracy is reported separately. This module
# only orchestrates frame capture / ROI selection around anpr.PlateRecognitionPipeline;
# all detection, preprocessing, OCR and consensus logic lives in anpr/.

def _vehicle_region_provider(frame_bgr):
    """
    Try the vehicle-crop ROI first (faster, less background to confuse the
    detector on a small plate), then fall back to the full frame. Vehicle
    detection here reuses the existing YOLOv8n COCO vehicle model above —
    this is unrelated to the plate-model mirror issue and is left as-is.
    """
    regions = []
    crop, bbox = detect_vehicle_crop(frame_bgr)
    if bbox is not None:
        x1, y1, _, _ = bbox
        regions.append((crop, (x1, y1), "vehicle_roi"))
    regions.append((frame_bgr, (0, 0), "full_frame"))
    return regions


def detect_plate_multi_frame(frames: list, debug_session_id: str = None):
    """
    Run the full multi-frame ANPR pipeline (detector -> crop QA -> rectify
    -> OCR -> safe correction -> consensus) and return a
    anpr.models.PlateRecognitionResult. Never raises — a missing/broken
    model or camera failure comes back as a NOT_DETECTED result with a
    human-readable rejection_reason so manual entry can take over.
    """
    pipeline = get_pipeline()
    return pipeline.process_frames(
        frames, region_provider=_vehicle_region_provider,
        debug_session_id=debug_session_id,
    )


def detect_plate(frame_bgr) -> str:
    """
    Compatibility wrapper for call sites that only want a bare string from a
    single frame (e.g. the exit page's live "anpr-now" quick check). Reuses
    the same already-loaded detector/OCR singletons — a lightweight
    single-frame consensus requirement is applied only for this call, no
    model is reloaded.
    """
    if not OPENCV_AVAILABLE or frame_bgr is None:
        return ""
    pipeline = get_pipeline()
    quick_config = dict(pipeline.config)
    quick_config["plate_consensus_frames"] = 1
    quick_pipeline = _anpr.PlateRecognitionPipeline(pipeline.detector, pipeline.ocr, quick_config)
    result = quick_pipeline.process_frames([frame_bgr], region_provider=_vehicle_region_provider)
    return _anpr.detect_plate_string(result)


# ── VEHICLE COLOUR / TYPE / BRAND ────────────────────────────────────────────
# Recognition lives in the attributes/ package (localisation, heads, voting).
# This module only orchestrates capture and reports progress, exactly as it
# does for ANPR. Attributes are ADVISORY: they never influence allow/deny.

COLOR_TOLERANCE = 0.35

_attr_pipeline = None
_attr_pipeline_lock = threading.Lock()


def get_attribute_pipeline():
    """Build the attribute pipeline once and reuse it for the process life."""
    global _attr_pipeline
    if _attr_pipeline is not None:
        return _attr_pipeline
    with _attr_pipeline_lock:
        if _attr_pipeline is None:
            _attr_pipeline = va.build_pipeline(va.settings.load())
    return _attr_pipeline


def reset_attribute_pipeline():
    """Force a rebuild on next use — call after attr_* settings change."""
    global _attr_pipeline
    with _attr_pipeline_lock:
        _attr_pipeline = None


def attribute_model_status() -> dict:
    """Readiness for the settings/model-status interface. Loads no model."""
    try:
        return get_attribute_pipeline().status()
    except Exception as exc:
        return {"enabled": False, "error": f"{type(exc).__name__}: {exc}"}


def warm_up_attributes():
    """
    Build the backbone in the background at start-up. The first forward pass
    through a fresh torch module is several times slower than steady state,
    so paying it here keeps the first vehicle of the day inside the budget.
    """
    def _warm():
        try:
            get_attribute_pipeline().warm_up()
        except Exception as exc:
            print(f"[gate_manager] Attribute warm-up skipped: {exc}")

    threading.Thread(target=_warm, daemon=True, name="AttrWarmup").start()


def analyse_vehicle_attributes(frames: list, plate_box=None, vehicle_boxes=None,
                               progress=None):
    """
    Run colour/type/brand over the frames the plate stage already used.

    `vehicle_boxes` are COCO boxes this module already computed during
    capture — passing them in avoids a second YOLO pass, measured at ~297 ms
    per 1280x720 frame on the Pi.

    Never raises: a missing model, an unlocalisable vehicle or a corrupt
    frame all come back as a structured result, so entry always remains
    completable by hand.
    """
    try:
        return get_attribute_pipeline().process_frames(
            frames, plate_box=plate_box,
            vehicle_boxes_for_frame=(lambda i: vehicle_boxes) if vehicle_boxes else None,
            progress=progress)
    except Exception as exc:
        print(f"[gate_manager] Attribute pipeline error: {type(exc).__name__}: {exc}")
        return va.VehicleAttributeResult.failed(
            len(frames or []), f"attribute pipeline error: {type(exc).__name__}")


def _run_vehicle_attributes(cid: str, push, frames: list, plate_result=None) -> dict:
    """
    Shared tail of both vehicle-capture flows, run AFTER the plate result
    has already been pushed — the barrier never waits on a brand classifier.
    """
    settings = va.settings.load()
    if not settings.get("attr_enabled", True):
        return {}

    plate_box = getattr(plate_result, "bounding_box", None) if plate_result else None
    push("step", "Reading vehicle attributes…")
    result = analyse_vehicle_attributes(
        frames, plate_box=plate_box, progress=lambda text: push("info", text))
    temp_set(cid, "vehicle_attributes", result.to_dict())

    api = result.to_api_dict()
    if result.status == "VEHICLE_NOT_LOCALISED":
        push("warning",
             "Could not tell which vehicle carries this plate — attributes skipped",
             vehicle_attributes=api)
        return api

    parts = [f"{name.capitalize()}: {value.value} ({value.confidence * 100:.0f}%)"
             for name, value in result.values().items() if value.is_usable]
    if parts:
        confirmed = all(not v.needs_operator_review
                        for v in result.values().values() if v.is_usable)
        push("ok" if confirmed else "warning",
             "Vehicle attributes — " + ", ".join(parts)
             + ("" if confirmed else " — advisory, verify before accepting"),
             vehicle_attributes=api)
    else:
        push("info", "Vehicle attributes unavailable — enter them manually if needed",
             vehicle_attributes=api)
    return api


# ── TEMP CAPTURE STORE ────────────────────────────────────────────────────────
_temp: dict = {}
_temp_lock  = threading.Lock()


def new_capture_id(owner=None) -> str:
    """A capture id, optionally bound to the gate session that created it."""
    cid = uuid.uuid4().hex[:16]
    with _temp_lock:
        _temp[cid] = {"owner_session": owner} if owner is not None else {}
    return cid


def claim_capture(cid: str, owner) -> bool:
    """
    Bind a capture to the gate session using it. Returns False when another
    session already owns it, so one operator's capture — including the
    identity recorded from its face/fingerprint search — can never be used
    to log another operator's entry.
    """
    if not cid:
        return False
    with _temp_lock:
        entry = _temp.setdefault(cid, {})
        current = entry.get("owner_session")
        if current is None:
            entry["owner_session"] = owner
            return True
        return current == owner


def temp_get(cid: str) -> dict:
    with _temp_lock:
        return dict(_temp.get(cid, {}))


def temp_set(cid: str, key: str, value):
    with _temp_lock:
        _temp.setdefault(cid, {})[key] = value


def temp_clear(cid: str):
    with _temp_lock:
        _temp.pop(cid, None)


# ── VEHICLE CAPTURE (STEP 1: plate + colour) ─────────────────────────────────
_vehicle_sessions: dict = {}
_vs_lock = threading.Lock()


def start_vehicle_session(cid: str) -> bool:
    with _vs_lock:
        if cid in _vehicle_sessions:
            return False
        _vehicle_sessions[cid] = queue.Queue()
    return True


def get_vehicle_queue(cid: str):
    return _vehicle_sessions.get(cid)


def end_vehicle_session(cid: str):
    with _vs_lock:
        _vehicle_sessions.pop(cid, None)


def _push_vehicle(cid: str, t: str, txt: str, **kw):
    q = _vehicle_sessions.get(cid)
    if q:
        q.put({"type": t, "text": txt, **kw})


def snap_vehicle_frame(cid: str) -> dict:
    """Grab one raw frame from the camera and save it. Synchronous — no SSE, no analysis."""
    if not OPENCV_AVAILABLE:
        return {"error": "OpenCV not installed"}
    if not fm.ensure_camera_running():
        return {"error": "Camera unavailable — check connection"}
    fm.request_capture()
    rgb = fm.get_captured_frame(timeout=10)
    if rgb is None:
        return {"error": "No frame received from camera"}
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    path = os.path.join(GATE_PHOTOS_DIR, f"vehicle_{cid}.jpg")
    cv2.imwrite(path, bgr)
    temp_set(cid, "vehicle_photo", path)
    temp_set(cid, "vehicle_image_source", "camera")
    return {"ok": True, "url": f"/gate/photo/vehicle_{cid}.jpg",
            "image_source": "camera"}


# ── UPLOADED VEHICLE IMAGE ───────────────────────────────────────────────────
# An operator can analyse a still image instead of the live camera. The
# analysis path is identical (run_vehicle_capture), but the PROVENANCE is
# not: a camera capture is evidence the vehicle was physically at the gate,
# an uploaded file is not. That difference is recorded on the trip as
# vehicle_image_source and shown in the UI, so nobody reads an uploaded
# photo as proof of presence.

#: Accepted upload container formats. Checked by DECODING the bytes, not by
#: trusting the filename or the declared content type.
UPLOAD_MIN_DIMENSION = 64


def save_uploaded_vehicle_image(cid: str, data: bytes, original_name: str = "") -> dict:
    """
    Validate an uploaded image and store it as this capture's vehicle photo.

    The file is decoded from memory and RE-ENCODED to JPEG rather than being
    written through. That normalises the format, strips EXIF (which can
    carry location and device metadata the gate has no reason to keep), and
    means a file that merely looks like an image cannot be stored under a
    .jpg name.

    Returns {"ok": True, "url": ...} or {"error": "..."} — never raises.
    """
    if not OPENCV_AVAILABLE or not NUMPY_AVAILABLE:
        return {"error": "OpenCV not installed"}

    c = cfg.load()
    if not c.get("gate_upload_enabled", True):
        return {"error": "Image upload is disabled in configuration"}

    max_bytes = int(c.get("gate_upload_max_mb", 12)) * 1024 * 1024
    if not data:
        return {"error": "No image data received"}
    if len(data) > max_bytes:
        return {"error": f"Image is larger than the "
                         f"{c.get('gate_upload_max_mb', 12)} MB limit"}

    try:
        buffer = np.frombuffer(data, dtype=np.uint8)
        bgr = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    except Exception:
        bgr = None
    if bgr is None or bgr.size == 0:
        return {"error": "That file is not a readable image "
                         "(JPEG or PNG expected)"}

    height, width = bgr.shape[:2]
    if height < UPLOAD_MIN_DIMENSION or width < UPLOAD_MIN_DIMENSION:
        return {"error": f"Image is too small to analyse "
                         f"({width}x{height}; minimum "
                         f"{UPLOAD_MIN_DIMENSION}x{UPLOAD_MIN_DIMENSION})"}

    max_pixels = int(c.get("gate_upload_max_pixels", 40_000_000))
    if height * width > max_pixels:
        return {"error": f"Image resolution is too large ({width}x{height})"}

    # Shrink large photos before they are saved: every later step (save,
    # re-read, plate and vehicle detection) is faster on a smaller image,
    # and the detectors work at far below phone-camera resolution anyway.
    # The shorter side never drops below the minimum analysable size.
    max_side = int(c.get("gate_upload_max_side", 1920) or 0)
    if max_side > 0 and max(height, width) > max_side:
        scale = max(max_side / float(max(height, width)),
                    UPLOAD_MIN_DIMENSION / float(min(height, width)))
        if scale < 1.0:
            bgr = cv2.resize(bgr, (max(1, round(width * scale)), max(1, round(height * scale))),
                             interpolation=cv2.INTER_AREA)
            height, width = bgr.shape[:2]

    # The filename is constructed here, never taken from the upload, so a
    # crafted name cannot escape the gate photos directory.
    path = os.path.join(GATE_PHOTOS_DIR, f"vehicle_{cid}.jpg")
    if not cv2.imwrite(path, bgr, [cv2.IMWRITE_JPEG_QUALITY, 92]):
        return {"error": "Could not save the uploaded image"}

    temp_set(cid, "vehicle_photo", path)
    temp_set(cid, "vehicle_image_source", "upload")
    # A previous camera capture's results must not survive a new upload.
    temp_set(cid, "plate_result", None)
    temp_set(cid, "plate_result_object", None)
    temp_set(cid, "vehicle_attributes", None)
    return {
        "ok": True,
        "url": f"/gate/photo/vehicle_{cid}.jpg",
        "width": int(width),
        "height": int(height),
        "image_source": "upload",
    }


def _capture_plate_frame_burst(count: int, interval: float = 0.05) -> list:
    """
    Grab `count` consecutive raw frames from the running camera worker over
    a short window (so a stable-but-not-frozen vehicle gets several looks),
    then keep only the sharpest, best-exposed ones. Never depends on a
    single frame.
    """
    frames = []
    for _ in range(count):
        with fm._cam_lock:
            raw = fm._cam_state.get("raw")
        if raw is not None:
            frames.append(raw.copy())
        time.sleep(interval)
    return frames


def _run_plate_recognition(cid: str, push, frames: list) -> dict:
    """
    Shared tail-end of both vehicle capture flows: run the multi-frame ANPR
    pipeline, push the structured result over SSE, and stash it for
    /gate/entry/confirm. Returns the result's API dict.
    """
    c = cfg.load()
    frame_count = c.get("plate_capture_frame_count", 10)
    best_frames = _anpr_pp.select_best_frames(frames, count=max(3, frame_count // 2))

    push("step", f"Analysing {len(best_frames)} frame(s) for the plate…")
    result = detect_plate_multi_frame(best_frames, debug_session_id=cid)

    temp_set(cid, "plate_number", result.plate_number or "")
    temp_set(cid, "plate_result", result.to_dict())
    # The full object, so the attribute stage can reuse the plate BOX for
    # the containment linkage without re-running plate detection.
    temp_set(cid, "plate_result_object", result)
    temp_set(cid, "plate_source", "auto" if result.plate_number else "")
    temp_set(cid, "plate_confidence", result.overall_confidence)

    api = result.to_api_dict()
    if result.status == "CONFIRMED":
        push("ok", f"Plate detected: {result.display_plate} "
                    f"({result.consensus_count}/{result.total_frames} frames agree)", **api)
    elif result.status == "LOW_CONFIDENCE":
        push("warning", f"Low-confidence plate reading: {result.display_plate} — "
                         f"please confirm or correct it", **api)
    else:
        push("warning", "No plate detected — enter manually", **api)
    return api


def run_vehicle_auto_capture(cid: str, on_complete):
    """
    Auto-detect a vehicle in the live camera feed using YOLO, then capture a
    short burst of frames and run the ANPR pipeline across them.
    Mirrors how run_face_capture automatically captures when a face is detected.
    """
    def push(t, txt, **kw):
        _push_vehicle(cid, t, txt, **kw)

    if not OPENCV_AVAILABLE:
        push("error", "OpenCV not installed")
        on_complete(None); end_vehicle_session(cid); return

    if not fm.ensure_camera_running():
        push("error", "Camera unavailable — check connection")
        on_complete(None); end_vehicle_session(cid); return

    if not get_pipeline().detector.is_ready():
        push("warning", "Nigerian plate model not available — capture will still "
                         "work, plate must be entered manually")

    vehicle_model = _get_vehicle_yolo()
    if vehicle_model is None:
        push("warning", "YOLO not available — capturing frame directly")

    push("step", "Scanning for vehicle…")
    imgsz = vehicle_detection_imgsz()

    CONSEC_NEEDED = 3       # consecutive detections before auto-capture
    CONF_THRESH   = 0.45    # vehicle detection confidence threshold
    CHECK_INTERVAL = 0.20   # seconds between frame checks
    TIMEOUT        = 30.0   # seconds before giving up

    detected_count = 0
    deadline       = time.time() + TIMEOUT
    captured_bgr   = None

    while time.time() < deadline:
        with fm._cam_lock:
            raw = fm._cam_state.get("raw")

        if raw is None:
            time.sleep(0.1)
            continue

        # fm._cam_state["raw"] is already BGR (a raw copy of the OpenCV
        # capture frame) — do not re-convert it, that swaps R/B channels.
        bgr = raw

        if vehicle_model is not None:
            try:
                results = vehicle_model(
                    bgr, classes=_VEHICLE_CLASSES, conf=CONF_THRESH, imgsz=imgsz,
                    verbose=False
                )[0]
                boxes = results.boxes
                if boxes is not None and len(boxes) > 0:
                    best_conf = float(boxes.conf.max().cpu())
                    detected_count += 1
                    push("waiting",
                         f"Vehicle detected — hold still… ({detected_count}/{CONSEC_NEEDED})")
                    if detected_count >= CONSEC_NEEDED:
                        captured_bgr = bgr.copy()
                        break
                else:
                    if detected_count > 0:
                        push("waiting", "Scanning for vehicle…")
                    detected_count = 0
            except Exception as e:
                push("info", f"Detection error: {e}")
                detected_count = 0
        else:
            # No YOLO — wait briefly then capture whatever is in frame
            time.sleep(2.0)
            with fm._cam_lock:
                raw = fm._cam_state.get("raw")
            if raw is not None:
                captured_bgr = raw.copy()
            break

        time.sleep(CHECK_INTERVAL)

    if captured_bgr is None:
        push("error", "No vehicle detected — try again or capture manually")
        on_complete(None); end_vehicle_session(cid); return

    # Burst-capture several frames while the vehicle is stable — a single
    # frame is never enough to trust a plate reading on.
    c = cfg.load()
    push("step", "Capturing frame burst…")
    frames = _capture_plate_frame_burst(c.get("plate_capture_frame_count", 10))
    if not frames:
        frames = [captured_bgr]

    # Save the sharpest frame as the vehicle photo shown in the UI.
    best_for_photo = _anpr_pp.select_best_frames(frames, count=1)
    photo_frame = best_for_photo[0] if best_for_photo else captured_bgr
    path = os.path.join(GATE_PHOTOS_DIR, f"vehicle_{cid}.jpg")
    cv2.imwrite(path, photo_frame)
    temp_set(cid, "vehicle_photo", path)
    temp_set(cid, "vehicle_image_source", "camera")
    push("ok", "Vehicle captured ✓")

    api = _run_plate_recognition(cid, push, frames)
    # Attribute recognition runs after ANPR and can never block it: a
    # failure here still leaves the plate result and manual entry intact.
    attributes = _run_vehicle_attributes(cid, push, frames,
                                         temp_get(cid).get("plate_result_object"))

    push("success", "Analysis complete ✓",
         vehicle_url=f"/gate/photo/vehicle_{cid}.jpg",
         vehicle_attributes=attributes,
         **api)
    on_complete(temp_get(cid))
    end_vehicle_session(cid)


def run_vehicle_capture(cid: str, on_complete):
    """Step 1 analysis: read the already-snapped vehicle photo → ANPR + colour via SSE."""
    def push(t, txt, **kw):
        _push_vehicle(cid, t, txt, **kw)

    if not OPENCV_AVAILABLE:
        push("error", "OpenCV not installed")
        on_complete(None); end_vehicle_session(cid); return

    path = temp_get(cid).get("vehicle_photo")
    if not path or not os.path.exists(path):
        push("error", "No vehicle photo — take a photo first")
        on_complete(None); end_vehicle_session(cid); return

    bgr = cv2.imread(path)
    if bgr is None:
        push("error", "Could not read vehicle photo file")
        on_complete(None); end_vehicle_session(cid); return

    if not get_pipeline().detector.is_ready():
        push("warning", "Nigerian plate model not available — plate must be entered manually")

    push("step", "Detecting license plate…")
    # Only one still photo is available in this flow (no burst) — the
    # pipeline will correctly report LOW_CONFIDENCE rather than CONFIRMED,
    # since a single frame can never meet the multi-frame consensus bar.
    api = _run_plate_recognition(cid, push, [bgr])
    attributes = _run_vehicle_attributes(cid, push, [bgr],
                                         temp_get(cid).get("plate_result_object"))

    push("success", "Analysis complete ✓",
         vehicle_url=f"/gate/photo/vehicle_{cid}.jpg",
         vehicle_attributes=attributes,
         **api)
    on_complete(temp_get(cid))
    end_vehicle_session(cid)


# ── FACE CAPTURE (STEP 2: encoding + cropped photo) ──────────────────────────
_face_sessions: dict = {}
_fss_lock = threading.Lock()


def start_face_session(cid: str) -> bool:
    with _fss_lock:
        if cid in _face_sessions:
            return False
        _face_sessions[cid] = queue.Queue()
    return True


def get_face_queue(cid: str):
    return _face_sessions.get(cid)


def end_face_session(cid: str):
    with _fss_lock:
        _face_sessions.pop(cid, None)


def _push_face_cap(cid: str, t: str, txt: str, **kw):
    q = _face_sessions.get(cid)
    if q:
        q.put({"type": t, "text": txt, **kw})


def run_face_capture(cid: str, on_complete, holders_with_photos=None):
    """Step 2: grab frame → SFace embedding + cropped photo + optional DB lookup."""
    def push(t, txt, **kw):
        _push_face_cap(cid, t, txt, **kw)

    def finish(result):
        fm.stop_camera()
        on_complete(result)
        end_face_session(cid)

    # A new capture replaces any earlier identity result for this capture;
    # a failed one leaves none (-> identity unresolved, never "guest").
    temp_set(cid, "identity_face", None)

    if not OPENCV_AVAILABLE:
        push("error", "OpenCV not available")
        finish(None); return

    if not fm.FACE_CV_AVAILABLE:
        push("error", "OpenCV YuNet/SFace not available (need OpenCV 4.5.4+)")
        finish(None); return

    # Ensure face models are loaded (download if missing)
    det, rec = fm._get_face_cv_models()
    if det is None or rec is None:
        push("step", "Downloading face models (first-time setup ~37 MB)…")
        fm.download_face_models(push_fn=push)
        det, rec = fm._get_face_cv_models()
        if det is None:
            push("error", "Face models unavailable — place ONNX files in models/")
            finish(None); return

    if not fm.ensure_camera_running():
        push("error", "Camera unavailable — check connection")
        finish(None); return

    push("step", "Capturing driver face…")
    time.sleep(1.2)
    fm.request_capture()
    rgb = fm.get_captured_frame(timeout=12)
    if rgb is None:
        push("error", "No frame received from camera")
        finish(None); return

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    push("step", "Detecting face with YuNet…")
    emb, face_box = fm.get_face_embedding(bgr)
    if emb is None:
        push("error", "No face detected — ensure driver is looking at camera")
        finish(None); return

    # Crop face from YuNet box: [x, y, w, h, ...]
    x, y, fw, fh_b = int(face_box[0]), int(face_box[1]), int(face_box[2]), int(face_box[3])
    pad = 35
    ih, iw = bgr.shape[:2]
    ct = max(0, y - pad); cl = max(0, x - pad)
    cb = min(ih, y + fh_b + pad); cr = min(iw, x + fw + pad)
    face_bgr = bgr[ct:cb, cl:cr]

    f_path = os.path.join(GATE_PHOTOS_DIR, f"face_{cid}.jpg")
    cv2.imwrite(f_path, face_bgr)
    temp_set(cid, "face_photo", f_path)
    temp_set(cid, "face_encoding", emb.tolist())

    # ── DB face lookup: compare live embedding against all registered passport photos ──
    # The outcome is recorded server-side as this capture's face identity
    # result; /gate/entry/confirm reads it from here, never from the browser.
    if holders_with_photos is not None:
        push("step", f"Searching database ({len(holders_with_photos)} record(s))…")
        best_holder, best_dist, compared = None, 1.0, 0
        for holder in holders_with_photos:
            photo_path = holder.get("photo_path", "")
            if not photo_path or not os.path.exists(photo_path):
                continue
            ref_bgr = cv2.imread(photo_path)
            if ref_bgr is None:
                continue
            ref_emb, _ = fm.get_face_embedding(ref_bgr)
            if ref_emb is None:
                continue
            compared += 1
            dist = fm.face_distance(ref_emb, emb)
            if dist < best_dist:
                best_dist = dist
                best_holder = holder

        if best_holder and best_dist <= fm.FACE_CV_THRESHOLD:
            temp_set(cid, "identity_face", {
                "result": "match", "holder_id": best_holder.get("holder_id"),
                "holder_uid": best_holder.get("holder_uid"),
                "distance": round(float(best_dist), 4)})
        elif holders_with_photos and compared == 0:
            # Registered photos exist but none could be compared: a failed
            # search, not evidence that the driver is unregistered.
            temp_set(cid, "identity_face", {"result": "error",
                                            "reason": "no registered photo could be compared"})
        else:
            temp_set(cid, "identity_face", {"result": "no_match", "compared": compared})

        if best_holder and best_dist <= fm.FACE_CV_THRESHOLD:
            confidence = round((1.0 - best_dist) * 100, 1)
            pp = best_holder.get("photo_path", "")
            push("db_match",
                 f"Record found: {best_holder['name']} (confidence {confidence}%)",
                 holder_id=best_holder.get("holder_id"),
                 name=best_holder.get("name", ""),
                 confidence=confidence,
                 date_of_birth=best_holder.get("date_of_birth"),
                 blood_group=best_holder.get("blood_group"),
                 license_number=best_holder.get("license_number"),
                 license_class=best_holder.get("license_class"),
                 expiry_date=best_holder.get("expiry_date"),
                 photo_url=(f"/photos/{os.path.basename(pp)}" if pp else None),
                 match_method="face")
        elif holders_with_photos and compared == 0:
            push("warning", "Could not compare against any registered photo — "
                            "identity unresolved")
        else:
            conf_pct = round((1.0 - best_dist) * 100, 1) if best_dist < 1.0 else 0
            push("db_notfound",
                 f"No matching record found in database (best confidence: {conf_pct}%)")

    push("success", "Driver face captured ✓",
         face_url=f"/gate/photo/face_{cid}.jpg")
    finish(temp_get(cid))


# ── EXIT VERIFY SESSION (SSE) ─────────────────────────────────────────────────
_exit_sessions: dict = {}
_ev_lock = threading.Lock()


def start_exit_session(trip_id: int) -> bool:
    with _ev_lock:
        if trip_id in _exit_sessions:
            return False
        _exit_sessions[trip_id] = queue.Queue()
    return True


def get_exit_queue(trip_id: int):
    return _exit_sessions.get(trip_id)


def end_exit_session(trip_id: int):
    with _ev_lock:
        _exit_sessions.pop(trip_id, None)


def _push_exit(trip_id: int, t: str, txt: str, **kw):
    q = _exit_sessions.get(trip_id)
    if q:
        q.put({"type": t, "text": txt, **kw})


# ── EXIT AUTHORIZATION ────────────────────────────────────────────────────────
# Exit authorizations are not kept in memory here. The face/fingerprint
# comparisons below hand their result to `on_result`, which the app wires to
# gate_verification.record_biometric_result(): it stores the comparison and,
# for a match against THIS trip's own entry capture, writes the exit
# authorization to the database for the gate session that started the scan.
# on_result runs before the final SSE message, so the page never asks to
# close the trip before the authorization exists.


def _report(on_result, result) -> bool:
    """Call on_result(result); True only when it authorized the exit."""
    if on_result is None:
        return False
    try:
        return bool(on_result(result))
    except Exception as exc:
        print(f"[gate_manager] Could not record the exit result: {type(exc).__name__}")
        return False


def run_exit_verify(trip_id: int, stored_encoding: list, on_complete, on_result=None):
    """
    Capture live frame → SFace compare against THIS trip's entry capture →
    on_result(result) (records it; a match authorizes the exit) → final SSE
    message → on_complete(result). Camera is started here and stopped when done.
    """
    def push(t, txt, **kw):
        _push_exit(trip_id, t, txt, **kw)

    def finish(result):
        fm.stop_camera()
        on_complete(result)
        end_exit_session(trip_id)

    if not OPENCV_AVAILABLE:
        push("error", "OpenCV not available")
        finish(None); return

    if not fm.ensure_camera_running():
        push("error", "Camera unavailable")
        finish(None); return

    push("step", "Capturing exit frame…")
    time.sleep(1.2)
    fm.request_capture()
    rgb = fm.get_captured_frame(timeout=12)
    if rgb is None:
        push("error", "No frame received from camera")
        finish(None); return

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    exit_photo = os.path.join(GATE_PHOTOS_DIR,
                              f"exit_{trip_id}_{uuid.uuid4().hex[:6]}.jpg")
    cv2.imwrite(exit_photo, bgr)

    push("step", "Detecting driver face with YuNet…")
    live_emb, _ = fm.get_face_embedding(bgr)
    if live_emb is None:
        result = {"error": "no_face", "exit_photo": exit_photo, "decision": "NO_FACE"}
        _report(on_result, result)
        push("error", "No face detected — ask driver to look at camera")
        finish(result)
        return

    dist = fm.face_distance(stored_encoding, live_emb)
    confidence = round((1.0 - dist) * 100, 1)
    face_match = dist <= FACE_TOLERANCE

    push("ok" if face_match else "warning",
         f"Face {'MATCH' if face_match else 'MISMATCH'} — "
         f"confidence {confidence}% (distance {round(dist, 3)})",
         distance=round(dist, 3), confidence=confidence)

    decision = "MATCH" if face_match else "MISMATCH"
    exit_photo_url = "/gate/photo/" + os.path.basename(exit_photo)
    result = {
        "face_match": face_match,
        "face_distance": round(dist, 3),
        "confidence": confidence,
        "exit_photo": exit_photo,
        "exit_photo_url": exit_photo_url,
        "decision": decision,
    }

    authorized = _report(on_result, result)
    if face_match and authorized:
        text = "Face matches this trip's entry capture — exit authorized ✓"
    elif face_match:
        text = ("Face matches, but the exit could not be authorized "
                "(the trip may already be closed).")
    else:
        text = ("Face does not match this trip's entry capture — try the fingerprint "
                "or the fallback.")
    push("success" if face_match else "denied", text,
         decision=decision, authorized=authorized, exit_photo_url=exit_photo_url,
         confidence=confidence, face_distance=round(dist, 3))
    finish(result)


# ── FINGERPRINT CAPTURE (ENTRY) ───────────────────────────────────────────────
_ENTRY_FP_KEY = "__gate_entry_fp__"


def start_fp_entry(capture_id: str) -> bool:
    return fp_mgr.start_session(_ENTRY_FP_KEY + capture_id)


def get_fp_entry_queue(capture_id: str):
    return fp_mgr.get_session_queue(_ENTRY_FP_KEY + capture_id)


def cancel_fp_entry(capture_id: str):
    fp_mgr.end_session(_ENTRY_FP_KEY + capture_id)


def run_fp_entry(capture_id: str, on_complete, all_templates=None):
    key = _ENTRY_FP_KEY + capture_id
    # Recorded only when the whole capture succeeds (see below).
    temp_set(capture_id, "identity_fp", None)
    identity = None

    def push(t, txt, **kw):
        q = fp_mgr.get_session_queue(key)
        if q:
            q.put({"type": t, "text": txt, **kw})

    sensor, err = fp_mgr._get_sensor()
    if not sensor:
        push("error", f"Sensor unavailable: {err}")
        on_complete(None)
        fp_mgr.end_session(key)
        return

    # Flush stale bytes left over from any previous session
    try:
        if fp_mgr._uart:
            fp_mgr._uart.reset_input_buffer()
            fp_mgr._uart.reset_output_buffer()
            time.sleep(0.15)
    except Exception:
        pass

    try:
        push("step", "Place finger on sensor…")
        deadline = time.time() + 30
        while True:
            if time.time() > deadline:
                push("error", "Timed out waiting for finger")
                on_complete(None)
                return
            i = sensor.get_image()
            if i == adafruit_fingerprint.OK:
                push("ok", "Fingerprint image captured ✓")
                break
            elif i == adafruit_fingerprint.NOFINGER:
                push("waiting", "Waiting…")
                time.sleep(0.5)
            else:
                push("error", "Imaging error — try again")
                on_complete(None)
                return

        if sensor.image_2_tz(1) != adafruit_fingerprint.OK:
            push("error", "Could not process fingerprint")
            on_complete(None)
            return

        # ── DB lookup FIRST while live template is still in CharBuffer1 ──────
        # get_fpdata() communicates with the sensor and can disturb CharBuffer1,
        # so we do all comparisons before downloading the template to Python.
        if all_templates is not None:
            push("step", f"Searching database ({len(all_templates)} record(s))…")
            best_entry, best_score, compared = None, 0, 0
            for entry in all_templates:
                try:
                    if fp_mgr._uart:
                        fp_mgr._uart.reset_input_buffer()
                        fp_mgr._uart.reset_output_buffer()
                        time.sleep(0.1)
                    sensor.send_fpdata(entry["template"], sensorbuffer="char", slot=2)
                    time.sleep(0.2)
                    sensor.compare_templates()
                    time.sleep(0.1)
                    raw = sensor.confidence
                    score = raw[0] if isinstance(raw, tuple) else (raw if raw is not None else 0)
                    score = int(score)
                    compared += 1
                    push("info", f"Checked {entry.get('name','?')}: score {score}")
                    if score > best_score:
                        best_score = score
                        best_entry = entry
                except Exception as ex:
                    push("warning", f"Skipped {entry.get('name','?')}: {ex}")
                    continue

            if best_entry and best_score >= fp_mgr.CONFIDENCE_THRESHOLD:
                identity = {"result": "match", "holder_id": best_entry.get("holder_id"),
                            "holder_uid": best_entry.get("holder_uid"), "score": best_score}
            elif all_templates and compared == 0:
                identity = {"result": "error",
                            "reason": "no enrolled fingerprint could be compared"}
            else:
                identity = {"result": "no_match", "compared": compared}

            if best_entry and best_score >= fp_mgr.CONFIDENCE_THRESHOLD:
                push("db_match",
                     f"Record found: {best_entry['name']} (score: {best_score})",
                     holder_id=best_entry.get("holder_id"),
                     name=best_entry.get("name", ""),
                     score=best_score,
                     date_of_birth=best_entry.get("date_of_birth"),
                     blood_group=best_entry.get("blood_group"),
                     license_number=best_entry.get("license_number"),
                     license_class=best_entry.get("license_class"),
                     expiry_date=best_entry.get("expiry_date"),
                     photo_url=best_entry.get("photo_url"),
                     match_method="fingerprint")
            else:
                push("db_notfound",
                     f"No matching record found in database (best score: {best_score})")

        # ── Download template AFTER comparisons are done ─────────────────────
        data = sensor.get_fpdata(sensorbuffer="char", slot=1)
        if not data:
            push("error", "Failed to read template from sensor")
            on_complete(None)
            return

        template = list(data)
        temp_set(capture_id, "fingerprint_template", template)
        temp_set(capture_id, "identity_fp", identity)

        push("success", f"Fingerprint captured ({len(template)} bytes) ✓")
        on_complete(template)

    except Exception as exc:
        push("error", f"Sensor error — please try again ({exc})")
        on_complete(None)
    finally:
        fp_mgr.end_session(key)


# ── FINGERPRINT VERIFY (EXIT) ────────────────────────────────────────────────
_EXIT_FP_KEY = "__gate_exit_fp__"


def start_fp_exit(trip_id: int) -> bool:
    return fp_mgr.start_session(_EXIT_FP_KEY + str(trip_id))


def get_fp_exit_queue(trip_id: int):
    return fp_mgr.get_session_queue(_EXIT_FP_KEY + str(trip_id))


def cancel_fp_exit(trip_id: int):
    fp_mgr.end_session(_EXIT_FP_KEY + str(trip_id))


def run_fp_exit(trip_id: int, stored_template: list, on_complete, on_result=None):
    """
    Verify the fingerprint at exit against THIS trip's stored entry
    template. on_result(result) records it — a match authorizes the exit —
    before the final SSE message.
    """
    key = _EXIT_FP_KEY + str(trip_id)

    def push(t, txt, **kw):
        q = fp_mgr.get_session_queue(key)
        if q:
            q.put({"type": t, "text": txt, **kw})

    sensor, err = fp_mgr._get_sensor()
    if not sensor:
        push("error", f"Sensor unavailable: {err}")
        on_complete(None); fp_mgr.end_session(key); return

    push("step", "Place finger on sensor to verify…")
    deadline = time.time() + 30
    while True:
        if time.time() > deadline:
            push("error", "Timed out waiting for fingerprint")
            on_complete(None); fp_mgr.end_session(key); return
        i = sensor.get_image()
        if i == adafruit_fingerprint.OK:
            push("ok", "Fingerprint image captured ✓"); break
        elif i == adafruit_fingerprint.NOFINGER:
            push("waiting", "Waiting for finger…"); time.sleep(0.5)
        else:
            push("error", "Imaging error — try again")
            on_complete(None); fp_mgr.end_session(key); return

    if sensor.image_2_tz(1) != adafruit_fingerprint.OK:
        push("error", "Could not process fingerprint")
        on_complete(None); fp_mgr.end_session(key); return

    push("step", "Comparing with entry fingerprint…")
    try:
        sensor.send_fpdata(stored_template, sensorbuffer="char", slot=2)
        time.sleep(0.2)
        sensor.compare_templates()
        raw = sensor.confidence
        score = raw[0] if isinstance(raw, tuple) else (raw if raw is not None else 0)
        score = int(score)
    except Exception as e:
        push("error", f"Comparison error: {e}")
        on_complete(None); fp_mgr.end_session(key); return

    match = score >= FP_THRESHOLD
    result = {"match": match, "score": score}
    authorized = _report(on_result, result)
    if match and authorized:
        push("success", f"Fingerprint MATCH — score {score} ✓ exit authorized",
             authorized=True)
    elif match:
        push("success", f"Fingerprint MATCH — score {score}, but the exit could not be "
                        f"authorized (the trip may already be closed).", authorized=False)
    else:
        push("notfound", f"Fingerprint MISMATCH — score {score} (need ≥ {FP_THRESHOLD})",
             authorized=False)

    on_complete(result)
    fp_mgr.end_session(key)


# ── VEHICLE PRESENCE DETECTION ────────────────────────────────────────────────
# MOG2 background subtraction: detects when a vehicle fills the gate frame.
# Broadcasts events to all subscribed SSE clients.

COVERAGE_ENTER = 0.15   # fraction of frame that must be foreground → vehicle present
COVERAGE_CLEAR = 0.05   # fraction below which → vehicle gone
STABLE_SECS    = 1.5    # seconds of stable reading before emitting event
DETECT_FPS     = 8      # detection loop frequency
WARMUP_SECS    = 3.0    # seconds to build background model before emitting events

_detect_running  = threading.Event()
_detect_thread   = None
_detect_lock     = threading.Lock()
_detect_state    = {"event": "clear", "coverage": 0.0}

_detect_subs     = []
_detect_sub_lock = threading.Lock()


def _broadcast_detect(event: dict):
    with _detect_sub_lock:
        dead = []
        for q in _detect_subs:
            try:
                q.put_nowait(event)
            except queue.Full:
                dead.append(q)
        for d in dead:
            try:
                _detect_subs.remove(d)
            except ValueError:
                pass


def subscribe_detect() -> queue.Queue:
    q = queue.Queue(maxsize=20)
    with _detect_sub_lock:
        _detect_subs.append(q)
    return q


def unsubscribe_detect(q):
    with _detect_sub_lock:
        try:
            _detect_subs.remove(q)
        except ValueError:
            pass


def get_detect_state() -> dict:
    return dict(_detect_state)


def ensure_detection_running():
    """Start the vehicle detection worker if not already running."""
    global _detect_thread
    with _detect_lock:
        if _detect_running.is_set() and _detect_thread and _detect_thread.is_alive():
            return
        _detect_running.set()
        _detect_thread = threading.Thread(
            target=_detection_worker, daemon=True, name="VehicleDetect"
        )
        _detect_thread.start()


def _detection_worker():
    if not OPENCV_AVAILABLE or not NUMPY_AVAILABLE:
        _detect_running.clear()
        return

    sub = cv2.createBackgroundSubtractorMOG2(
        history=400, varThreshold=40, detectShadows=False
    )
    morph_k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    # Warmup: build background model without emitting events
    warmup_end = time.time() + WARMUP_SECS
    while time.time() < warmup_end and _detect_running.is_set():
        with fm._cam_lock:
            frame = fm._cam_state.get("frame")
        if frame is not None:
            small = cv2.resize(frame, (160, 120))
            sub.apply(small)
        time.sleep(0.12)

    state        = "clear"
    stable_since = None
    interval     = 1.0 / DETECT_FPS

    while _detect_running.is_set():
        with fm._cam_lock:
            frame = fm._cam_state.get("frame")

        if frame is None:
            time.sleep(0.1)
            continue

        small    = cv2.resize(frame, (160, 120))
        fg       = sub.apply(small)
        fg       = cv2.morphologyEx(fg, cv2.MORPH_OPEN,  morph_k)
        fg       = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, morph_k)
        coverage = float(np.count_nonzero(fg)) / fg.size

        _detect_state["coverage"] = round(coverage, 3)

        now = time.time()

        if state == "clear":
            if coverage >= COVERAGE_ENTER:
                if stable_since is None:
                    stable_since = now
                elif now - stable_since >= STABLE_SECS:
                    state        = "present"
                    stable_since = None
                    _detect_state["event"] = "present"
                    _broadcast_detect({
                        "event":    "vehicle_detected",
                        "coverage": round(coverage, 2),
                    })
            else:
                stable_since = None

        elif state == "present":
            if coverage < COVERAGE_CLEAR:
                if stable_since is None:
                    stable_since = now
                elif now - stable_since >= STABLE_SECS:
                    state        = "clear"
                    stable_since = None
                    _detect_state["event"] = "clear"
                    _broadcast_detect({"event": "vehicle_cleared"})
            else:
                stable_since = None

        time.sleep(interval)
