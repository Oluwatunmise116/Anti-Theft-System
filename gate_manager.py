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

GATE_PHOTOS_DIR   = "gate_photos"
os.makedirs(GATE_PHOTOS_DIR, exist_ok=True)

FACE_TOLERANCE  = 0.60   # SFace cosine distance; same-person threshold ~0.637
FP_THRESHOLD    = 30
COLOR_TOLERANCE = 0.35

# ── YOLO PLATE MODEL ──────────────────────────────────────────────────────────

_PLATE_MODEL_PATH = os.path.join("models", "license_plate_detector.pt")
# Mirror URLs for the YOLOv8 license plate detector (tried in order)
_PLATE_MODEL_URLS = [
    # Same model & filename as the computervisioneng/automatic-number-plate-recognition repo
    "https://raw.githubusercontent.com/Muhammad-Zeerak-Khan/"
    "Automatic-License-Plate-Recognition-using-YOLOv8/main/license_plate_detector.pt",
    # Pi-optimised variant
    "https://raw.githubusercontent.com/ahasera/"
    "alpr-YOLOv8-YOLOv4Tiny/main/models/yolov8/best.pt",
    # Additional mirror
    "https://raw.githubusercontent.com/Arijit1080/"
    "Licence-Plate-Detection-using-YOLO-V8/main/best.pt",
]

_plate_yolo       = None
_plate_yolo_lock  = threading.Lock()


def _get_plate_yolo():
    global _plate_yolo
    if _plate_yolo is not None:
        return _plate_yolo
    if not YOLO_AVAILABLE or not os.path.exists(_PLATE_MODEL_PATH):
        return None
    with _plate_yolo_lock:
        if _plate_yolo is not None:
            return _plate_yolo
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _plate_yolo = _YOLO_CLASS(_PLATE_MODEL_PATH)
        except Exception:
            pass
    return _plate_yolo


def download_plate_model(push_fn=None) -> bool:
    """Download the YOLO plate model if not already present. Returns True when ready."""
    if os.path.exists(_PLATE_MODEL_PATH):
        return True
    if not YOLO_AVAILABLE:
        if push_fn:
            push_fn("warning", "ultralytics not installed — run: pip install ultralytics")
        return False
    os.makedirs("models", exist_ok=True)
    if push_fn:
        push_fn("step", "Downloading plate detection model (~6 MB)…")
    import urllib.request
    tmp = _PLATE_MODEL_PATH + ".tmp"
    for url in _PLATE_MODEL_URLS:
        try:
            urllib.request.urlretrieve(url, tmp)
            os.rename(tmp, _PLATE_MODEL_PATH)
            if push_fn:
                push_fn("ok", "Plate model downloaded ✓")
            return True
        except Exception as e:
            if push_fn:
                push_fn("info", f"Mirror failed ({e}) — trying next…")
    if push_fn:
        push_fn("warning", "All mirrors failed — plate detection will use OCR fallback")
    return False


def plate_model_status() -> dict:
    return {
        "yolo_available": YOLO_AVAILABLE,
        "model_present":  os.path.exists(_PLATE_MODEL_PATH),
        "model_loaded":   _plate_yolo is not None,
        "model_path":     _PLATE_MODEL_PATH,
    }


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


def detect_vehicle_crop(frame_bgr):
    """
    Run YOLOv8n to find the largest vehicle in frame.
    Returns (crop_bgr, (x1,y1,x2,y2)) or (frame_bgr, None) if nothing detected.
    """
    model = _get_vehicle_yolo()
    if model is None or not NUMPY_AVAILABLE:
        return frame_bgr, None
    h, w = frame_bgr.shape[:2]
    try:
        results = model(frame_bgr, classes=_VEHICLE_CLASSES, conf=0.25, verbose=False)[0]
        if results.boxes is None or len(results.boxes) == 0:
            return frame_bgr, None
        boxes = results.boxes
        areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes.xyxy.cpu().numpy()]
        best  = int(np.argmax(areas))
        x1, y1, x2, y2 = boxes.xyxy[best].cpu().numpy().astype(int)
        pad = 4
        x1c = max(0, x1 - pad); y1c = max(0, y1 - pad)
        x2c = min(w, x2 + pad); y2c = min(h, y2 + pad)
        return frame_bgr[y1c:y2c, x1c:x2c], (x1, y1, x2, y2)
    except Exception:
        return frame_bgr, None


# ── ANPR ──────────────────────────────────────────────────────────────────────

_ocr_reader = None
_ocr_lock   = threading.Lock()
# Nigerian plate format: 2-3 letters + 2-4 digits + 1-3 letters (e.g. KJA456GH, LSD123AB)
_PLATE_RE   = re.compile(r'^[A-Z]{1,3}[0-9]{2,4}[A-Z]{1,3}$')
# Words that appear in camera overlay text and must never be returned as plates
_OVERLAY_WORDS = frozenset([
    'FACE', 'DETECTED', 'CAPTURING', 'POSITION', 'HOLD', 'STILL',
    'FRAME', 'CAMERA', 'WAITING', 'INSIDE', 'DENIED', 'GRANTED',
])

# OCR character correction maps (from util.py in the referenced repo, adapted for Nigerian plates)
_INT_TO_CHAR = {'0': 'O', '1': 'I', '5': 'S', '6': 'G', '8': 'B', '3': 'J', '4': 'A', '2': 'Z'}
_CHAR_TO_INT = {'O': '0', 'I': '1', 'S': '5', 'G': '6', 'B': '8', 'J': '3', 'A': '4', 'Z': '2'}

# Known Nigerian plate structures (first_letters, digits, last_letters) ordered by frequency
_NG_STRUCTURES = [
    (3, 3, 2),   # 8 chars: AAA-201-KJ  ← most common (Lagos, Abuja, …)
    (2, 3, 2),   # 7 chars: AA-123-BC
    (3, 4, 2),   # 9 chars: AAA-1234-KJ
    (2, 4, 2),   # 8 chars: AA-1234-BC
    (3, 2, 2),   # 7 chars: AAA-12-BC
    (3, 3, 3),   # 9 chars: AAA-201-KJA
    (2, 2, 2),   # 6 chars: AA-12-BC
    (1, 3, 2),   # 6 chars: A-123-BC
    (3, 2, 3),   # 8 chars: AAA-12-BCD
    (2, 3, 3),   # 8 chars: AA-123-BCD
    (2, 2, 3),   # 7 chars: AA-12-BCD
]


def _correct_nigeria_plate(text: str) -> str:
    """
    Template-based OCR correction for Nigerian plates.

    Tries every plausible L/D/L structure and scores each by how many input
    characters are already the right type for their zone — the structure that
    needs the fewest corrections wins.  This handles the common case where OCR
    produces 'Z' or 'O' at digit positions: those are alphabetic, so a
    boundary-scan approach wrongly extends the letter zone over them.
    """
    clean = re.sub(r'[^A-Z0-9]', '', text.upper())
    n = len(clean)
    if n < 5 or n > 10:
        return clean

    best_score  = -9999
    best_result = clean

    for l1, d, l2 in _NG_STRUCTURES:
        if l1 + d + l2 != n:
            continue

        corrected = ''
        for j, c in enumerate(clean):
            if j < l1:
                corrected += _INT_TO_CHAR.get(c, c)   # digit→letter in letter zone
            elif j < l1 + d:
                corrected += _CHAR_TO_INT.get(c, c)   # letter→digit in digit zone
            else:
                corrected += _INT_TO_CHAR.get(c, c)   # digit→letter in letter zone

        # Score: chars already in the right type for their zone cost 3;
        # chars that needed correction cost 1 (both are acceptable, but prefer fewer corrections)
        score = 0
        for j, c_in in enumerate(clean):
            in_letter = (j < l1) or (j >= l1 + d)
            if in_letter:
                score += 3 if c_in.isalpha() else 1
            else:
                score += 3 if c_in.isdigit() else 1

        if _PLATE_RE.match(corrected):
            score += 15   # strong bonus for matching the full Nigerian regex

        if score > best_score:
            best_score  = score
            best_result = corrected

    return best_result


def _get_ocr():
    global _ocr_reader
    with _ocr_lock:
        if _ocr_reader is None:
            try:
                import easyocr
                _ocr_reader = easyocr.Reader(['en'], gpu=False, verbose=False)
            except Exception:
                pass
    return _ocr_reader


def _ocr_plate_crop(reader, bgr_crop) -> list:
    """
    OCR a plate crop. Upscales if small, tries CLAHE + sharpen + Otsu + binary-inverse variants.
    Returns list of (_, text, conf) tuples.
    """
    ch, cw = bgr_crop.shape[:2]
    if cw < 1 or ch < 1:
        return []
    gray = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2GRAY)
    if cw < 320:
        scale = 320 / cw
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    clahe    = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    variants = [enhanced]

    if NUMPY_AVAILABLE:
        k     = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharp = cv2.filter2D(enhanced, -1, k)
        variants.append(sharp)

    _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(thresh)

    # Binary-inverse at fixed threshold (same as reference ANPR repo — dark text on light plate)
    _, thresh_inv = cv2.threshold(gray, 64, 255, cv2.THRESH_BINARY_INV)
    variants.append(thresh_inv)

    out = []
    seen = set()
    for img in variants:
        try:
            for det in reader.readtext(
                img,
                allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-',
                paragraph=False,
            ):
                key = det[1].upper()
                if key not in seen:
                    seen.add(key)
                    out.append(det)
        except Exception:
            pass
    return out


def _ocr_hits_to_candidates(ocr_results, yolo_conf=1.0) -> list:
    """
    Convert raw easyocr results → (score, plate) candidates.
    Applies Nigerian plate correction and format check.
    """
    candidates = []
    for (_, text, ocr_conf) in ocr_results:
        clean = re.sub(r'[^A-Z0-9]', '', text.upper())
        if len(clean) < 5 or len(clean) > 10 or clean in _OVERLAY_WORDS:
            continue
        corrected = _correct_nigeria_plate(clean)
        has_letters = any(c.isalpha() for c in corrected)
        has_digits  = any(c.isdigit() for c in corrected)
        if not (has_letters and has_digits):
            continue
        bonus = 0.5 if _PLATE_RE.match(corrected) else 0.0
        score = (ocr_conf + bonus) * (0.4 + 0.6 * yolo_conf)
        candidates.append((score, corrected))
    return candidates


def detect_plate_verbose(frame_bgr) -> tuple:
    """
    YOLO-first ANPR pipeline for Nigerian plates.

    1. YOLO detects plate bounding boxes → OCR each crop.
       Character-correction maps fix common OCR errors (O↔0, I↔1, S↔5 …).
    2. Falls back to multi-region OCR scan if YOLO not available or finds nothing.

    Returns (plate_string, log_lines).
    """
    if not OPENCV_AVAILABLE:
        return '', ['OpenCV not available']
    reader = _get_ocr()
    if reader is None:
        return '', ['OCR reader unavailable']

    h, w   = frame_bgr.shape[:2]
    hits   = []          # log lines
    cands  = []          # (score, plate) candidates

    # ── PASS 0: vehicle crop (yolov8n) to reduce background noise ────────────
    # Mirrors the reference ANPR repo: detect vehicle body first, then find plate inside it.
    vehicle_crop, vehicle_bbox = detect_vehicle_crop(frame_bgr)
    if vehicle_bbox is not None:
        hits.append(f"Vehicle body detected — searching plate within crop")
    search_frames = []
    if vehicle_bbox is not None:
        search_frames.append(('vehicle_crop', vehicle_crop))
    search_frames.append(('full_frame', frame_bgr))

    # ── PASS 1: YOLO plate detector ──────────────────────────────────────────
    model = _get_plate_yolo()
    if model is not None:
        for frame_label, search_frame in search_frames:
            fh, fw = search_frame.shape[:2]
            try:
                results = model(search_frame, conf=0.20, verbose=False)[0]
                boxes   = results.boxes
                if boxes is not None and len(boxes):
                    order = boxes.conf.cpu().numpy().argsort()[::-1]
                    hits.append(f"YOLO [{frame_label}]: {len(boxes)} plate region(s) detected")
                    for i in order[:5]:
                        x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)
                        yconf = float(boxes.conf[i])
                        pad = 5
                        x1c = max(0, x1-pad); y1c = max(0, y1-pad)
                        x2c = min(fw, x2+pad); y2c = min(fh, y2+pad)
                        crop = search_frame[y1c:y2c, x1c:x2c]
                        ocr_res = _ocr_plate_crop(reader, crop)
                        for (_, text, oconf) in ocr_res:
                            clean = re.sub(r'[^A-Z0-9]', '', text.upper())
                            if clean:
                                hits.append(f"  OCR: {clean!r} conf={oconf:.2f} yolo={yconf:.2f}")
                        cands.extend(_ocr_hits_to_candidates(ocr_res, yolo_conf=yconf))
                else:
                    hits.append(f"YOLO [{frame_label}]: no plate regions found")
            except Exception as e:
                hits.append(f"YOLO error [{frame_label}]: {e}")
            # If we already found strong candidates from vehicle crop, skip full frame
            if frame_label == 'vehicle_crop' and any(s > 0.7 for s, _ in cands):
                hits.append("Strong candidates from vehicle crop — skipping full frame scan")
                break
    else:
        hits.append("YOLO model not loaded — using OCR scan")

    if cands:
        cands.sort(reverse=True)
        return cands[0][1], hits

    # ── PASS 2: multi-region OCR fallback ────────────────────────────────────
    hits.append("Running multi-region OCR scan…")

    def _scan(bgr_crop, label):
        ocr_res = _ocr_plate_crop(reader, bgr_crop)
        for (_, text, conf) in ocr_res:
            clean = re.sub(r'[^A-Z0-9]', '', text.upper())
            if len(clean) >= 4:
                hits.append(f"  [{label}] {clean!r} conf={conf:.2f}")
        cands.extend(_ocr_hits_to_candidates(ocr_res))

    _scan(frame_bgr, 'full')
    _scan(frame_bgr[h // 2:, :], 'lower-half')
    _scan(frame_bgr[int(h * 0.55):int(h * 0.90), :], 'strip')
    _scan(frame_bgr[:h // 2, :], 'upper-half')

    # Contour-based plate region detection
    try:
        gray  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        blur  = cv2.bilateralFilter(gray, 11, 17, 17)
        edges = cv2.Canny(blur, 30, 200)
        cnts, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:25]
        for c in cnts:
            peri   = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.018 * peri, True)
            if len(approx) == 4:
                x, y, bw, bh = cv2.boundingRect(approx)
                ar = bw / bh if bh > 0 else 0
                if 1.8 < ar < 7.0 and bw > 50 and bh > 12:
                    pad = 6
                    crop = frame_bgr[max(0, y-pad):min(h, y+bh+pad),
                                     max(0, x-pad):min(w, x+bw+pad)]
                    _scan(crop, 'contour')
    except Exception:
        pass

    if not cands:
        return '', hits
    cands.sort(reverse=True)
    return cands[0][1], hits


def detect_plate(frame_bgr) -> str:
    plate, _ = detect_plate_verbose(frame_bgr)
    return plate


# ── VEHICLE COLOR ─────────────────────────────────────────────────────────────

_COLOR_RULES = [
    ("white",  lambda h, s, v: s < 40 and v > 180),
    ("black",  lambda h, s, v: v < 60),
    ("silver", lambda h, s, v: s < 50 and 60 <= v <= 180),
    ("red",    lambda h, s, v: (h < 15 or h > 165) and s >= 80),
    ("orange", lambda h, s, v: 15 <= h < 30 and s >= 80),
    ("yellow", lambda h, s, v: 30 <= h < 45 and s >= 80),
    ("green",  lambda h, s, v: 45 <= h < 90 and s >= 60),
    ("blue",   lambda h, s, v: 105 <= h < 135 and s >= 60),
    ("purple", lambda h, s, v: 135 <= h <= 165 and s >= 60),
    ("grey",   lambda h, s, v: True),
]

_COLOR_BADGES = {
    "white": "#f3f4f6", "black": "#1f2937", "silver": "#9ca3af",
    "red": "#ef4444", "orange": "#f97316", "yellow": "#eab308",
    "green": "#22c55e", "blue": "#3b82f6", "purple": "#a855f7",
    "grey": "#6b7280", "unknown": "#d1d5db",
}


def detect_color(frame_bgr) -> tuple:
    """Return (color_name, [h, s, v]) using YOLO vehicle crop + saturation-priority cluster."""
    if not OPENCV_AVAILABLE or not NUMPY_AVAILABLE:
        return "unknown", [0, 0, 0]

    # YOLO vehicle detection → focused crop (skip roof/sky and wheels/road)
    crop, bbox = detect_vehicle_crop(frame_bgr)
    ch, cw = crop.shape[:2]
    roi = crop[int(ch * 0.15):int(ch * 0.80), int(cw * 0.05):int(cw * 0.95)]
    if roi.size == 0:
        roi = crop

    hsv    = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    pixels = hsv.reshape(-1, 3).astype(np.float32)
    if len(pixels) < 10:
        return "unknown", [0, 0, 0]

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 15, 1.0)
    try:
        _, labels, centers = cv2.kmeans(
            pixels, 5, None, criteria, 3, cv2.KMEANS_PP_CENTERS
        )
        counts = np.bincount(labels.flatten(), minlength=5)

        # Prefer the most-populated chromatically rich cluster (paint colour).
        # Large dark areas like grilles, tyres, and shadows are achromatic
        # (low S or low V), so they are skipped in this pass.
        best_hh = best_ss = best_vv = None
        best_count = -1
        for center, count in zip(centers, counts):
            hh, ss, vv = int(center[0]), int(center[1]), int(center[2])
            if ss >= 40 and vv >= 40 and count > best_count:
                best_count           = count
                best_hh, best_ss, best_vv = hh, ss, vv

        if best_hh is None:
            # No saturated cluster — entire scene is achromatic (white/black/silver)
            dom = centers[np.argmax(counts)]
            best_hh, best_ss, best_vv = int(dom[0]), int(dom[1]), int(dom[2])

        hh, ss, vv = best_hh, best_ss, best_vv

    except Exception:
        avg = hsv.mean(axis=(0, 1))
        hh, ss, vv = int(avg[0]), int(avg[1]), int(avg[2])

    for name, test in _COLOR_RULES:
        if test(hh, ss, vv):
            return name, [hh, ss, vv]
    return "grey", [hh, ss, vv]


def color_distance(hsv1: list, hsv2: list) -> float:
    """Normalized HSV distance 0–1."""
    if not hsv1 or not hsv2:
        return 1.0
    dh = min(abs(hsv1[0] - hsv2[0]), 180 - abs(hsv1[0] - hsv2[0])) / 90.0
    ds = abs(hsv1[1] - hsv2[1]) / 255.0
    dv = abs(hsv1[2] - hsv2[2]) / 255.0
    return dh * 0.5 + ds * 0.25 + dv * 0.25


def color_hex(name: str) -> str:
    return _COLOR_BADGES.get(name, "#d1d5db")


# ── TEMP CAPTURE STORE ────────────────────────────────────────────────────────
_temp: dict = {}
_temp_lock  = threading.Lock()


def new_capture_id() -> str:
    cid = uuid.uuid4().hex[:10]
    with _temp_lock:
        _temp[cid] = {}
    return cid


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
    return {"ok": True, "url": f"/gate/photo/vehicle_{cid}.jpg"}


def run_vehicle_auto_capture(cid: str, on_complete):
    """
    Auto-detect a vehicle in the live camera feed using YOLO, then capture and run ANPR.
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

    # Ensure YOLO model is present (downloads ~6 MB on first use)
    if YOLO_AVAILABLE and not os.path.exists(_PLATE_MODEL_PATH):
        download_plate_model(push_fn=push)
        _get_plate_yolo()

    vehicle_model = _get_vehicle_yolo()
    if vehicle_model is None:
        push("warning", "YOLO not available — capturing frame directly")

    push("step", "Scanning for vehicle…")

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

        bgr = cv2.cvtColor(raw, cv2.COLOR_RGB2BGR)

        if vehicle_model is not None:
            try:
                results = vehicle_model(
                    bgr, classes=_VEHICLE_CLASSES, conf=CONF_THRESH, verbose=False
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
                captured_bgr = cv2.cvtColor(raw, cv2.COLOR_RGB2BGR)
            break

        time.sleep(CHECK_INTERVAL)

    if captured_bgr is None:
        push("error", "No vehicle detected — try again or capture manually")
        on_complete(None); end_vehicle_session(cid); return

    # Save the captured frame
    path = os.path.join(GATE_PHOTOS_DIR, f"vehicle_{cid}.jpg")
    cv2.imwrite(path, captured_bgr)
    temp_set(cid, "vehicle_photo", path)
    push("ok", "Vehicle captured ✓")

    # Run ANPR on the captured frame
    push("step", "Detecting license plate…")
    plate, log_lines = detect_plate_verbose(captured_bgr)
    for line in log_lines:
        push("info", line)
    if plate:
        push("ok", f"Plate detected: {plate}", plate=plate)
    else:
        push("warning", "No plate detected — enter manually", plate="")
    temp_set(cid, "plate_number", plate or "")

    push("success", "Analysis complete ✓",
         vehicle_url=f"/gate/photo/vehicle_{cid}.jpg",
         plate=plate or "")
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

    # Ensure YOLO model is present (downloads ~6 MB on first use)
    if YOLO_AVAILABLE and not os.path.exists(_PLATE_MODEL_PATH):
        download_plate_model(push_fn=push)
        _get_plate_yolo()   # load into memory now

    push("step", "Detecting license plate…")
    plate, log_lines = detect_plate_verbose(bgr)
    for line in log_lines:
        push("info", line)
    if plate:
        push("ok", f"Plate detected: {plate}", plate=plate)
    else:
        push("warning", "No plate detected — enter manually", plate="")
    temp_set(cid, "plate_number", plate or "")

    push("success", "Analysis complete ✓",
         vehicle_url=f"/gate/photo/vehicle_{cid}.jpg",
         plate=plate or "")
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
    if holders_with_photos:
        push("step", f"Searching database ({len(holders_with_photos)} record(s))…")
        best_holder, best_dist = None, 1.0
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
            dist = fm.face_distance(ref_emb, emb)
            if dist < best_dist:
                best_dist = dist
                best_holder = holder

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


def run_exit_verify(trip_id: int, stored_encoding: list, on_complete):
    """
    Capture live frame → SFace face compare → on_complete(result).
    Camera is started here and stopped when done.
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
        push("error", "No face detected — ask driver to look at camera")
        finish({"error": "no_face", "exit_photo": exit_photo, "decision": "DENIED"})
        return

    dist = fm.face_distance(stored_encoding, live_emb)
    confidence = round((1.0 - dist) * 100, 1)
    face_match = dist <= FACE_TOLERANCE

    push("ok" if face_match else "warning",
         f"Face {'MATCH' if face_match else 'MISMATCH'} — "
         f"confidence {confidence}% (distance {round(dist, 3)})",
         distance=round(dist, 3), confidence=confidence)

    decision = "GRANTED" if face_match else "DENIED"
    result = {
        "face_match": face_match,
        "face_distance": round(dist, 3),
        "confidence": confidence,
        "exit_photo": exit_photo,
        "exit_photo_url": "/gate/photo/" + os.path.basename(exit_photo),
        "decision": decision,
    }

    push("success" if face_match else "denied",
         f"EXIT {decision}", decision=decision)
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
        if all_templates:
            push("step", f"Searching database ({len(all_templates)} record(s))…")
            best_entry, best_score = None, 0
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
                    push("info", f"Checked {entry.get('name','?')}: score {score}")
                    if score > best_score:
                        best_score = score
                        best_entry = entry
                except Exception as ex:
                    push("warning", f"Skipped {entry.get('name','?')}: {ex}")
                    continue

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


def run_fp_exit(trip_id: int, stored_template: list, on_complete):
    """Verify fingerprint at exit against stored entry template."""
    key = _EXIT_FP_KEY + str(trip_id)

    def push(t, txt):
        fp_mgr._push(key, t, txt)

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
    if match:
        push("success", f"Fingerprint MATCH — score {score} ✓")
    else:
        push("notfound", f"Fingerprint MISMATCH — score {score} (need ≥ {FP_THRESHOLD})")

    on_complete({"match": match, "score": score})
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
