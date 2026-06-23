"""
Face Recognition Manager — persistent camera architecture.

The camera is owned by a single background thread (_camera_worker) that runs
continuously. generate_video_frames() and the search thread both read from a
shared frame buffer — no open/close races between scans.

Face recognition uses OpenCV's built-in YuNet face detector and SFace recogniser
(ArcFace family) — no dlib/face_recognition dependency.
"""
import threading
import queue
import time
import os
import glob
import subprocess
import urllib.request

try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False

# Check for YuNet + SFace support (OpenCV 4.5.4+)
FACE_CV_AVAILABLE = (
    OPENCV_AVAILABLE and
    hasattr(cv2, "FaceDetectorYN") and
    hasattr(cv2, "FaceRecognizerSF")
)

import config as cfg

USB_CAMERA_INDEX   = cfg.get("camera_index", 0)
CAPTURE_DELAY      = 2.5

# SFace cosine distance threshold.
# OpenCV docs: cosine similarity 0.363 → same person.
# We convert: distance = 1 − similarity  →  same person: dist < 0.637.
# Use 0.60 to be slightly conservative.
FACE_CV_THRESHOLD  = 0.60

# ── FACE MODEL PATHS & URLS ───────────────────────────────────────────────────
_FACE_DET_MODEL = os.path.join("models", "face_detection_yunet_2023mar.onnx")
_FACE_REC_MODEL = os.path.join("models", "face_recognition_sface_2021dec.onnx")

# media.githubusercontent.com serves actual Git-LFS binary (not the pointer)
_FACE_DET_URLS = [
    "https://raw.githubusercontent.com/opencv/opencv_zoo/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx",
]
_FACE_REC_URLS = [
    "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/"
    "face_recognition_sface/face_recognition_sface_2021dec.onnx",
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_recognition_sface/face_recognition_sface_2021dec.onnx",
]

_yunet        = None
_sface        = None
_face_cv_lock = threading.Lock()


# ── CAMERA DETECTION ──────────────────────────────────────────────────────────

def _v4l2_device_names() -> dict:
    """Parse v4l2-ctl --list-devices to get device path → friendly name mapping."""
    names = {}
    try:
        out = subprocess.check_output(
            ["v4l2-ctl", "--list-devices"], stderr=subprocess.DEVNULL, timeout=3
        ).decode()
        current_name = "Unknown"
        for line in out.splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith("/dev/video"):
                names[line] = current_name
            elif not line.startswith("/dev/"):
                # Strip trailing " (platform:...)" or "(usb-...)" from name
                current_name = line.split(" (")[0].strip()
    except Exception:
        pass
    return names


def _try_open(index: int) -> bool:
    """Open a camera index, try to read a frame, return True if successful. Suppresses OpenCV stderr."""
    import sys
    # Suppress OpenCV's WARN messages by temporarily redirecting stderr at the fd level
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    old_stderr = os.dup(2)
    os.dup2(devnull_fd, 2)
    try:
        cap = cv2.VideoCapture(index)
        opened = cap.isOpened()
        readable = False
        if opened:
            ret, _ = cap.read()
            readable = ret
        cap.release()
    finally:
        os.dup2(old_stderr, 2)
        os.close(old_stderr)
        os.close(devnull_fd)
    return readable


def list_cameras() -> list:
    """
    Return list of dicts describing available capture-capable cameras:
      {"index": int, "path": str, "name": str, "active": bool}
    Includes OpenCV index 0–9 plus any /dev/video* that open cleanly.
    Filters out Pi ISP backend nodes (pispbe / hevc / isp / codec).
    """
    if not OPENCV_AVAILABLE:
        return []

    v4l2_names = _v4l2_device_names()
    found = {}  # index → info

    # --- Try /dev/video* paths first ---
    for path in sorted(glob.glob("/dev/video*")):
        try:
            idx = int(path.replace("/dev/video", ""))
        except ValueError:
            continue
        name = v4l2_names.get(path, f"Camera {idx}")
        if any(skip in name.lower() for skip in ("pispbe", "hevc", "isp", "codec")):
            continue
        if _try_open(idx) and idx not in found:
            found[idx] = {"index": idx, "path": path, "name": name}

    # --- Also probe low indices (USB cameras on some setups) ---
    for idx in range(10):
        if idx in found:
            continue
        if _try_open(idx):
            path = f"/dev/video{idx}"
            name = v4l2_names.get(path, f"USB Camera (index {idx})")
            found[idx] = {"index": idx, "path": path, "name": name}

    active = USB_CAMERA_INDEX
    result = []
    for idx in sorted(found):
        entry = dict(found[idx])
        entry["active"] = (idx == active)
        result.append(entry)
    return result


def switch_camera(index: int):
    """Change the active camera index, persist to config, restart worker."""
    global USB_CAMERA_INDEX
    USB_CAMERA_INDEX = index
    cfg.save({"camera_index": index})
    if _cam_running.is_set():
        stop_camera()
        ensure_camera_running()

# ── CAMERA WORKER ─────────────────────────────────────────────────────────────
_cam_lock    = threading.Lock()
_cam_running = threading.Event()   # set = worker thread is alive
_cam_thread  = None
_cam_state   = {"frame": None, "raw": None}  # "frame"=annotated for MJPEG, "raw"=clean for ANPR/capture

# ── CAPTURE PIPE ──────────────────────────────────────────────────────────────
_capture_event = threading.Event()        # set = worker should capture next frame
_frame_queue   = queue.Queue(maxsize=1)   # captured RGB frame lands here

# ── SESSION / RESULT ──────────────────────────────────────────────────────────
_sessions     = {}
_session_lock = threading.Lock()
_last_result  = {}
FACE_SEARCH_KEY = "__face_search__"


# ── CAMERA WORKER THREAD ──────────────────────────────────────────────────────

def _camera_worker():
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    cap = cv2.VideoCapture(USB_CAMERA_INDEX)
    if not cap.isOpened():
        _cam_running.clear()
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    for _ in range(5):   # warmup — discard first frames so exposure settles
        cap.read()

    try:
        while _cam_running.is_set():
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.05)
                continue

            raw = frame.copy()   # clean frame before any overlays — used for capture/ANPR

            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
            )

            for (x, y, w, h) in faces:
                col = (201, 168, 76)
                c, t = 18, 2
                cv2.rectangle(frame, (x, y), (x+w, y+h), col, 1)
                cv2.line(frame, (x,     y),     (x+c,   y),     col, t)
                cv2.line(frame, (x,     y),     (x,     y+c),   col, t)
                cv2.line(frame, (x+w,   y),     (x+w-c, y),     col, t)
                cv2.line(frame, (x+w,   y),     (x+w,   y+c),   col, t)
                cv2.line(frame, (x,     y+h),   (x+c,   y+h),   col, t)
                cv2.line(frame, (x,     y+h),   (x,     y+h-c), col, t)
                cv2.line(frame, (x+w,   y+h),   (x+w-c, y+h),   col, t)
                cv2.line(frame, (x+w,   y+h),   (x+w,   y+h-c), col, t)
                cv2.putText(frame, "FACE DETECTED", (x, max(y-8, 14)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)

            capturing = _capture_event.is_set()
            if capturing:
                label, lcol = "CAPTURING...", (76, 175, 130)
            elif len(faces) > 0:
                label, lcol = "FACE IN FRAME - HOLD STILL", (201, 168, 76)
            else:
                label, lcol = "POSITION FACE IN FRAME", (168, 184, 204)

            cv2.putText(frame, label, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, lcol, 2)

            # Capture RGB frame for recognition when requested — always use raw (no overlays)
            if capturing:
                rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
                _capture_event.clear()
                try:
                    _frame_queue.put_nowait(rgb)
                except queue.Full:
                    pass

            with _cam_lock:
                _cam_state["frame"] = frame   # annotated — for MJPEG stream
                _cam_state["raw"]   = raw     # clean — for ANPR / color / face encoding

            time.sleep(0.033)

    finally:
        cap.release()
        with _cam_lock:
            _cam_state["frame"] = None
            _cam_state["raw"]   = None
        _cam_running.clear()


def ensure_camera_running():
    """Start the camera worker if not already running. Returns True when ready."""
    global _cam_thread

    if _cam_running.is_set() and _cam_thread and _cam_thread.is_alive():
        return True

    # Clear any stale captured frames from a previous session
    while not _frame_queue.empty():
        try:
            _frame_queue.get_nowait()
        except queue.Empty:
            break
    _capture_event.clear()

    _cam_running.set()
    _cam_thread = threading.Thread(target=_camera_worker, daemon=True, name="CameraWorker")
    _cam_thread.start()

    # Wait up to 5 s for the first frame (confirms camera opened successfully)
    deadline = time.time() + 5.0
    while time.time() < deadline:
        if not _cam_running.is_set():
            return False   # camera failed to open
        with _cam_lock:
            if _cam_state["frame"] is not None:
                return True
        time.sleep(0.1)

    return _cam_running.is_set()


def stop_camera():
    """Signal the camera worker to stop and wait for it to exit."""
    global _cam_thread
    _cam_running.clear()
    if _cam_thread:
        _cam_thread.join(timeout=3.0)
        _cam_thread = None
    with _cam_lock:
        _cam_state["frame"] = None


def generate_video_frames():
    """Yield MJPEG frames from the running camera worker. Does NOT own the camera."""
    if not OPENCV_AVAILABLE:
        return
    while _cam_running.is_set():
        with _cam_lock:
            frame = _cam_state["frame"]
        if frame is None:
            time.sleep(0.05)
            continue
        ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if not ok:
            continue
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n"
               + jpeg.tobytes()
               + b"\r\n")
        time.sleep(0.033)


# ── SESSION HELPERS ───────────────────────────────────────────────────────────

def start_session():
    with _session_lock:
        if FACE_SEARCH_KEY in _sessions:
            return False
        _sessions[FACE_SEARCH_KEY] = queue.Queue()
    return True


def get_session_queue():
    return _sessions.get(FACE_SEARCH_KEY)


def end_session():
    with _session_lock:
        _sessions.pop(FACE_SEARCH_KEY, None)


def _push(msg_type, text):
    q = _sessions.get(FACE_SEARCH_KEY)
    if q:
        q.put({"type": msg_type, "text": text})


def request_capture():
    while not _frame_queue.empty():
        try:
            _frame_queue.get_nowait()
        except queue.Empty:
            break
    _capture_event.set()


def get_captured_frame(timeout=15):
    try:
        return _frame_queue.get(timeout=timeout)
    except queue.Empty:
        return None


# ── FACE CV MODEL MANAGEMENT ──────────────────────────────────────────────────

def _valid_onnx(path: str, min_bytes: int = 5_000) -> bool:
    """True if file exists, is large enough, and isn't a Git-LFS pointer."""
    if not os.path.exists(path):
        return False
    if os.path.getsize(path) < min_bytes:
        return False
    with open(path, "rb") as f:
        return not f.read(30).startswith(b"version https://git-lfs")


def _get_face_cv_models():
    """Lazy-load YuNet + SFace. Returns (detector, recogniser) or (None, None)."""
    global _yunet, _sface
    if _yunet is not None and _sface is not None:
        return _yunet, _sface
    if not FACE_CV_AVAILABLE:
        return None, None
    if not _valid_onnx(_FACE_DET_MODEL) or not _valid_onnx(_FACE_REC_MODEL, 1_000_000):
        return None, None
    with _face_cv_lock:
        if _yunet is not None and _sface is not None:
            return _yunet, _sface
        try:
            _yunet = cv2.FaceDetectorYN.create(
                _FACE_DET_MODEL, "",
                (320, 320),
                score_threshold=0.85,
                nms_threshold=0.3,
                backend_id=cv2.dnn.DNN_BACKEND_OPENCV,
                target_id=cv2.dnn.DNN_TARGET_CPU,
            )
            _sface = cv2.FaceRecognizerSF.create(
                _FACE_REC_MODEL, "",
                backend_id=cv2.dnn.DNN_BACKEND_OPENCV,
                target_id=cv2.dnn.DNN_TARGET_CPU,
            )
        except Exception as e:
            print(f"[face_manager] Face model load error: {e}")
            _yunet = None
            _sface = None
    return _yunet, _sface


def download_face_models(push_fn=None) -> bool:
    """Download YuNet + SFace ONNX models from OpenCV Zoo. Returns True when both are ready."""
    os.makedirs("models", exist_ok=True)
    ok = True
    specs = [
        (_FACE_DET_URLS, _FACE_DET_MODEL, "face detector (YuNet ~230 KB)", 5_000),
        (_FACE_REC_URLS, _FACE_REC_MODEL, "face recogniser (SFace ~37 MB)", 1_000_000),
    ]
    for urls, path, name, min_bytes in specs:
        if _valid_onnx(path, min_bytes):
            continue
        if push_fn:
            push_fn("step", f"Downloading {name}…")
        downloaded = False
        for url in urls:
            try:
                tmp = path + ".tmp"
                urllib.request.urlretrieve(url, tmp)
                if _valid_onnx(tmp, min_bytes):
                    os.replace(tmp, path)
                    if push_fn:
                        push_fn("ok", f"Downloaded {name} ✓")
                    downloaded = True
                    break
                else:
                    os.remove(tmp)
                    if push_fn:
                        push_fn("info", f"Mirror returned invalid file — trying next…")
            except Exception as e:
                if push_fn:
                    push_fn("info", f"Mirror failed ({e}) — trying next…")
        if not downloaded:
            if push_fn:
                push_fn("warning", f"Could not download {name}. "
                        "Place the ONNX file in the models/ folder manually.")
            ok = False
    return ok


def face_model_status() -> dict:
    return {
        "face_cv_available": FACE_CV_AVAILABLE,
        "yunet_present":     _valid_onnx(_FACE_DET_MODEL),
        "sface_present":     _valid_onnx(_FACE_REC_MODEL, 1_000_000),
        "models_loaded":     _yunet is not None and _sface is not None,
    }


def get_face_embedding(bgr_frame):
    """
    Detect the largest face in bgr_frame and return (embedding, face_box).
    embedding is a (1, 128) numpy array for cosine comparison with SFace.
    Returns (None, None) if no face detected or models unavailable.
    """
    det, rec = _get_face_cv_models()
    if det is None:
        return None, None
    h, w = bgr_frame.shape[:2]
    det.setInputSize((w, h))
    _, faces = det.detect(bgr_frame)
    if faces is None or len(faces) == 0:
        return None, None
    best = max(faces, key=lambda f: float(f[2]) * float(f[3]))
    try:
        aligned = rec.alignCrop(bgr_frame, best)
        emb = rec.feature(aligned)
    except Exception:
        return None, None
    return emb, best


def face_distance(emb1, emb2) -> float:
    """
    Cosine distance between two SFace embeddings: 0.0 = identical, 1.0 = different.
    Accepts numpy arrays or plain lists. Same-person threshold: < 0.637 (we use 0.60).
    """
    if emb1 is None or emb2 is None:
        return 1.0
    _, rec = _get_face_cv_models()
    if rec is None:
        return 1.0
    try:
        import numpy as _np
        e1 = _np.array(emb1, dtype=_np.float32).reshape(1, -1)
        e2 = _np.array(emb2, dtype=_np.float32).reshape(1, -1)
        if e1.shape != e2.shape:
            return 1.0
        score = rec.match(e1, e2, cv2.FaceRecognizerSF_FR_COSINE)
        return float(1.0 - max(0.0, min(1.0, score)))
    except Exception:
        return 1.0


# ── FACE SEARCH ───────────────────────────────────────────────────────────────

def run_face_search(holders_with_photos, on_complete):
    def push(t, txt):
        _push(t, txt)

    if not FACE_CV_AVAILABLE:
        push("error", "OpenCV YuNet/SFace not available (need OpenCV 4.5.4+)")
        on_complete(None); end_session(); return

    if not _cam_running.is_set():
        push("error", "Camera is not running — open the video feed first")
        on_complete(None); end_session(); return

    # Ensure models are present
    det, rec = _get_face_cv_models()
    if det is None or rec is None:
        push("step", "Face models not found — downloading…")
        if not download_face_models(push_fn=push):
            push("error", "Face models unavailable. Place ONNX files in models/ folder.")
            on_complete(None); end_session(); return
        det, rec = _get_face_cv_models()
        if det is None:
            push("error", "Failed to load face models after download.")
            on_complete(None); end_session(); return

    candidates = [h for h in holders_with_photos
                  if h.get("photo_path") and os.path.exists(h["photo_path"])]
    if not candidates:
        push("error", "No passport photos found in database.")
        on_complete(None); end_session(); return

    push("step", f"Loading {len(candidates)} reference photo(s)…")
    reference_encodings = []
    for h in candidates:
        bgr = cv2.imread(h["photo_path"])
        if bgr is None:
            push("warning", f"Cannot read photo for {h['name']} — skipping")
            continue
        emb, _ = get_face_embedding(bgr)
        if emb is not None:
            reference_encodings.append((h, emb))
            push("ok", f"Loaded: {h['name']}")
        else:
            push("warning", f"No face in photo for {h['name']} — skipping")

    if not reference_encodings:
        push("error", "No usable face encodings. Check photo quality.")
        on_complete(None); end_session(); return

    push("step", f"{len(reference_encodings)} reference(s) ready. Look at the camera…")
    time.sleep(CAPTURE_DELAY)

    push("step", "Capturing face from camera…")
    request_capture()
    live_image = get_captured_frame(timeout=15)

    if live_image is None:
        push("error", "No frame received. Ensure camera feed is open.")
        on_complete(None); end_session(); return

    push("ok", "Frame captured ✓")
    push("step", "Detecting face with YuNet…")

    live_bgr = cv2.cvtColor(live_image, cv2.COLOR_RGB2BGR)
    live_emb, _ = get_face_embedding(live_bgr)
    if live_emb is None:
        push("error", "No face detected. Improve lighting and try again.")
        on_complete(None); end_session(); return

    push("ok", "Face detected (SFace embedding) ✓")
    push("step", f"Comparing against {len(reference_encodings)} stored face(s)…")

    best_match, best_dist = None, 1.0
    for holder, ref_emb in reference_encodings:
        dist = face_distance(ref_emb, live_emb)
        push("info", f"Checked {holder['name']}: distance={dist:.3f}")
        if dist < best_dist:
            best_dist  = dist
            best_match = holder

    confidence_pct = round((1.0 - best_dist) * 100, 1)

    if best_dist <= FACE_CV_THRESHOLD:
        push("success", f"Match: {best_match['name']} (confidence {confidence_pct}%)")
        result = {**best_match, "distance": best_dist, "confidence": confidence_pct}
        _last_result.clear()
        _last_result.update(result)
        on_complete(result)
    else:
        push("notfound",
             f"No match. Best confidence: {confidence_pct}% "
             f"(threshold: {round((1 - FACE_CV_THRESHOLD) * 100)}%)")
        _last_result.clear()
        on_complete(None)

    end_session()


def get_last_result():
    return dict(_last_result)
