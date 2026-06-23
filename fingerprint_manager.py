"""
Fingerprint Manager — integrates AS608 sensor with Flask dashboard.
Runs enrollment in a background thread and streams progress via a queue.
"""
import threading
import queue
import time

# Sensor is optional — gracefully fails if not connected (e.g. dev machine)
try:
    import serial
    import adafruit_fingerprint
    SENSOR_AVAILABLE = True
except ImportError:
    SENSOR_AVAILABLE = False

# ── Global state ──────────────────────────────────────────────────────────────

_uart = None
_finger = None
_sessions = {}        # { holder_id: Queue }  — one queue per active enrollment
_session_lock = threading.Lock()

SERIAL_PORT = "/dev/ttyAMA0"
BAUD_RATE   = 57600


def _get_sensor():
    """Lazy-init the serial connection. Returns (finger, error_msg)."""
    global _uart, _finger
    if not SENSOR_AVAILABLE:
        return None, "adafruit_fingerprint library not installed"
    try:
        if _uart is None or not _uart.is_open:
            _uart = serial.Serial(SERIAL_PORT, baudrate=BAUD_RATE, timeout=1)
            _finger = adafruit_fingerprint.Adafruit_Fingerprint(_uart)
        return _finger, None
    except Exception as e:
        return None, str(e)


# ── Session helpers ───────────────────────────────────────────────────────────

def start_session(holder_id: int) -> bool:
    """Start a new enrollment session. Returns False if one is already running."""
    with _session_lock:
        if holder_id in _sessions:
            return False
        _sessions[holder_id] = queue.Queue()
    return True


def get_session_queue(holder_id: int):
    return _sessions.get(holder_id)


def end_session(holder_id: int):
    with _session_lock:
        _sessions.pop(holder_id, None)


def _push(holder_id: int, msg_type: str, text: str):
    """Push a message onto the holder's queue."""
    q = _sessions.get(holder_id)
    if q:
        q.put({"type": msg_type, "text": text})


# ── Enrollment thread ─────────────────────────────────────────────────────────

def run_enrollment(holder_id: int, name: str, on_complete):
    """
    Run full enrollment in a background thread.
    Pushes SSE-style messages to the holder's queue.
    Calls on_complete(template_data) on success or on_complete(None) on failure.
    """
    def push(msg_type, text):
        _push(holder_id, msg_type, text)

    finger, err = _get_sensor()
    if not finger:
        push("error", f"Sensor not available: {err}")
        on_complete(None)
        end_session(holder_id)
        return

    # ── Capture two images ────────────────────────────────────────────────────
    for img_num in range(1, 3):
        if img_num == 1:
            push("step", "Place finger firmly on sensor…")
        else:
            push("step", "Place the SAME finger on sensor again…")

        # Wait for finger
        timeout = 30  # seconds
        start = time.time()
        while True:
            if time.time() - start > timeout:
                push("error", "Timed out waiting for finger. Please try again.")
                on_complete(None)
                end_session(holder_id)
                return

            i = finger.get_image()
            if i == adafruit_fingerprint.OK:
                push("ok", "Image captured ✓")
                break
            elif i == adafruit_fingerprint.NOFINGER:
                push("waiting", "Waiting for finger…")
                time.sleep(0.5)
            elif i == adafruit_fingerprint.IMAGEFAIL:
                push("error", "Imaging error. Please try again.")
                on_complete(None)
                end_session(holder_id)
                return

        # Template the image
        push("step", "Processing image…")
        i = finger.image_2_tz(img_num)
        if i == adafruit_fingerprint.OK:
            push("ok", f"Image {img_num} templated ✓")
        elif i == adafruit_fingerprint.IMAGEMESS:
            push("error", "Image too messy — please clean finger and try again.")
            on_complete(None)
            end_session(holder_id)
            return
        elif i == adafruit_fingerprint.FEATUREFAIL:
            push("error", "Could not identify fingerprint features.")
            on_complete(None)
            end_session(holder_id)
            return
        elif i == adafruit_fingerprint.INVALIDIMAGE:
            push("error", "Invalid image captured.")
            on_complete(None)
            end_session(holder_id)
            return
        else:
            push("error", "Templating failed.")
            on_complete(None)
            end_session(holder_id)
            return

        if img_num == 1:
            push("step", "Remove finger from sensor…")
            time.sleep(1)
            while finger.get_image() != adafruit_fingerprint.NOFINGER:
                time.sleep(0.2)
            push("ok", "Finger removed ✓")

    # ── Create model ──────────────────────────────────────────────────────────
    push("step", "Creating fingerprint model…")
    i = finger.create_model()
    if i == adafruit_fingerprint.OK:
        push("ok", "Model created ✓")
    elif i == adafruit_fingerprint.ENROLLMISMATCH:
        push("error", "Fingerprints did not match — please try again with the same finger.")
        on_complete(None)
        end_session(holder_id)
        return
    else:
        push("error", "Failed to create model.")
        on_complete(None)
        end_session(holder_id)
        return

    # ── Download template to Pi ───────────────────────────────────────────────
    push("step", "Downloading template to Raspberry Pi…")
    data = finger.get_fpdata(sensorbuffer="char", slot=1)
    if not data:
        push("error", "Failed to download template from sensor.")
        on_complete(None)
        end_session(holder_id)
        return

    push("ok", f"Template downloaded ({len(data)} bytes) ✓")
    push("success", f"Fingerprint enrolled successfully for {name}!")
    on_complete(list(data))
    end_session(holder_id)


# ── Search / Identify thread ──────────────────────────────────────────────────

SEARCH_SESSION_KEY = "__search__"
CONFIDENCE_THRESHOLD = 30


def start_search_session() -> bool:
    """Start a fingerprint search session. Returns False if one is already running."""
    return start_session(SEARCH_SESSION_KEY)


def get_search_queue():
    return get_session_queue(SEARCH_SESSION_KEY)


def end_search_session():
    end_session(SEARCH_SESSION_KEY)


def _push_search(msg_type: str, text: str):
    _push(SEARCH_SESSION_KEY, msg_type, text)


def run_search(all_templates: list, on_complete):
    """
    Capture a live finger, compare against all stored templates.
    all_templates: list of {"holder_id": int, "name": str, "template": list}
    on_complete(result): result = {"holder_id", "name", "score"} or None if no match
    """
    def push(msg_type, text):
        _push_search(msg_type, text)

    finger, err = _get_sensor()
    if not finger:
        push("error", f"Sensor not available: {err}")
        on_complete(None)
        end_search_session()
        return

    if not all_templates:
        push("error", "No fingerprints enrolled in the database yet.")
        on_complete(None)
        end_search_session()
        return

    # ── Capture live finger into buffer 1 ────────────────────────────────────
    push("step", "Place finger on sensor…")
    timeout = 30
    start = time.time()
    while True:
        if time.time() - start > timeout:
            push("error", "Timed out. No finger detected.")
            on_complete(None)
            end_search_session()
            return
        i = finger.get_image()
        if i == adafruit_fingerprint.OK:
            push("ok", "Fingerprint image captured ✓")
            break
        elif i == adafruit_fingerprint.NOFINGER:
            push("waiting", "Waiting for finger…")
            time.sleep(0.5)
        elif i == adafruit_fingerprint.IMAGEFAIL:
            push("error", "Imaging error. Please try again.")
            on_complete(None)
            end_search_session()
            return

    # Template into buffer 1 — stays there throughout all comparisons
    if finger.image_2_tz(1) != adafruit_fingerprint.OK:
        push("error", "Failed to process fingerprint image.")
        on_complete(None)
        end_search_session()
        return

    push("ok", "Image processed ✓")
    push("step", f"Comparing against {len(all_templates)} enrolled fingerprint(s)…")

    best_id   = None
    best_name = None
    best_score = 0

    for entry in all_templates:
        holder_id     = entry["holder_id"]
        name          = entry["name"]
        stored_data   = entry["template"]

        # Flush UART then load stored template into buffer 2
        try:
            _uart.reset_input_buffer()
            _uart.reset_output_buffer()
            time.sleep(0.1)

            finger.send_fpdata(stored_data, sensorbuffer="char", slot=2)
            time.sleep(0.2)

            finger.compare_templates()
            time.sleep(0.1)

            raw = finger.confidence
            score = raw[0] if isinstance(raw, tuple) else (raw if raw is not None else 0)

            push("info", f"Checked {name}: score {score}")

            if score > best_score:
                best_score = score
                best_id    = holder_id
                best_name  = name

        except Exception as e:
            push("warning", f"Skipped {name}: {e}")
            continue

    if best_score >= CONFIDENCE_THRESHOLD:
        push("success", f"Match found: {best_name} (score: {best_score})")
        on_complete({"holder_id": best_id, "name": best_name, "score": best_score})
    else:
        push("notfound", f"No match found. Best score: {best_score} (threshold: {CONFIDENCE_THRESHOLD})")
        on_complete(None)

    end_search_session()