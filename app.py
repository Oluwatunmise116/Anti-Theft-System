from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_from_directory, Response
import os
import threading
import json
import time

# Secrets (FLASK_SECRET_KEY, ROBASE_API_KEY) come from the environment or a
# local, git-ignored .env — loaded before `database`, which reads
# LICENSE_DB_PATH at import. Real environment variables win.
try:
    from dotenv import load_dotenv
    load_dotenv(override=False)
except ImportError:
    pass

import database as db
import fingerprint_manager as fp
import face_manager as fm
import gate_manager as gm
import config as cfg
import auth
import gate_verification as gv
import verification_settings as vs
import wifi_uplink as wifi
import functools
from anpr import postprocessing as anpr_post
import attributes as va
from werkzeug.utils import secure_filename
from datetime import datetime

app = Flask(__name__)
# Staff sign-in, server-side gate sessions and CSRF protection for every
# route (see auth.py). The session signing key comes from FLASK_SECRET_KEY.
auth.init_app(app)

UPLOAD_FOLDER = "photos"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
# Hard ceiling on any request body, so an oversized upload is refused by
# Werkzeug before the application reads it into memory. Kept above
# gate_upload_max_mb so the friendlier per-image message is what an operator
# normally sees.
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024


@app.errorhandler(413)
def _payload_too_large(_error):
    return jsonify({"error": "That file is too large to upload."}), 413


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def _json_body() -> dict:
    """The JSON object body, or {} for a missing/non-JSON/non-object body."""
    data = request.get_json(silent=True)
    return data if isinstance(data, dict) else {}


def _text(value) -> str:
    return value.strip() if isinstance(value, str) else ""


def _capture_id(value) -> str:
    text = _text(value)
    return text if len(text) <= 64 else ""


def _trip_id(value):
    """A positive int trip id, or None. JSON booleans are rejected."""
    if isinstance(value, bool):
        return None
    try:
        trip_id = int(value)
    except (TypeError, ValueError):
        return None
    return trip_id if trip_id > 0 else None


def _capture_denied(capture_id):
    """None when this gate session may use the capture, else a 403 response."""
    if gm.claim_capture(capture_id, auth.current_staff().session_id):
        return None
    return jsonify({"error": "This capture belongs to another gate session — "
                             "reload the page."}), 403


def _verification_response(exc, **extra):
    return jsonify({**extra, **exc.payload()}), exc.status


def _upload_max_side() -> int:
    """Longest side, in pixels, the browser shrinks an upload to before sending."""
    try:
        return max(640, int(cfg.get("gate_upload_max_side", 1920)))
    except (TypeError, ValueError):
        return 1920


# ── DASHBOARD ──────────────────────────────────────────

@app.route("/")
def dashboard():
    stats = db.get_stats()
    holders = db.get_all_holders()[:5]  # recent 5
    return render_template("dashboard.html", stats=stats, recent=holders)


# ── HOLDERS ────────────────────────────────────────────

@app.route("/holders")
def holders():
    query = request.args.get("q", "")
    if query:
        holders_list = db.search_holders(query)
    else:
        holders_list = db.get_all_holders()
    return render_template("holders.html", holders=holders_list, query=query)


@app.route("/holders/add", methods=["GET", "POST"])
def add_holder():
    if request.method == "POST":
        data = {
            "surname": request.form["surname"].upper(),
            "first_name": request.form["first_name"],
            "date_of_birth": request.form["date_of_birth"],
            "sex": request.form["sex"],
            "height_cm": float(request.form["height_cm"] or 0),
            "blood_group": request.form["blood_group"],
            "street_address": request.form["street_address"],
            "state": request.form["state"],
            "phone_number": request.form["phone_number"],
            "next_of_kin": request.form["next_of_kin"],
            "next_of_kin_phone": request.form["next_of_kin_phone"],
            "religion": request.form["religion"],
            "nationality": request.form["nationality"],
        }
        holder_id = db.add_holder(data)
        if not db.normalize_holder_phone(data["phone_number"]):
            flash("The phone number was not recognised as a valid number. SMS exit codes "
                  "cannot be sent to this driver until it is corrected.", "error")

        # License info
        if request.form.get("license_number"):
            db.add_license({
                "holder_id": holder_id,
                "license_number": request.form["license_number"],
                "date_of_issue": request.form["date_of_issue"],
                "expiry_date": request.form["expiry_date"],
                "license_class": request.form["license_class"],
                "state_of_issue": request.form["state_of_issue"],
                "endorsements": request.form.get("endorsements", ""),
                "authorized_by": request.form.get("authorized_by", ""),
            })

        # Passport photo
        if "photo" in request.files:
            file = request.files["photo"]
            if file and allowed_file(file.filename):
                filename = secure_filename(f"holder_{holder_id}_passport.jpg")
                filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
                file.save(filepath)
                db.save_photo(holder_id, "passport", filepath)

        flash(f"Holder #{holder_id} registered successfully!", "success")
        return redirect(url_for("view_holder", holder_id=holder_id))

    return render_template("add_holder.html")


@app.route("/holders/<int:holder_id>")
def view_holder(holder_id):
    record = db.get_full_record(holder_id)
    if not record:
        flash("Holder not found.", "error")
        return redirect(url_for("holders"))
    photos = db.get_photos(holder_id)
    has_fingerprint = db.get_fingerprint(holder_id) is not None
    return render_template("view_holder.html", record=record, photos=photos,
                           has_fingerprint=has_fingerprint,
                           sms=gv.holder_sms_status(record))


@app.route("/holders/<int:holder_id>/edit", methods=["GET", "POST"])
def edit_holder(holder_id):
    record = db.get_full_record(holder_id)
    if not record:
        flash("Holder not found.", "error")
        return redirect(url_for("holders"))

    if request.method == "POST":
        data = {
            "surname": request.form["surname"].upper(),
            "first_name": request.form["first_name"],
            "date_of_birth": request.form["date_of_birth"],
            "sex": request.form["sex"],
            "height_cm": float(request.form["height_cm"] or 0),
            "blood_group": request.form["blood_group"],
            "street_address": request.form["street_address"],
            "state": request.form["state"],
            "phone_number": request.form["phone_number"],
            "next_of_kin": request.form["next_of_kin"],
            "next_of_kin_phone": request.form["next_of_kin_phone"],
            "religion": request.form["religion"],
            "nationality": request.form["nationality"],
        }
        staff = auth.current_staff()
        # SMS exit codes go to this number, so only an administrator may
        # change where they go (setting, replacing or removing it).
        if db.normalize_holder_phone(data["phone_number"]) != record.get("phone_e164") \
                and not staff.is_admin:
            flash("Only an administrator can change a driver's phone number, because SMS exit "
                  "codes are sent to it. Nothing was saved.", "error")
            return redirect(url_for("edit_holder", holder_id=holder_id))
        changes = db.update_holder(holder_id, data, staff_user_id=staff.user_id)
        if changes["sms_challenges_cancelled"]:
            flash("The phone number changed, so SMS codes already sent to the old number "
                  "were cancelled.", "success")
        if not changes["phone_valid"]:
            flash("The phone number was not recognised as a valid number. SMS exit codes "
                  "cannot be sent to this driver until it is corrected.", "error")

        # Update license
        license_data = {
            "license_number": request.form["license_number"],
            "date_of_issue": request.form["date_of_issue"],
            "expiry_date": request.form["expiry_date"],
            "license_class": request.form["license_class"],
            "state_of_issue": request.form["state_of_issue"],
            "endorsements": request.form.get("endorsements", ""),
            "authorized_by": request.form.get("authorized_by", ""),
        }
        existing_license = db.get_license(holder_id)
        if existing_license:
            db.update_license(holder_id, license_data)
        else:
            license_data["holder_id"] = holder_id
            db.add_license(license_data)

        # Update photo if provided
        if "photo" in request.files:
            file = request.files["photo"]
            if file and file.filename and allowed_file(file.filename):
                filename = secure_filename(f"holder_{holder_id}_passport.jpg")
                filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
                file.save(filepath)
                db.save_photo(holder_id, "passport", filepath)

        flash("Record updated successfully!", "success")
        return redirect(url_for("view_holder", holder_id=holder_id))

    return render_template("edit_holder.html", record=record)


@app.route("/holders/<int:holder_id>/delete", methods=["POST"])
def delete_holder(holder_id):
    db.delete_holder(holder_id, staff_user_id=auth.current_staff().user_id)
    flash("Holder deleted successfully.", "success")
    return redirect(url_for("holders"))


# ── PHOTOS ─────────────────────────────────────────────

@app.route("/photos/<path:filename>")
def serve_photo(filename):
    # DB may store "photos/holder_1_passport.jpg" or just "holder_1_passport.jpg"
    if filename.startswith("photos/"):
        filename = filename[len("photos/"):]
    photos_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), UPLOAD_FOLDER)
    return send_from_directory(photos_dir, filename)


# ── FINGERPRINT ENROLLMENT ─────────────────────────────

@app.route("/holders/<int:holder_id>/enroll/start", methods=["POST"])
def enroll_start(holder_id):
    record = db.get_full_record(holder_id)
    if not record:
        return jsonify({"error": "Holder not found"}), 404
    name = f"{record['surname']} {record['first_name']}"
    if not fp.start_session(holder_id):
        return jsonify({"error": "Enrollment already in progress"}), 409
    def on_complete(template_data):
        if template_data:
            db.save_fingerprint(holder_id, template_data)
    thread = threading.Thread(target=fp.run_enrollment, args=(holder_id, name, on_complete), daemon=True)
    thread.start()
    return jsonify({"started": True, "holder_id": holder_id})


@app.route("/holders/<int:holder_id>/enroll/stream")
def enroll_stream(holder_id):
    def event_stream():
        q = fp.get_session_queue(holder_id)
        if not q:
            yield 'data: {"type": "error", "text": "No active session. Click Enroll to start."}\n\n'
            return
        while True:
            try:
                msg = q.get(timeout=45)
                yield f"data: {json.dumps(msg)}\n\n"
                if msg["type"] in ("success", "error"):
                    break
            except Exception:
                yield 'data: {"type": "error", "text": "Session timed out."}\n\n'
                break
    return Response(event_stream(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/holders/<int:holder_id>/enroll/cancel", methods=["POST"])
def enroll_cancel(holder_id):
    fp.end_session(holder_id)
    return jsonify({"cancelled": True})


@app.route("/api/fingerprint/status/<int:holder_id>")
def api_fingerprint_status(holder_id):
    has_fp = db.get_fingerprint(holder_id) is not None
    return jsonify({"holder_id": holder_id, "enrolled": has_fp})


@app.route("/api/stats")
def api_stats():
    return jsonify(db.get_stats())



# ── FINGERPRINT SEARCH ─────────────────────────────────

@app.route("/fingerprint-search")
def fingerprint_search():
    return render_template("fingerprint_search.html")


@app.route("/fingerprint-search/start", methods=["POST"])
def fingerprint_search_start():
    if not fp.start_search_session():
        return jsonify({"error": "A search is already in progress"}), 409

    all_templates = db.get_all_fingerprints()
    result_holder = {}
    fp._last_search_result = {}

    def on_complete(result):
        if result:
            result_holder.update(result)
            fp._last_search_result = result

    thread = threading.Thread(
        target=fp.run_search,
        args=(all_templates, on_complete),
        daemon=True
    )
    thread.start()
    return jsonify({"started": True})


@app.route("/fingerprint-search/stream")
def fingerprint_search_stream():
    def event_stream():
        q = fp.get_search_queue()
        if not q:
            yield "data: " + json.dumps({"type":"error","text":"No active search session."}) + "\n\n"
            return
        while True:
            try:
                msg = q.get(timeout=45)
                yield "data: " + json.dumps(msg) + "\n\n"
                if msg["type"] in ("success", "notfound", "error"):
                    break
            except Exception:
                yield "data: " + json.dumps({"type":"error","text":"Session timed out."}) + "\n\n"
                break
    return Response(event_stream(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/fingerprint-search/cancel", methods=["POST"])
def fingerprint_search_cancel():
    fp.end_search_session()
    return jsonify({"cancelled": True})


@app.route("/fingerprint-search/last-match")
def fingerprint_search_last_match():
    result = getattr(fp, "_last_search_result", {})
    if not result:
        return jsonify({"holder_id": None})
    # Enrich with full record
    holder_id = result.get("holder_id")
    record = db.get_full_record(holder_id) if holder_id else None
    photos = db.get_photos(holder_id) if holder_id else {}
    if record:
        return jsonify({
            "holder_id": holder_id,
            "name": f"{record['surname']}, {record['first_name']}",
            "score": result.get("score"),
            "date_of_birth": record.get("date_of_birth"),
            "blood_group": record.get("blood_group"),
            "license_number": record.get("license_number"),
            "license_class": record.get("license_class"),
            "expiry_date": record.get("expiry_date"),
            "photo_path": photos.get("passport", ""),
        })
    return jsonify({"holder_id": None})


# ── FACE RECOGNITION SEARCH ────────────────────────────

@app.route("/face-search")
def face_search():
    return render_template("face_search.html")


@app.route("/face-search/start", methods=["POST"])
def face_search_start():
    if not fm._cam_running.is_set():
        return jsonify({"error": "Camera is not ready — open the video feed first"}), 503
    if not fm.start_session():
        return jsonify({"error": "A face search is already in progress"}), 409

    # Get all holders that have passport photos
    all_holders = db.get_all_holders()
    holders_with_photos = []
    for h in all_holders:
        photos = db.get_photos(h["id"])
        if photos.get("passport"):
            holders_with_photos.append({
                "holder_id": h["id"],
                "name": f"{h['surname']}, {h['first_name']}",
                "photo_path": photos["passport"],
                "date_of_birth": h.get("date_of_birth"),
                "blood_group": h.get("blood_group"),
                "license_number": h.get("license_number"),
                "license_class": h.get("license_class"),
                "expiry_date": h.get("expiry_date"),
            })

    def on_complete(result):
        pass  # result already stored in face_manager._last_result

    thread = threading.Thread(
        target=fm.run_face_search,
        args=(holders_with_photos, on_complete),
        daemon=True
    )
    thread.start()
    return jsonify({"started": True})


@app.route("/face-search/stream")
def face_search_stream():
    def event_stream():
        q = fm.get_session_queue()
        if not q:
            yield "data: " + json.dumps({"type": "error", "text": "No active session."}) + "\n\n"
            return
        while True:
            try:
                msg = q.get(timeout=60)
                yield "data: " + json.dumps(msg) + "\n\n"
                if msg["type"] in ("success", "notfound", "error"):
                    break
            except Exception:
                yield "data: " + json.dumps({"type": "error", "text": "Session timed out."}) + "\n\n"
                break
    return Response(event_stream(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/face-search/cancel", methods=["POST"])
def face_search_cancel():
    fm.end_session()
    return jsonify({"cancelled": True})


@app.route("/face-search/video-feed")
def face_search_video_feed():
    """MJPEG stream from the persistent camera worker."""
    if not fm.ensure_camera_running():
        return Response("Camera unavailable — check connection", status=503)
    return Response(
        fm.generate_video_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/face-search/last-match")
def face_search_last_match():
    result = fm.get_last_result()
    if not result or not result.get("holder_id"):
        return jsonify({"holder_id": None})
    holder_id = result["holder_id"]
    record = db.get_full_record(holder_id)
    photos = db.get_photos(holder_id)
    if record:
        return jsonify({
            "holder_id": holder_id,
            "name": f"{record['surname']}, {record['first_name']}",
            "confidence": result.get("confidence"),
            "date_of_birth": record.get("date_of_birth"),
            "blood_group": record.get("blood_group"),
            "license_number": record.get("license_number"),
            "license_class": record.get("license_class"),
            "expiry_date": record.get("expiry_date"),
            "photo_path": photos.get("passport", ""),
        })
    return jsonify({"holder_id": None})

# ── GATE: VEHICLE DETECTION ───────────────────────────

@app.route("/gate/detect/stream")
def gate_detect_stream():
    """SSE: vehicle_detected / vehicle_cleared events from MOG2 worker."""
    gm.ensure_detection_running()
    q = gm.subscribe_detect()

    def event_stream():
        # Send current state immediately so the UI is in sync on connect
        yield f"data: {json.dumps(gm.get_detect_state())}\n\n"
        try:
            while True:
                try:
                    event = q.get(timeout=20)
                    yield f"data: {json.dumps(event)}\n\n"
                except Exception:
                    yield 'data: {"event":"ping"}\n\n'
        except GeneratorExit:
            gm.unsubscribe_detect(q)

    return Response(event_stream(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/gate/detect/anpr-now", methods=["POST"])
def gate_detect_anpr_now():
    """Grab the current frame and run ANPR — used by exit page on vehicle detect."""
    if not fm._cam_running.is_set():
        if not fm.ensure_camera_running():
            return jsonify({"plate": ""})
    fm.request_capture()
    import cv2 as _cv2
    rgb = fm.get_captured_frame(timeout=8)
    if rgb is None:
        return jsonify({"plate": ""})
    bgr   = _cv2.cvtColor(rgb, _cv2.COLOR_RGB2BGR)
    plate = gm.detect_plate(bgr)
    fm.stop_camera()
    return jsonify({"plate": plate or ""})


# ── GATE: CAMERA FEED ─────────────────────────────────

@app.route("/gate/video-feed")
def gate_video_feed():
    if not fm.ensure_camera_running():
        return Response("Camera unavailable", status=503)
    return Response(fm.generate_video_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


# ── GATE: ENTRY ────────────────────────────────────────

@app.route("/gate/entry")
def gate_entry():
    capture_id = gm.new_capture_id(owner=auth.current_staff().session_id)
    settings = vs.load()
    return render_template("gate_entry.html", capture_id=capture_id,
                           passcode_min=settings.passcode_min_length,
                           passcode_max=settings.passcode_max_length,
                           upload_max_side=_upload_max_side())


@app.route("/gate/entry/identity")
def gate_entry_identity():
    """
    The server's view of the driver's identity for this capture, from its
    own face/fingerprint search results. The entry page uses it to choose
    between the SMS code fallback, guest passcode and 'unresolved'.
    """
    capture_id = _capture_id(request.args.get("capture_id"))
    if not capture_id:
        return jsonify({"error": "Missing capture_id"}), 400
    denied = _capture_denied(capture_id)
    if denied:
        return denied
    return jsonify(gv.entry_identity_view(gm.temp_get(capture_id)))


@app.route("/gate/model/status")
def gate_model_status():
    """
    Readiness of every model the gate uses. The vehicle-attribute entry
    reports what is present on disk without loading anything, so polling
    this route never pulls a model into memory.
    """
    return jsonify({
        "plate": gm.plate_model_status(),
        "face":  fm.face_model_status(),
        "vehicle_attributes": gm.attribute_model_status(),
    })


@app.route("/gate/attributes/options")
def gate_attribute_options():
    """The closed label spaces, for the operator's correction dropdowns."""
    return jsonify({
        "colours": [{"label": c, "display": c.capitalize()} for c in va.COLOURS],
        "types": [{"label": t, "display": t.replace("_", " ").capitalize()}
                  for t in va.TYPES],
        "brands": [{"label": b, "display": b.replace("-", " ").title()}
                   for b in va.BRANDS],
    })


@app.route("/gate/model/download", methods=["POST"])
def gate_model_download():
    """
    Face models (from the OpenCV Zoo) may still be fetched on demand. The
    Nigerian plate detector is NOT downloadable at runtime — it must be
    trained/exported as an explicit offline deployment step (see
    README_ANPR.md) and placed at the configured plate_model_path. This
    intentionally replaces the old behaviour of fetching an unrelated
    generic plate model from third-party GitHub mirrors.
    """
    model = request.json.get("model", "plate") if request.is_json else "plate"
    if model == "face":
        ok = fm.download_face_models()
        fm._get_face_cv_models()   # load into memory after download
        return jsonify({"ok": ok, **fm.face_model_status()})
    return jsonify({
        "ok": False,
        "error": "The Nigerian plate detector cannot be downloaded automatically. "
                 "Train it with scripts/train_plate_detector.py and export it with "
                 "scripts/export_plate_model.py, then place the weights at the "
                 "configured plate_model_path. See README_ANPR.md.",
        **gm.plate_model_status(),
    }), 400


@app.route("/gate/entry/vehicle/snap", methods=["POST"])
def gate_entry_vehicle_snap():
    """Grab a single camera frame and save it — no analysis yet."""
    capture_id = _capture_id(_json_body().get("capture_id"))
    if not capture_id:
        return jsonify({"error": "Missing capture_id"}), 400
    denied = _capture_denied(capture_id)
    if denied:
        return denied
    result = gm.snap_vehicle_frame(capture_id)
    return jsonify(result)


@app.route("/gate/entry/vehicle/upload", methods=["POST"])
def gate_entry_vehicle_upload():
    """
    Accept an uploaded still image as this capture's vehicle photo.

    The image is analysed by exactly the same pipeline the camera path uses
    (plate detection, then vehicle attributes) — the caller follows this
    with /gate/entry/vehicle/start as if a photo had been snapped.

    An uploaded image is NOT evidence the vehicle was at the gate, so its
    provenance is recorded and shown. It does not change any authorization
    rule: entry still needs a plate, and exit still needs a verified face,
    fingerprint or passcode.
    """
    capture_id = _capture_id(request.form.get("capture_id"))
    if not capture_id:
        return jsonify({"error": "Missing capture_id"}), 400
    denied = _capture_denied(capture_id)
    if denied:
        return denied
    if "image" not in request.files:
        return jsonify({"error": "No image file was uploaded"}), 400

    upload = request.files["image"]
    if not upload or not upload.filename:
        return jsonify({"error": "No image file was uploaded"}), 400

    # The extension check is a fast reject for obvious mistakes; the real
    # validation is decoding the bytes in save_uploaded_vehicle_image().
    if not allowed_file(upload.filename):
        return jsonify({"error": "Only JPEG and PNG images are accepted"}), 400

    result = gm.save_uploaded_vehicle_image(capture_id, upload.read(),
                                            upload.filename)
    if "error" in result:
        return jsonify(result), 400
    return jsonify(result)


@app.route("/gate/entry/vehicle/start", methods=["POST"])
def gate_entry_vehicle_start():
    capture_id = _capture_id(_json_body().get("capture_id"))
    if not capture_id:
        return jsonify({"error": "Missing capture_id"}), 400
    denied = _capture_denied(capture_id)
    if denied:
        return denied
    if not gm.start_vehicle_session(capture_id):
        return jsonify({"error": "Vehicle capture already in progress"}), 409
    thread = threading.Thread(
        target=gm.run_vehicle_capture,
        args=(capture_id, lambda _: None),
        daemon=True,
    )
    thread.start()
    return jsonify({"started": True})


@app.route("/gate/entry/vehicle/auto-start", methods=["POST"])
def gate_entry_vehicle_auto_start():
    """Start auto-detect: YOLO watches the live feed and captures when a vehicle is stable."""
    capture_id = _capture_id(_json_body().get("capture_id"))
    if not capture_id:
        return jsonify({"error": "Missing capture_id"}), 400
    denied = _capture_denied(capture_id)
    if denied:
        return denied
    if not gm.start_vehicle_session(capture_id):
        return jsonify({"error": "Vehicle capture already in progress"}), 409
    thread = threading.Thread(
        target=gm.run_vehicle_auto_capture,
        args=(capture_id, lambda _: None),
        daemon=True,
    )
    thread.start()
    return jsonify({"started": True})


@app.route("/gate/entry/vehicle/stream/<capture_id>")
def gate_entry_vehicle_stream(capture_id):
    denied = _capture_denied(capture_id)
    if denied:
        return denied

    def event_stream():
        q = gm.get_vehicle_queue(capture_id)
        if not q:
            yield 'data: {"type":"error","text":"No active vehicle capture session."}\n\n'
            return
        while True:
            try:
                msg = q.get(timeout=45)
                yield f"data: {json.dumps(msg)}\n\n"
                if msg["type"] in ("success", "error"):
                    break
            except Exception:
                yield 'data: {"type":"error","text":"Session timed out."}\n\n'
                break
    return Response(event_stream(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/gate/entry/face/start", methods=["POST"])
def gate_entry_face_start():
    capture_id = _capture_id(_json_body().get("capture_id"))
    if not capture_id:
        return jsonify({"error": "Missing capture_id"}), 400
    denied = _capture_denied(capture_id)
    if denied:
        return denied
    if not gm.start_face_session(capture_id):
        return jsonify({"error": "Face capture already in progress"}), 409

    # Build enriched holder list for DB lookup after face capture
    holders_with_photos = []
    for h in db.get_all_holders():
        photos = db.get_photos(h["id"])
        if photos.get("passport"):
            holders_with_photos.append({
                "holder_id":      h["id"],
                "holder_uid":     h.get("holder_uid"),
                "name":           f"{h['surname']}, {h['first_name']}",
                "photo_path":     photos["passport"],
                "date_of_birth":  h.get("date_of_birth"),
                "blood_group":    h.get("blood_group"),
                "license_number": h.get("license_number"),
                "license_class":  h.get("license_class"),
                "expiry_date":    h.get("expiry_date"),
            })

    thread = threading.Thread(
        target=gm.run_face_capture,
        args=(capture_id, lambda _: None, holders_with_photos),
        daemon=True,
    )
    thread.start()
    return jsonify({"started": True})


@app.route("/gate/entry/face/stream/<capture_id>")
def gate_entry_face_stream(capture_id):
    denied = _capture_denied(capture_id)
    if denied:
        return denied

    def event_stream():
        q = gm.get_face_queue(capture_id)
        if not q:
            yield 'data: {"type":"error","text":"No active face capture session."}\n\n'
            return
        while True:
            try:
                msg = q.get(timeout=45)
                yield f"data: {json.dumps(msg)}\n\n"
                if msg["type"] in ("success", "error"):
                    break
            except Exception:
                yield 'data: {"type":"error","text":"Session timed out."}\n\n'
                break
    return Response(event_stream(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/gate/entry/fp/start", methods=["POST"])
def gate_entry_fp_start():
    capture_id = _capture_id(_json_body().get("capture_id"))
    if not capture_id:
        return jsonify({"error": "Missing capture_id"}), 400
    denied = _capture_denied(capture_id)
    if denied:
        return denied
    if not gm.start_fp_entry(capture_id):
        return jsonify({"error": "Fingerprint capture already in progress"}), 409

    # Build enriched fingerprint list for DB lookup after fp scan
    all_templates = db.get_all_fingerprints()
    all_holders   = {h["id"]: h for h in db.get_all_holders()}
    for t in all_templates:
        h      = all_holders.get(t["holder_id"], {})
        photos = db.get_photos(t["holder_id"])
        pp     = photos.get("passport", "")
        t.update({
            "date_of_birth":  h.get("date_of_birth"),
            "blood_group":    h.get("blood_group"),
            "license_number": h.get("license_number"),
            "license_class":  h.get("license_class"),
            "expiry_date":    h.get("expiry_date"),
            "photo_url":      (f"/photos/{os.path.basename(pp)}" if pp else None),
        })

    thread = threading.Thread(
        target=gm.run_fp_entry,
        args=(capture_id, lambda _: None, all_templates),
        daemon=True,
    )
    thread.start()
    return jsonify({"started": True})


@app.route("/gate/entry/fp/stream/<capture_id>")
def gate_entry_fp_stream(capture_id):
    denied = _capture_denied(capture_id)
    if denied:
        return denied

    def event_stream():
        q = gm.get_fp_entry_queue(capture_id)
        if not q:
            yield 'data: {"type":"error","text":"No active fingerprint session."}\n\n'
            return
        while True:
            try:
                msg = q.get(timeout=40)
                yield f"data: {json.dumps(msg)}\n\n"
                if msg["type"] in ("success", "error"):
                    break
            except Exception:
                yield 'data: {"type":"error","text":"Session timed out."}\n\n'
                break
    return Response(event_stream(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/gate/entry/fp/cancel", methods=["POST"])
def gate_entry_fp_cancel():
    capture_id = _capture_id(_json_body().get("capture_id"))
    if capture_id and not _capture_denied(capture_id):
        gm.cancel_fp_entry(capture_id)
    return jsonify({"cancelled": True})


# ── GATE: EXIT FINGERPRINT VERIFY ──────────────────────

def _biometric_recorder(trip_id, method):
    """
    on_result callback for the exit face/fingerprint comparison. It runs in
    the capture thread, before the result is pushed to the page: it stores
    the comparison and, for a match against THIS trip's own entry capture,
    authorizes the exit for the gate session that started the scan (read
    here, inside the request). Returns True when the exit was authorized.
    """
    staff = auth.current_staff()

    def record(result):
        if not result:
            return False
        if method == gv.METHOD_FACE:
            return gv.record_biometric_result(trip_id, staff, method, result.get("face_match"),
                                              result.get("face_distance"),
                                              result.get("exit_photo"))
        return gv.record_biometric_result(trip_id, staff, method, result.get("match"),
                                          result.get("score"))
    return record


@app.route("/gate/exit/fp/start", methods=["POST"])
def gate_exit_fp_start():
    trip_id = _trip_id(_json_body().get("trip_id"))
    if not trip_id:
        return jsonify({"error": "trip_id required"}), 400
    trip = db.get_trip(trip_id)
    if not trip or trip["status"] != "INSIDE":
        return jsonify({"error": "Trip not found or already closed"}), 404
    if not trip.get("fingerprint_template"):
        return jsonify({"no_fp": True, "msg": "No fingerprint enrolled — skipping FP check"})
    if not gm.start_fp_exit(trip_id):
        return jsonify({"error": "Fingerprint verification already in progress"}), 409
    thread = threading.Thread(
        target=gm.run_fp_exit,
        args=(trip_id, trip["fingerprint_template"], lambda result: None,
              _biometric_recorder(trip_id, gv.METHOD_FINGERPRINT)),
        daemon=True,
    )
    thread.start()
    return jsonify({"started": True})


@app.route("/gate/exit/fp/stream/<int:trip_id>")
def gate_exit_fp_stream(trip_id):
    def event_stream():
        q = gm.get_fp_exit_queue(trip_id)
        if not q:
            yield 'data: {"type":"error","text":"No active fingerprint session."}\n\n'
            return
        while True:
            try:
                msg = q.get(timeout=40)
                yield f"data: {json.dumps(msg)}\n\n"
                if msg["type"] in ("success", "notfound", "error"):
                    break
            except Exception:
                yield 'data: {"type":"error","text":"Session timed out."}\n\n'
                break
    return Response(event_stream(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/gate/exit/fp/cancel", methods=["POST"])
def gate_exit_fp_cancel():
    trip_id = _trip_id(_json_body().get("trip_id"))
    if trip_id:
        gm.cancel_fp_exit(trip_id)
    return jsonify({"cancelled": True})


@app.route("/gate/entry/confirm", methods=["POST"])
def gate_entry_confirm():
    data  = _json_body()
    staff = auth.current_staff()
    cid   = _capture_id(data.get("capture_id"))
    if cid:
        denied = _capture_denied(cid)
        if denied:
            return denied
    plate = anpr_post.normalize_plate_for_storage(_text(data.get("plate")))
    if not plate:
        return jsonify({"error": "Plate number is required"}), 400

    captured    = gm.temp_get(cid) if cid else {}
    auto_plate  = captured.get("plate_number", "") or ""
    plate_result = captured.get("plate_result") or {}
    auto_status  = plate_result.get("status", "NOT_DETECTED")
    auto_conf    = plate_result.get("overall_confidence", 0.0)

    if not auto_plate:
        plate_source = "manual_entry"
    elif plate != auto_plate:
        plate_source = "manual_correction"
    else:
        plate_source = "auto"
        # A reading the operator left untouched must still be explicitly
        # confirmed before it can grant entry when it wasn't CONFIRMED by
        # the pipeline itself — no low-confidence result is auto-accepted.
        if auto_status != "CONFIRMED" and not data.get("confirm_low_confidence"):
            return jsonify({
                "error": "Low-confidence plate reading — please review the plate crop "
                         "and confirm or correct it before logging entry.",
                "requires_confirmation": True,
                "status": auto_status,
            }), 409

    # Vehicle attributes: advisory. A missing or failed attribute model
    # leaves these columns NULL and never blocks the entry from being
    # logged, exactly as a missing plate model does not.
    attribute_result = va.VehicleAttributeResult.from_dict(
        captured.get("vehicle_attributes"))
    raw_corrections = data.get("attribute_corrections")
    corrections = {
        key: str(value).strip()
        for key, value in (raw_corrections if isinstance(raw_corrections, dict) else {}).items()
        if key in ("colour", "type", "brand") and str(value or "").strip()
    }
    # Only values from the closed label spaces are accepted — an operator
    # cannot silently widen a label space through this route.
    for key, value in list(corrections.items()):
        if value.lower() not in va.LABEL_SPACES[key]:
            return jsonify({
                "error": f"'{value}' is not a valid {key}. Choose one of: "
                         + ", ".join(va.LABEL_SPACES[key])}), 400
        corrections[key] = value.lower()
    attributes = db.build_attribute_columns(attribute_result, corrections)
    # Camera capture vs uploaded file. Advisory provenance, never an input to
    # any authorization rule — but an operator reviewing a trip must be able
    # to see that a photo was supplied rather than taken at the gate.
    attributes["vehicle_image_source"] = captured.get("vehicle_image_source") or "camera"

    # Driver identity and exit verification mode, from this capture's
    # server-side face/fingerprint results. Validated BEFORE the trip exists
    # (a registered driver cannot be given a passcode; a guest must set a
    # valid one), so a rejected entry leaves nothing half-saved.
    try:
        identity = gv.plan_entry(captured, data, staff)
    except gv.VerificationError as exc:
        return _verification_response(exc)

    trip_id = db.create_trip(
        plate_number         = plate,
        vehicle_photo_path   = captured.get("vehicle_photo"),
        face_photo_path      = captured.get("face_photo"),
        face_encoding        = captured.get("face_encoding"),
        fingerprint_template = captured.get("fingerprint_template"),
        notes                = _text(data.get("notes"))[:500],
        plate_source         = plate_source,
        plate_confidence     = auto_conf if plate_source == "auto" else None,
        attributes           = attributes,
        identity             = identity,
    )

    # Organise captured files into a per-trip folder
    trip_folder = os.path.join("gate_photos", f"trip_{trip_id}")
    os.makedirs(trip_folder, exist_ok=True)

    new_vehicle_path = None
    new_face_path    = None

    v_src = captured.get("vehicle_photo")
    if v_src and os.path.exists(v_src):
        new_vehicle_path = os.path.join(trip_folder, "vehicle.jpg")
        os.rename(v_src, new_vehicle_path)

    f_src = captured.get("face_photo")
    if f_src and os.path.exists(f_src):
        new_face_path = os.path.join(trip_folder, "face.jpg")
        os.rename(f_src, new_face_path)

    fp_template = captured.get("fingerprint_template")
    if fp_template:
        fp_path = os.path.join(trip_folder, "fingerprint.bin")
        with open(fp_path, "wb") as fh:
            fh.write(bytes(fp_template) if isinstance(fp_template, list) else fp_template)

    if new_vehicle_path or new_face_path:
        db.update_trip_photos(trip_id, new_vehicle_path, new_face_path)

    if cid:
        gm.temp_clear(cid)

    return jsonify({"ok": True, "trip_id": trip_id, "plate": plate,
                    "identity_status": identity["identity_status"],
                    "verification_mode": identity.get("verification_mode"),
                    "sms_fallback_available": gv.entry_identity_view(captured)
                    .get("sms_fallback_available")})


# ── GATE: EXIT ─────────────────────────────────────────

@app.route("/gate/exit")
def gate_exit():
    capture_id = gm.new_capture_id(owner=auth.current_staff().session_id)
    return render_template("gate_exit.html", capture_id=capture_id,
                           mock_delivery=vs.load().mock_delivery,
                           upload_max_side=_upload_max_side())


@app.route("/gate/exit/lookup")
def gate_exit_lookup():
    """
    Find the open trip for a plate and, when the exit capture produced
    attributes, compare them with what was recorded at entry.

    A disagreement sets `requires_operator_confirmation`, which the exit
    page surfaces through the SAME low-confidence confirmation dialog the
    plate stage already uses. It is recorded on the trip as evidence and it
    NEVER denies an exit: /gate/exit/confirm still requires a verified
    face/fingerprint match or the trip's fallback (SMS code / guest passcode).
    Given realistic brand accuracy, auto-denying on an attribute mismatch
    would strand legitimate residents weekly — and a cloned plate on a
    different vehicle is precisely the case that needs a human, not a relay.

    `verification` tells the page which control to show. It is derived from
    the identity stored on the trip at entry and carries no secrets.
    """
    plate = request.args.get("plate", "").strip().upper()
    if not plate:
        return jsonify({"error": "Plate required"}), 400
    capture_id = _capture_id(request.args.get("capture_id"))
    if capture_id:
        denied = _capture_denied(capture_id)
        if denied:
            return denied
    trip = db.get_open_trip(plate)
    if not trip:
        return jsonify({"found": False, "plate": plate})

    payload = {
        "found": True,
        "trip_id": trip["id"],
        "plate": trip["plate_number"],
        "entry_time": trip["entry_time"],
        "face_photo_url": gate_photo_url_filter(trip.get("face_photo_path")) or None,
        "vehicle_photo_url": gate_photo_url_filter(trip.get("vehicle_photo_path")) or None,
        "has_face": trip.get("face_encoding") is not None,
        "has_fingerprint": trip.get("fingerprint_template") is not None,
        "entry_attributes": _entry_attributes_payload(trip),
        "verification": gv.trip_verification_view(trip, auth.current_staff()),
    }

    stored = gm.temp_get(capture_id).get("vehicle_attributes") if capture_id else None
    if stored:
        exit_result = va.VehicleAttributeResult.from_dict(stored)
        comparison = _compare_attributes(trip, exit_result)
        db.record_exit_attributes(trip["id"], None, None)
        db.update_trip_attributes(trip["id"], {
            "exit_vehicle_color": (exit_result.colour.value
                                   if exit_result.colour.is_usable else None),
            "exit_vehicle_type": (exit_result.type.value
                                  if exit_result.type.is_usable else None),
            "exit_vehicle_brand": (exit_result.brand.value
                                   if exit_result.brand.is_usable else None),
            "color_match": comparison["flags"].get("colour"),
            "type_match": comparison["flags"].get("type"),
            "brand_match": comparison["flags"].get("brand"),
        })
        db.record_attribute_mismatch(trip["id"], comparison["mismatched"],
                                     comparison["reason"])
        payload["exit_attributes"] = exit_result.to_api_dict()
        payload["attribute_comparison"] = comparison
    return jsonify(payload)


def _compare_attributes(trip: dict, exit_result) -> dict:
    """
    Entry-vs-exit comparison. Advisory evidence for a human.

    An attribute is compared only when BOTH sides produced a usable value.
    Anything else is "not comparable", never a mismatch: an absent
    measurement is not evidence of a swapped vehicle.
    """
    column_for = {"colour": "vehicle_color", "type": "vehicle_type",
                  "brand": "vehicle_brand"}
    rows, mismatched, flags = [], [], {}

    for attribute, column in column_for.items():
        entry_value = (trip.get(column) or "").strip().lower() or None
        exit_value_obj = getattr(exit_result, attribute)
        exit_value = exit_value_obj.value if exit_value_obj.is_usable else None

        if not entry_value or not exit_value:
            reason = ("Not recorded at entry" if not entry_value
                      else "Not recognised at exit")
            rows.append({"attribute": attribute, "entry": entry_value,
                         "exit": exit_value, "match": None, "reason": reason})
            flags[attribute] = None
            continue

        match = entry_value == exit_value
        flags[attribute] = 1 if match else 0
        rows.append({"attribute": attribute, "entry": entry_value,
                     "exit": exit_value, "match": match,
                     "reason": ("matches the entry record" if match
                                else "differs from the entry record")})
        if not match:
            mismatched.append(attribute)

    return {
        "comparisons": rows,
        "flags": flags,
        "mismatched": mismatched,
        "reason": ("; ".join(f"{a}: {trip.get(column_for[a])} at entry vs "
                             f"{getattr(exit_result, a).value} at exit"
                             for a in mismatched) or None),
        # Routes to the operator-confirmation dialog. Never to a relay.
        "requires_operator_confirmation": bool(mismatched),
        "advisory_only": True,
    }


def _entry_attributes_payload(trip: dict) -> dict:
    """Stored entry attributes, shaped for the exit page. No filesystem paths."""
    return {
        "colour": trip.get("vehicle_color"),
        "colour_confidence": trip.get("vehicle_color_confidence"),
        "colour_source": trip.get("vehicle_color_source"),
        "type": trip.get("vehicle_type"),
        "type_confidence": trip.get("vehicle_type_confidence"),
        "type_source": trip.get("vehicle_type_source"),
        "brand": trip.get("vehicle_brand"),
        "brand_confidence": trip.get("vehicle_brand_confidence"),
        "brand_source": trip.get("vehicle_brand_source"),
        "status": trip.get("attr_status"),
        "coco_class": trip.get("attr_coco_class"),
    }


@app.route("/gate/exit/verify/start", methods=["POST"])
def gate_exit_verify_start():
    """Face comparison against this trip's entry capture; a match authorizes the exit."""
    trip_id = _trip_id(_json_body().get("trip_id"))
    if not trip_id:
        return jsonify({"error": "trip_id required"}), 400
    trip = db.get_trip(trip_id)
    if not trip or trip["status"] != "INSIDE":
        return jsonify({"error": "Trip not found or already closed"}), 404
    if not trip.get("face_encoding"):
        return jsonify({"error": "No face data on this entry record"}), 400
    if not gm.start_exit_session(trip_id):
        return jsonify({"error": "Verification already in progress"}), 409

    thread = threading.Thread(
        target=gm.run_exit_verify,
        args=(trip_id,
              trip["face_encoding"],
              lambda result: None,
              _biometric_recorder(trip_id, gv.METHOD_FACE)),
        daemon=True,
    )
    thread.start()
    return jsonify({"started": True})


@app.route("/gate/exit/verify/stream/<int:trip_id>")
def gate_exit_verify_stream(trip_id):
    def event_stream():
        q = gm.get_exit_queue(trip_id)
        if not q:
            yield 'data: {"type":"error","text":"No active verify session."}\n\n'
            return
        while True:
            try:
                msg = q.get(timeout=45)
                yield f"data: {json.dumps(msg)}\n\n"
                if msg["type"] in ("success", "denied", "error"):
                    break
            except Exception:
                yield 'data: {"type":"error","text":"Session timed out."}\n\n'
                break
    return Response(event_stream(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/gate/exit/last-result/<int:trip_id>")
def gate_exit_last_result(trip_id):
    """Return exit photo path so the UI can display the live capture."""
    trip = db.get_trip(trip_id)
    if not trip or not trip.get("exit_face_photo_path"):
        return jsonify({"exit_photo_url": None})
    return jsonify({"exit_photo_url": gate_photo_url_filter(trip["exit_face_photo_path"])})


@app.route("/gate/exit/confirm", methods=["POST"])
def gate_exit_confirm():
    """
    Close an open trip as GRANTED. Requires an unexpired exit authorization
    for THIS trip, created by THIS gate session, which exists only after:
      - a face or fingerprint match against THIS trip's own entry capture
        (/gate/exit/verify/start, /gate/exit/fp/start), or
      - the SMS code sent to the trip's registered driver was verified
        (/gate/exit/sms/verify — registered trips only), or
      - the guest passcode set at entry was verified
        (/gate/exit/passcode/verify — guest trips only).
    The authorization is consumed in the same transaction that closes the
    trip. The client sends only ids — file paths come from the server.
    """
    data    = _json_body()
    trip_id = _trip_id(data.get("trip_id"))
    if not trip_id:
        return jsonify({"error": "trip_id required", "code": "bad_input"}), 400

    staff = auth.current_staff()
    cid = _capture_id(data.get("capture_id"))
    extra_files = []
    if cid and gm.claim_capture(cid, staff.session_id):
        face = gm.temp_get(cid).get("face_photo")
        if face:
            extra_files.append(face)
    try:
        result = gv.confirm_exit(trip_id, staff, extra_files=extra_files)
    except gv.VerificationError as exc:
        return _verification_response(exc)
    if extra_files:
        gm.temp_clear(cid)
    return jsonify(result)


# ── GATE: TRIPS LIST ───────────────────────────────────

@app.route("/gate/trips")
def gate_trips():
    active  = db.get_active_trips()
    history = db.get_all_trips(limit=100)
    return render_template("gate_trips.html", active=active, history=history)


@app.route("/gate/trips/clear-history", methods=["POST"])
@auth.admin_required
def gate_trips_clear_history():
    deleted = db.clear_trip_history()
    flash(f"Trip history cleared — {deleted} completed record{'s' if deleted != 1 else ''} removed.", "success")
    return redirect(url_for("gate_trips"))


@app.route("/gate/trips/clear-all", methods=["POST"])
@auth.admin_required
def gate_trips_clear_all():
    deleted = db.clear_all_trips()
    flash(f"All trips cleared — {deleted} record{'s' if deleted != 1 else ''} removed.", "success")
    return redirect(url_for("gate_trips"))


@app.route("/gate/trips/<int:trip_id>/delete", methods=["POST"])
@auth.admin_required
def gate_trip_delete(trip_id):
    db.delete_trip(trip_id)
    flash("Trip record deleted.", "success")
    return redirect(url_for("gate_trips"))


@app.route("/gate/trips/<int:trip_id>/attributes", methods=["POST"])
def gate_trip_update_attributes(trip_id):
    """
    Operator correction of a stored trip's vehicle attributes.

    Only the three attribute values can be changed here: the plate, the
    biometrics and the trip status are untouchable through this route. Each
    corrected value is stored with `*_source` = manual_correction (or
    manual_entry when there was no automatic reading), so an edit is always
    distinguishable from a model output.
    """
    trip = db.get_trip(trip_id)
    if not trip:
        return jsonify({"error": "Trip not found"}), 404

    data = _json_body()
    column_for = {"colour": "vehicle_color", "type": "vehicle_type",
                  "brand": "vehicle_brand"}
    updates = {}
    for attribute, column in column_for.items():
        if attribute not in data:
            continue
        value = str(data.get(attribute) or "").strip().lower()
        if value and value not in va.LABEL_SPACES[attribute]:
            return jsonify({
                "error": f"'{value}' is not a valid {attribute}. Choose one of: "
                         + ", ".join(va.LABEL_SPACES[attribute])}), 400
        had_automatic = bool(trip.get(column)) and \
            trip.get(f"{column}_source") == va.SOURCE_AUTO
        updates[column] = value or None
        updates[f"{column}_confidence"] = None
        updates[f"{column}_votes"] = None
        updates[f"{column}_source"] = (
            (va.SOURCE_MANUAL_CORRECTION if had_automatic else va.SOURCE_MANUAL_ENTRY)
            if value else None)

    if not updates:
        return jsonify({"error": "No attribute values supplied"}), 400

    db.update_trip_attributes(trip_id, updates)
    return jsonify({"ok": True, "trip_id": trip_id,
                    "attributes": _entry_attributes_payload(db.get_trip(trip_id))})


# ── GATE: SERVE PHOTOS ─────────────────────────────────

def _safe_gate_path(base_dir: str, filename: str):
    """
    Resolve `filename` inside `base_dir`, or None if it escapes.

    send_from_directory already refuses traversal, but gate photos can
    contain personal information, so the check is made explicit and applied
    before any filesystem access. Absolute paths, "..", and symlinks that
    resolve outside the directory are all rejected.
    """
    if not filename or os.path.isabs(filename) or "\x00" in filename:
        return None
    candidate = os.path.realpath(os.path.join(base_dir, filename))
    root = os.path.realpath(base_dir)
    if candidate != root and not candidate.startswith(root + os.sep):
        return None
    return candidate


@app.route("/gate/photo/<path:filename>")
def gate_photo(filename):
    photos_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gate_photos")
    if _safe_gate_path(photos_dir, filename) is None:
        return jsonify({"error": "Not found"}), 404
    return send_from_directory(photos_dir, filename)


@app.route("/gate/debug/<path:filename>")
def gate_debug_photo(filename):
    """
    Serve ANPR debug artifacts (raw/rectified crops, preprocessing
    variants). Gate images can contain personal information, so this route
    only works while plate_debug_mode is enabled in config — it is not a
    permanent public path, and debug artifacts are cleaned up automatically
    (see anpr debug cleanup in gate_manager / scripts docs).
    """
    if not cfg.get("plate_debug_mode", False):
        return jsonify({"error": "Debug mode is disabled"}), 404
    debug_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gate_photos", "debug")
    if _safe_gate_path(debug_dir, filename) is None:
        return jsonify({"error": "Not found"}), 404
    return send_from_directory(debug_dir, filename)


# ── EXIT FALLBACK: SMS OTP (registered) / PASSCODE (guest) ─────────────────
# For when face and fingerprint fail. The fallback is fixed by the identity
# stored on the trip at entry; every route below re-checks it server-side
# (gate_verification._open_trip), so calling the other route — or a hidden
# button — gets a 409, not a bypass. The destination number is always read
# from the trip's linked holder record, never from the request.

def _send_sms_otp():
    trip_id = _trip_id(_json_body().get("trip_id"))
    if not trip_id:
        return jsonify({"error": "trip_id required", "code": "bad_input"}), 400
    try:
        return jsonify(gv.request_sms_otp(trip_id, auth.current_staff()))
    except gv.VerificationError as exc:
        return _verification_response(exc)


@app.route("/gate/exit/sms/send", methods=["POST"])
def gate_exit_sms_send():
    """Text a Robase code to the trip's registered driver."""
    return _send_sms_otp()


@app.route("/gate/exit/sms/resend", methods=["POST"])
def gate_exit_sms_resend():
    """Send a new code; a code Robase accepts replaces the earlier ones."""
    return _send_sms_otp()


@app.route("/gate/exit/sms/verify", methods=["POST"])
def gate_exit_sms_verify():
    """Check the driver's SMS code with Robase (registered trips only)."""
    data = _json_body()
    trip_id = _trip_id(data.get("trip_id"))
    if not trip_id:
        return jsonify({"valid": False, "error": "trip_id required", "code": "bad_input"}), 400
    try:
        return jsonify(gv.verify_sms_otp(trip_id, data.get("code"), auth.current_staff()))
    except gv.VerificationError as exc:
        return _verification_response(exc, valid=False)


@app.route("/gate/exit/sms/status")
def gate_exit_sms_status():
    """Delivery state of this screen's active code (informational)."""
    trip_id = _trip_id(request.args.get("trip_id"))
    if not trip_id:
        return jsonify({"error": "trip_id required", "code": "bad_input"}), 400
    return jsonify(gv.sms_delivery_status(trip_id, auth.current_staff()))


@app.route("/gate/exit/passcode/verify", methods=["POST"])
def gate_exit_passcode_verify():
    """Verify the passcode a guest driver set at entry (guest trips only)."""
    data = _json_body()
    trip_id = _trip_id(data.get("trip_id"))
    if not trip_id:
        return jsonify({"valid": False, "error": "trip_id required", "code": "bad_input"}), 400
    try:
        return jsonify(gv.verify_passcode(trip_id, data.get("passcode"), auth.current_staff()))
    except gv.VerificationError as exc:
        return _verification_response(exc, valid=False)


# ── ADMIN: TRIP RECONCILIATION ─────────────────────────────────────────────

def _sms_ready_holders():
    """Holders a trip can be assigned to: only those SMS codes can reach."""
    seen, out = set(), []
    for holder in db.get_all_holders():
        if holder["id"] in seen:
            continue
        seen.add(holder["id"])
        status = gv.holder_sms_status(holder)
        if status["phone_valid"]:
            out.append({"id": holder["id"], "phone_hint": status["phone_hint"],
                        "name": f"{holder['surname']}, {holder['first_name']}"})
    return out


@app.route("/admin/verification")
@auth.admin_required
def admin_verification():
    return render_template("admin_verification.html",
                           trips=gv.trips_needing_review(),
                           sms_ready_holders=_sms_ready_holders(),
                           events=gv.recent_security_events(40))


@app.route("/admin/trips/<int:trip_id>/reconcile", methods=["POST"])
@auth.admin_required
def admin_trip_reconcile(trip_id):
    try:
        gv.reconcile_trip(trip_id, request.form.get("action", ""), auth.current_staff(),
                          holder_id=_trip_id(request.form.get("holder_id")),
                          note=request.form.get("note", ""))
    except gv.VerificationError as exc:
        flash(exc.message, "error")
    else:
        flash(f"Trip #{trip_id} updated.", "success")
    return redirect(url_for("admin_verification"))


@app.context_processor
def inject_review_count():
    """Trips awaiting reconciliation or unlocking, for the nav badge."""
    staff = auth.current_staff()
    if staff is None or not staff.is_admin:
        return {"review_count": 0}
    conn = db.get_connection()
    try:
        trips = conn.execute("SELECT COUNT(*) FROM trips WHERE status='INSIDE' AND "
                             "(verification_mode IS NULL OR verification_locked=1)").fetchone()[0]
    finally:
        conn.close()
    return {"review_count": trips}


# ── SETTINGS ───────────────────────────────────────────

@app.route("/settings")
def settings():
    cameras = fm.list_cameras()
    active_index = cfg.get("camera_index", 0)
    # Model readiness is read from disk only — opening Settings never loads
    # an attribute model into memory.
    staff = auth.current_staff()
    return render_template("settings.html", cameras=cameras, active_index=active_index,
                           attr_status=gm.attribute_model_status(),
                           show_wifi=bool(staff and staff.is_admin))


@app.route("/settings/camera/select", methods=["POST"])
def settings_camera_select():
    index = request.form.get("index")
    if index is None:
        flash("No camera index provided.", "error")
        return redirect(url_for("settings"))
    fm.switch_camera(int(index))
    flash(f"Camera switched to index {index}. Feed restarting…", "success")
    return redirect(url_for("settings"))


# ── INTERNET WI-FI (USB adapter uplink) ────────────────
# Every change is made by the root-owned helper (see wifi_uplink.py and
# README_WIFI_UPLINK.md), which only ever touches the configured USB adapter.

def _wifi_api(view):
    """Administrators only, and the CSRF token is required even on reads."""
    @functools.wraps(view)
    def wrapper(*args, **kwargs):
        if not auth._csrf_ok():
            return jsonify({"error": "Security token missing or invalid — reload the page "
                                     "and try again.", "code": "csrf"}), 400
        try:
            return view(*args, **kwargs)
        except wifi.WifiError as exc:
            return jsonify({"error": exc.message, **exc.as_dict()}), exc.http_status
    return auth.admin_required(wrapper)


@app.route("/settings/wifi/status")
@_wifi_api
def wifi_status():
    return jsonify(wifi.get_service().status(force=request.args.get("refresh") == "1"))


@app.route("/settings/wifi/scan", methods=["POST"])
@_wifi_api
def wifi_scan():
    return jsonify(wifi.get_service().scan())


@app.route("/settings/wifi/connect", methods=["POST"])
@_wifi_api
def wifi_connect():
    job = wifi.get_service().start_connect(_json_body(), auth.current_staff())
    return jsonify({"job": job}), 202


@app.route("/settings/wifi/reconnect", methods=["POST"])
@_wifi_api
def wifi_reconnect():
    job = wifi.get_service().start_reconnect(_json_body().get("uuid"), auth.current_staff())
    return jsonify({"job": job}), 202


@app.route("/settings/wifi/jobs/<job_id>")
@_wifi_api
def wifi_job(job_id):
    job = wifi.get_service().job(job_id)
    if job is None:
        return jsonify({"error": "That Wi-Fi operation is no longer tracked.", "code": "not_found"}), 404
    return jsonify({"job": job})


@app.route("/settings/wifi/disconnect", methods=["POST"])
@_wifi_api
def wifi_disconnect():
    return jsonify(wifi.get_service().disconnect(auth.current_staff()))


@app.route("/settings/wifi/forget", methods=["POST"])
@_wifi_api
def wifi_forget():
    return jsonify(wifi.get_service().forget(_json_body().get("uuid"), auth.current_staff()))


@app.route("/settings/wifi/connectivity")
@_wifi_api
def wifi_connectivity():
    return jsonify(wifi.get_service().connectivity(force=request.args.get("refresh") == "1"))


@app.route("/api/cameras")
def api_cameras():
    return jsonify(fm.list_cameras())


@app.route("/api/cameras/<int:index>/snapshot")
def api_camera_snapshot(index):
    """Capture a single test frame from a specific camera index and return as JPEG."""
    if not fm.OPENCV_AVAILABLE:
        return Response("OpenCV not available", status=503)
    import cv2
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        return Response("Camera not available", status=503)
    # Warm up — discard first few frames
    for _ in range(3):
        cap.read()
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        return Response("Could not read frame", status=503)
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ok:
        return Response("Encode failed", status=500)
    return Response(buf.tobytes(), mimetype="image/jpeg")


# ── REPORTS ────────────────────────────────────────────

@app.route("/reports")
def reports():
    stats = db.get_stats()
    all_holders = db.get_all_holders()
    expired = [h for h in all_holders if h.get("expiry_date") and h["expiry_date"] < datetime.now().strftime("%Y-%m-%d")]
    return render_template("reports.html", stats=stats, expired=expired,
                           attr_report=_attribute_report())


def _attribute_report(limit=500) -> dict:
    """
    Coverage of attribute recognition across recent trips.

    Reports how often each attribute was RECOGNISED and how often an
    operator corrected it — not how often it was CORRECT. Accuracy needs
    ground-truth labels: see
    'python tools/build_local_plate_test_set.py evaluate-attributes'.
    """
    trips = db.get_all_trips(limit=limit)
    total = len(trips)
    column_for = {"colour": "vehicle_color", "type": "vehicle_type",
                  "brand": "vehicle_brand"}
    match_column = {"colour": "color_match", "type": "type_match",
                    "brand": "brand_match"}

    recognised = {a: 0 for a in column_for}
    corrected = {a: 0 for a in column_for}
    compared = {a: 0 for a in column_for}
    differed = {a: 0 for a in column_for}
    not_localised = 0

    for trip in trips:
        if trip.get("attr_status") == "VEHICLE_NOT_LOCALISED":
            not_localised += 1
        for attribute, column in column_for.items():
            if trip.get(column):
                if trip.get(f"{column}_source") == "auto":
                    recognised[attribute] += 1
                else:
                    corrected[attribute] += 1
            flag = trip.get(match_column[attribute])
            if flag is not None:
                compared[attribute] += 1
                if flag == 0:
                    differed[attribute] += 1

    def pct(n):
        return round(n * 100.0 / total, 1) if total else 0.0

    return {
        "total_trips": total,
        "recognised": recognised,
        "recognised_pct": {k: pct(v) for k, v in recognised.items()},
        "operator_set": corrected,
        "compared": compared,
        "differed": differed,
        "vehicle_not_localised": not_localised,
    }


_EXIT_METHOD_LABELS = {
    "face": "face match", "fingerprint": "fingerprint match", "sms_otp": "SMS code",
    "passcode": "guest passcode",
    # Values written by earlier versions, kept readable in the history.
    "PASSCODE": "guest passcode", "TELEGRAM_OTP": "Telegram code (retired)",
}


@app.template_filter("exit_method_label")
def exit_method_label_filter(method):
    return _EXIT_METHOD_LABELS.get(method or "", method or "")


@app.template_filter("epoch_time")
def epoch_time_filter(value):
    """Unix timestamp (security tables) -> local 'YYYY-MM-DD HH:MM:SS'."""
    try:
        return datetime.fromtimestamp(float(value)).strftime("%Y-%m-%d %H:%M:%S")
    except (TypeError, ValueError, OSError):
        return "—"


@app.template_filter("gate_photo_url")
def gate_photo_url_filter(path):
    """Convert a stored gate photo path → a /gate/photo/... URL."""
    if not path:
        return ""
    # Strip leading gate_photos/ prefix (stored paths always start with it)
    rel = path
    for prefix in ("gate_photos/", "gate_photos\\"):
        if rel.startswith(prefix):
            rel = rel[len(prefix):]
            break
    return "/gate/photo/" + rel.replace("\\", "/")


#: Fixed swatch table keyed by the CLOSED colour label space. Never derived
#: from model output, so a label can carry no styling payload.
_COLOUR_SWATCHES = {
    "white": "#f3f4f6", "black": "#1f2937", "silver": "#c0c4c9", "grey": "#6b7280",
    "red": "#ef4444", "blue": "#3b82f6", "green": "#22c55e", "gold": "#d4af37",
    "brown": "#78503c", "orange": "#f97316", "yellow": "#eab308",
    "unknown": "#d1d5db",
}


@app.template_filter("colour_swatch")
def colour_swatch_filter(label):
    return _COLOUR_SWATCHES.get((label or "").strip().lower(), _COLOUR_SWATCHES["unknown"])


@app.template_filter("attr_label")
def attr_label_filter(label):
    """`mercedes-benz` -> `Mercedes Benz`, for display only."""
    if not label:
        return "—"
    return str(label).replace("_", " ").replace("-", " ").title()


@app.template_filter("source_badge")
def source_badge_filter(source):
    """Provenance badge: an operator edit must be visible at a glance."""
    return {
        "auto": ("badge-grey", "auto"),
        "manual_correction": ("badge-amber", "corrected"),
        "manual_entry": ("badge-amber", "operator"),
    }.get(source, ("badge-grey", ""))


@app.template_filter("confidence_pct")
def confidence_pct_filter(value):
    """0.842 -> '84%'. Blank for a missing confidence (e.g. operator-set)."""
    try:
        return f"{float(value) * 100:.0f}%"
    except (TypeError, ValueError):
        return ""


@app.template_filter("match_badge")
def match_badge_filter(value):
    """Tri-state match column -> (badge class, text)."""
    if value == 1:
        return ("badge-green", "Match")
    if value == 0:
        return ("badge-red", "Differs")
    return ("badge-grey", "n/a")


@app.context_processor
def inject_now():
    return {"now": datetime.now().strftime("%Y-%m-%d")}


def _port_available(host: str, port: int) -> bool:
    """False when something is already listening on host:port."""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind((host, port))
        except OSError:
            return False
    return True


if __name__ == "__main__":
    _host = os.environ.get("FLASK_RUN_HOST", "0.0.0.0")
    _port = int(os.environ.get("FLASK_RUN_PORT", "5000"))
    # Checked before any model is loaded, so a second copy exits cleanly
    # instead of aborting while the warm-up threads are still starting.
    if not _port_available(_host, _port):
        print(f"[startup] Port {_port} is already in use — another copy of the app (for "
              f"example its systemd service) is probably running. Stop or restart that "
              f"copy instead (find it with: ss -ltnp 'sport = :{_port}'), or set "
              f"FLASK_RUN_PORT.")
        raise SystemExit(1)

    _settings = vs.load()
    print(f"[startup] APP_ENV={_settings.app_env}; SMS OTP fallback (Robase): "
          + ("MOCK — development only, no SMS is sent"
             if _settings.mock_delivery else "live"))
    try:
        import robase_client as _robase
        _robase.get_client(_settings)
    except vs.ConfigurationError as _exc:
        print(f"[startup]   SMS OTP fallback disabled until configured: {_exc}")
    if auth.count_users() == 0:
        print("[startup]   No staff accounts yet — create one with: "
              "python scripts/manage_staff.py create <username> --role admin")

    # Pre-load the OCR backend in the background so the first plate scan doesn't stall
    gm.warm_up_ocr()
    # The COCO vehicle model too: it is loaded before Auto-Detect's first SSE
    # message, so a cold load makes the button look unresponsive.
    gm.warm_up_vehicle_detector()
    # Prune any stale ANPR debug artifacts from previous runs before serving requests
    gm.cleanup_debug_artifacts()
    # Report vehicle-attribute model readiness, and optionally warm the
    # models in the background. A missing attribute model is logged and the
    # gate starts normally — attribute recognition is never required.
    # Report attribute-model readiness and warm the backbone in the
    # background. A missing attribute model is logged and the gate starts
    # normally — attributes are never required.
    _attr = gm.attribute_model_status()
    _clf = _attr.get("classifier", {})
    print(f"[startup] Vehicle attributes: enabled={_attr.get('enabled')} "
          f"model={'trained' if _clf.get('trained') else 'NOT TRAINED'} "
          f"({_clf.get('version')})")
    if not _clf.get("model_present"):
        print("[startup]   No attribute checkpoint — colour falls back to the HSV "
              "baseline; body style and brand report UNKNOWN. Train a head with "
              "scripts/train_vehicle_attributes.py.")
    if not _attr.get("logo_detector", {}).get("model_present"):
        print("[startup]   No vehicle-logo detector — brand stays UNKNOWN "
              "(never guessed from the body crop).")
    gm.warm_up_attributes()
    # The Werkzeug debugger executes code from the browser, so it is only
    # available when explicitly requested in development.
    _debug = _settings.app_env == "development" and \
        os.environ.get("FLASK_DEBUG", "").strip().lower() in ("1", "true", "yes")
    app.run(host=_host, port=_port, debug=_debug, use_reloader=False, threaded=True)