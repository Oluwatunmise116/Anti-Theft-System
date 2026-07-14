from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_from_directory, Response
import os
import threading
import json
import time
import database as db
import fingerprint_manager as fp
import face_manager as fm
import gate_manager as gm
import config as cfg
from werkzeug.utils import secure_filename
from datetime import datetime

app = Flask(__name__)
app.secret_key = "license_admin_secret_2024"

UPLOAD_FOLDER = "photos"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


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
    return render_template("view_holder.html", record=record, photos=photos, has_fingerprint=has_fingerprint)


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
        db.update_holder(holder_id, data)

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
    db.delete_holder(holder_id)
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
    capture_id = gm.new_capture_id()
    return render_template("gate_entry.html", capture_id=capture_id)


@app.route("/gate/model/status")
def gate_model_status():
    return jsonify({
        "plate": gm.plate_model_status(),
        "face":  fm.face_model_status(),
    })


@app.route("/gate/model/download", methods=["POST"])
def gate_model_download():
    model = request.json.get("model", "plate") if request.is_json else "plate"
    if model == "face":
        ok = fm.download_face_models()
        fm._get_face_cv_models()   # load into memory after download
        return jsonify({"ok": ok, **fm.face_model_status()})
    ok = gm.download_plate_model()
    return jsonify({"ok": ok, **gm.plate_model_status()})


@app.route("/gate/entry/vehicle/snap", methods=["POST"])
def gate_entry_vehicle_snap():
    """Grab a single camera frame and save it — no analysis yet."""
    capture_id = request.json.get("capture_id")
    if not capture_id:
        return jsonify({"error": "Missing capture_id"}), 400
    result = gm.snap_vehicle_frame(capture_id)
    return jsonify(result)


@app.route("/gate/entry/vehicle/start", methods=["POST"])
def gate_entry_vehicle_start():
    capture_id = request.json.get("capture_id")
    if not capture_id:
        return jsonify({"error": "Missing capture_id"}), 400
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
    capture_id = request.json.get("capture_id")
    if not capture_id:
        return jsonify({"error": "Missing capture_id"}), 400
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
    capture_id = request.json.get("capture_id")
    if not capture_id:
        return jsonify({"error": "Missing capture_id"}), 400
    if not gm.start_face_session(capture_id):
        return jsonify({"error": "Face capture already in progress"}), 409

    # Build enriched holder list for DB lookup after face capture
    holders_with_photos = []
    for h in db.get_all_holders():
        photos = db.get_photos(h["id"])
        if photos.get("passport"):
            holders_with_photos.append({
                "holder_id":      h["id"],
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
    capture_id = request.json.get("capture_id")
    if not capture_id:
        return jsonify({"error": "Missing capture_id"}), 400
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
    capture_id = request.json.get("capture_id")
    if capture_id:
        gm.cancel_fp_entry(capture_id)
    return jsonify({"cancelled": True})


# ── GATE: EXIT FINGERPRINT VERIFY ──────────────────────

@app.route("/gate/exit/fp/start", methods=["POST"])
def gate_exit_fp_start():
    trip_id = request.json.get("trip_id")
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
        args=(trip_id, trip["fingerprint_template"], lambda _: None),
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
    trip_id = request.json.get("trip_id")
    if trip_id:
        gm.cancel_fp_exit(trip_id)
    return jsonify({"cancelled": True})


@app.route("/gate/entry/confirm", methods=["POST"])
def gate_entry_confirm():
    data = request.json
    cid  = data.get("capture_id")
    plate = data.get("plate", "").strip().upper()
    if not plate:
        return jsonify({"error": "Plate number is required"}), 400

    captured = gm.temp_get(cid)
    trip_id = db.create_trip(
        plate_number         = plate,
        vehicle_photo_path   = captured.get("vehicle_photo"),
        face_photo_path      = captured.get("face_photo"),
        face_encoding        = captured.get("face_encoding"),
        fingerprint_template = captured.get("fingerprint_template"),
        notes                = data.get("notes", ""),
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

    gm.temp_clear(cid)

    # Store driver-set passcode if provided
    passcode = data.get("passcode", "").strip()
    if passcode and passcode.isdigit() and 4 <= len(passcode) <= 8:
        db.set_trip_passcode(trip_id, passcode)

    return jsonify({"ok": True, "trip_id": trip_id, "plate": plate})


# ── GATE: EXIT ─────────────────────────────────────────

@app.route("/gate/exit")
def gate_exit():
    capture_id = gm.new_capture_id()
    return render_template("gate_exit.html", capture_id=capture_id)


@app.route("/gate/exit/lookup")
def gate_exit_lookup():
    plate = request.args.get("plate", "").strip().upper()
    if not plate:
        return jsonify({"error": "Plate required"}), 400
    trip = db.get_open_trip(plate)
    if not trip:
        return jsonify({"found": False, "plate": plate})
    return jsonify({
        "found": True,
        "trip_id": trip["id"],
        "plate": trip["plate_number"],
        "entry_time": trip["entry_time"],
        "face_photo_url": gate_photo_url_filter(trip.get("face_photo_path")) or None,
        "vehicle_photo_url": gate_photo_url_filter(trip.get("vehicle_photo_path")) or None,
        "has_face": trip.get("face_encoding") is not None,
        "has_fingerprint": trip.get("fingerprint_template") is not None,
    })


@app.route("/gate/exit/verify/start", methods=["POST"])
def gate_exit_verify_start():
    trip_id = request.json.get("trip_id")
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
              lambda r: None),
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
        return jsonify({"exit_photo_url": None, "exit_photo_path": None})
    return jsonify({
        "exit_photo_url":  gate_photo_url_filter(trip["exit_face_photo_path"]),
        "exit_photo_path": trip["exit_face_photo_path"],
    })


@app.route("/gate/exit/confirm", methods=["POST"])
def gate_exit_confirm():
    """
    Close an open trip as GRANTED. The client cannot dictate the decision —
    a "GRANTED" here requires a server-side exit authorization already
    recorded for this exact trip_id, produced only by:
      - a face match against THIS trip's own stored face_encoding
        (gm.run_exit_verify), or
      - a fingerprint match against THIS trip's own stored fingerprint
        template (gm.run_fp_exit), or
      - a passcode verified against THIS trip's own passcode
        (/gate/exit/otp/verify).
    A person merely existing in the general holders database is never
    sufficient — see gm.authorize_exit / gm.consume_exit_authorization.
    """
    data    = request.json or {}
    trip_id = data.get("trip_id")
    if not trip_id:
        return jsonify({"error": "trip_id required"}), 400
    trip_id = int(trip_id)

    trip = db.get_trip(trip_id)
    if not trip or trip["status"] != "INSIDE":
        return jsonify({"error": "Trip not found or already closed"}), 404

    auth = gm.consume_exit_authorization(trip_id)
    if not auth:
        return jsonify({
            "error": "No verified exit authorization for this trip — "
                     "the biometric must match this trip's own driver, "
                     "or a valid trip passcode must be verified first."
        }), 403

    exit_photo    = data.get("exit_photo_path") or auth.get("exit_photo")
    face_distance = auth.get("face_distance")

    db.close_trip(
        trip_id              = trip_id,
        exit_result          = "GRANTED",
        exit_face_photo_path = exit_photo,
        face_distance        = face_distance,
    )
    return jsonify({"ok": True, "decision": "GRANTED", "method": auth.get("method")})


# ── GATE: TRIPS LIST ───────────────────────────────────

@app.route("/gate/trips")
def gate_trips():
    active  = db.get_active_trips()
    history = db.get_all_trips(limit=100)
    return render_template("gate_trips.html", active=active, history=history)


@app.route("/gate/trips/clear-history", methods=["POST"])
def gate_trips_clear_history():
    deleted = db.clear_trip_history()
    flash(f"Trip history cleared — {deleted} completed record{'s' if deleted != 1 else ''} removed.", "success")
    return redirect(url_for("gate_trips"))


@app.route("/gate/trips/clear-all", methods=["POST"])
def gate_trips_clear_all():
    deleted = db.clear_all_trips()
    flash(f"All trips cleared — {deleted} record{'s' if deleted != 1 else ''} removed.", "success")
    return redirect(url_for("gate_trips"))


@app.route("/gate/trips/<int:trip_id>/delete", methods=["POST"])
def gate_trip_delete(trip_id):
    db.delete_trip(trip_id)
    flash("Trip record deleted.", "success")
    return redirect(url_for("gate_trips"))


# ── GATE: SERVE PHOTOS ─────────────────────────────────

@app.route("/gate/photo/<path:filename>")
def gate_photo(filename):
    photos_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gate_photos")
    return send_from_directory(photos_dir, filename)


# ── TRIP PASSCODE ──────────────────────────────────────

@app.route("/gate/exit/otp/verify", methods=["POST"])
def gate_exit_otp_verify():
    data    = request.json or {}
    trip_id = data.get("trip_id")
    code    = str(data.get("code", "")).strip()
    if not trip_id:
        return jsonify({"valid": False, "error": "trip_id required"}), 400
    if not code or not code.isdigit() or not (4 <= len(code) <= 8):
        return jsonify({"valid": False, "error": "Enter a 4–8 digit passcode"}), 400
    if not db.verify_trip_passcode(int(trip_id), code):
        return jsonify({"valid": False, "error": "Incorrect passcode"}), 400
    # Proof this specific trip's delegation passcode was verified — not just
    # a general holders-database match on the delegate's own biometrics.
    gm.authorize_exit(int(trip_id), "passcode")
    return jsonify({"valid": True})


# ── SETTINGS ───────────────────────────────────────────

@app.route("/settings")
def settings():
    cameras = fm.list_cameras()
    active_index = cfg.get("camera_index", 0)
    return render_template("settings.html", cameras=cameras, active_index=active_index)


@app.route("/settings/camera/select", methods=["POST"])
def settings_camera_select():
    index = request.form.get("index")
    if index is None:
        flash("No camera index provided.", "error")
        return redirect(url_for("settings"))
    fm.switch_camera(int(index))
    flash(f"Camera switched to index {index}. Feed restarting…", "success")
    return redirect(url_for("settings"))


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
    return render_template("reports.html", stats=stats, expired=expired)


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


@app.context_processor
def inject_now():
    return {"now": datetime.now().strftime("%Y-%m-%d")}


if __name__ == "__main__":
    # Pre-load EasyOCR in the background so the first plate scan doesn't stall
    gm.warm_up_ocr()
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False, threaded=True)