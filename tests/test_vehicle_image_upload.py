"""
Uploading a still image as an alternative to the live camera.

The upload feeds exactly the same analysis pipeline as a camera capture, so
what needs testing is the boundary: what is accepted, what is refused, and
whether the difference in PROVENANCE survives into the trip record. An
uploaded photo is not evidence the vehicle was at the gate.
"""
import io

import cv2
import numpy as np
import pytest

import app as application
import database as db
import gate_manager as gm
from gate_helpers import staff_client


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "upload.db"))
    db.init_db()
    application.app.config["TESTING"] = True
    with staff_client(application.app) as c:
        yield c


def encode(width=640, height=480, fmt=".jpg"):
    rng = np.random.default_rng(3)
    image = rng.integers(0, 255, (height, width, 3), dtype=np.uint8)
    ok, buffer = cv2.imencode(fmt, image)
    assert ok
    return buffer.tobytes()


def upload(client, capture_id, data, filename="car.jpg"):
    return client.post("/gate/entry/vehicle/upload",
                       data={"capture_id": capture_id,
                             "image": (io.BytesIO(data), filename)},
                       content_type="multipart/form-data")


# ── accepted ──────────────────────────────────────────────────────────────

def test_a_jpeg_upload_is_accepted_and_becomes_the_vehicle_photo(client, tmp_path):
    cid = gm.new_capture_id()
    response = upload(client, cid, encode())
    assert response.status_code == 200
    body = response.get_json()
    assert body["ok"] is True
    assert body["width"] == 640 and body["height"] == 480
    assert body["image_source"] == "upload"
    assert body["url"].endswith(f"vehicle_{cid}.jpg")

    session = gm.temp_get(cid)
    assert session["vehicle_image_source"] == "upload"
    assert session["vehicle_photo"].endswith(f"vehicle_{cid}.jpg")


def test_a_png_upload_is_accepted(client):
    cid = gm.new_capture_id()
    assert upload(client, cid, encode(fmt=".png"), "car.png").status_code == 200


def test_the_stored_file_is_re_encoded_as_jpeg(client):
    """
    Re-encoding normalises the format and strips EXIF, so a PNG (or a file
    with embedded metadata) cannot be stored verbatim under a .jpg name.
    """
    cid = gm.new_capture_id()
    upload(client, cid, encode(fmt=".png"), "car.png")
    path = gm.temp_get(cid)["vehicle_photo"]
    with open(path, "rb") as fh:
        assert fh.read(3) == b"\xff\xd8\xff"        # JPEG SOI marker


def test_a_new_upload_clears_the_previous_analysis(client):
    """Stale plate or attribute results must not survive a new image."""
    cid = gm.new_capture_id()
    gm.temp_set(cid, "plate_result", {"plate_number": "OLD123AB"})
    gm.temp_set(cid, "vehicle_attributes", {"colour": {"value": "red"}})
    upload(client, cid, encode())
    session = gm.temp_get(cid)
    assert session["plate_result"] is None
    assert session["vehicle_attributes"] is None


# ── refused ───────────────────────────────────────────────────────────────

def test_a_missing_capture_id_is_refused(client):
    assert upload(client, "", encode()).status_code == 400


def test_a_request_with_no_file_is_refused(client):
    response = client.post("/gate/entry/vehicle/upload",
                           data={"capture_id": gm.new_capture_id()},
                           content_type="multipart/form-data")
    assert response.status_code == 400
    assert "No image file" in response.get_json()["error"]


def test_a_non_image_file_is_refused(client):
    cid = gm.new_capture_id()
    response = upload(client, cid, b"#!/bin/sh\nrm -rf /\n", "payload.jpg")
    assert response.status_code == 400
    assert "not a readable image" in response.get_json()["error"]
    # Nothing was stored.
    assert "vehicle_photo" not in gm.temp_get(cid)


def test_a_disallowed_extension_is_refused(client):
    assert upload(client, gm.new_capture_id(), encode(), "car.svg").status_code == 400


def test_a_tiny_image_is_refused(client):
    response = upload(client, gm.new_capture_id(), encode(16, 16))
    assert response.status_code == 400
    assert "too small" in response.get_json()["error"]


def test_an_oversized_file_is_refused(client, monkeypatch):
    import config as cfg

    real_load = cfg.load
    monkeypatch.setattr(cfg, "load", lambda: {**real_load(), "gate_upload_max_mb": 0.0001})
    response = upload(client, gm.new_capture_id(), encode(1280, 720))
    assert response.status_code == 400
    assert "larger than" in response.get_json()["error"]


def test_an_excessive_resolution_is_refused(client, monkeypatch):
    """Guards against a decompression bomb sized to pass the byte limit."""
    import config as cfg

    real_load = cfg.load
    monkeypatch.setattr(cfg, "load", lambda: {**real_load(), "gate_upload_max_pixels": 1000})
    response = upload(client, gm.new_capture_id(), encode(640, 480))
    assert response.status_code == 400
    assert "resolution is too large" in response.get_json()["error"]


def test_upload_can_be_disabled(client, monkeypatch):
    import config as cfg

    real_load = cfg.load
    monkeypatch.setattr(cfg, "load", lambda: {**real_load(), "gate_upload_enabled": False})
    response = upload(client, gm.new_capture_id(), encode())
    assert response.status_code == 400
    assert "disabled" in response.get_json()["error"]


def test_a_crafted_filename_cannot_escape_the_photo_directory(client):
    """The stored name is constructed server-side, never taken from the upload."""
    cid = gm.new_capture_id()
    response = upload(client, cid, encode(), "../../../../etc/passwd.jpg")
    assert response.status_code == 200
    stored = gm.temp_get(cid)["vehicle_photo"]
    assert stored.endswith(f"vehicle_{cid}.jpg")
    assert ".." not in stored


# ── provenance ────────────────────────────────────────────────────────────

def test_an_uploaded_image_is_recorded_as_such_on_the_trip(client):
    cid = gm.new_capture_id()
    upload(client, cid, encode())
    gm.temp_set(cid, "plate_number", "KJA456GH")
    gm.temp_set(cid, "plate_result", {"status": "CONFIRMED", "overall_confidence": 0.9})

    response = client.post("/gate/entry/confirm",
                           json={"capture_id": cid, "plate": "KJA456GH",
                                     # no face/fingerprint search in these tests
                                     "accept_unresolved": True})
    trip = db.get_trip(response.get_json()["trip_id"])
    assert trip["vehicle_image_source"] == "upload"


def test_a_camera_capture_is_recorded_as_camera(client):
    cid = gm.new_capture_id()
    gm.temp_set(cid, "vehicle_image_source", "camera")
    gm.temp_set(cid, "plate_number", "KJA456GH")
    gm.temp_set(cid, "plate_result", {"status": "CONFIRMED", "overall_confidence": 0.9})
    response = client.post("/gate/entry/confirm",
                           json={"capture_id": cid, "plate": "KJA456GH",
                                     # no face/fingerprint search in these tests
                                     "accept_unresolved": True})
    assert db.get_trip(response.get_json()["trip_id"])["vehicle_image_source"] == "camera"


def test_provenance_defaults_to_camera_when_unset(client):
    cid = gm.new_capture_id()
    gm.temp_set(cid, "plate_number", "KJA456GH")
    gm.temp_set(cid, "plate_result", {"status": "CONFIRMED", "overall_confidence": 0.9})
    response = client.post("/gate/entry/confirm",
                           json={"capture_id": cid, "plate": "KJA456GH",
                                     # no face/fingerprint search in these tests
                                     "accept_unresolved": True})
    assert db.get_trip(response.get_json()["trip_id"])["vehicle_image_source"] == "camera"


def test_the_trip_log_flags_an_uploaded_photo(client, tmp_path):
    trip_id = db.create_trip("KJA456GH", "gate_photos/vehicle_x.jpg", None, None,
                             attributes={"vehicle_image_source": "upload"})
    body = client.get("/gate/trips").get_data(as_text=True)
    assert "uploaded" in body


def test_an_upload_does_not_weaken_plate_confirmation(client):
    """A single still image still yields a low-confidence plate needing review."""
    cid = gm.new_capture_id()
    upload(client, cid, encode())
    gm.temp_set(cid, "plate_number", "KJA456GH")
    gm.temp_set(cid, "plate_result", {"status": "LOW_CONFIDENCE",
                                      "overall_confidence": 0.4})
    response = client.post("/gate/entry/confirm",
                           json={"capture_id": cid, "plate": "KJA456GH",
                                     # no face/fingerprint search in these tests
                                     "accept_unresolved": True})
    assert response.status_code == 409
    assert response.get_json()["requires_confirmation"] is True


def test_an_upload_grants_no_exit_authorization(client):
    """Uploading an image must not authorise anything."""
    trip_id = db.create_trip("KJA456GH", None, None, None)
    cid = gm.new_capture_id()
    upload(client, cid, encode())
    assert client.post("/gate/exit/confirm", json={"trip_id": trip_id}).status_code == 403
