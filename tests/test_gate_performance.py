"""
The Pi performance settings: smaller vehicle-detector input, shrunk uploads
and a lighter live preview. No camera or model is used.
"""
import io
import os

import cv2
import numpy as np
import pytest

import app as application
import config as cfg
import database as db
import face_manager as fm
import gate_manager as gm
from gate_helpers import staff_client


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "perf.db"))
    db.init_db()
    application.app.config["TESTING"] = True
    with staff_client(application.app) as c:
        yield c


def with_config(monkeypatch, **overrides):
    real_load = cfg.load
    monkeypatch.setattr(cfg, "load", lambda: {**real_load(), **overrides})


def photo(width, height):
    """A smooth test photo (compresses small, unlike random noise)."""
    image = np.full((height, width, 3), 120, dtype=np.uint8)
    cv2.rectangle(image, (width // 4, height // 4), (width * 3 // 4, height * 3 // 4),
                  (30, 60, 200), -1)
    ok, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
    assert ok
    return buffer.tobytes()


def upload(client, cid, data):
    return client.post("/gate/entry/vehicle/upload",
                       data={"capture_id": cid, "image": (io.BytesIO(data), "car.jpg")},
                       content_type="multipart/form-data")


# ── vehicle detector input size ──────────────────────────────────────────────

@pytest.mark.parametrize("configured,expected", [(320, 320), (640, 640), (330, 320),
                                                 (64, 192), (5000, 1280), ("abc", 320)])
def test_the_vehicle_detector_size_is_clamped_to_a_multiple_of_32(monkeypatch,
                                                                  configured, expected):
    monkeypatch.setattr(cfg, "get", lambda key, default=None:
                        configured if key == "vehicle_detection_imgsz" else default)
    assert gm.vehicle_detection_imgsz() == expected


def test_the_vehicle_detector_runs_at_the_configured_size(monkeypatch):
    calls = []

    class FakeResult:
        boxes = None

    def fake_model(frame, **kwargs):
        calls.append(kwargs)
        return [FakeResult()]

    monkeypatch.setattr(gm, "_get_vehicle_yolo", lambda: fake_model)
    monkeypatch.setattr(gm, "vehicle_detection_imgsz", lambda: 320)
    gm.detect_vehicle_crop_detailed(np.zeros((720, 1280, 3), dtype=np.uint8))
    assert calls and calls[0]["imgsz"] == 320


def test_the_shipped_config_uses_the_faster_settings():
    shipped = cfg.load()
    assert shipped["vehicle_detection_imgsz"] == 320
    assert shipped["plate_capture_frame_count"] == 6
    assert shipped["gate_upload_max_side"] == 1920
    assert (shipped["preview_max_width"], shipped["preview_max_fps"]) == (640, 15)


# ── uploads ──────────────────────────────────────────────────────────────────

def test_a_large_upload_is_shrunk_before_it_is_saved(client):
    cid = gm.new_capture_id()
    response = upload(client, cid, photo(4000, 3000))
    body = response.get_json()
    assert response.status_code == 200, body
    assert (body["width"], body["height"]) == (1920, 1440)
    stored = cv2.imread(gm.temp_get(cid)["vehicle_photo"])
    assert stored.shape[:2] == (1440, 1920)
    os.remove(gm.temp_get(cid)["vehicle_photo"])


def test_a_small_upload_keeps_its_size(client):
    cid = gm.new_capture_id()
    body = upload(client, cid, photo(1280, 720)).get_json()
    assert (body["width"], body["height"]) == (1280, 720)
    os.remove(gm.temp_get(cid)["vehicle_photo"])


def test_shrinking_never_goes_below_the_minimum_size(client, monkeypatch):
    with_config(monkeypatch, gate_upload_max_side=640)
    cid = gm.new_capture_id()
    body = upload(client, cid, photo(3000, 70)).get_json()
    assert body["height"] >= gm.UPLOAD_MIN_DIMENSION and body["width"] <= 3000
    os.remove(gm.temp_get(cid)["vehicle_photo"])


def test_the_pages_shrink_uploads_in_the_browser_first(client):
    for path in ("/gate/entry", "/gate/exit"):
        page = client.get(path).get_data(as_text=True)
        assert "const UPLOAD_MAX_SIDE = 1920;" in page
        assert "shrinkImage(file)" in page and "function shrinkImage(" in page


# ── live preview ─────────────────────────────────────────────────────────────

def test_the_preview_is_downscaled_for_display(monkeypatch):
    with_config(monkeypatch, preview_max_width=640, preview_max_fps=30)
    was_running = fm._cam_running.is_set()
    monkeypatch.setitem(fm._cam_state, "frame", np.full((720, 1280, 3), 90, dtype=np.uint8))
    fm._cam_running.set()
    try:
        chunk = next(fm.generate_video_frames())
    finally:
        if not was_running:
            fm._cam_running.clear()
    jpeg = chunk.split(b"\r\n\r\n", 1)[1].rsplit(b"\r\n", 1)[0]
    frame = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert frame.shape[:2] == (360, 640)


@pytest.mark.parametrize("width,fps,expected", [(640, 15, (640, 1 / 15)),
                                                (10, 999, (160, 1 / 30)),
                                                ("x", 15, (640, 1 / 15))])
def test_preview_settings_are_clamped(monkeypatch, width, fps, expected):
    with_config(monkeypatch, preview_max_width=width, preview_max_fps=fps)
    assert fm._preview_settings() == pytest.approx(expected)
