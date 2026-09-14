"""
Attribute persistence, migration and provenance.

Real temporary SQLite throughout — the migration is the thing under test.
"""
import json
import sqlite3

import pytest

import database as db
from attributes.models import (AttributeStatus, AttributeValue, ScoredLabel,
                               SOURCE_AUTO, SOURCE_MANUAL_CORRECTION,
                               SOURCE_MANUAL_ENTRY, VehicleAttributeResult)


PRE_ATTRIBUTE_TRIPS_SCHEMA = """
    CREATE TABLE trips (
        id                   INTEGER PRIMARY KEY AUTOINCREMENT,
        plate_number         TEXT NOT NULL,
        vehicle_color        TEXT,
        vehicle_color_hsv    TEXT,
        vehicle_photo_path   TEXT,
        face_photo_path      TEXT,
        face_encoding        TEXT,
        fingerprint_template TEXT,
        status               TEXT DEFAULT 'INSIDE'
                                 CHECK(status IN ('INSIDE','EXITED','DENIED')),
        entry_time           TEXT DEFAULT (datetime('now','localtime')),
        exit_time            TEXT,
        exit_face_photo_path TEXT,
        exit_result          TEXT,
        face_distance        REAL,
        color_match          INTEGER,
        notes                TEXT,
        passcode             TEXT,
        plate_source         TEXT DEFAULT 'manual_entry',
        plate_confidence     REAL
    );
"""


@pytest.fixture()
def fresh_db(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "fresh.db"))
    db.init_db()
    return db


@pytest.fixture()
def legacy_db(tmp_path, monkeypatch):
    path = str(tmp_path / "legacy.db")
    conn = sqlite3.connect(path)
    conn.executescript(PRE_ATTRIBUTE_TRIPS_SCHEMA)
    conn.execute("INSERT INTO trips (plate_number, vehicle_color, face_encoding, "
                 "status, plate_source, plate_confidence) VALUES (?,?,?,?,?,?)",
                 ("KJA456GH", "red", json.dumps([0.1, 0.2]), "INSIDE", "auto", 0.83))
    conn.execute("INSERT INTO trips (plate_number, status) VALUES ('ABC123DE','EXITED')")
    conn.commit()
    conn.close()
    monkeypatch.setattr(db, "DB_PATH", path)
    return path


def make_result(colour="red", body_type="sedan", brand="toyota", votes=2):
    def value(attribute, label, confidence, source=SOURCE_AUTO):
        return AttributeValue(attribute=attribute, value=label, confidence=confidence,
                              status=AttributeStatus.CONFIRMED.value, votes=votes,
                              total_frames=3, source=source)
    result = VehicleAttributeResult(
        colour=value("colour", colour, 0.82),
        type=value("type", body_type, 0.74),
        brand=value("brand", brand, 0.68),
        vehicle_box=(10, 20, 300, 240), coco_class="car",
        frames_processed=3, frames_with_vehicle=3, frames_with_logo=2,
        processing_time_ms=91.4,
        model_versions={"attributes": "mnv3@arch1", "logo": "logo.pt"})
    result.brand.top_k = [ScoredLabel(brand, 0.68), ScoredLabel("lexus", 0.11)]
    result.status = result.roll_up_status()
    return result


# ── migration ─────────────────────────────────────────────────────────────

def test_new_database_has_every_attribute_column(fresh_db):
    conn = db.get_connection()
    columns = {row[1] for row in conn.execute("PRAGMA table_info(trips)")}
    conn.close()
    assert set(db.TRIP_ATTRIBUTE_COLUMNS_V2) - columns == set()


def test_migration_adds_columns_without_losing_trips(legacy_db):
    before = sqlite3.connect(legacy_db).execute("SELECT COUNT(*) FROM trips").fetchone()[0]
    db.init_db()
    after = sqlite3.connect(legacy_db).execute("SELECT COUNT(*) FROM trips").fetchone()[0]
    assert before == after == 2
    conn = db.get_connection()
    columns = {row[1] for row in conn.execute("PRAGMA table_info(trips)")}
    conn.close()
    assert set(db.TRIP_ATTRIBUTE_COLUMNS_V2) <= columns


def test_migration_is_idempotent(legacy_db):
    db.init_db()
    db.init_db()
    conn = db.get_connection()
    columns = [row[1] for row in conn.execute("PRAGMA table_info(trips)")]
    conn.close()
    assert len(columns) == len(set(columns))


def test_pre_attribute_trips_still_load(legacy_db):
    db.init_db()
    trip = db.get_open_trip("KJA456GH")
    assert trip["plate_number"] == "KJA456GH"
    assert trip["plate_source"] == "auto"          # ANPR provenance preserved
    assert trip["plate_confidence"] == 0.83
    assert trip["attr_status"] is None
    assert trip["vehicle_color_source"] is None
    assert trip["attr_vehicle_box_json"] is None


# ── writing ───────────────────────────────────────────────────────────────

def test_attributes_are_stored_with_confidence_votes_and_source(fresh_db):
    columns = db.build_attribute_columns(make_result())
    trip_id = db.create_trip("KJA456GH", None, None, None, attributes=columns)
    trip = db.get_trip(trip_id)

    assert trip["vehicle_color"] == "red"
    assert trip["vehicle_color_confidence"] == 0.82
    assert trip["vehicle_color_source"] == SOURCE_AUTO
    assert trip["vehicle_color_votes"] == 2
    assert trip["vehicle_type"] == "sedan"
    assert trip["vehicle_brand"] == "toyota"
    assert trip["attr_status"] == "CONFIRMED"
    assert trip["attr_coco_class"] == "car"
    assert trip["attr_vehicle_box_json"] == [10, 20, 300, 240]
    assert trip["attr_model_version"] == "mnv3@arch1"
    assert trip["vehicle_brand_top3_json"][0]["label"] == "toyota"


def test_unusable_values_are_stored_as_null(fresh_db):
    result = make_result()
    result.brand = AttributeValue.unknown("brand", "no logo detected")
    columns = db.build_attribute_columns(result)
    trip = db.get_trip(db.create_trip("X", None, None, None, attributes=columns))
    assert trip["vehicle_brand"] is None
    assert trip["vehicle_brand_confidence"] is None
    assert trip["vehicle_brand_source"] is None
    assert trip["vehicle_color"] == "red"          # others unaffected


def test_a_trip_with_no_attributes_still_saves(fresh_db):
    trip = db.get_trip(db.create_trip("ABC123DE", None, None, None))
    assert trip["plate_number"] == "ABC123DE"
    assert trip["vehicle_color"] is None
    assert trip["attr_status"] is None


# ── provenance ────────────────────────────────────────────────────────────

def test_a_correction_over_an_automatic_reading_is_manual_correction(fresh_db):
    columns = db.build_attribute_columns(make_result(), {"colour": "white"})
    trip = db.get_trip(db.create_trip("K", None, None, None, attributes=columns))
    assert trip["vehicle_color"] == "white"
    assert trip["vehicle_color_source"] == SOURCE_MANUAL_CORRECTION
    # A human assertion carries no model probability and no vote count.
    assert trip["vehicle_color_confidence"] is None
    assert trip["vehicle_color_votes"] is None
    # Untouched attributes keep their automatic provenance.
    assert trip["vehicle_type_source"] == SOURCE_AUTO


def test_a_correction_with_no_prior_reading_is_manual_entry(fresh_db):
    result = make_result()
    result.brand = AttributeValue.unknown("brand", "no logo detected")
    columns = db.build_attribute_columns(result, {"brand": "innoson"})
    trip = db.get_trip(db.create_trip("K", None, None, None, attributes=columns))
    assert trip["vehicle_brand"] == "innoson"
    assert trip["vehicle_brand_source"] == SOURCE_MANUAL_ENTRY


def test_mismatch_flags_are_recorded_for_operator_review(fresh_db):
    trip_id = db.create_trip("K", None, None, None,
                             attributes=db.build_attribute_columns(make_result()))
    db.record_attribute_mismatch(trip_id, ["colour", "brand"],
                                 "colour: red at entry vs blue at exit")
    trip = db.get_trip(trip_id)
    assert trip["attr_mismatch_flags_json"] == ["colour", "brand"]
    assert "red at entry" in trip["attribute_mismatch_reason"]
    # Recording a mismatch changes no status and closes nothing.
    assert trip["status"] == "INSIDE"


def test_closing_a_trip_keeps_the_attribute_evidence(fresh_db):
    trip_id = db.create_trip("K", None, None, None,
                             attributes=db.build_attribute_columns(make_result()))
    db.close_trip(trip_id, "GRANTED", None, 0.2)
    trip = db.get_trip(trip_id)
    assert trip["status"] == "EXITED"
    assert trip["face_encoding"] is None           # biometrics still cleared
    assert trip["vehicle_color"] == "red"          # non-biometric evidence kept
    assert trip["attr_status"] == "CONFIRMED"


def test_corrupt_attribute_json_does_not_break_a_trip(fresh_db):
    trip_id = db.create_trip("K", None, None, None)
    conn = db.get_connection()
    conn.execute("UPDATE trips SET attr_vehicle_box_json='{{{ broken' WHERE id=?",
                 (trip_id,))
    conn.commit()
    conn.close()
    assert db.get_trip(trip_id)["attr_vehicle_box_json"] is None


def test_update_ignores_columns_outside_the_attribute_set(fresh_db):
    trip_id = db.create_trip("KJA456GH", None, None, None)
    db.update_trip_attributes(trip_id, {"vehicle_color": "blue",
                                        "plate_number": "HACKED",
                                        "status": "EXITED"})
    trip = db.get_trip(trip_id)
    assert trip["vehicle_color"] == "blue"
    assert trip["plate_number"] == "KJA456GH"
    assert trip["status"] == "INSIDE"
