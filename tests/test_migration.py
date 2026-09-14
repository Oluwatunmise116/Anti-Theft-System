"""
The additive, repeatable migrations (v1: identity/staff/passcodes, v2: SMS
OTP replacing Telegram), run against databases shaped like the ones earlier
versions of the app created.
"""
import os
import sqlite3
import stat

import pytest
from werkzeug.security import check_password_hash

import app as application
import database as db
from gate_helpers import rows, staff_client

# The holders/trips/fingerprints tables as the original init_db() created them.
LEGACY_SCHEMA = """
    CREATE TABLE holders (
        id INTEGER PRIMARY KEY, surname TEXT NOT NULL, first_name TEXT NOT NULL,
        date_of_birth TEXT NOT NULL, sex TEXT CHECK(sex IN ('M', 'F')), height_cm REAL,
        blood_group TEXT, street_address TEXT, state TEXT, phone_number TEXT,
        next_of_kin TEXT, next_of_kin_phone TEXT, religion TEXT, nationality TEXT,
        created_at TEXT DEFAULT (datetime('now','localtime')),
        updated_at TEXT DEFAULT (datetime('now','localtime')));
    CREATE TABLE fingerprints (
        id INTEGER PRIMARY KEY AUTOINCREMENT, holder_id INTEGER NOT NULL UNIQUE,
        template_data TEXT NOT NULL, enrolled_at TEXT DEFAULT (datetime('now','localtime')));
    CREATE TABLE trips (
        id INTEGER PRIMARY KEY AUTOINCREMENT, plate_number TEXT NOT NULL,
        vehicle_color TEXT, vehicle_color_hsv TEXT, vehicle_photo_path TEXT,
        face_photo_path TEXT, face_encoding TEXT, fingerprint_template TEXT,
        status TEXT DEFAULT 'INSIDE' CHECK(status IN ('INSIDE','EXITED','DENIED')),
        entry_time TEXT DEFAULT (datetime('now','localtime')), exit_time TEXT,
        exit_face_photo_path TEXT, exit_result TEXT, face_distance REAL,
        color_match INTEGER, notes TEXT, passcode TEXT,
        plate_source TEXT DEFAULT 'manual_entry', plate_confidence REAL);
"""

# The Telegram tables the v1 (Telegram OTP) release created.
V1_TELEGRAM_SCHEMA = """
    CREATE TABLE schema_migrations (name TEXT PRIMARY KEY, applied_at REAL NOT NULL);
    CREATE TABLE telegram_links (
        id INTEGER PRIMARY KEY AUTOINCREMENT, holder_id INTEGER NOT NULL,
        holder_uid TEXT NOT NULL, telegram_user_id INTEGER NOT NULL, chat_id INTEGER NOT NULL,
        phone_e164 TEXT NOT NULL, status TEXT NOT NULL, source TEXT NOT NULL,
        approved_by INTEGER, created_at REAL NOT NULL, revoked_at REAL,
        revoked_reason TEXT, revoked_by INTEGER);
    CREATE TABLE telegram_enrollment_requests (
        id INTEGER PRIMARY KEY AUTOINCREMENT, telegram_user_id INTEGER NOT NULL,
        chat_id INTEGER NOT NULL, phone_e164 TEXT NOT NULL,
        candidate_holder_uids TEXT NOT NULL, reason TEXT NOT NULL, status TEXT NOT NULL,
        created_at REAL NOT NULL, resolved_at REAL, resolved_by INTEGER,
        resolved_holder_uid TEXT, note TEXT);
    CREATE TABLE otp_challenges (
        id INTEGER PRIMARY KEY AUTOINCREMENT, challenge_uid TEXT NOT NULL UNIQUE,
        trip_id INTEGER NOT NULL, holder_id INTEGER NOT NULL, holder_uid TEXT NOT NULL,
        telegram_link_id INTEGER NOT NULL, gate_session_id INTEGER NOT NULL,
        code_digest TEXT NOT NULL, status TEXT NOT NULL, attempts INTEGER NOT NULL DEFAULT 0,
        max_attempts INTEGER NOT NULL, created_at REAL NOT NULL, sent_at REAL,
        expires_at REAL, verified_at REAL, retry_not_before REAL,
        telegram_message_id INTEGER, delivery_error TEXT);
    CREATE TABLE exit_authorizations (
        id INTEGER PRIMARY KEY AUTOINCREMENT, trip_id INTEGER NOT NULL,
        method TEXT NOT NULL CHECK(method IN ('TELEGRAM_OTP','PASSCODE')),
        challenge_id INTEGER, gate_session_id INTEGER NOT NULL, status TEXT NOT NULL,
        created_at REAL NOT NULL, expires_at REAL NOT NULL, consumed_at REAL);
    CREATE TABLE app_state (key TEXT PRIMARY KEY, value TEXT NOT NULL, updated_at REAL NOT NULL);
"""


def _legacy_rows(conn):
    conn.executemany("INSERT INTO holders (id, surname, first_name, date_of_birth, sex, "
                     "phone_number) VALUES (?,?,?,?,?,?)",
                     [(1, "OKAFOR", "Ada", "1990-01-01", "F", "0803 123 4567"),
                      (2, "EZE", "Obi", "1985-05-05", "M", "12345"),
                      (3, "BELLO", "Sade", "1992-02-02", "F", None)])
    conn.execute("INSERT INTO fingerprints (holder_id, template_data) VALUES (1, '[1, 2, 3]')")
    conn.executemany("INSERT INTO trips (plate_number, status, passcode, face_encoding) "
                     "VALUES (?,?,?,?)",
                     [("AAA111AA", "INSIDE", "1234", "[0.1, 0.2]"),
                      ("BBB222BB", "INSIDE", None, None),
                      ("CCC333CC", "INSIDE", "12ab", None),
                      ("DDD444DD", "EXITED", "5678", None)])


@pytest.fixture()
def legacy(tmp_path, monkeypatch):
    """A database from before any exit-verification migration."""
    path = tmp_path / "license.db"
    conn = sqlite3.connect(path)
    conn.executescript(LEGACY_SCHEMA)
    _legacy_rows(conn)
    conn.commit()
    conn.close()
    monkeypatch.setattr(db, "DB_PATH", str(path))
    return tmp_path


@pytest.fixture()
def telegram_era(tmp_path, monkeypatch):
    """A database left by the v1 Telegram OTP release: identity columns,
    Telegram tables with rows, and only the v1 migration recorded."""
    path = tmp_path / "license.db"
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.executescript(LEGACY_SCHEMA + V1_TELEGRAM_SCHEMA)
    _legacy_rows(conn)
    cursor = conn.cursor()
    db._ensure_columns(cursor, "holders", db.HOLDER_IDENTITY_COLUMNS)
    db._ensure_columns(cursor, "trips", db.TRIP_VERIFICATION_COLUMNS)
    conn.execute("UPDATE holders SET holder_uid='uid-1', phone_e164='+2348031234567' WHERE id=1")
    # An open trip for a registered driver who was to receive a Telegram code,
    # and a closed one that left by Telegram code.
    conn.execute("INSERT INTO trips (plate_number, status, holder_id, holder_uid, "
                 "identity_status, verification_mode) VALUES "
                 "('TEL111AA', 'INSIDE', 1, 'uid-1', 'REGISTERED', 'TELEGRAM_OTP')")
    conn.execute("INSERT INTO trips (plate_number, status, holder_id, holder_uid, "
                 "identity_status, verification_mode, exit_verification_method) VALUES "
                 "('TEL222BB', 'EXITED', 1, 'uid-1', 'REGISTERED', 'TELEGRAM_OTP', 'TELEGRAM_OTP')")
    conn.execute("INSERT INTO telegram_links (holder_id, holder_uid, telegram_user_id, chat_id, "
                 "phone_e164, status, source, created_at) VALUES "
                 "(1, 'uid-1', 555, 555, '+2348031234567', 'ACTIVE', 'self_contact', 0)")
    conn.execute("INSERT INTO otp_challenges (challenge_uid, trip_id, holder_id, holder_uid, "
                 "telegram_link_id, gate_session_id, code_digest, status, max_attempts, "
                 "created_at) VALUES ('c1', 5, 1, 'uid-1', 1, 1, 'digest', 'ACTIVE', 3, 0)")
    conn.execute("INSERT INTO exit_authorizations (trip_id, method, gate_session_id, status, "
                 "created_at, expires_at) VALUES (5, 'TELEGRAM_OTP', 1, 'ACTIVE', 0, 9e9)")
    conn.execute("INSERT INTO app_state (key, value, updated_at) VALUES "
                 "('telegram_update_offset', '42', 0)")
    conn.execute("INSERT INTO schema_migrations VALUES (?, 0)", (db.MIGRATION_IDENTITY_V1,))
    conn.commit()
    conn.close()
    monkeypatch.setattr(db, "DB_PATH", str(path))
    return tmp_path


def trips_by_plate():
    return {t["plate_number"]: t for t in rows("SELECT * FROM trips")}


def tables():
    return {r["name"] for r in rows("SELECT name FROM sqlite_master WHERE type='table'")}


def snapshot(extra=()):
    return {table: rows(f"SELECT * FROM {table} ORDER BY rowid")
            for table in ("holders", "trips", "fingerprints", *extra)}


# ── from the original schema ─────────────────────────────────────────────────

def test_the_database_is_backed_up_before_it_is_migrated(legacy):
    db.init_db()
    (backup,) = list((legacy / "backups").iterdir())
    assert stat.S_IMODE(os.stat(backup).st_mode) == 0o600
    copy = sqlite3.connect(backup)
    copied_tables = {r[0] for r in copy.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert copy.execute("SELECT COUNT(*) FROM trips").fetchone()[0] == 4
    assert "schema_migrations" not in copied_tables          # taken before any change
    copy.close()


def test_holders_and_biometrics_are_preserved_and_given_a_uid(legacy):
    db.init_db()
    holders = rows("SELECT id, surname, phone_number, phone_e164, holder_uid FROM holders "
                   "ORDER BY id")
    assert [h["surname"] for h in holders] == ["OKAFOR", "EZE", "BELLO"]
    assert [h["phone_e164"] for h in holders] == ["+2348031234567", None, None]
    assert holders[0]["phone_number"] == "0803 123 4567"          # original kept as typed
    uids = [h["holder_uid"] for h in holders]
    assert all(uids) and len(set(uids)) == 3
    assert rows("SELECT holder_id, template_data FROM fingerprints") \
        == [{"holder_id": 1, "template_data": "[1, 2, 3]"}]


def test_open_legacy_trips_need_reconciliation_and_are_not_assumed_guests(legacy):
    db.init_db()
    trips = trips_by_plate()
    for plate in ("AAA111AA", "BBB222BB", "CCC333CC"):
        trip = trips[plate]
        assert trip["identity_status"] == "LEGACY_UNRESOLVED"
        assert trip["verification_mode"] is None and trip["holder_id"] is None
    assert trips["DDD444DD"]["identity_status"] == "LEGACY"
    assert trips["AAA111AA"]["face_encoding"] == "[0.1, 0.2]"      # face exit still possible


def test_legacy_passcodes_are_hashed_or_discarded(legacy):
    db.init_db()
    trips = trips_by_plate()
    assert all(t["passcode"] is None for t in trips.values())      # no plaintext remains
    assert check_password_hash(trips["AAA111AA"]["passcode_hash"], "1234")
    assert trips["BBB222BB"]["passcode_hash"] is None
    assert trips["CCC333CC"]["passcode_hash"] is None
    assert "discarded" in trips["CCC333CC"]["reconciliation_note"]


def test_the_migrations_are_repeatable(legacy):
    db.init_db()
    first = snapshot()
    db.init_db()
    db.init_db()
    assert snapshot() == first
    assert len(list((legacy / "backups").iterdir())) == 1
    assert {r["name"] for r in rows("SELECT name FROM schema_migrations")} == set(db.MIGRATIONS)


def test_a_fresh_database_gets_no_backup_and_no_telegram_tables(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "new.db"))
    db.init_db()
    assert not (tmp_path / "backups").exists()
    assert not tables() & set(db.RETIRED_TELEGRAM_TABLES)
    assert {"sms_otp_challenges", "trip_exit_authorizations"} <= tables()


# ── from the Telegram (v1) release ───────────────────────────────────────────

def test_v2_backs_up_and_keeps_every_telegram_record(telegram_era):
    db.init_db()
    (backup,) = list((telegram_era / "backups").iterdir())
    assert db.MIGRATION_SMS_OTP_V2 in backup.name
    assert set(db.RETIRED_TELEGRAM_TABLES) <= tables()            # never dropped
    for table in db.RETIRED_TELEGRAM_TABLES:
        if table != "telegram_enrollment_requests":
            assert rows(f"SELECT COUNT(*) AS n FROM {table}") == [{"n": 1}], table


def test_v2_moves_open_telegram_trips_to_the_sms_fallback(telegram_era):
    db.init_db()
    trips = trips_by_plate()
    open_trip, closed_trip = trips["TEL111AA"], trips["TEL222BB"]
    assert open_trip["verification_mode"] == "SMS_OTP"
    assert (open_trip["holder_id"], open_trip["holder_uid"]) == (1, "uid-1")   # same driver
    # History is left as it happened.
    assert closed_trip["verification_mode"] == "TELEGRAM_OTP"
    assert closed_trip["exit_verification_method"] == "TELEGRAM_OTP"


def test_v2_withdraws_unused_telegram_codes_and_authorizations(telegram_era):
    db.init_db()
    assert rows("SELECT status FROM otp_challenges") == [{"status": "CANCELLED"}]
    assert rows("SELECT status FROM exit_authorizations") == [{"status": "REVOKED"}]


def test_v2_is_repeatable(telegram_era):
    db.init_db()
    first = snapshot(extra=db.RETIRED_TELEGRAM_TABLES)
    db.init_db()
    assert snapshot(extra=db.RETIRED_TELEGRAM_TABLES) == first
    assert len(list((telegram_era / "backups").iterdir())) == 1


def test_a_moved_telegram_trip_can_use_the_sms_fallback(telegram_era, sms_mock):
    db.init_db()
    application.app.config["TESTING"] = True
    operator = staff_client(application.app, "op", "operator")
    trip_id = trips_by_plate()["TEL111AA"]["id"]
    response = operator.post("/gate/exit/sms/send", json={"trip_id": trip_id})
    assert response.status_code == 200
    assert sms_mock.sent[-1]["phone_number"] == "+2348031234567"


# ── safety ───────────────────────────────────────────────────────────────────

def test_initialisation_never_deletes_verification_state(gate_db):
    with db.transaction() as conn:
        conn.execute("CREATE TABLE driver_otps (id INTEGER PRIMARY KEY, note TEXT)")
        conn.execute("INSERT INTO driver_otps (note) VALUES ('kept')")
        conn.execute("INSERT INTO sms_otp_challenges (trip_id, holder_id, holder_uid, phone_e164, "
                     "gate_session_id, idempotency_key, status, max_attempts, created_at) "
                     "VALUES (1, 1, 'u', '+2348031234567', 1, 'k1', 'ACTIVE', 3, 0)")
        conn.execute("INSERT INTO trip_exit_authorizations (trip_id, method, gate_session_id, "
                     "status, created_at, expires_at) VALUES (1, 'face', 1, 'ACTIVE', 0, 1)")
    db.init_db()
    for table in ("driver_otps", "sms_otp_challenges", "trip_exit_authorizations"):
        assert rows(f"SELECT COUNT(*) AS n FROM {table}") == [{"n": 1}], table


def test_a_legacy_trip_exits_after_an_admin_confirms_the_guest(legacy, sms_mock):
    db.init_db()
    application.app.config["TESTING"] = True
    operator = staff_client(application.app, "op", "operator")
    admin = staff_client(application.app, "boss", "admin")
    trip_id = trips_by_plate()["AAA111AA"]["id"]

    def verify():
        return operator.post("/gate/exit/passcode/verify",
                             json={"trip_id": trip_id, "passcode": "1234"})

    assert verify().get_json()["code"] == "reconciliation_required"
    admin.post(f"/admin/trips/{trip_id}/reconcile",
               data={"action": "confirm_guest", "note": "driver confirmed at the office"})
    assert verify().get_json()["valid"] is True
    assert operator.post("/gate/exit/confirm", json={"trip_id": trip_id}).status_code == 200
