import sqlite3
import os
import json
import time
import uuid
from contextlib import contextmanager
from datetime import datetime

from werkzeug.security import generate_password_hash

import phone_utils
import verification_settings as vs

# LICENSE_DB_PATH lets tests (and alternative deployments) point at another
# file. Tests MUST set it before importing this module, because init_db()
# runs at import time — see tests/conftest.py.
DB_PATH = os.environ.get("LICENSE_DB_PATH", "license.db")
PHOTOS_DIR = "photos"
SIGNATURES_DIR = "signatures"
GATE_PHOTOS_DIR = "gate_photos"


def get_connection():
    conn = sqlite3.connect(DB_PATH, timeout=15)
    conn.row_factory = sqlite3.Row
    return conn


@contextmanager
def transaction():
    """
    BEGIN IMMEDIATE ... COMMIT on a dedicated connection.

    IMMEDIATE takes SQLite's write lock up front, so every check-then-write
    inside the block is atomic across threads, gunicorn workers and the
    command-line scripts. Any exception rolls the whole block back.
    """
    conn = sqlite3.connect(DB_PATH, timeout=15, isolation_level=None)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("BEGIN IMMEDIATE")
        yield conn
        conn.execute("COMMIT")
    except BaseException:
        try:
            conn.execute("ROLLBACK")
        except sqlite3.Error:
            pass
        raise
    finally:
        conn.close()


def log_security_event(conn, event, trip_id=None, holder_id=None,
                       staff_user_id=None, session_id=None, detail=None):
    """
    Append to the security audit log inside the caller's transaction.
    `detail` must never contain a code, passcode, token or full phone number.
    """
    conn.execute(
        "INSERT INTO security_events (created_at, event, trip_id, holder_id, "
        "staff_user_id, session_id, detail) VALUES (?,?,?,?,?,?,?)",
        (time.time(), event, trip_id, holder_id, staff_user_id, session_id,
         json.dumps(detail) if detail is not None else None))


def _phone_region():
    try:
        return vs.load().phone_default_region
    except vs.ConfigurationError:
        return "NG"


def normalize_holder_phone(raw):
    return phone_utils.normalize_phone(raw, _phone_region())


# ── VEHICLE ATTRIBUTE SCHEMA ──────────────────────────────────────────────────
# Every column added by the vehicle-attribute feature, with its declared
# type. Applied by _ensure_columns() on every start-up: adding a column that
# already exists is skipped, so this is safe to run against a database from
# any earlier version of the app and never touches existing rows.
TRIP_ATTRIBUTE_COLUMNS = {
    # entry-time recognition
    "vehicle_color":                          "TEXT",
    "vehicle_color_confidence":               "REAL",
    "vehicle_color_method":                   "TEXT",
    "vehicle_color_hsv":                      "TEXT",     # JSON [h, s, v]
    "vehicle_type":                           "TEXT",
    "vehicle_type_confidence":                "REAL",
    "vehicle_brand":                          "TEXT",
    "vehicle_brand_confidence":               "REAL",
    "vehicle_brand_top3_json":                "TEXT",     # JSON list
    "vehicle_yolo_class":                     "TEXT",
    "vehicle_attributes_status":              "TEXT",
    "vehicle_attributes_model_versions_json": "TEXT",     # JSON dict
    "vehicle_attributes_processing_ms":       "REAL",
    # exit-time recognition and comparison
    "exit_vehicle_color":                     "TEXT",
    "exit_vehicle_type":                      "TEXT",
    "exit_vehicle_brand":                     "TEXT",
    "color_match":                            "INTEGER",  # 1 / 0 / NULL
    "type_match":                             "INTEGER",
    "brand_match":                            "INTEGER",
    "attribute_mismatch_reason":              "TEXT",
}

# Columns added for the attributes/ package. The colour/type/brand VALUE
# and CONFIDENCE columns above are reused as-is; these add the provenance
# and vote count the new contract requires, so a trip written before this
# change still reads correctly and no column is duplicated.
#
# `*_source` carries the same vocabulary as plate_source
# ("auto" / "manual_correction" / "manual_entry"), which is what makes an
# operator edit distinguishable from an automatic reading.
TRIP_ATTRIBUTE_COLUMNS_V2 = {
    "vehicle_color_source":   "TEXT",
    "vehicle_color_votes":    "INTEGER",
    "vehicle_type_source":    "TEXT",
    "vehicle_type_votes":     "INTEGER",
    "vehicle_brand_source":   "TEXT",
    "vehicle_brand_votes":    "INTEGER",
    "attr_status":            "TEXT",
    "attr_coco_class":        "TEXT",
    "attr_vehicle_box_json":  "TEXT",    # JSON [x1, y1, x2, y2]
    "attr_model_version":     "TEXT",
    "attr_processing_ms":     "REAL",
    # Which attributes disagreed with the registered vehicle, for the
    # operator-review dialog. Advisory: never an input to allow/deny.
    "attr_mismatch_flags_json": "TEXT",
    # "camera" (taken at the gate) or "upload" (an operator supplied a file).
    # An uploaded image is not evidence the vehicle was present.
    "vehicle_image_source": "TEXT",
}

# Trip columns holding JSON, decoded on read by _decode_trip().
_TRIP_JSON_COLUMNS = ("vehicle_color_hsv", "vehicle_brand_top3_json",
                      "vehicle_attributes_model_versions_json",
                      "attr_vehicle_box_json", "attr_mismatch_flags_json")


# ── EXIT VERIFICATION SCHEMA (staff auth, SMS OTP fallback, guest passcode) ──
# Additive and repeatable: every statement is CREATE ... IF NOT EXISTS, an
# ADD COLUMN that skips existing columns, or an UPDATE that only touches
# rows still needing it. The first run against an existing database that
# is missing any of these migrations backs the file up first (see init_db).
#
# v1 added staff accounts, trip identity and hashed passcodes (its name is
# kept because existing databases already record it). v2 replaced the
# Telegram OTP fallback with Robase SMS OTP.
MIGRATION_IDENTITY_V1 = "2026_09_telegram_otp_v1"
MIGRATION_SMS_OTP_V2 = "2026_09_sms_otp_v2"
MIGRATIONS = (MIGRATION_IDENTITY_V1, MIGRATION_SMS_OTP_V2)

#: Tables used only by the retired Telegram OTP feature. v2 never drops
#: them: existing rows stay for the audit trail; nothing reads or writes them.
RETIRED_TELEGRAM_TABLES = ("telegram_links", "telegram_enrollment_requests",
                           "otp_challenges", "exit_authorizations", "app_state")

# Identity recorded at entry. A NULL holder_id alone never means "guest".
IDENTITY_REGISTERED = "REGISTERED"
IDENTITY_GUEST = "GUEST"
IDENTITY_UNRESOLVED = "UNRESOLVED"
IDENTITY_LEGACY_UNRESOLVED = "LEGACY_UNRESOLVED"   # open trip from before this feature
IDENTITY_LEGACY = "LEGACY"                         # closed trip from before this feature

# The trip's FALLBACK when face and fingerprint fail. Biometrics come first
# for every trip; this only decides what happens when they cannot be used.
MODE_SMS_OTP = "SMS_OTP"
MODE_PASSCODE = "PASSCODE"
LEGACY_MODE_TELEGRAM_OTP = "TELEGRAM_OTP"          # v1 value, migrated by v2

#: The previous backend accepted 4-8 digit passcodes; legacy codes in that
#: range are preserved (hashed), anything else is discarded.
LEGACY_PASSCODE_RANGE = (4, 8)

HOLDER_IDENTITY_COLUMNS = {
    # Random and never reused. Holder ids ARE reused (get_next_holder_id
    # fills gaps), so trip identities bind to this value: a recreated
    # holder #3 can never receive codes for a deleted holder #3's trip.
    "holder_uid": "TEXT",
    # The saved phone number in E.164, kept in step with phone_number.
    "phone_e164": "TEXT",
}

TRIP_VERIFICATION_COLUMNS = {
    "holder_id":                "INTEGER",
    "holder_uid":               "TEXT",
    "identity_status":          "TEXT",
    "identity_method":          "TEXT",
    "verification_mode":        "TEXT",
    "passcode_hash":            "TEXT",
    "passcode_failed_attempts": "INTEGER NOT NULL DEFAULT 0",
    "otp_failed_attempts":      "INTEGER NOT NULL DEFAULT 0",
    "otp_send_count":           "INTEGER NOT NULL DEFAULT 0",
    "verification_locked":      "INTEGER NOT NULL DEFAULT 0",
    "entry_session_id":         "INTEGER",
    "entry_staff_user_id":      "INTEGER",
    "exit_staff_user_id":       "INTEGER",
    "exit_verification_method": "TEXT",
    "exit_authorization_id":    "INTEGER",
    # The latest face / fingerprint comparison at exit, for the audit record.
    "exit_biometric_method":    "TEXT",
    "exit_biometric_match":     "INTEGER",
    "exit_biometric_score":     "REAL",
    "reconciled_by":            "INTEGER",
    "reconciled_at":            "REAL",
    "reconciliation_note":      "TEXT",
}

_SECURITY_TABLES_SQL = """
    CREATE TABLE IF NOT EXISTS schema_migrations (
        name        TEXT PRIMARY KEY,
        applied_at  REAL NOT NULL
    );

    CREATE TABLE IF NOT EXISTS staff_users (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        username        TEXT NOT NULL UNIQUE COLLATE NOCASE,
        password_hash   TEXT NOT NULL,
        role            TEXT NOT NULL CHECK(role IN ('admin','operator')),
        active          INTEGER NOT NULL DEFAULT 1,
        failed_logins   INTEGER NOT NULL DEFAULT 0,
        locked_until    REAL,
        created_at      REAL NOT NULL,
        last_login_at   REAL
    );

    -- Server-side gate sessions. The browser holds only a random token;
    -- the database holds its SHA-256, so OTP challenges and exit
    -- authorizations can be bound to a session that survives restarts and
    -- is shared by every worker.
    CREATE TABLE IF NOT EXISTS staff_sessions (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        token_hash      TEXT NOT NULL UNIQUE,
        user_id         INTEGER NOT NULL,
        created_at      REAL NOT NULL,
        expires_at      REAL NOT NULL,
        last_seen_at    REAL,
        revoked_at      REAL,
        user_agent      TEXT
    );

    -- One row per SMS OTP send, for the trip's registered driver. Robase
    -- generates, texts and checks the code; only its OTP id is stored here.
    -- Every row is bound to the trip, the holder, the destination number
    -- and the gate session that asked for it.
    CREATE TABLE IF NOT EXISTS sms_otp_challenges (
        id                  INTEGER PRIMARY KEY AUTOINCREMENT,
        trip_id             INTEGER NOT NULL,
        holder_id           INTEGER NOT NULL,
        holder_uid          TEXT NOT NULL,
        phone_e164          TEXT NOT NULL,
        gate_session_id     INTEGER NOT NULL,
        idempotency_key     TEXT NOT NULL UNIQUE,
        provider_otp_id     TEXT,
        status              TEXT NOT NULL CHECK(status IN
                                ('SENDING','ACTIVE','FAILED','UNCERTAIN','VERIFIED',
                                 'EXPIRED','EXHAUSTED','SUPERSEDED','CANCELLED')),
        attempts            INTEGER NOT NULL DEFAULT 0,
        failed_attempts     INTEGER NOT NULL DEFAULT 0,
        max_attempts        INTEGER NOT NULL,
        created_at          REAL NOT NULL,
        sent_at             REAL,
        expires_at          REAL,
        verified_at         REAL,
        retry_not_before    REAL,
        delivery_status     TEXT,
        provider_error      TEXT
    );
    CREATE INDEX IF NOT EXISTS ix_sms_otp_challenges_trip
        ON sms_otp_challenges(trip_id, status);
    CREATE INDEX IF NOT EXISTS ix_sms_otp_challenges_holder
        ON sms_otp_challenges(holder_uid, created_at);

    -- Proof that one verification method succeeded for one trip, from one
    -- gate session: face or fingerprint (primary), sms_otp or passcode
    -- (fallback). Consumed atomically when the trip is closed.
    CREATE TABLE IF NOT EXISTS trip_exit_authorizations (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        trip_id         INTEGER NOT NULL,
        method          TEXT NOT NULL CHECK(method IN
                            ('face','fingerprint','sms_otp','passcode')),
        challenge_id    INTEGER,
        gate_session_id INTEGER NOT NULL,
        status          TEXT NOT NULL CHECK(status IN ('ACTIVE','CONSUMED','REVOKED')),
        created_at      REAL NOT NULL,
        expires_at      REAL NOT NULL,
        consumed_at     REAL
    );
    CREATE UNIQUE INDEX IF NOT EXISTS ux_trip_exit_authorizations_active
        ON trip_exit_authorizations(trip_id) WHERE status = 'ACTIVE';

    CREATE TABLE IF NOT EXISTS security_events (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at      REAL NOT NULL,
        event           TEXT NOT NULL,
        trip_id         INTEGER,
        holder_id       INTEGER,
        staff_user_id   INTEGER,
        session_id      INTEGER,
        detail          TEXT
    );
    CREATE INDEX IF NOT EXISTS ix_security_events_time ON security_events(created_at);
"""

_SECURITY_INDEXES_SQL = """
    CREATE UNIQUE INDEX IF NOT EXISTS ux_holders_uid ON holders(holder_uid);
    CREATE INDEX IF NOT EXISTS ix_holders_phone_e164 ON holders(phone_e164);
    CREATE INDEX IF NOT EXISTS ix_trips_holder_uid ON trips(holder_uid);
"""


def _pending_migrations(path) -> list:
    """Migrations not yet recorded in an existing database at `path`
    (empty for a new or empty file, which has nothing to back up)."""
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return []
    conn = sqlite3.connect(path, timeout=15)
    try:
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        if "holders" not in tables and "trips" not in tables:
            return []
        recorded = set()
        if "schema_migrations" in tables:
            recorded = {r[0] for r in conn.execute("SELECT name FROM schema_migrations")}
        return [name for name in MIGRATIONS if name not in recorded]
    finally:
        conn.close()


def backup_database(label="manual"):
    """
    Consistent copy of the database (SQLite online-backup API) into a
    git-ignored backups/ directory next to it, readable only by its owner.
    Raises on failure — callers must not migrate without a backup.
    """
    src_path = os.path.abspath(DB_PATH)
    backup_dir = os.path.join(os.path.dirname(src_path), "backups")
    os.makedirs(backup_dir, mode=0o700, exist_ok=True)
    stem = os.path.splitext(os.path.basename(src_path))[0]
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    dest = os.path.join(backup_dir, f"{stem}-{label}-{stamp}-{uuid.uuid4().hex[:6]}.db")
    src = sqlite3.connect(src_path, timeout=15)
    out = sqlite3.connect(dest)
    try:
        src.backup(out)
    finally:
        out.close()
        src.close()
    os.chmod(dest, 0o600)
    return dest


def _backfill_holder_identity(c):
    rows = c.execute("SELECT id, holder_uid, phone_number, phone_e164 FROM holders").fetchall()
    for row in rows:
        updates = {}
        if not row["holder_uid"]:
            updates["holder_uid"] = uuid.uuid4().hex
        if row["phone_e164"] is None and row["phone_number"]:
            e164 = normalize_holder_phone(row["phone_number"])
            if e164:
                updates["phone_e164"] = e164
        if updates:
            c.execute(f"UPDATE holders SET {', '.join(f'{k}=?' for k in updates)} WHERE id=?",
                      list(updates.values()) + [row["id"]])


def _mark_legacy_trips(c):
    # Trips created before this feature have no server-confirmed identity.
    # Open ones need staff reconciliation before they can exit; a NULL
    # holder_id is NOT taken to mean "unregistered".
    c.execute("UPDATE trips SET identity_status=? WHERE identity_status IS NULL "
              "AND status='INSIDE'", (IDENTITY_LEGACY_UNRESOLVED,))
    c.execute("UPDATE trips SET identity_status=? WHERE identity_status IS NULL",
              (IDENTITY_LEGACY,))


def _secure_legacy_passcodes(c):
    """Replace plaintext trip passcodes with a password hash, or discard them."""
    low, high = LEGACY_PASSCODE_RANGE
    rows = c.execute("SELECT id, passcode, passcode_hash, reconciliation_note FROM trips "
                     "WHERE passcode IS NOT NULL").fetchall()
    for row in rows:
        code = str(row["passcode"]).strip()
        note = row["reconciliation_note"]
        if row["passcode_hash"]:
            c.execute("UPDATE trips SET passcode=NULL WHERE id=?", (row["id"],))
        elif code.isdigit() and low <= len(code) <= high:
            c.execute("UPDATE trips SET passcode_hash=?, passcode=NULL, reconciliation_note=? "
                      "WHERE id=?",
                      (generate_password_hash(code),
                       _append_note(note, "legacy passcode preserved as a hash"), row["id"]))
        else:
            c.execute("UPDATE trips SET passcode=NULL, reconciliation_note=? WHERE id=?",
                      (_append_note(note, "legacy passcode discarded: not a 4-8 digit code"),
                       row["id"]))


def _append_note(existing, addition):
    return f"{existing}; {addition}" if existing else addition


def _retire_telegram_state(c):
    """
    v2: Telegram OTP is gone. An open trip whose registered driver was to get
    a Telegram code keeps the same driver link and falls back to an SMS code
    instead. Unused Telegram codes and exit authorizations are withdrawn;
    the retired tables and their rows are otherwise left untouched.
    """
    tables = {r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    c.execute("UPDATE trips SET verification_mode=? WHERE verification_mode=? "
              "AND status='INSIDE'", (MODE_SMS_OTP, LEGACY_MODE_TELEGRAM_OTP))
    if "exit_authorizations" in tables:
        c.execute("UPDATE exit_authorizations SET status='REVOKED' WHERE status='ACTIVE'")
    if "otp_challenges" in tables:
        c.execute("UPDATE otp_challenges SET status='CANCELLED' "
                  "WHERE status IN ('SENDING','ACTIVE')")


def _apply_security_migration(conn):
    c = conn.cursor()
    c.executescript(_SECURITY_TABLES_SQL)
    _ensure_columns(c, "holders", HOLDER_IDENTITY_COLUMNS)
    _ensure_columns(c, "trips", TRIP_VERIFICATION_COLUMNS)
    _backfill_holder_identity(c)
    _mark_legacy_trips(c)
    _secure_legacy_passcodes(c)
    _retire_telegram_state(c)
    c.executescript(_SECURITY_INDEXES_SQL)
    now = time.time()
    for name in MIGRATIONS:
        c.execute("INSERT OR IGNORE INTO schema_migrations (name, applied_at) VALUES (?, ?)",
                  (name, now))
    conn.commit()


def _table_columns(c, table):
    """Existing column names for `table` (empty when the table is absent)."""
    try:
        return {row[1] for row in c.execute(f"PRAGMA table_info({table})").fetchall()}
    except Exception:
        return set()


def _ensure_columns(c, table, columns):
    """
    Idempotent additive migration. Adds only missing columns, one statement
    at a time, so a single failure cannot abort the rest of the migration.
    Returns the list of columns actually added.
    """
    existing = _table_columns(c, table)
    added = []
    for name, decl in columns.items():
        if name in existing:
            continue
        try:
            c.execute(f"ALTER TABLE {table} ADD COLUMN {name} {decl}")
            added.append(name)
        except Exception:
            pass
    return added


def safe_json_loads(value, default=None):
    """
    Decode a JSON column without ever raising. A trip written before this
    feature existed, or one whose JSON was truncated, must still load.
    """
    if value is None or value == "":
        return default
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return default


def safe_json_dumps(value):
    """Serialize for storage, returning NULL rather than raising on bad input."""
    if value is None:
        return None
    try:
        return json.dumps(value)
    except (TypeError, ValueError):
        return None


def init_db():
    os.makedirs(PHOTOS_DIR, exist_ok=True)
    os.makedirs(SIGNATURES_DIR, exist_ok=True)
    # Back up an existing database before any pending migration runs. If the
    # backup fails, the exception stops start-up: the database is never
    # migrated without a copy.
    pending = _pending_migrations(DB_PATH)
    if pending:
        backup_path = backup_database(f"pre-{pending[-1]}")
        print(f"[database] Backed up {DB_PATH} to {backup_path} before migration "
              f"{', '.join(pending)}")
    conn = get_connection()
    c = conn.cursor()
    c.executescript("""
        CREATE TABLE IF NOT EXISTS holders (
            id                  INTEGER PRIMARY KEY,
            surname             TEXT NOT NULL,
            first_name          TEXT NOT NULL,
            date_of_birth       TEXT NOT NULL,
            sex                 TEXT CHECK(sex IN ('M', 'F')),
            height_cm           REAL,
            blood_group         TEXT,
            street_address      TEXT,
            state               TEXT,
            phone_number        TEXT,
            next_of_kin         TEXT,
            next_of_kin_phone   TEXT,
            religion            TEXT,
            nationality         TEXT,
            created_at          TEXT DEFAULT (datetime('now','localtime')),
            updated_at          TEXT DEFAULT (datetime('now','localtime'))
        );

        CREATE TABLE IF NOT EXISTS licenses (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            holder_id           INTEGER NOT NULL,
            license_number      TEXT UNIQUE NOT NULL,
            date_of_issue       TEXT NOT NULL,
            expiry_date         TEXT NOT NULL,
            license_class       TEXT NOT NULL,
            state_of_issue      TEXT NOT NULL,
            endorsements        TEXT,
            authorized_by       TEXT,
            created_at          TEXT DEFAULT (datetime('now','localtime')),
            FOREIGN KEY (holder_id) REFERENCES holders(id)
        );

        CREATE TABLE IF NOT EXISTS fingerprints (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            holder_id           INTEGER NOT NULL UNIQUE,
            template_data       TEXT NOT NULL,
            enrolled_at         TEXT DEFAULT (datetime('now','localtime')),
            FOREIGN KEY (holder_id) REFERENCES holders(id)
        );

        CREATE TABLE IF NOT EXISTS photos (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            holder_id           INTEGER NOT NULL,
            photo_type          TEXT CHECK(photo_type IN ('passport', 'holder_signature', 'authorized_signature')),
            file_path           TEXT NOT NULL,
            uploaded_at         TEXT DEFAULT (datetime('now','localtime')),
            FOREIGN KEY (holder_id) REFERENCES holders(id)
        );

        CREATE TABLE IF NOT EXISTS gate_sessions (
            id                      INTEGER PRIMARY KEY AUTOINCREMENT,
            plate_number            TEXT NOT NULL,
            vehicle_photo_path      TEXT,
            face_photo_path         TEXT,
            face_encoding           TEXT,
            fingerprint_template    TEXT,
            entry_time              TEXT DEFAULT (datetime('now','localtime')),
            entry_gate              TEXT DEFAULT 'ENTRY',
            exit_time               TEXT,
            exit_gate               TEXT DEFAULT 'EXIT',
            exit_face_photo_path    TEXT,
            status                  TEXT DEFAULT 'inside'
                                        CHECK(status IN ('inside','exited','denied','released')),
            exit_decision           TEXT,
            notes                   TEXT
        );

        CREATE TABLE IF NOT EXISTS trips (
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
            notes                TEXT
        );
    """)
    # Initialisation never drops tables: an earlier version dropped
    # driver_otps/vehicle_driver_bindings/vehicles on every start, which
    # would silently destroy any state kept under those names.
    conn.commit()
    # Add passcode column to trips (idempotent migration)
    try:
        c.execute("ALTER TABLE trips ADD COLUMN passcode TEXT")
        conn.commit()
    except Exception:
        pass
    # ANPR provenance — required so a plate reading's trust level survives
    # into the audit trail (auto-recognised vs. operator-typed/corrected).
    try:
        c.execute("ALTER TABLE trips ADD COLUMN plate_source TEXT DEFAULT 'manual_entry'")
        conn.commit()
    except Exception:
        pass
    try:
        c.execute("ALTER TABLE trips ADD COLUMN plate_confidence REAL")
        conn.commit()
    except Exception:
        pass

    # Vehicle attribute columns (colour / type / brand) — see
    # TRIP_ATTRIBUTE_COLUMNS. Added idempotently so an existing license.db
    # migrates in place: no trip record is ever dropped or rewritten.
    _ensure_columns(c, "trips", TRIP_ATTRIBUTE_COLUMNS)
    _ensure_columns(c, "trips", TRIP_ATTRIBUTE_COLUMNS_V2)
    conn.commit()

    # Operator corrections live in their own table, separate from the trip
    # record: they are future training data, never part of an authorization
    # decision, and pruning them can never disturb the audit trail.
    c.execute("""
        CREATE TABLE IF NOT EXISTS attribute_corrections (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            trip_id             INTEGER,
            attribute           TEXT NOT NULL,
            original_label      TEXT,
            original_confidence REAL,
            original_status     TEXT,
            corrected_label     TEXT NOT NULL,
            image_reference     TEXT,
            model_version       TEXT,
            created_at          TEXT DEFAULT (datetime('now','localtime'))
        )
    """)
    conn.commit()

    # Staff auth, trip identity, SMS OTP challenges and exit authorizations.
    # Repeatable; see MIGRATIONS.
    _apply_security_migration(conn)
    conn.close()


def get_next_holder_id(c):
    """Find the lowest unused holder ID starting from 1."""
    c.execute("SELECT id FROM holders ORDER BY id")
    existing = {row[0] for row in c.fetchall()}
    i = 1
    while i in existing:
        i += 1
    return i


def add_holder(data):
    conn = get_connection()
    c = conn.cursor()
    next_id = get_next_holder_id(c)
    data["id"] = next_id
    # A fresh uid even when the numeric id is reused — see HOLDER_IDENTITY_COLUMNS.
    data["holder_uid"] = uuid.uuid4().hex
    data["phone_e164"] = normalize_holder_phone(data.get("phone_number"))
    c.execute("""
        INSERT INTO holders (
            id, surname, first_name, date_of_birth, sex, height_cm,
            blood_group, street_address, state, phone_number,
            next_of_kin, next_of_kin_phone, religion, nationality,
            holder_uid, phone_e164
        ) VALUES (
            :id, :surname, :first_name, :date_of_birth, :sex, :height_cm,
            :blood_group, :street_address, :state, :phone_number,
            :next_of_kin, :next_of_kin_phone, :religion, :nationality,
            :holder_uid, :phone_e164
        )
    """, data)
    conn.commit()
    conn.close()
    return next_id


def revoke_holder_sms_access(conn, holder_uid, reason):
    """
    Cancel a holder's unfinished SMS OTP challenges and withdraw unused exit
    authorizations earned with them. Used when the phone number changes or
    the holder is deleted, so a code sent to an old number can no longer
    open the gate. Runs inside the caller's transaction and returns the
    number of challenges cancelled.
    """
    if not holder_uid:
        return 0
    conn.execute(
        "UPDATE trip_exit_authorizations SET status='REVOKED' WHERE status='ACTIVE' "
        "AND method='sms_otp' AND challenge_id IN "
        "(SELECT id FROM sms_otp_challenges WHERE holder_uid=?)", (holder_uid,))
    return conn.execute(
        "UPDATE sms_otp_challenges SET status='CANCELLED', provider_error=? "
        "WHERE holder_uid=? AND status IN ('SENDING','ACTIVE')",
        (f"cancelled: {reason}", holder_uid)).rowcount


def get_holder(holder_id):
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT * FROM holders WHERE id = ?", (holder_id,))
    row = c.fetchone()
    conn.close()
    return dict(row) if row else None


def get_all_holders():
    conn = get_connection()
    c = conn.cursor()
    c.execute("""
        SELECT h.*, l.license_number, l.expiry_date, l.license_class,
               CASE WHEN f.holder_id IS NOT NULL THEN 1 ELSE 0 END as has_fingerprint,
               p.file_path as photo_path
        FROM holders h
        LEFT JOIN licenses l ON h.id = l.holder_id
        LEFT JOIN fingerprints f ON h.id = f.holder_id
        LEFT JOIN photos p ON h.id = p.holder_id AND p.photo_type = 'passport'
        ORDER BY h.surname, h.first_name
    """)
    rows = c.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def search_holders(query):
    conn = get_connection()
    c = conn.cursor()
    c.execute("""
        SELECT h.*, l.license_number, l.expiry_date, l.license_class,
               CASE WHEN f.holder_id IS NOT NULL THEN 1 ELSE 0 END as has_fingerprint,
               p.file_path as photo_path
        FROM holders h
        LEFT JOIN licenses l ON h.id = l.holder_id
        LEFT JOIN fingerprints f ON h.id = f.holder_id
        LEFT JOIN photos p ON h.id = p.holder_id AND p.photo_type = 'passport'
        WHERE h.surname LIKE ? OR h.first_name LIKE ? OR l.license_number LIKE ?
        ORDER BY h.surname
    """, (f"%{query}%", f"%{query}%", f"%{query}%"))
    rows = c.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def update_holder(holder_id, data, staff_user_id=None):
    """
    Returns {"phone_changed", "phone_valid", "sms_challenges_cancelled"}.

    A change to the normalised phone number cancels, in the same
    transaction, any SMS code already sent to the old number. (Who may
    change a number is decided by the caller — see app.edit_holder.)
    """
    data["holder_id"] = holder_id
    data["updated_at"] = datetime.now().isoformat()
    data["phone_e164"] = normalize_holder_phone(data.get("phone_number"))
    result = {"phone_changed": False, "phone_valid": data["phone_e164"] is not None,
              "sms_challenges_cancelled": 0}
    with transaction() as conn:
        old = conn.execute("SELECT holder_uid, phone_e164 FROM holders WHERE id=?",
                           (holder_id,)).fetchone()
        conn.execute("""
            UPDATE holders SET
                surname=:surname, first_name=:first_name, date_of_birth=:date_of_birth,
                sex=:sex, height_cm=:height_cm, blood_group=:blood_group,
                street_address=:street_address, state=:state, phone_number=:phone_number,
                next_of_kin=:next_of_kin, next_of_kin_phone=:next_of_kin_phone,
                religion=:religion, nationality=:nationality, updated_at=:updated_at,
                phone_e164=:phone_e164
            WHERE id=:holder_id
        """, data)
        if old and old["phone_e164"] != data["phone_e164"]:
            result["phone_changed"] = True
            result["sms_challenges_cancelled"] = revoke_holder_sms_access(
                conn, old["holder_uid"], "phone_number_changed")
            log_security_event(conn, "holder_phone_changed", holder_id=holder_id,
                               staff_user_id=staff_user_id,
                               detail={"sms_challenges_cancelled":
                                       result["sms_challenges_cancelled"]})
    return result


def delete_holder(holder_id, staff_user_id=None):
    with transaction() as c:
        # Fetch all photo file paths before deleting DB records
        photo_rows = c.execute("SELECT file_path FROM photos WHERE holder_id=?",
                               (holder_id,)).fetchall()
        holder = c.execute("SELECT holder_uid FROM holders WHERE id=?",
                           (holder_id,)).fetchone()
        if holder:
            revoke_holder_sms_access(c, holder["holder_uid"], "holder_deleted")
            log_security_event(c, "holder_deleted", holder_id=holder_id,
                               staff_user_id=staff_user_id)

        # Delete DB records
        c.execute("DELETE FROM fingerprints WHERE holder_id=?", (holder_id,))
        c.execute("DELETE FROM photos WHERE holder_id=?", (holder_id,))
        c.execute("DELETE FROM licenses WHERE holder_id=?", (holder_id,))
        c.execute("DELETE FROM holders WHERE id=?", (holder_id,))

    # Delete photo files from disk
    for row in photo_rows:
        file_path = row["file_path"]
        if file_path and os.path.exists(file_path):
            os.remove(file_path)


def add_license(data):
    conn = get_connection()
    c = conn.cursor()
    c.execute("""
        INSERT INTO licenses (
            holder_id, license_number, date_of_issue, expiry_date,
            license_class, state_of_issue, endorsements, authorized_by
        ) VALUES (
            :holder_id, :license_number, :date_of_issue, :expiry_date,
            :license_class, :state_of_issue, :endorsements, :authorized_by
        )
    """, data)
    lid = c.lastrowid
    conn.commit()
    conn.close()
    return lid


def get_license(holder_id):
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT * FROM licenses WHERE holder_id=?", (holder_id,))
    row = c.fetchone()
    conn.close()
    return dict(row) if row else None


def update_license(holder_id, data):
    data["holder_id"] = holder_id
    conn = get_connection()
    c = conn.cursor()
    c.execute("""
        UPDATE licenses SET
            license_number=:license_number, date_of_issue=:date_of_issue,
            expiry_date=:expiry_date, license_class=:license_class,
            state_of_issue=:state_of_issue, endorsements=:endorsements,
            authorized_by=:authorized_by
        WHERE holder_id=:holder_id
    """, data)
    conn.commit()
    conn.close()


def get_full_record(holder_id):
    conn = get_connection()
    c = conn.cursor()
    c.execute("""
        SELECT h.*, l.license_number, l.date_of_issue, l.expiry_date,
               l.license_class, l.state_of_issue, l.endorsements, l.authorized_by
        FROM holders h
        LEFT JOIN licenses l ON h.id = l.holder_id
        WHERE h.id=?
    """, (holder_id,))
    row = c.fetchone()
    conn.close()
    return dict(row) if row else None


def save_fingerprint(holder_id, template_data):
    conn = get_connection()
    c = conn.cursor()
    c.execute("""
        INSERT INTO fingerprints (holder_id, template_data)
        VALUES (?, ?)
        ON CONFLICT(holder_id) DO UPDATE SET
            template_data=excluded.template_data,
            enrolled_at=datetime('now','localtime')
    """, (holder_id, json.dumps(template_data)))
    conn.commit()
    conn.close()


def get_fingerprint(holder_id):
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT template_data FROM fingerprints WHERE holder_id=?", (holder_id,))
    row = c.fetchone()
    conn.close()
    return json.loads(row["template_data"]) if row else None


def get_all_fingerprints():
    conn = get_connection()
    c = conn.cursor()
    c.execute("""
        SELECT f.holder_id, f.template_data, h.surname, h.first_name, h.holder_uid
        FROM fingerprints f JOIN holders h ON f.holder_id = h.id
    """)
    rows = c.fetchall()
    conn.close()
    return [{"holder_id": r["holder_id"], "holder_uid": r["holder_uid"],
             "name": f"{r['surname']} {r['first_name']}",
             "template": json.loads(r["template_data"])} for r in rows]


def save_photo(holder_id, photo_type, file_path):
    conn = get_connection()
    c = conn.cursor()
    c.execute("DELETE FROM photos WHERE holder_id=? AND photo_type=?", (holder_id, photo_type))
    c.execute("INSERT INTO photos (holder_id, photo_type, file_path) VALUES (?,?,?)", (holder_id, photo_type, file_path))
    conn.commit()
    conn.close()


def get_photos(holder_id):
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT * FROM photos WHERE holder_id=?", (holder_id,))
    rows = c.fetchall()
    conn.close()
    return {r["photo_type"]: r["file_path"] for r in rows}


def get_stats():
    conn = get_connection()
    c = conn.cursor()
    stats = {}
    c.execute("SELECT COUNT(*) as total FROM holders")
    stats["total_holders"] = c.fetchone()["total"]
    c.execute("SELECT COUNT(*) as total FROM licenses")
    stats["total_licenses"] = c.fetchone()["total"]
    c.execute("SELECT COUNT(*) as total FROM fingerprints")
    stats["total_fingerprints"] = c.fetchone()["total"]
    c.execute("SELECT COUNT(*) as total FROM licenses WHERE expiry_date < date('now','localtime')")
    stats["expired"] = c.fetchone()["total"]
    c.execute("SELECT COUNT(*) as total FROM licenses WHERE expiry_date BETWEEN date('now','localtime') AND date('now','localtime', '+30 days')")
    stats["expiring_soon"] = c.fetchone()["total"]
    c.execute("SELECT COUNT(*) as total FROM trips WHERE status='INSIDE'")
    stats["active_trips"] = c.fetchone()["total"]
    c.execute("SELECT COUNT(*) as total FROM trips WHERE date(entry_time)=date('now','localtime')")
    stats["today_trips"] = c.fetchone()["total"]
    conn.close()
    return stats


# ── TRIPS ─────────────────────────────────────────────────────────────────────

def create_trip(plate_number, vehicle_photo_path, face_photo_path, face_encoding,
                fingerprint_template=None, notes="",
                plate_source="manual_entry", plate_confidence=None,
                attributes=None, identity=None):
    """
    plate_source is one of "auto" (unedited ANPR result), "manual_correction"
    (operator edited an ANPR suggestion), or "manual_entry" (typed with no
    ANPR suggestion) — preserved for the audit trail.

    `attributes` is the optional vehicle-attribute payload built by
    build_trip_attributes(). It is written in the same INSERT so a trip is
    never briefly visible without its attributes, and omitting it leaves
    every attribute column NULL — exactly what a pre-feature trip looks
    like, which the readers already tolerate.
    """
    columns = ["plate_number", "vehicle_photo_path", "face_photo_path",
               "face_encoding", "fingerprint_template", "notes",
               "plate_source", "plate_confidence"]
    values = [
        plate_number.strip().upper(),
        vehicle_photo_path,
        face_photo_path,
        json.dumps(face_encoding) if face_encoding else None,
        json.dumps(fingerprint_template) if fingerprint_template else None,
        notes,
        plate_source,
        plate_confidence,
    ]
    _allowed_attribute_columns = {**TRIP_ATTRIBUTE_COLUMNS, **TRIP_ATTRIBUTE_COLUMNS_V2}
    for name, value in (attributes or {}).items():
        if name in _allowed_attribute_columns:
            columns.append(name)
            values.append(value)

    # Server-confirmed identity and exit verification mode, written in the
    # same INSERT as the trip. Without one the trip is UNRESOLVED — never
    # silently a guest. See gate_verification.plan_entry().
    identity = dict(identity or {})
    identity.setdefault("identity_status", IDENTITY_UNRESOLVED)
    for name, value in identity.items():
        if name in TRIP_VERIFICATION_COLUMNS and name not in columns:
            columns.append(name)
            values.append(value)

    conn = get_connection()
    c = conn.cursor()
    c.execute(
        f"INSERT INTO trips ({', '.join(columns)}) "
        f"VALUES ({', '.join('?' * len(values))})",
        values,
    )
    trip_id = c.lastrowid
    conn.commit()
    conn.close()
    return trip_id


def build_trip_attributes(result, corrections=None):
    """
    VehicleAttributeResult (+ optional operator corrections) -> a mapping of
    trip columns.

    Corrections are a dict like {"color": "red", "brand": "toyota"}. A
    corrected value replaces the stored label, its confidence is cleared
    (a human assertion has no model probability) and the method is recorded
    as "operator_confirmed" so an audit can tell the two apart. The
    automatic value it replaced is preserved in the model-versions JSON
    blob under "operator_corrections".
    """
    if result is None:
        return {}
    corrections = {k: v for k, v in (corrections or {}).items() if v}

    color, vtype, brand = result.color, result.vehicle_type, result.brand
    hsv = ((color.evidence or {}).get("hsv") or {}).get("hsv")

    def label_of(pred, key):
        if key in corrections:
            return corrections[key]
        return pred.label if pred.is_usable else None

    def confidence_of(pred, key):
        if key in corrections:
            return None
        return round(float(pred.confidence), 4) if pred.is_usable else None

    versions = dict(result.model_versions or {})
    if corrections:
        versions["operator_corrections"] = {
            key: {
                "automatic_label": getattr(result, attr).label,
                "automatic_confidence": round(float(getattr(result, attr).confidence), 4),
                "automatic_status": getattr(result, attr).status,
                "corrected_label": value,
            }
            for key, attr, value in (
                ("color", "color", corrections.get("color")),
                ("vehicle_type", "vehicle_type", corrections.get("vehicle_type")),
                ("brand", "brand", corrections.get("brand")),
            ) if value
        }

    top3 = [t.to_dict() if hasattr(t, "to_dict") else dict(t)
            for t in (brand.top_predictions or [])[:3]]

    return {
        "vehicle_color": label_of(color, "color"),
        "vehicle_color_confidence": confidence_of(color, "color"),
        "vehicle_color_method": ("operator_confirmed" if "color" in corrections
                                 else color.method),
        "vehicle_color_hsv": safe_json_dumps(hsv),
        "vehicle_type": label_of(vtype, "vehicle_type"),
        "vehicle_type_confidence": confidence_of(vtype, "vehicle_type"),
        "vehicle_brand": label_of(brand, "brand"),
        "vehicle_brand_confidence": confidence_of(brand, "brand"),
        "vehicle_brand_top3_json": safe_json_dumps(top3) if top3 else None,
        "vehicle_yolo_class": result.yolo_class,
        "vehicle_attributes_status": result.status,
        "vehicle_attributes_model_versions_json": safe_json_dumps(versions) if versions else None,
        "vehicle_attributes_processing_ms": round(float(result.processing_time_ms), 2),
    }


def build_attribute_columns(result, corrections=None):
    """
    attributes.VehicleAttributeResult (+ optional operator corrections) ->
    a mapping of trip columns.

    `corrections` is {"colour": "red", "type": "suv", "brand": "toyota"}.
    A corrected value replaces the stored label, clears the confidence (a
    human assertion has no model probability) and sets `*_source` to
    manual_correction or manual_entry, so the audit trail always
    distinguishes an operator edit from an automatic reading.
    """
    if result is None:
        return {}
    corrections = {k: str(v).strip() for k, v in (corrections or {}).items()
                   if str(v or "").strip()}

    # The colour columns predate this package and use the American spelling.
    column_for = {"colour": "vehicle_color", "type": "vehicle_type",
                  "brand": "vehicle_brand"}
    out = {
        "attr_status": result.status,
        "attr_coco_class": result.coco_class,
        "attr_vehicle_box_json": safe_json_dumps(list(result.vehicle_box)
                                                 if result.vehicle_box else None),
        "attr_model_version": (result.model_versions or {}).get("attributes"),
        "attr_processing_ms": round(float(result.processing_time_ms), 2),
    }

    for attribute, column in column_for.items():
        value = getattr(result, attribute)
        if attribute in corrections:
            from attributes.models import AttributeValue
            corrected = AttributeValue.operator(attribute, corrections[attribute], value)
            out[column] = corrected.value
            out[f"{column}_confidence"] = None
            out[f"{column}_source"] = corrected.source
            out[f"{column}_votes"] = None
        else:
            out[column] = value.value if value.is_usable else None
            out[f"{column}_confidence"] = (round(float(value.confidence), 4)
                                           if value.is_usable else None)
            out[f"{column}_source"] = value.source if value.is_usable else None
            out[f"{column}_votes"] = value.votes if value.is_usable else None

    top3 = [s.to_dict() if hasattr(s, "to_dict") else dict(s)
            for s in (result.brand.top_k or [])[:3]]
    out["vehicle_brand_top3_json"] = safe_json_dumps(top3) if top3 else None
    return out


def record_attribute_mismatch(trip_id, mismatch_flags, reason=None):
    """
    Record which attributes disagreed with the registered vehicle.

    ADVISORY ONLY. This exists so the operator-confirmation dialog can show
    what differed; nothing reads it to allow or deny a movement.
    """
    return update_trip_attributes(trip_id, {
        "attr_mismatch_flags_json": safe_json_dumps(mismatch_flags or []),
        "attribute_mismatch_reason": reason,
    })


def update_trip_attributes(trip_id, attributes):
    """Apply an attribute mapping to an existing trip. Unknown keys are ignored."""
    allowed = {**TRIP_ATTRIBUTE_COLUMNS, **TRIP_ATTRIBUTE_COLUMNS_V2}
    updates = {k: v for k, v in (attributes or {}).items() if k in allowed}
    if not updates:
        return False
    conn = get_connection()
    c = conn.cursor()
    c.execute(
        f"UPDATE trips SET {', '.join(f'{k}=?' for k in updates)} WHERE id=?",
        list(updates.values()) + [trip_id],
    )
    affected = c.rowcount
    conn.commit()
    conn.close()
    return affected > 0


def record_exit_attributes(trip_id, exit_result=None, comparison=None):
    """
    Store the exit-time attribute readings and their comparison against the
    entry record.

    This is evidence only: it records what was seen and whether it matched,
    and has no bearing on whether the exit is authorised.
    """
    updates = {}
    if exit_result is not None:
        updates["exit_vehicle_color"] = (exit_result.color.label
                                         if exit_result.color.is_usable else None)
        updates["exit_vehicle_type"] = (exit_result.vehicle_type.label
                                        if exit_result.vehicle_type.is_usable else None)
        updates["exit_vehicle_brand"] = (exit_result.brand.label
                                         if exit_result.brand.is_usable else None)
    if comparison is not None:
        updates["color_match"] = comparison.match_flag("color")
        updates["type_match"] = comparison.match_flag("vehicle_type")
        updates["brand_match"] = comparison.match_flag("brand")
        updates["attribute_mismatch_reason"] = comparison.mismatch_reason
    if not updates:
        return False
    conn = get_connection()
    c = conn.cursor()
    c.execute(
        f"UPDATE trips SET {', '.join(f'{k}=?' for k in updates)} WHERE id=?",
        list(updates.values()) + [trip_id],
    )
    affected = c.rowcount
    conn.commit()
    conn.close()
    return affected > 0


def _decode_trip(row):
    """
    Row -> dict with every JSON column decoded.

    Uses safe_json_loads throughout: a trip written before the attribute
    columns existed has NULLs (decoded to None), and a corrupted blob
    degrades to None rather than making the whole trip unreadable.
    """
    t = dict(row)
    for key in ("face_encoding", "fingerprint_template"):
        if t.get(key):
            t[key] = safe_json_loads(t[key])
    for key in _TRIP_JSON_COLUMNS:
        if key in t and t.get(key):
            t[key] = safe_json_loads(t[key])
    return t


def get_open_trip(plate_number):
    conn = get_connection()
    c = conn.cursor()
    c.execute("""
        SELECT * FROM trips
        WHERE plate_number=? COLLATE NOCASE AND status='INSIDE'
        ORDER BY entry_time DESC LIMIT 1
    """, (plate_number.strip().upper(),))
    row = c.fetchone()
    conn.close()
    if not row:
        return None
    return _decode_trip(row)


def get_trip(trip_id):
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT * FROM trips WHERE id=?", (trip_id,))
    row = c.fetchone()
    conn.close()
    if not row:
        return None
    return _decode_trip(row)


def close_trip(trip_id, exit_result, exit_face_photo_path, face_distance):
    conn = get_connection()
    c = conn.cursor()
    new_status = "EXITED" if exit_result == "GRANTED" else "DENIED"
    # Fetch paths before nulling them
    c.execute("SELECT vehicle_photo_path, face_photo_path FROM trips WHERE id=?", (trip_id,))
    old = c.fetchone()
    # Biometrics and photos are cleared on close, as before. The vehicle
    # attribute columns are deliberately NOT cleared: they are
    # non-biometric observations that form the anti-theft audit trail.
    c.execute("""
        UPDATE trips SET
            status=?, exit_time=datetime('now','localtime'),
            exit_face_photo_path=NULL, exit_result=?,
            face_distance=?,
            face_encoding=NULL, fingerprint_template=NULL,
            vehicle_photo_path=NULL, face_photo_path=NULL
        WHERE id=?
    """, (new_status, exit_result, face_distance, trip_id))
    conn.commit()
    conn.close()
    remove_trip_files(trip_id, [exit_face_photo_path] +
                      ([old["vehicle_photo_path"], old["face_photo_path"]] if old else []))


def remove_gate_file(path):
    """
    Delete a gate capture, but only if it resolves inside gate_photos/.
    Paths reach here from the database; this check means a bad value can
    never delete an arbitrary file.
    """
    if not path:
        return False
    root = os.path.realpath(GATE_PHOTOS_DIR)
    candidate = os.path.realpath(path)
    if not candidate.startswith(root + os.sep) or not os.path.isfile(candidate):
        return False
    try:
        os.remove(candidate)
        return True
    except OSError:
        return False


def remove_trip_files(trip_id, paths=()):
    """Delete a closed trip's folder and any loose capture files."""
    _delete_trip_folder(trip_id)
    for path in paths:
        remove_gate_file(path)


def _delete_orphan_verification_rows(c):
    """SMS challenges and exit authorizations of trips that no longer exist."""
    c.execute("DELETE FROM sms_otp_challenges WHERE trip_id NOT IN (SELECT id FROM trips)")
    c.execute("DELETE FROM trip_exit_authorizations "
              "WHERE trip_id NOT IN (SELECT id FROM trips)")


def update_trip_photos(trip_id, vehicle_photo_path, face_photo_path):
    conn = get_connection()
    c = conn.cursor()
    c.execute("UPDATE trips SET vehicle_photo_path=?, face_photo_path=? WHERE id=?",
              (vehicle_photo_path, face_photo_path, trip_id))
    conn.commit()
    conn.close()


def _delete_trip_folder(trip_id):
    """Remove gate_photos/trip_{id}/ directory and all contents."""
    import shutil
    folder = os.path.join("gate_photos", f"trip_{trip_id}")
    if os.path.isdir(folder):
        try:
            shutil.rmtree(folder)
        except Exception:
            pass


def delete_trip(trip_id):
    """Delete a single trip record (any status) and its associated photo folder."""
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT vehicle_photo_path, face_photo_path, exit_face_photo_path FROM trips WHERE id=?",
              (trip_id,))
    row = c.fetchone()
    c.execute("DELETE FROM trips WHERE id=?", (trip_id,))
    _delete_orphan_verification_rows(c)
    conn.commit()
    conn.close()
    remove_trip_files(trip_id, [row["vehicle_photo_path"], row["face_photo_path"],
                                row["exit_face_photo_path"]] if row else [])


def get_active_trips():
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT * FROM trips WHERE status='INSIDE' ORDER BY entry_time DESC")
    rows = c.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_all_trips(limit=100):
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT * FROM trips ORDER BY entry_time DESC LIMIT ?", (limit,))
    rows = c.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def clear_trip_history():
    """Delete all completed trips (EXITED / DENIED). Active (INSIDE) trips are kept."""
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT id, vehicle_photo_path, face_photo_path, exit_face_photo_path FROM trips WHERE status != 'INSIDE'")
    rows = c.fetchall()
    c.execute("DELETE FROM trips WHERE status != 'INSIDE'")
    deleted = c.rowcount
    _delete_orphan_verification_rows(c)
    conn.commit()
    conn.close()
    for row in rows:
        remove_trip_files(row["id"], [row["vehicle_photo_path"], row["face_photo_path"],
                                      row["exit_face_photo_path"]])
    return deleted


def clear_all_trips():
    """Delete ALL trips (including INSIDE) and their photo folders."""
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT id, vehicle_photo_path, face_photo_path, exit_face_photo_path FROM trips")
    rows = c.fetchall()
    c.execute("DELETE FROM trips")
    deleted = c.rowcount
    _delete_orphan_verification_rows(c)
    conn.commit()
    conn.close()
    for row in rows:
        remove_trip_files(row["id"], [row["vehicle_photo_path"], row["face_photo_path"],
                                      row["exit_face_photo_path"]])
    return deleted


# ── TRIP PASSCODE ─────────────────────────────────────────────────────────────
# Guest passcodes are stored only as a password hash (trips.passcode_hash)
# and verified with an attempt limit in gate_verification.verify_passcode().
# The old plaintext trips.passcode column is kept only so the migration can
# hash legacy values; nothing writes to it any more.

# ── OPERATOR ATTRIBUTE CORRECTIONS (optional training-data collection) ───────
# Separate from trips on purpose: this is future fine-tuning data, never an
# input to any authorization decision. See vehicle_attributes/feedback.py
# for the privacy rules and the retention policy.

def record_attribute_correction(attribute, corrected_label, original_label=None,
                                original_confidence=None, original_status=None,
                                image_reference=None, model_version=None,
                                trip_id=None):
    conn = get_connection()
    c = conn.cursor()
    c.execute("""
        INSERT INTO attribute_corrections
            (trip_id, attribute, original_label, original_confidence,
             original_status, corrected_label, image_reference, model_version)
        VALUES (?,?,?,?,?,?,?,?)
    """, (trip_id, attribute, original_label, original_confidence,
          original_status, corrected_label, image_reference, model_version))
    row_id = c.lastrowid
    conn.commit()
    conn.close()
    return row_id


def get_attribute_corrections(limit=200, attribute=None):
    conn = get_connection()
    c = conn.cursor()
    if attribute:
        c.execute("""SELECT * FROM attribute_corrections WHERE attribute=?
                     ORDER BY created_at DESC LIMIT ?""", (attribute, limit))
    else:
        c.execute("SELECT * FROM attribute_corrections ORDER BY created_at DESC LIMIT ?",
                  (limit,))
    rows = c.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def prune_attribute_corrections(max_age_days=30):
    """Delete corrections past the configured retention window."""
    conn = get_connection()
    c = conn.cursor()
    c.execute("DELETE FROM attribute_corrections "
              "WHERE created_at < datetime('now','localtime', ?)",
              (f"-{int(max_age_days)} days",))
    deleted = c.rowcount
    conn.commit()
    conn.close()
    return deleted


def count_attribute_corrections():
    conn = get_connection()
    c = conn.cursor()
    try:
        c.execute("SELECT COUNT(*) AS total FROM attribute_corrections")
        total = c.fetchone()["total"]
    except Exception:
        total = 0
    conn.close()
    return total


init_db()