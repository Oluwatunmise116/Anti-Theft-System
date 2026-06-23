import sqlite3
import os
import json
from datetime import datetime

DB_PATH = "license.db"
PHOTOS_DIR = "photos"
SIGNATURES_DIR = "signatures"


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    os.makedirs(PHOTOS_DIR, exist_ok=True)
    os.makedirs(SIGNATURES_DIR, exist_ok=True)
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
            created_at          TEXT DEFAULT (datetime('now')),
            updated_at          TEXT DEFAULT (datetime('now'))
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
            created_at          TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (holder_id) REFERENCES holders(id)
        );

        CREATE TABLE IF NOT EXISTS fingerprints (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            holder_id           INTEGER NOT NULL UNIQUE,
            template_data       TEXT NOT NULL,
            enrolled_at         TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (holder_id) REFERENCES holders(id)
        );

        CREATE TABLE IF NOT EXISTS photos (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            holder_id           INTEGER NOT NULL,
            photo_type          TEXT CHECK(photo_type IN ('passport', 'holder_signature', 'authorized_signature')),
            file_path           TEXT NOT NULL,
            uploaded_at         TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (holder_id) REFERENCES holders(id)
        );

        CREATE TABLE IF NOT EXISTS gate_sessions (
            id                      INTEGER PRIMARY KEY AUTOINCREMENT,
            plate_number            TEXT NOT NULL,
            vehicle_photo_path      TEXT,
            face_photo_path         TEXT,
            face_encoding           TEXT,
            fingerprint_template    TEXT,
            entry_time              TEXT DEFAULT (datetime('now')),
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
            entry_time           TEXT DEFAULT (datetime('now')),
            exit_time            TEXT,
            exit_face_photo_path TEXT,
            exit_result          TEXT,
            face_distance        REAL,
            color_match          INTEGER,
            notes                TEXT
        );

        DROP TABLE IF EXISTS vehicle_driver_bindings;
        DROP TABLE IF EXISTS vehicles;
    """)
    conn.commit()
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
    c.execute("""
        INSERT INTO holders (
            id, surname, first_name, date_of_birth, sex, height_cm,
            blood_group, street_address, state, phone_number,
            next_of_kin, next_of_kin_phone, religion, nationality
        ) VALUES (
            :id, :surname, :first_name, :date_of_birth, :sex, :height_cm,
            :blood_group, :street_address, :state, :phone_number,
            :next_of_kin, :next_of_kin_phone, :religion, :nationality
        )
    """, data)
    conn.commit()
    conn.close()
    return next_id


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


def update_holder(holder_id, data):
    data["holder_id"] = holder_id
    data["updated_at"] = datetime.now().isoformat()
    conn = get_connection()
    c = conn.cursor()
    c.execute("""
        UPDATE holders SET
            surname=:surname, first_name=:first_name, date_of_birth=:date_of_birth,
            sex=:sex, height_cm=:height_cm, blood_group=:blood_group,
            street_address=:street_address, state=:state, phone_number=:phone_number,
            next_of_kin=:next_of_kin, next_of_kin_phone=:next_of_kin_phone,
            religion=:religion, nationality=:nationality, updated_at=:updated_at
        WHERE id=:holder_id
    """, data)
    conn.commit()
    conn.close()


def delete_holder(holder_id):
    conn = get_connection()
    c = conn.cursor()

    # Fetch all photo file paths before deleting DB records
    c.execute("SELECT file_path FROM photos WHERE holder_id=?", (holder_id,))
    photo_rows = c.fetchall()

    # Delete DB records
    c.execute("DELETE FROM fingerprints WHERE holder_id=?", (holder_id,))
    c.execute("DELETE FROM photos WHERE holder_id=?", (holder_id,))
    c.execute("DELETE FROM licenses WHERE holder_id=?", (holder_id,))
    c.execute("DELETE FROM holders WHERE id=?", (holder_id,))
    conn.commit()
    conn.close()

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
            enrolled_at=datetime('now')
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
        SELECT f.holder_id, f.template_data, h.surname, h.first_name
        FROM fingerprints f JOIN holders h ON f.holder_id = h.id
    """)
    rows = c.fetchall()
    conn.close()
    return [{"holder_id": r["holder_id"], "name": f"{r['surname']} {r['first_name']}", "template": json.loads(r["template_data"])} for r in rows]


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
    c.execute("SELECT COUNT(*) as total FROM licenses WHERE expiry_date < date('now')")
    stats["expired"] = c.fetchone()["total"]
    c.execute("SELECT COUNT(*) as total FROM licenses WHERE expiry_date BETWEEN date('now') AND date('now', '+30 days')")
    stats["expiring_soon"] = c.fetchone()["total"]
    c.execute("SELECT COUNT(*) as total FROM trips WHERE status='INSIDE'")
    stats["active_trips"] = c.fetchone()["total"]
    c.execute("SELECT COUNT(*) as total FROM trips WHERE date(entry_time)=date('now')")
    stats["today_trips"] = c.fetchone()["total"]
    conn.close()
    return stats


# ── TRIPS ─────────────────────────────────────────────────────────────────────

def create_trip(plate_number, vehicle_photo_path, face_photo_path, face_encoding,
                fingerprint_template=None, notes=""):
    conn = get_connection()
    c = conn.cursor()
    c.execute("""
        INSERT INTO trips
            (plate_number, vehicle_photo_path, face_photo_path, face_encoding,
             fingerprint_template, notes)
        VALUES (?,?,?,?,?,?)
    """, (
        plate_number.strip().upper(),
        vehicle_photo_path,
        face_photo_path,
        json.dumps(face_encoding) if face_encoding else None,
        json.dumps(fingerprint_template) if fingerprint_template else None,
        notes,
    ))
    trip_id = c.lastrowid
    conn.commit()
    conn.close()
    return trip_id


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
    t = dict(row)
    if t.get("face_encoding"):
        t["face_encoding"] = json.loads(t["face_encoding"])
    if t.get("fingerprint_template"):
        t["fingerprint_template"] = json.loads(t["fingerprint_template"])
    if t.get("vehicle_color_hsv"):
        t["vehicle_color_hsv"] = json.loads(t["vehicle_color_hsv"])
    return t


def get_trip(trip_id):
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT * FROM trips WHERE id=?", (trip_id,))
    row = c.fetchone()
    conn.close()
    if not row:
        return None
    t = dict(row)
    if t.get("face_encoding"):
        t["face_encoding"] = json.loads(t["face_encoding"])
    if t.get("fingerprint_template"):
        t["fingerprint_template"] = json.loads(t["fingerprint_template"])
    if t.get("vehicle_color_hsv"):
        t["vehicle_color_hsv"] = json.loads(t["vehicle_color_hsv"])
    return t


def close_trip(trip_id, exit_result, exit_face_photo_path, face_distance):
    conn = get_connection()
    c = conn.cursor()
    new_status = "EXITED" if exit_result == "GRANTED" else "DENIED"
    # Fetch paths before nulling them
    c.execute("SELECT vehicle_photo_path, face_photo_path FROM trips WHERE id=?", (trip_id,))
    old = c.fetchone()
    c.execute("""
        UPDATE trips SET
            status=?, exit_time=datetime('now'),
            exit_face_photo_path=NULL, exit_result=?,
            face_distance=?,
            face_encoding=NULL, fingerprint_template=NULL,
            vehicle_photo_path=NULL, face_photo_path=NULL
        WHERE id=?
    """, (new_status, exit_result, face_distance, trip_id))
    conn.commit()
    conn.close()
    # Delete per-trip folder and any loose exit photos
    _delete_trip_folder(trip_id)
    if exit_face_photo_path and os.path.exists(exit_face_photo_path):
        try:
            os.remove(exit_face_photo_path)
        except Exception:
            pass
    if old:
        for path in [old["vehicle_photo_path"], old["face_photo_path"]]:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except Exception:
                    pass


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
    conn.commit()
    conn.close()
    _delete_trip_folder(trip_id)
    if row:
        for path in [row["vehicle_photo_path"], row["face_photo_path"], row["exit_face_photo_path"]]:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except Exception:
                    pass


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
    conn.commit()
    conn.close()
    for row in rows:
        _delete_trip_folder(row["id"])
        for path in [row["vehicle_photo_path"], row["face_photo_path"], row["exit_face_photo_path"]]:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except Exception:
                    pass
    return deleted


def clear_all_trips():
    """Delete ALL trips (including INSIDE) and their photo folders."""
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT id, vehicle_photo_path, face_photo_path, exit_face_photo_path FROM trips")
    rows = c.fetchall()
    c.execute("DELETE FROM trips")
    deleted = c.rowcount
    conn.commit()
    conn.close()
    for row in rows:
        _delete_trip_folder(row["id"])
        for path in [row["vehicle_photo_path"], row["face_photo_path"], row["exit_face_photo_path"]]:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except Exception:
                    pass
    return deleted


init_db()