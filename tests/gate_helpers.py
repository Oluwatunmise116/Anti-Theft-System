"""Helpers for the staff-auth, biometric, SMS OTP and guest-passcode tests."""
import re

from werkzeug.security import generate_password_hash

import auth
import database as db

PASSWORD = "correct horse battery"

HOLDER_FIELDS = ("surname", "first_name", "date_of_birth", "sex", "height_cm",
                 "blood_group", "street_address", "state", "phone_number",
                 "next_of_kin", "next_of_kin_phone", "religion", "nationality")

#: Entry templates stored on a trip. Their content is irrelevant: the tests
#: fake the comparison, but a trip must HAVE a template to be matched.
FACE = [0.1] * 8
FINGERPRINT = [1, 2, 3, 4]


# ── staff ────────────────────────────────────────────────────────────────────

def create_staff(username="operator1", role="operator"):
    return auth.create_user(username, PASSWORD, role)


def login(client, username, password=PASSWORD):
    """Sign in through the real login form; every later request from this
    client carries the session's CSRF token."""
    page = client.get("/login")
    token = re.search(rb'name="csrf_token" value="([^"]+)"', page.data).group(1).decode()
    response = client.post("/login", data={"username": username, "password": password,
                                           "csrf_token": token})
    assert response.status_code == 302, response.data[:300]
    with client.session_transaction() as sess:
        client.environ_base["HTTP_X_CSRF_TOKEN"] = sess["csrf"]
    return client


def staff_client(app, username="operator1", role="operator"):
    create_staff(username, role)
    return login(app.test_client(), username)


def staff_context(client):
    with client.session_transaction() as sess:
        token = sess["sid"]
    return auth.load_session(token)


# ── holders ──────────────────────────────────────────────────────────────────

def add_holder(surname="OKAFOR", first_name="Ada", phone="0803 123 4567"):
    return db.add_holder({
        "surname": surname, "first_name": first_name, "date_of_birth": "1990-01-01",
        "sex": "F", "height_cm": 170.0, "blood_group": "O+",
        "street_address": "1 Gate Road", "state": "Lagos", "phone_number": phone,
        "next_of_kin": "", "next_of_kin_phone": "", "religion": "", "nationality": "Nigerian",
    })


def holder_row(holder_id):
    return rows("SELECT * FROM holders WHERE id=?", (holder_id,))[0]


def change_phone(holder_id, phone):
    record = holder_row(holder_id)
    data = {key: record[key] for key in HOLDER_FIELDS}
    data["phone_number"] = phone
    return db.update_holder(holder_id, data)


def edit_form(holder_id, phone):
    """The full holder edit form, as the browser posts it."""
    record = holder_row(holder_id)
    form = {key: "" if record[key] is None else str(record[key]) for key in HOLDER_FIELDS}
    form.update(phone_number=phone, license_number=f"LIC-{holder_id}",
                date_of_issue="2020-01-01", expiry_date="2030-01-01", license_class="B",
                state_of_issue="Lagos")
    return form


def last_sms_code(mock):
    """The code a driver would read from the most recent mocked SMS."""
    return mock.sent[-1]["code"]


# ── trips ────────────────────────────────────────────────────────────────────

def registered_trip(holder_id, plate="KJA456GH", face=None, fingerprint=None):
    holder = holder_row(holder_id)
    return db.create_trip(plate, None, None, face, fingerprint_template=fingerprint, identity={
        "identity_status": db.IDENTITY_REGISTERED, "identity_method": "face",
        "holder_id": holder_id, "holder_uid": holder["holder_uid"],
        "verification_mode": db.MODE_SMS_OTP})


def guest_trip(passcode="4821", plate="GST123AA", face=None, fingerprint=None):
    return db.create_trip(plate, None, None, face, fingerprint_template=fingerprint, identity={
        "identity_status": db.IDENTITY_GUEST, "identity_method": "face+fingerprint",
        "verification_mode": db.MODE_PASSCODE,
        "passcode_hash": generate_password_hash(passcode)})


def unresolved_trip(plate="UNR111AA", passcode=None, face=None,
                    identity_status=db.IDENTITY_UNRESOLVED):
    identity = {"identity_status": identity_status}
    if passcode:
        identity["passcode_hash"] = generate_password_hash(passcode)
    return db.create_trip(plate, None, None, face, identity=identity)


def trip_row(trip_id):
    return rows("SELECT * FROM trips WHERE id=?", (trip_id,))[0]


def rows(sql, params=()):
    conn = db.get_connection()
    try:
        return [dict(r) for r in conn.execute(sql, params).fetchall()]
    finally:
        conn.close()
