"""
Exit verification.

Primary — face or fingerprint, compared with THIS trip's own entry capture
(gate_manager.run_exit_verify / run_fp_exit). A match authorizes the exit
for the gate session that ran the scan. No code is needed.

Fallback — for when biometrics fail or are unavailable. Which fallback is
fixed by the identity recorded on the trip at entry, never by the browser:
  * registered driver (verification_mode SMS_OTP): a one-time code that
    Robase generates, texts to the phone number on the trip's linked holder
    record, and verifies;
  * guest (PASSCODE): the passcode chosen at entry, stored only as a hash;
  * unresolved or legacy (NULL): no fallback until an administrator
    reconciles the trip. Biometrics still work.
An SMS failure never moves a registered driver to the passcode.

Every success creates one exit authorization bound to the trip and to the
gate session that earned it; /gate/exit/confirm consumes it in the same
transaction that closes the trip. All state lives in SQLite and every
check-then-write runs in db.transaction() (BEGIN IMMEDIATE). Calls to Robase
never run inside a transaction.
"""
import logging
import math
import time
import uuid

from werkzeug.security import check_password_hash, generate_password_hash

import database as db
import phone_utils
import robase_client as robase
import verification_settings as vs

log = logging.getLogger(__name__)

MODE_SMS = db.MODE_SMS_OTP
MODE_PASSCODE = db.MODE_PASSCODE
MODES = (MODE_SMS, MODE_PASSCODE)

METHOD_FACE = "face"
METHOD_FINGERPRINT = "fingerprint"
METHOD_SMS = "sms_otp"
METHOD_PASSCODE = "passcode"
BIOMETRIC_METHODS = (METHOD_FACE, METHOD_FINGERPRINT)
FALLBACK_METHOD = {MODE_SMS: METHOD_SMS, MODE_PASSCODE: METHOD_PASSCODE}

REGISTERED = db.IDENTITY_REGISTERED
GUEST = db.IDENTITY_GUEST
UNRESOLVED = db.IDENTITY_UNRESOLVED

RECONCILE_ACTIONS = ("assign_holder", "confirm_guest", "unlock")

#: Patched by tests to move time forward without sleeping.
_clock = time.time


class VerificationError(Exception):
    def __init__(self, message, status=400, code="invalid", **extra):
        super().__init__(message)
        self.message = message
        self.status = status
        self.code = code
        self.extra = extra

    def payload(self) -> dict:
        return {"error": self.message, "code": self.code, **self.extra}


def _settings(settings):
    return settings or vs.load()


# ── INPUT ────────────────────────────────────────────────────────────────────

def clean_code(value) -> str:
    """Codes arrive as JSON strings. None/missing -> ''. Any other type is
    rejected instead of guessed at (a JSON number would lose leading zeros)."""
    if value is None:
        return ""
    if not isinstance(value, str):
        raise VerificationError("The code must be sent as text.", 400, "bad_input")
    return value.strip()


def _is_digits(code: str) -> bool:
    return code.isascii() and code.isdigit()


def validate_passcode(value, settings=None) -> str:
    s = _settings(settings)
    code = clean_code(value)
    rule = f"{s.passcode_min_length}–{s.passcode_max_length} digit"
    if not code:
        raise VerificationError(f"Enter a {rule} passcode.", 400, "passcode_required")
    if not _is_digits(code) or not s.passcode_min_length <= len(code) <= s.passcode_max_length:
        raise VerificationError(f"The passcode must be a {rule} number.", 400, "passcode_invalid")
    return code


# ── HOLDERS AND PHONE NUMBERS ────────────────────────────────────────────────

def _holder_row(conn, holder_id, holder_uid):
    if not holder_id or not holder_uid:
        return None
    return conn.execute("SELECT id, holder_uid, surname, first_name, phone_number, phone_e164 "
                        "FROM holders WHERE id=? AND holder_uid=?",
                        (holder_id, holder_uid)).fetchone()


def _sms_destination(holder, settings):
    """The E.164 number SMS codes go to: the linked holder's saved number,
    normalised and validated again. None when missing or invalid."""
    if holder is None:
        return None
    return phone_utils.normalize_phone(holder["phone_e164"] or holder["phone_number"],
                                       settings.phone_default_region)


def _name(holder) -> str:
    return f"{holder['surname']}, {holder['first_name']}"


def holder_sms_status(holder: dict, settings=None) -> dict:
    """Whether SMS codes can reach this holder. No full number is returned."""
    s = _settings(settings)
    phone = phone_utils.normalize_phone(holder.get("phone_e164") or holder.get("phone_number"),
                                        s.phone_default_region)
    return {"phone_valid": phone is not None, "phone_hint": phone_utils.mask_phone(phone)}


# ── ENTRY ────────────────────────────────────────────────────────────────────

def resolve_entry_identity(captured: dict, settings=None) -> dict:
    """
    Identity from the server-side results of this capture's face and
    fingerprint searches (written by gate_manager, never by the browser).

      * any biometric matched a holder, and all matches agree -> REGISTERED
      * face AND fingerprint searches completed with no match   -> GUEST
      * anything else (not run, failed, conflicting)            -> UNRESOLVED
    """
    s = _settings(settings)
    checks = {"face": captured.get("identity_face"),
              "fingerprint": captured.get("identity_fp")}
    matches = {m: c for m, c in checks.items()
               if isinstance(c, dict) and c.get("result") == "match"}
    no_match = {m for m, c in checks.items()
                if isinstance(c, dict) and c.get("result") == "no_match"}

    if matches:
        identities = {(c.get("holder_id"), c.get("holder_uid")) for c in matches.values()}
        if len(identities) != 1:
            return {"status": UNRESOLVED, "methods": sorted(matches),
                    "reason": "Face and fingerprint matched different registered holders."}
        holder_id, holder_uid = identities.pop()
        conn = db.get_connection()
        try:
            holder = _holder_row(conn, holder_id, holder_uid)
        finally:
            conn.close()
        if holder is None:
            return {"status": UNRESOLVED, "methods": sorted(matches),
                    "reason": "The matched holder record has changed or been deleted."}
        phone = _sms_destination(holder, s)
        return {"status": REGISTERED, "mode": MODE_SMS, "methods": sorted(matches),
                "holder_id": holder["id"], "holder_uid": holder["holder_uid"],
                "driver_name": _name(holder), "sms_fallback_available": phone is not None,
                "phone_hint": phone_utils.mask_phone(phone)}

    if no_match == {"face", "fingerprint"}:
        return {"status": GUEST, "mode": MODE_PASSCODE, "methods": ["face", "fingerprint"]}

    missing = [m for m in ("face", "fingerprint") if m not in no_match]
    return {"status": UNRESOLVED, "methods": sorted(no_match),
            "reason": f"Identity check incomplete: {' and '.join(missing)} search not completed."}


def entry_identity_view(captured: dict, settings=None) -> dict:
    """What the entry page needs to choose its fallback controls."""
    s = _settings(settings)
    identity = resolve_entry_identity(captured, s)
    view = {"identity_status": identity["status"], "verification_mode": identity.get("mode"),
            "methods": identity.get("methods", []), "reason": identity.get("reason"),
            "passcode": {"min_length": s.passcode_min_length,
                         "max_length": s.passcode_max_length}}
    if identity["status"] == REGISTERED:
        view.update(driver_name=identity["driver_name"],
                    sms_fallback_available=identity["sms_fallback_available"],
                    phone_hint=identity["phone_hint"])
    return view


def plan_entry(captured: dict, data: dict, staff, settings=None) -> dict:
    """
    Validate everything an entry needs BEFORE the trip is created, and
    return the identity columns for db.create_trip(). Raises
    VerificationError; nothing is written.
    """
    s = _settings(settings)
    identity = resolve_entry_identity(captured, s)
    status = identity["status"]
    raw_passcode = data.get("passcode")
    columns = {"identity_status": status,
               "identity_method": "+".join(identity.get("methods") or []) or None,
               "entry_session_id": staff.session_id,
               "entry_staff_user_id": staff.user_id}

    if status == REGISTERED:
        if clean_code(raw_passcode):
            raise VerificationError(
                "This driver is registered: exit uses face or fingerprint, with an SMS code as "
                "the fallback. A guest passcode is not accepted for registered drivers.",
                409, "registered_driver_passcode_rejected",
                identity_status=status, verification_mode=MODE_SMS)
        # A missing phone number does not block entry: face and fingerprint
        # remain the primary exit, and correcting the number on the holder
        # record later restores the SMS fallback for this trip.
        columns.update(holder_id=identity["holder_id"], holder_uid=identity["holder_uid"],
                       verification_mode=MODE_SMS)
    elif status == GUEST:
        code = validate_passcode(raw_passcode, s)
        columns.update(verification_mode=MODE_PASSCODE, passcode_hash=generate_password_hash(code))
    else:
        if data.get("accept_unresolved") is not True:
            raise VerificationError(
                f"{identity['reason']} Complete the face and fingerprint checks, or record the "
                "entry as unresolved — face or fingerprint can still clear it at exit, and an "
                "administrator must reconcile it before any fallback can be used.",
                409, "identity_unresolved", identity_status=status,
                requires_acknowledgement=True)
        code = clean_code(raw_passcode)
        if code:
            # Kept so an administrator can later confirm a guest without
            # needing the driver to choose a new code.
            columns["passcode_hash"] = generate_password_hash(validate_passcode(code, s))
        columns.update(verification_mode=None, reconciliation_note=identity["reason"])
    return columns


# ── SHARED CHECKS ────────────────────────────────────────────────────────────

def _open_trip(conn, trip_id, expected_mode=None):
    """An open trip whose fallback is `expected_mode` and not locked."""
    trip = conn.execute("SELECT * FROM trips WHERE id=?", (trip_id,)).fetchone()
    if trip is None:
        raise VerificationError("Trip not found.", 404, "trip_not_found")
    if trip["status"] != "INSIDE":
        raise VerificationError("This trip is already closed.", 409, "trip_closed")
    mode = trip["verification_mode"]
    if mode not in MODES:
        raise VerificationError(
            "This trip's driver identity was not resolved at entry, so it has no fallback "
            "until an administrator reconciles it. Face or fingerprint can still be used.",
            409, "reconciliation_required", verification_mode=None)
    if expected_mode and mode != expected_mode:
        if mode == MODE_SMS:
            raise VerificationError(
                "This vehicle entered with a registered driver. If face and fingerprint fail, "
                "the fallback is an SMS code to that driver — a passcode is not accepted.",
                409, "wrong_verification_mode", verification_mode=mode)
        raise VerificationError(
            "This vehicle entered with a guest driver. If face and fingerprint fail, the "
            "fallback is the passcode set at entry.", 409, "wrong_verification_mode",
            verification_mode=mode)
    if trip["verification_locked"]:
        raise VerificationError(
            "Too many failed code or passcode attempts for this trip. An administrator must "
            "review it. Face or fingerprint can still be used.", 423, "verification_locked")
    return trip


def _grant_authorization(conn, trip_id, method, staff, s, now, challenge_id=None) -> int:
    conn.execute("UPDATE trip_exit_authorizations SET status='REVOKED' WHERE trip_id=? "
                 "AND status='ACTIVE'", (trip_id,))
    return conn.execute(
        "INSERT INTO trip_exit_authorizations (trip_id, method, challenge_id, gate_session_id, "
        "status, created_at, expires_at) VALUES (?,?,?,?, 'ACTIVE', ?, ?)",
        (trip_id, method, challenge_id, staff.session_id, now,
         now + s.exit_authorization_ttl_seconds)).lastrowid


def _authorization_fits(method, trip) -> bool:
    """Biometric authorizations fit any trip; a fallback only its own mode."""
    return method in BIOMETRIC_METHODS or FALLBACK_METHOD.get(trip["verification_mode"]) == method


# ── PRIMARY: FACE / FINGERPRINT ──────────────────────────────────────────────

def record_biometric_result(trip_id, staff, method, match, score=None, exit_photo=None,
                            settings=None) -> bool:
    """
    Store a face/fingerprint comparison against THIS trip's own entry
    capture and, when it matched, authorize the exit for the gate session
    that ran the scan. Returns True when an authorization was created.
    """
    if method not in BIOMETRIC_METHODS:
        raise ValueError(f"not a biometric method: {method}")
    s = _settings(settings)
    now = _clock()
    authorized = stored = False
    replaced = None
    with db.transaction() as conn:
        trip = conn.execute("SELECT status, exit_face_photo_path, face_encoding, "
                            "fingerprint_template FROM trips WHERE id=?", (trip_id,)).fetchone()
        if trip is not None and trip["status"] == "INSIDE":
            stored = True
            replaced = trip["exit_face_photo_path"] if exit_photo else None
            conn.execute("""
                UPDATE trips SET exit_biometric_method=?, exit_biometric_match=?,
                    exit_biometric_score=?,
                    face_distance=CASE WHEN ?='face' THEN ? ELSE face_distance END,
                    exit_face_photo_path=COALESCE(?, exit_face_photo_path)
                WHERE id=?
            """, (method, None if match is None else int(bool(match)), score,
                  method, score, exit_photo, trip_id))
            # Only a comparison against a template this trip actually stored
            # can have matched it.
            template = trip["face_encoding"] if method == METHOD_FACE \
                else trip["fingerprint_template"]
            if match is True and template and staff is not None:
                _grant_authorization(conn, trip_id, method, staff, s, now)
                authorized = True
            db.log_security_event(conn, "exit_biometric_result", trip_id=trip_id,
                                  staff_user_id=getattr(staff, "user_id", None),
                                  session_id=getattr(staff, "session_id", None),
                                  detail={"method": method, "match": match,
                                          "authorized": authorized})
    if not stored:
        db.remove_gate_file(exit_photo)
    elif replaced and replaced != exit_photo:
        db.remove_gate_file(replaced)
    return authorized


# ── FALLBACK: SMS OTP (registered drivers) ───────────────────────────────────

def _sms_client(s):
    try:
        return robase.get_client(s)
    except vs.ConfigurationError as exc:
        log.error("SMS OTP is not configured: %s", exc)
        raise VerificationError("SMS OTP is not configured on this server. Use face or "
                                "fingerprint, or contact the administrator.",
                                503, "sms_not_configured") from None


def _send_error(error) -> VerificationError:
    kind = error.kind
    if kind == "rate_limited":
        wait = int(error.retry_after or 60)
        return VerificationError(f"The SMS provider is limiting codes to this number. Try "
                                 f"again in {wait} s.", 429, "sms_rate_limited",
                                 retry_after=wait)
    if kind == "insufficient_credits":
        return VerificationError("The SMS account has run out of credit, so no code was sent. "
                                 "Use face or fingerprint, or ask an administrator to top up "
                                 "the Robase balance.", 503, "sms_insufficient_credits")
    if kind == "unauthorized":
        return VerificationError("The SMS provider rejected this server's API key, so no code "
                                 "was sent. Contact the administrator.", 503, "sms_send_failed",
                                 provider_error=kind)
    if kind == "rejected":
        message = {
            "invalid_phone": "The SMS provider rejected the driver's phone number. Ask an "
                             "administrator to check the number on the holder record.",
            "country_not_supported": "The SMS provider cannot send to the driver's phone "
                                     "number's country.",
        }.get(error.error_type, "The SMS provider rejected the request, so no code was sent.")
        return VerificationError(message, 502, "sms_send_failed",
                                 provider_error=error.error_type or kind)
    return VerificationError("The SMS provider did not confirm the send, so no code is active "
                             "and nothing was authorized. If the driver receives a code it "
                             "will not work — wait for the countdown, then send a new code.",
                             504, "sms_send_uncertain", provider_error=kind)


def request_sms_otp(trip_id, staff, settings=None, client=None) -> dict:
    """Ask Robase to text a new code to the trip's registered driver.
    Also serves resend: a delivered code replaces the earlier ones."""
    s = _settings(settings)
    client = client or _sms_client(s)
    now = _clock()
    with db.transaction() as conn:
        trip = _open_trip(conn, trip_id, MODE_SMS)
        holder = _holder_row(conn, trip["holder_id"], trip["holder_uid"])
        if holder is None:
            raise VerificationError("The registered driver on this trip no longer matches a "
                                    "holder record. Use face or fingerprint, or ask an "
                                    "administrator to reconcile the trip.", 409, "holder_changed")
        phone = _sms_destination(holder, s)
        if phone is None:
            raise VerificationError("The registered driver's record has no valid phone number, "
                                    "so an SMS code cannot be sent. Use face or fingerprint, "
                                    "or ask an administrator to correct the number.",
                                    409, "phone_missing")

        last = conn.execute("SELECT created_at, retry_not_before FROM sms_otp_challenges "
                            "WHERE trip_id=? ORDER BY id DESC LIMIT 1", (trip_id,)).fetchone()
        if last:
            ready_at = max(last["created_at"] + s.otp_resend_cooldown_seconds,
                           last["retry_not_before"] or 0)
            if now < ready_at:
                wait = int(math.ceil(ready_at - now))
                raise VerificationError(f"Please wait {wait} s before sending another code.",
                                        429, "sms_cooldown", retry_after=wait)
        if trip["otp_send_count"] >= s.otp_max_sends_per_trip:
            raise VerificationError("The SMS code limit for this trip has been reached. Use "
                                    "face or fingerprint, or ask an administrator to review "
                                    "the trip.", 429, "sms_trip_send_limit")

        recent = conn.execute(
            "SELECT COUNT(*) AS sends, COALESCE(SUM(failed_attempts), 0) AS failed, "
            "MIN(created_at) AS oldest FROM sms_otp_challenges "
            "WHERE holder_uid=? AND created_at >= ?",
            (holder["holder_uid"], now - s.otp_holder_window_seconds)).fetchone()
        if recent["sends"] >= s.otp_max_sends_per_holder_window or \
                recent["failed"] >= s.otp_max_failed_per_holder_window:
            wait = int(math.ceil((recent["oldest"] or now) + s.otp_holder_window_seconds - now))
            raise VerificationError("Too many codes or failed attempts for this driver "
                                    "recently. Try again later or ask an administrator.",
                                    429, "sms_driver_limit", retry_after=max(wait, 1))

        idempotency_key = uuid.uuid4().hex
        challenge_id = conn.execute(
            "INSERT INTO sms_otp_challenges (trip_id, holder_id, holder_uid, phone_e164, "
            "gate_session_id, idempotency_key, status, max_attempts, created_at) "
            "VALUES (?,?,?,?,?,?, 'SENDING', ?, ?)",
            (trip_id, holder["id"], holder["holder_uid"], phone, staff.session_id,
             idempotency_key, s.otp_max_attempts, now)).lastrowid
        conn.execute("UPDATE trips SET otp_send_count = otp_send_count + 1 WHERE id=?",
                     (trip_id,))
        db.log_security_event(conn, "sms_otp_requested", trip_id=trip_id,
                              holder_id=holder["id"], staff_user_id=staff.user_id,
                              session_id=staff.session_id,
                              detail={"challenge_id": challenge_id})

    # The network call runs outside any transaction. The idempotency key
    # lets the client retry a timed-out send without a second SMS or charge.
    sent = error = None
    try:
        sent = client.send_otp(phone, code_length=s.otp_length, ttl_seconds=s.otp_ttl_seconds,
                               idempotency_key=idempotency_key,
                               language=s.sms_language or None,
                               metadata={"purpose": "gate_exit", "trip_id": str(trip_id)})
    except robase.RobaseError as exc:
        error = exc
    except Exception as exc:  # a client bug must not leave anything usable
        error = robase.RobaseError(f"Unexpected SMS client failure ({type(exc).__name__})",
                                   kind="malformed")

    done = _clock()
    with db.transaction() as conn:
        if error is None:
            row = conn.execute("SELECT c.status, t.status AS trip_status, t.verification_mode "
                               "FROM sms_otp_challenges c JOIN trips t ON t.id = c.trip_id "
                               "WHERE c.id=?", (challenge_id,)).fetchone()
            if not row or row["status"] != "SENDING" or row["trip_status"] != "INSIDE" or \
                    row["verification_mode"] != MODE_SMS:
                conn.execute("UPDATE sms_otp_challenges SET status='CANCELLED', "
                             "provider_otp_id=? WHERE id=?", (sent.otp_id, challenge_id))
                outcome = VerificationError("The trip changed while the code was being sent, "
                                            "so the code was cancelled.", 409, "sms_cancelled")
            else:
                expires_at = done + s.otp_ttl_seconds
                if sent.expires_at and done < sent.expires_at < expires_at:
                    expires_at = sent.expires_at       # never outlive Robase's own expiry
                # Only a code Robase accepted replaces earlier ones; unused
                # SMS authorizations from earlier codes are withdrawn too.
                conn.execute("UPDATE sms_otp_challenges SET status='SUPERSEDED' WHERE trip_id=? "
                             "AND id != ? AND status IN ('ACTIVE','SENDING')",
                             (trip_id, challenge_id))
                conn.execute("UPDATE trip_exit_authorizations SET status='REVOKED' "
                             "WHERE trip_id=? AND status='ACTIVE' AND method=?",
                             (trip_id, METHOD_SMS))
                conn.execute("UPDATE sms_otp_challenges SET status='ACTIVE', provider_otp_id=?, "
                             "sent_at=?, expires_at=?, delivery_status='queued' WHERE id=?",
                             (sent.otp_id, done, expires_at, challenge_id))
                db.log_security_event(conn, "sms_otp_sent", trip_id=trip_id,
                                      staff_user_id=staff.user_id, session_id=staff.session_id,
                                      detail={"challenge_id": challenge_id})
                outcome = {"sent": True, "expires_in": int(round(expires_at - done)),
                           "resend_available_in": s.otp_resend_cooldown_seconds,
                           "attempts_allowed": s.otp_max_attempts,
                           "code_length": s.otp_length,
                           "destination_hint": phone_utils.mask_phone(phone),
                           "delivery_status": "queued",
                           "mock_delivery": bool(getattr(client, "is_mock", False))}
        else:
            retry_not_before = done + error.retry_after if error.retry_after else None
            conn.execute("UPDATE sms_otp_challenges SET status=?, provider_error=?, "
                         "retry_not_before=? WHERE id=?",
                         ("UNCERTAIN" if error.uncertain else "FAILED",
                          f"{error.kind}:{error.error_type or ''}"[:120], retry_not_before,
                          challenge_id))
            db.log_security_event(conn, "sms_otp_send_failed", trip_id=trip_id,
                                  staff_user_id=staff.user_id, session_id=staff.session_id,
                                  detail={"challenge_id": challenge_id, "kind": error.kind,
                                          "error_type": error.error_type})
            outcome = _send_error(error)

    if error is not None:
        log.warning("SMS OTP send failed for trip %s: %s", trip_id, error.kind)
    if isinstance(outcome, VerificationError):
        raise outcome
    return outcome


def verify_sms_otp(trip_id, code, staff, settings=None, client=None) -> dict:
    """Check the driver's code with Robase against the trip's latest active
    challenge. A match authorizes the exit for this gate session."""
    s = _settings(settings)
    code = clean_code(code)
    if not (_is_digits(code) and len(code) == s.otp_length):
        raise VerificationError(f"Enter the {s.otp_length}-digit code from the SMS.",
                                400, "otp_format")
    client = client or _sms_client(s)

    # 1. Reserve an attempt atomically, so parallel requests cannot spend
    #    more than the attempt budget.
    with db.transaction() as conn:
        reserved = _reserve_attempt(conn, trip_id, staff, s, _clock())
    if isinstance(reserved, VerificationError):
        raise reserved
    challenge_id, otp_id = reserved

    # 2. Ask Robase, outside any transaction.
    result = error = None
    try:
        result = client.verify_otp(otp_id, code, idempotency_key=uuid.uuid4().hex)
    except robase.RobaseError as exc:
        error = exc
    except Exception as exc:
        error = robase.RobaseError(f"Unexpected SMS client failure ({type(exc).__name__})",
                                   kind="malformed")

    # 3. Record the outcome; consume the challenge only if it is still the
    #    trip's active one.
    with db.transaction() as conn:
        outcome = _finish_verification(conn, trip_id, challenge_id, staff, s, _clock(),
                                       result, error)
    if isinstance(outcome, VerificationError):
        raise outcome
    return outcome


def _reserve_attempt(conn, trip_id, staff, s, now):
    trip = _open_trip(conn, trip_id, MODE_SMS)
    challenge = conn.execute("SELECT * FROM sms_otp_challenges WHERE trip_id=? "
                             "AND status='ACTIVE' ORDER BY id DESC LIMIT 1",
                             (trip_id,)).fetchone()
    if challenge is None:
        raise VerificationError("There is no active SMS code for this trip. Send a new code.",
                                409, "sms_no_active_challenge")
    if challenge["gate_session_id"] != staff.session_id:
        db.log_security_event(conn, "sms_otp_cross_session_attempt", trip_id=trip_id,
                              staff_user_id=staff.user_id, session_id=staff.session_id)
        return VerificationError("This code was requested from a different gate session. "
                                 "Send a new code from this screen.", 403, "sms_other_session")
    holder = _holder_row(conn, trip["holder_id"], trip["holder_uid"])
    if challenge["holder_uid"] != trip["holder_uid"] or \
            _sms_destination(holder, s) != challenge["phone_e164"]:
        conn.execute("UPDATE sms_otp_challenges SET status='CANCELLED' WHERE id=?",
                     (challenge["id"],))
        return VerificationError("The driver's record changed after this code was sent. "
                                 "Send a new code.", 409, "sms_recipient_changed")
    if challenge["expires_at"] is None or now >= challenge["expires_at"]:
        conn.execute("UPDATE sms_otp_challenges SET status='EXPIRED' WHERE id=?",
                     (challenge["id"],))
        return VerificationError("This code has expired. Send a new code.", 410, "otp_expired")
    if challenge["attempts"] >= challenge["max_attempts"]:
        # Every attempt is spent or still being checked. The status is left
        # to those checks, so a correct code already in flight still counts.
        raise VerificationError("No attempts are left for this code. Send a new code.",
                                400, "otp_attempts_exhausted", attempts_remaining=0)
    conn.execute("UPDATE sms_otp_challenges SET attempts = attempts + 1 WHERE id=?",
                 (challenge["id"],))
    return challenge["id"], challenge["provider_otp_id"]


def _set_status(conn, challenge_id, status):
    conn.execute("UPDATE sms_otp_challenges SET status=? WHERE id=? AND status='ACTIVE'",
                 (status, challenge_id))


def _finish_verification(conn, trip_id, challenge_id, staff, s, now, result, error):
    challenge = conn.execute("SELECT * FROM sms_otp_challenges WHERE id=?",
                             (challenge_id,)).fetchone()
    trip = conn.execute("SELECT * FROM trips WHERE id=?", (trip_id,)).fetchone()

    def refund():
        # Robase did not check the code, so the reserved attempt is returned.
        conn.execute("UPDATE sms_otp_challenges SET attempts = MAX(attempts - 1, 0) "
                     "WHERE id=?", (challenge_id,))

    if error is not None:
        db.log_security_event(conn, "sms_otp_verify_error", trip_id=trip_id,
                              staff_user_id=staff.user_id, session_id=staff.session_id,
                              detail={"challenge_id": challenge_id, "kind": error.kind})
        log.warning("SMS OTP verification error for trip %s: %s", trip_id, error.kind)
        if error.kind == "expired":
            _set_status(conn, challenge_id, "EXPIRED")
            return VerificationError("This code has expired. Send a new code.", 410,
                                     "otp_expired")
        if error.kind == "attempts_exhausted":
            _set_status(conn, challenge_id, "EXHAUSTED")
            return VerificationError("No attempts are left for this code. Send a new code.",
                                     400, "otp_attempts_exhausted", attempts_remaining=0)
        if error.kind == "not_found":
            _set_status(conn, challenge_id, "CANCELLED")
        if error.kind in ("already_verified", "not_found"):
            # "Already verified" can also mean a parallel check of the same
            # code won; that request records the success, so this one only
            # reports failure and leaves the challenge alone.
            return VerificationError("The SMS provider no longer accepts this code, so nothing "
                                     "was authorized. Send a new code.", 409, "sms_code_unusable")
        if error.kind == "rate_limited":
            refund()
            wait = int(error.retry_after or 60)
            return VerificationError(f"The SMS provider is limiting checks for this number. "
                                     f"Try again in {wait} s.", 429, "sms_rate_limited",
                                     retry_after=wait)
        if error.kind in ("unauthorized", "insufficient_credits", "rejected"):
            refund()
            return VerificationError("The SMS provider could not check the code, so nothing "
                                     "was authorized. Try again, or use face or fingerprint.",
                                     503, "sms_verify_failed", provider_error=error.kind)
        # Timeout, network, 5xx or a malformed answer: the outcome is
        # unknown, so nothing is authorized and the attempt stays counted.
        return VerificationError("The SMS provider did not confirm the code, so nothing was "
                                 "authorized. Try verifying again, or send a new code.",
                                 504, "sms_verify_uncertain", provider_error=error.kind)

    if challenge is None or challenge["status"] != "ACTIVE":
        # Replaced by a newer code, cancelled or consumed while Robase answered.
        return VerificationError("This is no longer the trip's active code. Use the latest "
                                 "code or send a new one.", 409, "sms_code_superseded")
    if trip is None or trip["status"] != "INSIDE":
        _set_status(conn, challenge_id, "CANCELLED")
        return VerificationError("This trip is already closed.", 409, "trip_closed")

    if result.valid:
        if trip["verification_mode"] != MODE_SMS or trip["verification_locked"]:
            _set_status(conn, challenge_id, "CANCELLED")
            return VerificationError("The trip changed while the code was being checked, so "
                                     "nothing was authorized.", 409, "sms_cancelled")
        consumed = conn.execute("UPDATE sms_otp_challenges SET status='VERIFIED', verified_at=? "
                                "WHERE id=? AND status='ACTIVE'", (now, challenge_id)).rowcount
        if consumed != 1:
            return VerificationError("This is no longer the trip's active code.", 409,
                                     "sms_code_superseded")
        _grant_authorization(conn, trip_id, METHOD_SMS, staff, s, now, challenge_id=challenge_id)
        db.log_security_event(conn, "sms_otp_verified", trip_id=trip_id,
                              holder_id=trip["holder_id"], staff_user_id=staff.user_id,
                              session_id=staff.session_id,
                              detail={"challenge_id": challenge_id})
        return {"valid": True, "method": METHOD_SMS,
                "authorization_expires_in": s.exit_authorization_ttl_seconds}

    # A wrong code.
    failed = challenge["failed_attempts"] + 1
    exhausted = failed >= challenge["max_attempts"] or \
        result.attempts_remaining == 0 or result.status == "failed"
    conn.execute("UPDATE sms_otp_challenges SET failed_attempts=?, status=? WHERE id=?",
                 (failed, "EXHAUSTED" if exhausted else "ACTIVE", challenge_id))
    trip_failed = trip["otp_failed_attempts"] + 1
    locked = trip_failed >= s.otp_max_failed_per_trip
    conn.execute("UPDATE trips SET otp_failed_attempts=?, verification_locked=? WHERE id=?",
                 (trip_failed, 1 if locked else trip["verification_locked"], trip_id))
    db.log_security_event(conn, "sms_otp_failed", trip_id=trip_id,
                          staff_user_id=staff.user_id, session_id=staff.session_id,
                          detail={"challenge_id": challenge_id,
                                  "failed_attempts": failed, "trip_locked": locked})
    if locked:
        return VerificationError("Incorrect code. Too many failed attempts for this trip — an "
                                 "administrator must review it. Face or fingerprint can still "
                                 "be used.", 423, "verification_locked")
    if exhausted:
        return VerificationError("Incorrect code. No attempts are left for this code — send a "
                                 "new code.", 400, "otp_attempts_exhausted",
                                 attempts_remaining=0)
    return VerificationError("Incorrect code.", 400, "otp_incorrect",
                             attempts_remaining=challenge["max_attempts"] - failed)


def sms_delivery_status(trip_id, staff, settings=None, client=None) -> dict:
    """
    Delivery state of this session's active code, from GET /v1/otp/{id}.
    Informational only; a failed delivery makes the code unusable so the
    operator sends a new one. Never raises for provider problems.
    """
    s = _settings(settings)
    conn = db.get_connection()
    try:
        challenge = conn.execute("SELECT id, provider_otp_id, gate_session_id "
                                 "FROM sms_otp_challenges WHERE trip_id=? AND status='ACTIVE' "
                                 "ORDER BY id DESC LIMIT 1", (trip_id,)).fetchone()
    finally:
        conn.close()
    if challenge is None or challenge["gate_session_id"] != staff.session_id:
        return {"delivery_status": None}
    try:
        info = (client or robase.get_client(s)).get_otp(challenge["provider_otp_id"])
    except (vs.ConfigurationError, robase.RobaseError):
        return {"delivery_status": "unknown"}
    status = info.get("status")
    delivery = "delivered" if info.get("delivered") else {
        "pending": "queued", "sent": "sent", "verified": "verified",
        "expired": "expired", "failed": "failed"}.get(status, "unknown")
    with db.transaction() as conn:
        if status == "failed":
            conn.execute("UPDATE sms_otp_challenges SET status='FAILED', delivery_status='failed', "
                         "provider_error=? WHERE id=? AND status='ACTIVE'",
                         (f"delivery:{info.get('failure_reason') or 'failed'}"[:120],
                          challenge["id"]))
        else:
            conn.execute("UPDATE sms_otp_challenges SET delivery_status=? WHERE id=?",
                         (delivery, challenge["id"]))
    return {"delivery_status": delivery}


# ── FALLBACK: GUEST PASSCODE ─────────────────────────────────────────────────

def verify_passcode(trip_id, passcode, staff, settings=None) -> dict:
    s = _settings(settings)
    code = validate_passcode(passcode, s)
    now = _clock()
    with db.transaction() as conn:
        outcome = _verify_passcode_tx(conn, trip_id, code, staff, s, now)
    if isinstance(outcome, VerificationError):
        raise outcome
    return outcome


def _verify_passcode_tx(conn, trip_id, code, staff, s, now):
    trip = _open_trip(conn, trip_id, MODE_PASSCODE)
    if not trip["passcode_hash"]:
        raise VerificationError("No passcode is stored for this trip. An administrator must "
                                "reconcile it.", 409, "passcode_missing")
    if not check_password_hash(trip["passcode_hash"], code):
        failed = trip["passcode_failed_attempts"] + 1
        locked = failed >= s.passcode_max_attempts
        conn.execute("UPDATE trips SET passcode_failed_attempts=?, verification_locked=? "
                     "WHERE id=?", (failed, 1 if locked else trip["verification_locked"], trip_id))
        db.log_security_event(conn, "passcode_failed", trip_id=trip_id,
                              staff_user_id=staff.user_id, session_id=staff.session_id,
                              detail={"attempt": failed, "trip_locked": locked})
        if locked:
            return VerificationError("Incorrect passcode. Too many failed attempts — an "
                                     "administrator must review this trip. Face or "
                                     "fingerprint can still be used.", 423,
                                     "verification_locked")
        return VerificationError("Incorrect passcode.", 400, "passcode_incorrect",
                                 attempts_remaining=s.passcode_max_attempts - failed)

    _grant_authorization(conn, trip_id, METHOD_PASSCODE, staff, s, now)
    db.log_security_event(conn, "passcode_verified", trip_id=trip_id,
                          staff_user_id=staff.user_id, session_id=staff.session_id)
    return {"valid": True, "method": METHOD_PASSCODE,
            "authorization_expires_in": s.exit_authorization_ttl_seconds}


# ── EXIT ─────────────────────────────────────────────────────────────────────

def confirm_exit(trip_id, staff, extra_files=()) -> dict:
    """Consume this session's authorization and close the trip, atomically."""
    now = _clock()
    with db.transaction() as conn:
        trip = conn.execute("SELECT * FROM trips WHERE id=?", (trip_id,)).fetchone()
        if trip is None:
            raise VerificationError("Trip not found.", 404, "trip_not_found")
        if trip["status"] != "INSIDE":
            raise VerificationError("This trip is already closed.", 409, "trip_closed")
        auth = conn.execute("SELECT * FROM trip_exit_authorizations WHERE trip_id=? "
                            "AND status='ACTIVE'", (trip_id,)).fetchone()
        if auth is None or auth["gate_session_id"] != staff.session_id or \
                auth["expires_at"] <= now or not _authorization_fits(auth["method"], trip):
            raise VerificationError(
                "No verified exit authorization for this trip from this gate session. Verify "
                "the driver by face or fingerprint, or use the trip's fallback first.",
                403, "exit_not_authorized")

        consumed = conn.execute("UPDATE trip_exit_authorizations SET status='CONSUMED', "
                                "consumed_at=? WHERE id=? AND status='ACTIVE'",
                                (now, auth["id"])).rowcount
        # Biometrics, photos and the passcode hash are cleared on close;
        # vehicle attributes and the verification record are kept.
        closed = conn.execute("""
            UPDATE trips SET status='EXITED', exit_time=datetime('now','localtime'),
                exit_result='GRANTED', exit_verification_method=?, exit_authorization_id=?,
                exit_staff_user_id=?, exit_face_photo_path=NULL, face_encoding=NULL,
                fingerprint_template=NULL, vehicle_photo_path=NULL, face_photo_path=NULL,
                passcode_hash=NULL
            WHERE id=? AND status='INSIDE'
        """, (auth["method"], auth["id"], staff.user_id, trip_id)).rowcount
        if consumed != 1 or closed != 1:
            raise VerificationError("This trip was closed by another request.", 409,
                                    "exit_conflict")
        db.log_security_event(conn, "exit_granted", trip_id=trip_id, holder_id=trip["holder_id"],
                              staff_user_id=staff.user_id, session_id=staff.session_id,
                              detail={"method": auth["method"], "authorization_id": auth["id"]})
        paths = [trip["exit_face_photo_path"], trip["vehicle_photo_path"], trip["face_photo_path"]]

    db.remove_trip_files(trip_id, paths + list(extra_files))
    return {"ok": True, "decision": "GRANTED", "method": auth["method"]}


def trip_verification_view(trip: dict, staff, settings=None) -> dict:
    """What the exit page needs. Never includes a code, a hash, a provider
    id or the full phone number."""
    s = _settings(settings)
    now = _clock()
    mode = trip.get("verification_mode")
    view = {"mode": mode if mode in MODES else None,
            "identity_status": trip.get("identity_status"),
            "locked": bool(trip.get("verification_locked")),
            "reconciliation_required": mode not in MODES,
            "biometrics": {"face": bool(trip.get("face_encoding")),
                           "fingerprint": bool(trip.get("fingerprint_template"))},
            "authorized": False, "authorized_method": None,
            "mock_delivery": s.mock_delivery}
    conn = db.get_connection()
    try:
        auth = conn.execute("SELECT gate_session_id, expires_at, method "
                            "FROM trip_exit_authorizations WHERE trip_id=? AND status='ACTIVE'",
                            (trip["id"],)).fetchone()
        if auth and auth["gate_session_id"] == staff.session_id and auth["expires_at"] > now \
                and _authorization_fits(auth["method"], trip):
            view.update(authorized=True, authorized_method=auth["method"])
        if mode == MODE_SMS:
            holder = _holder_row(conn, trip.get("holder_id"), trip.get("holder_uid"))
            phone = _sms_destination(holder, s)
            challenge = conn.execute("SELECT * FROM sms_otp_challenges WHERE trip_id=? "
                                     "ORDER BY id DESC LIMIT 1", (trip["id"],)).fetchone()
            sms = {"available": phone is not None, "code_length": s.otp_length,
                   "destination_hint": phone_utils.mask_phone(phone) if phone else None,
                   "active": False, "expires_in": None, "attempts_remaining": None,
                   "resend_available_in": 0, "delivery_status": None,
                   "sends_remaining": max(0, s.otp_max_sends_per_trip
                                          - (trip.get("otp_send_count") or 0))}
            if challenge:
                ready_at = max(challenge["created_at"] + s.otp_resend_cooldown_seconds,
                               challenge["retry_not_before"] or 0)
                sms["resend_available_in"] = max(0, int(math.ceil(ready_at - now)))
                if challenge["status"] == "ACTIVE" and challenge["expires_at"] and \
                        challenge["expires_at"] > now and \
                        challenge["gate_session_id"] == staff.session_id:
                    sms.update(active=True, expires_in=int(challenge["expires_at"] - now),
                               attempts_remaining=challenge["max_attempts"]
                               - challenge["attempts"],
                               delivery_status=challenge["delivery_status"])
            view["driver_name"] = _name(holder) if holder else None
            view["sms"] = sms
        elif mode == MODE_PASSCODE:
            view["passcode"] = {
                "min_length": s.passcode_min_length, "max_length": s.passcode_max_length,
                "attempts_remaining": max(0, s.passcode_max_attempts
                                          - (trip.get("passcode_failed_attempts") or 0))}
    finally:
        conn.close()
    return view


# ── ADMINISTRATOR RECONCILIATION ─────────────────────────────────────────────

def reconcile_trip(trip_id, action, staff, holder_id=None, note="", settings=None) -> str:
    """
    assign_holder  unresolved/legacy trip -> registered holder (SMS fallback)
    confirm_guest  unresolved/legacy trip with a stored passcode -> guest
    unlock         clear attempt counters and the lock after a review
    """
    s = _settings(settings)
    if staff is None or not staff.is_admin:
        raise VerificationError("An administrator account is required.", 403, "admin_required")
    if action not in RECONCILE_ACTIONS:
        raise VerificationError("Unknown reconciliation action.", 400, "bad_action")
    note = (note or "").strip()[:500]
    if not note:
        raise VerificationError("Record a note explaining the decision.", 400, "note_required")

    now = _clock()
    with db.transaction() as conn:
        trip = conn.execute("SELECT * FROM trips WHERE id=?", (trip_id,)).fetchone()
        if trip is None:
            raise VerificationError("Trip not found.", 404, "trip_not_found")
        if trip["status"] != "INSIDE":
            raise VerificationError("This trip is already closed.", 409, "trip_closed")
        resolved = trip["verification_mode"] in MODES
        stamp = f"[{staff.username}] {action}: {note}"
        full_note = f"{trip['reconciliation_note']}; {stamp}" if trip["reconciliation_note"] \
            else stamp

        if action in ("assign_holder", "confirm_guest") and resolved:
            raise VerificationError("This trip's identity is already resolved; only unlocking "
                                    "is possible.", 409, "already_resolved")

        if action == "assign_holder":
            holder = conn.execute("SELECT id, holder_uid, surname, first_name, phone_number, "
                                  "phone_e164 FROM holders WHERE id=?", (holder_id,)).fetchone()
            if holder is None:
                raise VerificationError("Holder not found.", 404, "holder_not_found")
            if _sms_destination(holder, s) is None:
                raise VerificationError(f"{_name(holder)} has no valid phone number, so the SMS "
                                        "fallback could not reach them. Correct the number "
                                        "first.", 409, "phone_missing")
            conn.execute("UPDATE trips SET holder_id=?, holder_uid=?, identity_status=?, "
                         "identity_method='staff_reconciliation', verification_mode=?, "
                         "reconciled_by=?, reconciled_at=?, reconciliation_note=? WHERE id=?",
                         (holder["id"], holder["holder_uid"], REGISTERED, MODE_SMS,
                          staff.user_id, now, full_note, trip_id))
        elif action == "confirm_guest":
            if not trip["passcode_hash"]:
                raise VerificationError("No passcode is stored for this trip, so it cannot be "
                                        "confirmed as a guest. Assign a registered holder "
                                        "instead.", 409, "passcode_missing")
            conn.execute("UPDATE trips SET identity_status=?, "
                         "identity_method='staff_reconciliation', verification_mode=?, "
                         "reconciled_by=?, reconciled_at=?, reconciliation_note=? WHERE id=?",
                         (GUEST, MODE_PASSCODE, staff.user_id, now, full_note, trip_id))
        else:
            conn.execute("UPDATE trips SET verification_locked=0, passcode_failed_attempts=0, "
                         "otp_failed_attempts=0, otp_send_count=0, reconciled_by=?, "
                         "reconciled_at=?, reconciliation_note=? WHERE id=?",
                         (staff.user_id, now, full_note, trip_id))

        conn.execute("UPDATE sms_otp_challenges SET status='CANCELLED' WHERE trip_id=? "
                     "AND status IN ('ACTIVE','SENDING')", (trip_id,))
        conn.execute("UPDATE trip_exit_authorizations SET status='REVOKED' WHERE trip_id=? "
                     "AND status='ACTIVE'", (trip_id,))
        db.log_security_event(conn, "trip_reconciled", trip_id=trip_id,
                              holder_id=holder_id if action == "assign_holder" else None,
                              staff_user_id=staff.user_id, session_id=staff.session_id,
                              detail={"action": action})
    return action


def trips_needing_review() -> list:
    conn = db.get_connection()
    try:
        return [dict(r) for r in conn.execute("""
            SELECT id, plate_number, entry_time, identity_status, verification_mode,
                   verification_locked, passcode_hash IS NOT NULL AS has_passcode,
                   reconciliation_note, passcode_failed_attempts, otp_failed_attempts,
                   otp_send_count
            FROM trips
            WHERE status='INSIDE' AND (verification_mode IS NULL OR verification_locked=1)
            ORDER BY entry_time
        """)]
    finally:
        conn.close()


def recent_security_events(limit=50) -> list:
    conn = db.get_connection()
    try:
        return [dict(r) for r in conn.execute(
            "SELECT e.created_at, e.event, e.trip_id, e.holder_id, u.username "
            "FROM security_events e LEFT JOIN staff_users u ON u.id = e.staff_user_id "
            "ORDER BY e.id DESC LIMIT ?", (limit,))]
    finally:
        conn.close()
