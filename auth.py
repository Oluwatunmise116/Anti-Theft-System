"""
Staff authentication, server-side gate sessions and CSRF protection.

Every route except /login needs a signed-in staff member. The browser's
signed cookie holds only a random session token and a CSRF token; the
session itself lives in the staff_sessions table, so it
  * survives an application restart and is shared by every worker,
  * can be revoked (logout, disabled account, password reset), and
  * gives OTP challenges and exit authorizations a stable gate session to
    bind to.

State-changing requests (anything but GET/HEAD/OPTIONS) must carry the CSRF
token, either as the X-CSRF-Token header (fetch) or as a csrf_token form
field. base.html adds both automatically.

Roles: 'operator' runs the gate. 'admin' can also change drivers' phone
numbers (SMS exit codes go to them), reconcile unresolved or locked trips
and delete trip records.
"""
import functools
import hashlib
import hmac
import logging
import os
import secrets
import time
from dataclasses import dataclass
from datetime import timedelta
from urllib.parse import urlparse

from flask import (Blueprint, flash, g, jsonify, redirect, render_template,
                   request, session, url_for)
from werkzeug.security import check_password_hash, generate_password_hash

import database as db
import verification_settings as vs

log = logging.getLogger(__name__)
bp = Blueprint("auth", __name__)

ROLES = ("admin", "operator")
MIN_PASSWORD_LENGTH = 10
MAX_FAILED_LOGINS = 5
LOCKOUT_SECONDS = 15 * 60
SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})
PUBLIC_ENDPOINTS = frozenset({"auth.login", "static"})
_SESSION_TOUCH_INTERVAL = 60

# Checked when the username is unknown, so a failed login takes about as
# long whether or not the account exists.
_DUMMY_HASH = generate_password_hash(secrets.token_hex(16))


@dataclass(frozen=True)
class StaffContext:
    session_id: int
    user_id: int
    username: str
    role: str

    @property
    def is_admin(self) -> bool:
        return self.role == "admin"


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


# ── ACCOUNTS ─────────────────────────────────────────────────────────────────

def validate_password(password):
    if len(password or "") < MIN_PASSWORD_LENGTH:
        raise ValueError(f"Password must be at least {MIN_PASSWORD_LENGTH} characters.")


def create_user(username, password, role="operator"):
    username = (username or "").strip()
    if not username or len(username) > 64 or \
            not all(ch.isalnum() or ch in "._-@" for ch in username):
        raise ValueError("Username must be 1-64 characters: letters, digits, '.', '_', '-', '@'.")
    if role not in ROLES:
        raise ValueError(f"Role must be one of: {', '.join(ROLES)}.")
    validate_password(password)
    with db.transaction() as conn:
        if conn.execute("SELECT 1 FROM staff_users WHERE username=?", (username,)).fetchone():
            raise ValueError(f"User {username!r} already exists.")
        cur = conn.execute(
            "INSERT INTO staff_users (username, password_hash, role, created_at) "
            "VALUES (?,?,?,?)", (username, generate_password_hash(password), role, time.time()))
        db.log_security_event(conn, "staff_user_created", staff_user_id=cur.lastrowid,
                              detail={"role": role})
        return cur.lastrowid


def set_password(username, password):
    validate_password(password)
    with db.transaction() as conn:
        user = conn.execute("SELECT id FROM staff_users WHERE username=?", (username,)).fetchone()
        if not user:
            raise ValueError(f"No user named {username!r}.")
        conn.execute("UPDATE staff_users SET password_hash=?, failed_logins=0, locked_until=NULL "
                     "WHERE id=?", (generate_password_hash(password), user["id"]))
        _revoke_user_sessions(conn, user["id"])
        db.log_security_event(conn, "staff_password_reset", staff_user_id=user["id"])


def set_active(username, active: bool):
    with db.transaction() as conn:
        user = conn.execute("SELECT id FROM staff_users WHERE username=?", (username,)).fetchone()
        if not user:
            raise ValueError(f"No user named {username!r}.")
        conn.execute("UPDATE staff_users SET active=? WHERE id=?", (int(bool(active)), user["id"]))
        if not active:
            _revoke_user_sessions(conn, user["id"])
        db.log_security_event(conn, "staff_user_enabled" if active else "staff_user_disabled",
                              staff_user_id=user["id"])


def list_users():
    conn = db.get_connection()
    try:
        return [dict(r) for r in conn.execute(
            "SELECT id, username, role, active, created_at, last_login_at "
            "FROM staff_users ORDER BY username")]
    finally:
        conn.close()


def count_users() -> int:
    conn = db.get_connection()
    try:
        return conn.execute("SELECT COUNT(*) FROM staff_users").fetchone()[0]
    finally:
        conn.close()


def authenticate(username, password, now=None):
    """The user row on success, else None. Locks an account for
    LOCKOUT_SECONDS after MAX_FAILED_LOGINS consecutive failures."""
    now = now or time.time()
    username = (username or "").strip()
    password = password or ""
    with db.transaction() as conn:
        user = conn.execute("SELECT * FROM staff_users WHERE username=?", (username,)).fetchone()
        if not user or not user["active"]:
            check_password_hash(_DUMMY_HASH, password)
            db.log_security_event(conn, "login_failed", detail={"reason": "unknown_or_disabled"})
            return None
        if user["locked_until"] and user["locked_until"] > now:
            db.log_security_event(conn, "login_rejected_locked", staff_user_id=user["id"])
            return None
        if not check_password_hash(user["password_hash"], password):
            failed = user["failed_logins"] + 1
            locked_until = now + LOCKOUT_SECONDS if failed >= MAX_FAILED_LOGINS else None
            conn.execute("UPDATE staff_users SET failed_logins=?, locked_until=? WHERE id=?",
                         (0 if locked_until else failed, locked_until, user["id"]))
            db.log_security_event(conn, "login_failed", staff_user_id=user["id"],
                                  detail={"locked": bool(locked_until)})
            return None
        conn.execute("UPDATE staff_users SET failed_logins=0, locked_until=NULL, last_login_at=? "
                     "WHERE id=?", (now, user["id"]))
        db.log_security_event(conn, "login_succeeded", staff_user_id=user["id"])
        return dict(user)


# ── SESSIONS ─────────────────────────────────────────────────────────────────

def start_session(user_id, user_agent="", now=None) -> str:
    token = secrets.token_urlsafe(32)
    now = now or time.time()
    hours = vs.load().staff_session_hours
    with db.transaction() as conn:
        cur = conn.execute(
            "INSERT INTO staff_sessions (token_hash, user_id, created_at, expires_at, "
            "last_seen_at, user_agent) VALUES (?,?,?,?,?,?)",
            (_token_hash(token), user_id, now, now + hours * 3600, now, (user_agent or "")[:200]))
        db.log_security_event(conn, "session_started", staff_user_id=user_id,
                              session_id=cur.lastrowid)
    return token


def load_session(token, now=None):
    if not token or not isinstance(token, str):
        return None
    now = now or time.time()
    conn = db.get_connection()
    try:
        row = conn.execute("""
            SELECT s.id AS session_id, s.last_seen_at, u.id AS user_id, u.username, u.role
            FROM staff_sessions s JOIN staff_users u ON u.id = s.user_id
            WHERE s.token_hash=? AND s.revoked_at IS NULL AND s.expires_at > ? AND u.active = 1
        """, (_token_hash(token), now)).fetchone()
        if row and (row["last_seen_at"] or 0) < now - _SESSION_TOUCH_INTERVAL:
            conn.execute("UPDATE staff_sessions SET last_seen_at=? WHERE id=?",
                         (now, row["session_id"]))
            conn.commit()
    finally:
        conn.close()
    if not row:
        return None
    return StaffContext(session_id=row["session_id"], user_id=row["user_id"],
                        username=row["username"], role=row["role"])


def revoke_session(session_id):
    with db.transaction() as conn:
        conn.execute("UPDATE staff_sessions SET revoked_at=? WHERE id=? AND revoked_at IS NULL",
                     (time.time(), session_id))


def _revoke_user_sessions(conn, user_id):
    conn.execute("UPDATE staff_sessions SET revoked_at=? WHERE user_id=? AND revoked_at IS NULL",
                 (time.time(), user_id))


# ── REQUEST HOOKS ────────────────────────────────────────────────────────────

def csrf_token() -> str:
    token = session.get("csrf")
    if not token:
        token = secrets.token_urlsafe(32)
        session["csrf"] = token
    return token


def _csrf_ok() -> bool:
    expected = session.get("csrf")
    supplied = request.headers.get("X-CSRF-Token") or request.form.get("csrf_token") or ""
    return bool(expected) and hmac.compare_digest(str(expected), str(supplied))


def _wants_html() -> bool:
    return not request.is_json and "text/html" in request.headers.get("Accept", "")


def _deny(status, message):
    if _wants_html():
        if status == 401:
            target = request.full_path if request.method in SAFE_METHODS else None
            return redirect(url_for("auth.login", next=_safe_next(target)))
        return message, status
    return jsonify({"error": message}), status


def _before_request():
    g.staff = None
    endpoint = request.endpoint
    if endpoint is None or endpoint in PUBLIC_ENDPOINTS:
        if endpoint is not None and request.method not in SAFE_METHODS and not _csrf_ok():
            return _deny(400, "The form expired — reload the page and try again.")
        return None

    staff = load_session(session.get("sid"))
    if staff is None:
        session.pop("sid", None)
        return _deny(401, "Sign in required.")
    if request.method not in SAFE_METHODS and not _csrf_ok():
        return _deny(400, "Security token missing or invalid — reload the page and try again.")
    g.staff = staff
    return None


def current_staff():
    return g.get("staff")


def admin_required(view):
    @functools.wraps(view)
    def wrapper(*args, **kwargs):
        staff = current_staff()
        if staff is None or not staff.is_admin:
            if _wants_html():
                flash("That action needs an administrator account.", "error")
                return redirect(url_for("dashboard"))
            return jsonify({"error": "An administrator account is required."}), 403
        return view(*args, **kwargs)
    return wrapper


def _safe_next(target):
    """Only same-site relative paths are allowed as a post-login redirect."""
    if not target or not target.startswith("/") or target.startswith("//") or "\\" in target:
        return None
    parsed = urlparse(target)
    if parsed.scheme or parsed.netloc:
        return None
    return target.rstrip("?") or None


@bp.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        user = authenticate(request.form.get("username"), request.form.get("password"))
        if user is None:
            flash("Invalid username or password, or the account is temporarily locked.", "error")
            return render_template("login.html", no_users=count_users() == 0), 401
        token = start_session(user["id"], request.user_agent.string or "")
        next_url = _safe_next(request.args.get("next"))
        # A new session and CSRF token on every sign-in (no fixation).
        session.clear()
        session.permanent = True
        session["sid"] = token
        session["csrf"] = secrets.token_urlsafe(32)
        return redirect(next_url or url_for("dashboard"))

    if load_session(session.get("sid")):
        return redirect(url_for("dashboard"))
    return render_template("login.html", no_users=count_users() == 0)


@bp.route("/logout", methods=["POST"])
def logout():
    staff = current_staff()
    if staff:
        revoke_session(staff.session_id)
    session.clear()
    flash("Signed out.", "success")
    return redirect(url_for("auth.login"))


def _secret_key(settings) -> str:
    try:
        return vs.flask_secret_key()
    except vs.ConfigurationError:
        if not settings.is_development:
            raise
        log.warning("FLASK_SECRET_KEY is not set — using a temporary key. Sessions will "
                    "not survive a restart. Set it in .env for anything but development.")
        return secrets.token_hex(32)


def init_app(app):
    settings = vs.load()
    app.secret_key = _secret_key(settings)
    app.config.update(
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
        # Enable when the app is served over HTTPS (reverse proxy / TLS).
        SESSION_COOKIE_SECURE=os.environ.get("SESSION_COOKIE_SECURE", "").strip().lower()
        in ("1", "true", "yes"),
        PERMANENT_SESSION_LIFETIME=timedelta(hours=settings.staff_session_hours),
    )
    app.register_blueprint(bp)
    app.before_request(_before_request)
    app.jinja_env.globals["csrf_token"] = csrf_token

    @app.context_processor
    def _inject_staff():
        return {"current_staff": g.get("staff")}
