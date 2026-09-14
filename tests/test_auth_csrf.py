"""Staff sign-in, server-side gate sessions, CSRF and administrator-only routes."""
import re

import pytest

import app as application
import auth
from gate_helpers import PASSWORD, create_staff, login, rows, staff_client


@pytest.fixture()
def flask_app(gate_db):
    application.app.config["TESTING"] = True
    return application.app


def _login_attempt(client, username, password):
    page = client.get("/login")
    token = re.search(rb'name="csrf_token" value="([^"]+)"', page.data).group(1).decode()
    return client.post("/login", data={"username": username, "password": password,
                                       "csrf_token": token})


def test_the_hardcoded_secret_key_is_gone(flask_app):
    assert flask_app.secret_key != "license_admin_secret_2024"
    assert len(flask_app.secret_key) >= 32


@pytest.mark.parametrize("path", ["/", "/holders", "/holders/1", "/gate/entry", "/gate/exit",
                                  "/gate/trips", "/gate/exit/lookup?plate=X",
                                  "/admin/verification", "/gate/photo/x.jpg", "/photos/x.jpg"])
def test_every_page_requires_sign_in(flask_app, path):
    client = flask_app.test_client()
    assert client.get(path).status_code == 401
    browser = client.get(path, headers={"Accept": "text/html"})
    assert browser.status_code == 302 and "/login" in browser.headers["Location"]


@pytest.mark.parametrize("path", ["/gate/entry/confirm", "/gate/exit/confirm",
                                  "/gate/exit/sms/send", "/gate/exit/sms/resend",
                                  "/gate/exit/sms/verify", "/gate/exit/passcode/verify",
                                  "/gate/exit/verify/start", "/gate/exit/fp/start",
                                  "/holders/1/delete", "/holders/1/edit"])
def test_gate_actions_require_sign_in(flask_app, path):
    assert flask_app.test_client().post(path, json={"trip_id": 1}).status_code == 401


def test_repeated_failed_logins_lock_the_account(flask_app):
    create_staff("op", "operator")
    client = flask_app.test_client()
    for _ in range(auth.MAX_FAILED_LOGINS):
        assert _login_attempt(client, "op", "not the password").status_code == 401
    # Locked: even the right password is refused for now.
    assert _login_attempt(client, "op", PASSWORD).status_code == 401
    assert _login_attempt(client, "nobody", PASSWORD).status_code == 401


def test_login_without_a_csrf_token_is_refused(flask_app):
    create_staff("op", "operator")
    client = flask_app.test_client()
    client.get("/login")
    assert client.post("/login", data={"username": "op", "password": PASSWORD}).status_code == 400


def test_state_changing_requests_need_the_csrf_token(flask_app):
    client = staff_client(flask_app)
    del client.environ_base["HTTP_X_CSRF_TOKEN"]
    response = client.post("/gate/exit/confirm", json={"trip_id": 1})
    assert response.status_code == 400
    assert "token" in response.get_json()["error"].lower()


def test_a_csrf_token_from_another_session_is_refused(flask_app):
    alice = staff_client(flask_app, "alice", "operator")
    bob = staff_client(flask_app, "bob", "operator")
    alice.environ_base["HTTP_X_CSRF_TOKEN"] = bob.environ_base["HTTP_X_CSRF_TOKEN"]
    assert alice.post("/gate/exit/confirm", json={"trip_id": 1}).status_code == 400


def test_sign_in_rotates_the_session_and_logout_revokes_it(flask_app):
    create_staff("op", "operator")
    client = flask_app.test_client()
    client.get("/login")
    with client.session_transaction() as sess:
        pre_login_csrf = sess.get("csrf")
    login(client, "op")
    with client.session_transaction() as sess:
        assert sess["csrf"] != pre_login_csrf
        token = sess["sid"]
    assert auth.load_session(token) is not None

    cookie = client.get_cookie("session").value
    assert client.post("/logout").status_code == 302
    assert auth.load_session(token) is None
    replay = flask_app.test_client()
    replay.set_cookie("session", cookie)
    assert replay.get("/gate/entry").status_code == 401


def test_only_a_hash_of_the_session_token_is_stored(flask_app):
    client = staff_client(flask_app)
    with client.session_transaction() as sess:
        token = sess["sid"]
    stored = [r["token_hash"] for r in rows("SELECT token_hash FROM staff_sessions")]
    assert len(stored) == 1 and token not in stored


def test_operators_cannot_use_administrator_routes(flask_app):
    client = staff_client(flask_app, "op", "operator")
    assert client.get("/admin/verification").status_code == 403
    assert client.post("/admin/trips/1/reconcile",
                       data={"action": "unlock", "note": "x"}).status_code == 403
    assert client.post("/gate/trips/clear-all").status_code == 403
    assert client.post("/gate/trips/1/delete").status_code == 403


def test_administrators_can_open_the_review_page(flask_app):
    assert staff_client(flask_app, "boss", "admin").get("/admin/verification").status_code == 200


def test_the_login_page_and_flow_are_unchanged(flask_app):
    page = flask_app.test_client().get("/login").get_data(as_text=True)
    for fragment in ("SecureDrive", "Staff sign-in", 'name="username"', 'name="password"',
                     'name="csrf_token"', "Sign in"):
        assert fragment in page
    client = staff_client(flask_app, "op", "operator")
    assert client.get("/gate/exit").status_code == 200


def test_the_retired_telegram_admin_routes_are_gone(flask_app):
    admin = staff_client(flask_app, "boss", "admin")
    for path in ("/admin/enrollment/1/approve", "/admin/enrollment/1/reject",
                 "/holders/1/telegram/revoke"):
        assert admin.post(path, data={"note": "x"}).status_code == 404


def test_disabling_an_account_ends_its_sessions(flask_app):
    client = staff_client(flask_app, "op", "operator")
    auth.set_active("op", False)
    assert client.get("/gate/entry").status_code == 401


def test_post_login_redirects_stay_on_this_site():
    assert auth._safe_next("/gate/exit?plate=A") == "/gate/exit?plate=A"
    for target in ("https://evil.example/", "//evil.example/x", "/\\evil.example",
                   "javascript:alert(1)", None, ""):
        assert auth._safe_next(target) is None


@pytest.mark.parametrize("role", ["operator", "admin"])
def test_every_page_renders_for_signed_in_staff(flask_app, role):
    from gate_helpers import add_holder, guest_trip, unresolved_trip
    holder_id = add_holder()
    guest_trip()
    unresolved_trip()
    client = staff_client(flask_app, f"user-{role}", role)
    pages = ["/", "/holders", f"/holders/{holder_id}", f"/holders/{holder_id}/edit",
             "/holders/add", "/gate/entry", "/gate/exit", "/gate/trips", "/reports",
             "/face-search", "/fingerprint-search"]
    for path in pages:
        response = client.get(path)
        assert response.status_code == 200, path
        assert b'name="csrf-token"' in response.data, path
    trips = client.get("/gate/trips").get_data(as_text=True)
    assert "Guest passcode" in trips and "Unresolved" in trips
    assert ("Clear All" in trips) == (role == "admin")


def test_weak_passwords_are_refused(gate_db):
    with pytest.raises(ValueError):
        auth.create_user("op", "short", "operator")
    with pytest.raises(ValueError):
        auth.create_user("op", PASSWORD, "superuser")
