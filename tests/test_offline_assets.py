"""
Pages must load completely from the Pi alone.

Devices on the Secure_Drive hotspot have no internet, so any script,
stylesheet or font fetched from the internet makes the browser stall until it
times out (21 s per page on Windows) before anything is drawn.
"""
import glob
import os
import re

import pytest

import app as application
import face_manager as fm
from gate_helpers import staff_client

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXTERNAL = re.compile(r"""<(?:script|link|img|iframe)\b[^>]*\b(?:src|href)\s*=\s*["']\s*(?:https?:)?//""", re.I)


@pytest.mark.parametrize("path", sorted(glob.glob(os.path.join(ROOT, "templates", "*.html"))),
                         ids=os.path.basename)
def test_no_template_loads_anything_from_the_internet(path):
    source = open(path, encoding="utf-8").read()
    assert not EXTERNAL.search(source), EXTERNAL.search(source).group(0)


def test_the_font_stylesheet_only_points_at_local_files():
    css = open(os.path.join(ROOT, "static", "fonts", "fonts.css"), encoding="utf-8").read()
    urls = re.findall(r"url\(([^)]+)\)", css)
    assert len(urls) == css.count("@font-face") == 16
    for url in urls:
        assert url.startswith("files/") and os.path.isfile(os.path.join(ROOT, "static", "fonts", url))


@pytest.fixture()
def admin(gate_db, monkeypatch):
    application.app.config["TESTING"] = True
    monkeypatch.setattr(fm, "list_cameras", lambda: [])
    return staff_client(application.app, "admin-assets", "admin")


def test_pages_use_the_local_fonts_and_only_the_dashboard_loads_chartjs(admin):
    login = application.app.test_client().get("/login").get_data(as_text=True)
    assert "/static/fonts/fonts.css" in login
    dashboard = admin.get("/").get_data(as_text=True)
    assert "/static/fonts/fonts.css" in dashboard
    assert "/static/vendor/chart.js-4.4.0/chart.umd.js" in dashboard
    assert dashboard.index("chart.umd.js") < dashboard.index("new Chart(")
    for path in ("/settings", "/gate/entry", "/gate/exit"):
        assert "chart.umd.js" not in admin.get(path).get_data(as_text=True)


@pytest.mark.parametrize("asset", ["fonts/fonts.css", "vendor/chart.js-4.4.0/chart.umd.js"])
def test_static_assets_are_served_without_signing_in(gate_db, asset):
    response = application.app.test_client().get(f"/static/{asset}")
    assert response.status_code == 200 and len(response.data) > 1000
    response.close()
