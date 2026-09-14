"""
Every JavaScript handler referenced from markup must actually be defined.

This exists because an editing mistake silently deleted ten functions from
the entry page — including startAutoDetect, the Auto-Detect Vehicle button's
handler — and every server-side test still passed, because the page returned
200 with a dead button. These checks are cheap and catch that class of
breakage.
"""
import re
from pathlib import Path

import pytest

TEMPLATES = Path(__file__).resolve().parent.parent / "templates"
INTERACTIVE_PAGES = ["gate_entry.html", "gate_exit.html", "gate_trips.html",
                     "settings.html", "reports.html", "dashboard.html",
                     "face_search.html", "fingerprint_search.html"]


def page_source(name):
    return (TEMPLATES / name).read_text()


def scripts(source):
    return "\n".join(re.findall(r"<script[^>]*>(.*?)</script>", source, re.S))


@pytest.mark.parametrize("page", INTERACTIVE_PAGES)
def test_every_inline_handler_is_defined(page):
    source = page_source(page)
    js = scripts(source)
    handlers = set()
    for attribute in ("onclick", "onchange", "oninput", "onsubmit", "onkeydown"):
        for expression in re.findall(rf'{attribute}="([^"]+)"', source):
            # Strip string literals first: prose inside a confirm() message
            # such as "Clear completed trips (EXITED…)" would otherwise look
            # like a call to trips().
            without_strings = re.sub(r"'[^']*'|&#39;[^&]*&#39;", "''", expression)
            handlers.update(re.findall(r"\b([a-zA-Z_]\w*)\s*\(", without_strings))

    builtins = {"confirm", "alert", "return", "if", "event", "parseInt", "fetch"}
    for name in sorted(handlers - builtins):
        assert re.search(rf"\bfunction\s+{name}\s*\(", js) \
            or re.search(rf"\b(?:const|let|var)\s+{name}\s*=", js), \
            f"{page}: handler {name}() is referenced in markup but never defined"


@pytest.mark.parametrize("page", INTERACTIVE_PAGES)
def test_script_blocks_are_balanced(page):
    js = scripts(page_source(page))
    for opener, closer in (("{", "}"), ("(", ")"), ("[", "]")):
        assert js.count(opener) == js.count(closer), \
            f"{page}: unbalanced {opener}{closer} in inline script"


def test_entry_page_wires_the_attribute_panel_into_the_vehicle_stream():
    """The attribute payload must be rendered by the capture SSE handler."""
    js = scripts(page_source("gate_entry.html"))
    assert "showAttrPanel(msg.vehicle_attributes)" in js
    assert "function showAttrPanel(" in js


def test_entry_page_auto_detect_posts_to_the_capture_route():
    js = scripts(page_source("gate_entry.html"))
    assert "function startAutoDetect(" in js
    assert "/gate/entry/vehicle/auto-start" in js
    assert "/gate/entry/vehicle/stream/" in js


def test_exit_page_renders_the_attribute_comparison():
    js = scripts(page_source("gate_exit.html"))
    assert "function renderAttributeComparison(" in js
    assert "renderAttributeComparison(d.entry_attributes" in js
