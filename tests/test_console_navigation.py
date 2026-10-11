"""Verify the console's single navigation model from shipped markup and fake-DOM scripts."""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

STATIC = Path(__file__).resolve().parents[1] / "synria_lerobot" / "console" / "static"


def test_global_nav_has_three_destinations_and_flows_have_one_exit() -> None:
    """Home, Sessions and Help are the only top-level places; flows show a breadcrumb and exit."""
    markup = (STATIC / "index.html").read_text(encoding="utf-8")
    nav = re.search(r'<nav id="global-nav".*?</nav>', markup, re.S)
    assert nav is not None
    links = re.findall(r'<a href="(#/[a-z]+)" data-route="([a-z]+)">([A-Za-z]+)</a>', nav.group(0))
    assert links == [("#/home", "home", "Home"), ("#/sessions", "sessions", "Sessions"),
                     ("#/help", "help", "Help")]
    assert 'id="flow-nav"' in markup and 'id="breadcrumb"' in markup and 'id="exit-flow"' in markup
    for stale in ('data-screen="record"', 'data-screen="review"', 'data-screen="trash"',
                  'id="open-help"', 'id="review-session"'):
        assert stale not in markup
    assert 'id="route-announcer"' in markup and 'aria-live="polite"' in markup
    assert 'id="token-notice"' in markup and 'id="status-strip"' in markup
    assert 'id="sessions-more"' in markup and 'id="reindex"' in markup
    assert 'id="trash-list"' in markup
    assert '<script src="/router.js" defer></script>' in markup
    assert '<script src="/home.js" defer></script>' in markup
    assert '<script src="/teach.js" defer></script>' in markup
    assert 'id="teach-sentence"' in markup and 'id="teach-save"' in markup


def test_every_screen_has_a_focusable_title_and_route_labels_are_unique() -> None:
    """Each screen carries a heading focus lands on, and no label points at two places."""
    markup = (STATIC / "index.html").read_text(encoding="utf-8")
    for screen in ("home", "intro", "teach", "sessions", "help", "new", "record", "review",
                   "summary"):
        pattern = rf'<section id="{screen}" class="screen[^"]*"[^>]*>(.*?)</section>'
        section = re.search(pattern, markup, re.S)
        assert section is not None, screen
        assert 'class="screen-title" tabindex="-1"' in section.group(1), screen
    targets: dict[str, set[str]] = {}
    for route, label in re.findall(r'data-route="([^"]+)">([^<]+)</a>', markup):
        targets.setdefault(label.strip(), set()).add(route)
    assert all(len(routes) == 1 for routes in targets.values()), targets


def test_router_and_home_status_behave_with_fakes() -> None:
    """Parse, build and round-trip every route and describe status rows without a browser."""
    executable = shutil.which("node")
    if executable is None:
        pytest.skip("optional JavaScript runtime is unavailable")
    result = subprocess.run(
        [executable, str(Path(__file__).with_name("console_router.cjs")),
         str(STATIC / "router.js"), str(STATIC / "home.js")],
        check=True, capture_output=True, text=True, timeout=15,
    )
    assert "router and home status verified" in result.stdout
    for script in ("router.js", "home.js", "app.js", "operator-guide.js"):
        subprocess.run([executable, "--check", str(STATIC / script)], check=True, timeout=15)
