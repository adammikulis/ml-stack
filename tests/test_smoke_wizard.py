"""The bundle smoke test's wizard walk finishes setup against the page the daemon serves."""

import importlib.util
from pathlib import Path

import pytest
import test_fleet_page as fleet_page

browser = fleet_page.browser
daemon = fleet_page.daemon
joined = fleet_page.joined
open_page = fleet_page.open_page
no_release_lookup = fleet_page.no_release_lookup
pytestmark = pytest.mark.slow

SMOKE = Path(__file__).resolve().parents[1] / "packaging" / "smoke.py"


def _smoke():
    spec = importlib.util.spec_from_file_location("bundle_smoke", SMOKE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_walk_creates_the_pool_and_reaches_the_end_of_setup(daemon, open_page, monkeypatch):
    from ml_stack.fleet import automatic_clusters
    from ml_stack.fleet.discovery import memberships

    monkeypatch.setattr(automatic_clusters, "offers", lambda port=None: [])
    page, errors = open_page(daemon)
    page.wait_for_selector("#first-run:not([hidden])")
    _smoke().first_run(page, "ci-runner", fleet_page.WORDS, "ci")
    _smoke().sign_in(page, fleet_page.WORDS)
    assert "ci" in [m.group for m in memberships(daemon.keyfile)]
    assert not errors


def test_the_walk_opens_every_screen_of_a_signed_in_machine(joined, open_page):
    page, errors = open_page(joined, cookie=joined.cookie)
    _smoke().every_screen(page)
    assert not errors
