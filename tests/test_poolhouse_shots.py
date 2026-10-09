"""Regenerates the screenshots in docs/brand-ui: every main screen, in both vocabularies.

Skipped unless ``ML_STACK_UI_SHOTS`` is set, because it writes into the docs folder. The
devices, the shared project and the board messages are invented samples (the daemon here has no
second machine and no project registry), served to the page the way the real routes would.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.slow

import test_fleet_page as fleet_page  # noqa: E402

browser = fleet_page.browser
daemon = fleet_page.daemon
joined = fleet_page.joined
no_release_lookup = fleet_page.no_release_lookup
open_page = fleet_page.open_page
with_peers = fleet_page.with_peers



@pytest.fixture(autouse=True)
def only_when_asked():
    """Writing into docs/ is opt-in."""
    if not os.environ.get("ML_STACK_UI_SHOTS"):
        pytest.skip("writes docs/brand-ui; set ML_STACK_UI_SHOTS")


OUT = Path(__file__).resolve().parent.parent / "docs" / "brand-ui"
SIZES = {"desktop": {"width": 1400, "height": 900}, "phone": {"width": 390, "height": 844}}
PROJECT = "a" * 32


def device(name, vendor, ram, slots_free, **more):
    """One invented device: ``more`` may carry gpu, serving, paused and is_self."""
    slots, free = slots_free
    return {"name": name, "port": 8770, "host": "10.0.0.9", "base_url": f"http://10.0.0.{len(name)}:8770",
            "is_self": more.get("is_self", False), "clusters": ["home"], "slots": slots, "free": free, "queued": 0,
            "busy": free == 0,
            "device": {"vendor": vendor, "cuda": vendor == "nvidia", "gpu": more.get("gpu", ""), "cpus": 16, "ram_gb": ram,
                       "ram_used_gb": ram / 3,
                       "availability": {"paused": more.get("paused", False), "unavailable_because": "paused"},
                       "serving": [{"models": [more["serving"]], "slots": slots}] if more.get("serving") else []}}


SAMPLE_DEVICES = [
    device("studio-mac", "apple", 128.0, (4, 3), gpu="M-series GPU", serving="thornfield-8B-Q4_K_M", is_self=True),
    device("garage-tower", "nvidia", 64.0, (4, 0), gpu="Marrowgate 5000", serving="marrowgate-A3B-Q4"),
    device("attic-laptop", "cpu", 32.0, (2, 2)),
    device("shed-nuc", "amd", 48.0, (2, 1), gpu="Greenhollow 7", paused=True),
]
SAMPLE_BOARD = {"ok": True, "state": "connected",
                "agents": [{"label": "reviewer", "role": "agent", "online": True, "model": "thornfield-8B"},
                           {"label": "builder", "role": "agent", "online": True},
                           {"label": "night-watch", "role": "agent", "online": False}],
                "messages": [{"from": "builder", "text": "Merged the pool screen branch; tests are green on the garage tower."},
                             {"from": "reviewer", "text": "Two small notes on the sidebar spacing, nothing blocking."},
                             {"from": "night-watch", "text": "Quiet night. One device paused itself at 02:10 and resumed."}]}


def sample(page):
    """Answer the project routes with the invented sample, leave every other route to the daemon."""
    def respond(route):
        url = route.request.url
        body = {"ok": True, "projects": [{"id": PROJECT, "name": "ledger-service", "state": "available", "is_self": True},
                                         {"id": "b" * 32, "name": "field-notes", "state": "available", "is_self": False, "peer": "attic-laptop"}],
                "candidates": [], "devices": []} if url.rstrip("/").endswith("/ui/projects") else SAMPLE_BOARD
        route.fulfill(status=200, content_type="application/json", body=json.dumps(body))
    page.route("**/ui/projects", respond)
    page.route(f"**/ui/projects/{PROJECT}/board", respond)


@pytest.fixture
def full_pool(joined):
    joined.ui.peers = lambda force=False: SAMPLE_DEVICES
    return joined


@pytest.mark.parametrize("vocab", ["professional", "friendly"])
@pytest.mark.parametrize("size", SIZES)
def test_main_screens(full_pool, open_page, vocab, size):
    OUT.mkdir(parents=True, exist_ok=True)
    for name, route in (("chat", "chat"), ("devices", "cluster"), ("projects", "projects"), ("settings", "settings")):
        page, errors = open_page(full_pool, path=f"/ui/?vocab={vocab}#{route}", cookie=full_pool.cookie)
        page.set_viewport_size(SIZES[size])
        sample(page)
        page.reload()
        page.wait_for_function(f"() => window.fleetModel && window.fleetModel.route === '{route}'")
        if route == "settings":
            page.click('[data-section="developer"]')
        page.wait_for_timeout(900)
        page.screenshot(path=str(OUT / f"poolhouse-{vocab}-{name}-{size}.png"), full_page=False)
        assert errors == []


@pytest.mark.parametrize("vocab", ["professional", "friendly"])
@pytest.mark.parametrize("size", SIZES)
def test_first_run(daemon, open_page, vocab, size):
    OUT.mkdir(parents=True, exist_ok=True)
    page, errors = open_page(daemon, path=f"/ui/?vocab={vocab}")
    page.set_viewport_size(SIZES[size])
    page.wait_for_selector("#first-run-card h1")
    page.wait_for_timeout(500)
    page.screenshot(path=str(OUT / f"poolhouse-{vocab}-setup-{size}.png"))
    assert errors == []
