"""Fleet cards display advertised addresses while local requests use loopback."""

from types import SimpleNamespace

import pytest
from test_fleet_initial_setup import initial
from test_fleet_ui import Serving, no_release_lookup

from poolhouse.fleet import automatic_clusters
from poolhouse.fleet.projects import ProjectRegistry
from poolhouse.fleet.ui import UI
from poolhouse.scrape.browser import Window, browser

__all__ = ["no_release_lookup"]


@pytest.mark.parametrize("host", ["http://192.168.4.8:8770", "https://192.168.4.9:8770", ""])
def test_self_address_uses_registered_lan_host_and_preserves_local_requests(host, monkeypatch):
    interface = UI(name="cedar", peer_port=8770)
    interface.projects = SimpleNamespace(host=host)
    row = interface.itself()
    assert row["display_url"] == host
    assert row["base_url"] == "http://127.0.0.1:8770"
    assert row["host"] == "127.0.0.1"
    monkeypatch.setattr(interface, "peers", lambda: [row])
    monkeypatch.setattr(interface, "bench_state", lambda: {})
    displayed = interface.fleet()["peers"][0]
    assert displayed["display_url"] == host
    assert displayed["base_url"] == "http://127.0.0.1:8770"


def test_unregistered_self_address_has_no_invented_lan_host():
    row = UI(name="cedar", peer_port=8770).itself()
    assert row["display_url"] == ""
    assert row["base_url"] == "http://127.0.0.1:8770"


@pytest.mark.slow
@pytest.mark.parametrize("advertised", ["http://192.168.4.8:8770", ""])
def test_fleet_card_displays_authoritative_lan_address(tmp_path, monkeypatch, advertised):
    pytest.importorskip("playwright.sync_api", reason="poolhouse[scrape]")
    monkeypatch.setattr(automatic_clusters, "offers", lambda port=None: [])
    served = Serving(tmp_path, secure=False)
    try:
        served.ui.projects = ProjectRegistry(tmp_path / "projects", "device", host=advertised)
        served.ui.peer_port = served.port
        served.ui.report = lambda: {"gpu": "test accelerator", "ram_gb": 32}
        assert initial(served)[0] == 200
        served.ui.setup_finished()
        with browser(Window(tmp_path / "browser", channel="chromium", headless=True)) as page:
            page.goto(f"http://127.0.0.1:{served.port}/ui/?display_url=https://untrusted.invalid/#cluster")
            card = page.locator("cluster-view .peer.self .url")
            card.wait_for(state="visible")
            assert card.inner_text() == (advertised or f"http://127.0.0.1:{served.port}")
            assert page.locator("cluster-view .peer.self").get_by_text("test accelerator", exact=True).is_visible()
            assert page.evaluate("async () => (await fleetModel.api('/ui/peers')).peers[0].base_url") == f"http://127.0.0.1:{served.port}"
    finally:
        served.close()
