"""Session negotiation orders protected calls in the browser."""
from __future__ import annotations

import pytest
from test_fleet_initial_setup import initial
from test_fleet_ui import Serving, no_release_lookup
from test_workspace_board_ui import browser

from ml_stack.fleet import automatic_clusters
from ml_stack.fleet.discovery import mint_cluster

__all__ = ["browser", "no_release_lookup"]


@pytest.fixture
def device(tmp_path, monkeypatch):
    monkeypatch.setattr(automatic_clusters, "offers", lambda port=None: [])
    served = Serving(tmp_path, secure=False)
    try:
        yield served
    finally:
        served.close()


DELAY_SESSION = """
(() => {
  const request = window.fetch.bind(window);
  window.bootstrapRequests = [];
  window.fetch = async (path, options) => {
    if (path === '/ui/setup/local-session') {
      window.bootstrapPending = true;
      await new Promise(resolve => setTimeout(resolve, 800));
      const response = await request(path, options);
      window.bootstrapFinished = true;
      return response;
    }
    if (path === '/ui/room' || path === '/ui/workspace/jobs' || path === '/ui/peers') {
      window.bootstrapRequests.push({path, negotiated:!!window.bootstrapFinished});
    }
    return request(path, options);
  };
})();
"""


@pytest.mark.slow
def test_delayed_local_session_orders_eager_requests_and_expiry_still_signs_in(device, browser):
    assert initial(device)[0] == 200
    device.ui.setup_finished()
    page = browser.new_page()
    try:
        page.add_init_script(DELAY_SESSION)
        page.goto(f"http://127.0.0.1:{device.port}/ui/#cluster")
        page.wait_for_function("() => window.bootstrapPending === true")
        cancelled = page.evaluate("""async () => {
          const control = new AbortController();
          const pending = fleetModel.api('/ui/peers', {signal:control.signal});
          control.abort();
          try { await pending; return false; }
          catch (error) { return error.name === 'AbortError'; }
        }""")
        assert cancelled
        assert page.evaluate("bootstrapRequests.length") == 0
        page.wait_for_function("() => fleetModel.route === 'cluster'")
        page.wait_for_function("() => bootstrapRequests.some(r => r.path === '/ui/room') && bootstrapRequests.some(r => r.path === '/ui/workspace/jobs')")
        assert page.evaluate("bootstrapRequests.every(r => r.negotiated)")
        with device.ui.sessions._lock:
            for session in device.ui.sessions._sessions.values():
                session.expires_at = 0
        status = page.evaluate("""async () => {
          const result = await fleetModel.api('/ui/peers');
          fleetModel.refused(result);
          return result.status;
        }""")
        assert status == 401
        page.wait_for_function("() => fleetModel.route === 'sign-in'")
        # a development pool on this computer is signed in to by choosing it; otherwise by passphrase
        assert page.get_by_role("heading", name="Sign in", exact=True).is_visible()
        assert page.locator("#signin-pools button").count() or page.get_by_role("button", name="Sign in", exact=True).is_visible()
    finally:
        page.close()


@pytest.mark.slow
@pytest.mark.parametrize("mode,expected", [("first", "first-run"), ("prod", "sign-in")])
def test_bootstrap_preserves_setup_and_password_routes(device, browser, mode, expected):
    if mode == "prod":
        mint_cluster("private", device.keyfile, mode="prod")
        device.ui.settings.setup_done = True
    page = browser.new_page()
    try:
        page.goto(f"http://127.0.0.1:{device.port}/ui/#cluster")
        page.wait_for_function(f"() => fleetModel.route === '{expected}'")
        assert page.evaluate("fleetModel.requestedRoute") == "cluster"
    finally:
        page.close()
