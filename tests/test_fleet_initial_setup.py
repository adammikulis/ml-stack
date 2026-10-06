"""First-run mode selection and local browser admission over real sockets."""
from __future__ import annotations

import pytest
from test_fleet_ui import Serving

from ml_stack.fleet import automatic_clusters
from ml_stack.fleet.daemon import DaemonOptions, DaemonRuntime
from ml_stack.fleet.discovery import memberships, mint_cluster
from ml_stack.fleet.settings import Settings


@pytest.fixture
def device(tmp_path, monkeypatch):
    monkeypatch.setattr(automatic_clusters, "offers", lambda port=None: [])
    served = Serving(tmp_path)
    try:
        yield served
    finally:
        served.close()


def browser_headers(device):
    return {"Origin": f"http://127.0.0.1:{device.port}", "Sec-Fetch-Site": "same-origin"}


def initial(device, mode="dev", **extra):
    return device.call("/ui/setup/initial", method="POST", headers=browser_headers(device),
                       body={"name": "quillhaven", "cluster_mode": mode, **extra})


def test_dev_name_creates_automatic_membership_without_finishing_wizard(device):
    status, state, headers = initial(device)
    assert status == 200
    assert state["name"] == "quillhaven"
    assert state["cluster_mode"] == "dev"
    assert state["needs_setup"] is True
    assert memberships(device.keyfile)[0].mode == "dev"
    assert "HttpOnly" in headers["Set-Cookie"]
    assert Settings.load(device.ui.settings_path).cluster_mode == "dev"


def test_prod_name_stays_unpaired_and_keeps_full_wizard(device):
    status, state, _ = initial(device, "prod")
    assert status == 200
    assert state["needs_setup"] is True
    assert state["cluster_mode"] == "prod"
    assert memberships(device.keyfile) == []


@pytest.mark.parametrize("changes", [
    {"Origin": "http://foreign.invalid"}, {"Sec-Fetch-Site": "cross-site"},
    {"Authorization": "Bearer agent"}, {"X-ML-Stack-Token": "agent"},
    {"X-ML-Stack-Agent": "helper"}, {"Host": "[malformed"},
])
def test_initial_rejects_foreign_and_agent_requests(device, changes):
    headers = {**browser_headers(device), **changes}
    status, _, _ = device.call("/ui/setup/initial", method="POST", headers=headers,
                               body={"name": "quillhaven", "cluster_mode": "dev"})
    assert status == 403
    assert not memberships(device.keyfile)


def test_initial_rejects_mode_change_and_nonboolean_automatic(device):
    mint_cluster("private", device.keyfile, mode="prod")
    assert initial(device)[0] == 400
    assert initial(device, "prod", automatic="yes")[0] == 400
    assert memberships(device.keyfile)[0].mode == "prod"


def test_returning_dev_local_session_requires_completed_setup(device):
    initial(device)
    def request():
        return device.call("/ui/setup/local-session", method="POST", headers=browser_headers(device))
    assert request()[0] == 400
    device.ui.setup_finished()
    status, _, headers = request()
    assert status == 200
    assert device.ui.authed(headers["Set-Cookie"].split(";")[0])


def test_fresh_browser_daemon_defers_admission(tmp_path, monkeypatch):
    key = tmp_path / "cluster.key"
    monkeypatch.setattr(automatic_clusters, "offers", lambda port=None: [])
    runtime = DaemonRuntime(DaemonOptions(root=tmp_path / "root", cluster_key_path=key, initial_setup=True))
    runtime.configure()
    assert not memberships(key)
    assert runtime.effective_mode == "dev"


def test_saved_prod_daemon_stays_unpaired_on_restart(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    Settings(cluster_mode="prod").save(root / "settings.json")
    runtime = DaemonRuntime(DaemonOptions(root=root, cluster_key_path=tmp_path / "cluster.key"))
    runtime.configure()
    assert runtime.effective_mode == "prod"
    assert not memberships(runtime.cluster_key_path)


def test_token_browser_session_cannot_mint_owner_session(device):
    token_session = device.ui.sessions.open("token")
    cookie = device.ui.sessions.cookie_header(token_session).split(";")[0]
    status, _, _ = device.call("/ui/setup/initial", method="POST", headers=browser_headers(device),
                               cookie=cookie, body={"name": "quillhaven", "cluster_mode": "dev"})
    assert status == 403
    assert not memberships(device.keyfile)


def test_updater_waits_for_downloads_and_background_installations():
    from types import SimpleNamespace

    runtime = DaemonRuntime(DaemonOptions())
    runtime.runner = SimpleNamespace(status=lambda: {"busy": False})
    runtime.downloads = SimpleNamespace(active=lambda: [])
    runtime.interface = None
    assert runtime.background_busy() is False
    runtime.downloads.active = lambda: [SimpleNamespace(state="getting")]
    assert runtime.background_busy() is True
    runtime.downloads.active = lambda: [SimpleNamespace(state="done")]
    assert runtime.background_busy() is False
    runtime.interface = SimpleNamespace(setup_jobs=SimpleNamespace(active=lambda: True))
    assert runtime.background_busy() is True
    runtime.interface.setup_jobs.active = lambda: False
    assert runtime.background_busy() is False


def test_completed_prod_cannot_open_initial_session_anonymously(device):
    mint_cluster("private", device.keyfile, mode="prod")
    device.ui.setup_finished()
    status, body, _ = initial(device, "prod")
    assert status == 400
    assert "sign in" in body["error"]
    assert len(device.ui.sessions) == 0


def test_fresh_automatic_dev_can_choose_prod_explicitly(device):
    automatic_clusters.ensure(device.keyfile, mode="dev")
    assert initial(device, "prod")[0] == 200
    assert not memberships(device.keyfile)
    assert Settings.load(device.ui.settings_path).cluster_mode == "prod"


def test_initial_never_drops_manual_dev_membership(device):
    from dataclasses import replace

    from ml_stack.fleet.discovery import adopt

    member = mint_cluster("private", device.keyfile, mode="dev")
    adopt(replace(member, selection="manual"), device.keyfile)
    assert initial(device, "prod")[0] == 400
    assert memberships(device.keyfile)[0].group == "private"
