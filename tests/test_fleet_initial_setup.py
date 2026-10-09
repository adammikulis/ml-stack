"""First-run mode selection and local browser admission over real sockets."""
from __future__ import annotations

import pytest
from launch_support import browser as browser_headers, sign_in
from test_fleet_ui import Serving

from poolhouse.fleet import automatic_clusters
from poolhouse.fleet.daemon import DaemonOptions, DaemonRuntime
from poolhouse.fleet.discovery import memberships, mint_cluster
from poolhouse.fleet.settings import Settings


@pytest.fixture
def device(tmp_path, monkeypatch):
    monkeypatch.setattr(automatic_clusters, "offers", lambda port=None: [])
    served = Serving(tmp_path)
    try:
        yield served
    finally:
        served.close()


def initial(device, mode="dev", cookie=None, **extra):
    return device.call("/ui/setup/initial", method="POST", headers=browser_headers(device),
                       cookie=sign_in(device) if cookie is None else cookie,
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
    {"Authorization": "Bearer agent"}, {"X-Poolhouse-Token": "agent"},
    {"X-Poolhouse-Agent": "helper"}, {"Host": "[malformed"},
])
def test_initial_rejects_foreign_and_agent_requests(device, changes):
    headers = {**browser_headers(device), **changes}
    status, _, _ = device.call("/ui/setup/initial", method="POST", headers=headers,
                               cookie=sign_in(device), body={"name": "quillhaven", "cluster_mode": "dev"})
    assert status == 403
    assert not memberships(device.keyfile)


def test_initial_rejects_mode_change_and_nonboolean_automatic(device):
    cookie = sign_in(device)
    mint_cluster("private", device.keyfile, mode="prod")
    assert initial(device, cookie=cookie)[0] == 400
    assert initial(device, "prod", cookie=cookie, automatic="yes")[0] == 400
    assert memberships(device.keyfile)[0].mode == "prod"


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
    token_session = device.ui.sessions.open("token", "token")
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


def test_completed_prod_hands_no_launch_session_and_refuses_initial(device):
    from launch_support import redeem, ticket

    mint_cluster("private", device.keyfile, mode="prod")
    device.ui.setup_finished()
    status, issued, _ = ticket(device)
    assert status == 200
    status, _, headers = redeem(device, issued["ticket"])
    assert status == 401
    assert "Set-Cookie" not in headers
    status, _, _ = device.call("/ui/setup/initial", method="POST", headers=browser_headers(device),
                                  body={"name": "quillhaven", "cluster_mode": "prod"})
    assert status == 403
    assert len(device.ui.sessions) == 0
    assert {"event": "session.refused", "reason": "production-needs-passphrase",
            "source": "127.0.0.1"} in device.rows


def test_fresh_automatic_dev_can_choose_prod_explicitly(device):
    automatic_clusters.ensure(device.keyfile, mode="dev")
    assert initial(device, "prod")[0] == 200
    assert not memberships(device.keyfile)
    assert Settings.load(device.ui.settings_path).cluster_mode == "prod"


def test_initial_never_drops_manual_dev_membership(device):
    from dataclasses import replace

    from poolhouse.fleet.discovery import adopt

    member = mint_cluster("private", device.keyfile, mode="dev")
    adopt(replace(member, selection="manual"), device.keyfile)
    assert initial(device, "prod")[0] == 400
    assert memberships(device.keyfile)[0].group == "private"


@pytest.mark.redteam
@pytest.mark.parametrize("changes", [
    {"Origin": "http://foreign.invalid"}, {"Sec-Fetch-Site": "cross-site"},
    {"Authorization": "Bearer agent"}, {"X-Poolhouse-Token": "agent"},
    {"X-Poolhouse-Agent": "helper"}, {"Host": "rebound.invalid"},
])
def test_owner_admission_refuses_foreign_and_agent_requests_even_with_a_launch_session(device, changes):
    status, _, headers = device.call("/ui/setup/initial", method="POST", cookie=sign_in(device),
                                     headers={**browser_headers(device), **changes},
                                     body={"name": "quillhaven", "cluster_mode": "dev"})
    assert status == 403
    assert "Set-Cookie" not in headers
    assert len(device.ui.sessions) == 1
    assert not memberships(device.keyfile)


@pytest.mark.redteam
def test_owner_admission_refuses_lan_and_missing_ui_header(device):
    from poolhouse.fleet.discovery import primary_ip

    body = {"name": "quillhaven", "cluster_mode": "dev"}
    cookie = sign_in(device)
    for options in ({"host": primary_ip()}, {"ui_header": False}):
        status, _, headers = device.call("/ui/setup/initial", method="POST", cookie=cookie,
                                         headers=browser_headers(device), body=body, **options)
        assert status == 403
        assert "Set-Cookie" not in headers
    assert len(device.ui.sessions) == 1
    assert not memberships(device.keyfile)


@pytest.mark.redteam
@pytest.mark.parametrize("body", [
    {"name": "../escaped", "cluster_mode": "dev"},
    {"name": "quillhaven", "cluster_mode": "owner"},
    {"name": "quillhaven", "cluster_mode": "dev", "automatic": "yes"},
    {"name": "quillhaven", "cluster_mode": "dev", "root": "../escaped"},
    {"name": "x" * 5000, "cluster_mode": "dev"},
    ["quillhaven", "dev"],
])
def test_initial_rejects_hostile_body_without_identity_or_membership_changes(device, body):
    status, _, headers = device.call("/ui/setup/initial", method="POST", headers=browser_headers(device),
                                     cookie=sign_in(device), body=body)
    assert status == 400
    assert "Set-Cookie" not in headers
    assert len(device.ui.sessions) == 1
    assert not memberships(device.keyfile)
    assert device.ui.name == "studio"


@pytest.mark.redteam
def test_setup_jobs_refuses_untrusted_readers_and_ignores_external_paths(device, tmp_path):
    from poolhouse.fleet.setup_jobs import Jobs

    device.ui.root = tmp_path / "traind"
    device.ui.setup_jobs = Jobs(device.ui.root)
    mint_cluster("private", device.keyfile, mode="dev")
    device.ui.setup_finished()
    for expected, options in ((401, {}), (401, {"cookie": "poolhouse-session=forged"}),
                              (403, {"ui_header": False}), (401, {"headers": {"Host": "rebound.invalid"}})):
        status, _, _ = device.call("/ui/setup/jobs", **options)
        assert status == expected
    outside = tmp_path / "external.json"
    outside.write_text('{"secret":"external-job-store"}')
    session = device.ui.sessions.open("setup", "launch-ticket")
    cookie = device.ui.sessions.cookie_header(session).split(";")[0]
    status, body, _ = device.call("/ui/setup/jobs?root=../external.json&path=../external.json", cookie=cookie)
    assert status == 200
    assert body == {"jobs": []}
    assert outside.read_text() == '{"secret":"external-job-store"}'
