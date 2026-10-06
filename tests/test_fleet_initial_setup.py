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


@pytest.mark.redteam
@pytest.mark.parametrize("path", ["/ui/setup/initial", "/ui/setup/local-session"])
@pytest.mark.parametrize("changes", [
    {"Origin": "http://foreign.invalid"}, {"Sec-Fetch-Site": "cross-site"},
    {"Authorization": "Bearer agent"}, {"X-ML-Stack-Token": "agent"},
    {"X-ML-Stack-Agent": "helper"}, {"Host": "rebound.invalid"},
])
def test_owner_admission_routes_refuse_foreign_and_agent_requests(device, path, changes):
    status, _, headers = device.call(path, method="POST", headers={**browser_headers(device), **changes},
                                     body={"name": "quillhaven", "cluster_mode": "dev"})
    assert status == 403
    assert "Set-Cookie" not in headers
    assert len(device.ui.sessions) == 0
    assert not memberships(device.keyfile)


@pytest.mark.redteam
@pytest.mark.parametrize("path", ["/ui/setup/initial", "/ui/setup/local-session"])
def test_owner_admission_routes_refuse_lan_and_missing_ui_header(device, path):
    from ml_stack.fleet.discovery import primary_ip

    body = {"name": "quillhaven", "cluster_mode": "dev"}
    for options in ({"host": primary_ip()}, {"ui_header": False}):
        status, _, headers = device.call(path, method="POST", headers=browser_headers(device), body=body, **options)
        assert status == 403
        assert "Set-Cookie" not in headers
    assert len(device.ui.sessions) == 0
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
    status, _, headers = device.call("/ui/setup/initial", method="POST", headers=browser_headers(device), body=body)
    assert status == 400
    assert "Set-Cookie" not in headers
    assert len(device.ui.sessions) == 0
    assert not memberships(device.keyfile)
    assert device.ui.name == "studio"


@pytest.mark.redteam
@pytest.mark.parametrize("mode", ["dev", "prod"])
def test_local_session_cannot_upgrade_agent_cookie_or_prod_membership(device, mode):
    mint_cluster("private", device.keyfile, mode=mode)
    device.ui.setup_finished()
    token_session = device.ui.sessions.open("token")
    cookie = device.ui.sessions.cookie_header(token_session).split(";")[0]
    status, _, headers = device.call("/ui/setup/local-session", method="POST", headers=browser_headers(device), cookie=cookie)
    assert status == 403
    assert "Set-Cookie" not in headers
    assert len(device.ui.sessions) == 1
    if mode == "prod":
        status, _, headers = device.call("/ui/setup/local-session", method="POST", headers=browser_headers(device))
        assert status == 400
        assert "Set-Cookie" not in headers
        assert len(device.ui.sessions) == 1


@pytest.mark.redteam
def test_setup_jobs_refuses_untrusted_readers_and_ignores_external_paths(device, tmp_path):
    from ml_stack.fleet.setup_jobs import Jobs

    device.ui.root = tmp_path / "traind"
    device.ui.setup_jobs = Jobs(device.ui.root)
    mint_cluster("private", device.keyfile, mode="dev")
    device.ui.setup_finished()
    for expected, options in ((401, {}), (401, {"cookie": "ml-stack-session=forged"}),
                              (403, {"ui_header": False}), (401, {"headers": {"Host": "rebound.invalid"}})):
        status, _, _ = device.call("/ui/setup/jobs", **options)
        assert status == expected
    outside = tmp_path / "external.json"
    outside.write_text('{"secret":"external-job-store"}')
    session = device.ui.sessions.open("setup")
    cookie = device.ui.sessions.cookie_header(session).split(";")[0]
    status, body, _ = device.call("/ui/setup/jobs?root=../external.json&path=../external.json", cookie=cookie)
    assert status == 200
    assert body == {"jobs": []}
    assert outside.read_text() == '{"secret":"external-job-store"}'
