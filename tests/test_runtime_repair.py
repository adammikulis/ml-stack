import json
import os
import re
import subprocess
import threading
import time
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from ml_stack.fleet import daemon, runtime_repair
from ml_stack.fleet.initial_setup_routes import InitialSetupRoutes
from ml_stack.fleet.runtime_repair_routes import RuntimeRepairRoutes
from ml_stack.fleet.session import Sessions
from ml_stack.fleet.setup_jobs import Jobs


@pytest.mark.slow
def test_frozen_agent_runtime_probe_and_repair_dispatch(tmp_path):
    binary = os.environ.get("ML_STACK_FROZEN_BINARY", "")
    if not binary:
        pytest.skip("set ML_STACK_FROZEN_BINARY to the built standalone daemon")
    environment = {**os.environ, "ML_STACK_HOME": str(tmp_path / "home"), "PYTHONPATH": "",
                   "PYTHON_KEYRING_BACKEND": "keyring.backends.null.Keyring"}
    probe = subprocess.run([binary, "--check-agent-runtime"], env=environment,
                           capture_output=True, text=True, timeout=30, check=True)
    observed = json.loads(probe.stdout)
    assert observed["runtime_ready"] and re.fullmatch("[0-9a-f]{40}", observed["commit"])
    repair = subprocess.run([binary, "-m", "ml_stack.cli.daemon", "--root", str(tmp_path / "daemon"),
                             "--agent-runtime-job", "a" * 32], env=environment, capture_output=True, text=True, timeout=30)
    assert repair.returncode == 1 and "repair job is not waiting" in repair.stderr


def test_waiting_repair_is_durable_and_does_not_block_downloads(tmp_path):
    queue = Jobs(tmp_path)
    repair = queue.start(runtime_repair.KIND, {}, None)
    assert repair["state"] == "waiting"
    assert not queue.active()
    assert queue.start(runtime_repair.KIND, {}, None)["id"] == repair["id"]
    finished = threading.Event()
    download = queue.start("library", {}, lambda progress: finished.set() or {"ok": True})
    assert finished.wait(5)
    deadline = time.monotonic() + 5
    while queue.active() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert next(row for row in queue.all() if row["id"] == download["id"])["state"] == "done"
    recovered = Jobs(tmp_path)
    assert next(row for row in recovered.all() if row["id"] == repair["id"])["state"] == "waiting"


def test_job_updates_merge_newer_worker_state(tmp_path):
    queue = Jobs(tmp_path)
    stale = queue.start(runtime_repair.KIND, {}, None)
    latest = dict(stale)
    queue._update(latest, state="done", result={"runtime_ready": True})
    queue._update(stale, worker_pid=123)
    row = queue.all()[0]
    assert row["state"] == "done" and row["result"]["runtime_ready"] is True


class Route(InitialSetupRoutes, RuntimeRepairRoutes):
    def __init__(self, tmp_path):
        sessions = Sessions()
        person = sessions.open("setup")
        self.ui = SimpleNamespace(root=tmp_path, sessions=sessions, setup_jobs=None,
                                  _setup_jobs_lock=threading.Lock(), peer_port=8770)
        self.cookie = "ml_stack_ui=" + person.sid
        self.path = "/ui/agent-runtime/install"
        self.method = "POST"
        self.client_ip = "127.0.0.1"
        self.host_header = "127.0.0.1:8770"
        self.headers = {"Origin": "http://127.0.0.1:8770", "Sec-Fetch-Site": "same-origin", "Content-Length": "2"}
        self.payload = {}
        self.replies = []

    def header(self, key, default=""):
        return self.headers.get(key, default)

    def body(self):
        return self.payload

    def send(self, code, payload):
        self.replies.append((code, payload))


@pytest.mark.parametrize("action", ["install", "repair"])
@pytest.mark.parametrize("change", ["foreign-origin", "remote", "agent", "token", "no-session", "foreign-host"])
def test_install_requires_authenticated_local_browser(tmp_path, monkeypatch, change, action):
    route = Route(tmp_path)
    route.path = "/ui/agent-runtime/" + action
    if change == "foreign-origin":
        route.headers["Origin"] = "https://other.example"
    elif change == "remote":
        route.client_ip = "192.0.2.5"
    elif change == "agent":
        route.headers["X-ML-Stack-Agent"] = "worker"
    elif change == "token":
        route.cookie = "ml_stack_ui=" + route.ui.sessions.open("token").sid
    elif change == "no-session":
        route.cookie = ""
    elif change == "foreign-host":
        route.host_header = "other.example:8770"
    called = []
    monkeypatch.setattr(runtime_repair, "request", lambda *args: called.append(args))
    assert route.route()
    assert route.replies[0][0] == 403 and not called


@pytest.mark.parametrize("action", ["install", "repair", "status"])
@pytest.mark.parametrize("payload", [{"path": "/tmp/runtime"}, {"command": "pip install agents"}, {"commit": "a" * 40}, []])
def test_install_never_accepts_browser_artifact_or_commands(tmp_path, monkeypatch, payload, action):
    route = Route(tmp_path)
    route.path = "/ui/agent-runtime/" + action
    route.payload = payload
    called = []
    monkeypatch.setattr(runtime_repair, "request", lambda *args: called.append(args))
    assert (route.public_route() if action == "status" else route.route())
    assert route.replies[0][0] == 400 and not called


def test_install_queues_fixed_action_with_server_person_provenance(tmp_path, monkeypatch):
    route = Route(tmp_path)
    monkeypatch.setattr(runtime_repair, "resume", lambda *args: None)
    assert route.route()
    assert route.replies[0][0] == 202
    row = route.ui.setup_jobs.all()[0]
    assert row["request"] == {"repair": False} and row["kind"] == runtime_repair.KIND
    assert row["provenance"]["authentication"] == "setup"


def test_status_does_not_expose_artifact_paths_or_person_provenance(tmp_path, monkeypatch):
    route = Route(tmp_path)
    route.path = "/ui/agent-runtime/status"
    route.cookie = ""
    monkeypatch.setattr(runtime_repair, "resume", lambda *args: None)
    route.ui.setup_jobs = Jobs(tmp_path)
    row = route.ui.setup_jobs.start(runtime_repair.KIND, {}, None, provenance={"private": "person"})
    route.ui.setup_jobs._update(row, result={"candidate": {"path": "/private/artifact"}})
    assert route.public_route()
    assert set(route.replies[0][1]["job"]) == {"id", "state", "note", "error"}


def test_repair_rechecks_selected_artifact_before_admission(tmp_path, monkeypatch):
    queue = Jobs(tmp_path)
    row = queue.start(runtime_repair.KIND, {}, None)
    queue._update(row, result={"candidate": {"sha256": "a" * 64}})
    monkeypatch.setattr(runtime_repair.updates, "running_path", lambda: Path("/owned/Poolside.app"))
    monkeypatch.setattr(runtime_repair, "Candidates", lambda root: SimpleNamespace(select=lambda name: {"sha256": "b" * 64}))
    admitted = []
    monkeypatch.setattr(runtime_repair, "request_replacement", lambda *args: admitted.append(args))
    assert runtime_repair.install(tmp_path, row["id"]) == 1
    assert not admitted and queue.all()[0]["state"] == "failed"


def test_permanent_control_denial_does_not_retry(tmp_path, monkeypatch):
    row = {"result": {"port": 8770, "instance": "chosen"}}
    monkeypatch.setattr(runtime_repair, "already_running", lambda port: {"launcher_control": "chosen"})
    called = []

    def denied(*args):
        called.append(args)
        raise runtime_repair.ControlError("Owned control record is invalid")

    monkeypatch.setattr(runtime_repair, "request_replacement", denied)
    with pytest.raises(runtime_repair.ControlError, match="invalid"):
        runtime_repair._admit(tmp_path, row, {"commit": "a" * 40})
    assert len(called) == 1


def test_public_errors_redact_environment_credentials(tmp_path, monkeypatch):
    secret = "runtime-install-private-token"
    monkeypatch.setenv("TEST_ACCESS_TOKEN", secret)
    route = Route(tmp_path)
    route.path = "/ui/agent-runtime/status"
    monkeypatch.setattr(runtime_repair, "resume", lambda *args: None)
    route.ui.setup_jobs = Jobs(tmp_path)
    row = route.ui.setup_jobs.start(runtime_repair.KIND, {}, None)
    route.ui.setup_jobs._update(row, state="failed", error="Denied " + secret)
    assert route.public_route()
    assert secret not in route.replies[0][1]["job"]["error"]


@pytest.mark.parametrize("reported", [None, "b" * 8, "b" * 39 + "c", "b" * 40])
def test_replacement_requires_exact_full_health_commit_and_preserves_recovery(tmp_path, monkeypatch, reported):
    root = tmp_path / "root"
    queue = Jobs(root)
    row = queue.start(runtime_repair.KIND, {}, None)
    target = tmp_path / "Poolside.app"
    target.mkdir()
    (target / "identity").write_text("previous")
    candidate = {"sha256": "a" * 64, "commit": "b" * 40}
    queue._update(row, result={"candidate": candidate, "port": 8770, "instance": "chosen"})
    cache = root / "runtime-candidates"
    cache.mkdir()
    with zipfile.ZipFile(cache / (candidate["sha256"] + ".zip"), "w") as archive:
        archive.writestr("Poolside.app/identity", "replacement")
    monkeypatch.setattr(runtime_repair.updates, "running_path", lambda: target)
    monkeypatch.setattr(runtime_repair, "Candidates", lambda root: SimpleNamespace(select=lambda name: candidate))
    monkeypatch.setattr(runtime_repair, "_admit", lambda *args: 8770)
    starts = []
    monkeypatch.setattr(runtime_repair, "_start", lambda *args: starts.append(args) or SimpleNamespace(poll=lambda: 1))
    health = iter([{"commit": reported, "launcher_control": "replacement"} if reported else None,
                   {"commit": "previous"}])
    monkeypatch.setattr(runtime_repair, "wait_for_health", lambda *args, **kwargs: next(health))
    monkeypatch.setattr(runtime_repair, "already_running", lambda port: None)
    monkeypatch.setattr(runtime_repair.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(
        stdout=json.dumps({"commit": candidate["commit"], "runtime_ready": True})))
    if reported == candidate["commit"]:
        assert runtime_repair.install(root, row["id"]) == 0
        assert queue.all()[0]["state"] == "done" and queue.all()[0]["result"]["runtime_ready"] is True
        assert (target / "identity").read_text() == "replacement"
        assert len(starts) == 1 and not target.with_name(target.name + ".old").exists()
        return
    assert runtime_repair.install(root, row["id"]) == 1
    assert (target / "identity").read_text() == "previous"
    assert len(starts) == 2
    assert not target.with_name(target.name + ".old").exists()
    retained = list(tmp_path.glob("ml-stack-failed-runtime-*/Poolside.app/identity"))
    assert len(retained) == 1 and retained[0].read_text() == "replacement"
    assert "restored and restarted" in queue.all()[0]["error"]


@pytest.mark.slow
def test_missing_runtime_install_runs_in_background_and_readiness_enables_chat(tmp_path, playwright, monkeypatch):
    from playwright.sync_api import expect
    from test_fleet_ui import Serving

    from ml_stack import agent_dependency
    from ml_stack.fleet.conversations import Conversations
    from ml_stack.fleet.serving import Serving as ModelsServing
    from ml_stack.testing.fakes import FakeLlamaServer, Served

    ready = {"value": False}
    monkeypatch.setattr(agent_dependency, "problem", lambda: "" if ready["value"] else "The agent runtime is not installed.")
    monkeypatch.setattr(runtime_repair, "resume", lambda *args: None)
    model = FakeLlamaServer(Served(model="qwen3.8-test.gguf", pieces=("hello",)))
    served = Serving(tmp_path)
    served.ui.root = tmp_path
    served.ui.setup_jobs = Jobs(tmp_path)
    served.ui.settings.setup_done = True
    served.ui.serving = ModelsServing(tmp_path / "models.json")
    served.ui.serving.register(model.port, ["qwen3.8-test.gguf"])
    served.ui.conversations = Conversations(tmp_path / "chats")
    from ml_stack.scrape.browser import Window, browser

    driver = browser(Window(profile=tmp_path / "browser", channel="chromium"), playwright)
    page = driver.__enter__()
    try:
        person = served.ui.sessions.open("setup")
        page.context.add_cookies([{"name": "ml_stack_ui", "value": person.sid,
                                  "url": f"http://127.0.0.1:{served.port}"}])
        page.goto(f"http://127.0.0.1:{served.port}/ui#chat")
        install = page.get_by_role("button", name="Install agent runtime", exact=True)
        expect(install).to_be_enabled()
        expect(page.locator("chat-view #chat-send")).to_be_disabled()
        install.click()
        expect(install).to_be_disabled()
        expect(page.locator("chat-view #chat-note")).to_contain_text("Waiting to install")
        row = served.ui.setup_jobs.all()[0]
        assert row["state"] == "waiting" and not served.ui.setup_jobs.active()
        page.locator("fleet-nav a[href='#settings']").click()
        expect(page.locator("settings-view")).to_be_visible()
        served.ui.setup_jobs._update(row, state="done", note="Agent runtime ready")
        ready["value"] = True
        page.get_by_role("link", name="Conversations", exact=True).click()
        expect(page.locator("chat-view #chat-send")).to_be_enabled(timeout=10000)
        expect(page.get_by_role("button", name="Install agent runtime", exact=True)).to_have_count(0)
        assert not model.sent_to("/v1/chat/completions")
    finally:
        driver.__exit__(None, None, None)
        served.close()
        model.close()


@pytest.mark.parametrize("client", [
    ({"Origin": "https://foreign.example"}, "127.0.0.1", "127.0.0.1:8770", ""),
    ({}, "192.0.2.5", "127.0.0.1:8770", ""),
    ({"X-ML-Stack-Agent": "worker"}, "127.0.0.1", "127.0.0.1:8770", ""),
    ({}, "127.0.0.1", "foreign.example:8770", ""),
    ({"Authorization": "Bearer untrusted"}, "127.0.0.1", "127.0.0.1:8770", ""),
])
def test_status_refuses_hostile_browser_before_resuming_jobs(tmp_path, monkeypatch, client):
    headers, address, host, cookie = client
    route = Route(tmp_path)
    route.path = "/ui/agent-runtime/status"
    route.headers.update(headers)
    route.client_ip, route.host_header, route.cookie = address, host, cookie
    monkeypatch.setattr(runtime_repair, "resume", lambda *args: pytest.fail("unauthorized request resumed jobs"))
    assert route.public_route()
    assert route.replies[0][0] == 403


def test_central_daemon_protocol_runs_only_the_selected_job(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(runtime_repair, "install", lambda root, job: calls.append((root, job)) or 7)
    monkeypatch.setattr(daemon, "serve", lambda options: pytest.fail("worker protocol started a daemon"))
    assert daemon.run(["--root", str(tmp_path), "--agent-runtime-job", "a" * 32]) == 7
    assert calls == [(tmp_path, "a" * 32)]


@pytest.mark.parametrize("job", ["../job", "a" * 33, "A" * 32, "x" * 32, "a;$(touch unwanted)", ""])
def test_central_daemon_protocol_rejects_hostile_job_ids_before_execution(tmp_path, monkeypatch, job):
    monkeypatch.setattr(runtime_repair, "install", lambda *args: pytest.fail("invalid job executed"))
    monkeypatch.setattr(daemon, "serve", lambda options: pytest.fail("invalid job started daemon"))
    with pytest.raises(SystemExit) as failure:
        daemon.run(["--root", str(tmp_path), "--agent-runtime-job", job])
    assert failure.value.code == 2


def test_preparation_launches_fixed_central_protocol_without_credentials(tmp_path, monkeypatch):
    queue = Jobs(tmp_path)
    row = queue.start(runtime_repair.KIND, {"repair": True}, None)
    ui = SimpleNamespace(root=tmp_path, peer_port=8770)
    candidate = {"commit": "a" * 40, "sha256": "b" * 64}
    monkeypatch.setattr(runtime_repair.updates, "running_path", lambda: Path("/owned/Poolside.app"))
    monkeypatch.setattr(runtime_repair, "Candidates", lambda root: SimpleNamespace(select=lambda name: candidate))
    monkeypatch.setattr(runtime_repair, "wait_for_health", lambda *args, **kwargs: {"launcher_control": "generation"})
    monkeypatch.setenv("TEST_ACCESS_TOKEN", "private-worker-credential")
    calls = []
    monkeypatch.setattr(runtime_repair, "start_process", lambda argv, **kwargs: calls.append((argv, kwargs)) or SimpleNamespace(pid=123))
    monkeypatch.setattr(runtime_repair, "started_at", lambda pid: 11)
    runtime_repair._prepare(ui, queue)
    argv, options = calls[0]
    assert argv == [runtime_repair.sys.executable, "-m", "ml_stack.cli.daemon", "--root", str(tmp_path), "--agent-runtime-job", row["id"]]
    assert "TEST_ACCESS_TOKEN" not in options["env"] and "shell" not in options
    assert queue.all()[0]["worker_born"] == 11


def test_failed_candidate_never_launches_worker(tmp_path, monkeypatch):
    queue = Jobs(tmp_path)
    row = queue.start(runtime_repair.KIND, {"repair": True}, None)
    ui = SimpleNamespace(root=tmp_path, peer_port=8770)
    monkeypatch.setattr(runtime_repair.updates, "running_path", lambda: Path("/owned/Poolside.app"))
    def reject(name):
        raise ValueError("Registered candidate artifact has changed.")
    monkeypatch.setattr(runtime_repair, "Candidates", lambda root: SimpleNamespace(select=reject))
    monkeypatch.setattr(runtime_repair, "start_process", lambda *args, **kwargs: pytest.fail("changed candidate spawned worker"))
    runtime_repair._prepare(ui, queue)
    saved = next(item for item in queue.all() if item["id"] == row["id"])
    assert saved["state"] == "failed" and "changed" in saved["error"]


def test_replacement_launch_uses_literal_paths_and_secret_free_environment(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setenv("TEST_ACCESS_TOKEN", "private-replacement-credential")
    monkeypatch.setattr(runtime_repair, "start_process", lambda argv, **kwargs: calls.append((argv, kwargs)))
    target, root = tmp_path / "Poolside.app", tmp_path / "root;$(not-a-command)"
    runtime_repair._start(target, root, 8770)
    argv, options = calls[0]
    assert argv == [str(target / "Contents/MacOS/ml-stack-headless"), "--port", "8770", "--root", str(root), "--no-browser"]
    assert "TEST_ACCESS_TOKEN" not in options["env"] and "shell" not in options


@pytest.mark.slow
def test_failed_runtime_job_survives_reload_and_retries_through_actual_api(tmp_path, playwright, monkeypatch):
    from playwright.sync_api import expect
    from test_fleet_ui import Serving

    from ml_stack import agent_dependency
    from ml_stack.fleet.conversations import Conversations
    from ml_stack.fleet.serving import Serving as ModelsServing
    from ml_stack.scrape.browser import Window, browser
    from ml_stack.testing.fakes import FakeLlamaServer, Served

    monkeypatch.setattr(agent_dependency, "problem", lambda: "The agent runtime is not installed.")
    monkeypatch.setattr(runtime_repair, "resume", lambda *args: None)
    model = FakeLlamaServer(Served(model="qwen3.8-test.gguf", pieces=("hello",)))
    served = Serving(tmp_path)
    served.ui.root = tmp_path
    served.ui.setup_jobs = Jobs(tmp_path)
    served.ui.settings.setup_done = True
    served.ui.serving = ModelsServing(tmp_path / "models.json")
    served.ui.serving.register(model.port, ["qwen3.8-test.gguf"])
    served.ui.conversations = Conversations(tmp_path / "chats")
    try:
        with browser(Window(profile=tmp_path / "retry-browser", channel="chromium"), playwright) as page:
            person = served.ui.sessions.open("setup")
            page.context.add_cookies([{"name": "ml_stack_ui", "value": person.sid,
                                      "url": f"http://127.0.0.1:{served.port}"}])
            page.goto(f"http://127.0.0.1:{served.port}/ui/#chat", wait_until="domcontentloaded")
            page.get_by_role("button", name="Install agent runtime", exact=True).click()
            expect(page.locator("chat-view #chat-note")).to_contain_text("Waiting to install")
            row = served.ui.setup_jobs.all()[0]
            served.ui.setup_jobs._update(row, state="failed", error="Reviewed artifact unavailable", note="Installation failed")
            retry = page.get_by_role("button", name="Retry installation", exact=True)
            expect(retry).to_be_enabled(timeout=10000)
            expect(page.locator("chat-view #chat-note")).to_contain_text("Reviewed artifact unavailable")
            page.reload(wait_until="domcontentloaded")
            expect(retry).to_be_enabled()
            expect(page.locator("chat-view #chat-note")).to_contain_text("Reviewed artifact unavailable")
            retry.click()
            expect(retry).to_have_count(0)
            expect(page.get_by_role("button", name="Install agent runtime", exact=True)).to_be_disabled()
            rows = served.ui.setup_jobs.all()
            assert len(rows) == 2 and rows[0]["state"] == "failed" and rows[1]["state"] == "waiting"
            assert rows[1]["request"] == {"repair": False}
    finally:
        served.close()
        model.close()
