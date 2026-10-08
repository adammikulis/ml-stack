"""A process on this machine that forges the page's request headers cannot become the person.

Every request goes over a real loopback socket to a real daemon UI with a real project
workspace; the only credential that opens the first session is a launch ticket the daemon
issued against its launch secret.
"""

from __future__ import annotations

import http.client
import json
import re
import sys
import time

import pytest
from launch_support import browser, hand_typed, redeem, sign_in, ticket
from test_fleet_ui import WORDS, Serving
from test_redteam_human_floor import run_command

from ml_stack import home
from ml_stack.fleet import session as session_module
from ml_stack.fleet.launch_secret import HEADER, LaunchSecret
from ml_stack.fleet.projects import ProjectRegistry, identity
from ml_stack.net import git
from ml_stack.workspace.identity import HUMAN
from ml_stack.workspace.remote_host import WorkspaceHost

pytestmark = pytest.mark.redteam


@pytest.fixture
def window(tmp_path, monkeypatch):
    monkeypatch.setenv("ML_STACK_WORKSPACE_HOME", str(tmp_path / "machine-workspace"))
    checkout = tmp_path / "experiment"
    checkout.mkdir()
    git.run(["init"], cwd=checkout)
    git.run(["remote", "add", "origin", "https://code.example.invalid/team/person.git"], cwd=checkout)
    registry = ProjectRegistry(tmp_path / "registry", "fixture-device", (checkout,), "http://127.0.0.1:8770")
    project = identity(checkout)
    host = WorkspaceHost(registry)
    registry.claim_authority(project)
    server = Serving(tmp_path / "ui", secure=False)
    server.ui.settings.setup_done = True
    server.ui.projects, server.ui.workspaces = registry, host
    server.ui.audit = lambda event, **fields: server.rows.append({"event": event, **fields}) or host.audit(event, **fields)
    server.project = project
    server.workspace = host.workspace(project)
    try:
        yield server
    finally:
        server.close()


def people(window) -> list[str]:
    return [name for name, entry in window.workspace.registry._load().items() if entry.get("role") == HUMAN]


def connect(window, cookie="", headers=None):
    return window.call(f"/ui/projects/{window.project}/board/connect", method="POST", body={},
                       cookie=cookie, headers={**browser(window), **(headers or {})})


def raw_post(window, path, headers, body=b"{}"):
    conn = http.client.HTTPConnection("127.0.0.1", window.port, timeout=10)
    try:
        conn.request("POST", path, body=body, headers={"Content-Type": "application/json", "X-ML-Stack-UI": "1",
                                                       "Content-Length": str(len(body)), **headers})
        reply = conn.getresponse()
        return reply.status, reply.getheader("Set-Cookie"), reply.read()
    finally:
        conn.close()


FORGED = {"Host": "127.0.0.1", "Sec-Fetch-Site": "same-origin"}


def test_forged_page_headers_without_a_ticket_open_no_session_and_make_no_person(window):
    forged = {**FORGED, "Origin": f"http://127.0.0.1:{window.port}", "Host": f"127.0.0.1:{window.port}"}
    status, cookie, _ = raw_post(window, "/ui/setup/initial", forged,
                                 json.dumps({"name": "quillhaven", "cluster_mode": "dev"}).encode())
    assert status == 403 and cookie is None
    status, cookie, _ = raw_post(window, "/ui/setup/local-session", forged)
    assert status == 404 and cookie is None
    status, cookie, _ = raw_post(window, "/ui/session", forged)
    assert status == 401 and cookie is None
    status, cookie, _ = raw_post(window, f"/ui/projects/{window.project}/board/connect", forged)
    assert status == 403 and cookie is None
    assert len(window.ui.sessions) == 0
    assert people(window) == []


def test_a_wrong_ticket_opens_nothing(window):
    status, _, headers = redeem(window, "not-a-ticket")
    assert status == 401 and "Set-Cookie" not in headers
    assert connect(window)[0] == 403
    assert people(window) == []


def test_a_ticket_opens_one_session_and_only_once(window):
    status, issued, _ = ticket(window)
    assert status == 200
    status, _, headers = redeem(window, issued["ticket"])
    assert status == 200
    cookie = headers["Set-Cookie"].split(";", 1)[0]
    assert connect(window, cookie)[0] == 200
    assert people(window) == ["local-person"]
    status, _, headers = redeem(window, issued["ticket"])
    assert status == 401 and "Set-Cookie" not in headers


def test_an_expired_ticket_opens_nothing(window, monkeypatch):
    monkeypatch.setattr(session_module, "TICKET_TTL_S", 0.05)
    status, issued, _ = ticket(window)
    assert status == 200
    time.sleep(0.15)
    status, _, headers = redeem(window, issued["ticket"])
    assert status == 401 and "Set-Cookie" not in headers


def test_a_session_without_a_credentialed_origin_cannot_connect_or_act_as_the_person(window):
    bare = window.ui.sessions.open("setup")
    cookie = window.ui.sessions.cookie_header(bare).split(";", 1)[0]
    assert connect(window, cookie)[0] == 403
    assert people(window) == []
    assert window.call(f"/ui/projects/{window.project}/board/agents", cookie=cookie)[0] == 403


def test_a_cluster_created_with_no_credential_leaves_a_session_that_cannot_connect(window):
    status, _, headers = window.call("/ui/setup/join", method="POST",
                                     body={"mode": "create", "passphrase": WORDS, "group": "home"})
    assert status == 200
    cookie = headers["Set-Cookie"].split(";", 1)[0]
    assert connect(window, cookie)[0] == 403
    assert people(window) == []


def test_a_passphrase_session_connects(window):
    window.call("/ui/setup/join", method="POST", body={"mode": "create", "passphrase": WORDS, "group": "home"})
    status, _, headers = window.call("/ui/session", method="POST", body={"passphrase": WORDS, "group": "home"})
    assert status == 200
    assert connect(window, headers["Set-Cookie"].split(";", 1)[0])[0] == 200
    assert people(window) == ["local-person"]


def test_the_credentialed_connect_is_recorded_in_the_project_audit_chain(window):
    cookie = sign_in(window)
    assert connect(window, cookie)[0] == 200
    rows = [row for row in window.workspace.audit_log.rows() if row.get("event") == "person.connect"]
    assert rows and rows[-1]["origin"] == "launch-ticket" and rows[-1]["project"] == window.project


def test_a_ticket_is_asked_for_only_by_the_machine_itself_with_its_secret(window):
    secret = window.ui.launch.value
    assert ticket(window, "wrong-secret")[0] == 403
    assert window.call("/ui/launch/ticket", method="POST")[0] == 403
    assert window.call("/ui/launch/ticket", method="POST", headers={HEADER: secret, "Origin": "http://evil.example"})[0] == 403
    assert window.call("/ui/launch/ticket", method="POST", headers={HEADER: secret, "Sec-Fetch-Site": "same-origin"})[0] == 403
    assert window.call("/ui/launch/ticket", method="POST", headers={HEADER: secret, "Host": "rebound.example"})[0] == 403
    assert window.call("/ui/launch/ticket", method="GET", headers={HEADER: secret})[0] == 405
    assert ticket(window)[0] == 200


def test_guessing_the_secret_is_held_off_even_for_the_right_secret(window):
    for _ in range(4):
        assert ticket(window, "guess")[0] == 403
    status, body, _ = ticket(window)
    assert status == 429 and "ticket" not in body


def test_guessing_tickets_is_held_off(window):
    for index in range(4):
        assert redeem(window, f"guess-{index}")[0] == 401
    issued = ticket(window)
    assert issued[0] == 200
    assert redeem(window, issued[1]["ticket"])[0] == 429


def test_only_a_few_tickets_wait_at_once(window):
    for _ in range(8):
        assert ticket(window)[0] == 200
    assert ticket(window)[0] == 429


def test_an_uncredentialed_session_cannot_hand_out_a_ticket(window):
    bare = window.ui.sessions.open("setup")
    cookie = window.ui.sessions.cookie_header(bare).split(";", 1)[0]
    status, body, _ = window.call("/ui/launch-ticket", method="POST", cookie=cookie)
    assert status == 403 and "ticket" not in body
    credentialed = sign_in(window)
    assert window.call("/ui/launch-ticket", method="POST", cookie=credentialed)[0] == 200


def test_the_audit_rows_name_every_mint_and_refusal_and_hold_no_ticket_or_secret(window):
    secret = window.ui.launch.value
    ticket(window, "wrong")
    redeem(window, "wrong-ticket")
    _, issued, _ = ticket(window)
    redeem(window, issued["ticket"])
    window.call("/ui/setup/initial", method="POST", headers=browser(window),
                body={"name": "quillhaven", "cluster_mode": "dev"})
    events = {(row["event"], row.get("reason") or row.get("by") or row.get("origin")) for row in window.rows}
    assert {("launch.refused", "secret"), ("session.refused", "ticket"), ("launch.ticket", "launch-secret"),
            ("session.open", "launch-ticket"), ("session.refused", "setup-without-credential")} <= events
    text = json.dumps(window.rows)
    assert secret not in text and issued["ticket"] not in text and "wrong-ticket" not in text


def test_a_token_session_is_not_a_person(window):
    bare = window.ui.sessions.open("token", "token")
    cookie = window.ui.sessions.cookie_header(bare).split(";", 1)[0]
    assert connect(window, cookie)[0] == 403
    assert people(window) == []


@pytest.mark.slow
def test_a_page_opened_by_hand_says_how_to_open_it_and_a_launched_page_does_not(window, playwright):
    from playwright.sync_api import expect

    base = f"http://127.0.0.1:{window.port}/ui/"
    with playwright.chromium.launch(headless=True) as chromium:
        page = chromium.new_page()
        with hand_typed():
            page.goto(base)
        expect(page.locator("#launch-needed")).to_be_visible()
        expect(page.locator("#launch-needed-command")).to_have_text("ml-stack peers open")
        launched = chromium.new_page()
        launched.goto(f"{base}?launch_ticket={ticket(window)[1]['ticket']}")
        launched.wait_for_selector("#app:not([hidden])")
        expect(launched.locator("#launch-needed")).to_be_hidden()
        assert "launch_ticket" not in launched.url


@pytest.fixture
def machine(window, tmp_path, monkeypatch):
    """The window's daemon, recorded under the state root a child process of this test will read."""
    monkeypatch.setenv("ML_STACK_HOME", str(tmp_path / "home"))
    window.ui.launch = LaunchSecret(home.state("traind"), window.port)
    return window


def launch_tickets(machine) -> list[dict]:
    return [row for row in machine.rows if row["event"] == "launch.ticket"]


@pytest.mark.skipif(sys.platform == "win32", reason="requires a POSIX pseudoterminal")
def test_the_open_command_hands_a_person_at_a_terminal_a_one_use_address(machine, tmp_path):
    done = run_command(tmp_path, ("ml_stack.fleet.peers", "main", ["open", "--print"]), agent=False, terminal=True)
    assert done.returncode == 0, done.stdout
    found = re.search(rf"http://127\.0\.0\.1:{machine.port}/ui/\?launch_ticket=([\w-]+)", done.stdout)
    assert found, done.stdout
    assert redeem(machine, found.group(1))[0] == 200
    assert redeem(machine, found.group(1))[0] == 401


@pytest.mark.parametrize("agent,terminal", [(True, True), (False, False), (True, False)])
def test_the_open_command_hands_an_agent_or_a_pipe_nothing(machine, tmp_path, agent, terminal):
    if terminal and sys.platform == "win32":
        pytest.skip("requires a POSIX pseudoterminal")
    done = run_command(tmp_path, ("ml_stack.fleet.peers", "main", ["open", "--print"]), agent=agent, terminal=terminal)
    assert done.returncode != 0
    assert "launch_ticket" not in done.stdout + (done.stderr or "")
    assert launch_tickets(machine) == []


GATES = ("peers open", "signed-in local person")


def gated(reply) -> bool:
    """Whether the daemon refused the request for want of a credentialed session."""
    status, body, _ = reply
    return status == 403 and any(gate in str(body.get("error", "")) for gate in GATES)


@pytest.mark.parametrize("path", ["/ui/board/agents", "/ui/tasks", "/ui/coordination", "/ui/history/events",
                                  "/ui/projects/PROJECT/board/agents"])
def test_the_routes_that_act_for_the_person_answer_only_a_credentialed_session(window, path):
    path = path.replace("PROJECT", window.project)
    bare = window.ui.sessions.open("setup")
    uncredentialed = window.ui.sessions.cookie_header(bare).split(";", 1)[0]
    assert gated(window.call(path, headers=browser(window)))
    assert gated(window.call(path, cookie=uncredentialed, headers=browser(window)))
    assert not gated(window.call(path, cookie=sign_in(window), headers=browser(window)))


def test_credential_changes_need_a_credentialed_session(window):
    from ml_stack import credentials

    body = {"name": "HF_TOKEN", "value": "isolated-secret"}
    bare = window.ui.sessions.open("setup")
    uncredentialed = window.ui.sessions.cookie_header(bare).split(";", 1)[0]
    for cookie in ("", uncredentialed):
        status, _, _ = window.call("/ui/credentials", method="POST", body=body, cookie=cookie, headers=browser(window))
        assert status == 403
        assert credentials.get("HF_TOKEN") is None
    status, _, _ = window.call("/ui/credentials", method="POST", body=body, cookie=sign_in(window), headers=browser(window))
    assert status == 200 and credentials.get("HF_TOKEN") == "isolated-secret"


def test_the_wiring_limit_refuses_an_uncredentialed_session_before_asking_anyone(window):
    bare = window.ui.sessions.open("setup")
    cookie = window.ui.sessions.cookie_header(bare).split(";", 1)[0]
    status, answer, _ = window.call("/ui/room/reset", method="POST", body={}, cookie=cookie, headers=browser(window))
    assert status == 403 and "not opened with a credential" in answer["error"]
