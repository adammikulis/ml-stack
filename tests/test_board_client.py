"""The Python client of the node: its socket protocol, starting a dead node, the board a directory belongs to."""

from __future__ import annotations

import os
import signal
import stat
import subprocess
import threading
import time
from pathlib import Path

import pytest
from node_kit import BOARD

from ml_stack.board import client as board_client, credentials, node_start, place, session

pytest_plugins = ["node_kit"]


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def git(*argv: str, cwd: Path) -> None:
    subprocess.run(["git", "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
                    "-c", "commit.gpgsign=false", *argv], cwd=cwd, check=True, capture_output=True)


def test_a_killed_node_is_started_again_and_keeps_every_token(workspace_node):
    who = workspace_node.member("kept")
    first = workspace_node.client.call("status")["pid"]
    os.kill(first, signal.SIGKILL)
    deadline = time.monotonic() + 5
    while alive(first) and time.monotonic() < deadline:
        time.sleep(0.01)
    after = workspace_node.client.call("whoami", BOARD, who.token)
    assert after["name"] == who.name
    assert workspace_node.client.call("status")["pid"] != first


def test_clients_that_find_the_socket_dead_at_once_start_exactly_one_node(workspace_node):
    workspace_node.stop()
    pids, errors = [], []

    def ask():
        try:
            pids.append(board_client.Client(workspace_node.state).call("hello")["pid"])
        except Exception as err:  # noqa: BLE001 - the test reports whatever a client hit
            errors.append(err)

    threads = [threading.Thread(target=ask) for _ in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(30)
    assert not errors and len(set(pids)) == 1 and len(pids) == 6
    listed = subprocess.run(["pgrep", "-f", f"poolside-node run --state {workspace_node.state}"],
                            capture_output=True, text=True, check=False).stdout.split()
    assert len(listed) == 1


def test_no_binary_is_said_plainly_and_a_refusal_carries_the_nodes_code(workspace_node, monkeypatch):
    with pytest.raises(board_client.Denied) as denied:
        workspace_node.client.call("whoami", BOARD, "not-a-token")
    assert denied.value.code == "denied"
    with pytest.raises(board_client.Invalid):
        workspace_node.client.call("no_such_method")
    with pytest.raises(board_client.Quota):
        workspace_node.client.call("post", BOARD, "x", blob="x" * (2 * 1024 * 1024))
    workspace_node.stop()
    monkeypatch.setenv(node_start.BIN_ENV, str(workspace_node.state / "missing"))
    with pytest.raises(node_start.NodeUnavailable, match="no poolside-node binary"):
        workspace_node.client.call("hello")


def test_a_token_is_kept_in_a_private_file_and_a_loose_one_is_refused(workspace_node):
    who = workspace_node.member("keeper")
    path = workspace_node.state / "client" / BOARD / f"{who.name}.token"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600 and stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    assert credentials.load(workspace_node.state, BOARD, who.name) == who.token
    path.chmod(0o640)
    with pytest.raises(board_client.Denied, match="lets others read it"):
        credentials.load(workspace_node.state, BOARD, who.name)
    with pytest.raises(board_client.Denied, match="missing"):
        credentials.load(workspace_node.state, BOARD, "claude-000000")


def test_connect_finds_the_token_by_agent_name_environment_or_file_and_refuses_none(workspace_node, monkeypatch, tmp_path):
    who = workspace_node.member("finder")
    assert session.connect(agent=who.name).whoami().name == who.name
    monkeypatch.setenv(session.AGENT_ENV, who.name)
    assert session.connect().name == who.name
    monkeypatch.delenv(session.AGENT_ENV)
    monkeypatch.setenv(session.TOKEN_ENV, who.token)
    assert session.connect().name == who.name
    monkeypatch.delenv(session.TOKEN_ENV)
    file = tmp_path / "token"
    file.write_text(who.token + "\n")
    assert session.connect(token_file=str(file)).name == who.name
    with pytest.raises(board_client.Denied, match="no token"):
        session.connect()


def test_a_repository_and_all_its_worktrees_are_one_board_and_a_stranger_directory_has_none(workspace_node, monkeypatch, tmp_path):
    monkeypatch.delenv("ML_STACK_BOARD")
    repo = tmp_path / "shop"
    repo.mkdir()
    git("init", "-q", "-b", "main", cwd=repo)
    git("commit", "-q", "--allow-empty", "-m", "start", cwd=repo)
    linked = tmp_path / "shop-feature"
    git("worktree", "add", "-q", str(linked), "-b", "feature", cwd=repo)
    (repo / "src").mkdir()
    first = place.resolve(workspace_node.client, repo)
    assert first.startswith("shop-") and place.resolve(workspace_node.client, linked) == first
    assert place.resolve(workspace_node.client, repo / "src") == first
    other = tmp_path / "other"
    other.mkdir()
    git("init", "-q", "-b", "main", cwd=other)
    assert place.resolve(workspace_node.client, other) != first
    plain = tmp_path / "plain"
    plain.mkdir()
    with pytest.raises(board_client.Denied, match="not part of a project"):
        place.resolve(workspace_node.client, plain)
    assert place.resolve(workspace_node.client, plain, named="chosen") == "chosen"
    monkeypatch.setenv("ML_STACK_BOARD", "elsewhere")
    assert place.resolve(workspace_node.client, repo) == "elsewhere"


def test_a_registered_session_is_found_again_by_its_native_id_on_the_same_board(workspace_node):
    made = workspace_node.member("native-1")
    assert session.find("claude-code", "native-1") == made.name
    assert session.find("claude-code", "native-2") == ""
    assert session.find("codex", "native-1") == ""
    again = session.register(workspace_node.client, BOARD, session.Native("claude-opus-5-5", "claude-code", "native-1"))
    assert (again.name, again.created) == (made.name, False)


def test_a_session_reads_the_cursor_it_kept_and_each_message_comes_once(workspace_node):
    lead, other = workspace_node.member("lead"), workspace_node.member("other")
    sender, reader = workspace_node.session(lead), workspace_node.session(other)
    sender.post(other.name, "note", "one")
    page = reader.read(reader.cursor("inbox"), inbox=True)
    assert [e.text for e in page.entries] == ["one"]
    reader.keep("inbox", page.cursor)
    assert reader.read(reader.cursor("inbox"), inbox=True).entries == []
    sender.post(other.name, "note", "two")
    assert [e.text for e in reader.read(reader.cursor("inbox"), inbox=True).entries] == ["two"]
    assert reader.cursor("announce") == {}
