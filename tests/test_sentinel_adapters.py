"""The hooks that connect other modules' events to a sentinel."""

from __future__ import annotations

import logging
import subprocess
import sys

import pytest

from ml_stack.sentinel import Mode, Sentinel, State
from ml_stack.sentinel.adapters import (
    GuardLogHandler,
    agent_gate,
    broker_listener,
    note_refusal,
    serve_hooks,
)
from ml_stack.serve.process import kill_process_tree, pid_exists


@pytest.fixture
def node(tmp_path):
    return Sentinel(tmp_path / "s", mode=Mode.ENFORCE, roots=[tmp_path])


def test_guard_warnings_become_findings_for_the_session(node):
    logger = logging.getLogger("ml_stack.guard")
    handler = GuardLogHandler(node, session=lambda: "s1")
    logger.addHandler(handler)
    try:
        for _ in range(5):
            logger.warning("%s: %s at %s", "untrusted", "Deny", "after_tool_call")
        logger.warning("%s: %s at %s", "untrusted", "Confirm", "before_tool_call")
        logger.debug("%s: %s at %s", "x", "Rewrite", "after_tool_call")
    finally:
        logger.removeHandler(handler)
    kinds = [e.kind for e in node.bus.recent(kind="guard.")]
    assert kinds.count("guard.denied") == 5 and "guard.repeated_denials" in kinds
    assert node.session_frozen("s1")


def test_unrelated_log_records_are_ignored(node):
    handler = GuardLogHandler(node)
    record = logging.LogRecord("ml_stack.guard", logging.WARNING, __file__, 1,
                               "plain message %s", ("only one",), None)
    handler.emit(record)
    assert node.bus.recent(kind="guard.") == []


def test_broker_events_are_logged_and_count_toward_the_caller(node):
    on = broker_listener(node)
    for _ in range(130):
        on("request", {"caller": "agent-7", "model": "m"})
    assert len(node.bus.recent(kind="broker.request")) == 130
    assert node.store.state_of("caller", "agent-7") == State.QUARANTINED
    assert not node.tool_allowed("anything", caller="agent-7")
    assert node.tool_allowed("anything", caller="agent-8")


def test_a_refused_fetch_is_an_event(node):
    note_refusal(node, "169.254.169.254", "metadata address")
    assert node.bus.recent(kind="httpguard.refused")[0].subject == "host:169.254.169.254"


def test_the_agent_gate_refuses_sentinel_verbs_frozen_sessions_and_disabled_tools(node):
    gate = agent_gate(node)
    assert gate("read_file", {"path": "notes.md"}, session="s1") == ""
    assert gate("shell", {"command": "ml-stack security mode off"}, session="s1")
    node.store.quarantine(("tool", "shell"), "abuse", None)
    assert "disabled" in gate("shell", {"command": "ls"}, session="s1")
    node.store.quarantine(("session", "s2"), "steered", None)
    assert "frozen" in gate("read_file", {"path": "a"}, session="s2")


def test_quarantining_a_server_or_its_model_stops_that_process_and_no_other(node, tmp_path):
    model = tmp_path / "m.gguf"
    model.write_bytes(b"GGUF")
    mine = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
    bystander = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
    records = {8101: {"pid": mine.pid, "model": str(model)}}
    stopped = []

    def stop(port: int) -> None:
        stopped.append(port)
        kill_process_tree(records.pop(port)["pid"], grace_s=1.0)

    serve_hooks(node, lambda: dict(records), stop)
    try:
        node.store.quarantine(("server", "port:9999"), "not ours", None)
        assert stopped == [] and pid_exists(mine.pid)
        node.store.quarantine(("model", str(model)), "hash mismatch", None)
        assert stopped == [8101]
        mine.wait(timeout=10)
        assert pid_exists(bystander.pid)
        records[8102] = {"pid": bystander.pid, "model": "/elsewhere/other.gguf"}
        node.store.quarantine(("server", "port:8102"), "binary changed", None)
        bystander.wait(timeout=10)
        assert stopped == [8101, 8102]
    finally:
        for proc in (mine, bystander):
            if proc.poll() is None:
                proc.kill()
                proc.wait()
