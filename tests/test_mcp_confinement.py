"""What a tool call may reach through ``bench_run``, ``conversation_compact`` and ``speech_transcribe``."""

from __future__ import annotations

import json

import pytest

from ml_stack import mcp


@pytest.mark.parametrize("argv", [
    ["sweep", "--binary", "/tmp/evil"],
    ["sweep", "--binary=/tmp/evil"],
    ["sweep", "--bin", "/tmp/evil"],
    ["sweep", "--store", "/etc"],
    ["sweep", "--kept", "/tmp/k"],
    ["sweep", "--serve-arg", "--api-key"],
    ["sweep", "--on", "x=http://169.254.169.254"],
    ["sweep", "--serve", "--binary"],
    ["report", "--md", "/tmp/x"],
    ["--binary", "/tmp/evil"],
    [],
])
def test_bench_run_refuses_what_is_not_allowed(argv, monkeypatch):
    from ml_stack.bench import underway

    monkeypatch.setattr(underway, "detach", lambda a: pytest.fail("detached " + repr(a)))
    with pytest.raises(ValueError):
        mcp.bench_run(argv)


def test_bench_run_passes_an_allowed_line_through():
    assert mcp._checked_bench_argv(["sweep", "--serve", "m.gguf", "--smoke", "--sample=3"]) == [
        "sweep", "--serve", "m.gguf", "--smoke", "--sample=3"]


def test_compact_refuses_a_file_outside_the_home_and_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path / "..")
    outside = tmp_path.parent.parent / "elsewhere.json"
    with pytest.raises(ValueError):
        mcp.conversation_compact(str(outside), 100)


def test_compact_reads_a_file_in_the_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "chat.json").write_text(json.dumps([{"role": "user", "content": "hi"}]))
    assert mcp.conversation_compact("chat.json", 1000)["strategy"] == "none"


def test_compact_takes_a_server_only_when_the_broker_holds_it(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "chat.json").write_text("[]")
    with pytest.raises(ValueError):
        mcp.conversation_compact("chat.json", 1000, url="http://attacker.example:9")


def test_transcribe_refuses_a_file_outside_the_home_and_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError):
        mcp.speech_transcribe("/etc/hosts")
