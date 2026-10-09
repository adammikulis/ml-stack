"""Text that becomes a command line: no process is started through a shell, the one shell line
is shown and confirmed first, and hostile argument text stays one argv item."""

from __future__ import annotations

import ast
import os
import subprocess
import types
from pathlib import Path

import pytest

from ml_stack import checks, jobs, mcp

SRC = Path(__file__).resolve().parent.parent / "src" / "ml_stack"
HOSTILE = ["x; touch /tmp/pwned", "$(id)", "`id`", "a | b > c", "--exec=/bin/sh", "a\nb",
           "'; rm -rf ~ #", "\u202ehidden"]
SHELLS = {"os.system", "os.popen", "asyncio.create_subprocess_shell", "subprocess.getoutput",
          "subprocess.getstatusoutput"}


def dotted(node: ast.AST) -> str:
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    return ".".join([node.id if isinstance(node, ast.Name) else "?", *reversed(parts)])


def shell_sites() -> list[str]:
    sites = []
    for path in sorted(SRC.rglob("*.py")):
        if "/testing/" in path.as_posix():
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.Call):
                continue
            keyed = any(k.arg == "shell" and not (isinstance(k.value, ast.Constant)
                                                  and k.value.value is False)
                        for k in node.keywords)
            if keyed or dotted(node.func) in SHELLS:
                sites.append(f"{path.relative_to(SRC).as_posix()}:{node.lineno}")
    return sites


def test_no_process_is_started_through_a_shell_except_the_confirmed_fix_line():
    assert shell_sites() == []


def finding(marker: Path) -> checks.Finding:
    return checks.Finding(name="x", good=False, said="broken", fix=["touch", str(marker)])


def test_the_only_shell_line_is_one_the_person_was_shown_and_confirmed(tmp_path, monkeypatch):
    marker = tmp_path / "ran"
    monkeypatch.setattr("sys.stdin", types.SimpleNamespace(isatty=lambda: False))
    checks.ask([finding(marker)])
    assert not marker.exists()
    monkeypatch.setattr("sys.stdin", types.SimpleNamespace(isatty=lambda: True))
    for refusal in ("", "n", "no", "yes please", "y; rm -rf ~"):
        monkeypatch.setattr("builtins.input", lambda _prompt="", said=refusal: said)
        checks.ask([finding(marker)])
        assert not marker.exists(), refusal
    monkeypatch.setattr("builtins.input", lambda _prompt="": "y")
    checks.ask([finding(marker)])
    assert marker.exists()


class Recorded:
    def __init__(self) -> None:
        self.calls: list[tuple[object, dict]] = []

    def __call__(self, command, *args, **kwargs):
        self.calls.append((command, kwargs))
        return types.SimpleNamespace(pid=os.getpid() + 1, poll=lambda: 0)


@pytest.fixture
def recorded(monkeypatch):
    spy = Recorded()
    monkeypatch.setattr(subprocess, "Popen", spy)
    return spy


@pytest.mark.parametrize("text", HOSTILE)
def test_a_hostile_argument_is_one_argv_item_and_never_reaches_a_shell(tmp_path, recorded, text):
    jobs.detach("ml_stack.serve.cli", ["up", text], log=tmp_path / "a.log")
    if text.startswith("-"):
        assert mcp.call("serve_up", {"model": text, "extra": [text]})["isError"]
        assert mcp.call("models_fetch", {"reference": text})["isError"]
        assert len(recorded.calls) == 1
        return
    mcp.serve_up(text, extra=[text])
    seen_before = len(recorded.calls)
    mcp.models_fetch(text)
    assert seen_before == 2 and len(recorded.calls) == 3
    for command, kwargs in recorded.calls:
        assert isinstance(command, list) and text in command
        assert not kwargs.get("shell")


@pytest.mark.parametrize("text", ["x" * 100_000, "a\x00b"], ids=["oversized-name", "nul-name"])
def test_a_name_no_file_system_can_hold_is_an_error_and_starts_nothing(recorded, text):
    for tool, arguments in (("models_fetch", {"reference": text}), ("serve_up", {"model": text})):
        assert mcp.call(tool, arguments)["isError"], (tool, text[:10])
    assert recorded.calls == []
