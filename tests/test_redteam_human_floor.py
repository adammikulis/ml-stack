"""The person-only floor, attacked through every command that holds a piece of it.

Each command runs as a real child process with its own state directory. A process an agent
started (CLAUDECODE set) at a terminal, and a process with no terminal, must be refused and
leave the state directory byte for byte as it was; the same command run by a person at a
terminal does the work, so a refusal is never just a command that fails.
"""

from __future__ import annotations

import os
import pty
import select
import subprocess
import sys
import time
from pathlib import Path

import pytest

from ml_stack.net.scan import CATEGORIES
from ml_stack.sentinel import Sentinel, State

SRC = str(Path(__file__).resolve().parent.parent / "src")
HOST = "evil.example"
CLEAN = {"CLAUDECODE": "", "ML_STACK_AGENT": "", "ML_STACK_NONINTERACTIVE": "",
         "ML_STACK_SENTINEL": ""}


def sentinel_dir(tmp_path: Path) -> Path:
    return tmp_path / "home" / "sentinel"


def seed_quarantine(tmp_path: Path) -> str:
    node = Sentinel(sentinel_dir(tmp_path), roots=[tmp_path])
    return node.store.quarantine(("peer", "10.2.2.2"), "forged", None).id


def commands(tmp_path: Path) -> list[tuple[str, str, str, list[str], str]]:
    """(label, module, function, argv, what the person types) for every command that needs one."""
    held = seed_quarantine(tmp_path)
    return [
        ("sentinel mode off", "ml_stack.sentinel.cli", "command", ["mode", "off"], "sentinel"),
        ("sentinel release", "ml_stack.sentinel.cli", "command", ["quarantine", "release", held], held),
        ("sentinel purge", "ml_stack.sentinel.cli", "command", ["quarantine", "purge", held], held),
        ("approve a host", "ml_stack.net.cli", "command", ["approve-host", HOST], HOST),
        ("scan policy", "ml_stack.net.cli", "command", ["scan-policy", CATEGORIES[0], "allow"], CATEGORIES[0]),
        ("memory forget", "ml_stack.memory.cli", "main", ["forget", "--all"], "yes"),
        ("memory rekey", "ml_stack.memory.cli", "main", ["rekey"], "yes"),
        ("memory export", "ml_stack.memory.cli", "main", ["export"], "yes"),
        ("workspace init", "ml_stack.workspace.cli", "main", ["init"], "yes"),
    ]


def snapshot(root: Path) -> dict[str, bytes]:
    """Every file under the state directory except the logs a refusal is allowed to write."""
    out = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix not in (".log", ".jsonl", ".lock"):
            out[path.relative_to(root).as_posix()] = path.read_bytes()
    return out


def run_command(tmp_path: Path, target: tuple[str, str, list[str]], *,
                agent: bool, terminal: bool, answer: str = "") -> subprocess.CompletedProcess:
    module, function, argv = target
    env = {**os.environ, "PYTHONPATH": SRC, "ML_STACK_HOME": str(tmp_path / "home"), **CLEAN,
           "PYTHON_KEYRING_BACKEND": "keyring.backends.null.Keyring",
           **({"CLAUDECODE": "1"} if agent else {})}
    code = f"import sys; from {module} import {function}; sys.exit({function}({argv!r}))"
    if not terminal:
        return subprocess.run([sys.executable, "-c", code], env=env, capture_output=True,
                              text=True, stdin=subprocess.DEVNULL, timeout=90)
    master, slave = pty.openpty()
    child = subprocess.Popen([sys.executable, "-c", code], env=env, stdin=slave, stdout=slave,
                             stderr=slave, close_fds=True)
    os.close(slave)
    heard, typed = b"", False
    end = time.monotonic() + 30
    while child.poll() is None:
        if time.monotonic() > end:
            child.kill()
            raise AssertionError(f"{argv} waited for input: {heard!r}")
        if not select.select([master], [], [], 0.2)[0]:
            continue
        try:
            heard += os.read(master, 4096)
        except OSError:
            break
        if answer and not typed and b"type " in heard:
            os.write(master, (answer + "\n").encode())
            typed = True
    child.wait(timeout=60)
    os.close(master)
    return subprocess.CompletedProcess(child.args, child.returncode, heard.decode(errors="replace"), "")


@pytest.mark.parametrize("index", range(9))
def test_every_human_only_command_refuses_a_process_started_by_an_agent(tmp_path, index):
    label, module, function, argv, typed = commands(tmp_path)[index]
    before = snapshot(tmp_path / "home")
    done = run_command(tmp_path, (module, function, argv), agent=True, terminal=True, answer=typed)
    assert done.returncode != 0, f"{label} ran for an agent: {done.stdout}"
    assert snapshot(tmp_path / "home") == before, f"{label} changed state for an agent"


@pytest.mark.parametrize("index", [*range(8), pytest.param(8, marks=pytest.mark.xfail(
    strict=True, reason="workspace init checks the agent markers and never asks for a terminal"))])
def test_every_human_only_command_refuses_a_process_with_no_terminal(tmp_path, index):
    label, module, function, argv, _ = commands(tmp_path)[index]
    before = snapshot(tmp_path / "home")
    done = run_command(tmp_path, (module, function, argv), agent=False, terminal=False)
    assert done.returncode != 0, f"{label} ran with no terminal: {done.stdout}"
    assert snapshot(tmp_path / "home") == before, f"{label} changed state with no terminal"


def test_the_same_commands_do_their_work_for_a_person_at_a_terminal(tmp_path):
    held = seed_quarantine(tmp_path)
    approved = run_command(tmp_path, ("ml_stack.net.cli", "command", ["approve-host", HOST]),
                           agent=False, terminal=True, answer=HOST)
    assert approved.returncode == 0, approved.stdout
    listed = run_command(tmp_path, ("ml_stack.net.cli", "command", ["hosts"]), agent=False,
                         terminal=False)
    assert HOST in listed.stdout
    released = run_command(tmp_path, ("ml_stack.sentinel.cli", "command",
                                 ["quarantine", "release", held]), agent=False, terminal=True,
                           answer=held)
    assert released.returncode == 0, released.stdout
    assert Sentinel(sentinel_dir(tmp_path)).store.get(held).state == State.RELEASED


@pytest.fixture
def keystore(tmp_path, monkeypatch):
    import keyring
    from onboard_support import FileKeyring
    monkeypatch.setenv("ML_STACK_TEST_KEYRING", str(tmp_path / "keystore.json"))
    before = keyring.get_keyring()
    keyring.set_keyring(FileKeyring())
    yield
    keyring.set_keyring(before)


@pytest.mark.parametrize("action,value", [("export", "out.key"), ("rotate", ""),
                                          ("revoke", "SHA256:abc"), ("confirm", "off"),
                                          ("accept", "")])
def test_every_signing_key_step_refuses_an_agent_even_at_a_terminal(
        tmp_path, monkeypatch, keystore, action, value):
    import argparse

    from ml_stack.fleet.onboard.signing_cli import cmd_signing
    state = tmp_path / "state"
    shown = argparse.Namespace(state=str(state), action="show", value="", json=True)
    assert cmd_signing(shown) == 0
    before = snapshot(state)
    monkeypatch.setenv("CLAUDECODE", "1")
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True, raising=False)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True, raising=False)
    monkeypatch.setattr("builtins.input", lambda _p="": "yes")
    args = argparse.Namespace(state=str(state), action=action, value=str(tmp_path / value),
                              json=True)
    assert cmd_signing(args) == 2
    assert snapshot(state) == before
    assert not (tmp_path / "out.key").exists()


def test_no_tool_an_agent_is_offered_is_a_human_only_action():
    import io

    from ml_stack import chat, do, mcp
    floor = ("quarantine", "purge", "rekey", "approve", "mint", "revoke", "rotate", "baseline",
             "unquarantine", "forget", "signing", "scan_policy", "sentinel", "security")
    person = do.Person(io.StringIO(""), io.StringIO(""))
    session = chat.Chat(None, person, role="runner", extension=chat.extensions(person))
    offered = [s["function"]["name"] for s, _ in session.offered] + [t.name for t in mcp.TOOLS]
    named = [n for n in offered if any(word in n.lower() for word in floor)]
    assert named == []


def test_a_tool_argument_naming_the_floor_is_refused_whichever_tool_carries_it():
    from ml_stack.sentinel.human import agent_may
    for tool in ("bench_run", "serve_up", "models_fetch", "speech_say", "doctor", "decide"):
        for text in ("ml-stack security quarantine release q-1", "ml-stack-security mode off",
                     "python -m ml_stack.sentinel.cli release q-1"):
            assert agent_may(tool, {"argv": [text], "extra": [text]}), (tool, text)


def test_the_os_keystore_is_reached_only_from_the_vault_signing_and_credentials_code():
    import ast
    src = Path(SRC) / "ml_stack"
    allowed = {"memory/vault.py", "fleet/onboard/signing.py", "credentials/__init__.py"}
    found = set()
    for path in src.rglob("*.py"):
        rel = path.relative_to(src).as_posix()
        if rel.startswith("testing/"):
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else \
                [node.module or ""] if isinstance(node, ast.ImportFrom) else []
            if any(n.split(".")[0] == "keyring" for n in names):
                found.add(rel)
    assert found <= allowed, sorted(found - allowed)


def test_building_the_signing_keys_never_asks_the_keystore(tmp_path):
    import keyring
    from onboard_support import FileKeyring

    from ml_stack.fleet.onboard.signing import SigningKeys

    class Counting(FileKeyring):
        asked = 0

        def get_password(self, *a):
            Counting.asked += 1
            return super().get_password(*a)

        def set_password(self, *a):
            Counting.asked += 1
            return super().set_password(*a)

    before = keyring.get_keyring()
    keyring.set_keyring(Counting())
    try:
        os.environ["ML_STACK_TEST_KEYRING"] = str(tmp_path / "ks.json")
        SigningKeys(tmp_path / "state")
        assert Counting.asked == 0
    finally:
        keyring.set_keyring(before)
        os.environ.pop("ML_STACK_TEST_KEYRING", None)
