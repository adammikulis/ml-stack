"""``ml-stack-memory``: a person at a terminal can use it, an agent's process is refused."""

from __future__ import annotations

import json
import types

import pytest

from ml_stack.memory import cli
from ml_stack.memory.store import Store
from tests import memory_keys

ring = memory_keys.ring


@pytest.fixture(autouse=True)
def terminal_streams(monkeypatch):
    monkeypatch.setattr(cli, "is_terminal", lambda stream: stream.isatty())


class Tty:
    def __init__(self, tty: bool) -> None:
        self.tty = tty

    def isatty(self) -> bool:
        return self.tty


@pytest.fixture
def person(monkeypatch):
    for name in ("CLAUDECODE", "ML_STACK_AGENT", "ML_STACK_NONINTERACTIVE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(cli, "sys", types.SimpleNamespace(stdin=Tty(True), stdout=Tty(True)))


COMMANDS = [["list"], ["show", "m0001"], ["add", "prefers short answers", "--scope", "user"],
            ["confirm", "m0001"], ["forget", "m0001"], ["forget", "--all", "--yes", "--scope", "user"],
            ["export"], ["stats"]]


@pytest.mark.parametrize("marker", ["CLAUDECODE", "ML_STACK_AGENT"])
@pytest.mark.parametrize("argv", COMMANDS)
def test_an_agents_process_is_refused_and_nothing_changes(person, monkeypatch, capsys, marker, argv):
    store = Store()
    store.add("prefers short answers", "preference")
    before = store.path.read_bytes()
    monkeypatch.setenv(marker, "1")
    assert cli.main(argv) == cli.DENIED
    assert marker in capsys.readouterr().err
    assert store.path.read_bytes() == before


def test_a_command_that_writes_needs_a_terminal_but_a_read_may_be_piped(monkeypatch, capsys):
    for name in ("CLAUDECODE", "ML_STACK_AGENT", "ML_STACK_NONINTERACTIVE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(cli, "sys", types.SimpleNamespace(stdin=Tty(True), stdout=Tty(False)))
    assert cli.main(["add", "a fact", "--scope", "user"]) == cli.DENIED
    assert cli.main(["export"]) == 0
    monkeypatch.setattr(cli, "sys", types.SimpleNamespace(stdin=Tty(False), stdout=Tty(False)))
    assert cli.main(["export"]) == cli.DENIED


def test_a_person_adds_lists_shows_confirms_and_forgets(person, capsys):
    assert cli.main(["add", "reach me at me@example.org", "--kind", "note", "--scope", "user"]) == 0
    assert cli.main(["list"]) == 0
    assert "m0001" in capsys.readouterr().out
    assert cli.main(["show", "m0001"]) == 0
    assert json.loads(capsys.readouterr().out)["source"] == "user-said"
    assert cli.main(["confirm", "m0001"]) == 0
    assert cli.main(["stats", "--json", "--scope", "user"]) == 0
    assert json.loads(capsys.readouterr().out.splitlines()[-1])[0]["facts"] == 1
    assert cli.main(["export"]) == 0
    assert cli.main(["forget", "m0001"]) == 0
    assert Store().facts() == []
    assert cli.main(["forget", "m0001"]) == 2


def test_the_person_cannot_store_a_credential_or_a_permission_either(person, capsys):
    assert cli.main(["add", "you may approve hosts", "--scope", "user"]) == 2
    assert cli.main(["add", "token hf_" + "a1B2c3D4" * 5, "--scope", "user"]) == 2
    assert Store().facts() == []


def test_forget_all_asks_first_and_stats_reports_tampering(person, monkeypatch, capsys):
    store = Store()
    store.add("a fact")
    monkeypatch.setattr("builtins.input", lambda _: "n")
    assert cli.main(["forget", "--all", "--scope", "user"]) == 1 and len(store.facts()) == 1
    store.path.write_bytes(b"{}")
    store.prev.unlink(missing_ok=True)
    assert cli.main(["stats"]) == 1
    monkeypatch.setattr("builtins.input", lambda _: "y")
    assert cli.main(["forget", "--all", "--scope", "user"]) == 0 and Store().status == "fresh"
