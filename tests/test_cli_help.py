"""Every ``ml-stack`` command answers ``--help``, and the README names only flags that exist.

Nothing else in the suite catches a flag that does not exist. Three silent edits in one
afternoon left ``--sample`` and ``--short`` documented but never added to the bench, so
argparse matched ``--short`` to ``--shortlist`` by prefix and refused it with a message about
the wrong flag. A ``--help`` that is asserted is the cheapest guard there is: each command
here is imported and its ``main`` called, so a fresh checkout tests itself without being
installed. No model is served, and nothing under ``~/.ml-stack`` is touched.
"""

from __future__ import annotations

import argparse
import importlib
import re
import shlex
import tomllib
from pathlib import Path
from unittest import mock

import pytest

REPO = Path(__file__).resolve().parent.parent
README = REPO / "README.md"


def _scripts() -> dict[str, str]:
    """``command -> module:attr`` for every entry point ``pyproject.toml`` installs."""
    table = tomllib.loads((REPO / "pyproject.toml").read_text())["project"]["scripts"]
    return dict(table)


SCRIPTS = _scripts()
"""What is installed, read from the same table ``pip`` reads."""


def _main_of(command: str):
    module, _, attr = SCRIPTS[command].partition(":")
    return getattr(importlib.import_module(module), attr or "main")


class _Captured(Exception):
    """Raised inside ``main`` to hand its parser out before it parses anything."""

    def __init__(self, parser: argparse.ArgumentParser) -> None:
        self.parser = parser


def parser_of(command: str) -> argparse.ArgumentParser:
    """The parser a command builds, taken out of ``main`` at the moment it would parse.

    None of the entry points exposes its parser, and none should have to: the object is
    caught on its way into ``parse_known_args`` (which ``parse_args`` also goes through)
    and handed back before any argument is read.
    """
    def grab(self: argparse.ArgumentParser, *_: object, **__: object) -> None:
        raise _Captured(self)

    with mock.patch.object(argparse.ArgumentParser, "parse_known_args", grab):
        try:
            _main_of(command)([])
        except _Captured as caught:
            return caught.parser
    raise AssertionError(f"{command}'s main never parsed its arguments")


def parsers_of(parser: argparse.ArgumentParser) -> dict[str, argparse.ArgumentParser]:
    """Every parser under this one, by the subcommand path that reaches it; '' is the top."""
    found = {"": parser}
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            for name, sub in action.choices.items():
                for path, deeper in parsers_of(sub).items():
                    found[f"{name} {path}".strip()] = deeper
    return found


def flags_of(parser: argparse.ArgumentParser, *paths: str) -> set[str]:
    """The ``--flags`` accepted at the named subcommand paths, or everywhere under ``parser``.

    A subcommand also accepts the top parser's own options before it, so '' is always
    included when a path is asked for.
    """
    everything = parsers_of(parser)
    chosen = everything.values() if not paths else [everything[p] for p in {"", *paths}]
    return {s for one in chosen for s in one._option_string_actions if s.startswith("--")}


PARSERS = {command: parser_of(command) for command in SCRIPTS}
SUBCOMMANDS = [(command, path) for command, parser in PARSERS.items()
               for path in parsers_of(parser) if path]


@pytest.fixture(autouse=True)
def bench_at_home(tmp_path, monkeypatch):
    """The bench takes a lock under its home before measuring; keep that out of ~/.ml-stack."""
    from ml_stack.graph import bench
    monkeypatch.setattr(bench, "HOME", tmp_path / "bench")


@pytest.mark.parametrize("command", sorted(SCRIPTS), ids=sorted(SCRIPTS))
def test_every_command_answers_help(command, capsys):
    with pytest.raises(SystemExit) as left:
        _main_of(command)(["--help"])
    assert left.value.code == 0
    assert "usage:" in capsys.readouterr().out


@pytest.mark.parametrize("command", sorted(c for c in SCRIPTS if c.startswith("ml-stack-")))
def test_the_umbrella_hands_back_the_same_help(command, capsys):
    """``ml-stack do --help`` is ``ml-stack-do --help``; ``ml-stack train run`` joins the words."""
    words = command[len("ml-stack-"):].split("-")
    with pytest.raises(SystemExit) as left:
        _main_of(command)(["--help"])
    assert left.value.code == 0
    direct = capsys.readouterr().out
    with pytest.raises(SystemExit) as left:
        _main_of("ml-stack")([*words, "--help"])
    assert left.value.code == 0
    assert capsys.readouterr().out == direct


@pytest.mark.parametrize(("command", "path"), SUBCOMMANDS,
                         ids=[f"{c} {p}" for c, p in SUBCOMMANDS])
def test_every_subcommand_answers_help(command, path, capsys):
    with pytest.raises(SystemExit) as left:
        _main_of(command)([*path.split(), "--help"])
    assert left.value.code == 0
    assert "usage:" in capsys.readouterr().out


@pytest.mark.parametrize("command", [c for c, p in PARSERS.items() if len(parsers_of(p)) > 1])
def test_the_top_level_help_names_every_subcommand(command, capsys):
    """A subcommand the usage line does not show is one nobody finds.

    A hand-written ``metavar="{a,b,c}"`` on the subparsers is a list that goes stale the
    day a fourth is added -- ``drafts`` and ``memory`` were both missing from their
    command's ``--help`` for exactly that reason.
    """
    with pytest.raises(SystemExit):
        _main_of(command)(["--help"])
    usage = capsys.readouterr().out.split("\n\n", 1)[0]
    for path in parsers_of(PARSERS[command]):
        if path:
            assert re.search(rf"\b{re.escape(path)}\b", usage), \
                f"the usage line of {command} --help does not name its subcommand {path!r}"


# -- the README ------------------------------------------------------------------------

FLAG = re.compile(r"(?<![\w-])--[a-z][a-z0-9-]*")
CODE_SPAN = re.compile(r"`([^`]*)`")


def _sections(text: str) -> dict[str, list[tuple[int, str]]]:
    """README ``##`` sections, each as numbered lines."""
    out: dict[str, list[tuple[int, str]]] = {}
    title = ""
    for number, line in enumerate(text.splitlines(), start=1):
        if line.startswith("## "):
            title = line[3:].strip()
            out[title] = []
        elif title:
            out[title].append((number, line))
    return out


def _command_lines(text: str) -> list[tuple[int, str]]:
    """Every line of a fenced block that starts an ``ml-stack`` command, continuations joined."""
    found: list[tuple[int, str]] = []
    fenced = False
    pending: tuple[int, str] | None = None
    for number, line in enumerate(text.splitlines(), start=1):
        if line.startswith("```"):
            fenced = not fenced
            continue
        if not fenced:
            continue
        if pending:
            start, sofar = pending
            line, pending = f"{sofar} {line.strip()}", None
            number = start
        if line.rstrip().endswith("\\"):
            pending = (number, line.rstrip()[:-1])
            continue
        if line.strip().startswith("ml-stack"):
            found.append((number, line.strip()))
    return found


def _documented() -> list[tuple[str, str, tuple[str, ...], set[str]]]:
    """``(where, command, subcommand paths, flags)`` for every place the README names a flag.

    Two places: a line of a fenced code block, which names its subcommand and so is held to
    that subcommand's parser; and a row of "The commands" table, held to the whole command.
    Prose is not scanned: "Serving a model" rightly names llama-server's own flags, the
    ones ``ServerSpec`` emits, and no parser here is meant to accept those.
    """
    text = README.read_text()
    sections = _sections(text)
    found = []

    for number, line in _command_lines(text):
        words = shlex.split(line, posix=True)
        command, rest = words[0], words[1:]
        if command not in SCRIPTS:
            continue
        subs = [rest[0]] if rest and rest[0] in parsers_of(PARSERS[command]) else []
        flags = {w for w in rest if FLAG.fullmatch(w)}
        if flags:
            found.append((f"README.md:{number}", command, tuple(subs), flags))

    for number, line in sections.get("The commands", []):
        if not line.startswith("| `ml-stack"):
            continue
        command = line.split("`", 2)[1].split()[0]
        flags = {f for span in CODE_SPAN.findall(line) for f in FLAG.findall(span)}
        if flags:
            found.append((f"README.md:{number}", command, (), flags))
    return found


DOCUMENTED = _documented()


def test_the_readme_was_read():
    """A regex that matches nothing would make the test below pass by saying nothing."""
    assert any(cmd == "ml-stack-serve" and subs == ("up",) for _, cmd, subs, _ in DOCUMENTED)
    assert any(cmd == "ml-stack-bench" and "--rates" in flags for _, cmd, _, flags in DOCUMENTED)


@pytest.mark.parametrize(("where", "command", "subs", "flags"), DOCUMENTED,
                         ids=[f"{w} {c}" for w, c, _, _ in DOCUMENTED])
def test_every_flag_the_readme_names_is_one_the_command_accepts(where, command, subs, flags):
    accepted = flags_of(PARSERS[command], *subs)
    missing = flags - accepted
    assert not missing, (f"{where} documents {sorted(missing)} for {command} "
                         f"{' '.join(subs)}, which accepts none of them")


# -- the bench, where the last three bugs were -----------------------------------------

def test_every_measuring_subcommand_takes_a_sample_and_no_two_define_it_twice():
    """``drafts`` defines ``--sample`` on its own and the shared block adds it to the rest.

    If the shared block ever included ``drafts`` too, argparse would raise on the duplicate
    while building the parser, and ``parser_of`` would have failed before this test ran;
    what is asserted here is that each still accepts it.
    """
    from ml_stack.graph.bench import MEASURING
    for sub in MEASURING:
        assert "--sample" in flags_of(PARSERS["ml-stack-bench"], sub), sub
    for sub in ("run", "sweep"):
        assert {"--short", "--smoke"} <= flags_of(PARSERS["ml-stack-bench"], sub), sub


def test_the_bench_refuses_an_abbreviated_flag_rather_than_guessing():
    """``--short`` bound to ``--shortlist`` by prefix was the bug; a prefix binds nothing now."""
    for path, parser in parsers_of(PARSERS["ml-stack-bench"]).items():
        assert parser.allow_abbrev is False, f"bench {path or 'top level'} allows abbreviation"
    with pytest.raises(SystemExit) as left:
        PARSERS["ml-stack-bench"].parse_args(["run", "x", "--shortl", "3"])
    assert left.value.code == 2


@pytest.mark.parametrize("sub", ("run", "sweep", "drafts"))
def test_the_help_of_a_measuring_subcommand_names_no_queue(sub, capsys):
    """``--no-queue`` is taken out of argv before the parser sees it, so only the parser
    can tell anyone it exists."""
    with pytest.raises(SystemExit):
        _main_of("ml-stack-bench")([sub, "--help"])
    assert "--no-queue" in capsys.readouterr().out
