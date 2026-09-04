"""``ml-stack`` -- one command in front of every ``ml-stack-<command>``.

``ml-stack bench sweep ...`` runs ``ml-stack-bench sweep ...`` in this process; bare, it
starts the app the way ``ml_stack.fleet.launch`` does.
"""

from __future__ import annotations

import argparse
import contextlib
import difflib
import importlib
import io
import sys
import tomllib
from importlib.metadata import entry_points
from pathlib import Path
from typing import Callable

__all__ = ["PREFIX", "commands", "load", "main"]

PREFIX = "ml-stack-"
PYPROJECT = Path(__file__).resolve().parents[2] / "pyproject.toml"


def commands() -> dict[str, str]:
    """``word -> module:attr`` for every installed ``ml-stack-<word>``, and the checkout's."""
    found: dict[str, str] = {}
    for point in entry_points(group="console_scripts"):
        if point.name.startswith(PREFIX):
            found[point.name[len(PREFIX):]] = point.value
    if PYPROJECT.exists():
        table = tomllib.loads(PYPROJECT.read_text()).get("project", {}).get("scripts", {})
        for name, target in table.items():
            if name.startswith(PREFIX):
                found.setdefault(name[len(PREFIX):], target)
    return found


def load(target: str) -> Callable[..., int | None]:
    """The callable ``module:attr`` names."""
    module, _, attr = target.partition(":")
    return getattr(importlib.import_module(module), attr or "main")


def _first_help_line(target: str) -> str:
    out = io.StringIO()
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
            load(target)(["--help"])
    except SystemExit:
        pass
    except Exception as failed:  # noqa: BLE001  (a missing extra is reported, not raised)
        return f"({type(failed).__name__}: {failed})"
    usage, _, body = out.getvalue().partition("\n\n")
    for paragraph in body.split("\n\n"):
        line = paragraph.strip().splitlines()[0] if paragraph.strip() else ""
        if line and not paragraph.startswith(" ") and not line.endswith(":"):
            return line
    return usage.strip().removeprefix("usage: ").splitlines()[0] if usage.strip() else ""


def _parser(table: dict[str, str]) -> argparse.ArgumentParser:
    words = sorted(table)
    listed = "\n".join(f"  {w.replace('-', ' ')}" for w in words)
    ap = argparse.ArgumentParser(
        prog="ml-stack",
        usage="ml-stack [--list] [<command> [args...]]",
        description="Run an ml-stack-<command>, or with none, start the app.",
        epilog=f"commands (each is also ml-stack-<command>):\n{listed}\n\n"
               "ml-stack help <command> is that command's help; ml-stack help lists them.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true",
                    help="every command with the first line of its help")
    return ap


def _unknown(words: list[str], table: dict[str, str]) -> int:
    asked = " ".join(words)
    spoken = {w.replace("-", " "): w for w in table}
    near = difflib.get_close_matches(asked, spoken, n=5, cutoff=0.4)
    near = near or [s for s in sorted(spoken) if s.startswith(words[0][:2])]
    hint = ", ".join(near) if near else "ml-stack --list shows them all"
    print(f"ml-stack: '{asked}' is not a command; nearest: {hint}", file=sys.stderr)
    return 2


HELP_LINE = "every command with the first line of its help, or one command's own help"


def listing(table: dict[str, str] | None = None) -> str:
    """Every command, one line each: the word as `ml-stack` takes it and the first line of
    its help. `help` itself is described in a sentence rather than asked, or it would ask
    itself forever."""
    table = commands() if table is None else table
    width = max((len(w) for w in table), default=0)
    return "\n".join(
        f"{word.replace('-', ' '):<{width}}  "
        + (HELP_LINE if word == "help" else _first_help_line(table[word]))
        for word in sorted(table))


def _help_parser(table: dict[str, str]) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="ml-stack-help",
        usage="ml-stack help [<command>...]   (also ml-stack-help)",
        description="Every command with the first line of its help; with a command named, "
                    "that command's own help (ml-stack help bench sweep is ml-stack-bench "
                    "sweep --help).",
        epilog=listing(table), formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("words", nargs="*", metavar="command")
    return ap


def help_main(argv: list[str] | None = None) -> int:
    """``ml-stack help`` and ``ml-stack-help``: with no words, every command with the first
    line of its help; with words, that command's own ``--help``. Adam, 2026-09-03: "is
    there ml-stack-help".
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    table = commands()
    ap = _help_parser(table)
    words = list(ap.parse_args(argv).words)
    if not words:
        print(ap.format_help())
        return 0
    for k in range(len(words), 0, -1):
        word = "-".join(words[:k])
        if word in table and word != "help":
            try:
                result = load(table[word])([*words[k:], "--help"])
            except SystemExit as left:
                return int(left.code or 0)
            return 0 if result is None else int(result)
    return _unknown(words, table)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    table = commands()
    if argv and not argv[0].startswith("-"):
        for k in range(len(argv), 0, -1):
            word = "-".join(argv[:k])
            if word in table:
                result = load(table[word])(argv[k:])
                return 0 if result is None else int(result)
        return _unknown(argv, table)

    known, rest = _parser(table).parse_known_args(argv)
    if known.list:
        print(listing(table))
        return 0
    from .fleet.launch import main as app
    return app(rest)


if __name__ == "__main__":
    raise SystemExit(main())
