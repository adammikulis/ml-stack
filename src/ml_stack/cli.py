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
               "ml-stack <command> --help is that command's help.",
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
        width = max((len(w) for w in table), default=0)
        for word in sorted(table):
            print(f"{word.replace('-', ' '):<{width}}  {_first_help_line(table[word])}")
        return 0
    from .fleet.launch import main as app
    return app(rest)


if __name__ == "__main__":
    raise SystemExit(main())
