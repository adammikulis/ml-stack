#!/usr/bin/env python3
"""Ask one checkout what a command answers to, as JSON on stdout.

    probe.py            every command word this checkout installs
    probe.py serve      that command, its subcommands, their help, and what a bare argv does

Run by path, never imported: the ``ml_stack`` it reads is the one on ``PYTHONPATH``.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from collections.abc import Iterator
from typing import Any

from ml_stack.cli import commands, load


def taken(target: str) -> argparse.ArgumentParser | None:
    """The parser a command builds, taken as it parses ``--help``."""
    seen: list[argparse.ArgumentParser] = []
    real = argparse.ArgumentParser.parse_known_args

    def spy(self: argparse.ArgumentParser, args: Any = None,
            namespace: Any = None) -> Any:
        if not seen:
            seen.append(self)
        return real(self, args, namespace)

    argparse.ArgumentParser.parse_known_args = spy  # type: ignore[method-assign]
    try:
        with contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            load(target)(["--help"])
    except SystemExit:
        pass
    finally:
        argparse.ArgumentParser.parse_known_args = real  # type: ignore[method-assign]
    return seen[0] if seen else None


def children(parser: argparse.ArgumentParser) -> list[tuple[str, argparse.ArgumentParser]]:
    """Each subcommand this parser adds, in the order it was declared."""
    for action in parser._actions:
        if hasattr(action, "add_parser"):
            return list(action.choices.items())
    return []


def refuses_empty(parser: argparse.ArgumentParser) -> bool:
    """Whether this parser rejects a bare argv, so running it only prints usage."""
    try:
        with contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            parser.parse_known_args([])
    except SystemExit:
        return True
    return False


def nodes(words: tuple[str, ...], parser: argparse.ArgumentParser) -> Iterator[dict[str, Any]]:
    """This parser and every parser under it, each as ``words``, ``help`` and ``refuses``."""
    yield {"words": list(words), "help": parser.format_help(),
           "refuses": refuses_empty(parser)}
    for name, child in children(parser):
        yield from nodes((*words, name), child)


def tree(word: str) -> dict[str, Any]:
    """One command's nodes; a command that will not load leaves this process non-zero."""
    table = commands()
    parser = taken(table[word] if word else "ml_stack.cli:main")
    if parser is None:
        return {"word": word, "error": "built no parser"}
    return {"word": word, "nodes": list(nodes((word,) if word else (), parser))}


def run(argv: list[str]) -> int:
    """Print the word list, or one command's tree, as JSON."""
    if not argv:
        json.dump(["", *sorted(commands())], sys.stdout)
        return 0
    json.dump(tree(argv[0]), sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(run(sys.argv[1:]))
