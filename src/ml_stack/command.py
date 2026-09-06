"""How a command declares itself: shared options, subcommands, and the parser they build.

``SHARED`` holds the options more than one command takes -- ``--port``, ``--model``,
``--json``, ``--yes``, ``--out``, ``--context``, ``--timeout``, ``--parallel``,
``--dry-run``. `option` takes one by name and changes what this command needs changed;
`flag` declares one of a command's own. A `Group` collects them and answers as ``main``.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = ["SHARED", "Command", "Group", "Option", "flag", "option"]


@dataclass(frozen=True, slots=True)
class Option:
    """One ``add_argument`` call, held until a parser asks for it."""

    flags: tuple[str, ...]
    kwargs: dict[str, Any] = field(default_factory=dict)

    def but(self, **changes: Any) -> Option:
        """The same option with the named keywords changed."""
        return Option(self.flags, {**self.kwargs, **changes})

    def add_to(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(*self.flags, **self.kwargs)


SHARED: dict[str, Option] = {
    "port": Option(("--port",), {"type": int, "help": "the port to use"}),
    "model": Option(("--model",), {"default": "", "help": "which model"}),
    "json": Option(("--json",), {"action": "store_true",
                                 "help": "print one JSON object instead of the listing"}),
    "yes": Option(("--yes",), {"action": "store_true", "help": "do not ask before acting"}),
    "out": Option(("--out",), {"default": "", "help": "where to write it"}),
    "context": Option(("--context",), {"type": int, "help": "tokens of context"}),
    "timeout": Option(("--timeout",), {"type": float, "default": None,
                                       "help": "seconds to wait"}),
    "parallel": Option(("--parallel",), {"type": int, "help": "how many at once"}),
    "dry-run": Option(("--dry-run",), {"action": "store_true", "dest": "dry_run",
                                       "help": "print the plan and do nothing"}),
}
"""The options declared once here, taken by name with `option`."""


def option(name: str, **changes: Any) -> Option:
    """The shared option ``name``, with anything named here changed."""
    shared = SHARED[name]
    return shared.but(**changes) if changes else shared


def flag(*flags: str, **kwargs: Any) -> Option:
    """An option or positional of one command's own."""
    return Option(tuple(flags), dict(kwargs))


@dataclass(frozen=True, slots=True)
class Command:
    """One subcommand: what it is called, its one line of help, its options, its handler."""

    name: str
    help: str
    run: Callable[[argparse.Namespace], int | None]
    options: tuple[Option, ...] = ()
    kwargs: dict[str, Any] = field(default_factory=dict)


class Group:
    """The commands one console script answers to, and the parser they build.

    With subcommands registered it builds ``prog sub [options]``; with ``run`` given and
    none registered it is a single command whose options sit at the top. ``main`` is
    `run`.
    """

    def __init__(self, prog: str, description: str, *,
                 options: Sequence[Option] = (),
                 run: Callable[[argparse.Namespace], int | None] | None = None,
                 **kwargs: Any) -> None:
        self.prog = prog
        self.description = description
        self.options = tuple(options)
        self.handler = run
        self.kwargs = kwargs
        self.commands: list[Command] = []

    def command(self, name: str, *, help: str, options: Sequence[Option] = (),
                **kwargs: Any) -> Callable[[Callable[..., int | None]], Callable[..., int | None]]:
        """Register the decorated function as the subcommand ``name``."""
        def take(fn: Callable[..., int | None]) -> Callable[..., int | None]:
            self.add(name, fn, help=help, options=options, **kwargs)
            return fn

        return take

    def add(self, name: str, fn: Callable[..., int | None], *, help: str,
            options: Sequence[Option] = (), **kwargs: Any) -> None:
        """Register a handler written elsewhere as the subcommand ``name``."""
        self.commands.append(Command(name, help, fn, tuple(options), dict(kwargs)))

    def parser(self) -> argparse.ArgumentParser:
        """The parser these commands build."""
        ap = argparse.ArgumentParser(prog=self.prog, description=self.description,
                                     **self.kwargs)
        for one in self.options:
            one.add_to(ap)
        if self.commands:
            sub = ap.add_subparsers(dest="cmd", required=True)
            for command in self.commands:
                child = sub.add_parser(command.name, help=command.help, **command.kwargs)
                for one in command.options:
                    one.add_to(child)
                child.set_defaults(run=command.run)
        return ap

    def run(self, argv: Sequence[str] | None = None) -> int:
        """Parse ``argv`` and run the command it names; returns its exit code."""
        args = self.parser().parse_args(None if argv is None else list(argv))
        handler = getattr(args, "run", None) or self.handler
        if handler is None:
            raise SystemExit(2)
        return int(handler(args) or 0)
