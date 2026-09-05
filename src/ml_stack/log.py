"""How this library talks to a person: `say` for output, `warn` for trouble, `die` to stop.

Left alone, `say` writes to stdout and `warn` to stderr, exactly as `print` does. `to` and
`to_file` send both to a callback or a file instead, so a daemon or a program embedding
this library gets the same text without the calling code changing.
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import NoReturn, TextIO

__all__ = ["Listener", "die", "listen", "say", "to", "to_file", "warn"]

Listener = Callable[[str, str], None]
"""``on(stream, text)`` -- ``"out"`` or ``"err"``, and the text as it would have printed."""

_LISTENER: Listener | None = None


def _write(stream: str, parts: tuple[object, ...], sep: str, end: str,
           flush: bool) -> None:
    text = sep.join(str(one) for one in parts) + end
    if _LISTENER is not None:
        _LISTENER(stream, text)
        return
    where: TextIO = sys.stdout if stream == "out" else sys.stderr
    print(text, end="", file=where, flush=flush)


def say(*parts: object, sep: str = " ", end: str = "\n", flush: bool = False) -> None:
    """Write a command's output where a person will read it."""
    _write("out", parts, sep, end, flush)


def warn(*parts: object, sep: str = " ", end: str = "\n", flush: bool = False) -> None:
    """Write something that went wrong where a person will read it."""
    _write("err", parts, sep, end, flush)


def die(message: object = "", code: int = 1) -> NoReturn:
    """Write ``message`` as a warning and stop with ``code``."""
    if message != "":
        warn(message)
    raise SystemExit(code)


def listen(listener: Listener | None) -> Listener | None:
    """Send every line to ``listener`` rather than the console; returns the previous one."""
    global _LISTENER
    was, _LISTENER = _LISTENER, listener
    return was


@contextmanager
def to(listener: Listener | None) -> Iterator[None]:
    """Send every line to ``listener`` for the duration of the block."""
    was = listen(listener)
    try:
        yield
    finally:
        listen(was)


@contextmanager
def to_file(path: str | Path, *, append: bool = True) -> Iterator[Path]:
    """Send every line to ``path`` for the duration of the block."""
    where = Path(path)
    where.parent.mkdir(parents=True, exist_ok=True)
    handle = where.open("a" if append else "w", encoding="utf-8")

    def onto(stream: str, text: str) -> None:
        handle.write(text)
        handle.flush()

    try:
        with to(onto):
            yield where
    finally:
        handle.close()
