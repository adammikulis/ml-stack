"""Stop a model server nobody is using, so the memory it holds goes back.

A server holds its weights and every seat's cache for as long as it runs, and a machine
that answered one question at nine o'clock is still carrying ninety gigabytes at five. What
is missing is not a way to stop one -- that is `ServerManager.release` -- it is knowing
which one nobody wants.

**Idleness is asked, not remembered by whoever asked last.** Writing down "last used"
would only be as reliable as every caller remembering to write it, and an adopted server
has no caller here at all. So a server is asked instead: ``GET /slots`` says whether any
slot is processing right now, and a port not *seen* busy since it was looked at is idle for
that long. That never mistakes a long single answer for idleness, because the slot is
processing throughout.

**What the looks find is kept, so a later pass is not starting from nothing.** One look can
only ever report zero, which would make a single pass useless against any real threshold.
So each look writes what it saw (`state_path()`) and idleness adds up across looks.

**A gap in the looking is a gap in the evidence.** Only an interval short enough to be an
observation counts (`TRUST_S`): a server not seen busy at nine and not seen busy at five was
not therefore idle all day, and crediting that gap would stop a server somebody used at
noon. A longer gap starts the clock again, so a pass after a silent afternoon reports
nearly nothing and stops nothing, while a machine whose daemon has been looking every
minute can act at once.

A server that does not answer at all is left alone. Not answering is a reason to look at a
machine, not a reason to kill something on it.

    with watching(older_than=300):   # a thread, looking every minute
        ...
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from ml_stack import home

__all__ = ["Idleness", "TRUST_S", "busy_now", "ports_idle", "state_path",
           "reclaim_idle", "watching"]

#: How often the watcher looks, unless told otherwise.
EVERY_S = 60.0

#: The longest gap between two looks that still counts as having watched. Anything longer
#: is time nobody observed, and starts the clock again.
TRUST_S = 300.0


class _Default:
    """Stands for "the usual file", so ``state=None`` can mean "keep nothing"."""


DEFAULT = _Default()


def state_path() -> Path:
    """Where the looks are kept, so a later pass is not starting from nothing."""
    return home.moved("idle.json")


def busy_now(base_url: str, *, timeout: float = 2.0) -> bool | None:
    """Whether any slot on this server is processing. None when it does not say.

    llama.cpp's ``/slots`` is a list, one entry a seat, each carrying ``is_processing``.
    A build served with ``--no-slots``, or anything else on the port, says nothing -- and
    nothing is not idle.
    """
    from ml_stack.http import ServerError, request_json

    try:
        slots = request_json(f"{base_url.rstrip('/')}/slots", timeout=timeout)
    except ServerError:
        return None
    if not isinstance(slots, list):
        return None
    seen = False
    for slot in slots:
        if not isinstance(slot, Mapping):
            continue
        state = slot.get("is_processing")
        if isinstance(state, bool):
            seen = True
            if state:
                return True
    return False if seen else None


class Idleness:
    """How long each server has gone without being seen busy.

    Every look is written to ``state`` and idleness adds up across looks, but only over
    intervals no longer than ``trust``: a gap in the looking is a gap in the evidence, and
    cannot count towards stopping anything.
    """

    def __init__(self, *, clock: Callable[[], float] = time.time,
                 probe: Callable[[str], bool | None] = busy_now,
                 state: Path | str | None | _Default = DEFAULT,
                 trust: float = TRUST_S) -> None:
        self.clock = clock
        self.probe = probe
        self.trust = trust
        if isinstance(state, _Default):
            state = state_path()
        self.state = Path(state) if state is not None else None
        self.seen: dict[int, dict[str, float]] = self._read()

    def _read(self) -> dict[int, dict[str, float]]:
        if self.state is None:
            return {}
        try:
            held = json.loads(self.state.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        out: dict[int, dict[str, float]] = {}
        for port, row in (held.items() if isinstance(held, dict) else ()):
            if isinstance(row, Mapping) and str(port).isdigit():
                out[int(port)] = {"idle": float(row.get("idle") or 0.0),
                                  "looked": float(row.get("looked") or 0.0)}
        return out

    def _write(self) -> None:
        if self.state is None:
            return
        try:
            self.state.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.state.with_suffix(".json.tmp")
            tmp.write_text(json.dumps({str(p): r for p, r in self.seen.items()},
                                      indent=2, sort_keys=True), encoding="utf-8")
            os.replace(tmp, self.state)
        except OSError:  # a reading that cannot be kept is still a reading
            pass

    def look(self, servers: Mapping[int, Mapping[str, Any]]) -> dict[int, float]:
        """``{port: seconds idle}`` for every server that answered. One that did not is
        left out, and a port that has gone is forgotten."""
        now = self.clock()
        idle: dict[int, float] = {}
        for port, entry in servers.items():
            where = str(entry.get("base_url") or f"http://127.0.0.1:{port}")
            busy = self.probe(where)
            if busy is None:
                continue
            row = self.seen.setdefault(port, {"idle": 0.0, "looked": now})
            watched = now - row["looked"]
            row["idle"] = 0.0 if (busy or watched > self.trust) else row["idle"] + watched
            row["looked"] = now
            idle[port] = row["idle"]
        for gone in set(self.seen) - set(servers):
            self.seen.pop(gone, None)
        self._write()
        return idle


def _recorded() -> dict[int, dict]:
    from ml_stack.serve.manager import recorded_servers

    return recorded_servers()


def _stop(port: int, entry: Mapping[str, Any]) -> bool:
    """Stop the server on ``port``. True when a process was ended."""
    from ml_stack.platform import stop_pid
    from ml_stack.serve.manager import _DEFAULT

    pid = entry.get("pid")
    if isinstance(pid, int) and pid > 0:
        stop_pid(pid)
    return _DEFAULT.reclaim(int(port)) or isinstance(pid, int)


def reclaim_idle(*, older_than: float, idleness: Idleness | None = None,
                 servers: Callable[[], Mapping[int, Mapping[str, Any]]] = _recorded,
                 stop: Callable[[int, Mapping[str, Any]], bool] = _stop,
                 say: Callable[[str], None] = lambda _line: None) -> list[int]:
    """Stop every recorded server idle for longer than ``older_than`` seconds.

    Returns the ports stopped. ``idleness`` carries the clocks between calls; without one
    a fresh reading is taken, which can only report zero, so a caller that means to reclaim
    keeps its own.
    """
    if older_than <= 0:
        return []
    held = dict(servers())
    idle = (idleness or Idleness()).look(held)
    stopped: list[int] = []
    for port, seconds in sorted(idle.items()):
        if seconds < older_than:
            continue
        entry = held.get(port) or {}
        model = str(entry.get("model") or "")
        if stop(port, entry):
            stopped.append(port)
            say(f"reclaimed port {port} after {seconds:.0f}s idle"
                + (f" ({model.rsplit('/', 1)[-1]})" if model else ""))
    return stopped


@contextmanager
def watching(*, older_than: float, every: float = EVERY_S,
             idleness: Idleness | None = None,
             servers: Callable[[], Mapping[int, Mapping[str, Any]]] = _recorded,
             stop: Callable[[int, Mapping[str, Any]], bool] = _stop,
             say: Callable[[str], None] = lambda _line: None) -> Iterator[threading.Event]:
    """Reclaim idle servers every ``every`` seconds for the length of the block.

    Yields the event that ends it, so a caller can stop early. Nothing is reclaimed on the
    way in: the first look only starts the clocks.
    """
    done = threading.Event()
    watcher = idleness or Idleness()

    def loop() -> None:
        while not done.is_set():
            try:
                reclaim_idle(older_than=older_than, idleness=watcher, servers=servers,
                             stop=stop, say=say)
            except Exception as why:  # noqa: BLE001 - a watcher that dies stops watching
                say(f"reclaim: {type(why).__name__}: {why}")
            done.wait(every)

    thread = threading.Thread(target=loop, name="ml-stack-reclaim", daemon=True)
    thread.start()
    try:
        yield done
    finally:
        done.set()
        thread.join(timeout=max(1.0, min(every, 5.0)))


def ports_idle(seconds: float = 0.0, *, servers: Callable[[], Mapping[int, Mapping[str, Any]]] = _recorded,
               idleness: Idleness | None = None) -> dict[int, float]:
    """One reading of how long each recorded server has been idle. For a person looking."""
    return {port: idle for port, idle in (idleness or Idleness()).look(dict(servers())).items()
            if idle >= seconds}
