"""``ml-stack-serve up`` is a lease: asking the broker for a server and holding it.

`up` writes a request, starts a detached holder process, and waits for the holder to say what
the broker did. The holder asks the broker for a lease with the shape that was asked for; the
broker admits it against memory, queues it behind other leases when memory is short, picks the
port, and starts the server. The holder then keeps the lease until `down`, until the idle
timeout, or until it is killed (the broker drops the lease of a holder that has ended). The
server is stopped only when no other lease uses it.

A hold is a small record under ``<state>/holds/``: who holds it, what, since when, and what
the broker said. Nothing here starts llama-server; `ml_stack.serve.grant` makes sure of it.
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import json
import logging
import os
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ml_stack import home, hub, jobs
from ml_stack.files import read_json, write_json
from ml_stack.lock import only_one
from ml_stack.platform import stop_pid
from ml_stack.serve import admission, broker_wire
from ml_stack.serve.backend import ServerFailed, ServerSpec
from ml_stack.serve.broker import Ask, BrokerError
from ml_stack.serve.broker_wire import spec_to_json
from ml_stack.serve.leases import recorded_servers
from ml_stack.serve.matching import model_matches
from ml_stack.serve.process import pid_exists, started_at
from ml_stack.serve.reclaim import busy_now
from ml_stack.units import human_bytes

__all__ = ["Hold", "Refusal", "Terms", "check_fits", "down", "find", "holds", "run_holder", "shape_of",
           "up"]

logger = logging.getLogger(__name__)
DEFAULT_WAIT_S = 600.0
START_TOLERANCE_S = 2.0
POLL_S = 0.2
STARTING_S = 300.0
"""A hold whose holder has not started is given up on after this long."""
_STOPS: dict[str, threading.Event] = {}
"""Holders running as threads of this process (when the broker does too), by hold id."""


class Refusal(Exception):
    """The lease will not be granted; ``lines`` are what a person is told, one per line."""

    def __init__(self, *lines: str) -> None:
        super().__init__(lines[0])
        self.lines = lines


@dataclasses.dataclass
class Hold:
    """One `up`: the shape asked for, who holds it and what the broker said."""

    id: str
    shape: str
    model: str
    context: int
    parallel: int
    status: str = "starting"  # starting | queued | ready | refused | released
    pid: int = 0
    started: float = 0.0
    since: float = 0.0
    port: int = 0
    base_url: str = ""
    lease: str = ""
    memory: int = 0
    idle_s: float = 0.0
    error: str = ""
    adopted: bool = False

    @classmethod
    def read(cls, path: Path) -> Hold | None:
        held = read_json(path, {})
        names = {f.name for f in dataclasses.fields(cls)}
        try:
            return cls(**{k: v for k, v in held.items() if k in names}) if held else None
        except TypeError:
            return None

    def alive(self) -> bool:
        """Whether the process that holds this is the one that started it."""
        if not self.pid or not pid_exists(self.pid):
            return False
        began = started_at(self.pid)
        return not self.started or (began is not None
                                    and abs(began - self.started) <= START_TOLERANCE_S)


def hold_dir() -> Path:
    return home.state("holds")


def log_path(hold_id: str) -> Path:
    return home.cache("logs", f"hold-{hold_id}.log")


def _path(hold_id: str) -> Path:
    return hold_dir() / f"{hold_id}.json"


def _request_path(hold_id: str) -> Path:
    return hold_dir() / f"{hold_id}.request.json"


def shape_of(spec: ServerSpec) -> str:
    """A key for what was asked: the model and every setting that shapes the server, not the
    port (the broker picks that)."""
    body = {k: v for k, v in spec_to_json(spec).items() if k not in ("port", "process")}
    return hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest()[:12]


def holds() -> list[Hold]:
    """Every hold whose holder is running; the records of ones that are gone are removed."""
    out = []
    for path in sorted(hold_dir().glob("*.json")):
        if path.name.endswith(".request.json"):
            continue
        hold = Hold.read(path)
        if hold is not None and (hold.alive() or (
                hold.status == "starting" and not hold.pid
                and time.time() - hold.since < STARTING_S)):
            out.append(hold)
        else:
            path.unlink(missing_ok=True)
            _request_path(path.stem).unlink(missing_ok=True)
    return out


def find(shape: str) -> Hold | None:
    """The live hold with this shape that is usable (queued or ready)."""
    return next((h for h in holds()
                 if h.shape == shape and h.status in ("starting", "queued", "ready")), None)


def target(which: str) -> list[Hold]:
    """The holds a ``down`` argument names: a port, or part of a model name."""
    word = which.strip()
    return [h for h in holds()
            if (word.isdigit() and h.port == int(word))
            or (not word.isdigit() and word and model_matches(h.model, word))]


# ---------------------------------------------------------------------------------- asking
def check_fits(spec: ServerSpec, *, budget: int | None = None, total: int | None = None,
               ask: str = "") -> int:
    """The bytes ``spec`` will take; raises `Refusal` when it could not fit on this machine
    even with nothing else running, saying the longest context that does."""
    wanted = admission.estimate_bytes(spec)
    budget = hub.room() if budget is None else budget
    if not wanted or not budget or wanted <= budget:
        return wanted
    low, high, best = 0, int(spec.context) // 256, 0
    while low <= high:
        mid = (low + high) // 2
        if admission.estimate_bytes(dataclasses.replace(spec, context=mid * 256)) <= budget:
            best, low = mid * 256, mid + 1
        else:
            high = mid - 1
    name = Path(str(spec.model)).name
    lines = [f"refused: {name} at {int(spec.context):,} tokens needs about "
             f"{human_bytes(wanted)} and this machine lets a model use {human_bytes(budget)}"]
    if best:
        lines.append(f"  the longest context that fits is {best:,} tokens: "
                     f"ml-stack-serve up {ask or name} --context {best}")
    else:
        lines.append("  the weights alone do not fit")
    installed = hub.total_memory() if total is None else total
    if installed and wanted <= installed * 0.9:
        lines.append(f"  the limit can be raised by a person (never by an agent): "
                     f"ml-stack-serve memory --for {ask or name} --ctx {int(spec.context)} "
                     f"--apply")
    raise Refusal(*lines)


def _write(hold: Hold) -> None:
    write_json(_path(hold.id), dataclasses.asdict(hold))


@dataclasses.dataclass(frozen=True)
class Terms:
    """How `up` asks: how long the lease may queue, whether to wait for it, when to give it
    up, the port it would like, and the bytes it will take."""

    wait_s: float = DEFAULT_WAIT_S
    wait: bool = True
    idle_s: float = 0.0
    port: int = 0
    weight: int = 0
    patience_s: float = 120.0
    """How long the holder may stay silent before `up` gives up on it."""


def up(spec: ServerSpec, *, manager: Any, terms: Terms = Terms(),  # noqa: B008 - frozen
       say: Callable[[str], None] = print) -> Hold:
    """A hold on a server for ``spec``: the one already held for this shape, else a new
    holder that asks the broker. Returns when the server is ready, or (``wait`` false) when
    it is queued. Raises `Refusal` with the broker's reason when the lease is refused.

    The holder is started by `_detach`: a process that outlives this terminal.
    """
    wait_s, wait, idle_s, asked_port, weight, patience_s = (
        terms.wait_s, terms.wait, terms.idle_s, terms.port, terms.weight, terms.patience_s)
    shape = shape_of(spec)
    hold_dir().mkdir(parents=True, exist_ok=True)
    # Finding a hold for this shape and making one are one step: two `up`s at once must not
    # start two holders, so the second finds the first's record and adopts it.
    with only_one(hold_dir() / "up.lock", wait=True, timeout=30.0, announce=lambda _l: None):
        hold = find(shape)
        adopted = hold is not None
        if hold is None:
            hold = Hold(id=os.urandom(4).hex(), shape=shape, model=str(spec.model),
                        context=int(spec.context), parallel=int(spec.parallel or 1),
                        idle_s=idle_s, memory=weight, since=time.time())
            _write(hold)
            write_json(_request_path(hold.id), {
                "spec": spec_to_json(spec), "wait_s": wait_s, "idle_s": idle_s,
                "port": asked_port, "weight": weight,
                "options": {"backend": manager.backend.options()}
                if hasattr(manager.backend, "options") else {}})
    if adopted:
        say(f"adopted the lease {hold.id}"
            + (f" on port {hold.port}" if hold.port else f" ({hold.status})"))
    else:
        _detach(hold.id, manager)

    told, deadline = "", time.monotonic() + patience_s
    while True:
        now = Hold.read(_path(hold.id)) or hold
        if now.status == "ready":
            return dataclasses.replace(now, adopted=adopted)
        if now.status in ("refused", "released"):
            _path(hold.id).unlink(missing_ok=True)
            _request_path(hold.id).unlink(missing_ok=True)
            raise Refusal(*(now.error or "the lease was refused").splitlines())
        if now.status == "queued":
            place = _place(manager, now.pid)
            if place and place != told:
                told = place
                say(place)
            if not wait:
                say(f"queued; it will start when memory frees: ml-stack-serve status "
                    f"(lease {hold.id}); ml-stack-serve down {hold.id} gives it up")
                return now
            deadline = max(deadline, time.monotonic() + patience_s)
        if time.monotonic() > deadline:
            raise Refusal(f"the lease holder did not answer within {patience_s:.0f}s; "
                          f"see {log_path(hold.id)}")
        if now.pid and not now.alive() and now.status != "ready":
            raise Refusal(f"the lease holder ended before the broker answered; "
                          f"see {log_path(hold.id)}")
        time.sleep(POLL_S)


def _place(manager: Any, pid: int) -> str:
    """Where this holder stands in the broker's queue, in words; "" when it does not."""
    with contextlib.suppress(BrokerError, OSError, ValueError, KeyError, AttributeError):
        queue = manager.broker.snapshot()["queue"]
        for at, waiting in enumerate(queue, start=1):
            if waiting["pid"] == pid:
                why = waiting.get("blocked_by") or "behind other leases"
                return f"queued, #{at} of {len(queue)}: {why}"
    return ""


def _detach(hold_id: str, manager: Any) -> None:
    """Start the holder: a detached process that outlives this terminal. When the broker
    runs inside this process (``ML_STACK_BROKER_LOCAL=1``, the tests) no other process could
    reach it, so the holder is a thread of this one."""
    if os.environ.get(broker_wire.LOCAL_ENV) == "1":
        threading.Thread(target=run_holder, args=(hold_id,),
                         kwargs={"manager": manager, "tick_s": 0.2}, daemon=True).start()
        return
    jobs.detach("ml_stack.serve.cli", ["hold", hold_id], log=log_path(hold_id))


# ------------------------------------------------------------------------------ the holder
def run_holder(hold_id: str, *, manager: Any, **kept: Any) -> int:
    """`_hold`, and when it fails for a reason of its own, the record says so for `up`."""
    try:
        return _hold(hold_id, manager=manager, **kept)
    except (BrokerError, ServerFailed, OSError, ValueError, TypeError, KeyError,
            RuntimeError) as why:
        logger.exception("the holder of %s failed", hold_id)
        hold = Hold.read(_path(hold_id))
        if hold is not None:
            hold.status, hold.error = "refused", f"the lease holder failed: {why!r}"
            _write(hold)
        return 1


def _hold(hold_id: str, *, manager: Any,
               busy: Callable[[str], bool | None] | None = None,
               stop: threading.Event | None = None,
               tick_s: float = 1.0) -> int:
    """Be the holder of ``hold_id``: ask the broker, say what it answered, keep the lease
    until stopped or idle, release it and stop the server if nobody else uses it."""
    hold = Hold.read(_path(hold_id))
    request = read_json(_request_path(hold_id), {})
    if hold is None or not request:
        return 2
    stop = stop or threading.Event()
    _STOPS[hold_id] = stop
    busy = busy or busy_now
    hold.pid = os.getpid()
    hold.started = started_at(hold.pid) or time.time()
    hold.status = "queued"
    _write(hold)
    spec = dict(request["spec"])
    ask = Ask(purpose=f"serve:{hold.shape}", models=(str(spec.pop("model")),), pid=hold.pid,
              label=f"ml-stack-serve up {Path(hold.model).name}",
              weight=int(request.get("weight") or 0),
              spec={k: v for k, v in spec.items() if k != "port"},
              port=int(request.get("port") or 0), options=request.get("options") or {})
    granted = None
    try:
        granted = manager.broker.lease(ask, timeout=float(request.get("wait_s") or DEFAULT_WAIT_S))
    except (BrokerError, OSError, ValueError) as why:
        hold.status, hold.error = "refused", _refused(str(why))
        _write(hold)
        return 1
    hold.status, hold.port, hold.base_url = "ready", granted.port, granted.base_url
    hold.lease, hold.since, hold.adopted = granted.lease, time.time(), bool(granted.shared)
    _write(hold)
    idle_s, quiet = float(request.get("idle_s") or 0.0), time.monotonic()
    try:
        while not stop.wait(tick_s):
            if _server_gone(manager, granted.port):
                break
            if busy(granted.base_url) is not False:
                quiet = time.monotonic()
            elif idle_s and time.monotonic() - quiet >= idle_s:
                break
    finally:
        _release(manager, granted)
        hold.status = "released"
        _write(hold)
        _path(hold_id).unlink(missing_ok=True)
        _request_path(hold_id).unlink(missing_ok=True)
        _STOPS.pop(hold_id, None)
    return 0


def _server_gone(manager: Any, port: int) -> bool:
    """Whether the server recorded on ``port`` has died, so there is nothing to hold."""
    entry = recorded_servers(manager.state_file).get(port)
    pid = entry.get("pid") if entry else None
    return isinstance(pid, int) and not pid_exists(pid)


def _refused(why: str) -> str:
    """The broker's words for a refusal, trimmed to what a person needs."""
    return why.replace("\n", " ")


def _release(manager: Any, granted: Any) -> None:
    with contextlib.suppress(BrokerError, OSError):
        manager.broker.release(granted.lease)
        manager.broker.stop(granted.port)


# ------------------------------------------------------------------------------------ down
def down(holds_: list[Hold], *, manager: Any, wait_s: float = 30.0) -> list[tuple[Hold, str]]:
    """Release each hold; what happened to its server, one line each."""
    out = []
    for one in holds_:
        event = _STOPS.get(one.id)
        if event is not None:
            event.set()
        elif one.pid:
            with contextlib.suppress(OSError):
                stop_pid(one.pid)
        deadline = time.monotonic() + wait_s
        while _path(one.id).exists() and time.monotonic() < deadline:
            time.sleep(POLL_S)
        users = _users(manager, one.port)
        out.append((one, f"still used by {users}; left running" if users else "server stopped"))
    return out


def _users(manager: Any, port: int) -> str:
    with contextlib.suppress(BrokerError, OSError, ValueError, KeyError, AttributeError):
        for server in manager.broker.snapshot()["servers"]:
            if server["port"] == port and server["holders"]:
                return ", ".join(f"pid {h['pid']}" + (f" ({h['label']})" if h["label"] else "")
                                 for h in server["holders"])
    return ""
