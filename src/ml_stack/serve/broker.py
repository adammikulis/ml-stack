"""The one place on a machine a model server is asked for.

A client asks for a purpose and the models it would accept, and gets a lease on a server:
one already serving an acceptable model is shared, otherwise one is started on a port the
broker picks. A request for a purpose whose server is held with another model waits in a
first-come queue until its holders let go; nothing held is stopped to make room. Holders
are processes, so a lease whose pid has ended is released by `reap`, and a server of ours
nobody holds is stopped once it has been idle for ``idle_s`` and is not answering anything.
A server put up with ``ml-stack-serve up`` is held by itself until it is taken down.
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import os
import re
import sys
import threading
import time
import uuid
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from ml_stack import activity, gate
from ml_stack.client import is_healthy, reported_models, serving_params
from ml_stack.files import read_json, write_json
from ml_stack.hub import free_memory
from ml_stack.serve import canaries, grant, guarded, lease_history, provenance, unmanaged
from ml_stack.serve.backend import LlamaServerBackend, ServerFailed, ServerInfo, ServerSpec
from ml_stack.serve.events import Caller, Growth
from ml_stack.serve.leases import recorded_servers
from ml_stack.serve.manager import BESIDE_HEADROOM, Measuring, ServerManager, Starting
from ml_stack.serve.matching import model_matches
from ml_stack.serve.ports import DEFAULT_HOST, free_port, port_is_free
from ml_stack.serve.preflight import PreflightFailed
from ml_stack.serve.process import every_server, kill_process_tree, pid_exists
from ml_stack.serve.reclaim import busy_now
from ml_stack.serve.weights import weight_of

__all__ = ["Ask", "Broker", "BrokerError", "Grant", "Held", "Waiting"]

IDLE_S = 600.0
POLL_S = 0.5
_SPEC_FIELDS = frozenset(f.name for f in dataclasses.fields(ServerSpec)) - {"model", "port"}


_QUANT = re.compile(r"(?i)(?<![a-z0-9])((?:IQ|Q)\d(?:_[A-Z0-9]+)*|BF16|F16|F32)(?![a-z0-9])")
_FLAGS = ("context", "n_gpu_layers", "parallel", "mtp", "spec_type", "flash_attn", "embedding")


def _shape_of(settings: Mapping[str, Any]) -> dict[str, Any]:
    """The settings of a server that decide whether it can be shared, from an ask's spec, a
    `ServerSpec` as a dict or a lease record."""
    out: dict[str, Any] = {}
    for key in ("context", "parallel", "cache_type_k", "cache_type_v", "spec_type", "mtp",
                "embedding"):
        if key in settings and settings[key] is not None:
            out[key] = settings[key]
    return out


def _named(ask: Ask) -> str:
    return "model:" + Path(ask.models[0]).name


def _flags(ask: Ask) -> dict[str, Any]:
    """The settings a lease record keeps: quant (from the file name), the flags that shape the
    run, the share of the card, and the draft head's file name."""
    quant = _QUANT.search(Path(ask.models[0]).name)
    out: dict[str, Any] = {"quant": quant.group(1) if quant else "", "weight": ask.weight}
    out.update({k: ask.spec[k] for k in _FLAGS if k in ask.spec})
    if ask.spec.get("draft"):
        out["draft"] = Path(str(ask.spec["draft"])).name
    return out


def who() -> str:
    """This process as a person would recognise it in the queue."""
    return " ".join([Path(sys.argv[0]).name, *sys.argv[1:]])[:120] if sys.argv else ""


class BrokerError(RuntimeError):
    """A lease was not granted: it timed out, its asker ended, or the server would not start."""


@dataclass(frozen=True)
class Ask:
    """What a client wants: a purpose, the models it accepts (the first is started), and
    the server settings to start it with."""

    purpose: str
    models: tuple[str, ...]
    pid: int
    label: str = ""
    weight: int = 0
    spec: Mapping[str, Any] = field(default_factory=dict)
    #: a port the asker would like; the broker uses it only when it is free, else picks one
    port: int = 0
    #: how to start it: ``{"backend": {...}}`` names the binary or build, as for `start`
    options: Mapping[str, Any] = field(default_factory=dict)
    #: what the asker says about the lease (`provenance.asked`), and the process it asks for
    claim: Mapping[str, Any] = field(default_factory=dict)
    behalf: int = 0
    #: the broker's own record of the lease, made when it arrives
    record: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_json(cls, body: Mapping[str, Any]) -> Ask:
        """An ask from a request body. Raises ``ValueError`` on a malformed one."""
        models = tuple(str(m) for m in body.get("models") or () if str(m))
        purpose = str(body.get("purpose") or "")
        if not purpose or not models:
            raise ValueError("a lease names a purpose and at least one model")
        spec = dict(body.get("spec") or {})
        unknown = sorted(set(spec) - _SPEC_FIELDS)
        if unknown:
            raise ValueError(f"not server settings: {', '.join(unknown)}")
        return cls(purpose=purpose, models=models, pid=int(body.get("pid") or 0),
                   label=str(body.get("label") or ""), weight=int(body.get("weight") or 0),
                   spec=spec, port=int(body.get("port") or 0),
                   options=dict(body.get("options") or {}),
                   claim=dict(body.get("claim") or {}), behalf=int(body.get("behalf") or 0))

    def server_spec(self, port: int) -> ServerSpec:
        """The spec that starts this ask's first model on ``port``."""
        settings = {k: tuple(v) if isinstance(v, list) else v for k, v in self.spec.items()}
        return ServerSpec(model=self.models[0], port=port, **settings)


@dataclass
class Held:
    """A server the broker knows about, and the leases on it (lease id -> pid, label)."""

    port: int
    model: str
    names: tuple[str, ...] = ()
    purpose: str = ""
    pid: int | None = None
    ours: bool = True
    #: running without a record in the lease registry: counted, listed, never leased from
    unmanaged: bool = False
    loading: bool = False
    weight: int = 0
    idle_since: float = 0.0
    holders: dict[str, tuple[int, str]] = field(default_factory=dict)
    #: lease id -> why it was taken and by whom (`provenance.observe`)
    records: dict[str, dict[str, Any]] = field(default_factory=dict)
    info: ServerInfo | None = None
    #: what ``/props`` said this server actually loaded (its ``model_path``), fetched
    #: once when the server was found rather than per ask -- ``None`` when it could
    #: not be read.
    loaded_file: str | None = None
    #: the settings it was started with that decide whether another ask may share it
    #: (``context``, ``parallel``, ``cache_type_k``/``v``, ``mtp``, ``spec_type``,
    #: ``embedding``); a setting not known is not compared
    shape: dict[str, Any] = field(default_factory=dict)

    def short_of(self, asked: Mapping[str, Any]) -> str:
        """Why this server cannot serve ``asked`` (an ask's spec), or "" when it can: it must
        hold at least the context and slots asked for and run the cache type, MTP head and
        speculation kind asked for. Sharing by model alone served a 256K request from a
        32K server."""
        have = self.shape
        for key, what in (("context", "tokens of context"), ("parallel", "slot(s)")):
            want, got = asked.get(key), have.get(key)
            if isinstance(want, int) and want > 0 and isinstance(got, int) and got < want:
                return f"port {self.port} serves {got:,} {what}, {want:,} asked"
        for key in ("cache_type_k", "cache_type_v", "spec_type"):
            want, got = asked.get(key), have.get(key)
            if want and got is not None and want != got:
                return f"port {self.port} serves {key} {got!r}, {want!r} asked"
        for key in ("mtp", "embedding"):
            want, got = asked.get(key), have.get(key)
            if want is not None and got is not None and bool(want) != bool(got):
                return f"port {self.port} serves {key}={bool(got)}, {bool(want)} asked"
        return ""

    @property
    def base_url(self) -> str:
        return f"http://{DEFAULT_HOST}:{self.port}"

    def serves(self, models: Iterable[str]) -> bool:
        """Whether this server is serving one of ``models``."""
        names = self.names or (self.model,)
        return any(model_matches(name, wanted, loaded_file=self.loaded_file)
                   for name in names for wanted in models)

    def held_by_others(self, pid: int) -> list[tuple[int, str]]:
        return [who for who in self.holders.values() if who[0] != pid]

    def said(self) -> dict[str, Any]:
        return {"port": self.port, "model": self.model, "purpose": self.purpose,
                "pid": self.pid, "ours": self.ours, "unmanaged": self.unmanaged,
                "loading": self.loading,
                "base_url": self.base_url,
                "holders": [{**provenance.record_from(self.records.get(lease, {})),
                             "lease": lease, "pid": pid, "label": label}
                            for lease, (pid, label) in self.holders.items()]}


@dataclass
class Waiting:
    """An ask in the queue, and what it is waiting on."""

    lease: str
    ask: Ask
    since: float
    blocked_by: str = ""


@dataclass(frozen=True)
class Grant:
    """A lease: its id and the server it is on."""

    lease: str
    purpose: str
    model: str
    port: int
    base_url: str
    shared: bool

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


class Broker:
    """Every model server on this machine, who holds each one, and who is waiting."""

    def __init__(self, manager: ServerManager | None = None, *, idle_s: float = IDLE_S,
                 room: Callable[[], int | None] = free_memory,
                 alive: Callable[[int | None], bool] = pid_exists,
                 scan: Callable[[], list[dict]] = every_server) -> None:
        self.manager = manager or ServerManager()
        self.scan = scan
        #: whether a server's slots are processing, asked before an idle one is stopped
        self.busy: Callable[[str], bool | None] = busy_now
        #: told which server is stopped and why
        self.say: Callable[[str], Any] = lambda _line: None
        #: who holds what, beside the lease record, so a restart does not forget
        self.held_file = self.manager.state_file.with_name("broker-leases.json")
        self.history_file = self.manager.state_file.with_name("lease-history.ladybug")
        self._ended: list[tuple[str, str, dict[str, Any]]] = []
        self.idle_s = idle_s
        self.room = room
        self.alive = alive
        self.servers: dict[int, Held] = {}
        self.managers: dict[str, ServerManager] = {}
        self.queue: list[Waiting] = []
        self.claims: dict[str, dict[str, Any]] = {}
        #: lease -> (pid, cores, since) for the test runners sharing this machine's cores
        self.cores: dict[str, tuple[int, int, float]] = {}
        self.cpus = os.cpu_count() or 1
        self._unmanaged_told: set[int] = set()
        self._cond = threading.Condition()

    # ------------------------------------------------------------------ servers by spec
    def _manager_for(self, options: Mapping[str, Any]) -> ServerManager:
        """The manager that starts servers the way ``options["backend"]`` says."""
        wanted = dict(options.get("backend") or {})
        if not wanted:
            return self.manager
        key = json.dumps(wanted, sort_keys=True)
        with self._cond:
            if key not in self.managers:
                self.managers[key] = self.manager.with_backend(LlamaServerBackend(**wanted))
            return self.managers[key]

    def start(self, spec: ServerSpec, caller: Caller | None = None, *,
              timeout: float | None = None,
              options: Mapping[str, Any] | None = None) -> ServerInfo:
        """A server for exactly ``spec``, held for the caller until `drop`. One already up
        that serves it is shared; otherwise one is started once the machine has the memory."""
        caller = caller or Caller()
        options = dict(options or {})
        if why := guarded.blocked(spec.model):
            guarded.report("refused", caller=caller.label or f"pid-{caller.pid}", why="model held")
            raise guarded.SentinelRefused(why)
        manager = self._manager_for(options)
        how = Starting(**{**{k: v for k, v in options.items() if k != "backend"},
                          "who": caller.label or who()})
        with grant.broker_grant():
            info = manager._start_server(spec, timeout=timeout, how=how,
                                         on_event=caller.on_event, say=caller.say)
        pid = caller.pid or os.getpid()
        return self._held_by(info, spec, pid, caller.label or who(),
                             provenance.observe(pid, caller.claim))

    def _held_by(self, info: ServerInfo, spec: ServerSpec, pid: int, label: str,
                 record: Mapping[str, Any]) -> ServerInfo:
        lease = uuid.uuid4().hex
        entry = recorded_servers(self.manager.state_file).get(info.port) or {}
        with self._cond:
            held = self.servers.get(info.port)
            if held is None or held.pid != info.pid:
                held = Held(port=info.port, model=str(spec.model), pid=info.pid,
                            ours=not entry.get("unmanaged"), names=(str(spec.model),),
                            info=info, shape=_shape_of(
                                {f.name: getattr(spec, f.name)
                                 for f in dataclasses.fields(spec)}))
                self.servers[info.port] = held
            elif not info.adopted:
                held.info = info
            held.holders[lease] = (pid, label)
            held.records[lease] = dict(record)
            self._write_held()
            self._cond.notify_all()
        return replace(info, lease=lease)

    def drop(self, info: ServerInfo, *, grace_s: float = 5.0) -> None:
        """Let go of the lease ``info`` was granted under. The server is stopped when
        nobody else holds it."""
        keep = False
        with self._cond:
            held = self.servers.get(info.port)
            if held is not None:
                self._end(held, info.lease)
                keep = bool(held.holders or held.loading or not held.ours)
                if not keep:
                    self.servers.pop(info.port, None)
                self._write_held()
                self._cond.notify_all()
        self._flush_ended()
        if keep:
            return
        if held is not None:
            self._stop(held)
        elif not info.adopted and info.pid and self.alive(info.pid):
            self.manager._stop_server(info, grace_s=grace_s)

    def escalate(self, spec: ServerSpec, growth: Growth, caller: Caller | None = None, *,
                 options: Mapping[str, Any] | None = None) -> ServerInfo:
        """Grow the server on ``spec.port`` as ``growth`` says."""
        with grant.broker_grant():
            info = self._manager_for(options or {})._escalate(spec, growth, caller or Caller())
        with self._cond:
            held = self.servers.get(info.port)
            if held is not None:
                held.info, held.pid = info, info.pid
        return info

    def detach(self, info: ServerInfo) -> None:
        """Record the server under its own pid, held by nobody but itself."""
        self.manager._detach(info)
        with self._cond:
            held = self.servers.get(info.port)
            if held is not None and info.pid:
                held.holders[f"up-{info.port}"] = (info.pid, "ml-stack-serve up")
                held.records[f"up-{info.port}"] = held.records.get(info.lease, {})
                self._end(held, info.lease, keep_record=True)
                self._write_held()

    # ------------------------------------------------------------------ leases
    def lease(self, ask: Ask, *, timeout: float) -> Grant:
        """A lease for ``ask``, waiting up to ``timeout`` seconds for its turn. Servers
        started outside the broker since it last looked are taken in first, so an ask never
        loads a second copy of one of them. A card somebody else is measuring is waited on
        like any other holder, so a measurement is never spoiled and never refuses a lease
        that could have had its turn."""
        caller = ask.label or f"pid-{ask.pid}"
        ask = replace(ask, record=provenance.observe(ask.pid, ask.claim, behalf=ask.behalf))
        try:
            grant = self._lease(ask, caller, timeout)
        except BrokerError as why:
            activity.record("model.lease", subject=_named(ask), outcome="refused",
                            refs={"for": caller, "purpose": ask.purpose}, meta={"why": str(why)})
            raise
        activity.record("model.lease", subject=_named(ask), outcome="shared" if grant.shared else "granted",
                        refs={"for": caller, "purpose": ask.purpose, "lease": grant.lease},
                        meta={"port": grant.port, "why": ask.record["reason"], **_flags(ask)})
        return grant

    def _lease(self, ask: Ask, caller: str, timeout: float) -> Grant:
        if why := guarded.caller_blocked(caller):
            raise BrokerError(why)
        guarded.report("lease", caller=caller, purpose=ask.purpose)
        allowed = tuple(m for m in ask.models if not guarded.blocked(m))
        if not allowed:
            guarded.report("refused", caller=caller, why="model held by sentinel")
            raise BrokerError(guarded.blocked(ask.models[0]))
        ask = replace(ask, models=allowed)
        self.adopt()
        deadline = time.monotonic() + timeout
        while True:
            waiting = Waiting(lease=uuid.uuid4().hex, ask=ask, since=time.monotonic())
            with self._cond:
                self.queue.append(waiting)
                turn = self._wait_for_turn(waiting, deadline)
            if isinstance(turn, Grant):
                return turn
            placeholder, evicted = turn
            for held in evicted:
                self.say(f"stopping port {held.port} ({held.model}) to make room for "
                         f"{ask.models[0]} ({ask.purpose})")
                self._stop(held)
            try:
                return self._start(waiting, placeholder)
            except Measuring as why:
                if time.monotonic() >= deadline:
                    raise BrokerError(f"waited for {ask.purpose} ({ask.models[0]}): {why}") from why
                time.sleep(POLL_S)

    def release(self, lease: str) -> bool:
        """Let go of ``lease``. False when nothing held it."""
        with self._cond:
            for held in self.servers.values():
                if lease in held.holders:
                    self._end(held, lease)
                    if not held.holders:
                        held.idle_since = time.monotonic()
                    self._cond.notify_all()
                    self._write_held()
                    found = True
                    break
            else:
                found = False
        self._flush_ended()
        return found

    def _end(self, held: Held, lease: str, *, keep_record: bool = False) -> None:
        """Take ``lease`` off ``held`` and queue its history row. Called with the lock held."""
        held.holders.pop(lease, None)
        record = held.records.pop(lease, None)
        if record is not None and not keep_record:
            self._ended.append((held.model, lease, record))

    def _flush_ended(self) -> None:
        """Write the history rows of the leases that ended since the last call."""
        with self._cond:
            ended, self._ended = self._ended, []
        for model, lease, record in ended:
            try:
                lease_history.append(self.history_file, lease_history.ended_row(model, lease, record))
            except (OSError, RuntimeError, ValueError, KeyError) as why:
                self.say(f"lease {lease[:8]} ended; its history row was not written: {why}")

    def _write_held(self) -> None:
        """Record who holds what, so a broker that restarts does not unload a server its
        holder is still using. Called with the lock held."""
        write_json(self.held_file, {
            str(held.port): {"purpose": held.purpose, "model": held.model,
                             "holders": {lease: list(who) for lease, who in held.holders.items()},
                             "why": held.records}
            for held in self.servers.values() if held.holders and not held.loading})

    def _read_held(self) -> dict[int, dict[str, Any]]:
        """What the last broker recorded, by port; whatever cannot be read is nothing."""
        kept = read_json(self.held_file, {})
        out: dict[int, dict[str, Any]] = {}
        for port, entry in (kept.items() if isinstance(kept, dict) else ()):
            if str(port).isdigit() and isinstance(entry, dict):
                out[int(port)] = entry
        return out

    def _wait_for_turn(self, waiting: Waiting, deadline: float) -> Grant | tuple[Held, list[Held]]:
        ask = waiting.ask
        while True:
            if waiting not in self.queue:
                raise BrokerError(f"the lease for {ask.purpose} was dropped: pid {ask.pid} ended")
            if self._first_for_purpose(waiting):
                match, evict, why = self._decide(ask)
                if match is not None and not match.loading:
                    return self._share(waiting, match)
                if match is None and not why:
                    return self._place(waiting, evict)
                waiting.blocked_by = why
            left = deadline - time.monotonic()
            if left <= 0:
                self.queue.remove(waiting)
                self._cond.notify_all()
                raise BrokerError(f"waited for {ask.purpose} ({ask.models[0]}) and it did not "
                                  f"come free: {waiting.blocked_by or 'queued behind others'}")
            self._cond.wait(min(left, POLL_S))

    def _first_for_purpose(self, waiting: Waiting) -> bool:
        ahead = next(w for w in self.queue if w.ask.purpose == waiting.ask.purpose)
        return ahead is waiting

    def _decide(self, ask: Ask) -> tuple[Held | None, list[Held], str]:
        """``(server to share, servers to stop first, why it must wait)``."""
        # A shape asked for by `up` is shared only with a server held for that same shape:
        # taking in one that happens to serve the model is how a caller got the wrong slots.
        exact = ask.purpose.startswith("serve:")
        mine = [h for h in self.servers.values() if not h.unmanaged
                and (h.purpose == ask.purpose
                     or (not h.purpose and not exact and h.serves(ask.models)))]
        fits = [h for h in mine if h.serves(ask.models) and not h.short_of(ask.spec)]
        match = next(iter(fits), None)
        if match is None:
            mine, short = [h for h in mine if h not in fits], [
                h.short_of(ask.spec) for h in mine if h.serves(ask.models) and h.short_of(ask.spec)]
            if short and not any(h.held_by_others(ask.pid) or h.loading for h in mine):
                self.say(f"{ask.models[0]}: " + "; ".join(short) + "; starting one that fits")
        if match is not None:
            return match, [], f"{match.model} is loading on port {match.port}" if match.loading else ""
        busy = [h for h in mine if h.loading or h.held_by_others(ask.pid)]
        if busy:
            return None, [], "; ".join(
                [*(h.short_of(ask.spec) for h in busy if h.serves(ask.models)
                   and h.short_of(ask.spec)), *(self._held_said(h) for h in busy)])
        evict = [h for h in mine if h.ours]
        return self._fit(ask, evict)

    def _fit(self, ask: Ask, evict: list[Held]) -> tuple[None, list[Held], str]:
        need, room = ask.weight or weight_of(ask.models[0]), self.room()
        if not need or room is None:
            return None, evict, ""
        freed = sum(h.weight for h in evict)
        spare = sorted((h for h in self.servers.values() if h.ours and not h.loading
                        and not h.holders and h not in evict),
                       key=lambda h: h.idle_since)
        while need > (room + freed) * BESIDE_HEADROOM and spare:
            evict.append(spare.pop(0))
            freed += evict[-1].weight
        if need <= (room + freed) * BESIDE_HEADROOM:
            return None, evict, ""
        held = [h for h in self.servers.values() if h.holders or h.loading]
        if not held:
            return None, evict, ""
        return None, [], (f"{ask.models[0]} needs {need >> 20} MiB and {room >> 20} MiB is "
                          f"free; " + "; ".join(self._held_said(h) for h in held))

    @staticmethod
    def _held_said(held: Held) -> str:
        who = ", ".join(f"pid {pid}" + (f" ({label})" if label else "")
                        for pid, label in held.holders.values()) or "loading"
        return f"port {held.port} serves {held.model} for {held.purpose or 'no purpose'}, held by {who}"

    def _share(self, waiting: Waiting, held: Held) -> Grant:
        ask = waiting.ask
        held.purpose = held.purpose or ask.purpose
        lease = next((lease for lease, (pid, _) in held.holders.items() if pid == ask.pid),
                     waiting.lease)
        held.holders[lease] = (ask.pid, ask.label)
        held.records[lease] = dict(ask.record)
        self.queue.remove(waiting)
        self._write_held()
        self._cond.notify_all()
        return Grant(lease=lease, purpose=ask.purpose, model=held.model, port=held.port,
                     base_url=held.base_url, shared=True)

    def _place(self, waiting: Waiting, evict: list[Held]) -> tuple[Held, list[Held]]:
        ask = waiting.ask
        for held in evict:
            self.servers.pop(held.port, None)
        for held in [h for h in self.servers.values() if h.purpose == ask.purpose]:
            for lease in [k for k, v in held.holders.items() if v[0] == ask.pid]:
                self._end(held, lease)
        asked = ask.port if ask.port and port_is_free(ask.port) and not any(
            h.port == ask.port for h in self.servers.values()) else 0
        placeholder = Held(port=asked or free_port(), model=ask.models[0], purpose=ask.purpose,
                           loading=True, weight=ask.weight or weight_of(ask.models[0]),
                           shape=_shape_of(ask.spec),
                           holders={waiting.lease: (ask.pid, ask.label)},
                           records={waiting.lease: dict(ask.record)})
        self.servers[placeholder.port] = placeholder
        self.queue.remove(waiting)
        return placeholder, evict

    def _start(self, waiting: Waiting, placeholder: Held) -> Grant:
        spec = waiting.ask.server_spec(placeholder.port)
        try:
            with grant.broker_grant():
                info = self._manager_for(waiting.ask.options)._start_server(
                    spec, how=Starting(roam=False, who=waiting.ask.label))
        except Measuring:
            with self._cond:
                self.servers.pop(placeholder.port, None)
                self._cond.notify_all()
            raise
        except (ServerFailed, PreflightFailed, OSError, TypeError, ValueError) as exc:
            with self._cond:
                self.servers.pop(placeholder.port, None)
                self._cond.notify_all()
            raise BrokerError(f"{spec.model} did not start on port {spec.port}: {exc}") from exc
        params = serving_params(info.base_url)
        with self._cond:
            placeholder.info, placeholder.pid, placeholder.loading = info, info.pid, False
            placeholder.names = (str(spec.model), *reported_models(info.base_url))
            placeholder.loaded_file = params.model if params else None
            self._write_held()
            self._cond.notify_all()
        ask = waiting.ask
        return Grant(lease=waiting.lease, purpose=ask.purpose, model=placeholder.model,
                     port=placeholder.port, base_url=info.base_url, shared=False)

    def _stop(self, held: Held) -> None:
        if held.info is not None and not held.info.adopted:
            self.manager._stop_server(held.info)
        elif held.pid and self.alive(held.pid):
            kill_process_tree(held.pid)

    def stop(self, port: int, *, force: bool = False) -> dict[str, Any]:
        """Stop the server on ``port``. Refused while anyone holds it, unless ``force``."""
        with self._cond:
            held = self.servers.get(port)
            if held is None:
                return {"stopped": False, "why": f"nothing the broker knows is on port {port}"}
            if not held.ours:
                return {"stopped": False, "why": f"port {port} was not started through ml-stack"}
            if (held.holders or held.loading) and not force:
                return {"stopped": False, "why": self._held_said(held)}
            self.servers.pop(port)
            for lease in list(held.holders):
                self._end(held, lease)
            self._cond.notify_all()
        self._flush_ended()
        self._stop(held)
        return {"stopped": True, "port": port, "model": held.model}

    # ------------------------------------------------------------------ claims
    def claim(self, name: str, pid: int, info: Mapping[str, Any], *,
              timeout: float = 0.0) -> dict[str, Any]:
        """Hold ``name`` for ``pid``, or say who holds it. Waits up to ``timeout`` for it."""
        deadline = time.monotonic() + timeout
        with self._cond:
            while True:
                held = self.claims.get(name)
                if held is None or held["pid"] == pid or not self.alive(held["pid"]):
                    self.claims[name] = {"pid": pid, "info": dict(info), "since": time.time()}
                    return {"granted": True, **self.claims[name]}
                left = deadline - time.monotonic()
                if left <= 0:
                    return {"granted": False, **held}
                self._cond.wait(min(left, POLL_S))

    def unclaim(self, name: str, pid: int) -> bool:
        """Let go of ``name`` if ``pid`` holds it."""
        with self._cond:
            held = self.claims.get(name)
            if held is None or held["pid"] != pid:
                return False
            del self.claims[name]
            self._cond.notify_all()
            return True

    # ------------------------------------------------------------------ cores
    def take_cores(self, pid: int, want: int) -> dict[str, Any]:
        """Cores for ``pid`` to run tests on: ``want`` at most, what is free at least one.

        Never refuses and never waits -- a suite that cannot have what it asked for runs
        narrower rather than queueing.
        """
        with self._cond:
            self.cores = {k: v for k, v in self.cores.items() if self.alive(v[0])}
            taken = sum(cores for _, cores, _ in self.cores.values())
            cores = max(1, min(max(1, want), self.cpus - taken))
            lease = uuid.uuid4().hex
            self.cores[lease] = (pid, cores, time.time())
            return {"lease": lease, "cores": cores, "cpus": self.cpus, "taken": taken}

    def give_back_cores(self, lease: str) -> bool:
        """Let go of a core grant. False when nothing held it."""
        with self._cond:
            return self.cores.pop(lease, None) is not None

    # ------------------------------------------------------------------ upkeep
    def reap(self) -> list[Held]:
        """Drop every lease, wait and claim whose pid has ended, forget servers that have
        died, and stop the idle ones of ours. Returns the servers stopped."""
        now = time.monotonic()
        with self._cond:
            for held in list(self.servers.values()):
                for lease, (pid, _) in list(held.holders.items()):
                    if not self.alive(pid):
                        self._end(held, lease)
                        held.idle_since = now
                if not held.loading and held.pid and not self.alive(held.pid):
                    for lease in list(held.holders):
                        self._end(held, lease)
                    del self.servers[held.port]
            self.queue = [w for w in self.queue if self.alive(w.ask.pid)]
            self.claims = {k: v for k, v in self.claims.items() if self.alive(v["pid"])}
            self.cores = {k: v for k, v in self.cores.items() if self.alive(v[0])}
            idle = [h for h in self.servers.values() if h.ours and not h.holders
                    and not h.loading and now - h.idle_since > self.idle_s]
        answering = [h for h in idle if self.busy(h.base_url)]
        with self._cond:
            for held in answering:
                held.idle_since = time.monotonic()
            idle = [h for h in idle if h not in answering
                    and self.servers.get(h.port) is h and not h.holders]
            for held in idle:
                del self.servers[held.port]
            self._write_held()
            self._cond.notify_all()
        self._flush_ended()
        for held in idle:
            self.say(f"stopping port {held.port} ({held.model}): nobody has held it for "
                     f"{now - held.idle_since:.0f}s and it is answering nothing")
            self._stop(held)
        return idle

    def adopt(self) -> list[Held]:
        """Take in every server already running: one on the lease record is ours, held by
        its recorded owner while that process lives; any other llama-server is shared but
        never stopped."""
        now, found = time.monotonic(), []
        me, kept = os.getpid(), self._read_held()
        for port, entry in recorded_servers(self.manager.state_file).items():
            pid, owner = entry.get("pid"), entry.get("owner_pid")
            if not (isinstance(pid, int) and self.alive(pid)):
                continue
            was = kept.get(port, {})
            holders = {lease: (int(who[0]), str(who[1])) for lease, who in (was.get("holders") or {}).items()
                       if self.alive(int(who[0]))}
            why = {lease: rec for lease, rec in (was.get("why") or {}).items() if lease in holders}
            if not holders and isinstance(owner, int) and owner not in (pid, me) and self.alive(owner):
                holders = {f"recorded-{port}": (owner, "recorded owner")}
            if owner == pid:
                holders[f"up-{port}"] = (pid, "ml-stack-serve up")
            found.append(Held(port=port, model=str(entry.get("model") or ""), pid=pid,
                              ours=not entry.get("unmanaged"), purpose=str(was.get("purpose") or ""), idle_since=now, holders=holders,
                              records=why,
                              shape=_shape_of(entry),
                              info=ServerInfo(base_url=f"http://{DEFAULT_HOST}:{port}",
                                              port=port, pid=pid, backend="", adopted=True)))
        known = {h.port for h in found} | set(self.servers)
        for proc in self.scan():
            if proc["port"] not in known and not proc["defunct"]:
                taken = self._take_in(proc)
                if not taken and proc["port"] not in self._unmanaged_told:
                    self._unmanaged_told.add(proc["port"])
                    guarded.unmanaged_seen([proc])
                found.append(Held(port=proc["port"], model=proc["model"], pid=proc["pid"],
                                  ours=False, unmanaged=not taken, weight=proc["rss"],
                                  idle_since=now))
        with self._cond:
            for held in found:
                if held.port in self.servers or not is_healthy(held.base_url, timeout=2.0):
                    continue
                held.names = (held.model, *reported_models(held.base_url))
                params = serving_params(held.base_url)
                held.loaded_file = params.model if params else None
                self.servers[held.port] = held
            self._write_held()
            self._cond.notify_all()
        return found

    def _take_in(self, proc: Mapping[str, Any]) -> bool:
        """Adopt the unmanaged server ``proc`` when the setting is ``auto`` and it passes the
        checks; whether it was adopted."""
        if unmanaged.mode() != "auto":
            return False
        seen = unmanaged.examine(int(proc["port"]))
        if not seen.ok:
            self.say(f"port {proc['port']}: not adopting pid {proc['pid']} -- {seen.why}")
            return False
        self.manager.register_unmanaged(int(proc["port"]), seen)
        self.say(f"port {proc['port']}: adopted pid {seen.pid} serving {seen.model} "
                 f"({unmanaged.ENV}=auto)")
        return True

    def canary_targets(self) -> list[canaries.Target]:
        """The servers this broker started and holds, each to be asked through a lease so the
        broker knows who is using it. A lease that would start another server is let go unused."""
        out = []
        for one in self.snapshot()["servers"]:
            if one["ours"] and not one["unmanaged"] and not one["loading"]:
                out.append(canaries.Target(str(one["model"]), self._canary_lease(one)))
        return out

    def _canary_lease(self, one: dict[str, Any]) -> Callable[[], Any]:
        @contextlib.contextmanager
        def using() -> Iterator[str]:
            grant = self.lease(Ask(purpose=one["purpose"] or "chat", models=(one["model"],),
                                   pid=os.getpid(), label="sentinel-canary",
                                   claim=provenance.asked("sentinel canary check", "sentinel")),
                               timeout=canaries.LEASE_WAIT_S)
            try:
                if grant.port != one["port"]:
                    raise BrokerError("the canary lease was for another server")
                yield grant.base_url
            finally:
                self.release(grant.lease)
        return using

    def supervising(self) -> bool:
        """Whether anything is the broker's to look after: a server of ours, a held or loading
        one, a waiting ask, a claim or a core grant."""
        with self._cond:
            return bool(self.queue or self.claims or self.cores
                        or any(h.ours or h.holders or h.loading for h in self.servers.values()))

    def snapshot(self) -> dict[str, Any]:
        """Servers, their holders, the queue and the claims, for a person looking."""
        now = time.monotonic()
        with self._cond:
            return {
                "servers": [h.said() for h in sorted(self.servers.values(), key=lambda h: h.port)],
                "queue": [{"lease": w.lease, "purpose": w.ask.purpose, "model": w.ask.models[0],
                           "pid": w.ask.pid, "label": w.ask.label,
                           "reason": w.ask.record.get("reason") or provenance.NO_REASON,
                           "requester": w.ask.record.get("requester") or "(unknown)",
                           "waited_s": round(now - w.since, 1), "blocked_by": w.blocked_by}
                          for w in self.queue],
                "claims": {k: dict(v) for k, v in self.claims.items()},
                "requests": gate.snapshot(),
                "cores": {"cpus": self.cpus,
                          "grants": [{"lease": lease, "pid": pid, "cores": cores}
                                     for lease, (pid, cores, _since) in self.cores.items()]},
            }
