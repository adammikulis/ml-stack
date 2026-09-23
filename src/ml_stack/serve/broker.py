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

import dataclasses
import os
import threading
import time
import uuid
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from ml_stack.client import is_healthy, reported_models, serving_params
from ml_stack.files import read_json, write_json
from ml_stack.hub import free_memory
from ml_stack.serve.backend import ServerFailed, ServerInfo, ServerSpec
from ml_stack.serve.leases import recorded_servers
from ml_stack.serve.manager import BESIDE_HEADROOM, Measuring, ServerManager
from ml_stack.serve.matching import model_matches
from ml_stack.serve.ports import DEFAULT_HOST, free_port
from ml_stack.serve.process import every_server, kill_process_tree, pid_exists
from ml_stack.serve.reclaim import busy_now
from ml_stack.serve.weights import weight_of

__all__ = ["Ask", "Broker", "BrokerError", "Grant", "Held", "Waiting"]

IDLE_S = 600.0
POLL_S = 0.5
_SPEC_FIELDS = frozenset(f.name for f in dataclasses.fields(ServerSpec)) - {"model", "port"}


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
                   spec=spec)

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
    loading: bool = False
    weight: int = 0
    idle_since: float = 0.0
    holders: dict[str, tuple[int, str]] = field(default_factory=dict)
    info: ServerInfo | None = None
    #: what ``/props`` said this server actually loaded (its ``model_path``), fetched
    #: once when the server was found rather than per ask -- ``None`` when it could
    #: not be read.
    loaded_file: str | None = None

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
                "pid": self.pid, "ours": self.ours, "loading": self.loading,
                "base_url": self.base_url,
                "holders": [{"lease": lease, "pid": pid, "label": label}
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
                 scan: Callable[[], list[dict]] = every_server,
                 busy: Callable[[str], bool | None] = busy_now,
                 say: Callable[[str], Any] = lambda _line: None) -> None:
        self.manager = manager or ServerManager()
        self.scan = scan
        self.busy = busy
        self.say = say
        #: who holds what, beside the lease record, so a restart does not forget
        self.held_file = self.manager.state_file.with_name("broker-leases.json")
        self.idle_s = idle_s
        self.room = room
        self.alive = alive
        self.servers: dict[int, Held] = {}
        self.queue: list[Waiting] = []
        self.claims: dict[str, dict[str, Any]] = {}
        #: lease -> (pid, cores, since) for the test runners sharing this machine's cores
        self.cores: dict[str, tuple[int, int, float]] = {}
        self.cpus = os.cpu_count() or 1
        self._cond = threading.Condition()

    # ------------------------------------------------------------------ leases
    def lease(self, ask: Ask, *, timeout: float) -> Grant:
        """A lease for ``ask``, waiting up to ``timeout`` seconds for its turn. Servers
        started outside the broker since it last looked are taken in first, so an ask never
        loads a second copy of one of them. A card somebody else is measuring is waited on
        like any other holder, so a measurement is never spoiled and never refuses a lease
        that could have had its turn."""
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
                if held.holders.pop(lease, None) is not None:
                    if not held.holders:
                        held.idle_since = time.monotonic()
                    self._cond.notify_all()
                    self._write_held()
                    return True
        return False

    def _write_held(self) -> None:
        """Record who holds what, so a broker that restarts does not unload a server its
        holder is still using. Called with the lock held."""
        write_json(self.held_file, {
            str(held.port): {"purpose": held.purpose, "model": held.model,
                             "holders": {lease: list(who) for lease, who in held.holders.items()}}
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
        mine = [h for h in self.servers.values()
                if h.purpose == ask.purpose or (not h.purpose and h.serves(ask.models))]
        match = next((h for h in mine if h.serves(ask.models)), None)
        if match is not None:
            return match, [], f"{match.model} is loading on port {match.port}" if match.loading else ""
        busy = [h for h in mine if h.loading or h.held_by_others(ask.pid)]
        if busy:
            return None, [], "; ".join(self._held_said(h) for h in busy)
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
            held.holders = {k: v for k, v in held.holders.items() if v[0] != ask.pid}
        placeholder = Held(port=free_port(), model=ask.models[0], purpose=ask.purpose,
                           loading=True, weight=ask.weight or weight_of(ask.models[0]),
                           holders={waiting.lease: (ask.pid, ask.label)})
        self.servers[placeholder.port] = placeholder
        self.queue.remove(waiting)
        return placeholder, evict

    def _start(self, waiting: Waiting, placeholder: Held) -> Grant:
        spec = waiting.ask.server_spec(placeholder.port)
        try:
            info = self.manager.lease(spec, roam=False)
        except Measuring:
            with self._cond:
                self.servers.pop(placeholder.port, None)
                self._cond.notify_all()
            raise
        except (ServerFailed, OSError, TypeError, ValueError) as exc:
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
            self.manager.release(held.info)
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
            self._cond.notify_all()
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
                        del held.holders[lease]
                        held.idle_since = now
                if not held.loading and held.pid and not self.alive(held.pid):
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
            if not holders and isinstance(owner, int) and owner not in (pid, me) and self.alive(owner):
                holders = {f"recorded-{port}": (owner, "recorded owner")}
            if owner == pid:
                holders[f"up-{port}"] = (pid, "ml-stack-serve up")
            found.append(Held(port=port, model=str(entry.get("model") or ""), pid=pid,
                              purpose=str(was.get("purpose") or ""), idle_since=now, holders=holders,
                              info=ServerInfo(base_url=f"http://{DEFAULT_HOST}:{port}",
                                              port=port, pid=pid, backend="", adopted=True)))
        known = {h.port for h in found} | set(self.servers)
        for proc in self.scan():
            if proc["port"] not in known and not proc["defunct"]:
                found.append(Held(port=proc["port"], model=proc["model"], pid=proc["pid"],
                                  ours=False, weight=proc["rss"], idle_since=now))
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

    def snapshot(self) -> dict[str, Any]:
        """Servers, their holders, the queue and the claims, for a person looking."""
        now = time.monotonic()
        with self._cond:
            return {
                "servers": [h.said() for h in sorted(self.servers.values(), key=lambda h: h.port)],
                "queue": [{"lease": w.lease, "purpose": w.ask.purpose, "model": w.ask.models[0],
                           "pid": w.ask.pid, "label": w.ask.label,
                           "waited_s": round(now - w.since, 1), "blocked_by": w.blocked_by}
                          for w in self.queue],
                "claims": {k: dict(v) for k, v in self.claims.items()},
                "cores": {"cpus": self.cpus,
                          "grants": [{"lease": lease, "pid": pid, "cores": cores}
                                     for lease, (pid, cores, _since) in self.cores.items()]},
            }
