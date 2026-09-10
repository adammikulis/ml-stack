"""A model sweep spread over the fleet: `plan` puts each model on the idle peer with the
most room that fits it, `dispatch` sends the jobs, `wait` polls the daemons, and `gather`
brings every peer's runs home into one store.

`submit_bench` and `bench_export` are the two calls a peer answers, over its daemon or, for
a `fleet.measuring.Local`, in this process.
"""

from __future__ import annotations

import json
import re
import time
import urllib.parse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ml_stack.log import say
from ml_stack.units import human_bytes

from .gathering import import_runs
from .measuring import TAIL, Job, Local, Refused
from .remote import PeerError
from .sizing import estimate

__all__ = ["Handle", "Plan", "bench_export", "dispatch", "gather", "plan", "submit_bench",
           "wait"]


def submit_bench(peer: Any, job: Job) -> dict[str, Any]:
    """``POST /bench`` on ``peer``: the daemon's job record, or `Refused`."""
    if isinstance(peer, Local):
        return peer.host.submit(job).public()
    try:
        return peer._json("POST", "/bench", job.public())
    except PeerError as exc:
        answered = _answered(str(exc))
        if answered and answered[0] == 409:
            raise Refused(str(answered[1].get("refused") or "refused"),
                          str(answered[1].get("error") or exc)) from None
        raise


def bench_export(peer: Any, *, since: str = "", job: str = "", full: bool = True,
                 anyway: bool = False) -> dict[str, Any]:
    """``GET /bench/export`` on ``peer``: its runs since ``since`` or since ``job`` began."""
    if isinstance(peer, Local):
        return {**peer.host.export(since=since, job=job, full=full, anyway=anyway),
                "host": peer.name}
    query = urllib.parse.urlencode({k: v for k, v in (("since", since), ("job", job),
                                                        ("full", "1" if full else ""),
                                                        ("anyway", "1" if anyway else ""))
                                    if v})
    return peer._json("GET", f"/bench/export?{query}")


def _answered(message: str) -> tuple[int, dict[str, Any]] | None:
    """The status and JSON body out of a `PeerError`'s message, or None."""
    found = re.search(r"-> (\d{3}): (\{.*\})", message, re.S)
    if not found:
        return None
    try:
        return int(found.group(1)), json.loads(found.group(2))
    except ValueError:
        return None


# -- dispatch ------------------------------------------------------------------------
class Plan(dict):
    """``{peer: [model, ...]}`` -- and ``unplaced``, the models that fit nowhere, each with
    why not on every peer, so nothing is dropped silently."""

    def __init__(self) -> None:
        super().__init__()
        self.unplaced: list[tuple[str, str]] = []


def plan(models: Sequence[str], peers: Sequence[Any], *, needs: Mapping[str, int] | None = None,
         context: int = 32768, log: Callable[[str], None] = say) -> Plan:
    """Which peer serves which model.

    Each peer is asked ``health()`` for its name, whether it is idle (not busy, a free slot,
    not measuring) and its ``room_bytes``. Each model -- sized by ``needs`` when given, else
    by `estimate` at ``context`` -- goes, largest first, to an idle peer with room for it:
    the one holding the fewest models so far, the roomiest on a tie, so two models land on
    two machines rather than both on the biggest. A model nobody fits is listed under
    ``unplaced`` with every peer's reason, and printed. ``peers`` may include `here()`.
    """
    sized = dict(needs or {})
    for model in models:
        if model not in sized:
            sized[model] = int(estimate(model, context=context))
    seen: list[tuple[Any, str, dict[str, Any]]] = []
    for peer in peers:
        try:
            health = peer.health()
        except Exception as exc:  # noqa: BLE001 - a peer that does not answer is not idle
            log(f"  {getattr(peer, 'name', peer)}: did not answer ({exc})")
            continue
        seen.append((peer, _name_of(peer, health), health))
    out = Plan()
    for peer, _name, _health in seen:
        out[peer] = []
    log(f"planning {len(models)} model(s) over {len(seen)} peer(s):")
    for model in sorted(models, key=lambda m: -sized.get(m, 0)):
        need = sized.get(model, 0)
        fitting: list[tuple[Any, str, int]] = []
        reasons: list[str] = []
        for peer, name, health in seen:
            room = int(health.get("room_bytes") or 0)
            why = _not_idle(health)
            if why:
                reasons.append(f"{name}: {why}")
            elif room and need and need > room:
                reasons.append(f"{name}: room {human_bytes(room)} < {human_bytes(need)}")
            else:
                fitting.append((peer, name, room))
        if not fitting:
            said = "; ".join(reasons) or "no peers answered"
            out.unplaced.append((model, said))
            log(f"  {model:<40} {human_bytes(need):>7} fits nowhere: {said}")
            continue
        peer, name, room = min(fitting, key=lambda f: (len(out[f[0]]), -f[2]))
        out[peer].append(model)
        log(f"  {model:<40} {human_bytes(need):>7} -> {name} (room {human_bytes(room) if room else '?'})"
            + ("" if need else "  [size unknown; the peer's preflight decides]"))
    return out


def _name_of(peer: Any, health: Mapping[str, Any] | None = None) -> str:
    """What a peer calls itself: its beacon's name, else what ``/health`` says, else
    whatever it is addressed by -- a `Peer` made from a URL alone is named by the URL."""
    beacon = getattr(peer, "beacon", None)
    if beacon is not None and getattr(beacon, "name", ""):
        return str(beacon.name)
    if health is None:
        try:
            health = peer.health()
        except Exception:  # noqa: BLE001
            health = {}
    return str(health.get("name") or getattr(peer, "name", peer))


def _not_idle(health: Mapping[str, Any]) -> str:
    if health.get("measuring"):
        return "measuring"
    if health.get("busy"):
        return "busy"
    if int(health.get("free", 1) or 0) < 1:
        return "no free slot"
    return ""


@dataclass
class Handle:
    """One dispatched job: where it went, the daemon's id for it, and how it stands."""

    peer: Any
    job: Job
    id: str = ""
    name: str = ""
    state: str = "pending"
    """pending | running | done | failed | stopped | refused"""
    log: str = ""
    why: str = ""
    host: str = ""
    """What the peer calls itself, which is what `gather` files its runs under."""

    @property
    def peer_name(self) -> str:
        return self.host or str(getattr(self.peer, "name", self.peer))

    @property
    def ended(self) -> bool:
        return self.state in ("done", "failed", "stopped", "refused")


def dispatch(jobs: Mapping[Any, Job], *, log: Callable[[str], None] = say) -> list[Handle]:
    """Send each job to its peer. A refusal is a `Handle` in state ``refused`` with the
    reason, printed, not an exception: the rest of the sweep still goes out."""
    out: list[Handle] = []
    for peer, job in jobs.items():
        handle = Handle(peer=peer, job=job, host=_name_of(peer))
        try:
            answered = submit_bench(peer, job)
        except Refused as why:
            handle.state, handle.why = "refused", f"{why.kind}: {why}"
            log(f"  {handle.peer_name}: refused ({why.kind}) -- {why}")
        except Exception as exc:  # noqa: BLE001 - one unreachable peer must not stop the rest
            handle.state, handle.why = "refused", f"unreachable: {exc}"
            log(f"  {handle.peer_name}: unreachable -- {exc}")
        else:
            handle.id = str(answered.get("id") or "")
            handle.name = str(answered.get("name") or job.name)
            handle.state = str(answered.get("state") or "running")
            handle.log = str(answered.get("log") or "")
            log(f"  {handle.peer_name}: {handle.name} {handle.state} (job {handle.id}"
                + (f", log {handle.log}" if handle.log else "") + ")")
        out.append(handle)
    return out


def wait(handles: Sequence[Handle], *, poll_s: float = 20.0, timeout_s: float | None = None,
         log: Callable[[str], None] = say) -> list[Handle]:
    """Poll each peer's job until every one has ended, printing one line per change and
    the last `TAIL` lines of the log when a job ends. ``timeout_s`` bounds the whole wait;
    a job still running then is left in state ``running`` and said so."""
    deadline = time.monotonic() + timeout_s if timeout_s else None
    while True:
        for handle in handles:
            if handle.ended:
                continue
            try:
                current = handle.peer.job(handle.id)
            except Exception as exc:  # noqa: BLE001 - said, and asked again next round
                log(f"  {handle.peer_name}: could not read job {handle.id}: {exc}")
                continue
            state = str(current.get("state") or "")
            if state == handle.state:
                continue
            handle.state = state
            log(f"  {handle.peer_name}: {handle.name} {state}")
            if handle.ended:
                try:
                    tail = handle.peer.log(handle.id, tail=TAIL).rstrip()
                except Exception:  # noqa: BLE001
                    tail = ""
                for line in tail.splitlines():
                    log(f"      {line}")
        if all(h.ended for h in handles):
            return list(handles)
        if deadline is not None and time.monotonic() > deadline:
            for handle in handles:
                if not handle.ended:
                    log(f"  {handle.peer_name}: {handle.name} still {handle.state} after "
                        f"{timeout_s:.0f}s; left running")
            return list(handles)
        time.sleep(poll_s)




def gather(handles: Sequence[Handle], *, into: str | Path,
           log: Callable[[str], None] = say) -> dict[str, list[str]]:
    """Bring home what each dispatched job measured: every peer's runs kept since its job
    started, imported into ``into`` by `import_runs` with the peer's name as host and the
    peer's commit. A refused or never-started job has nothing to gather. Returns the keys
    written per peer; a peer whose export holds nothing is said, since a job marked done
    that kept no run is the thing worth noticing."""
    out: dict[str, list[str]] = {}
    for handle in handles:
        if not handle.id:
            continue
        try:
            answered = bench_export(handle.peer, job=handle.id, full=True)
        except Exception as exc:  # noqa: BLE001 - said, and the others still come home
            log(f"  {handle.peer_name}: could not export: {exc}")
            continue
        host = str(answered.get("host") or handle.peer_name)
        if not answered.get("runs"):
            log(f"  {host}: kept no run since {answered.get('since', '?')}"
                + (f" ({answered['skipped']} not over the invented community)"
                   if answered.get("skipped") else ""))
            out[host] = []
            continue
        out[host] = import_runs(answered, into, host=host,
                                commit=str(answered.get("commit") or ""), log=log)
    return out
