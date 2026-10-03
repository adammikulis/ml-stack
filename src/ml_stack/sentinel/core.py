"""The sentinel: detectors, the policy and the quarantine store joined to one event bus."""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ml_stack.lock import pid_alive
from ml_stack.sentinel import canary as canaries
from ml_stack.sentinel.canary import Baseline, Results
from ml_stack.sentinel.events import Bus, Event, EventLog, Severity
from ml_stack.sentinel.findings import Finding
from ml_stack.sentinel.heads_up import HeadsUp
from ml_stack.sentinel.honey import Honey
from ml_stack.sentinel.human import HumanGrant, protect
from ml_stack.sentinel.integrity import Manifest, Pin, check_file
from ml_stack.sentinel.policy import Mode, decide
from ml_stack.sentinel.rails import RailWatch, reads_like_instruction
from ml_stack.sentinel.rates import Abuse, PeerWatch, ToolMix
from ml_stack.sentinel.redaction import redact
from ml_stack.sentinel.score import Score
from ml_stack.sentinel.sealed import SealedFile
from ml_stack.sentinel.store import (
    KINDS,
    Holding,
    Record,
    State,
    Store,
    fingerprint,
    placeholder,
    sentinel_dir,
)
from ml_stack.sentinel.watch import scanner_state

__all__ = ["ENV", "Screened", "Sentinel"]

ENV = "ML_STACK_SENTINEL"
BECAUSE = "ML_STACK_SENTINEL_BECAUSE"
MISSING = "integrity.missing"
GONE_SCANS = 3
"""Scans in a row a pinned file may be gone before its pin is dropped."""


@dataclass(frozen=True, slots=True)
class Screened:
    """What a model may see in place of some text: ``text`` is the text or a placeholder,
    ``withheld`` the quarantine id when it was held."""

    text: str
    withheld: str = ""
    labelled: bool = False


class Sentinel:
    """One node's sentinel. Detectors call `handle` with findings; the policy decides
    whether a finding is recorded, watched or acted on."""

    def __init__(self, root: Path | None = None, *, mode: Mode | None = None,
                 dry_run: bool = False, roots: list[Path] | None = None,
                 clock: Callable[[], float] = time.time) -> None:
        self.root = Path(root) if root is not None else sentinel_dir()
        self.clock = clock
        protect(self.root)
        self.bus = Bus(EventLog(self.root / "events.log", anchor=self.root / "anchor.log"))
        self.store = Store(self.root, self.bus, roots=roots, clock=clock)
        self._config = SealedFile(self.root / "config.json")
        self.off_because, self._refused_off = "", False
        self.mode = mode or self._configured_mode()
        self.dry_run = dry_run
        self.manifest = Manifest(self.root / "manifest.json", clock)
        self.baselines = Baseline(self.root / "canaries.json")
        self.honey = Honey(state=self.root / "honey.json")
        self.peers, self.tools, self.rails = PeerWatch(), ToolMix(), RailWatch()
        self.abuse = Abuse()
        self.score = Score()
        self._flights: dict[str, int] = {}
        self._flight_lock = threading.Lock()
        self.last_scan = 0.0
        self._derived: dict[str, list[str]] = {}
        self.store.on_quarantine.setdefault("session", []).append(self._taint_derived)
        self.heads_up = HeadsUp(self.root / "notified.json", clock=clock, store=self.store,
                                bus=self.bus)
        self._gone: dict[str, int] = {}
        for kind in KINDS:
            self.store.on_quarantine.setdefault(kind, []).append(self.heads_up.on_quarantine)
        self._stop = threading.Event()
        self._verified = SealedFile(self.root / "verified.json")
        if self.off_because:
            self.bus.emit(Event("sentinel.opt_out", Severity.WARNING, "core", "opt_out:sentinel",
                                {"because": self.off_because}, clock()))
        elif mode is None and self._refused_off:
            self.bus.emit(Event("sentinel.off_refused", Severity.WARNING, "core", "",
                                {"why": f"{ENV}=off needs {BECAUSE}"}, clock()))
        if self.mode == Mode.OFF and not self.off_because:
            self.bus.emit(Event("sentinel.off", Severity.WARNING, "core", "", {}, clock()))

    def _configured_mode(self) -> Mode:
        named = os.environ.get(ENV, "").strip().lower()
        if named == Mode.OFF.value:
            self.off_because = os.environ.get(BECAUSE, "").strip()
            if self.off_because:
                return Mode.OFF
            self._refused_off = True
        elif named in Mode:
            return Mode(named)
        saved = self._config.load().payload.get("mode", "")
        return Mode(saved) if saved in Mode else Mode.GUARDED

    def set_mode(self, mode: Mode, grant: HumanGrant) -> None:
        """Change the mode. Needs a grant for ``mode`` on ``sentinel``."""
        grant.check("mode", "sentinel")
        self._config.save({"mode": mode.value})
        self.bus.emit(Event("sentinel.mode", Severity.NOTICE, "core", "",
                            {"from": self.mode.value, "to": mode.value}, self.clock()))
        self.mode = mode

    # -- findings --------------------------------------------------------------------
    def handle(self, found: Finding) -> str:
        """Record a finding and do what the policy says. Returns ``ignore``, ``watch``,
        ``act`` or ``would-act`` (dry run)."""
        what = decide(found, self.mode)
        if what == "ignore":
            return what
        self.bus.emit(found.event)
        reason = f"{found.event.kind}: {_summary(found.event.evidence)}"
        if what == "act" and self.dry_run:
            self.bus.emit(Event("policy.would_quarantine", Severity.NOTICE, "policy",
                                found.event.subject, {"kind": found.event.kind}, self.clock()))
            return "would-act"
        if what == "act":
            self.store.quarantine((found.kind, found.key), reason, dict(found.event.evidence),
                                  Holding(found.text, found.path or None, found.move))
        else:
            self.store.watch(found.kind, found.key, reason, dict(found.event.evidence))
        return what

    def handle_all(self, found: Iterable[Finding | None]) -> list[str]:
        return [self.handle(f) for f in found if f is not None]

    # -- what callers ask ------------------------------------------------------------
    def peer_blocked(self, peer: str) -> bool:
        return self.mode != Mode.OFF and self.store.blocked("peer", peer)

    def tool_allowed(self, name: str, caller: str = "") -> bool:
        if self.mode == Mode.OFF:
            return True
        return not (self.store.blocked("tool", name)
                    or (caller and self.store.blocked("caller", caller)))

    def session_frozen(self, session: str) -> bool:
        return self.mode != Mode.OFF and self.store.blocked("session", session)

    def credential_suspect(self, name: str) -> bool:
        return self.store.state_of("credential", name) == State.QUARANTINED

    def screen(self, text: str, source: str, *, session: str = "",
               verdict: Callable[[str, str], Any] | None = None, where: str = "message",
               ) -> Screened:
        """Text about to enter a model's context. A decoy value, content already held, or
        content ``verdict`` (a rail's ``input``) denies is replaced by a placeholder."""
        if self.mode == Mode.OFF:
            return Screened(text)
        self.handle_all(self.honey.scan(text, where, session=session, caller=source))
        held = self.store.find_fingerprint(fingerprint(text))
        if held is not None:
            return Screened(placeholder(held.id), held.id)
        if verdict is None:
            return Screened(text)
        answer = verdict(text, source)
        denied = bool(getattr(answer, "denied", False))
        rail, reason = str(getattr(answer, "rail", "")), str(getattr(answer, "reason", ""))
        if not (denied or (getattr(answer, "tainted", False) and reads_like_instruction(reason))):
            return Screened(text)
        who = session or "none"
        self.handle_all(self.rails.denied_text(who, rail, reason, source, text) if denied
                        else self.rails.tainted_text(who, rail, reason, text))
        again = self.store.find_fingerprint(fingerprint(text))
        if again is not None:
            return Screened(placeholder(again.id), again.id)
        return Screened(placeholder("unrecorded"), "unrecorded") if denied else Screened(text)

    @contextmanager
    def running(self, session: str) -> Iterator[None]:
        """Mark ``session`` as having a tool call in flight, so a hit on the decoy listener
        meanwhile can be attributed to it."""
        with self._flight_lock:
            self._flights[session] = self._flights.get(session, 0) + 1
        try:
            yield
        finally:
            with self._flight_lock:
                self._flights[session] -= 1
                if not self._flights[session]:
                    del self._flights[session]

    def in_flight(self) -> list[str]:
        """The sessions with a tool call running now."""
        with self._flight_lock:
            return sorted(self._flights)

    def decoy_hit(self, path: str) -> list[str]:
        """A request reached the decoy listener: every session with a tool call running is
        frozen (its call is the only way a loopback request can be attributed), and with none
        running the hit is recorded against nobody."""
        if self.mode == Mode.OFF:
            return []
        return self.handle_all(self.honey.endpoint_hit(path, self.in_flight()))

    def note(self, kind: str, key: str, signal: str, weight: float | None = None) -> list[str]:
        """Count one weak ``signal`` (a guard denial, a sandbox refusal, a refused lease)
        against the session or caller ``(kind, key)``. The score escalates it to watch and then
        to quarantine; a person's release starts the count again. Returns what the policy did."""
        if self.mode == Mode.OFF:
            return []
        if self.score.held(kind, key) and self.store.state_of(kind, key) == State.RELEASED:
            self.score.reset(kind, key)
        return self.handle_all(self.score.add(kind, key, signal, weight))

    def screen_output(self, text: str, *, session: str = "", source: str = "model") -> Screened:
        """A model's reply about to be returned or stored. A decoy value, content already
        held, or a secret-shaped string in it is suspect: in ``guarded`` mode the reply is
        returned with a visible label, in ``enforce`` it is held and replaced by a placeholder."""
        if self.mode == Mode.OFF or not text:
            return Screened(text)
        reasons = []
        seen = self.honey.scan(text, "model_output", session=session, caller=source)
        if seen:
            self.handle_all(seen)
            reasons.append("carries a decoy value")
        held = self.store.find_fingerprint(fingerprint(text)) or self.store.find_overlap(text)
        if held is not None:
            reasons.append(f"repeats held content {held.id}")
        if redact(text) != text:
            reasons.append("carries a secret-shaped string")
        if not reasons:
            return Screened(text)
        why = "; ".join(reasons)
        self.bus.emit(Event("output.suspect", Severity.WARNING, "screen",
                            f"session:{session}" if session else f"caller:{source}",
                            {"why": why, "bytes": len(text)}, self.clock()))
        if self.mode == Mode.ENFORCE and not self.dry_run:
            digest = fingerprint(text)[:12]
            record = self.store.quarantine(("message", f"{session or source}:{digest}"),
                                           f"model output {why}", {"bytes": len(text)},
                                           Holding(text=text))
            ident = record.id if record else "unrecorded"
            return Screened(placeholder(ident), ident)
        return Screened(f"[sentinel: this reply {why}]\n{text}", labelled=True)

    def register_derived(self, session: str, memory_key: str) -> None:
        """Note that the memory ``memory_key`` (a summary, note, KV slot) was built from
        ``session``, so freezing the session quarantines it too."""
        self._derived.setdefault(session, []).append(memory_key)

    def _taint_derived(self, record: Record) -> None:
        for key in self._derived.pop(record.key, []):
            self.store.quarantine(("memory", key), f"derived from frozen session {record.key}",
                                  {"session": record.key})

    def memory_trusted(self, key: str) -> bool:
        """Whether a stored summary, note or cache may be loaded."""
        return self.mode == Mode.OFF or not self.store.blocked("memory", key)

    def screen_memory(self, key: str, text: str, *, session: str = "") -> Screened:
        """A summary or note about to be stored or loaded. One that repeats a held
        sentence is held in its own right and replaced by a placeholder, so it is rebuilt
        from clean sources instead of reused."""
        if self.mode == Mode.OFF:
            return Screened(text)
        if session:
            self.register_derived(session, key)
        if not self.memory_trusted(key):
            held = self.store.find("memory", key)
            return Screened(placeholder(held.id if held else "tampered"), held.id if held else "")
        seen = self.honey.scan(text, "model_output", session=session, caller="memory")
        self.handle_all(seen)
        source = self.store.find_overlap(text) or self.store.find_fingerprint(fingerprint(text))
        if source is None and not seen:
            return Screened(text)
        what = f"repeats held content {source.id}" if source else "carries a decoy value"
        evidence = {"copies": source.id} if source else {"decoy": seen[0].event.evidence["decoy"]}
        self.bus.emit(Event("memory.poisoned", Severity.WARNING, "memory", f"memory:{key}",
                            evidence, self.clock()))
        if self.mode == Mode.OBSERVE or self.dry_run:
            return Screened(text)
        held = self.store.quarantine(("memory", key), what, evidence, Holding(text=text))
        ident = held.id if held else "unrecorded"
        return Screened(placeholder(ident), ident)

    def scrub_env(self, env: Mapping[str, str]) -> dict[str, str]:
        """``env`` without any variable named as a suspect credential."""
        return {k: v for k, v in env.items() if not self.credential_suspect(k)}

    def mcp_allowed(self, server: str) -> bool:
        return self.mode == Mode.OFF or not self.store.blocked("mcp_server", server)

    def chip(self) -> dict[str, Any]:
        """A status mark in the UI's verdict vocabulary (green, yellow, red, none)."""
        info = self.status()
        held = info["subjects"][State.QUARANTINED.value]
        watched = info["subjects"][State.WATCH.value]
        if self.mode == Mode.OFF:
            verdict, label = "none", "sentinel off"
        elif info["tampered"] or info["log_ok"] is False:
            verdict, label = "red", "sentinel: state or log tampered"
        elif held or watched:
            verdict, label = "yellow", f"sentinel: {held} held, {watched} watched"
        else:
            verdict, label = "green", f"sentinel: {self.mode.value}, nothing held"
        return {"verdict": verdict, "label": label, "held": held, "watched": watched}

    def screen_call(self, tool: str, arguments: Mapping[str, Any] | None, *,
                    session: str = "", caller: str = "") -> str:
        """Why a tool call must not run (a reason), or an empty string."""
        if self.mode == Mode.OFF:
            return ""
        flat = repr(arguments)
        self.handle_all(self.honey.scan(flat, "tool_args", session=session, caller=caller))
        self.handle_all([self.honey.tool_called(tool, session=session, caller=caller)])
        self.handle_all([self.tools.note(session or caller or "none", tool)])
        if session and self.session_frozen(session):
            return f"session {session} is frozen by sentinel"
        if not self.tool_allowed(tool, caller):
            return f"tool {tool} is disabled by sentinel"
        return ""

    # -- integrity -------------------------------------------------------------------
    def verify_before_load(self, path: Path | str, *, cached: bool = False) -> bool:
        """Hash a pinned file and compare it with its pin. False when it changed or is
        quarantined; an unpinned file passes. With ``cached`` a file whose size, mtime and
        inode are those of its last full verification against this pin is not hashed again."""
        pin = self.manifest.pins().get(str(Path(path).expanduser()))
        if pin is None or self.mode == Mode.OFF:
            return True
        if self.store.blocked(pin.kind, pin.path):
            return False
        if cached and self._verified_before(pin):
            return True
        found = check_file(pin, deep=True)
        if found is None:
            self._remember(pin)
            return True
        if found.event.kind != MISSING:
            self.handle(found)
        return self.mode == Mode.OBSERVE and found.event.kind != MISSING

    def _stat_key(self, pin: Pin) -> tuple[str, int, int, int] | None:
        path = Path(pin.path)
        try:
            info = path.stat()
            return str(path.resolve()), info.st_size, info.st_mtime_ns, info.st_ino
        except OSError:
            return None

    def _verified_before(self, pin: Pin) -> bool:
        key = self._stat_key(pin)
        entry = self._verified.load().payload.get("files", {}).get(key[0]) if key else None
        return bool(entry) and (entry["size"], entry["mtime_ns"], entry["inode"],
                                entry["sha256"]) == (*key[1:], pin.sha256) \
            and key[1] == pin.bytes

    def _remember(self, pin: Pin) -> None:
        key = self._stat_key(pin)
        if key is None:
            return
        files = dict(self._verified.load().payload.get("files", {}))
        files[key[0]] = {"size": key[1], "mtime_ns": key[2], "inode": key[3],
                         "sha256": pin.sha256}
        self._verified.save({"files": files})

    def scan(self, *, deep: bool = False) -> list[Finding]:
        """Check every pin (hashing every file when ``deep``) and the decoys. A pinned file
        that is gone is logged once and its pin dropped after `GONE_SCANS` scans in a row;
        nothing is quarantined for it."""
        out: list[Finding] = []
        for pin in self.manifest.pins().values():
            if self.store.state_of(pin.kind, pin.path) == State.QUARANTINED:
                continue
            found = check_file(pin, deep=deep)
            if found is not None:
                out.append(found)
            elif deep:
                self.manifest.refresh_stat(pin)
        self._heal([f for f in out if f.event.kind == MISSING])
        out.extend(self.honey.touched())
        self.handle_all([f for f in out if f.event.kind != MISSING])
        self.last_scan = self.clock()
        self.heads_up.poll()
        return out

    def _heal(self, missing: list[Finding]) -> None:
        """Count the scans a pinned file has been gone, drop the pin at `GONE_SCANS`, settle
        the record an older version quarantined for it, and clear a watched server whose
        process has exited."""
        gone = {f.path for f in missing}
        for path in [p for p in self._gone if p not in gone]:
            del self._gone[path]
        for found in missing:
            seen = self._gone[found.path] = self._gone.get(found.path, 0) + 1
            if seen == 1:
                self.bus.emit(found.event)
            if seen >= GONE_SCANS:
                self.manifest.unpin(found.path)
                self.bus.emit(Event("integrity.pin_dropped", Severity.NOTICE, "core",
                                    found.event.subject, {"scans": seen}, self.clock()))
                del self._gone[found.path]
        for record in self.store.records(state=State.QUARANTINED):
            if self.store.settle_missing(record.id):
                self.manifest.unpin(record.key)
        for record in self.store.records(kind="server", state=State.WATCH):
            pid = record.evidence.get("pid")
            if isinstance(pid, int) and not pid_alive(pid):
                self.store.clear("server", record.key, "the server process exited")

    def start(self, interval_s: float = 300.0, deep_every: int = 12) -> threading.Thread:
        """Scan on a timer in a background thread, hashing every file every ``deep_every``
        rounds. `stop` ends it."""
        self._stop.clear()

        def loop() -> None:
            rounds = 0
            while not self._stop.wait(interval_s):
                rounds += 1
                self.scan(deep=rounds % deep_every == 0)

        thread = threading.Thread(target=loop, name="sentinel-scan", daemon=True)
        thread.start()
        return thread

    def stop(self) -> None:
        self._stop.set()

    # -- canaries --------------------------------------------------------------------
    def baseline(self, model: str, ask: Callable[[str], str], runs: int = 8) -> Results:
        """Record how ``model`` answers the canary probes now."""
        results = canaries.run(ask, runs=runs)
        self.baselines.record(model, results)
        return results

    def canary(self, model: str, ask: Callable[[str], str], runs: int = 5) -> Finding | None:
        """Re-run the probes and compare with the baseline; handles and returns a drift
        finding. Records the baseline instead when there is none."""
        base = self.baselines.get(model)
        if base is None:
            self.baseline(model, ask, runs)
            return None
        found = canaries.drift_finding(model, base, canaries.run(ask, runs=runs))
        if found is not None:
            self.handle(found)
        return found

    # -- reporting -------------------------------------------------------------------
    def status(self) -> dict[str, Any]:
        records = self.store.records()
        by_state = {s.value: sum(r.state == s for r in records) for s in State}
        chain = self.bus.log.verify() if self.bus.log else None
        return {"mode": self.mode.value, "dry_run": self.dry_run,
                "tampered": self.store.tampered, "subjects": by_state,
                "pins": len(self.manifest.pins()),
                "decoys": len(self.honey.decoys()),
                "log_ok": None if chain is None else chain.ok,
                "log_records": 0 if chain is None else chain.records,
                "last_scan": self.last_scan, "off_because": self.off_because,
                "scanner": scanner_state(self.root, self.clock())}


def _summary(evidence: Mapping[str, Any]) -> str:
    return ", ".join(f"{k}={v}" for k, v in list(evidence.items())[:4])
