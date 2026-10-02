"""The sentinel: detectors, the policy and the quarantine store joined to one event bus."""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ml_stack.sentinel import canary as canaries
from ml_stack.sentinel.canary import Baseline, Results
from ml_stack.sentinel.events import Bus, Event, EventLog, Severity
from ml_stack.sentinel.findings import Finding
from ml_stack.sentinel.honey import Honey
from ml_stack.sentinel.human import HumanGrant, protect
from ml_stack.sentinel.integrity import Manifest, check_file
from ml_stack.sentinel.policy import Mode, decide
from ml_stack.sentinel.rails import RailWatch
from ml_stack.sentinel.rates import Abuse, PeerWatch, ToolMix
from ml_stack.sentinel.sealed import SealedFile
from ml_stack.sentinel.store import Holding, State, Store, fingerprint, placeholder, sentinel_dir

__all__ = ["ENV", "Screened", "Sentinel"]

ENV = "ML_STACK_SENTINEL"


@dataclass(frozen=True, slots=True)
class Screened:
    """What a model may see in place of some text: ``text`` is the text or a placeholder,
    ``withheld`` the quarantine id when it was held."""

    text: str
    withheld: str = ""


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
        self.mode = mode or self._configured_mode()
        self.dry_run = dry_run
        self.manifest = Manifest(self.root / "manifest.json", clock)
        self.baselines = Baseline(self.root / "canaries.json")
        self.honey = Honey(state=self.root / "honey.json")
        self.peers, self.tools, self.rails = PeerWatch(), ToolMix(), RailWatch()
        self.abuse = Abuse()
        self.last_scan = 0.0
        self._stop = threading.Event()
        if self.mode == Mode.OFF:
            self.bus.emit(Event("sentinel.off", Severity.WARNING, "core", "", {}, clock()))

    def _configured_mode(self) -> Mode:
        named = os.environ.get(ENV, "").strip().lower()
        if named in Mode:
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
        if verdict is not None:
            answer = verdict(text, source)
            if getattr(answer, "denied", False):
                found = self.rails.denied_text(session or "none", str(getattr(answer, "rail", "")),
                                               str(getattr(answer, "reason", "")), source, text)
                self.handle_all(found)
                again = self.store.find_fingerprint(fingerprint(text))
                if again is not None:
                    return Screened(placeholder(again.id), again.id)
                return Screened(placeholder("unrecorded"), "unrecorded")
        return Screened(text)

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
    def verify_before_load(self, path: Path | str) -> bool:
        """Hash a pinned file and compare it with its pin. False when it changed or is
        quarantined; an unpinned file passes."""
        pin = self.manifest.pins().get(str(Path(path).expanduser()))
        if pin is None or self.mode == Mode.OFF:
            return True
        if self.store.blocked(pin.kind, pin.path):
            return False
        found = check_file(pin, deep=True)
        if found is None:
            return True
        self.handle(found)
        return self.mode == Mode.OBSERVE

    def scan(self, *, deep: bool = False) -> list[Finding]:
        """Check every pin (hashing every file when ``deep``) and the decoys."""
        out: list[Finding] = []
        for pin in self.manifest.pins().values():
            if self.store.state_of(pin.kind, pin.path) == State.QUARANTINED:
                continue
            found = check_file(pin, deep=deep)
            if found is not None:
                out.append(found)
            elif deep:
                self.manifest.refresh_stat(pin)
        out.extend(self.honey.touched())
        self.handle_all(out)
        self.last_scan = self.clock()
        return out

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
                "last_scan": self.last_scan}


def _summary(evidence: Mapping[str, Any]) -> str:
    return ", ".join(f"{k}={v}" for k, v in list(evidence.items())[:4])
