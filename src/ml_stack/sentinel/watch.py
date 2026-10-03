"""What a long-running ml-stack process arms by default: the periodic scan with its heartbeat,
the decoy files, and the logged opt-outs."""

from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ml_stack.sentinel.events import Event, Severity
from ml_stack.sentinel.policy import Mode
from ml_stack.sentinel.sealed import SealedFile

if TYPE_CHECKING:
    from ml_stack.sentinel.core import Sentinel

__all__ = ["BECAUSE", "DEFAULT_INTERVAL_S", "ENV_SCAN", "Cadence", "Scanner", "arm_scan",
           "cadence", "ensure_honey", "opt_out", "scanner_state"]

logger = logging.getLogger("ml_stack.sentinel")

ENV_SCAN = "ML_STACK_SENTINEL_SCAN"
BECAUSE = "ML_STACK_SENTINEL_SCAN_BECAUSE"
DEFAULT_INTERVAL_S = 300.0
DEEP_EVERY = 12
HEARTBEAT = "scanner.json"


@dataclass(frozen=True, slots=True)
class Cadence:
    """Seconds between scans; 0 when switched off, with the reason given for it."""

    interval_s: float
    off_because: str = ""
    refused: str = ""


def cadence(env: Mapping[str, str] | None = None) -> Cadence:
    """The scan cadence ``ML_STACK_SENTINEL_SCAN`` names: seconds, or ``off`` together with
    ``ML_STACK_SENTINEL_SCAN_BECAUSE``. ``off`` without a reason, or anything unreadable, keeps
    the default and says so in ``refused``."""
    env = os.environ if env is None else env
    named = env.get(ENV_SCAN, "").strip().lower()
    if not named:
        return Cadence(DEFAULT_INTERVAL_S)
    if named == "off":
        because = env.get(BECAUSE, "").strip()
        if because:
            return Cadence(0.0, because)
        return Cadence(DEFAULT_INTERVAL_S, refused=f"{ENV_SCAN}=off needs {BECAUSE}")
    try:
        seconds = float(named)
    except ValueError:
        return Cadence(DEFAULT_INTERVAL_S, refused=f"{ENV_SCAN}={named!r} is not seconds or off")
    if seconds <= 0:
        return Cadence(DEFAULT_INTERVAL_S, refused=f"{ENV_SCAN} must be above zero")
    return Cadence(seconds)


def opt_out(sentinel: Sentinel, what: str, because: str) -> None:
    """Record that ``what`` runs without sentinel, for ``because``. A blank reason raises."""
    if not because.strip():
        raise ValueError(f"running {what} without sentinel needs a because=")
    message = f"sentinel: {what} runs without it: {because.strip()}"
    logger.warning(message)
    sentinel.bus.emit(Event("sentinel.opt_out", Severity.WARNING, "watch", f"opt_out:{what}",
                            {"because": because.strip()}, sentinel.clock()))


_PLANTED: set[str] = set()


def ensure_honey(sentinel: Sentinel) -> int:
    """Plant the decoy files under the state root unless this install has them; returns how
    many decoys the install holds."""
    where = str(sentinel.root)
    if where in _PLANTED and all(Path(d.path).exists() for d in sentinel.honey.decoys()):
        return len(sentinel.honey.decoys())
    have = sentinel.honey.decoys()
    if not have or any(not Path(d.path).exists() for d in have):
        have = sentinel.honey.plant() or sentinel.honey.decoys()
    _PLANTED.add(where)
    return len(have)


class Scanner:
    """The periodic scan of one sentinel in a daemon thread. It writes a heartbeat beside
    the sentinel's state so another process can tell the loop is running."""

    def __init__(self, sentinel: Sentinel, interval_s: float, *,
                 deep_every: int = DEEP_EVERY) -> None:
        self.sentinel, self.interval_s, self.deep_every = sentinel, interval_s, deep_every
        self._file = SealedFile(sentinel.root / HEARTBEAT)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._started = 0.0

    def start(self) -> Scanner:
        """Scan now, deeply, then every ``interval_s`` seconds until `stop`."""
        if self._thread is not None:
            return self
        self._started = self.sentinel.clock()
        self._beat(True)
        self._thread = threading.Thread(target=self._loop, name="sentinel-scan", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        """End the loop and mark the heartbeat as stopped."""
        self._stop.set()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout=10)
            self._beat(False)

    def _loop(self) -> None:
        rounds = 0
        self._once(deep=True)
        while not self._stop.wait(self.interval_s):
            rounds += 1
            self._once(deep=rounds % self.deep_every == 0)

    def _once(self, *, deep: bool) -> None:
        try:
            self.sentinel.scan(deep=deep)
        except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
            logger.warning("sentinel scan failed: %s", exc)
            self.sentinel.bus.emit(Event("sentinel.scan_failed", Severity.WARNING, "watch", "",
                                         {"error": type(exc).__name__}, self.sentinel.clock()))
        self._beat(True)

    def _beat(self, running: bool) -> None:
        self._file.save({"pid": os.getpid(), "interval_s": self.interval_s,
                         "deep_every": self.deep_every, "started": self._started,
                         "beat": self.sentinel.clock(), "running": running})


def arm_scan(sentinel: Sentinel) -> Scanner | None:
    """Start the periodic scan at the configured cadence, for a long-running process; the
    caller stops it. None when the mode is off or the scan was switched off with a reason."""
    if sentinel.mode == Mode.OFF:
        return None
    chosen = cadence()
    if chosen.refused:
        logger.warning("sentinel: %s; scanning every %.0fs", chosen.refused, chosen.interval_s)
        sentinel.bus.emit(Event("sentinel.scan_off_refused", Severity.WARNING, "watch", "",
                                {"why": chosen.refused}, sentinel.clock()))
    if chosen.interval_s <= 0:
        opt_out(sentinel, "the periodic scan", chosen.off_because)
        return None
    return Scanner(sentinel, chosen.interval_s).start()


def process_alive(pid: int) -> bool:
    """Whether a process with this id exists (signal 0 sends nothing). A process of another user counts as alive."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def scanner_state(root: Path, now: float | None = None) -> dict[str, Any]:
    """Whether a scan loop is armed on the sentinel at ``root``: a process wrote a heartbeat
    within three intervals and has not stopped it."""
    loaded = SealedFile(Path(root) / HEARTBEAT).load()
    beat = loaded.payload
    if not beat:
        return {"armed": False, "why": "no scan loop has run here"
                if loaded.status == "fresh" else f"heartbeat {loaded.status}"}
    now = time.time() if now is None else now
    pid, interval = int(beat.get("pid") or 0), float(beat.get("interval_s") or 0)
    out = {"armed": False, "pid": pid, "interval_s": interval, "last_beat": beat.get("beat")}
    if not beat.get("running"):
        return {**out, "why": "the scan loop was stopped"}
    if now - float(beat.get("beat") or 0) > 3 * interval + 5:
        return {**out, "why": "the scan loop has not reported within three intervals"}
    if not process_alive(pid):
        return {**out, "why": f"process {pid} that wrote the heartbeat is gone"}
    return {**out, "armed": True, "why": ""}

