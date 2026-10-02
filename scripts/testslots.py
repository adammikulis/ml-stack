"""A machine-wide CPU budget for test runs, shared by every checkout and every agent.

Problem: many agents each start a full test suite with several xdist workers at once, the machine saturates
(load average >> cores) and every run slows down, so everything waits on everything. This module makes test runs
queue instead: a run asks for `want` workers, waits (strictly first come, first served) until the shared budget has
room for at least `minimum`, runs with what it was granted, and gives the workers back when it exits (also when it is
killed: the lock dies with the process).

Protocol (stdlib only, so other repos can carry an identical copy and share the same budget):
  * directory  $DEV_TEST_SLOTS_DIR or ~/.cache/dev-test-slots
  * every participant creates `<time_ns>-<pid>.slot` (JSON: label, pid, want, minimum, granted, since) and holds an
    exclusive flock on it for as long as it lives, waiting or running. A slot whose flock can be taken is stale
    (its owner died) and is removed by whoever notices.
  * `mutex.lock` serialises decisions. Only the OLDEST waiting slot may be granted (no starvation); it gets
    min(want, budget - granted_total) workers once that is >= minimum.
  * budget = $DEV_TEST_BUDGET, else LOAD-AWARE: 3/4 of the cores normally, up to all the cores while the 1-minute load
    average is under 3/4 of the cores (the machine is idle: use it), down to half of the base while it is over twice the
    cores (it is already saturated: back off). Thresholds: $DEV_TEST_LOAD_LOW / $DEV_TEST_LOAD_HIGH (multiples of cores).
    The cap is re-evaluated every time a waiting run is considered; running leases are never shrunk.
  * HEAVY LANES: a few tests spawn multi-threaded tools (one `kicad-cli` DRC can use 8 cores). Those tests wrap
    themselves in `heavy_lane()`: at most $DEV_TEST_HEAVY_LANES (default cores/4, min 2) of them run at the same time
    machine-wide, across all runs. A lane is a leaf resource (a test holding one waits for nothing else), so it cannot
    deadlock; after $DEV_TEST_LANE_WAIT_S (default 600) a test runs anyway and says so.
  * $DEV_TEST_WAIT_S bounds the wait (default 3600 s); $DEV_TEST_SLOTS=off disables the queue (explicit, logged).

CLI:  python scripts/testslots.py status
      python scripts/testslots.py run --want 4 --min 2 --label NAME -- COMMAND ARGS...
            (queues, then runs COMMAND with DEV_TEST_WORKERS=<granted> in its environment; use it for any test
             command, e.g.  ... -- sh -c 'pytest -n $DEV_TEST_WORKERS tests')
"""
from __future__ import annotations

import contextlib
import fcntl
import json
import os
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path


def slots_dir() -> Path:
    d = Path(os.environ.get("DEV_TEST_SLOTS_DIR") or Path.home() / ".cache" / "dev-test-slots")
    d.mkdir(parents=True, exist_ok=True)
    return d


def _cores() -> int:
    return os.cpu_count() or 4


def base_budget(cores: int | None = None) -> int:
    return max(2, (cores or _cores()) * 3 // 4)


def cap_for(load1: float | None, cores: int | None = None, low: float | None = None, high: float | None = None) -> int:
    """The worker cap for a given 1-minute load average (pure, so it can be tested)."""
    cores = cores or _cores()
    low = low if low is not None else float(os.environ.get("DEV_TEST_LOAD_LOW", "0.75"))
    high = high if high is not None else float(os.environ.get("DEV_TEST_LOAD_HIGH", "2.0"))
    base = base_budget(cores)
    if load1 is None:
        return base
    if load1 < low * cores:
        return max(base, cores)
    if load1 > high * cores:
        return max(2, base // 2)
    return base


def budget() -> int:
    raw = os.environ.get("DEV_TEST_BUDGET")
    if raw and raw.isdigit() and int(raw) > 0:
        return int(raw)
    try:
        load1 = os.getloadavg()[0]
    except (OSError, AttributeError):
        load1 = None
    return cap_for(load1)


@dataclass
class Slot:
    path: Path
    data: dict

    @property
    def granted(self) -> int:
        return int(self.data.get("granted", 0))


@contextlib.contextmanager
def _mutex(d: Path) -> Iterator[None]:
    with (d / "mutex.lock").open("a+") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def _alive(path: Path) -> bool:
    """True while another process holds the slot's flock."""
    try:
        fd = os.open(path, os.O_RDWR)
    except OSError:
        return False
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        return True
    else:
        fcntl.flock(fd, fcntl.LOCK_UN)
        return False
    finally:
        os.close(fd)


def _read(d: Path, mine: Path | None = None) -> list[Slot]:
    out: list[Slot] = []
    for p in sorted(d.glob("*.slot")):
        if p != mine and not _alive(p):
            with contextlib.suppress(OSError):
                p.unlink()
            continue
        try:
            out.append(Slot(p, json.loads(p.read_text() or "{}")))
        except (OSError, ValueError):
            continue
    return out


def status() -> dict:
    d = slots_dir()
    with _mutex(d):
        slots = _read(d)
    running = [s for s in slots if s.granted > 0]
    return {"budget": budget(), "base": base_budget(), "in_use": sum(s.granted for s in running),
            "running": [s.data for s in running], "waiting": [s.data for s in slots if s.granted == 0]}


@dataclass
class Lease:
    workers: int
    waited: float


@contextlib.contextmanager
def lease(want: int, minimum: int | None = None, label: str = "tests", say=lambda m: print(m, file=sys.stderr, flush=True)
          ) -> Iterator[Lease]:
    """Wait for `minimum`..`want` workers of the shared budget; yield what was granted. Released on exit or death."""
    want = max(1, int(want))
    minimum = max(1, min(int(minimum if minimum is not None else min(2, want)), want))
    if os.environ.get("DEV_TEST_SLOTS", "").lower() == "off":
        say("testslots: queue disabled by DEV_TEST_SLOTS=off")
        yield Lease(want, 0.0)
        return
    d = slots_dir()
    path = d / f"{time.time_ns()}-{os.getpid()}.slot"
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    fcntl.flock(fd, fcntl.LOCK_EX)                       # held until we exit: this is the liveness signal
    me = {"label": label, "pid": os.getpid(), "want": want, "minimum": minimum, "granted": 0, "since": time.time()}
    path.write_text(json.dumps(me))
    t0 = time.time()
    deadline = t0 + float(os.environ.get("DEV_TEST_WAIT_S", "3600"))
    last_note = 0.0
    granted = 0
    try:
        while True:
            cap = max(budget(), minimum)                 # re-read every poll: load falls while a run waits
            with _mutex(d):
                slots = _read(d, mine=path)
                waiting = [s for s in slots if s.granted == 0]
                used = sum(s.granted for s in slots)
                head = waiting[0].path if waiting else None
                if head == path and cap - used >= minimum:
                    granted = min(want, cap - used)
                    me["granted"] = granted
                    path.write_text(json.dumps(me))
                    break
                ahead = [s.data.get("label") for s in waiting if s.path != path and s.path < path]
            if time.time() > deadline:
                raise TimeoutError(f"testslots: waited {time.time() - t0:.0f}s for {minimum} of {cap} workers "
                                   f"({used} in use); see `python scripts/testslots.py status`")
            if time.time() - last_note > 20:
                last_note = time.time()
                say(f"testslots: waiting for {minimum}-{want} workers ({used}/{cap} in use, "
                    f"{len(ahead)} run(s) ahead: {', '.join(map(str, ahead)) or 'none'})")
            time.sleep(0.4)
        yield Lease(granted, time.time() - t0)
    finally:
        with contextlib.suppress(OSError):
            path.unlink()
        os.close(fd)


def _lane_count() -> int:
    raw = os.environ.get("DEV_TEST_HEAVY_LANES")
    if raw and raw.isdigit() and int(raw) > 0:
        return int(raw)
    return max(2, _cores() // 4)


@contextlib.contextmanager
def heavy_lane(label: str = "heavy test", say=lambda m: print(m, file=sys.stderr, flush=True)) -> Iterator[None]:
    """Hold one of the machine-wide HEAVY lanes while a test that spawns a multi-threaded tool runs."""
    if os.environ.get("DEV_TEST_SLOTS", "").lower() == "off":
        yield
        return
    d = slots_dir()
    n = _lane_count()
    deadline = time.time() + float(os.environ.get("DEV_TEST_LANE_WAIT_S", "600"))
    fd = -1
    noted = False
    while fd < 0:
        for i in range(n):
            f = os.open(d / f"heavy-{i}.lane", os.O_RDWR | os.O_CREAT, 0o600)
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                os.close(f)
                continue
            fd = f
            break
        if fd >= 0:
            break
        if time.time() > deadline:
            say(f"testslots: no heavy lane free after the wait limit; running {label} anyway")
            break
        if not noted:
            noted = True
            say(f"testslots: {label} waits for one of {n} heavy lanes")
        time.sleep(0.2)
    try:
        yield
    finally:
        if fd >= 0:
            os.close(fd)                                  # closing releases the flock


def _run_command(argv: list[str]) -> int:
    import argparse
    import subprocess
    ap = argparse.ArgumentParser(prog="testslots run")
    ap.add_argument("--want", type=int, default=4)
    ap.add_argument("--min", dest="minimum", type=int, default=None)
    ap.add_argument("--label", default="command")
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args(argv)
    cmd = a.cmd[1:] if a.cmd[:1] == ["--"] else a.cmd
    if not cmd:
        ap.error("give a command after --")
    with lease(a.want, a.minimum, label=a.label) as got:
        print(f"testslots: running with {got.workers} worker(s) (waited {got.waited:.0f}s)", file=sys.stderr, flush=True)
        return subprocess.run(cmd, env={**os.environ, "DEV_TEST_WORKERS": str(got.workers)}).returncode


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["run"]:
        return _run_command(argv[1:])
    if argv[:1] != ["status"]:
        print(__doc__)
        return 2
    st = status()
    try:
        load = f"{os.getloadavg()[0]:.1f}"
    except (OSError, AttributeError):
        load = "?"
    print(f"budget {st['budget']} workers (base {st['base']}; load {load} on {_cores()} cores), {st['in_use']} in use, "
          f"{_lane_count()} heavy lane(s)")
    for s in st["running"]:
        print(f"  running  {s['granted']:>2}  pid {s['pid']:<7} {s['label']}")
    for s in st["waiting"]:
        print(f"  waiting  {s['minimum']}-{s['want']}  pid {s['pid']:<7} {s['label']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
