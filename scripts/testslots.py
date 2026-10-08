"""Machine-wide, file-lock-backed CPU permits and subprocess test admission."""
from __future__ import annotations

import contextlib
import contextvars
import json
import os
import secrets
import sys
import tempfile
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from ml_stack.lock import release, take

HEAVY_MODULES = frozenset({
    "test_serve_real_llama", "test_sentinel_real_model", "test_sentinel_wiring_serve",
    "test_serve_broker",
    "test_serve_three_callers", "test_spec_serve", "test_fleet_daemon", "test_fleet_bind",
    "test_fleet_join", "test_fleet_bench", "test_graph_bench", "test_graph_bench_animate",
    "test_graph_store_scale",
})

EXPORT_LEASE = contextvars.ContextVar("export_test_lease", default=True)
CHECK_CANCELLED = contextvars.ContextVar("check_test_cancelled", default=lambda: None)

def slots_dir() -> Path:
    d = Path(os.environ.get("DEV_TEST_SLOTS_DIR") or Path.home() / ".cache" / "dev-test-slots")
    d.mkdir(parents=True, exist_ok=True)
    return d


def _cores() -> int:
    return os.cpu_count() or 4


def base_budget(cores: int | None = None) -> int:
    return max(1, (cores or _cores()) - max(0, int(os.environ.get("DEV_TEST_RESERVED_CORES", "1"))))


def cap_for(load1: float | None, cores: int | None = None, low: float | None = None, high: float | None = None) -> int:
    """The worker cap for a given 1-minute load average (pure, so it can be tested)."""
    cores = cores or _cores()
    high = high if high is not None else float(os.environ.get("DEV_TEST_LOAD_HIGH", "2.0"))
    base = base_budget(cores)
    if load1 is None:
        return base
    if load1 > high * cores:
        return max(1, base // 2)
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
        while not take(fh):
            time.sleep(0.01)
        try:
            yield
        finally:
            release(fh)


def _alive(path: Path) -> bool:
    """True while another process holds the slot's file lock."""
    try:
        fd = os.open(path, os.O_RDWR)
    except OSError:
        return False
    try:
        if not take(fd):
            return True
        release(fd)
        return False
    finally:
        os.close(fd)


def _owns_inode(fd: int, path: Path) -> bool:
    try:
        held, published = os.fstat(fd), path.stat()
        return (held.st_dev, held.st_ino) == (published.st_dev, published.st_ino)
    except OSError:
        return False


def _valid_record(data: object) -> bool:
    if not isinstance(data, dict):
        return False
    numbers = ("pid", "want", "minimum", "granted")
    if any(type(data.get(key)) is not int for key in numbers):
        return False
    return (data["pid"] > 0 and data["want"] >= 0 and data["minimum"] > 0
            and data["granted"] >= 0 and (not data["want"] or max(data["minimum"], data["granted"]) <= data["want"])
            and isinstance(data.get("label"), str) and isinstance(data.get("token"), str)
            and type(data.get("backfills", 0)) is int and 0 <= data.get("backfills", 0) <= 2)


def _delete_shared_descriptor(path: Path) -> int:
    """Open a Windows record handle allowing replacement while its lock is held."""
    import ctypes
    import msvcrt
    from ctypes import wintypes

    create = ctypes.WinDLL("kernel32", use_last_error=True).CreateFileW
    create.argtypes = (wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p,
                       wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE)
    create.restype = wintypes.HANDLE
    handle = create(str(path), 0xC0000000, 7, None, 3, 0x80, None)
    if handle == wintypes.HANDLE(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    return msvcrt.open_osfhandle(handle, os.O_RDWR)


def _publish(path: Path, record: dict) -> int:
    """Publish a complete record whose inode already holds its liveness lock."""
    fd, name = tempfile.mkstemp(dir=path.parent, suffix=".pending")
    try:
        if sys.platform == "win32":
            os.close(fd)
            fd = _delete_shared_descriptor(Path(name))
        if not take(fd):
            raise RuntimeError("testslots: unpublished lease already locked")
        os.lseek(fd, 0, os.SEEK_SET)
        with os.fdopen(fd, "w", closefd=False) as stream:
            json.dump(record, stream)
        Path(name).replace(path)
        return fd
    except BaseException:
        os.close(fd)
        raise
    finally:
        Path(name).unlink(missing_ok=True)


def _read(d: Path, mine: Path | None = None) -> list[Slot]:
    out: list[Slot] = []
    for p in sorted(d.glob("*.slot")):
        if p != mine and not _alive(p):
            with contextlib.suppress(OSError):
                p.unlink()
            continue
        try:
            data = json.loads(p.read_text())
            if _valid_record(data):
                out.append(Slot(p, data))
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


def _reject_nested() -> None:
    if (os.environ.get("DEV_TEST_REMOTE_LEASE")
            and os.environ.get("DEV_TEST_REMOTE_BROKER") == str(slots_dir().resolve())):
        import testslots_rpc
        with testslots_rpc.request("acquire", label="nested CPU preflight", phase="collection"):
            pass
    inherited = os.environ.get("DEV_TEST_LEASE")
    if inherited:
        try:
            descriptor = json.loads(inherited)
            inherited_path = Path(descriptor["path"])
            record = json.loads(inherited_path.read_text())
            if (inherited_path.parent.resolve() == slots_dir().resolve()
                    and record.get("token") == descriptor["token"] and _alive(inherited_path)):
                raise RuntimeError("testslots: nested CPU run inside a live test lease; run it after releasing the parent permit")
        except (OSError, ValueError, KeyError, TypeError):
            pass


@contextlib.contextmanager
def _lease_environment(path: Path, token: str) -> Iterator[None]:
    if not EXPORT_LEASE.get():
        yield
        return
    previous = os.environ.get("DEV_TEST_LEASE")
    os.environ["DEV_TEST_LEASE"] = json.dumps({"path": str(path.resolve()), "token": token})
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("DEV_TEST_LEASE", None)
        else:
            os.environ["DEV_TEST_LEASE"] = previous


def _grant(me: Slot, waiting: list[Slot], capacity: tuple[int, int, int]) -> int:
    path, want, minimum = me.path, me.data["want"], me.data["minimum"]
    cap, used, running = capacity
    if not waiting or cap - used < minimum:
        return 0
    first = waiting[0]
    if first.path != path:
        if cap - used >= first.data["minimum"] or first.data.get("backfills", 0) >= 2:
            return 0
        first.data["backfills"] = first.data.get("backfills", 0) + 1
        first.path.write_text(json.dumps(first.data))
    if not want:
        return min(cap - used, max(minimum, cap // max(1, len(waiting) + running)))
    return min(want, cap - used)


def _restore_inode(path: Path, record: dict, fd: int) -> int:
    if _owns_inode(fd, path):
        return fd
    replacement = _publish(path, record)
    os.close(fd)
    return replacement


def _own_records(directory: Path, path: Path, record: dict) -> list[Slot] | None:
    slots = _read(directory, mine=path)
    owned = next((slot for slot in slots if slot.path == path), None)
    if owned is None:
        return None
    if owned.data["token"] != record["token"] or owned.granted != record["granted"]:
        raise RuntimeError("testslots: lease record ownership changed")
    return slots


def _agent_from_env() -> dict[str, str]:
    """The workspace agent the runner exported for this process, as plain strings."""
    try:
        data = json.loads(os.environ.get("DEV_TEST_AGENT", ""))
    except ValueError:
        return {}
    return {k: str(data[k]) for k in ("id", "label", "parent", "job") if isinstance(data, dict) and data.get(k)}


@contextlib.contextmanager
def lease(want: int, minimum: int | None = None, label: str = "tests", say=lambda m: print(m, file=sys.stderr, flush=True)
          ) -> Iterator[Lease]:
    """Acquire a bounded FIFO CPU lease, released on context exit or process death."""
    _reject_nested()
    auto = int(want) <= 0
    want = 0 if auto else max(1, int(want))
    minimum = max(1, int(minimum if minimum is not None else 1)) if auto \
        else max(1, min(int(minimum if minimum is not None else 1), want))
    if minimum > (int(os.environ["DEV_TEST_BUDGET"]) if os.environ.get("DEV_TEST_BUDGET", "").isdigit() else base_budget()):
        raise ValueError("testslots: minimum exceeds the configured CPU budget")
    d = slots_dir()
    path = d / f"{time.time_ns()}-{os.getpid()}-{secrets.token_hex(8)}.slot"
    me = {"label": label, "pid": os.getpid(), "want": want, "minimum": minimum, "granted": 0, "since": time.time(), "version": 1, "token": secrets.token_hex(24)}
    if _agent_from_env():
        me["agent"] = _agent_from_env()
    t0 = time.monotonic()
    deadline = t0 + float(os.environ.get("DEV_TEST_WAIT_S", "3600"))
    with _mutex(d):
        fd = _publish(path, me)
    last_note = t0
    last_state = None
    try:
        while True:
            CHECK_CANCELLED.get()()
            cap = budget()
            with _mutex(d):
                fd = _restore_inode(path, me, fd)
                slots = _own_records(d, path, me)
                if slots is None:
                    replacement = _publish(path, me)
                    os.close(fd)
                    fd = replacement
                    continue
                waiting = [s for s in slots if s.granted == 0]
                used = sum(s.granted for s in slots)
                granted = _grant(Slot(path, me), waiting, (cap, used, sum(slot.granted > 0 for slot in slots)))
                if granted:
                    me["granted"] = granted
                    path.write_text(json.dumps(me))
                    break
                ahead = [s.data.get("label") for s in waiting if s.path != path and s.path < path]
            now = time.monotonic()
            if now > deadline:
                raise TimeoutError(f"testslots: waited {now - t0:.0f}s for {minimum} of {cap} workers "
                                   f"({used} in use); see `python scripts/testslots.py status`")
            state = (minimum, want, cap, used, tuple(ahead))
            if now - last_note >= 20 or (state != last_state and now - last_note >= 1):
                last_note, last_state = now, state
                say(f"testslots: waiting for {minimum}-{want or 'auto'} workers ({used}/{cap} in use, "
                    f"{len(ahead)} run(s) ahead: {', '.join(map(str, ahead)) or 'none'})")
            time.sleep(0.01)
        with _lease_environment(path, me["token"]):
            yield Lease(granted, time.monotonic() - t0)
    finally:
        with _mutex(d):
            if sys.platform == "win32":
                os.close(fd)
            with contextlib.suppress(OSError):
                path.unlink()
            if sys.platform != "win32":
                os.close(fd)


def _lane_count() -> int:
    raw = os.environ.get("DEV_TEST_HEAVY_LANES")
    if raw and raw.isdigit() and int(raw) > 0:
        return int(raw)
    return max(2, _cores() // 4)


@contextlib.contextmanager
def heavy_lane(label: str = "heavy test", say=lambda m: print(m, file=sys.stderr, flush=True)) -> Iterator[None]:
    """Hold one of the machine-wide HEAVY lanes while a test that spawns a multi-threaded tool runs."""
    d = slots_dir()
    n = _lane_count()
    deadline = time.time() + float(os.environ.get("DEV_TEST_LANE_WAIT_S", "600"))
    fd = -1
    noted = False
    while fd < 0:
        CHECK_CANCELLED.get()()
        for i in range(n):
            f = os.open(d / f"heavy-{i}.lane", os.O_RDWR | os.O_CREAT, 0o600)
            try:
                if not take(f):
                    os.close(f)
                    continue
            except OSError:
                os.close(f)
                raise
            fd = f
            break
        if fd >= 0:
            break
        if time.time() > deadline:
            raise TimeoutError(f"testslots: no heavy lane free for {label} before the wait limit")
        if not noted:
            noted = True
            say(f"testslots: {label} waits for one of {n} heavy lanes")
        time.sleep(0.2)
    try:
        yield
    finally:
        if fd >= 0:
            os.close(fd)                                  # closing releases the lock


def _run_command(argv: list[str], elastic: bool = False) -> int:
    import argparse
    import subprocess
    ap = argparse.ArgumentParser(prog="testslots run")
    ap.add_argument("--want", type=lambda value: 0 if value == "auto" else int(value), default=0, help="workers wanted as a ceiling (0 = auto: what the machine can spare, a fair share)")
    ap.add_argument("--min", dest="minimum", type=int, default=None)
    ap.add_argument("--label", default="command")
    ap.add_argument("--container", action="store_true")
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args(argv)
    cmd = a.cmd[1:] if a.cmd[:1] == ["--"] else a.cmd
    if not cmd:
        ap.error("give a command after --")
    if elastic:
        import testslots_runner
        return testslots_runner.run_pytest(cmd, a.want, a.label, container=a.container)
    with lease(a.want, a.minimum, label=a.label) as got:
        print(f"testslots: running with {got.workers} worker(s) (waited {got.waited:.0f}s)", file=sys.stderr, flush=True)
        return subprocess.run(cmd, env={**os.environ, "DEV_TEST_WORKERS": str(got.workers)}).returncode


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["pytest"]:
        return _run_command(argv[1:], elastic=True)
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
        print(f"  running  {s['granted']:>2}  pid {s['pid']:<7} {_who(s)}")
    for s in st["waiting"]:
        print(f"  waiting  {s['minimum']}-{s['want'] or 'auto'}  pid {s['pid']:<7} {_who(s)}")
    return 0


def _who(slot: dict) -> str:
    """A status line's owner: the workspace agent when the lease names one, then its label."""
    agent = slot.get("agent")
    if not agent:
        return slot["label"]
    return f"{agent['id']}{'/' + agent['label'] if agent.get('label') else ''} ({slot['label']})"


if __name__ == "__main__":
    sys.exit(main())
