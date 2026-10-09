"""The node's supervisor: keeps one `poolside-node` running for a state directory, restarting it when it exits.

Crash-only. The node's state is its logs on disk, so a restart loses nothing; the supervisor only starts the binary the
state directory points at (or the selected runtime's), checked against its recorded checksum on every start, and starts it
again with a growing delay if it exits. It leaves when no verified binary can be found.
"""

from __future__ import annotations

import os
import signal
import subprocess
import time
from collections.abc import Callable
from pathlib import Path

from ml_stack import node_binary, runtime
from ml_stack.files import read_json, write_json
from ml_stack.lock import Busy, only_one
from ml_stack.platform import start_process

POINTER = "node-binary.json"
RUN = "node-run.json"
LOG = "node.log"
LOCK = "supervisor.lock"
MIN_DELAY_S = 0.1
MAX_DELAY_S = 30.0
STABLE_S = 60.0
GIVE_UP_AFTER = 5
POLL_S = 0.1


def pointer(state: Path) -> dict:
    """The binary a swap chose for this state directory ({binary, sha256}), or {} when the selected runtime decides."""
    row = read_json(state / POINTER, {})
    return row if isinstance(row, dict) else {}


def point(state: Path, binary: Path | None, sha: str = "") -> None:
    """Choose the binary the next start runs; None goes back to the selected runtime's."""
    if binary is None:
        (state / POINTER).unlink(missing_ok=True)
    else:
        write_json(state / POINTER, {"binary": str(binary), "sha256": sha})


def signature(state: Path) -> tuple[bytes, bytes]:
    """What decides the binary a start runs, as bytes: the pointer file and the runtime selection. A change restarts the node."""
    def read(path: Path) -> bytes:
        try:
            return path.read_bytes()
        except OSError:
            return b""
    return read(state / POINTER), read(runtime.directory() / "selected.json")


def resolve(state: Path) -> tuple[Path, str]:
    """The verified binary a start runs and its checksum: the pointer's when set, else the selected runtime's."""
    chosen = pointer(state)
    if chosen:
        binary = Path(str(chosen.get("binary", "")))
        if not binary.is_file() or node_binary.sha256(binary) != chosen.get("sha256"):
            raise node_binary.NodeBinaryError(f"{binary} does not match the checksum it was chosen with; refusing to start it")
        return binary, str(chosen["sha256"])
    prefix = runtime.selection_prefix()
    if prefix is None:
        raise node_binary.NodeBinaryError("no runtime is selected, so there is no node binary to start")
    binary = node_binary.verified(prefix)
    return binary, node_binary.record_of(prefix)["sha256"]


def running(state: Path) -> dict:
    """The record of the node this state directory last started: pid, binary, checksum, start time, supervisor pid."""
    row = read_json(state / RUN, {})
    return row if isinstance(row, dict) else {}


def _spawn(state: Path, binary: Path, sha: str, extra: list[str]) -> subprocess.Popen:
    with (state / LOG).open("ab") as out:
        child = start_process([str(binary), "run", "--state", str(state), *extra], stdin=subprocess.DEVNULL, stdout=out,
                              stderr=subprocess.STDOUT)
    before = running(state)
    previous = before.get("previous") if before.get("sha256") == sha else {"binary": before.get("binary"), "sha256": before.get("sha256")}
    write_json(state / RUN, {"pid": child.pid, "binary": str(binary), "sha256": sha, "started_at": time.time(),
                             "supervisor": os.getpid(), "previous": previous or {}})
    return child


def _wait(child: subprocess.Popen, state: Path, wanted: tuple[bytes, bytes], stopping: Callable[[], bool]) -> int | None:
    """Wait for the child to exit; None when asked to stop or when the chosen binary changed."""
    while True:
        try:
            return child.wait(timeout=POLL_S)
        except subprocess.TimeoutExpired:
            if stopping() or signature(state) != wanted:
                return None


def _pause(state: Path, seconds: float, seen: tuple[bytes, bytes], stopping: Callable[[], bool]) -> None:
    """Sleep up to ``seconds``, waking early when asked to stop or when a swap chose another binary."""
    end = time.monotonic() + seconds
    while time.monotonic() < end and not stopping() and signature(state) == seen:
        time.sleep(POLL_S)


def _note(state: Path, text: str) -> None:
    with (state / LOG).open("a", encoding="utf-8") as out:
        out.write(f"{time.strftime('%FT%T')} supervisor: {text}\n")


def supervise(state: Path, extra: list[str] | None = None) -> int:
    """Keep the node of ``state`` running until stopped; 0 on a stop, 1 when it cannot find a verified binary or is already supervised."""
    state.mkdir(mode=0o700, parents=True, exist_ok=True)
    stop = {"now": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(now=True))
    stopping = lambda: stop["now"]  # noqa: E731
    try:
        with only_one(state / LOCK, wait=False, note="node supervisor"):
            return _loop(state, extra or [], stopping)
    except Busy:
        return 1


def _loop(state: Path, extra: list[str], stopping: Callable[[], bool]) -> int:
    delay, refused = MIN_DELAY_S, 0
    while not stopping():
        seen = signature(state)
        try:
            binary, sha = resolve(state)
        except OSError as exc:
            refused += 1
            _note(state, f"cannot start the node: {exc}")
            if refused >= GIVE_UP_AFTER:
                return 1
            _pause(state, delay, seen, stopping)
            delay = min(delay * 2, MAX_DELAY_S)
            continue
        refused, began = 0, time.monotonic()
        child = _spawn(state, binary, sha, extra)
        code = _wait(child, state, seen, stopping)
        if code is None:
            _end(child)
        _note(state, f"node {child.pid} exited {child.returncode}")
        delay = MIN_DELAY_S if code is None or code == -signal.SIGTERM or time.monotonic() - began >= STABLE_S else min(delay * 2, MAX_DELAY_S)
        _pause(state, delay, seen, stopping)
    return 0


def _end(child: subprocess.Popen) -> None:
    """Stop a node the supervisor was told to leave or replace: ask, then insist."""
    child.terminate()
    try:
        child.wait(timeout=10)
    except subprocess.TimeoutExpired:
        child.kill()
        child.wait()

