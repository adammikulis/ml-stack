"""Run a command under a policy: choose the backend, start it in its own process group, hold it
to its limits, and report what the sandbox refused."""

from __future__ import annotations

import contextlib
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

if os.name == "posix":
    import resource
else:
    resource = None

from poolhouse.platform import start_process
from poolhouse.sandbox.backend import Backend, SandboxUnavailable
from poolhouse.sandbox.bubblewrap import Bubblewrap
from poolhouse.sandbox.policy import AllowUnsandboxed, Limits, Policy
from poolhouse.sandbox.seatbelt import Seatbelt

__all__ = ["Events", "Result", "SandboxViolation", "backend", "run", "wrapped"]

logger = logging.getLogger("poolhouse.sandbox")

Events = Callable[[str, dict[str, object]], None]
"""``on(event, fields)`` -- what happened in the sandbox, and what came with it."""


WEIGHT = ("network", "process-exec", "file-write", "file-read-data")
"""Operations in the order a refusal is worth reporting first."""


class SandboxViolation(RuntimeError):
    """The sandbox refused something the command tried."""


@dataclass(slots=True)
class Result:
    """How a run ended."""

    argv: list[str]
    returncode: int
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    truncated: bool = False
    sandboxed: bool = True
    backend: str = ""
    seconds: float = 0.0
    denials: list[dict[str, str]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out

    def check(self) -> Result:
        """This result, or `SandboxViolation` saying what was refused."""
        if self.denials:
            first = min(self.denials, key=lambda d: next(
                (i for i, prefix in enumerate(WEIGHT) if d["operation"].startswith(prefix)),
                len(WEIGHT)))
            raise SandboxViolation(
                f"the sandbox refused {first['operation']} {first['target']} "
                f"({len(self.denials)} refusal(s) running {Path(self.argv[0]).name})")
        return self


def backend() -> Backend:
    """The backend this platform uses; one that reports unavailable when there is none."""
    return Seatbelt() if sys.platform == "darwin" else Bubblewrap()


def wrapped(argv: Sequence[str], policy: Policy, *, via: Backend | None = None,
            unsandboxed: AllowUnsandboxed | None = None) -> tuple[list[str], str, Backend | None]:
    """``(argv, tag, backend)`` -- the backend is None when running without one -- that runs ``argv`` under ``policy``. When no backend can
    hold it, `SandboxUnavailable` unless ``unsandboxed`` names the reason to run without one."""
    policy = policy.validated()
    chosen = via or backend()
    state = chosen.available()
    if state.ok:
        made = chosen.wrap(argv, policy)
        return made.argv, made.tag, chosen
    if unsandboxed is None:
        raise SandboxUnavailable(
            f"{policy.name}: no sandbox is available ({chosen.name}: {state.reason}); the "
            f"command was not run. Pass AllowUnsandboxed(reason) to run it without one.")
    logger.warning("running WITHOUT a sandbox (%s): %s", unsandboxed.reason, argv[0])
    return list(argv), "", None


def _limits(limits: Limits) -> Callable[[], None]:
    if resource is None:
        raise SandboxUnavailable("process resource limits require a POSIX platform; command was not run")
    def apply() -> None:
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        for which, value in ((resource.RLIMIT_CPU, limits.cpu_seconds),
                             (resource.RLIMIT_FSIZE, limits.file_bytes),
                             (resource.RLIMIT_NOFILE, limits.open_files)):
            if value is not None:
                resource.setrlimit(which, (value, value))
    return apply


class _Capture(threading.Thread):
    def __init__(self, stream: Any, cap: int | None) -> None:
        super().__init__(daemon=True)
        self.stream, self.cap = stream, cap
        self.data = bytearray()
        self.over = False

    def run(self) -> None:
        read = getattr(self.stream, "read1", None) or self.stream.read
        while chunk := read(65536):
            room = len(chunk) if self.cap is None else max(0, self.cap - len(self.data))
            self.data += chunk[:room]
            if room < len(chunk):
                self.over = True


def _kill_group(pid: int) -> None:
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(pid, signal.SIGKILL)


def _wait(proc: subprocess.Popen[bytes], deadline: float | None, *captures: _Capture) -> bool:
    """Wait for ``proc`` to exit, or for the deadline or an output cap; True on the deadline."""
    while proc.poll() is None:
        if deadline is not None and time.time() >= deadline:
            return True
        if any(c.over for c in captures):
            return False
        time.sleep(0.01)
    return False


def run(argv: Sequence[str], policy: Policy, *,  # noqa: PLR0913 - one keyword per choice
        unsandboxed: AllowUnsandboxed | None = None,
        cwd: str | os.PathLike[str] | None = None, stdin: bytes | None = None,
        on_event: Events | None = None, via: Backend | None = None,
        diagnose: str = "failure") -> Result:
    """Run ``argv`` under ``policy``. The environment is exactly ``policy.env``. The process
    starts in a group of its own; the wall-clock limit or an output over the cap kills that
    group and nothing else. A run that fails with the sandbox having refused something carries
    the refusals in ``Result.denials`` and reports them to ``on_event``."""
    emit = _emitter(on_event)
    try:
        full, tag, chosen = wrapped(argv, policy, via=via, unsandboxed=unsandboxed)
    except SandboxUnavailable as exc:
        emit("sandbox.unavailable", severity="warning", command=str(argv[0]),
             policy=policy.name, reason=str(exc))
        raise
    name = chosen.name if chosen else "none"
    if chosen is None:
        emit("sandbox.unsandboxed", severity="warning", command=str(argv[0]),
             policy=policy.name, reason=unsandboxed.reason if unsandboxed else "")
    limits = policy.limits
    if cwd is None:
        cwd = (policy.write or policy.read or ("/",))[0]
    began = time.time()
    try:
        proc = start_process(
            full, stdin=subprocess.PIPE if stdin is not None else subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=HOOKS.environment(policy.env), cwd=cwd,
            preexec_fn=_limits(limits))
    except OSError as exc:
        raise SandboxUnavailable(f"{policy.name}: could not start {argv[0]}: {exc}") from exc
    out, err = _Capture(proc.stdout, limits.output_bytes), _Capture(proc.stderr, limits.output_bytes)
    out.start()
    err.start()
    if stdin is not None and proc.stdin is not None:
        with contextlib.suppress(OSError):
            proc.stdin.write(stdin)
            proc.stdin.close()
    deadline = None if limits.wall_seconds is None else began + limits.wall_seconds
    timed_out = _wait(proc, deadline, out, err)
    capped = out.over or err.over
    if timed_out or capped:
        _kill_group(proc.pid)
    proc.wait()
    _kill_group(proc.pid)
    out.join(2)
    err.join(2)
    result = Result(list(argv), proc.returncode, bytes(out.data).decode("utf-8", "replace"),
                    bytes(err.data).decode("utf-8", "replace"), timed_out, capped,
                    name != "none", name, round(time.time() - began, 3))
    if chosen is not None and _diagnose(diagnose, result):
        for wait in (0.5, 1.0, 1.5) if result.returncode != 0 else (0.0,):
            time.sleep(wait)
            result.denials = chosen.denials(tag, began)
            if result.denials:
                break
    if result.denials:
        emit("sandbox.denied", severity="warning", command=str(argv[0]), policy=policy.name,
             count=len(result.denials), first=result.denials[0], backend=name)
    if timed_out:
        emit("sandbox.timeout", severity="notice", command=str(argv[0]), policy=policy.name,
             seconds=limits.wall_seconds)
    return result


def _diagnose(when: str, result: Result) -> bool:
    if when == "always":
        return True
    return when == "failure" and (result.returncode != 0 or bool(result.stderr))


class Hooks:
    """What watches every run: ``events`` hears each event, ``scrub`` filters the environment a
    child gets. Set by the node's sentinel (`poolhouse.sentinel.default`) so that sandbox imports
    nothing from it."""

    def __init__(self) -> None:
        self.events: Events | None = None
        self.scrub: Callable[[Mapping[str, str]], dict[str, str]] | None = None

    def watch(self, events: Events | None,
              scrub: Callable[[Mapping[str, str]], dict[str, str]] | None) -> None:
        self.events, self.scrub = events, scrub

    def environment(self, env: Mapping[str, str]) -> dict[str, str]:
        return self.scrub(env) if self.scrub is not None else dict(env)


HOOKS = Hooks()


def _emitter(on_event: Events | None) -> Callable[..., None]:
    """Reports to the caller's ``on_event`` and to whatever watches every run (the sentinel)."""
    def emit(event: str, **fields: object) -> None:
        for call in (HOOKS.events, on_event):
            if call is not None:
                with contextlib.suppress(Exception):
                    call(event, dict(fields))
    return emit

