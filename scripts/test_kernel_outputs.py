"""Pinned supervisor output destinations outside protected state."""
from __future__ import annotations

import contextlib
import fcntl
import math
import os
import select
import stat
import sys
import time
from pathlib import Path

LIMIT = 16 * 1024 * 1024


def descriptor_path(fd: int) -> Path:
    raw = fcntl.fcntl(fd, getattr(fcntl, "F_GETPATH", 50), bytes(1024))
    value = os.fsdecode(raw.split(b"\0", 1)[0])
    if not value or not Path(value).is_absolute():
        raise RuntimeError("test confinement: output descriptor has no physical path")
    return Path(value)


class OutputSink:
    """Pin and validate an inherited output descriptor."""

    def __init__(self, fd: int, roots: tuple[Path, ...], *, private: bool = False):
        self.fd = os.dup(fd)
        self.roots = tuple(root.resolve() for root in roots)
        self.identities = {(value.st_dev, value.st_ino) for root in self.roots
                           if root.exists() for value in (root.stat(),)}
        self.private = private
        self.flags = fcntl.fcntl(self.fd, fcntl.F_GETFL)
        self.total = 0
        try:
            self.identity = self.validate()
        except BaseException:
            self.close()
            raise

    def validate(self) -> tuple[int, int]:
        info = os.fstat(self.fd)
        identity = info.st_dev, info.st_ino
        if not stat.S_ISREG(info.st_mode):
            if self.private:
                raise RuntimeError("test confinement: JUnit destination is not a regular file")
            if stat.S_ISSOCK(info.st_mode):
                raise RuntimeError("test confinement: output socket is not admitted")
            if stat.S_ISFIFO(info.st_mode):
                raw = fcntl.fcntl(self.fd, getattr(fcntl, "F_GETPATH", 50), bytes(1024))
                if raw.split(b"\0", 1)[0]:
                    raise RuntimeError("test confinement: named output FIFO is not admitted")
            if not (stat.S_ISFIFO(info.st_mode) or os.isatty(self.fd)
                    or info.st_rdev == Path("/dev/null").stat().st_rdev):
                raise RuntimeError("test confinement: unsupported output descriptor")
            return identity
        if (info.st_uid != os.getuid() or info.st_nlink != 1
                or info.st_mode & (0o077 if self.private else 0o022)):
            raise RuntimeError("test confinement: output file is not an owned unlinked-writer sink")
        path = descriptor_path(self.fd)
        resolved = path.resolve(strict=True)
        if path.is_symlink() or (resolved.stat().st_dev, resolved.stat().st_ino) != identity:
            raise RuntimeError("test confinement: output descriptor path changed")
        if any(resolved.is_relative_to(root) for root in self.roots):
            raise RuntimeError("test confinement: output overlaps protected roots")
        identities = self.identities | {(value.st_dev, value.st_ino) for root in self.roots
                                        if root.exists() for value in (root.stat(),)}
        for parent in (resolved, *resolved.parents):
            value = parent.stat()
            if (value.st_dev, value.st_ino) in identities:
                raise RuntimeError("test confinement: output physically aliases protected roots")
        return identity

    def write(self, value: bytes) -> None:
        info = os.fstat(self.fd)
        if self.total + len(value) > LIMIT or (stat.S_ISREG(info.st_mode) and info.st_size + len(value) > LIMIT):
            raise RuntimeError("test confinement: supervisor output exceeds bound")
        if self.validate() != self.identity:
            raise RuntimeError("test confinement: output identity changed")
        self.total += len(value)
        deadline = time.monotonic() + 5
        remaining = memoryview(value)
        fcntl.fcntl(self.fd, fcntl.F_SETFL, self.flags | os.O_NONBLOCK)
        try:
            while remaining:
                wait = deadline - time.monotonic()
                if wait <= 0 or not select.select([], [self.fd], [], wait)[1]:
                    raise RuntimeError("test confinement: supervisor output write timed out")
                try:
                    written = os.write(self.fd, remaining[:65536])
                except BlockingIOError:
                    continue
                if not written:
                    raise RuntimeError("test confinement: supervisor output write failed")
                remaining = remaining[written:]
        finally:
            fcntl.fcntl(self.fd, fcntl.F_SETFL, self.flags)

    def close(self) -> None:
        if self.fd >= 0:
            os.close(self.fd)
            self.fd = -1


def junit_sink(path: Path, roots: tuple[Path, ...]) -> OutputSink:
    if not path.is_absolute() or path.is_symlink():
        raise RuntimeError("test confinement: JUnit destination is a symlink or relative path")
    fd = os.open(path, os.O_WRONLY | os.O_NOFOLLOW)
    try:
        return OutputSink(fd, roots, private=True)
    finally:
        os.close(fd)


class ArtifactOutputs:
    """Hold fresh inert output files for a maintained test run."""

    def __init__(self, environment: dict[str, str], *, strict: bool = False):
        import secrets

        from ml_stack.activity.source_snapshot import private_namespace, protected_roots
        self.roots = protected_roots(environment)
        self.observation_seconds = min(15, float(environment.get("DEV_TEST_WAIT_S", "3600")))
        if not math.isfinite(self.observation_seconds) or self.observation_seconds <= 0:
            raise RuntimeError("test confinement: observation deadline must be finite and positive")
        self.directory = private_namespace(environment, "ml-stack-test-artifacts-")
        self.run_id = secrets.token_hex(32)
        self.sinks = {}
        self.proof = {}
        info = self.directory.stat()
        self.identity = info.st_dev, info.st_ino, info.st_uid
        try:
            for name in ("console.log", "junit.xml", "evidence.result.json"):
                fd = os.open(self.directory / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
                try:
                    self.sinks[name] = OutputSink(fd, self.roots, private=True)
                finally:
                    os.close(fd)
        except BaseException:
            self.close()
            raise

    def validate(self) -> None:
        info = self.directory.lstat()
        if ((info.st_dev, info.st_ino, info.st_uid) != self.identity or not stat.S_ISDIR(info.st_mode)
                or info.st_mode & 0o077 or self.directory.resolve() != self.directory):
            raise RuntimeError("test confinement: artifact namespace changed")
        for name, sink in self.sinks.items():
            if os.fstat(sink.fd).st_size > LIMIT:
                raise RuntimeError("test confinement: artifact file exceeds bound")
            if sink.validate() != sink.identity or descriptor_path(sink.fd) != self.directory / name:
                raise RuntimeError("test confinement: artifact descriptor changed")

    def boundary(self) -> dict:
        self.validate()
        return {"mode": "private-artifacts", "run_id": self.run_id, "directory": str(self.directory),
                "identity": self.identity, "files": {name: sink.identity for name, sink in self.sinks.items()},
                "proof": self.proof}

    def announce(self) -> None:
        import json
        if stat.S_ISREG(os.fstat(sys.stdout.fileno()).st_mode):
            sink = OutputSink(sys.stdout.fileno(), self.roots)
            sink.close()
        print("test: private artifacts " + json.dumps(str(self.directory)), flush=True)

    @contextlib.contextmanager
    def redirect(self):
        self.validate()
        saved = []
        try:
            for stream, fd in ((sys.stdout, 1), (sys.stderr, 2)):
                stream.flush()
                if stat.S_ISREG(os.fstat(fd).st_mode):
                    sink = OutputSink(fd, self.roots)
                    sink.close()
                saved.append((fd, os.dup(fd)))
                os.dup2(self.sinks["console.log"].fd, fd)
            yield
        finally:
            try:
                sys.stdout.flush()
                sys.stderr.flush()
            finally:
                for fd, original in reversed(saved):
                    os.dup2(original, fd)
                    os.close(original)

    def observe(self) -> None:
        self.validate()

    def close(self) -> None:
        for sink in self.sinks.values():
            sink.close()
        self.sinks.clear()
