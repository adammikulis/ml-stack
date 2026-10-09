"""Supervisor-owned precollection channels and kernel observations."""
from __future__ import annotations

import hashlib
import json
import math
import os
import secrets
import select
import subprocess
import sys
import time
from pathlib import Path

import psutil
import testslots_rpc
from test_kernel_endpoint import verify_socket

CHECKS = ["read", "write", "symlink", "hardlink-canary", "hardlink-manifest", "unix", "tcp", "signal", "descendant", "admission"]


def validate_result(value: dict, nonce: str, profile: str) -> None:
    if not isinstance(value, dict) or set(value) != {"nonce", "profile", "checks"} or value != {
            "nonce": nonce, "profile": profile, "checks": CHECKS}:
        raise RuntimeError("test confinement: invalid precollection attestation")


def owned_identity(process):
    """Return the live owned child's birth and UID."""
    try:
        if process.poll() is not None:
            return None
        child = psutil.Process(process.pid)
        birth, uid = child.create_time(), child.uids().real
        if uid != os.getuid() or process.poll() is not None:
            return None
        return birth, uid
    except (psutil.Error, OSError, AttributeError):
        return None


def timeout_observation(process, identity, started):
    """Return bounded metadata for the same unreaped owned child."""
    value = {"status": "unknown", "elapsed": time.monotonic() - started}
    try:
        if identity is None or owned_identity(process) != identity:
            return value
        child = psutil.Process(process.pid)
        executable, name, cpu = child.exe(), child.name(), child.cpu_times()
        if (len(os.fsencode(executable)) > 4096 or len(name.encode()) > 256
                or not all(math.isfinite(n) and n >= 0 for n in (cpu.user, cpu.system))
                or owned_identity(process) != identity):
            return value
        value.update(status="observed", pid=process.pid, birth=identity[0], uid=identity[1],
                     executable=executable, name=name, cpu_user=cpu.user, cpu_system=cpu.system)
    except (psutil.Error, OSError, AttributeError, UnicodeError):
        pass
    return value


class Attestation:
    """Hold private bootstrap descriptors and harmless denial targets."""

    def __init__(self, confined, arguments: list[str]):
        self.confined = confined
        self.descriptors: set[int] = set()
        self.denied = None
        self.target = None
        try:
            self.prepare(arguments)
        except BaseException:
            self.close()
            raise

    def prepare(self, arguments: list[str]) -> None:
        from test_kernel_git import observe
        self.git_observation = observe(self.confined.control)
        self.denied = testslots_rpc.Admission(False)
        target_code = "import signal,time,os;signal.signal(signal.SIGUSR1,signal.SIG_IGN);os.write(1,b'r');time.sleep(3600)"
        self.target = subprocess.Popen([sys.executable, "-I", "-S", "-c", target_code],
                                       stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                       close_fds=True, start_new_session=True, env={"PATH": "/usr/bin:/bin"})
        if not select.select([self.target.stdout], [], [], 5)[0] or self.target.stdout.read(1) != b"r":
            raise RuntimeError("test confinement: owned signal target did not start")
        self.birth = psutil.Process(self.target.pid).create_time()
        self.nonce = secrets.token_hex(32)
        self.canary = self.confined.canary.read_bytes()
        self.escape = self.confined.scratch / "precollection-escape"
        self.escape.symlink_to(self.confined.canary)
        self.report, output = os.pipe()
        acknowledgement, self.release = os.pipe()
        self.descriptors.update((self.report, output, acknowledgement, self.release))
        self.pass_fds = (output, acknowledgement)
        self.code = self.confined.control / "bootstrap.py"
        self.code.write_bytes(Path(__file__).with_name("test_kernel_bootstrap.py").read_bytes())
        self.code.chmod(0o400)
        self.code_hash = hashlib.sha256(self.code.read_bytes()).hexdigest()
        self.helper = self.confined.control / "test_kernel_endpoint.py"
        self.helper.write_bytes(Path(__file__).with_name("test_kernel_endpoint.py").read_bytes())
        self.helper.chmod(0o400)
        self.helper_hash = hashlib.sha256(self.helper.read_bytes()).hexdigest()
        self.manifest = self.confined.control / "bootstrap.json"
        self.manifest.touch(mode=0o600)
        self.arguments = arguments
        self.argv = [sys.executable, "-I", "-S", str(self.code), str(self.manifest), str(output), str(acknowledgement)]

    def seal(self, profile: Path) -> None:
        self.profile = hashlib.sha256(profile.read_bytes()).hexdigest()
        value = {"nonce": self.nonce, "profile": self.profile, "argv": self.arguments,
                 "git_observation": self.git_observation,
                 "canary": str(self.confined.canary), "escape": str(self.escape),
                 "denied_unix": self.confined.denied_endpoint, "denied_tcp": self.denied.endpoint,
                 "target_pid": self.target.pid, "admission": self.confined.environment["DEV_TEST_PYTEST_ENDPOINT"],
                 "identity": self.confined.environment["DEV_TEST_PYTEST_IDENTITY"], "helper": str(self.helper),
                 "terminal": self.confined.environment["DEV_TEST_PTY_ENDPOINT"],
                 "terminal_identity": self.confined.environment["DEV_TEST_PTY_IDENTITY"],
                 "token": self.confined.environment["DEV_TEST_PYTEST_TOKEN"]}
        data = json.dumps(value).encode()
        if len(data) > 65536:
            raise RuntimeError("test confinement: bootstrap manifest exceeds bound")
        self.manifest.write_bytes(data)
        self.manifest.chmod(0o400)
        self.manifest_hash = hashlib.sha256(self.manifest.read_bytes()).hexdigest()

    def launched(self, process) -> None:
        started = time.monotonic()
        identity = owned_identity(process)
        for fd in self.pass_fds:
            os.close(fd)
            self.descriptors.remove(fd)
        deadline = time.monotonic() + min(15, self.confined.observation_seconds)
        data = bytearray()
        while b"\n" not in data:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or process.poll() is not None or not select.select([self.report], [], [], remaining)[0]:
                if self.confined.artifacts is not None:
                    self.confined.artifacts.proof["precollection_timeout"] = timeout_observation(process, identity, started)
                raise RuntimeError("test confinement: precollection attestation timed out or exited")
            block = os.read(self.report, 4097 - len(data))
            if not block or len(data) + len(block) > 4096:
                raise RuntimeError("test confinement: precollection attestation EOF or overflow")
            data.extend(block)
        validate_result(json.loads(data), self.nonce, self.profile)
        self.observed()
        self.confined.recheck_images()
        os.write(self.release, b"a")
        for fd in (self.report, self.release):
            os.close(fd)
            self.descriptors.remove(fd)

    def observed(self) -> None:
        for endpoint, key in (("DEV_TEST_PYTEST_ENDPOINT", "DEV_TEST_PYTEST_IDENTITY"), ("DEV_TEST_PTY_ENDPOINT", "DEV_TEST_PTY_IDENTITY")):
            path = self.confined.environment[endpoint].removeprefix("unix:")
            verify_socket(path, json.loads(self.confined.environment[key]))
        if self.confined.canary.read_bytes() != self.canary or self.denied.request_count != 0:
            raise RuntimeError("test confinement: supervisor denial canary was accessed")
        if select.select([self.confined.denied_listener], [], [], 0)[0]:
            raise RuntimeError("test confinement: forbidden Unix listener received a connection")
        if self.target.poll() is not None or psutil.Process(self.target.pid).create_time() != self.birth:
            raise RuntimeError("test confinement: owned signal target identity changed")
        for path, digest in ((self.code, self.code_hash), (self.helper, self.helper_hash), (self.manifest, self.manifest_hash),
                             (self.confined.control / "profile.sb", self.profile)):
            if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise RuntimeError("test confinement: immutable bootstrap input changed")

    def close(self) -> None:
        for fd in self.descriptors:
            os.close(fd)
        self.descriptors.clear()
        if self.target is not None:
            if self.target.poll() is None:
                self.target.terminate()
                try:
                    self.target.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.target.kill()
                    self.target.wait()
            self.target.stdout.close()
        if self.denied is not None:
            self.denied.finish()
