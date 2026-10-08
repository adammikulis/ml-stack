"""Owned admission shutdown and escaped descendant exclusion probes."""
from __future__ import annotations

import json
import os
import secrets
import select
import subprocess
import sys
import time

import psutil
import testslots_rpc
from test_kernel_endpoint import connection, verify_socket

RETIRED = ["closed-cpu", "closed-terminal", "unlink", "rebind", "create-denied", "escaped-read", "escaped-tcp"]


def terminal_eof(stream) -> None:
    data = stream.read(4097)
    if len(data) > 4096:
        raise RuntimeError("test confinement: shutdown response exceeds bound")
    if data and json.loads(data) not in ({"error": "testslots: supervisor stopped"},
                                       {"error": "testslots: admission connection closed"}):
        raise RuntimeError("test confinement: shutdown response grants unexpected authority")
    if stream.read(1):
        raise RuntimeError("test confinement: admission connection survived shutdown")


def held_shutdown() -> None:
    server = testslots_rpc.UnixAdmission(False)
    channels = []
    finished = False
    try:
        identity = json.dumps(server.identity)
        channels.extend((connection(server.endpoint, identity), connection(server.endpoint, identity)))
        server.collected.set()
        channels[1].settimeout(10)
        with channels[1].makefile("rwb") as stream:
            testslots_rpc._write(stream, {"operation": "acquire", "token": server.token, "parent": None,
                                         "label": "kernel lifecycle probe", "phase": "test", "heavy": False})
            result = testslots_rpc._read(stream)
            if set(result) != {"lease"} or result["lease"] not in server.active:
                raise RuntimeError("test confinement: lifecycle probe did not hold an owned CPU lease")
            started = time.monotonic()
            server.finish()
            finished = True
            if time.monotonic() - started > 10 or server.active or server.running_threads or server.connections:
                raise RuntimeError("test confinement: held or idle admission did not stop")
            verify_socket(server.server_address, server.identity)
            channels[1].settimeout(1)
            terminal_eof(stream)
        channels[0].settimeout(1)
        with channels[0].makefile("rb") as stream:
            terminal_eof(stream)
    finally:
        for channel in channels:
            channel.close()
        if not finished:
            server.finish()


def retired_probe(confined) -> None:
    confined.bootstrap.observed()
    nonce = secrets.token_hex(32)
    report, output = os.pipe()
    acknowledgement, release = os.pipe()
    descriptors = {report, output, acknowledgement, release}
    process = None
    try:
        argv = [*confined.wrapped.argv[:4], sys.executable, "-I", "-S", str(confined.bootstrap.code),
                str(confined.bootstrap.manifest), "retired", nonce, str(output), str(acknowledgement)]
        process = subprocess.Popen(argv, env=confined.environment, close_fds=True, pass_fds=(output, acknowledgement),
                                   stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                                   start_new_session=True)
        for fd in (output, acknowledgement):
            os.close(fd)
            descriptors.remove(fd)
        result = read_report(report, min(10, confined.observation_seconds))
        if isinstance(result, dict) and set(result) == {"nonce", "error"} and result["nonce"] == nonce:
            raise RuntimeError("test confinement: retired probe failed: " + str(result["error"])[:256])
        if not isinstance(result, dict) or set(result) != {"nonce", "pid", "checks"} or result["nonce"] != nonce or result["checks"] != RETIRED:
            raise RuntimeError("test confinement: invalid escaped descendant report")
        observe_exit(result["pid"], argv, release, process.pid)
        if process.wait(timeout=5) != 0:
            raise RuntimeError("test confinement: retired endpoint probe failed")
        confined.bootstrap.observed()
    finally:
        for fd in descriptors:
            os.close(fd)
        if process is not None:
            try:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=5)
            finally:
                process.stderr.close()


def read_report(descriptor: int, seconds: float) -> dict:
    deadline = time.monotonic() + seconds
    data = bytearray()
    while b"\n" not in data:
        remaining = deadline - time.monotonic()
        if remaining <= 0 or not select.select([descriptor], [], [], remaining)[0]:
            raise RuntimeError("test confinement: escaped descendant proof timed out")
        block = os.read(descriptor, 4097 - len(data))
        if not block or len(data) + len(block) > 4096:
            raise RuntimeError("test confinement: escaped descendant proof EOF or overflow")
        data.extend(block)
    return json.loads(data)


def observe_exit(pid, argv: list[str], release: int, parent_pid: int = 0) -> None:
    if type(pid) is not int or pid <= 0:
        raise RuntimeError("test confinement: invalid owned probe PID")
    child = psutil.Process(pid)
    if child.uids().real != os.getuid() or child.cmdline() != argv[4:]:
        raise RuntimeError("test confinement: escaped probe identity does not match owned launch")
    birth = child.create_time()
    if os.getsid(pid) == parent_pid or os.getpgid(pid) == parent_pid:
        raise RuntimeError("test confinement: probe did not escape its original process group")
    os.write(release, b"a")
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            current = psutil.Process(pid)
            if current.create_time() != birth or current.status() == psutil.STATUS_ZOMBIE:
                return
        except psutil.NoSuchProcess:
            return
        time.sleep(.01)
    raise RuntimeError("test confinement: owned escaped probe did not exit")
