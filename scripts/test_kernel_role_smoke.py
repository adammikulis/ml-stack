"""Supervisor-owned fixed holder import and runtime-verification smoke."""
from __future__ import annotations

import hashlib
import json
import os
import selectors
import subprocess
import sys
import time
from contextlib import suppress
from dataclasses import replace
from pathlib import Path

import psutil
from test_kernel_role_home import IMPORT_NODE

from poolhouse.platform import start_process, terminate_process_group
from poolhouse.sandbox.seatbelt import Seatbelt

CONTROL_NODE = "tests/test_kernel_role_home.py::test_owned_smoke_reaps_exited_leader_and_known_stdout_child"
CONTROL_SCRIPT = (
    "import sys;assert sys.stdin.buffer.read(1)==b'G';import os,time,json;"
    "r,w=os.pipe();pid=os.fork();"
    "\nif pid==0:\n os.close(w);os.read(r,1);time.sleep(30);os._exit(0)"
    "\nos.close(r);print(json.dumps({'child':pid}),flush=True);"
    "assert sys.stdin.buffer.read(1)==b'C';os.close(w);os._exit(0)"
)

SCRIPT = (
    "import sys;assert sys.stdin.buffer.read(1)==b'G';import site;site.main();"
    "import json;from pathlib import Path;from importlib.metadata import distribution;"
    "import poolhouse,psutil;from poolhouse.serve import holder_protocol;"
    "from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey;"
    "from poolhouse import runtime;"
    "d=distribution('poolhouse');p=Path(poolhouse.__file__).resolve();"
    "r=runtime.Runtime(Path(sys.prefix),'a'*40,'0.1.0','darwin-arm64-cpython-313');"
    "assert runtime.verify(r)==r;"
    "key=Ed25519PrivateKey.generate();data=b'owned inert fixture';"
    "key.public_key().verify(key.sign(data),data);"
    "print(json.dumps({'prefix':sys.prefix,'package':str(p),'version':d.version,"
    "'stamp':p.with_name('fleet').joinpath('built-from').read_text().strip(),"
    "'holder':holder_protocol.__file__,'psutil':psutil.__file__}))"
)


def read_output(process, deadline: float) -> tuple[bytes, bytes]:
    buffers = {process.stdout: bytearray(), process.stderr: bytearray()}
    with selectors.DefaultSelector() as ready:
        for stream in buffers:
            ready.register(stream, selectors.EVENT_READ)
        while ready.get_map():
            remaining = deadline - time.monotonic() - 1
            if remaining <= 0 or not (events := ready.select(remaining)):
                raise TimeoutError("holder smoke: output deadline exceeded")
            for event, _ in events:
                block = os.read(event.fileobj.fileno(), 4096)
                if not block:
                    ready.unregister(event.fileobj)
                buffers[event.fileobj].extend(block)
                if sum(map(len, buffers.values())) > 8192:
                    raise RuntimeError("holder smoke: output exceeds bound")
    return bytes(buffers[process.stdout]), bytes(buffers[process.stderr])


def owned_group(process, expected: tuple, deadline: float) -> None:
    owner = psutil.Process(process.pid)
    before = (owner.create_time(), owner.uids().real)
    if before != expected[:2] or before[1] != os.getuid() or expected[2:] != (process.pid, process.pid):
        raise RuntimeError("holder smoke: owned process generation changed")
    try:
        if owner.status() != psutil.STATUS_ZOMBIE and (os.getsid(process.pid), os.getpgid(process.pid)) != expected[2:]:
            raise RuntimeError("holder smoke: owned session changed")
    except ProcessLookupError:
        if owner.status() != psutil.STATUS_ZOMBIE:
            raise RuntimeError("holder smoke: owned session unavailable") from None
    if (owner.create_time(), owner.uids().real) != expected[:2]:
        raise RuntimeError("holder smoke: owned process changed during group inspection")
    if live_group(process.pid, deadline):
        with suppress(ProcessLookupError):
            terminate_process_group(process, force=True)


def live_group(group: int, deadline: float) -> bool:
    live = False
    for pid in psutil.pids():
        if time.monotonic() >= deadline:
            raise TimeoutError("holder smoke: group observation deadline exceeded")
        try:
            if psutil.Process(pid).status() != psutil.STATUS_ZOMBIE and os.getpgid(pid) == group:
                live = True
        except (ProcessLookupError, psutil.NoSuchProcess):
            continue
        except (OSError, psutil.Error) as exc:
            raise RuntimeError("holder smoke: group membership unavailable") from exc
    return live


def capture_generation(process) -> tuple:
    owner = psutil.Process(process.pid)
    captured = (owner.create_time(), owner.uids().real, os.getsid(process.pid), os.getpgid(process.pid))
    if captured[1] != os.getuid() or captured[2:] != (process.pid, process.pid):
        raise RuntimeError("holder smoke: initial owned session unavailable")
    return captured


def execute(argv: list[str], environment: dict[str, str], deadline: float, state: dict | None = None, after_gate=None) -> dict:
    state = {} if state is None else state
    generation = None
    output, error = b"", b""
    process = start_process(argv, env=environment, close_fds=True, stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        state.update(pid=process.pid, gate_released=False, cleanup="pending")
        generation = capture_generation(process)
        state.update(started=generation[0], uid=generation[1], sid=generation[2], group=generation[3])
        if time.monotonic() >= deadline - 1:
            raise TimeoutError("holder smoke: capture deadline exceeded")
        process.stdin.write(b"G")
        process.stdin.flush()
        state["gate_released"] = True
        if after_gate is not None:
            after_gate(process, generation, deadline)
        process.stdin.close()
        output, error = read_output(process, deadline)
        owner = psutil.Process(process.pid)
        while owner.status() != psutil.STATUS_ZOMBIE:
            if time.monotonic() >= deadline - 1:
                raise TimeoutError("holder smoke: owned leader did not finish")
            time.sleep(.005)
    finally:
        cleanup_verified = False
        try:
            if generation is not None:
                owned_group(process, generation, deadline)
            else:
                process.stdin.close()
                process.kill()
            cleanup_verified = True
        except BaseException as exc:
            state.update(cleanup="unknown", cleanup_error=type(exc).__name__)
            raise
        finally:
            try:
                process.wait(timeout=max(.001, deadline - time.monotonic()))
                state["exit"] = process.returncode
                if cleanup_verified:
                    state["cleanup"] = ("captured group checked and leader reaped" if generation is not None
                                        else "pre-gate direct child reaped")
            finally:
                process.stdout.close()
                process.stderr.close()
                if not process.stdin.closed:
                    process.stdin.close()
    return {"exit": process.returncode, "stdout": output.decode(), "stderr": error.decode(errors="replace"),
            "pid": process.pid, "started": generation[0], "uid": generation[1], "group": process.pid}


def observe(confined, base):
    """Run the fixed smoke before pytest and publish its private receipt."""
    assets = confined.holder_assets
    if assets is None or set(assets.role_cases) != {IMPORT_NODE}:
        raise RuntimeError("holder smoke: exact normal-only case required")
    entry = assets.role_cases[IMPORT_NODE]
    runtime = entry["prepared"]["runtime"]
    if any(record["path"].endswith(".pth") for record in entry["prepared"]["files"]):
        raise RuntimeError("holder smoke: startup path hooks are not admitted")
    policy = replace(base, env={**base.env, "POOLHOUSE_HOME": entry["home"]}).validated()
    profile_path = confined.control / "holder-import-smoke.sb"
    wrapped = Seatbelt().wrap([runtime["python"], "-I", "-S", "-c", SCRIPT], policy, profile_path=profile_path)
    with profile_path.open("a") as stream:
        stream.write("\n" + "\n".join(assets.role_rules) + "\n")
    record = {"status": "pending", "scope": "fixed-holder-import-runtime-verification",
              "case": IMPORT_NODE, "bank_sha256": assets.role_bank_sha256,
              "profile_sha256": hashlib.sha256(profile_path.read_bytes()).hexdigest(),
              "script_sha256": hashlib.sha256(SCRIPT.encode()).hexdigest()}
    path = confined.control / "holder-import-smoke.json"
    try:
        confined.recheck_images()
        record.update(execute(wrapped.argv, dict(policy.env), time.monotonic() + 10, record))
        if record["exit"]:
            raise RuntimeError("holder smoke: fixed command failed: " + record["stderr"])
        value = json.loads(record.pop("stdout"))
        prefix = Path(runtime["prefix"])
        if (set(value) != {"prefix", "package", "version", "stamp", "holder", "psutil"}
                or value["prefix"] != str(prefix) or value["version"] != runtime["version"]
                or value["stamp"] != runtime["commit"]
                or Path(value["package"]) != prefix / "lib/python3.13/site-packages/poolhouse/__init__.py"
                or not all(Path(value[key]).is_relative_to(prefix) for key in ("holder", "psutil"))):
            raise RuntimeError("holder smoke: imported identity does not match pinned runtime")
        assets.recheck()
        record.update(status="passed", identity=value)
    except BaseException as exc:
        record.update(status="failed", error_type=type(exc).__name__)
        raise
    finally:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o400)
        with os.fdopen(fd, "w") as stream:
            json.dump(record, stream, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
    return path, record


def capture_stdout_child(process, generation, deadline: float, record: dict) -> None:
    raw = bytearray()
    with selectors.DefaultSelector() as ready:
        ready.register(process.stdout, selectors.EVENT_READ)
        while not raw.endswith(b"\n"):
            remaining = deadline - time.monotonic() - 1
            if remaining <= 0 or not ready.select(remaining):
                raise TimeoutError("holder control: child announcement deadline")
            block = os.read(process.stdout.fileno(), 1)
            if not block or len(raw) >= 256:
                raise RuntimeError("holder control: invalid child announcement")
            raw.extend(block)
    value = json.loads(raw)
    if type(value) is not dict or set(value) != {"child"} or type(value["child"]) is not int or value["child"] <= 0:
        raise RuntimeError("holder control: invalid child identity")
    child = psutil.Process(value["child"])
    identity = (child.create_time(), child.uids().real, os.getsid(child.pid), os.getpgid(child.pid))
    if (child.ppid() != process.pid or child.status() == psutil.STATUS_ZOMBIE
            or identity[0] < generation[0] or identity[1:] != (os.getuid(), process.pid, process.pid)
            or (child.create_time(), child.uids().real) != identity[:2]):
        raise RuntimeError("holder control: child generation unavailable")
    record.update(child_pid=child.pid, child_started=identity[0], child_uid=identity[1],
                  child_sid=identity[2], child_group=identity[3])
    process.stdin.write(b"C")
    process.stdin.flush()


def observe_control(confined, policy):
    profile = confined.control / "holder-control.sb"
    wrapped = Seatbelt().wrap([sys.executable, "-I", "-S", "-c", CONTROL_SCRIPT], policy, profile_path=profile)
    record = {"status": "pending", "scope": "known-exited-leader-stdout-child",
              "profile_sha256": hashlib.sha256(profile.read_bytes()).hexdigest(),
              "script_sha256": hashlib.sha256(CONTROL_SCRIPT.encode()).hexdigest()}
    path = confined.control / "holder-control.json"
    deadline = time.monotonic() + 10
    try:
        confined.recheck_images()
        try:
            execute(wrapped.argv, dict(policy.env), deadline, record,
                    lambda p, g, d: capture_stdout_child(p, g, d, record))
        except TimeoutError as exc:
            if (str(exc) != "holder smoke: output deadline exceeded"
                    or record.get("cleanup") != "captured group checked and leader reaped"
                    or record.get("exit") != 0 or "child_pid" not in record):
                raise
            record["expected_output_timeout"] = True
        else:
            raise RuntimeError("holder control: stdout child did not retain the pipe")
        while True:
            try:
                child = psutil.Process(record["child_pid"])
                if child.create_time() != record["child_started"]:
                    raise RuntimeError("holder control: known child generation changed")
                if child.status() == psutil.STATUS_ZOMBIE:
                    record["child_terminal"] = "zombie"
                    break
            except psutil.NoSuchProcess:
                record["child_terminal"] = "gone"
                break
            if time.monotonic() >= deadline:
                raise TimeoutError("holder control: known child remains live")
            time.sleep(.005)
        record["status"] = "passed"
    except BaseException as exc:
        record.update(status="failed", error_type=type(exc).__name__)
        raise
    finally:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o400)
        with os.fdopen(fd, "w") as stream:
            json.dump(record, stream, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
    return path, record
