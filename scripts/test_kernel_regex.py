"""Tiny native Seatbelt exact-language diagnostic."""
from __future__ import annotations

import hashlib
import json
import os
import selectors
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

from poolhouse.platform import start_process, terminate_process_group
from poolhouse.sandbox.path_language import expressions
from poolhouse.sandbox.policy import Net, checked_path
from poolhouse.sandbox.seatbelt import Seatbelt, quote

NODE = "tests/test_test_kernel_isolation.py::test_native_regex_matches_only_complete_listed_paths"
CHILD = '''import json,os,sys
listed,denied=json.loads(sys.argv[1])
for path in listed:
    if os.path.isdir(path):
        os.listdir(path)
    else:
        with open(path,"rb") as stream:
            assert stream.read(16)==b"owned"
for path in denied:
    try:
        with open(path,"rb") as stream:
            stream.read(1)
    except PermissionError:
        continue
    raise AssertionError("unlisted path was readable")
print(json.dumps({"listed":len(listed),"denied":len(denied)}))
'''


def language(paths: list[str]) -> str:
    """Return anchored native-quoted alternatives for checked complete paths."""
    if not 1 <= len(paths) <= 128 or len(set(paths)) != len(paths):
        raise RuntimeError("regex diagnostic: path count or uniqueness refused")
    for path in paths:
        checked_path(path, what="regex diagnostic listed path")
        if len(os.fsencode(path)) > 4096:
            raise RuntimeError("regex diagnostic: path bytes exceed bound")
    rendered = expressions(paths, quote)
    if len(rendered) != 1:
        raise RuntimeError("regex diagnostic: expected one bounded group")
    value = rendered[0]
    if len(value.encode()) > 65536:
        raise RuntimeError("regex diagnostic: representation exceeds bound")
    return value


def cases(control: Path) -> tuple[list[str], list[str]]:
    root = control / "regex-cases"
    root.mkdir(mode=0o700)
    names = ["a", "ab", "meta$[]()+*?{}^.|", 'quote"file', "back\\slash", "café", "漢字", "prefix"]
    listed = []
    denied = []
    for name in names:
        path = root / name
        path.write_bytes(b"owned")
        path.chmod(0o400)
        listed.append(str(path))
        for suffix in ("x", "\n"):
            other = root / (name + suffix)
            other.write_bytes(b"owned")
            other.chmod(0o400)
            denied.append(str(other))
    directory = root / "directory"
    directory.mkdir(mode=0o700)
    listed.append(str(directory))
    child = directory / "unlisted"
    child.write_bytes(b"owned")
    child.chmod(0o400)
    denied.append(str(child))
    return listed, denied


def execute(argv: list[str], environment: dict[str, str], deadline: float) -> dict:
    process = start_process(argv, env=environment, close_fds=True, stdin=subprocess.DEVNULL,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    buffers = {process.stdout: bytearray(), process.stderr: bytearray()}
    try:
        with selectors.DefaultSelector() as ready:
            for stream in buffers:
                ready.register(stream, selectors.EVENT_READ)
            while ready.get_map():
                remaining = deadline - time.monotonic() - 1
                if remaining <= 0 or not (events := ready.select(remaining)):
                    raise TimeoutError("regex diagnostic: native deadline exceeded")
                for event, _ in events:
                    block = os.read(event.fileobj.fileno(), 4096)
                    if not block:
                        ready.unregister(event.fileobj)
                    buffers[event.fileobj].extend(block)
                    if sum(map(len, buffers.values())) > 8192:
                        raise RuntimeError("regex diagnostic: child output exceeds bound")
        status = process.wait(timeout=max(.001, deadline - time.monotonic() - 1))
        if status or buffers[process.stderr]:
            raise RuntimeError(f"regex diagnostic: native exit {status}: "
                               + buffers[process.stderr].decode("utf-8", errors="replace"))
        return json.loads(buffers[process.stdout])
    finally:
        process.stdout.close()
        process.stderr.close()
        if process.poll() is None:
            try:
                terminate_process_group(process, force=True)
            except ProcessLookupError:
                pass
            finally:
                process.wait(timeout=max(.001, deadline - time.monotonic()))


def observe(confined, base) -> tuple[dict, Path]:
    """Compare native literal and native regex behavior on inert owned cases."""
    deadline = time.monotonic() + 5
    listed, denied = cases(confined.control)
    expression = language(listed)
    configs = [str(Path(prefix).resolve() / "pyvenv.cfg") for prefix in (sys.prefix, sys.base_prefix)
               if (Path(prefix).resolve() / "pyvenv.cfg").is_file()]
    policy = replace(base, name="native-regex-diagnostic", write=(), cache=None,
                     read_files=(*map(str, confined.images), *configs),
                     read_dirs=(), read_metadata=(*map(str, confined.loader_metadata), str(confined.control),
                                                 str(Path(listed[0]).parent), listed[-1]),
                     exec=(str(Path(sys.executable).resolve()),), net=Net.deny(), unix_sockets=(), unix_namespaces=(),
                     env={"PATH": os.defpath, "LC_ALL": "C"}).validated()
    payload = json.dumps([listed, denied])
    expected = {"listed": len(listed), "denied": len(denied)}
    records = {}
    for kind in ("literal", "regex"):
        profile = confined.control / ("tiny-" + kind + ".sb")
        wrapped = Seatbelt().wrap([sys.executable, "-I", "-S", "-c", CHILD, payload], policy, profile_path=profile)
        grant = expression if kind == "regex" else " ".join("(literal " + quote(path) + ")" for path in listed)
        with profile.open("a") as stream:
            stream.write("(allow file-read* " + grant + ")\n")
        profile.chmod(0o400)
        confined.recheck_images()
        result = execute(wrapped.argv, dict(policy.env), deadline)
        if result != expected:
            raise RuntimeError("regex diagnostic: unexpected native result")
        records[kind] = {**result, "exit": 0, "profile": hashlib.sha256(profile.read_bytes()).hexdigest()}
    report = confined.control / "native-regex.json"
    report.write_text(json.dumps(records, sort_keys=True))
    report.chmod(0o400)
    return {**records, "sha256": hashlib.sha256(report.read_bytes()).hexdigest()}, report
