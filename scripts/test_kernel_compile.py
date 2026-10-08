"""Bounded native compilation diagnostics for exact read components."""
from __future__ import annotations

import hashlib
import json
import os
import selectors
import subprocess
import time
from dataclasses import replace

from ml_stack.platform import start_process, terminate_process_group
from ml_stack.sandbox.path_language import expressions, paths_checked
from ml_stack.sandbox.policy import Net
from ml_stack.sandbox.seatbelt import Seatbelt, profile, quote

NODE = "tests/test_test_kernel_isolation.py::test_native_read_components_compile_independently"
LITERAL_NODE = "tests/test_test_kernel_isolation.py::test_explicit_literal_holder_inventory_runs_only_normal"


def literal_language(paths):
    """Return native literal rules for the checked complete path union."""
    unique = paths_checked(paths)
    groups = [unique[offset:offset + 128] for offset in range(0, len(unique), 128)]
    recovered = [path for group in groups for path in group]
    if recovered != unique:
        raise RuntimeError("literal profile: terminal union differs from input")
    lines = ["(allow file-read* " + " ".join("(literal " + quote(path) + ")" for path in group) + ")" for group in groups]
    text = "\n".join(lines)
    if len(text.encode()) > 32 * 1024 * 1024:
        raise RuntimeError("literal profile: output exceeds bound")
    digest = hashlib.sha256(json.dumps(unique, ensure_ascii=False).encode()).hexdigest()
    return lines, {"representation": "explicit-literal-experiment", "paths": len(unique),
                   "input_sha256": digest, "terminals_sha256": hashlib.sha256(json.dumps(recovered, ensure_ascii=False).encode()).hexdigest(), "rules": len(lines),
                   "output_bytes": len(text.encode()), "expressions_sha256": hashlib.sha256(text.encode()).hexdigest()}


def replace_exact_file_language(path, files):
    """Replace only the verified exact regex rules with the same literal union."""
    expected = ["(allow file-read* " + value + ")" for value in expressions(files, quote)]
    text = path.read_text()
    lines = text.splitlines()
    found = [line for line in lines if line.startswith("(allow file-read* (regex ")]
    if found != expected:
        raise RuntimeError("literal profile: generated exact file language changed")
    literals, receipt = literal_language(files)
    result = [line for line in lines if not line.startswith("(allow file-read* (regex ")]
    path.write_text("\n".join([*result, *literals]) + "\n")
    return receipt


def execute(argv, environment, deadline):
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
                    raise TimeoutError("compiler diagnostic: shared native deadline exceeded")
                for event, _ in events:
                    block = os.read(event.fileobj.fileno(), 4096)
                    if not block:
                        ready.unregister(event.fileobj)
                    buffers[event.fileobj].extend(block)
                    if sum(map(len, buffers.values())) > 8192:
                        raise RuntimeError("compiler diagnostic: output exceeds bound")
        status = process.wait(timeout=max(.001, deadline - time.monotonic() - 1))
        return {"exit": status, "stdout": buffers[process.stdout].decode(errors="replace"),
                "stderr": buffers[process.stderr].decode(errors="replace")}
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


def observe(confined, base):
    """Compile isolated read components with a fixed inert native executable."""
    deadline = time.monotonic() + 5
    rendered = profile(base, "/usr/bin/true", "read-component-inventory")
    files = sorted(set(base.read_files))
    file_digest = hashlib.sha256(json.dumps(files, ensure_ascii=False).encode()).hexdigest()
    components = {
        "literal_files": ["(allow file-read* " + " ".join("(literal " + quote(path) + ")" for path in files[offset:offset + 128]) + ")"
                          for offset in range(0, len(files), 128)],
        "files": ["(allow file-read* " + value + ")" for value in expressions(base.read_files, quote)],
        "metadata": [line for line in rendered.splitlines() if line.startswith("(allow file-read-metadata ")],
        "directories": [line for line in rendered.splitlines() if line.startswith("(allow file-read-data ")],
    }
    baseline = replace(base, name="read-component-diagnostic", read=(), write=(), cache=None,
                       read_files=(), read_dirs=(), read_metadata=(), exec=("/usr/bin/true",),
                       net=Net.deny(), unix_sockets=(), unix_namespaces=(), gpu=False,
                       env={"PATH": os.defpath, "LC_ALL": "C"}).validated()
    records = {}
    report = confined.control / "native-read-components.json"
    try:
        for name, lines in components.items():
            path = confined.control / ("read-component-" + name + ".sb")
            wrapped = Seatbelt().wrap(["/usr/bin/true"], baseline, profile_path=path)
            text = "\n".join(lines) + "\n"
            if len(text.encode()) > 32 * 1024 * 1024:
                raise RuntimeError("compiler diagnostic: component exceeds bound")
            with path.open("a") as stream:
                stream.write(text)
            path.chmod(0o400)
            confined.recheck_images()
            records[name] = {"rules": len(lines), "bytes": len(text.encode()),
                             "sha256": hashlib.sha256(text.encode()).hexdigest(),
                             "profile": hashlib.sha256(path.read_bytes()).hexdigest(), "status": "pending"}
            if name in ("files", "literal_files"):
                records[name].update(paths=len(files), input_sha256=file_digest)
            try:
                result = execute(wrapped.argv, dict(baseline.env), deadline)
            except Exception as error:
                records[name].update(status="unknown or failed", error=type(error).__name__)
                raise
            records[name].update(result, status="complete")
    finally:
        report.write_text(json.dumps(records, sort_keys=True))
        report.chmod(0o400)
        if confined.artifacts is not None:
            confined.artifacts.proof["native_read_components"] = {
                "records": records, "sha256": hashlib.sha256(report.read_bytes()).hexdigest()}
    if any(value["exit"] != 0 for value in records.values()):
        raise RuntimeError("compiler diagnostic: native component compilation failed")
    return report
