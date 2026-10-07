"""Windows launch preflight and the Linux model relay's isolation."""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

from ml_stack.fleet import wsl
from ml_stack.platform import launch
from ml_stack.sandbox import run
from ml_stack.sandbox.bubblewrap import Bubblewrap
from ml_stack.sandbox.policies import runtime_reads, system_env
from ml_stack.sandbox.policy import Limits, Net, Policy
from ml_stack.serve import socket_relay


def test_wsl_refuses_a_first_generation_distribution(monkeypatch):
    monkeypatch.setattr(wsl, "_read", lambda *args: json.dumps({
        "kernel": "microsoft", "python": [3, 12], "bwrap": "/usr/bin/bwrap", "home": "/home/test"}))
    with pytest.raises(wsl.WSLError, match="WSL 2"):
        wsl.prepare()


def test_wsl_refuses_missing_confinement(monkeypatch):
    monkeypatch.setattr(wsl, "_read", lambda *args: json.dumps({
        "kernel": "microsoft-standard-WSL2", "python": [3, 12], "bwrap": None}))
    with pytest.raises(wsl.WSLError, match="bubblewrap"):
        wsl.prepare()


def test_owned_process_preserves_hostile_arguments_without_a_shell(tmp_path):
    marker = tmp_path / "injected"
    payload = f"; touch {marker}; $(touch {marker}) `touch {marker}`\n"
    output = tmp_path / "arguments.json"
    script = "import json,pathlib,sys; pathlib.Path(sys.argv[1]).write_text(json.dumps(sys.argv[2:]))"
    child = launch([sys.executable, "-c", script, str(output), payload])
    assert child.wait(timeout=10) == 0
    assert json.loads(output.read_text()) == [payload]
    assert not marker.exists()


def test_wsl_read_preserves_hostile_text_and_unicode(monkeypatch, tmp_path):
    marker = tmp_path / "injected"
    payload = f"$(touch {marker}); `touch {marker}` \u2603\n"
    script = ("import json,sys; sys.stdout.reconfigure(encoding='utf-8'); "
              "print(json.dumps(sys.argv[1:], ensure_ascii=False))")
    monkeypatch.setattr(wsl, "command", lambda *args: [sys.executable, "-c", script, *args])
    assert json.loads(wsl._read("program", payload)) == ["program", payload]
    assert not marker.exists()


def _prepare_runtime(monkeypatch, tmp_path, *, installed="", returncode=0):
    wheel = tmp_path / "ml_stack-0.2-py3-none-any.whl"
    wheel.write_bytes(b"committed runtime one")
    calls, reads = [], []

    def read(*args):
        reads.append(args)
        if args[0] == "python3" and len(args) == 3:
            return json.dumps({"kernel": "microsoft-standard-WSL2", "python": [3, 12],
                               "bwrap": "/usr/bin/bwrap", "home": "/home/test"})
        if args[0] == "python3":
            return "1"
        if args[0] == "wslpath":
            return "/tmp/wheel path; $(touch injected) `touch injected`.whl"
        if len(args) > 2 and args[2] == wsl._CACHE_RUNTIME:
            return "/home/test/.local/share/ml-stack/runtime/ml-stack-wheels/" + "a" * 40 + "/" + wheel.name
        if args[0].endswith("/bin/python") and "p.read_text" in args[2]:
            return installed
        return ""

    def install(argv, **kwargs):
        calls.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, returncode)

    monkeypatch.setattr(wsl.runtime_wheel, "source_checkout", lambda: None)
    monkeypatch.setattr(wsl.runtime_wheel, "current_wheel", lambda: wheel)
    monkeypatch.setattr(wsl, "__file__", str(tmp_path / "site-packages/ml_stack/fleet/wsl.py"))
    monkeypatch.setattr(wsl, "_read", read)
    monkeypatch.setattr(wsl, "command", lambda *args: ["wsl.exe", "--exec", *args])
    monkeypatch.setattr(wsl.subprocess, "run", install)
    return wheel, calls, reads


def test_wsl_installer_uses_cached_wheel_without_source_checkout(monkeypatch, tmp_path):
    _, calls, _ = _prepare_runtime(monkeypatch, tmp_path)
    assert wsl.prepare().endswith("/bin/python")
    assert len(calls) == 3
    argv, kwargs = calls[1]
    assert argv[-1].startswith("/tmp/wheel path; $(touch injected) `touch injected`.whl[")
    assert "--upgrade" in argv
    assert "-e" not in argv
    assert not kwargs.get("shell")
    assert "--force-reinstall" in calls[0][0]
    assert "--no-deps" in calls[0][0]
    assert "--no-index" in calls[0][0]
    assert calls[0][0][-1].startswith("/tmp/wheel path;")
    assert "--force-reinstall" in calls[2][0]
    assert "--no-deps" in calls[2][0]
    assert "--no-index" in calls[2][0]
    assert "/ml-stack-wheels/" in calls[2][0][-1]


def test_wsl_runtime_reuses_matching_install_and_refreshes_changed_revision(monkeypatch, tmp_path):
    wheel, calls, reads = _prepare_runtime(monkeypatch, tmp_path)
    wsl.prepare()
    marker = reads[-1][-1]
    _, calls, reads = _prepare_runtime(monkeypatch, tmp_path, installed=marker)
    wsl.prepare()
    assert not calls
    wheel.write_bytes(b"committed runtime two")
    wsl.prepare()
    assert len(calls) == 3
    assert reads[-1][-1] != marker


@pytest.mark.parametrize("failure", [1, 2, 3])
def test_wsl_failed_install_does_not_record_marker(monkeypatch, tmp_path, failure):
    _, calls, reads = _prepare_runtime(monkeypatch, tmp_path)

    def install(argv, **kwargs):
        calls.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, int(len(calls) == failure))

    monkeypatch.setattr(wsl.subprocess, "run", install)
    with pytest.raises(wsl.WSLError, match="installation did not complete"):
        wsl.prepare()
    assert not any("write_text" in arg for args in reads for arg in args)


@pytest.mark.parametrize("has_source", [False, True])
def test_wsl_runtime_cache_preserves_revision_and_translates_source(tmp_path, has_source):
    import zipfile

    from ml_stack.fleet import runtime_wheel

    wheel = tmp_path / "ml_stack-0.2-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("ml_stack-0.2.dist-info/RECORD", "")
    original_source = tmp_path / "windows-source"
    translated_source = tmp_path / "linux-source"
    translated_source.mkdir()
    (translated_source / ".git").mkdir()
    commit = "a" * 40
    runtime_wheel.stamp(wheel, commit, original_source)
    original = wheel.read_bytes()
    prefix = tmp_path / "runtime"
    environment = dict(os.environ, PYTHONPATH=str(Path(wsl.__file__).resolve().parents[2]))
    child = launch([sys.executable, "-c", wsl._CACHE_RUNTIME, str(wheel),
                    str(translated_source) if has_source else "", str(prefix)],
                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=environment)
    output, errors = child.communicate(timeout=30)
    assert child.returncode == 0, errors.decode()
    cached = Path(output.decode().strip())
    assert cached == prefix / "ml-stack-wheels" / commit / wheel.name
    assert runtime_wheel.wheel_commit(cached) == commit
    with zipfile.ZipFile(cached) as archive:
        assert archive.read("ml_stack/fleet/source-checkout").decode().strip() == str((translated_source if has_source else original_source).resolve())
    assert wheel.read_bytes() == original


def test_wsl_failed_cache_does_not_record_marker(monkeypatch, tmp_path):
    _, _, reads = _prepare_runtime(monkeypatch, tmp_path)
    read = wsl._read

    def failing_read(*args):
        if len(args) > 2 and args[2] == wsl._CACHE_RUNTIME:
            raise wsl.WSLError("wheel cache failed")
        return read(*args)

    monkeypatch.setattr(wsl, "_read", failing_read)
    with pytest.raises(wsl.WSLError, match="wheel cache failed"):
        wsl.prepare()
    assert not any("write_text" in arg for args in reads for arg in args)


def test_wsl_missing_cached_wheel_requests_windows_update(monkeypatch, tmp_path):
    _prepare_runtime(monkeypatch, tmp_path)
    monkeypatch.setattr(wsl.runtime_wheel, "current_wheel", lambda: None)
    with pytest.raises(wsl.WSLError, match="Update the Windows"):
        wsl.prepare()


def test_wsl_start_preserves_service_setup_and_closes_owned_resources(monkeypatch):
    from types import SimpleNamespace

    events = []
    bridge = SimpleNamespace(config="bridge-config", close=lambda: events.append("bridge-close"))
    pipe = SimpleNamespace(close=lambda: events.append("pipe-close"))

    def wait(**kwargs):
        events.append(("wait", kwargs, "pipe-close" in events))
        return 0

    child = SimpleNamespace(stdin=pipe, wait=wait)
    launches = []

    def owned_launch(argv, **kwargs):
        launches.append((argv, kwargs))
        return child

    def unexpected_process(*args, **kwargs):
        pytest.fail("WSL launch managed another process")

    monkeypatch.setattr(wsl, "_read", lambda *args: "/mnt/c/runtime/python.exe")
    monkeypatch.setattr(wsl, "_bridge", lambda *args: bridge)
    monkeypatch.setattr(wsl, "command", lambda *args: ["wsl.exe", "--exec", *args])
    monkeypatch.setattr(wsl, "launch", owned_launch)
    monkeypatch.setattr(wsl.subprocess, "run", unexpected_process)
    monkeypatch.delenv("ML_STACK_HOME", raising=False)
    monkeypatch.delenv("ML_STACK_CACHE", raising=False)
    arguments = ["--port", "8771", "argument; $(touch injected)"]
    assert wsl.start(arguments, executable="/home/test/runtime/bin/python") == 0
    assert len(launches) == 1
    argv, kwargs = launches[0]
    assert argv[-len(arguments):] == arguments
    assert wsl.wsl_network.ENV + "=bridge-config" in argv
    assert kwargs == {"stdin": subprocess.PIPE}
    assert events == [("wait", {}, False), "pipe-close",
                      ("wait", {"timeout": 10}, True), "bridge-close"]


def test_model_namespace_allows_granted_reads_and_denies_host_loopback(tmp_path):
    if not sys.platform.startswith("linux") or not shutil.which("bwrap"):
        pytest.skip("requires Linux bubblewrap")
    allowed = tmp_path / "allowed"
    outside = tmp_path / "outside"
    allowed.mkdir()
    (allowed / "readable").write_text("granted")
    outside.write_text("private")
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        script = """import pathlib,socket,sys
assert pathlib.Path(sys.argv[1]).read_text() == 'granted'
try: pathlib.Path(sys.argv[2]).read_text()
except OSError: pass
else: raise AssertionError('outside read allowed')
sock=socket.socket(); sock.settimeout(.2)
try: sock.connect(('127.0.0.1', int(sys.argv[3])))
except OSError: pass
else: raise AssertionError('host loopback allowed')
print('isolated')
"""
        policy = Policy("test", read=(*runtime_reads(), str(allowed)),
                        exec=(os.path.realpath(sys.executable),), net=Net.deny(),
                        env=system_env(), limits=Limits(wall_seconds=5))
        result = run([os.path.realpath(sys.executable), "-c", script, str(allowed / "readable"),
                      str(outside), str(listener.getsockname()[1])], policy,
                     cwd=str(allowed), via=Bubblewrap(), diagnose="never")
        assert result.ok, result.stderr
        assert result.stdout.strip() == "isolated"


def test_relay_preserves_stream_and_releases_port_on_child_exit(tmp_path):
    if not sys.platform.startswith("linux"):
        pytest.skip("requires Unix sockets")
    target = str(tmp_path / "model.sock")
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    script = """import socket,sys
with socket.socket(socket.AF_UNIX) as server:
 server.bind(sys.argv[1]); server.listen()
 client,_=server.accept()
 with client:
  data=b''
  while chunk:=client.recv(65536): data+=chunk
  client.sendall(data[::-1])
"""
    diagnostics = tmp_path / "relay.log"
    with diagnostics.open("wb") as log:
        process = subprocess.Popen(socket_relay.arguments([sys.executable, "-c", script, target], port, target),
                                   stdout=log, stderr=subprocess.STDOUT)
    try:
        deadline = time.monotonic() + 20
        while not Path(target).exists() and time.monotonic() < deadline:
            assert process.poll() is None, diagnostics.read_text()
            time.sleep(.02)
        assert Path(target).exists(), "relay child did not create its Unix socket: " + diagnostics.read_text()
        payload = (b"GET /../../private HTTP/1.1\r\nOrigin: https://hostile.invalid\r\n"
                   b"\r\n$(touch injected)\x00\xff") * 2000
        try:
            client = socket.create_connection(("127.0.0.1", port), timeout=3)
        except ConnectionRefusedError as exc:
            raise AssertionError(f"relay exited {process.poll()}: {diagnostics.read_text()}") from exc
        with client:
            client.sendall(payload)
            client.shutdown(socket.SHUT_WR)
            received = bytearray()
            while chunk := client.recv(65536):
                received.extend(chunk)
        assert received == payload[::-1]
        assert process.wait(timeout=5) == 0
        with socket.socket() as released:
            released.bind(("127.0.0.1", port))
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=5)


def test_wsl_gpu_is_visible_inside_the_network_denied_namespace(tmp_path):
    binary = "/usr/lib/wsl/lib/nvidia-smi"
    if not sys.platform.startswith("linux") or not Path(binary).exists():
        pytest.skip("requires WSL CUDA")
    policy = Policy("gpu-test", read=("/usr/lib/wsl/lib",), exec=(binary,), gpu=True,
                    net=Net.deny(), env=system_env(LD_LIBRARY_PATH="/usr/lib/wsl/lib"),
                    limits=Limits(wall_seconds=10))
    result = run([binary, "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                 policy, cwd=str(tmp_path), diagnose="never")
    assert result.ok, result.stderr
    assert result.stdout.strip()
    assert all(int(value) > 0 for value in result.stdout.splitlines())
