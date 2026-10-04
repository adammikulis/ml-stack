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
    script = "import json,sys; print(json.dumps(sys.argv[1:], ensure_ascii=False))"
    monkeypatch.setattr(wsl, "command", lambda *args: [sys.executable, "-c", script, *args])
    assert json.loads(wsl._read("program", payload)) == ["program", payload]
    assert not marker.exists()


def test_wsl_installer_passes_a_hostile_source_path_as_one_argument(monkeypatch):
    source = "/tmp/source path; $(touch injected) `touch injected`"
    calls = []

    def read(*args):
        if args[0] == "python3":
            return json.dumps({"kernel": "microsoft-standard-WSL2", "python": [3, 12],
                               "bwrap": "/usr/bin/bwrap", "home": "/home/test"})
        if args[0] == "wslpath":
            return source
        return ""

    def install(argv, **kwargs):
        calls.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(wsl, "_read", read)
    monkeypatch.setattr(wsl, "command", lambda *args: ["wsl.exe", "--exec", *args])
    monkeypatch.setattr(wsl.subprocess, "run", install)
    assert wsl.prepare().endswith("/bin/python")
    assert len(calls) == 1
    argv, kwargs = calls[0]
    assert argv[argv.index("-e") + 1].startswith(source + "[")
    assert not kwargs.get("shell")


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
