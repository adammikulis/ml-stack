"""A llama-server ml-stack did not start is reported, counted, and adopted only on request.

Adoption is off until ``ML_STACK_ADOPT_UNMANAGED`` or the ``adopt_unmanaged`` limit says
``auto`` or ``ask``. Even then the listener is identified before anything is sent to it. The
hostile listener below is a real process that answers like llama-server on loopback; the
checks that refuse it read the real process table.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from http.server import BaseHTTPRequestHandler
from pathlib import Path

import pytest

from ml_stack import gate, limits
from ml_stack.http import Server
from ml_stack.serve import admission, unmanaged
from ml_stack.serve.backend import ServerFailed, ServerSpec
from ml_stack.serve.leases import lease_file, recorded_servers
from ml_stack.serve.manager import ServerManager, stop_all_servers
from ml_stack.serve.ports import free_port
from ml_stack.serve.process import pid_exists
from ml_stack.testing.fakes import FakeBackend, FakeLlamaServer, Served

SRC = str(Path(__file__).resolve().parent.parent / "src")
LLAMA = {"pid": 0, "ip": "127.0.0.1", "uid": os.getuid(), "user": "me",
         "exe": "/opt/llama.cpp/build/bin/llama-server", "rss": 5 * 1024 ** 3}


def listener_of(**changed):
    return lambda port: {**LLAMA, **changed}


@pytest.fixture
def served():
    made: list[FakeLlamaServer] = []

    def start(model: str = "model.gguf", **held) -> FakeLlamaServer:
        made.append(FakeLlamaServer(Served(model=f"/models/{model}", context=4096, **held)))
        return made[-1]

    yield start
    for fake in made:
        fake.close()


def test_adoption_is_off_unless_it_is_asked_for(monkeypatch):
    monkeypatch.delenv(unmanaged.ENV, raising=False)
    assert unmanaged.mode() == "off"
    limits.changed(adopt_unmanaged="ask")
    assert unmanaged.mode() == "ask"
    monkeypatch.setenv(unmanaged.ENV, "auto")
    assert unmanaged.mode() == "auto", "the environment wins over the limits file"
    monkeypatch.setenv(unmanaged.ENV, "sometimes")
    assert unmanaged.mode() == "ask", "a word that is not a mode is ignored"


def test_a_llama_server_of_this_user_on_loopback_passes(served):
    fake = served()
    seen = unmanaged.examine(fake.port, find=listener_of(pid=4242))
    assert seen.ok and seen.pid == 4242 and seen.model.endswith("model.gguf")


@pytest.mark.parametrize(("changed", "why"), [
    ({"ip": "0.0.0.0"}, "not loopback"),
    ({"ip": "192.168.1.20"}, "not loopback"),
    ({"uid": os.getuid() + 1, "user": "someone-else"}, "owned by someone-else"),
    ({"exe": "/usr/bin/python3"}, "not a recognised server binary"),
    ({"exe": "/tmp/llama-server-lookalike"}, "not a recognised server binary"),
    ({"exe": ""}, "not a recognised server binary"),
])
def test_a_listener_that_fails_an_identity_check_is_refused_before_it_is_asked_anything(
        served, changed, why):
    fake = served()
    seen = unmanaged.examine(fake.port, find=listener_of(**changed))
    assert not seen.ok and why in seen.why
    assert fake.requests == [], "nothing was sent to an unidentified listener"


def test_a_port_nothing_answers_on_is_refused():
    assert not unmanaged.examine(free_port(), find=lambda port: None).ok


def test_a_listener_whose_props_are_not_a_llama_servers_is_refused():
    class Imitation(BaseHTTPRequestHandler):
        def log_message(self, *_):
            return

        def do_GET(self):
            body = json.dumps({"hello": "world"}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    httpd = Server(("127.0.0.1", 0), Imitation)
    import threading

    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        seen = unmanaged.examine(httpd.server_port, find=listener_of(pid=1))
        assert not seen.ok and "/props is not a llama-server's" in seen.why
    finally:
        httpd.shutdown()
        httpd.server_close()


HOSTILE = """
import json, sys
from http.server import BaseHTTPRequestHandler, HTTPServer
log = sys.argv[2]
class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_GET(self):
        open(log, "a").write(self.path + "\\n")
        body = json.dumps({"status": "ok", "model_path": "/m/llama.gguf", "total_slots": 1,
                           "default_generation_settings": {"n_ctx": 4096}}).encode()
        self.send_response(200); self.send_header("Content-Length", str(len(body)))
        self.end_headers(); self.wfile.write(body)
    do_POST = do_GET
s = HTTPServer(("127.0.0.1", int(sys.argv[1])), H)
print("up", flush=True)
s.serve_forever()
"""


def test_a_local_process_imitating_llama_server_is_refused_without_being_asked_anything(
        tmp_path):
    port, log = free_port(), tmp_path / "seen.log"
    proc = subprocess.Popen([sys.executable, "-c", HOSTILE, str(port), str(log)],
                            stdout=subprocess.PIPE, text=True)
    assert proc.stdout is not None
    assert proc.stdout.readline().strip() == "up"
    try:
        seen = unmanaged.examine(port)
        assert not seen.ok
        assert "not a recognised server binary" in seen.why
        assert not log.exists(), "the imitation was never sent a request"
    finally:
        proc.kill()
        proc.wait()


def test_a_llama_server_is_counted_against_the_memory_and_listed_as_unmanaged(
        monkeypatch, tmp_path):
    limits.changed(memory_bytes=10 * 1024 ** 3)
    row = {"pid": 77, "port": 51000, "model": "/m/x.gguf", "rss": 8 * 1024 ** 3,
           "defunct": False}
    monkeypatch.setattr(unmanaged, "every_server", lambda: [row])
    assert unmanaged.unmanaged_servers({}) == [row]
    assert unmanaged.unmanaged_servers({51000: {}}) == []
    weights = tmp_path / "m.gguf"
    with weights.open("wb") as handle:
        handle.truncate(2 * 1024 ** 3)
    verdict = admission.check(ServerSpec(model=weights, port=1), {}, budget=10 * 1024 ** 3,
                              unmanaged=[row])
    assert verdict.rating == "red" and "unmanaged pid 77" in verdict.said()


def lease_beside(tmp_path, served_port: int, *, model: str, roam: bool, state=None):
    manager = ServerManager(FakeBackend(), state_file=state or tmp_path / "servers.json")
    told: list[str] = []
    manager.say = told.append
    return manager, told, lambda: manager.lease(
        ServerSpec(model=model, port=served_port), roam=roam, preflight=False)


def test_by_default_a_busy_port_with_an_unmanaged_server_is_left_alone(served, tmp_path):
    fake = served()
    manager, _, lease = lease_beside(tmp_path, fake.port, model="model.gguf", roam=True)
    info = lease()
    assert info.port != fake.port and not info.adopted
    assert recorded_servers(tmp_path / "servers.json").get(fake.port) is None
    manager2, _, strict = lease_beside(tmp_path, fake.port, model="model.gguf", roam=False)
    with pytest.raises(ServerFailed, match="ml-stack did not start"):
        strict()


def sleeping() -> subprocess.Popen:
    return subprocess.Popen(["sleep", "120"])


def test_auto_adopts_a_verified_server_and_it_is_queued_counted_and_never_stopped(
        served, tmp_path, monkeypatch):
    fake, process = served(), sleeping()
    monkeypatch.setenv(unmanaged.ENV, "auto")
    monkeypatch.setattr(unmanaged, "listener", listener_of(pid=process.pid))
    state = lease_file()
    manager, told, lease = lease_beside(tmp_path, fake.port, model="model.gguf", roam=False,
                                        state=state)
    try:
        info = lease()
        assert info.adopted and info.port == fake.port and info.pid == process.pid
        entry = recorded_servers(state)[fake.port]
        assert entry["unmanaged"] is True and entry["owner_pid"] == process.pid
        assert any("adopted" in line and str(process.pid) in line for line in told)
        assert admission.charge(entry) == LLAMA["rss"]
        assert gate.pool_of(f"http://127.0.0.1:{fake.port}/v1/chat/completions") == "gpu"

        manager.release(info)
        manager.stop_all()
        assert process.poll() is None, "releasing a lease on an adopted server stops nothing"
        assert fake.port in recorded_servers(state)
    finally:
        process.kill()
        process.wait()


def test_adopted_servers_survive_every_way_ml_stack_stops_servers(served, tmp_path,
                                                                   monkeypatch):
    fake, process = served(), sleeping()
    monkeypatch.setenv(unmanaged.ENV, "auto")
    monkeypatch.setattr(unmanaged, "listener", listener_of(pid=process.pid))
    manager = ServerManager(FakeBackend(), state_file=lease_file())
    try:
        manager.lease(ServerSpec(model="model.gguf", port=fake.port), roam=False, preflight=False)
        from ml_stack.serve import ops
        from ml_stack.serve.reclaim import _recorded

        stop_all_servers()
        assert process.poll() is None
        with pytest.raises(ops.Refused, match="not started by ml-stack"):
            ops.down(fake.port)
        assert process.poll() is None
        assert fake.port not in _recorded(), "idle reclaim does not see it"
        assert ops.orphans() == []
    finally:
        process.kill()
        process.wait()


def test_an_adopted_server_that_dies_is_dropped_from_the_registry(served, tmp_path, monkeypatch):
    fake, process = served(), sleeping()
    monkeypatch.setenv(unmanaged.ENV, "auto")
    monkeypatch.setattr(unmanaged, "listener", listener_of(pid=process.pid))
    manager, _, lease = lease_beside(tmp_path, fake.port, model="model.gguf", roam=False)
    lease()
    process.kill()
    process.wait()
    deadline = time.monotonic() + 5
    while pid_exists(process.pid) and time.monotonic() < deadline:
        time.sleep(0.05)
    manager._save()
    assert fake.port not in recorded_servers(tmp_path / "servers.json")


def test_auto_still_refuses_a_listener_that_fails_the_checks(served, tmp_path, monkeypatch):
    fake = served()
    monkeypatch.setenv(unmanaged.ENV, "auto")
    monkeypatch.setattr(unmanaged, "listener", listener_of(pid=1, exe="/usr/bin/python3"))
    manager, told, lease = lease_beside(tmp_path, fake.port, model="model.gguf", roam=True)
    info = lease()
    assert info.port != fake.port
    assert any("not adopting" in line and "recognised server binary" in line for line in told)
    assert recorded_servers(tmp_path / "servers.json").get(fake.port) is None


def test_auto_does_not_adopt_a_server_serving_something_else(served, tmp_path, monkeypatch):
    fake = served(model="other.gguf")
    monkeypatch.setenv(unmanaged.ENV, "auto")
    monkeypatch.setattr(unmanaged, "listener", listener_of(pid=1))
    manager, told, lease = lease_beside(tmp_path, fake.port, model="model.gguf", roam=True)
    assert lease().port != fake.port
    assert any("not adopting" in line and "model" in line for line in told)


def test_ask_adopts_only_when_the_confirm_hook_says_yes(served, tmp_path, monkeypatch):
    fake, process = served(), sleeping()
    monkeypatch.setenv(unmanaged.ENV, "ask")
    monkeypatch.setattr(unmanaged, "listener", listener_of(pid=process.pid))
    asked: list[str] = []
    try:
        manager, _, lease = lease_beside(tmp_path, fake.port, model="model.gguf", roam=True)
        assert lease().port != fake.port, "no hook set: nothing is adopted"

        manager.confirm = lambda what: asked.append(what) or False
        assert lease().port != fake.port
        assert asked and str(process.pid) in asked[0] and str(fake.port) in asked[0]
        assert recorded_servers(tmp_path / "servers.json").get(fake.port) is None

        manager.confirm = lambda what: True
        assert lease().port == fake.port
        assert recorded_servers(tmp_path / "servers.json")[fake.port]["unmanaged"]
    finally:
        process.kill()
        process.wait()
