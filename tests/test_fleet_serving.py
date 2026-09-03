"""Reaching a model server on another machine, through that machine's daemon."""

from __future__ import annotations

import json
import socket
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from ml_stack.fleet.daemon import JobRunner, load_or_create_token, make_handler
from ml_stack.fleet.serving import Endpoint, Serving, answers


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class FakeModelServer:
    """A real socket that streams like llama.cpp does, with real gaps between tokens."""

    def __init__(self, tokens: int = 8, gap: float = 0.25):
        self.disconnected = threading.Event()
        outer = self

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.0"

            def log_message(self, *a):
                pass

            def do_GET(self):
                if self.path in ("/health", "/v1/models"):
                    body = b'{"status":"ok"}'
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                else:
                    self.send_response(404)
                    self.end_headers()

            def do_POST(self):
                n = int(self.headers.get("Content-Length", "0"))
                req = json.loads(self.rfile.read(n) or b"{}")
                if not req.get("stream"):
                    body = json.dumps({"choices": [{"message": {"content": "hello"},
                                                    "finish_reason": "stop"}]}).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                try:
                    for i in range(tokens):
                        self.wfile.write(
                            f"data: {json.dumps({'i': i})}\n\n".encode())
                        self.wfile.flush()
                        time.sleep(gap)
                    self.wfile.write(b"data: [DONE]\n\n")
                except (BrokenPipeError, ConnectionResetError):
                    outer.disconnected.set()

        self.port = free_port()
        self.httpd = ThreadingHTTPServer(("127.0.0.1", self.port), H)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


class Daemon:
    def __init__(self, tmp_path, serving):
        root = tmp_path / "traind"
        files = root / "files"
        files.mkdir(parents=True)
        self.token = load_or_create_token(root)
        self.runner = JobRunner(root, files)
        self.port = free_port()
        self.httpd = ThreadingHTTPServer(
            ("127.0.0.1", self.port),
            make_handler(self.runner, files, self.token, "box", serving=serving))
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def post(self, path, body=None, token=None, stream=False):
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}{path}",
            data=json.dumps(body or {}).encode(), method="POST")
        req.add_header("Content-Type", "application/json")
        if token is not False:
            req.add_header("Authorization", f"Bearer {token or self.token}")
        return urllib.request.urlopen(req, timeout=30)

    def close(self):
        self.runner.shutdown()
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def model():
    m = FakeModelServer()
    try:
        yield m
    finally:
        m.close()


@pytest.fixture
def wired(tmp_path, model):
    serving = Serving(tmp_path / "serving.json")
    serving.register(model.port, ["qwen3-4b.gguf"], slots=4)
    d = Daemon(tmp_path, serving)
    try:
        yield d, serving, model
    finally:
        d.close()


class TestRegistry:
    def test_a_registered_server_that_died_is_not(self, tmp_path, model):
        """Registration is a claim. A beacon advertising a model nobody can reach
        sends work to a dead port."""
        s = Serving(tmp_path / "s.json")
        s.register(model.port, ["a.gguf"])
        model.close()
        assert s.live() == []

    @pytest.mark.slow

    def test_a_hung_server_does_not_stall_the_beacon(self, tmp_path):
        """A port that accepts and then says nothing is the slow case: every health
        path waits the full timeout. The beacon rebuilds this every 10s."""
        import socket as sk
        import time as clock

        listener = sk.socket()
        listener.bind(("127.0.0.1", 0))
        listener.listen(8)
        s = Serving(tmp_path / "s.json")
        s.register(listener.getsockname()[1], ["hung.gguf"])
        try:
            began = clock.monotonic()
            assert s.live(force=True) == []
            spent = clock.monotonic() - began
        finally:
            listener.close()
        assert spent < 5.0, f"one dead server cost {spent:.1f}s of a 10s beacon"

    def test_the_answer_is_reused_briefly_rather_than_reprobed(self, tmp_path, model):
        s = Serving(tmp_path / "s.json")
        s.register(model.port, ["a.gguf"])
        assert [x.port for x in s.live()] == [model.port]
        model.close()
        # Still cached: the beacon asked moments ago.
        assert [x.port for x in s.live()] == [model.port]
        assert s.live(force=True) == []

    def test_registering_clears_what_was_cached_and_a_live_server_is_live(self, tmp_path,
                                                                           model):
        s = Serving(tmp_path / "s.json")
        assert s.live() == []
        s.register(model.port, ["a.gguf"])
        assert [x.port for x in s.live()] == [model.port]

    def test_registering_the_same_port_twice_does_not_duplicate_it(self, tmp_path, model):
        s = Serving(tmp_path / "s.json")
        s.register(model.port, ["a.gguf"])
        s.register(model.port, ["b.gguf"])
        assert len(s.all()) == 1
        assert s.all()[0].models == ["b.gguf"]

    def test_a_model_can_be_found_by_part_of_its_name(self, tmp_path, model):
        s = Serving(tmp_path / "s.json")
        s.register(model.port, ["Qwen3-4B-Instruct.gguf"])
        assert s.port_for("qwen3") == model.port
        assert s.port_for("llama") is None

    def test_a_corrupt_registry_reports_nothing_rather_than_raising(self, tmp_path):
        path = tmp_path / "s.json"
        path.write_text("{ not json")
        assert Serving(path).all() == []


class TestProxy:
    def test_it_needs_the_token(self, wired):
        daemon, _, _ = wired
        with pytest.raises(urllib.error.HTTPError) as exc:
            daemon.post("/infer/v1/chat/completions", {}, token=False)
        assert exc.value.code == 401

    def test_a_plain_completion_comes_back(self, wired):
        daemon, _, _ = wired
        with daemon.post("/infer/v1/chat/completions",
                         {"messages": [{"role": "user", "content": "hi"}]}) as r:
            body = json.loads(r.read())
        assert body["choices"][0]["message"]["content"] == "hello"

    @pytest.mark.slow

    def test_a_streamed_completion_arrives_as_it_is_generated(self, wired):
        """read(n) blocks until it has n bytes and a token is tens of bytes, so a
        proxy that uses it delivers the whole completion at once."""
        daemon, _, _ = wired
        started = time.time()
        first = None
        with daemon.post("/infer/v1/chat/completions", {"stream": True}) as r:
            while True:
                block = r.read(64)
                if not block:
                    break
                if first is None:
                    first = time.time() - started
        total = time.time() - started

        assert first is not None
        # The whole completion takes tokens*gap. A buffered proxy delivers everything
        # at the end, so the first chunk would land at ~total.
        assert first < total - 0.5, (
            f"first chunk at {first:.2f}s of {total:.2f}s -- it was buffered")

    def test_a_machine_with_no_model_says_so(self, tmp_path):
        serving = Serving(tmp_path / "empty.json")
        daemon = Daemon(tmp_path, serving)
        try:
            with pytest.raises(urllib.error.HTTPError) as exc:
                daemon.post("/infer/v1/chat/completions", {})
            assert exc.value.code == 503
            assert "model server" in json.loads(exc.value.read())["error"]
        finally:
            daemon.close()

    def test_the_model_server_is_never_reachable_from_the_network(self, wired):
        """Only the daemon's port is exposed. llama.cpp has no authentication, so a
        model server on the LAN is an open inference endpoint."""
        _, _, model = wired
        from ml_stack.fleet.discovery import primary_ip

        with socket.socket() as s:
            s.settimeout(2)
            assert s.connect_ex((primary_ip(), model.port)) != 0

    def test_it_forwards_whatever_path_it_was_given(self, wired):
        daemon, _, _ = wired
        for path in ("/infer/v1/chat/completions", "/infer/completion",
                     "/infer/embedding"):
            with daemon.post(path, {}) as r:
                assert r.status == 200, path


class TestEndpoint:
    def test_it_hands_a_client_exactly_what_it_needs(self):
        """If this needs a change in Client, the design is wrong."""
        import inspect

        from ml_stack.client import Client

        kwargs = Endpoint(peer="gpubox", base_url="http://box:8770",
                          token="abc").client_kwargs()
        assert kwargs == {"base_url": "http://box:8770/infer", "api_key": "abc"}
        accepted = inspect.signature(Client.__init__).parameters
        assert set(kwargs) <= set(accepted)

    def test_an_unmodified_client_talks_through_the_proxy(self, wired):
        from ml_stack.client import Client

        daemon, _, _ = wired
        endpoint = Endpoint(peer="box", base_url=f"http://127.0.0.1:{daemon.port}",
                            token=daemon.token)
        reply = Client(**endpoint.client_kwargs()).chat(
            [{"role": "user", "content": "hi"}])
        assert reply.content == "hello"


def test_a_probe_reports_a_port_nothing_is_on(tmp_path):
    assert not answers(free_port(), timeout=1.0)


FAKE_SERVER = '''
import json, sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

argv = sys.argv[1:]
Path(sys.argv[0]).with_name("argv.json").write_text(json.dumps(argv))
port = int(argv[argv.index("--port") + 1])
name = Path(argv[argv.index("-m") + 1]).name


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def do_GET(self):
        body = json.dumps({"status": "ok", "data": [{"id": name}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
'''


@pytest.fixture
def fake_llama_server(tmp_path, monkeypatch):
    """A llama-server that really launches, binds the port it was given and answers."""
    import sys

    from ml_stack.serve import backend as backend_module

    monkeypatch.setattr(backend_module, "LOG_DIR", tmp_path / "logs")
    script = tmp_path / "server.py"
    script.write_text(FAKE_SERVER)
    binary = tmp_path / "llama-server"
    binary.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n')
    binary.chmod(0o755)
    return binary


@pytest.fixture
def manager(tmp_path, fake_llama_server):
    from ml_stack.serve import LlamaServerBackend, ServerManager

    return ServerManager(LlamaServerBackend(binary=fake_llama_server),
                         state_file=tmp_path / "servers.json")


@pytest.fixture
def gguf(tmp_path):
    path = tmp_path / "Qwen3-4B-Q4_K_M.gguf"
    path.write_bytes(b"GGUF" + b"\x00" * 64)
    return path


class TestStartingAModelWithoutTheInterface:
    def test_it_leases_a_server_that_answers(self, tmp_path, manager, gguf):
        from ml_stack.fleet.serving import start_model, stop_model

        started = start_model(tmp_path, gguf, manager=manager)
        try:
            assert answers(started.port, timeout=5.0)
            assert started.lease.pid
        finally:
            stop_model(started)

    def test_stopping_it_leaves_nothing_on_the_port(self, tmp_path, manager, gguf):
        from ml_stack.fleet.serving import start_model, stop_model

        started = start_model(tmp_path, gguf, manager=manager)
        stop_model(started)
        assert not answers(started.port, timeout=1.0)

    def test_the_registry_gains_and_loses_the_port(self, tmp_path, manager, gguf):
        from ml_stack.fleet.serving import start_model, stop_model

        registry = Serving(tmp_path / "serving.json")
        started = start_model(tmp_path, gguf, manager=manager, serving=registry)
        try:
            assert started.served is not None
            assert [s.port for s in registry.all()] == [started.port]
            assert registry.all()[0].models == ["Qwen3-4B-Q4_K_M.gguf"]
        finally:
            stop_model(started, serving=registry)
        assert registry.all() == []

    def test_the_name_it_registers_can_be_given(self, tmp_path, manager, gguf):
        from ml_stack.fleet.serving import start_model, stop_model

        registry = Serving(tmp_path / "serving.json")
        started = start_model(tmp_path, gguf, name="something else.gguf",
                              manager=manager, serving=registry)
        try:
            assert registry.all()[0].models == ["something else.gguf"]
        finally:
            stop_model(started, serving=registry)

    def test_the_context_length_reaches_the_server(self, tmp_path, manager, gguf):
        from ml_stack.fleet.serving import start_model, stop_model

        started = start_model(tmp_path, gguf, context=2048, manager=manager)
        stop_model(started)
        argv = json.loads((tmp_path / "argv.json").read_text())
        assert argv[argv.index("-c") + 1] == "2048"

    def test_a_draft_beside_the_model_is_served_with_it(self, tmp_path, manager, gguf):
        """A machine that fetched the draft and does not pass it paid for nothing."""
        from ml_stack.fleet.models import DRAFT_MARK
        from ml_stack.fleet.serving import start_model, stop_model

        draft = gguf.with_suffix(DRAFT_MARK + gguf.suffix)
        draft.write_bytes(b"GGUF" + b"\x00" * 64)

        started = start_model(tmp_path, gguf, manager=manager)
        stop_model(started)
        argv = json.loads((tmp_path / "argv.json").read_text())
        assert argv[argv.index("-md") + 1] == str(draft)
        assert argv[argv.index("-ngld") + 1] == "99"

    def test_a_given_port_is_the_one_used(self, tmp_path, manager, gguf):
        from ml_stack.fleet.serving import start_model, stop_model

        port = free_port()
        started = start_model(tmp_path, gguf, manager=manager, port=port)
        try:
            assert started.port == port
        finally:
            stop_model(started)
