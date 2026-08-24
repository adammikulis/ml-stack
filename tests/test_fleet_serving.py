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
    def test_a_registered_server_that_answers_is_live(self, tmp_path, model):
        s = Serving(tmp_path / "s.json")
        s.register(model.port, ["a.gguf"])
        assert [x.port for x in s.live()] == [model.port]

    def test_a_registered_server_that_died_is_not(self, tmp_path, model):
        """Registration is a claim. A beacon advertising a model nobody can reach
        sends work to a dead port."""
        s = Serving(tmp_path / "s.json")
        s.register(model.port, ["a.gguf"])
        model.close()
        assert s.live() == []

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

    def test_registering_clears_what_was_cached(self, tmp_path, model):
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
