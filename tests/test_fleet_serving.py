"""Reaching a model server on another machine, through that machine's daemon."""

from __future__ import annotations

import json
import socket
import struct
import threading
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler

import pytest

from ml_stack.fleet.api import Daemon, make_handler
from ml_stack.fleet.daemon import load_or_create_token
from ml_stack.fleet.jobs import JobRunner
from ml_stack.fleet.serving import Endpoint, Serving, answers
from ml_stack.http import Server, build_request
from ml_stack.testing.fakes import FakeLlamaServer, Served, fake_llama_binary


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Running:
    def __init__(self, tmp_path, serving):
        root = tmp_path / "traind"
        files = root / "files"
        files.mkdir(parents=True)
        self.token = load_or_create_token(root)
        self.runner = JobRunner(root, files)
        self.port = free_port()
        self.httpd = Server(
            ("127.0.0.1", self.port),
            make_handler(Daemon(self.runner, files, self.token, "box", serving=serving)))
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def post(self, path, body=None, token=None, stream=False):
        req = build_request(
            f"http://127.0.0.1:{self.port}{path}", data=json.dumps(body or {}).encode(),
            method="POST", headers={"Content-Type": "application/json"},
            token="" if token is False else (token or self.token))
        return urllib.request.urlopen(req, timeout=30)

    def close(self):
        self.runner.shutdown()
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def model():
    """A model server that streams a reply with real gaps between its pieces."""
    m = FakeLlamaServer(Served(model="qwen3-4b.gguf", answer="hello", gap=0.25,
                               pieces=tuple(f"tok{n}" for n in range(8))))
    try:
        yield m
    finally:
        m.close()


@pytest.fixture
def wired(tmp_path, model):
    serving = Serving(tmp_path / "serving.json")
    serving.register(model.port, ["qwen3-4b.gguf"], slots=4)
    d = Running(tmp_path, serving)
    try:
        yield d, serving, model
    finally:
        d.close()


@contextmanager
def metadata_server(body, *, redirect=""):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(302 if redirect else 200)
            if redirect:
                self.send_header("Location", redirect)
            self.end_headers()
            self.wfile.write(json.dumps(body).encode())

        def log_message(self, *_):
            pass

    server = Server(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        yield server.server_port
    finally:
        server.shutdown()
        server.server_close()
        worker.join()


class TestRegistry:
    def test_verified_unchanged_reads_do_not_create_a_write_lock(self, tmp_path, model):
        registry = Serving(tmp_path / "serving.json")
        registry.register(model.port, ["qwen3-4b.gguf"])
        expected = registry.live(force=True)
        lock_path = registry.path.with_suffix(".lock")
        lock_path.unlink()
        assert registry.live(force=True) == expected
        assert not lock_path.exists()

    def test_reconciliation_preserves_registration_changed_after_probe(self, tmp_path, model, monkeypatch):
        from dataclasses import replace

        from ml_stack.fleet import serving as module

        registry = Serving(tmp_path / "serving.json")
        original = registry.register(model.port, ["stale-claim.gguf"])
        current = replace(original, models=["concurrent-claim.gguf"])
        take_lock = module.lock.only_one

        @contextmanager
        def concurrent_registration(*args, **kwargs):
            with take_lock(*args, **kwargs) as held:
                registry._write([current])
                yield held

        monkeypatch.setattr(module.lock, "only_one", concurrent_registration)
        live = registry.live(force=True)
        assert live[0].models == ["qwen3-4b.gguf"]
        assert registry.all() == [current]

    @pytest.mark.parametrize("port", [-1, 0, 65536, True, "8080", None])
    def test_invalid_persisted_ports_are_never_probed(self, tmp_path, port):
        path = tmp_path / "serving.json"
        path.write_text(json.dumps([{"port": port, "models": ["claim.gguf"]}]))
        registry = Serving(path)
        assert registry.all() == registry.live(force=True) == []

    @pytest.mark.parametrize("body", [[], {"data": None}, {"data": 3}, {"data": [None, {"id": 3}, {"id": ""}]}])
    def test_malformed_metadata_does_not_advertise_registration_claims(self, tmp_path, body):
        with metadata_server(body) as port:
            registry = Serving(tmp_path / "serving.json")
            registry.register(port, ["claim.gguf"])
            assert registry.live(force=True) == []
            assert registry.all()[0].models == ["claim.gguf"]

    def test_metadata_redirect_cannot_borrow_another_endpoints_identity(self, tmp_path, model):
        with metadata_server({}, redirect=f"http://127.0.0.1:{model.port}/v1/models") as port:
            registry = Serving(tmp_path / "serving.json")
            registry.register(port, ["claim.gguf"])
            assert registry.live(force=True) == []
            assert registry.all()[0].models == ["claim.gguf"]

    def test_recycled_port_reports_its_actual_model_and_prunes_dead_rows(self, tmp_path, model):
        registry = Serving(tmp_path / "serving.json")
        registry.register(model.port, ["stale-claim.gguf"])
        registry.register(free_port(), ["dead-claim.gguf"])
        live = registry.live(force=True)
        assert [(one.port, one.models) for one in live] == [(model.port, ["qwen3-4b.gguf"])]
        assert [(one.port, one.models) for one in registry.all()] == [(model.port, ["qwen3-4b.gguf"])]

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

    def test_it_sends_the_leased_servers_key_upstream(self, wired):
        from ml_stack import serverkeys

        daemon, _, model = wired
        model.api_key = serverkeys.issue(model.port)
        with daemon.post("/infer/v1/chat/completions",
                         {"messages": [{"role": "user", "content": "hi"}]}) as r:
            assert json.loads(r.read())["choices"][0]["message"]["content"] == "hello"

    def test_an_unauthenticated_model_identity_is_not_advertised(self, wired):
        daemon, serving, model = wired
        model.api_key = "not-in-the-key-file"
        with pytest.raises(urllib.error.HTTPError) as exc:
            daemon.post("/infer/v1/chat/completions", {"messages": []})
        assert exc.value.code == 503
        assert serving.live(force=True) == []

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
        daemon = Running(tmp_path, serving)
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

    @pytest.mark.parametrize("path", [
        "/infer/v1/../props", "/infer/v1/%2e%2e/slots", "/infer/admin", "/infer/", "/infer",
        "/infer//etc/passwd", "/infer/v1/x%20y", "/infer/slots/0?action=erase",
        "/infer/props", "/infer/models",
    ])
    def test_a_path_that_is_not_the_model_servers_to_give_is_refused(self, wired, path):
        daemon, _, _ = wired
        with pytest.raises(urllib.error.HTTPError) as exc:
            daemon.post(path, {})
        assert exc.value.code == 403, path

    def test_the_caches_and_properties_of_the_model_server_are_read_not_written(self, wired):
        daemon, _, _ = wired
        for path in ("/infer/slots", "/infer/props"):
            with pytest.raises(urllib.error.HTTPError) as exc:
                daemon.post(path, {})
            assert exc.value.code == 403, path

    def test_a_path_cannot_name_another_host(self, wired):
        """`/infer@host:port/` would read as userinfo in front of a host the caller picked."""
        daemon, _, _ = wired
        other = FakeLlamaServer(Served(answer="elsewhere"))
        try:
            with pytest.raises(urllib.error.HTTPError) as exc:
                daemon.post(f"/infer@127.0.0.1:{other.port}/v1/chat/completions", {})
            assert exc.value.code == 404
        finally:
            other.close()


class TestEndpoint:
    def test_it_hands_a_client_exactly_what_it_needs(self):
        """If this needs a change in Client, the design is wrong."""
        import inspect

        from ml_stack.client import Client, Transport

        kwargs = Endpoint(peer="gpubox", base_url="http://box:8770",
                          token="abc").client_kwargs()
        assert kwargs == {"base_url": "http://box:8770/infer",
                          "transport": Transport(api_key="abc")}
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


def test_a_probe_takes_a_server_that_only_lists_models(server):
    from conftest import json_reply

    instance = server(lambda m, p, b: json_reply({"data": []})
                      if p == "/models" else (404, b"no such route"))
    assert answers(instance.port, timeout=2.0)


@pytest.fixture
def llama_binary(tmp_path, monkeypatch):
    """A llama-server that really launches, binds the port it was given and answers."""
    from ml_stack.serve import backend as backend_module

    monkeypatch.setattr(backend_module, "log_dir", lambda: tmp_path / "logs")
    return fake_llama_binary(tmp_path)


@pytest.fixture
def manager(tmp_path, llama_binary):
    from ml_stack.serve import LlamaServerBackend, ServerManager

    return ServerManager(LlamaServerBackend(binary=llama_binary),
                         state_file=tmp_path / "servers.json")


@pytest.fixture
def gguf(tmp_path):
    path = tmp_path / "Qwen3-4B-Q4_K_M.gguf"
    path.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 0, 0))
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
        from ml_stack.fleet.serving import start_model, stop_model
        from ml_stack.hub import DRAFT_MARK

        draft = gguf.with_suffix(DRAFT_MARK + gguf.suffix)
        draft.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 0, 0))

        started = start_model(tmp_path, gguf, manager=manager)
        stop_model(started)
        argv = json.loads((tmp_path / "argv.json").read_text())
        assert argv[argv.index("-md") + 1] == str(draft)
        assert argv[argv.index("--spec-draft-ngl") + 1] == "99"

    def test_a_given_port_is_the_one_used(self, tmp_path, manager, gguf):
        from ml_stack.fleet.serving import start_model, stop_model

        port = free_port()
        started = start_model(tmp_path, gguf, manager=manager, port=port)
        try:
            assert started.port == port
        finally:
            stop_model(started)
