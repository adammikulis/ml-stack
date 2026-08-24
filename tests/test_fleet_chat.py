"""Talking to a model, including from a machine that installed nothing to run one.

The test that matters here is the last one: a daemon with no model store, nothing
serving locally and no way to run a model still holds a conversation, because another
machine on the network is serving. That is the whole point of the fleet, and it is the
first thing a refactor would quietly break.
"""

from __future__ import annotations

import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from ml_stack.fleet.chat import find, targets
from ml_stack.fleet.daemon import JobRunner, load_or_create_token, make_handler
from ml_stack.fleet.discovery import join_cluster
from ml_stack.fleet.serving import Serving
from test_fleet_ui import WORDS, Serving as UIServing  # noqa: N811

PIECES = ["Hel", "lo", " there"]


def tmp_path_of(ui):
    return ui.files.parent


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def model_server():
    """A llama.cpp-shaped server that streams a reply a piece at a time."""

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"{}")

        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0"))
            self.rfile.read(length)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for piece in PIECES:
                frame = {"choices": [{"delta": {"content": piece}}]}
                self.wfile.write(f"data: {json.dumps(frame)}\n\n".encode())
                self.wfile.flush()
                time.sleep(0.05)
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()

    srv = ThreadingHTTPServer(("127.0.0.1", _free_port()), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        yield srv.server_address[1]
    finally:
        srv.shutdown()


@pytest.fixture
def host(tmp_path, model_server):
    """A second machine: a real daemon proxying to a real model server."""
    root = tmp_path / "host"
    files = root / "files"
    files.mkdir(parents=True)
    key = join_cluster(WORDS, group="home", path=tmp_path / "host.key")
    token = load_or_create_token(root, key)
    serving = Serving(root / "serving.json")
    serving.register(model_server, ["qwen3-4b.gguf"])
    runner = JobRunner(root, files)
    port = _free_port()
    httpd = ThreadingHTTPServer(
        ("127.0.0.1", port),
        make_handler(runner, files, token, "host", serving=serving))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        yield {"name": "host", "base_url": f"http://127.0.0.1:{port}",
               "is_self": False, "device": {"serving": serving.public()}}
    finally:
        runner.shutdown()
        httpd.shutdown()
        httpd.server_close()


class TestPickingWhereToSend:
    def test_a_machine_serving_nothing_still_sees_a_peers_model(self, host):
        found = targets([host], serving=None, token="t")
        assert [t.model for t in found] == ["qwen3-4b.gguf"]
        assert found[0].peer == "host"
        assert found[0].local is False
        assert found[0].url.endswith("/infer/v1/chat/completions")

    def test_this_machine_is_preferred_over_a_peer_holding_the_same_model(self, host):
        class Fake:
            def live(self):
                from ml_stack.fleet.serving import Served
                return [Served(port=9999, models=["qwen3-4b.gguf"])]

        found = targets([host], serving=Fake(), token="t")
        assert len(found) == 1
        assert found[0].local is True

    def test_it_never_offers_this_machines_own_beacon(self):
        me = {"name": "me", "base_url": "http://127.0.0.1:1", "is_self": True,
              "device": {"serving": [{"port": 1, "models": ["a.gguf"]}]}}
        assert targets([me], serving=None) == []

    def test_naming_a_model_loosely_still_finds_it(self, host):
        found = targets([host], serving=None, token="t")
        assert find(found, "qwen3").model == "qwen3-4b.gguf"
        assert find(found, "").model == "qwen3-4b.gguf"
        assert find(found, "nothing-like-this") is None
        assert find([], "anything") is None


class TestChattingThroughTheInterface:
    @pytest.fixture
    def bare(self, tmp_path, host):
        """A machine that installed nothing: no model store, nothing serving."""
        ui = UIServing(tmp_path / "bare", name="laptop")
        join_cluster(WORDS, group="home", path=ui.keyfile)
        ui.call("/ui/setup/join", method="POST",
                body={"passphrase": WORDS, "group": "home"})
        _, _, headers = ui.call("/ui/session", method="POST",
                                body={"passphrase": WORDS})
        assert ui.ui.models is None
        assert ui.ui.serving is None
        ui.ui._peers = (time.time(), [host])
        try:
            yield ui, headers["Set-Cookie"].split(";")[0]
        finally:
            ui.close()

    def test_it_lists_the_model_another_machine_is_serving(self, bare):
        ui, cookie = bare
        status, body, _ = ui.call("/ui/chat", cookie=cookie)
        assert status == 200
        assert [m["model"] for m in body["models"]] == ["qwen3-4b.gguf"]
        assert body["models"][0]["peer"] == "host"

    def test_a_machine_with_nothing_installed_holds_a_conversation(self, bare):
        ui, cookie = bare
        status, raw, headers = ui.call(
            "/ui/chat", method="POST", cookie=cookie,
            body={"model": "qwen3-4b.gguf",
                  "messages": [{"role": "user", "content": "hi"}]})
        assert status == 200
        assert headers.get("X-ML-Stack-Peer") == "host"
        text = raw["raw"] if "raw" in raw else json.dumps(raw)
        assert "Hel" in text and "there" in text

    def test_the_reply_arrives_as_it_is_generated(self, bare):
        """read(n) waits for n bytes and delivers the whole reply at once."""
        import urllib.request

        ui, cookie = bare
        req = urllib.request.Request(
            f"http://127.0.0.1:{ui.port}/ui/chat", method="POST",
            data=json.dumps({"model": "qwen3-4b.gguf",
                             "messages": [{"role": "user", "content": "hi"}]}).encode())
        req.add_header("Content-Type", "application/json")
        req.add_header("X-ML-Stack-UI", "1")
        req.add_header("Cookie", cookie)

        began = time.monotonic()
        arrived = []
        with urllib.request.urlopen(req, timeout=10) as r:
            while True:
                block = r.read1(4096)
                if not block:
                    break
                arrived.append((time.monotonic() - began, block))

        assert len(arrived) > 1, "the whole reply came in one piece"
        spread = arrived[-1][0] - arrived[0][0]
        assert spread > 0.05, f"every piece arrived at once ({spread:.3f}s apart)"

    def test_what_was_said_is_kept(self, bare, tmp_path):
        from ml_stack.fleet.conversations import Conversations

        ui, cookie = bare
        ui.ui.conversations = Conversations(tmp_path / "bare" / "chats")
        status, made, _ = ui.call("/ui/conversations", method="POST", cookie=cookie,
                                  body={"model": "qwen3-4b.gguf"})
        assert status == 201

        ui.call("/ui/chat", method="POST", cookie=cookie,
                body={"conversation": made["id"], "model": "qwen3-4b.gguf",
                      "messages": [{"role": "user", "content": "hi"}]})

        # A fresh handle, as a restarted daemon would have.
        kept = Conversations(tmp_path / "bare" / "chats").get(made["id"])
        assert [m.role for m in kept.messages] == ["user", "assistant"]
        assert kept.messages[0].content == "hi"
        assert kept.messages[1].content == "".join(PIECES)
        assert kept.title == "hi"

    def test_chats_are_listed_and_searchable_over_the_interface(self, bare, tmp_path):
        from ml_stack.fleet.conversations import Conversations

        ui, cookie = bare
        ui.ui.conversations = Conversations(tmp_path / "bare" / "chats")
        _, a, _ = ui.call("/ui/conversations", method="POST", cookie=cookie, body={})
        ui.ui.conversations.append(a["id"], "user", "about safetensors")
        _, b, _ = ui.call("/ui/conversations", method="POST", cookie=cookie, body={})
        ui.ui.conversations.append(b["id"], "user", "about quantising")

        _, listed, _ = ui.call("/ui/conversations", cookie=cookie)
        assert len(listed["conversations"]) == 2
        assert "messages" not in listed["conversations"][0]

        _, hit, _ = ui.call("/ui/conversations?q=safetensors", cookie=cookie)
        assert [c["id"] for c in hit["conversations"]] == [a["id"]]

        status, one, _ = ui.call(f"/ui/conversations/{a['id']}", cookie=cookie)
        assert status == 200
        assert one["messages"][0]["content"] == "about safetensors"

        status, _, _ = ui.call(f"/ui/conversations/{a['id']}", method="DELETE",
                               cookie=cookie)
        assert status == 200
        _, left, _ = ui.call("/ui/conversations", cookie=cookie)
        assert [c["id"] for c in left["conversations"]] == [b["id"]]

    def test_asking_for_a_chat_that_is_not_there_is_a_404(self, bare, tmp_path):
        from ml_stack.fleet.conversations import Conversations

        ui, cookie = bare
        ui.ui.conversations = Conversations(tmp_path / "bare" / "chats")
        assert ui.call("/ui/conversations/nope", cookie=cookie)[0] == 404

    def test_a_machine_that_cannot_run_a_model_says_so_rather_than_breaking(
            self, bare, monkeypatch):
        """The install that cannot serve is the common one. It must still answer."""
        from ml_stack.fleet import ui as ui_mod

        ui, cookie = bare
        ui.ui.serving = Serving(tmp_path_of(ui) / "serving.json")
        monkeypatch.setattr(ui_mod, "_can_serve", lambda: False)

        status, body, _ = ui.call("/ui/serving", cookie=cookie)
        assert status == 200
        assert body["can_serve"] is False
        assert body["running"] == []

        status, body, _ = ui.call("/ui/serving", method="POST", cookie=cookie,
                                  body={"name": "anything.gguf"})
        assert status == 501
        assert "network can serve one" in body["error"]

    def test_a_daemon_with_no_serving_at_all_answers_the_route(self, bare):
        ui, cookie = bare
        assert ui.ui.serving is None
        status, body, _ = ui.call("/ui/serving", cookie=cookie)
        assert status == 501
        assert "run a model" in body["error"]

    def test_asking_for_a_model_answers_before_it_has_arrived(self, bare, tmp_path):
        """A multi-gigabyte fetch must not be held open inside one request."""
        import os
        import time as clock
        from http.server import BaseHTTPRequestHandler
        from ml_stack.fleet.models import CHUNK, Downloads, Models

        payload = os.urandom(2 * CHUNK)

        class Slow(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                self.send_response(200)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                for i in range(0, len(payload), 65536):
                    self.wfile.write(payload[i:i + 65536])
                    self.wfile.flush()
                    clock.sleep(0.01)

        blob = ThreadingHTTPServer(("127.0.0.1", _free_port()), Slow)
        threading.Thread(target=blob.serve_forever, daemon=True).start()

        ui, cookie = bare
        shelf = tmp_path / "bare" / "models"
        shelf.mkdir(parents=True, exist_ok=True)
        ui.ui.models = Models([shelf], shelf)
        ui.ui.downloads = Downloads(ui.ui.models)
        try:
            began = clock.monotonic()
            status, body, _ = ui.call(
                "/ui/models", method="POST", cookie=cookie,
                body={"name": "big.gguf",
                      "source": f"http://127.0.0.1:{blob.server_address[1]}/big.gguf"})
            answered = clock.monotonic() - began

            assert status == 202, body
            assert body["state"] == "getting"
            assert answered < 1.0, f"the request waited {answered:.1f}s"

            saw_partial = False
            for _ in range(300):
                _, listing, _ = ui.call("/ui/models", cookie=cookie)
                rows = listing.get("getting") or []
                row = next((g for g in rows if g["id"] == body["id"]), None)
                if row is None:
                    break
                if 0 < row["done"] < row["total"]:
                    saw_partial = True
                if row["state"] != "getting":
                    assert row["state"] == "done", row
                    break
                clock.sleep(0.05)
        finally:
            blob.shutdown()

        assert saw_partial, "the screen could never show how far along it was"
        assert (shelf / "big.gguf").read_bytes() == payload

    def test_a_model_already_here_answers_at_once(self, bare, tmp_path):
        from ml_stack.fleet.models import Downloads, Models

        ui, cookie = bare
        shelf = tmp_path / "bare" / "models"
        shelf.mkdir(parents=True, exist_ok=True)
        (shelf / "here.gguf").write_bytes(b"x" * (2 * 1024 * 1024))
        ui.ui.models = Models([shelf], shelf)
        ui.ui.downloads = Downloads(ui.ui.models)

        status, body, _ = ui.call("/ui/models", method="POST", cookie=cookie,
                                  body={"name": "here.gguf"})
        assert status == 200
        assert body["name"] == "here.gguf"

    def test_with_nobody_serving_it_says_so_rather_than_failing(self, tmp_path):
        ui = UIServing(tmp_path / "alone", name="alone")
        try:
            ui.call("/ui/setup/join", method="POST",
                    body={"passphrase": WORDS, "group": "home"})
            _, _, headers = ui.call("/ui/session", method="POST",
                                    body={"passphrase": WORDS})
            cookie = headers["Set-Cookie"].split(";")[0]
            ui.ui._peers = (time.time(), [])
            status, body, _ = ui.call(
                "/ui/chat", method="POST", cookie=cookie,
                body={"messages": [{"role": "user", "content": "hi"}]})
            assert status == 503
            assert "serving a model" in body["error"]
        finally:
            ui.close()



class TestAnsweringToSeveralClusters:
    """The bearer token is derived from a cluster key, so a machine in two clusters
    has two tokens and must accept either."""

    def daemon(self, tmp_path, anchor):
        from http.server import ThreadingHTTPServer

        from ml_stack.fleet.daemon import JobRunner, load_or_create_token, make_handler
        from ml_stack.fleet.discovery import derive_token, memberships

        root = tmp_path / "traind"
        files = root / "files"
        files.mkdir(parents=True, exist_ok=True)
        rows = memberships(anchor)
        token = load_or_create_token(root, rows[0].key)
        runner = JobRunner(root, files)
        port = _free_port()
        httpd = ThreadingHTTPServer(
            ("127.0.0.1", port),
            make_handler(runner, files, token, "box",
                         cluster_key_path=anchor,
                         tokens=lambda: {derive_token(m.key)
                                         for m in memberships(anchor)}))
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        return f"http://127.0.0.1:{port}", runner, httpd

    def ask(self, base, token):
        import urllib.error
        import urllib.request

        # /health answers anyone: it is how a peer checks a machine is alive. /jobs
        # is behind the bearer token, which is what this is about.
        req = urllib.request.Request(f"{base}/jobs")
        req.add_header("Authorization", f"Bearer {token}")
        try:
            with urllib.request.urlopen(req, timeout=5) as r:
                return r.status
        except urllib.error.HTTPError as exc:
            return exc.code

    def test_either_cluster_can_reach_it(self, tmp_path):
        from ml_stack.fleet.discovery import derive_token, join

        anchor = tmp_path / "cluster.key"
        join(WORDS, group="home", path=anchor)
        join("a different set of words here", group="work", path=anchor)
        rows = {m.group: m.key for m in __import__(
            "ml_stack.fleet.discovery", fromlist=["memberships"]).memberships(anchor)}

        base, runner, httpd = self.daemon(tmp_path, anchor)
        try:
            assert self.ask(base, derive_token(rows["home"])) == 200
            assert self.ask(base, derive_token(rows["work"])) == 200, (
                "the second cluster was refused")
            assert self.ask(base, "not-a-real-token") == 401
        finally:
            runner.shutdown()
            httpd.shutdown()
            httpd.server_close()

    def test_a_cluster_it_left_is_refused(self, tmp_path):
        from ml_stack.fleet.discovery import derive_token, join, leave, memberships

        anchor = tmp_path / "cluster.key"
        join(WORDS, group="home", path=anchor)
        join("a different set of words here", group="work", path=anchor)
        gone = derive_token({m.group: m.key
                             for m in memberships(anchor)}["work"])

        base, runner, httpd = self.daemon(tmp_path, anchor)
        try:
            assert self.ask(base, gone) == 200
            leave("work", anchor)
            assert self.ask(base, gone) == 401, "it still answered a cluster it left"
        finally:
            runner.shutdown()
            httpd.shutdown()
            httpd.server_close()
