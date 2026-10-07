"""The installer's last screen, against a real daemon's /health on loopback or nothing at all."""

from __future__ import annotations

import json
import socket
import threading
from http.server import BaseHTTPRequestHandler

import pytest

from ml_stack.fleet import autostart, launch
from ml_stack.http import Server
from tests.cluster_support import join_cluster


class _Health(BaseHTTPRequestHandler):
    def do_GET(self):
        body = json.dumps({"name": "box", "machine": "m1"}).encode()
        self.send_response(200 if self.path == "/health" else 404)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


@pytest.fixture
def daemon():
    server = Server(("127.0.0.1", 0), _Health)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield server.server_address[1]
    server.shutdown()
    server.server_close()


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_with_nothing_answering_it_says_how_to_start_the_page():
    port = _free_port()
    lines = launch.last_screen("box", port=port)
    text = "\n".join(lines)
    assert "  machine     box" in lines
    assert "  open" not in text
    assert "cluster" not in text
    assert f"ml-stack                        -- starts it and opens http://127.0.0.1:{port}/ui/" \
        in text
    assert "ml-stack-cluster join --persist" in text


def test_with_a_daemon_answering_it_names_the_page_and_the_cluster(daemon):
    join_cluster("correct horse battery staple", group="attic")
    lines = launch.last_screen("box", track="main", port=daemon)
    assert f"  open        http://127.0.0.1:{daemon}/ui/" in lines
    assert "  cluster     attic" in lines
    assert "  updates     follows main, whenever nothing is running here" in lines
    assert not [line for line in lines if "starts it and opens" in line]


def test_the_version_is_not_said_twice():
    running = next(line for line in launch.last_screen("box", port=_free_port())
                   if line.startswith("  running"))
    words = running.split()
    assert len(words) == len(set(words)), running


def test_the_installer_prints_it(capsys):
    assert autostart.main(["done", "--name", "box"]) == 0
    assert "  machine     box" in capsys.readouterr().out


def test_browser_waits_through_slow_startup(monkeypatch, capsys):
    clock = iter([0.0, 21.0, 22.0])
    answers = iter([None, None, {"name": "box"}])
    opened = []
    stopped = threading.Event()
    monkeypatch.setattr(launch.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(launch, "_health", lambda port: next(answers))
    monkeypatch.setattr(stopped, "wait", lambda seconds: False)
    monkeypatch.setattr(launch.webbrowser, "open", opened.append)
    launch._open_when_ready(8770, True, stopped)
    assert opened == ["http://127.0.0.1:8770/ui/"]
    text = capsys.readouterr().out
    assert text.count("still starting") == 1
    assert "did not start" not in text


def test_browser_stops_waiting_when_daemon_exits(monkeypatch):
    stopped = threading.Event()
    opened = []

    def health(port):
        stopped.set()
        return None

    monkeypatch.setattr(launch, "_health", health)
    monkeypatch.setattr(launch.webbrowser, "open", opened.append)
    launch._open_when_ready(8770, True, stopped)
    assert not opened


def test_no_browser_still_waits_for_health(monkeypatch):
    monkeypatch.setattr(launch, "_health", lambda port: {})
    monkeypatch.setattr(launch.webbrowser, "open", lambda url: pytest.fail(url))
    launch._open_when_ready(8770, False, threading.Event())


def test_a_vcs_install_names_its_commit_on_the_done_screen(monkeypatch):
    from ml_stack.fleet import measuring, updates

    class VcsInstalled:
        def read_text(self, name):
            return ('{"url": "https://example.invalid/ml-stack.git", "vcs_info": '
                    '{"vcs": "git", "commit_id": "0ce5bc5' + "1" * 33 + '"}}')

    monkeypatch.setattr(measuring, "repo_root", lambda where: None)
    monkeypatch.setattr(measuring, "distribution", lambda name: VcsInstalled())
    monkeypatch.setattr(updates, "_COMMIT", [])
    monkeypatch.setattr(updates, "LAST", {})
    monkeypatch.setattr(updates, "commit_age_s", lambda commit: 0.0)
    running = next(line for line in launch.last_screen("box", port=_free_port())
                   if line.startswith("  running"))
    assert running.endswith("  0ce5bc5"), running
