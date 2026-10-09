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
    assert not any(line.startswith("  cluster     ") for line in lines)
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


def test_new_launcher_replaces_owned_older_idle_daemon_before_start(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(launch, "already_running", lambda _port: {"commit": "old", "launcher_control": "a" * 32})
    monkeypatch.setattr(launch, "state", lambda: {"commit": "new"})
    monkeypatch.setattr(launch, "request_replacement", lambda *args, **kwargs: calls.append(("replace", args)))
    monkeypatch.setattr(launch, "_wait_for_exit", lambda _port: True)
    monkeypatch.setattr(launch, "_open_when_ready", lambda *_args: None)
    assert launch.main(["--no-browser", "--root", str(tmp_path)],
                       daemon_main=lambda argv: calls.append(("start", argv)) or 0) == 0
    assert [row[0] for row in calls] == ["replace", "start"]
    assert calls[0][1][0] == tmp_path


def test_new_launcher_keeps_busy_or_unowned_daemon_and_reports_retry(monkeypatch, capsys):
    monkeypatch.setattr(launch, "already_running", lambda _port: {"commit": "old"})
    monkeypatch.setattr(launch, "state", lambda: {"commit": "new"})
    def refuse(*_args, **_kwargs):
        raise launch.ControlError("Daemon has active work; retry when it is idle.")
    monkeypatch.setattr(launch, "request_replacement", refuse)
    assert launch.main(["--no-browser"], daemon_main=lambda _argv: pytest.fail("started over active daemon")) == 1
    assert "retry" in capsys.readouterr().err


def test_launcher_reuses_exact_running_commit_without_replacement(monkeypatch):
    monkeypatch.setattr(launch, "already_running", lambda _port: {"name": "box", "commit": "a" * 40})
    monkeypatch.setattr(launch, "state", lambda: {"commit": "a" * 40})
    monkeypatch.setattr(launch, "request_replacement", lambda *_args: pytest.fail("replaced identical daemon"))
    assert launch.main(["--no-browser"], daemon_main=lambda _argv: pytest.fail("started duplicate")) == 0


def test_launcher_waits_for_tcp_close_when_health_is_unavailable(monkeypatch):
    from contextlib import nullcontext
    clock = iter([0.0, 1.0, 2.0, 3.0])
    connections = []
    def connect(*args, **kwargs):
        connections.append(args)
        if len(connections) == 1:
            return nullcontext()
        if len(connections) == 2:
            raise TimeoutError("listener is still reachable but busy")
        raise ConnectionRefusedError("listener is closed")
    monkeypatch.setattr(launch.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(launch.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(launch.socket, "create_connection", connect)
    monkeypatch.setattr(launch, "already_running", lambda _port: None)
    assert launch._wait_for_exit(8770)
    assert len(connections) == 3


def test_explicit_restart_replaces_same_commit_without_passing_flag_to_daemon(monkeypatch):
    flag = '--restart'
    calls = []
    monkeypatch.setattr(launch, 'already_running', lambda _port: {'commit': 'a' * 40})
    monkeypatch.setattr(launch, 'state', lambda: {'commit': 'a' * 40})
    monkeypatch.setattr(launch, 'request_replacement', lambda *args, **kwargs: calls.append(kwargs) or {'preserved': {'queued': 1, 'running': 1}})
    monkeypatch.setattr(launch, '_wait_for_exit', lambda _port: True)
    monkeypatch.setattr(launch, '_open_when_ready', lambda *_args: None)
    assert launch.main([flag, '--no-browser'], daemon_main=lambda args: calls.append(args) or 0) == 0
    assert calls[0]['restart'] == 'preserve'
    assert flag not in calls[1]


def test_launcher_refuses_the_removed_force_restart_flag_as_a_daemon_option(monkeypatch):
    known, rest = launch._arguments(['--restart', '--no-browser', '--force-restart'])
    assert known.restart and rest == ['--force-restart']


@pytest.fixture
def silent_port():
    """A port that accepts connections and never answers, as a daemon too loaded to reply."""
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(8)
    yield listener.getsockname()[1]
    listener.close()


def test_a_held_port_whose_health_probe_missed_never_starts_a_second_daemon(monkeypatch, silent_port, capsys):
    monkeypatch.setattr(launch, "already_running", lambda _port: None)
    monkeypatch.setattr(launch, "SLOW_HEALTH_TIMEOUT_S", 0.2, raising=False)
    monkeypatch.setattr(launch, "SLOW_HEALTH_TRIES", 2, raising=False)
    assert launch.main(["--no-browser", "--port", str(silent_port)],
                       daemon_main=lambda _argv: pytest.fail("started a daemon on a held port")) == 1
    assert "holds port" in capsys.readouterr().err


def test_a_held_port_that_answers_slowly_is_the_running_daemon(monkeypatch, capsys):
    server = Server(("127.0.0.1", 0), _Health)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        calls = iter([None])
        monkeypatch.setattr(launch, "already_running", lambda _port: next(calls, None))
        monkeypatch.setattr(launch, "same_commit", lambda *_args: True)
        monkeypatch.setattr(launch, "_open_when_ready", lambda *_args: None)
        assert launch.main(["--no-browser", "--port", str(server.server_address[1])],
                           daemon_main=lambda _argv: pytest.fail("started a duplicate")) == 0
        assert "already running" in capsys.readouterr().out
    finally:
        server.shutdown()
        server.server_close()


def test_a_port_nothing_listens_on_starts_the_daemon(monkeypatch):
    calls = []
    monkeypatch.setattr(launch, "already_running", lambda _port: None)
    monkeypatch.setattr(launch, "_open_when_ready", lambda *_args: None)
    assert launch.main(["--no-browser", "--port", str(_free_port())],
                       daemon_main=lambda argv: calls.append(argv) or 0) == 0
    assert calls
