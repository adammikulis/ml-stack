"""The two places autostart reaches outside itself: a process started from an argument list, and the daemon's loopback health probe."""

from __future__ import annotations

import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from poolhouse.fleet.autostart_apply import healthy
from poolhouse.fleet.autostart_backends import run

HOSTILE = "a b'c\"$x;`id`|&>\nz"


def test_run_hands_every_argument_to_the_program_unchanged():
    code, out = run([sys.executable, "-c", "import sys; sys.stdout.write(sys.argv[1])", HOSTILE])
    assert code == 0 and out == HOSTILE


def test_run_reports_a_program_that_cannot_start_as_a_failure():
    code, out = run(["/no/such/program", HOSTILE])
    assert code == 1 and out


def test_run_bounds_a_program_that_never_exits():
    started = time.monotonic()
    code, _ = run([sys.executable, "-c", "import time; time.sleep(30)"], timeout=1)
    assert code == 1 and time.monotonic() - started < 10


class Answer(BaseHTTPRequestHandler):
    def do_GET(self):
        mode = self.server.mode
        if mode == "slow":
            time.sleep(6)
        body = {"ok": b'{"ok": true}', "html": b"<html>nope", "slow": b"{}"}.get(mode, b"{}")
        self.send_response(500 if mode == "error" else 200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


@pytest.fixture
def server():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Answer)
    httpd.mode = "ok"
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield httpd
    httpd.shutdown()
    httpd.server_close()


def probe(server, timeout=1):
    return healthy({"kind": "http", "url": f"http://127.0.0.1:{server.server_port}/health", "timeout_s": timeout})


def test_a_daemon_that_answers_is_healthy(server):
    assert probe(server) is True


@pytest.mark.parametrize("mode", ["error", "html"])
def test_an_error_or_a_non_json_answer_is_not_healthy(server, mode):
    server.mode = mode
    assert probe(server) is False


def test_a_slow_answer_is_not_healthy(server):
    server.mode = "slow"
    started = time.monotonic()
    assert probe(server) is False
    assert time.monotonic() - started < 12


def test_nothing_listening_is_not_healthy(server):
    server.shutdown()
    server.server_close()
    assert probe(server) is False


def test_a_role_with_no_resident_process_has_nothing_to_probe():
    assert healthy({"kind": "loaded"}) is True
