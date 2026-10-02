"""Requests to servers ml-stack started are served one at a time per pool, across processes.

The servers here are real HTTP servers in this process that record the window in which each
generation was running; the callers are separate Python processes sending through
`ml_stack.http`, the path every consumer uses.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler
from pathlib import Path

import pytest

from ml_stack import gate, home
from ml_stack.http import Server, ServerError, request_json

SRC = str(Path(__file__).resolve().parent.parent / "src")
HOLD_S = 0.3


class Recorder:
    """A server that records when each generation began and ended."""

    def __init__(self, hold_s: float = HOLD_S) -> None:
        self.windows: list[tuple[float, float, str]] = []
        self.hold_s = hold_s
        self.lock = threading.Lock()
        recorder = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_: object) -> None:
                return

            def do_POST(self) -> None:
                self.rfile.read(int(self.headers.get("Content-Length") or 0))
                began = time.time()
                time.sleep(recorder.hold_s)
                ended = time.time()
                with recorder.lock:
                    recorder.windows.append((began, ended, self.headers.get("X-Who", "")))
                body = b'{"choices": [{"message": {"content": "ok"}}]}'
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:
                body = b"{}"
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.httpd = Server(("127.0.0.1", 0), Handler)
        self.port = self.httpd.server_port
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1/chat/completions"

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


def register(*servers: Recorder, pool: str = "gpu") -> None:
    path = home.state("servers.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({str(s.port): {"port": s.port, "pid": os.getpid(), "pool": pool,
                                              "owner_pid": os.getpid()} for s in servers}))


@pytest.fixture
def servers():
    made: list[Recorder] = []

    def make(n: int = 2, hold_s: float = HOLD_S) -> list[Recorder]:
        made.extend(Recorder(hold_s) for _ in range(n))
        register(*made)
        return made

    yield make
    for one in made:
        one.close()


def worker(url: str, who: str, *, env: dict[str, str] | None = None) -> subprocess.Popen:
    code = ("import sys\nfrom ml_stack.http import request_json\n"
            "request_json(sys.argv[1], payload={'messages': []}, headers={'X-Who': sys.argv[2]})\n")
    return subprocess.Popen([sys.executable, "-c", code, url, who],
                            env={**os.environ, "PYTHONPATH": SRC, **(env or {})},
                            stderr=subprocess.PIPE)


def overlap(windows: list[tuple[float, float, str]]) -> float:
    """The most time any two generation windows spent running at once."""
    worst = 0.0
    ordered = sorted(windows)
    for i, (_, end, _) in enumerate(ordered):
        for began, _, _ in ordered[i + 1:]:
            worst = max(worst, end - began)
    return worst


def run_all(procs: list[subprocess.Popen]) -> None:
    for proc in procs:
        _, err = proc.communicate(timeout=60)
        assert proc.returncode == 0, err.decode()


def test_requests_from_separate_processes_to_two_servers_never_overlap(servers):
    first, second = servers(2)
    urls = [first.url, second.url, first.url, second.url, first.url, second.url]
    began = time.time()
    run_all([worker(url, str(i)) for i, url in enumerate(urls)])
    windows = first.windows + second.windows
    assert len(windows) == 6
    assert overlap(windows) <= 0.0
    assert time.time() - began >= 6 * HOLD_S


def test_parallel_requests_are_an_explicit_opt_out_and_do_overlap(servers):
    first, second = servers(2)
    env = {gate.ENV_PARALLEL: "1"}
    run_all([worker(url, str(i), env=env) for i, url in enumerate([first.url, second.url] * 2)])
    assert overlap(first.windows + second.windows) > HOLD_S / 2


def test_requests_are_served_in_the_order_they_arrived(servers):
    first, second = servers(2)
    procs = []
    for i in range(4):
        procs.append(worker([first.url, second.url][i % 2], str(i)))
        time.sleep(0.25)
    run_all(procs)
    order = [who for _, _, who in sorted(first.windows + second.windows)]
    assert order == ["0", "1", "2", "3"]


HOLDER = """
import sys, time
from ml_stack import gate
with gate.turn(sys.argv[1]):
    print('held', flush=True)
    time.sleep(60)
"""


def holder(url: str) -> subprocess.Popen:
    proc = subprocess.Popen([sys.executable, "-c", HOLDER, url],
                            env={**os.environ, "PYTHONPATH": SRC},
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdout is not None
    assert proc.stdout.readline().strip() == b"held", proc.stderr.read() if proc.stderr else b""
    return proc


def test_a_holder_that_is_killed_frees_the_line(servers):
    (one,) = servers(1, hold_s=0.01)
    dead = holder(one.url)
    queued = worker(one.url, "after")
    time.sleep(0.5)
    assert queued.poll() is None, "the request waits while the holder is alive"
    dead.send_signal(signal.SIGKILL)
    dead.wait(timeout=10)
    began = time.time()
    run_all([queued])
    assert time.time() - began < 3.0
    assert [who for _, _, who in one.windows] == ["after"]


def test_a_request_that_waits_too_long_says_who_it_waited_for(servers, monkeypatch):
    (one,) = servers(1)
    held = holder(one.url)
    try:
        monkeypatch.setenv(gate.ENV_WAIT, "0.4")
        began = time.time()
        with pytest.raises(ServerError) as why:
            request_json(one.url, payload={})
        assert 0.4 <= time.time() - began < 3.0
        said = str(why.value)
        assert why.value.status == 429
        assert f"pid {held.pid}" in said and "gpu" in said
        assert gate.ENV_WAIT in said and gate.ENV_PARALLEL in said
        assert one.windows == []
        assert [r["pid"] for r in gate.snapshot()["gpu"]] == [held.pid]
    finally:
        held.kill()
        held.wait(timeout=10)
    assert gate.snapshot() == {}, "a timed-out request leaves no ticket behind"


def test_only_generation_on_registered_local_servers_is_queued(servers):
    (one,) = servers(1)
    held = holder(one.url)
    try:
        assert request_json(f"http://127.0.0.1:{one.port}/props", timeout=5) == {}
        free = Recorder(0.0)
        try:
            request_json(free.url, payload={}, timeout=5)
            assert len(free.windows) == 1, "a server nobody registered is not queued"
        finally:
            free.close()
    finally:
        held.kill()
        held.wait(timeout=10)


def test_pools_queue_apart(servers):
    first, second = servers(2)
    entries = json.loads(home.state("servers.json").read_text())
    entries[str(second.port)]["pool"] = "cpu"
    home.state("servers.json").write_text(json.dumps(entries))
    held = holder(first.url)
    try:
        request_json(second.url, payload={}, timeout=5)
        assert len(second.windows) == 1
    finally:
        held.kill()
        held.wait(timeout=10)


def test_a_thread_holding_the_pool_can_send_a_nested_request(servers):
    (one,) = servers(1, hold_s=0.01)
    with gate.turn(one.url):
        request_json(one.url, payload={}, timeout=5)
    assert len(one.windows) == 1


def test_the_turn_is_held_until_a_streamed_answer_is_read(servers):
    (one,) = servers(1, hold_s=0.01)
    from ml_stack.http import request_stream

    stream = request_stream(one.url, payload={}, timeout=5)
    try:
        next(stream, None)
    except ServerError:
        pass
    stream.close()
    assert gate.snapshot() == {}


def test_parallel_block_is_named_and_logged(servers, caplog):
    (one,) = servers(1, hold_s=0.01)
    held = holder(one.url)
    try:
        with caplog.at_level("WARNING", logger="ml_stack.gate"), gate.parallel("bench sweep"):
            request_json(one.url, payload={}, timeout=5)
        assert len(one.windows) == 1
        assert "bench sweep" in caplog.text
    finally:
        held.kill()
        held.wait(timeout=10)
