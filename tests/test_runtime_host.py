"""`ml-stack runtime restart-host`: the running host is replaced by the selected runtime, gracefully or not at all.

The host is a real HTTP server answering `/health` the way the daemon does; the runtime is a real selected tree from the
deploy tests' fake builder. The only seam is the detached process that would run the replacement: it records its command
instead of replacing anything, and flips the server's commit the way a finished replacement would.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler
from pathlib import Path

import pytest
from test_runtime_deploy import builder, commit, plan_for, world  # noqa: F401

from ml_stack import runtime, runtime_deploy, runtime_host
from ml_stack.http import Server

pytestmark = pytest.mark.slow
OLD = "0" * 40


class Host:
    """A stand-in daemon: answers /health with the commit it runs and, when it has one, its launcher-control instance."""

    def __init__(self, commit_id: str, control: str = "a" * 32) -> None:
        self.say = {"commit": commit_id, "name": "host", **({"launcher_control": control} if control else {})}
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                body = json.dumps(outer.say).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        self.server = Server(("127.0.0.1", 0), Handler)
        self.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def stop(self) -> None:
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def host():
    made = []

    def start(commit_id=OLD, **kw):
        made.append(Host(commit_id, **kw))
        return made[-1]

    yield start
    for one in made:
        one.stop()


@pytest.fixture
def deployed(world):  # noqa: F811
    repo, launchers, _ = world
    head = commit(repo, "a")
    runtime_deploy.ensure(plan_for(repo, launchers), builder=builder())
    return head


class Spawner:
    def __init__(self, serving=None, becomes=""):
        self.calls, self.serving, self.becomes = [], serving, becomes

    def __call__(self, module, argv, *, log):
        self.calls.append((module, list(argv), Path(log)))
        if self.serving is not None and self.becomes:
            self.serving.say["commit"] = self.becomes


def test_no_selected_runtime_means_nothing_to_restart_onto(world, host):  # noqa: F811
    spawn = Spawner()
    out = runtime_host.restart(host().port, Path("/r"), seams=runtime_host.Seams(spawn=spawn))
    assert out.action == "failed" and "no verified runtime" in out.detail and not spawn.calls


def test_a_host_that_is_not_running_is_left_for_the_next_start_to_pick_the_runtime(deployed):
    spawn = Spawner()
    free = Host(OLD)
    port = free.port
    free.stop()
    out = runtime_host.restart(port, Path("/r"), seams=runtime_host.Seams(spawn=spawn))
    assert (out.action, out.commit) == ("not-running", deployed) and not spawn.calls


def test_a_host_already_on_the_selected_runtime_is_not_restarted_unless_forced(deployed, host):
    running, spawn = host(deployed), Spawner()
    assert runtime_host.restart(running.port, Path("/r"), seams=runtime_host.Seams(spawn=spawn)).action == "current" and not spawn.calls
    assert runtime_host.restart(running.port, Path("/r"), seams=runtime_host.Seams(spawn=spawn), force=True, wait_s=0).action == "restarting"
    assert len(spawn.calls) == 1


def test_a_host_with_no_launcher_control_is_refused_and_never_signalled(deployed, host):
    running, spawn = host(control=""), Spawner()
    out = runtime_host.restart(running.port, Path("/r"), seams=runtime_host.Seams(spawn=spawn))
    assert out.action == "failed" and "predates" in out.detail and "Poolside" in out.detail and not spawn.calls


def test_a_behind_host_is_replaced_through_the_job_preserving_restart_on_its_port_and_root(deployed, host):
    running = host()
    spawn = Spawner(serving=running, becomes=deployed)
    out = runtime_host.restart(running.port, Path("/the/root"), seams=runtime_host.Seams(spawn=spawn), wait_s=10)
    assert (out.action, out.commit) == ("restarted", deployed)
    [(module, argv, log)] = spawn.calls
    assert module == "ml_stack.fleet.launch"
    assert argv == ["--restart", "--no-browser", "--port", str(running.port), "--root", "/the/root"]
    assert log == runtime.directory() / "restart.log"


def test_a_replacement_that_does_not_show_up_is_reported_not_forced(deployed, host):
    running, spawn = host(), Spawner()
    out = runtime_host.restart(running.port, Path("/r"), seams=runtime_host.Seams(spawn=spawn), wait_s=1)
    assert out.action == "failed" and "still runs" in out.detail and len(spawn.calls) == 1


def test_a_held_port_that_missed_the_health_probe_is_not_called_not_running(deployed):
    import socket
    spawn = Spawner()
    with socket.socket() as silent:
        silent.bind(("127.0.0.1", 0))
        silent.listen(4)
        seams = runtime_host.Seams(spawn=spawn, health=lambda _port: None)
        out = runtime_host.restart(silent.getsockname()[1], Path("/r"), seams=seams)
    assert out.action == "failed" and "holds port" in out.detail and not spawn.calls
