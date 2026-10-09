"""What a peer may be asked to run: `POST /jobs` takes an allowlist of poolhouse commands and nothing else."""

from __future__ import annotations

import json
import socket
import sys
import threading

import pytest

from poolhouse.fleet import calibration, commands
from poolhouse.fleet.api import Daemon, make_handler
from poolhouse.fleet.daemon import load_or_create_token
from poolhouse.fleet.jobs import JobRunner
from poolhouse.fleet.rates import Rates
from poolhouse.fleet.remote import Peer, PeerError
from poolhouse.fleet.work import Unit, run
from poolhouse.http import Server


class Box:
    """A real daemon with the allowlist it ships with."""

    def __init__(self, tmp_path, name: str) -> None:
        root = tmp_path / name
        (root / "files").mkdir(parents=True)
        token = load_or_create_token(root)
        self.runner = JobRunner(root)
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        self.httpd = Server(("127.0.0.1", port), make_handler(
            Daemon(self.runner, root / "files", token, name=name)))
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.peer = Peer(f"http://127.0.0.1:{port}", token)

    def close(self) -> None:
        self.runner.shutdown()
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def box(tmp_path):
    made = Box(tmp_path, "a")
    try:
        yield made
    finally:
        made.close()


@pytest.mark.parametrize("argv", [
    ["sh", "-c", "id"],
    ["/bin/sh", "-c", "id"],
    ["true"],
    [sys.executable, "-c", "print(1)"],
    ["python3", "-c", "print(1)"],
    ["python3", "-m", "http.server"],
    ["python3", "-m", "poolhouse.serve"],
    ["/usr/local/bin/poolhouse-bench", "--help"],
    ["poolhouse-claude", "hello"],
    [],
])
def test_a_program_that_is_not_on_the_list_is_refused_and_no_job_exists(box, argv):
    with pytest.raises(PeerError) as exc:
        box.peer.submit(argv)
    assert "400" in str(exc.value)
    assert box.peer.jobs() == []


@pytest.mark.parametrize("extra", [{"env": {"LD_PRELOAD": "/tmp/x.so"}}, {"cwd": "/"}, {"cwd": "/tmp", "env": {"A": "b"}}])
def test_an_environment_or_folder_from_the_wire_is_refused(box, extra):
    with pytest.raises(PeerError, match="cwd and env are not accepted"):
        box.peer._json("POST", "/jobs", {"argv": ["poolhouse-bench", "--help"], **extra})
    assert box.peer.jobs() == []


def test_an_allowed_command_runs_in_the_daemons_folder_with_its_own_interpreter(box):
    job = box.peer.submit(["python3", "-m", "poolhouse.fleet.calibration", "--budget", "0.05"])
    done = box.peer.wait(job["id"], poll_s=0.1, timeout_s=60)
    assert done["state"] == "done", box.peer.log(job["id"])
    assert done["argv"][0] == sys.executable and done["cwd"].endswith("files")
    assert json.loads(box.peer.log(job["id"], tail=5).strip().splitlines()[-1])["score"] > 0


def test_a_console_script_on_the_list_is_resolved_and_run(box):
    job = box.peer.submit(["poolhouse-bench", "--help"])
    assert job["argv"][0].endswith("poolhouse-bench") and job["argv"][1:] == ["--help"]
    done = box.peer.wait(job["id"], poll_s=0.1, timeout_s=60)
    assert done["state"] == "done", box.peer.log(job["id"])
    assert "usage" in box.peer.log(job["id"]).lower()


def test_a_fleet_sweep_runs_end_to_end_on_two_daemons(tmp_path):
    boxes = [Box(tmp_path, "a"), Box(tmp_path, "b")]
    try:
        units = [Unit(id=f"u{i}", argv=calibration.argv(0.05)) for i in range(4)]
        placements = run(units, [b.peer for b in boxes], kind="calibrate",
                         rates=Rates(tmp_path / "rates.json"), poll_s=0.1)
    finally:
        for b in boxes:
            b.close()
    assert all(p.ok for p in placements), [p.error for p in placements]
    assert len({p.base_url for p in placements}) >= 1


def test_the_list_resolves_a_name_and_refuses_a_path():
    assert commands.allowed(["python", "-m", "poolhouse.fleet.calibration"])[0] == sys.executable
    assert commands.allowed(["poolhouse-bench", "sweep"])[1:] == ["sweep"]
    with pytest.raises(ValueError, match="not a command a peer may be asked to run"):
        commands.allowed(["./poolhouse-bench"])
