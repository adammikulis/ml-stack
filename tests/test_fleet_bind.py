"""The daemon listens on this machine alone until it is told, or the machine joins a cluster."""

import socket
import subprocess
import sys
import time

import pytest

from ml_stack.fleet.daemon import ALL_INTERFACES, LOOPBACK, bind_address
from ml_stack.fleet.discovery import primary_ip


def test_the_default_is_this_machine_only():
    assert bind_address(None, lan=False, joined=False) == LOOPBACK
    assert bind_address("", lan=False, joined=False) == LOOPBACK


def test_a_cluster_member_or_an_explicit_lan_listens_on_the_network():
    assert bind_address(None, lan=True, joined=False) == ALL_INTERFACES
    assert bind_address(None, lan=False, joined=True) == ALL_INTERFACES


def test_a_named_host_wins_over_everything():
    assert bind_address("127.0.0.1", lan=True, joined=True) == "127.0.0.1"
    assert bind_address("192.0.2.7", lan=False, joined=False) == "192.0.2.7"


def _free_port():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _daemon(tmp_path, port, *flags):
    return subprocess.Popen(
        [sys.executable, "-m", "ml_stack.fleet.daemon", "--root", str(tmp_path / "traind"),
         "--bench-home", str(tmp_path / "bench"), "--port", str(port), "--no-announce",
         "--no-web", "--cluster-key", str(tmp_path / "none.key"), *flags],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _reachable(address, port, *, wait_s=20.0):
    deadline = time.monotonic() + wait_s
    while time.monotonic() < deadline:
        try:
            socket.create_connection((address, port), timeout=1.0).close()
            return True
        except OSError:
            time.sleep(0.2)
    return False


@pytest.mark.slow
@pytest.mark.parametrize("flags, on_the_lan", [((), False), (("--lan",), True)])
def test_a_daemon_started_with_no_flags_cannot_be_reached_from_the_lan(tmp_path, flags,
                                                                       on_the_lan):
    lan_address = primary_ip()
    if lan_address.startswith("127."):
        pytest.skip("this machine has no address other than loopback")
    port = _free_port()
    proc = _daemon(tmp_path, port, *flags)
    try:
        assert _reachable("127.0.0.1", port), "the daemon did not come up"
        assert _reachable(lan_address, port, wait_s=2.0) is on_the_lan
    finally:
        proc.terminate()
        proc.wait(timeout=20)


def _ui_daemon(tmp_path, *, ui_from_lan):
    import threading

    from ml_stack.fleet.api import Daemon, make_handler
    from ml_stack.fleet.daemon import load_or_create_token
    from ml_stack.fleet.framing import LimitedServer
    from ml_stack.fleet.jobs import JobRunner
    from ml_stack.fleet.ui import UI

    root = tmp_path / "traind"
    (root / "files").mkdir(parents=True)
    runner = JobRunner(root)
    handler = make_handler(Daemon(runner, root / "files", load_or_create_token(root),
                                  ui=UI(name="box", cluster_key_path=tmp_path / "k"),
                                  ui_from_lan=ui_from_lan))
    httpd = LimitedServer((ALL_INTERFACES, 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, runner


def _status(address, port):
    import http.client

    conn = http.client.HTTPConnection(address, port, timeout=5)
    conn.request("GET", "/ui/", headers={"X-ML-Stack-UI": "1"})
    got = conn.getresponse().status
    conn.close()
    return got


@pytest.mark.parametrize("ui_from_lan, from_the_lan", [(False, 403), (True, 200)])
def test_the_web_interface_answers_other_machines_only_when_told_to(tmp_path, ui_from_lan,
                                                                    from_the_lan):
    lan_address = primary_ip()
    if lan_address.startswith("127."):
        pytest.skip("this machine has no address other than loopback")
    httpd, runner = _ui_daemon(tmp_path, ui_from_lan=ui_from_lan)
    try:
        assert _status("127.0.0.1", httpd.server_port) == 200
        assert _status(lan_address, httpd.server_port) == from_the_lan
    finally:
        runner.shutdown()
        httpd.shutdown()
        httpd.server_close()
