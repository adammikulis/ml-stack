"""The daemon listens on this machine alone until it is told, or the machine joins a cluster."""

import contextlib
import socket
import subprocess
import sys
import time

import pytest

from poolhouse.fleet.daemon import ALL_INTERFACES, LOOPBACK, bind_address
from poolhouse.fleet.discovery import primary_ip


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
        [sys.executable, "-m", "poolhouse.fleet.daemon", "--root", str(tmp_path / "traind"),
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

    from poolhouse.fleet.api import Daemon, make_handler
    from poolhouse.fleet.daemon import load_or_create_token
    from poolhouse.fleet.framing import LimitedServer
    from poolhouse.fleet.jobs import JobRunner
    from poolhouse.fleet.ui import UI

    root = tmp_path / "traind"
    (root / "files").mkdir(parents=True)
    runner = JobRunner(root)
    handler = make_handler(Daemon(runner, root / "files", load_or_create_token(root),
                                  ui=UI(name="box", cluster_key_path=tmp_path / "k"),
                                  ui_from_lan=ui_from_lan))
    from poolhouse.fleet import tls

    context = tls.server_context(tls.identity(tmp_path / "tls", "box"))
    httpd = LimitedServer((ALL_INTERFACES, 0), handler, tls=context)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, runner


def _status(address, port):
    import http.client

    from poolhouse.fleet.onboard.pairing import unverified_context

    conn = (http.client.HTTPConnection(address, port, timeout=5) if address.startswith("127.")
            else http.client.HTTPSConnection(address, port, timeout=5, context=unverified_context()))
    conn.request("GET", "/ui/", headers={"X-Poolhouse-UI": "1"})
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


@pytest.mark.slow
def test_a_lan_daemon_serves_its_certificate_and_refuses_plain_http_from_the_lan(tmp_path):
    import ssl

    lan_address = primary_ip()
    if lan_address.startswith("127."):
        pytest.skip("this machine has no address other than loopback")
    port = _free_port()
    proc = _daemon(tmp_path, port, "--lan")
    try:
        assert _reachable("127.0.0.1", port), "the daemon did not come up"
        deadline = time.monotonic() + 20
        cert = tmp_path / "traind" / "tls" / "cert.pem"
        while not cert.exists() and time.monotonic() < deadline:
            time.sleep(0.2)
        # A stranger: it trusts the daemon's certificate but presents none of its own, as a browser would.
        # (`tls.pinned_context` would present this machine's identity, which the daemon does not know.)
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        ctx.minimum_version = ssl.TLSVersion.TLSv1_3
        ctx.check_hostname = False
        ctx.load_verify_locations(cadata=cert.read_text())
        with socket.create_connection((lan_address, port), timeout=5) as raw, \
                ctx.wrap_socket(raw) as secure:
            secure.sendall(b"GET /health HTTP/1.0\r\n\r\n")
            said = b""
            while chunk := secure.recv(4096):
                said += chunk
            assert b'"ok": true' in said
        with socket.create_connection((lan_address, port), timeout=5) as raw:
            raw.sendall(b"GET /health HTTP/1.0\r\n\r\n")
            with contextlib.suppress(ConnectionResetError):
                assert raw.recv(100) == b""
    finally:
        proc.terminate()
        proc.wait(timeout=20)
