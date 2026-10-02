"""A daemon on a LAN is spoken to by whoever is on it: what it will not take, over real sockets."""

import json
import socket
import threading
import time

import pytest

from ml_stack import macauth
from ml_stack.fleet.api import Daemon, make_handler
from ml_stack.fleet.daemon import load_or_create_token
from ml_stack.fleet.framing import LimitedServer
from ml_stack.fleet.jobs import JobRunner
from ml_stack.fleet.remote import Peer


@pytest.fixture
def served(tmp_path):
    root = tmp_path / "traind"
    files = root / "files"
    files.mkdir(parents=True)
    token = load_or_create_token(root)
    runner = JobRunner(root)
    handler = make_handler(Daemon(runner, files, token, name="alpha"))
    handler.timeout = 1.0
    handler.header_s = 1.0
    httpd = LimitedServer(("127.0.0.1", 0), handler, most=4)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    port = httpd.server_port
    yield {"port": port, "token": token, "files": files, "peer": Peer(f"http://127.0.0.1:{port}", token)}
    runner.shutdown()
    httpd.shutdown()
    httpd.server_close()


def talk(port, raw: bytes, *, read_s=3.0, then=b"") -> tuple[int, bytes]:
    """Send raw bytes, read the status line and body the daemon answers with."""
    with socket.create_connection(("127.0.0.1", port), timeout=read_s) as sock:
        sock.sendall(raw)
        if then:
            sock.sendall(then)
        chunks = []
        try:
            while True:
                got = sock.recv(65536)
                if not got:
                    break
                chunks.append(got)
        except TimeoutError:
            pass
    reply = b"".join(chunks)
    if not reply:
        return 0, b""
    head, _, body = reply.partition(b"\r\n\r\n")
    return int(head.split()[1]), body


def signed(served, method, path, body=b"", extra=""):
    """A request line and headers carrying a valid signature."""
    host = f"127.0.0.1:{served['port']}"
    auth = macauth.sign(served["token"], method, f"http://{host}{path}", body)["Authorization"]
    return (f"{method} {path} HTTP/1.1\r\nHost: {host}\r\nAuthorization: {auth}\r\n{extra}"
            f"Content-Length: {len(body)}\r\nConnection: close\r\n\r\n").encode() + body


# -- who may ask --------------------------------------------------------------------------


def test_an_unsigned_health_check_learns_only_that_the_daemon_is_there(served):
    status, body = talk(served["port"], b"GET /health HTTP/1.0\r\n\r\n")
    assert status == 200 and json.loads(body) == {"ok": True}


def test_a_signed_health_check_gets_the_whole_answer(served):
    assert served["peer"].health()["name"] == "alpha"


def test_a_bearer_token_is_not_accepted(served):
    status, _ = talk(served["port"], (f"GET /jobs HTTP/1.0\r\nAuthorization: Bearer "
                                      f"{served['token']}\r\n\r\n").encode())
    assert status == 401


def test_a_request_without_credentials_is_refused(served):
    for path in ("/jobs", "/models", "/files/x", "/bench", "/availability", "/fetch"):
        assert talk(served["port"], f"GET {path} HTTP/1.0\r\n\r\n".encode())[0] == 401, path
    assert talk(served["port"], b"POST /jobs HTTP/1.0\r\nContent-Length: 2\r\n\r\n{}")[0] == 401
    assert talk(served["port"], b"PUT /files/x HTTP/1.0\r\nContent-Length: 1\r\n\r\nx")[0] == 401


def test_a_captured_request_cannot_be_sent_again(served):
    raw = signed(served, "GET", "/jobs")
    assert talk(served["port"], raw)[0] == 200
    status, body = talk(served["port"], raw)
    assert status == 401 and b"already seen" in body


def test_a_signed_request_cannot_be_altered(served):
    raw = signed(served, "POST", "/jobs", b'{"argv": ["true"]}')
    forged = raw.replace(b'["true"]', b'["fals"]')
    assert talk(served["port"], forged)[0] == 401
    assert talk(served["port"], signed(served, "GET", "/jobs").replace(b"GET /jobs", b"GET /jobs?x=1"))[0] == 401


def test_an_address_that_keeps_guessing_is_locked_out(served):
    wrong = b"GET /jobs HTTP/1.0\r\nAuthorization: ML-Stack-MAC k=00000000,t=1,n=" + b"n" * 24 + b",s=" + b"0" * 64 + b"\r\n\r\n"
    codes = [talk(served["port"], wrong)[0] for _ in range(14)]
    assert 429 in codes and codes[0] == 401
    assert talk(served["port"], signed(served, "GET", "/jobs"))[0] == 429


# -- what a request may claim about its size ----------------------------------------------


@pytest.mark.parametrize("length", ["abc", "-5", "1e3", "0x10", "5.0", "", "+5", "1 2", "9" * 40])
def test_a_content_length_that_is_not_a_plain_number_is_refused(served, length):
    status, _ = talk(served["port"], signed(served, "POST", "/jobs", b"{}").replace(
        b"Content-Length: 2", f"Content-Length: {length}".encode()))
    assert status in (400, 413)


def test_two_content_lengths_that_disagree_are_refused(served):
    raw = signed(served, "POST", "/jobs", b"{}", extra="Content-Length: 5\r\n")
    assert talk(served["port"], raw)[0] == 400


def test_a_huge_content_length_is_refused_without_reading_it(served):
    started = time.monotonic()
    raw = signed(served, "POST", "/jobs", b"{}").replace(b"Content-Length: 2",
                                                         b"Content-Length: 999999999999")
    assert talk(served["port"], raw)[0] == 413
    assert time.monotonic() - started < 1.0


def test_an_upload_over_the_cap_is_refused_before_its_body_is_sent(served):
    raw = signed(served, "PUT", "/files/big", b"x").replace(
        b"Content-Length: 1", f"Content-Length: {25 << 20}".encode())
    assert talk(served["port"], raw)[0] == 413


def test_a_body_that_ends_early_is_refused(served):
    raw = signed(served, "POST", "/jobs", b"{}").replace(b"Content-Length: 2",
                                                         b"Content-Length: 50")
    status, _ = talk(served["port"], raw[:-2] + b"{}")
    assert status in (400, 401, 408)


def test_a_body_that_stops_arriving_ends_at_the_timeout(served):
    started = time.monotonic()
    raw = signed(served, "POST", "/jobs", b"{}").replace(b"Content-Length: 2",
                                                         b"Content-Length: 50")
    status, _ = talk(served["port"], raw[:-2] + b"{", read_s=6.0)
    assert status in (0, 400, 401, 408) and time.monotonic() - started < 5.0


def test_a_body_that_is_not_json_is_refused_not_dropped(served):
    assert talk(served["port"], signed(served, "POST", "/jobs", b"not json"))[0] == 400
    assert talk(served["port"], signed(served, "POST", "/jobs", b"[1,2]"))[0] == 400


@pytest.mark.parametrize("header", [
    "bytes 5-2/10", "bytes 0-9/5", "bytes a-b/c", "bytes 0-4/*x", "bytes 0--1/0",
    "bytes -1-5/10", "bytes 0-2/10", "bytes 0-99999999999999999999/99999999999999999999",
    "items 0-0/1", "bytes 0-0", "bytes 0-0/1,2-2/3",
])
def test_a_content_range_that_does_not_fit_its_body_is_refused(served, header):
    status, _ = talk(served["port"], signed(served, "PUT", "/files/r.bin", b"abcde",
                                           extra=f"Content-Range: {header}\r\n"))
    assert status == 400
    assert not list(served["files"].glob("r.bin*"))


def test_a_content_range_that_fits_is_taken(served):
    status, _ = talk(served["port"], signed(served, "PUT", "/files/r.bin", b"abcde",
                                           extra="Content-Range: bytes 0-4/5\r\n"))
    assert status == 200 and (served["files"] / "r.bin").read_bytes() == b"abcde"


def test_a_range_that_cannot_be_served_is_refused(served):
    (served["files"] / "s.bin").write_bytes(b"0123456789")
    for header in ("bytes=5-2", "bytes=a-b", "bytes=-5", "items=0-1"):
        status, _ = talk(served["port"], signed(served, "GET", "/files/s.bin",
                                               extra=f"Range: {header}\r\n"))
        assert status == 400, header
    status, _ = talk(served["port"], signed(served, "GET", "/files/s.bin",
                                           extra="Range: bytes=50-\r\n"))
    assert status == 416


def test_query_numbers_that_are_not_numbers_fall_back(served):
    job = served["peer"].submit(["true"])
    time.sleep(0.5)
    status, _ = talk(served["port"], signed(served, "GET", f"/jobs/{job['id']}/log?tail=abc"))
    assert status == 200
    status, _ = talk(served["port"], signed(served, "GET", f"/jobs/{job['id']}/metrics?since=-3"))
    assert status == 200


# -- how slowly and how many ------------------------------------------------------------


def test_a_client_that_sends_its_headers_slowly_is_hung_up_on(served):
    started = time.monotonic()
    with socket.create_connection(("127.0.0.1", served["port"]), timeout=6.0) as sock:
        sock.sendall(b"GET /health HTTP/1.1\r\nHost: x\r\n")
        for _ in range(5):
            time.sleep(0.4)
            try:
                sock.sendall(b"X-Slow: 1\r\n")
            except OSError:
                break
        try:
            gone = sock.recv(10) == b""
        except OSError:
            gone = True
    assert gone and time.monotonic() - started < 5.0


def test_more_connections_than_the_cap_get_503_and_the_daemon_recovers(served):
    held = [socket.create_connection(("127.0.0.1", served["port"])) for _ in range(4)]
    try:
        time.sleep(0.3)
        status, _ = talk(served["port"], b"GET /health HTTP/1.0\r\n\r\n")
        assert status == 503
    finally:
        for sock in held:
            sock.close()
    time.sleep(1.5)
    assert talk(served["port"], b"GET /health HTTP/1.0\r\n\r\n")[0] == 200
