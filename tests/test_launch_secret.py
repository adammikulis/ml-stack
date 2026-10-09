"""The launch secret file, and what the window does with a daemon that answers badly."""

from __future__ import annotations

import json
import stat
import sys
from http.server import BaseHTTPRequestHandler
from pathlib import Path

import pytest
from conftest import threaded_server

from poolhouse.fleet import launch_open
from poolhouse.fleet.launch_secret import DIRECTORY, FILE, HEADER, LaunchError, LaunchSecret, read

pytestmark = pytest.mark.redteam


def test_the_secret_is_random_private_and_versioned(tmp_path):
    first = LaunchSecret(tmp_path, 8770)
    second = LaunchSecret(tmp_path / "other", 8770)
    assert first.value != second.value and len(first.value) >= 43
    record = json.loads(first.path.read_text())
    assert record == {"version": 1, "secret": first.value, "port": 8770}
    assert read(tmp_path) == record
    if sys.platform != "win32":
        assert stat.S_IMODE(first.path.stat().st_mode) == 0o600
        assert stat.S_IMODE(first.path.parent.stat().st_mode) == 0o700


def test_a_new_daemon_run_replaces_the_secret(tmp_path):
    old = LaunchSecret(tmp_path, 8770)
    new = LaunchSecret(tmp_path, 8770)
    assert read(tmp_path)["secret"] == new.value != old.value
    assert not new.matches(old.value) and new.matches(new.value)
    assert not new.matches("") and not new.matches("é")


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX permission bits")
def test_a_redirected_or_shared_secret_is_never_read_or_written(tmp_path):
    held = LaunchSecret(tmp_path / "a", 8770)
    held.path.chmod(0o640)
    with pytest.raises(LaunchError, match="mode 640"):
        read(tmp_path / "a")
    held.path.chmod(0o600)
    target = tmp_path / "elsewhere.json"
    target.write_text(held.path.read_text())
    target.chmod(0o600)
    held.path.unlink()
    held.path.symlink_to(target)
    with pytest.raises(LaunchError, match="symlink"):
        read(tmp_path / "a")
    (tmp_path / "linked").mkdir()
    (tmp_path / "linked" / DIRECTORY).symlink_to(tmp_path / "a" / DIRECTORY)
    with pytest.raises(LaunchError, match="symlink"):
        LaunchSecret(tmp_path / "linked", 8770)
    assert (tmp_path / "linked" / DIRECTORY).readlink()


def test_a_missing_or_foreign_record_is_refused(tmp_path):
    with pytest.raises(LaunchError):
        read(tmp_path)
    held = LaunchSecret(tmp_path, 8770)
    for text in ("not json", "[]", json.dumps({"version": 2, "secret": "x", "port": 1}),
                 json.dumps({"version": 1, "secret": 5, "port": 1}),
                 json.dumps({"version": 1, "secret": "x", "port": "1"})):
        held.path.write_text(text)
        held.path.chmod(0o600)
        with pytest.raises(LaunchError):
            read(tmp_path)


def daemon(answer, seen):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self):
            seen.append({"path": self.path, "headers": {k.lower(): v for k, v in self.headers.items()},
                         "body": self.rfile.read(int(self.headers.get("Content-Length") or 0))})
            status, payload = answer
            self.send_response(status)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    return Handler


def record_for(url, root):
    LaunchSecret(root, int(url.rsplit(":", 1)[1]))


def test_the_window_asks_with_the_secret_and_gets_the_page_with_a_ticket(tmp_path):
    seen = []
    with_ticket = (200, json.dumps({"ticket": "abc-DEF_1", "expires_at": 1.0}).encode())
    with threaded_server(daemon(with_ticket, seen)) as url:
        record_for(url, tmp_path)
        port = url.rsplit(":", 1)[1]
        assert launch_open.ticket_url(tmp_path) == f"http://127.0.0.1:{port}/ui/?launch_ticket=abc-DEF_1"
        assert launch_open.page_url(int(port), tmp_path).endswith("?launch_ticket=abc-DEF_1")
        assert seen[0]["path"] == "/ui/launch/ticket"
        assert seen[0]["headers"][HEADER.lower()] == read(tmp_path)["secret"]
        assert "cookie" not in seen[0]["headers"] and "origin" not in seen[0]["headers"]


@pytest.mark.parametrize("answer", [
    (500, b"boom"), (403, b'{"error":"no"}'), (200, b"not json"), (200, b"[]"), (200, b"{}"),
    (200, b'{"ticket": 5}'),
])
def test_a_daemon_that_answers_badly_gives_the_window_the_bare_page(tmp_path, answer):
    with threaded_server(daemon(answer, [])) as url:
        record_for(url, tmp_path)
        port = int(url.rsplit(":", 1)[1])
        with pytest.raises(LaunchError):
            launch_open.ticket_url(tmp_path)
        assert launch_open.page_url(port, tmp_path) == f"http://127.0.0.1:{port}/ui/"


def test_a_record_for_another_port_gives_the_bare_page(tmp_path):
    seen = []
    with threaded_server(daemon((200, b'{"ticket":"abc"}'), seen)) as url:
        record_for(url, tmp_path)
        assert launch_open.page_url(65000, tmp_path) == "http://127.0.0.1:65000/ui/"


def test_the_record_path_is_where_the_daemon_writes_it(tmp_path):
    assert LaunchSecret(tmp_path, 1).path == Path(tmp_path) / DIRECTORY / FILE
