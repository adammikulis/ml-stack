"""The training daemon: real HTTP, real subprocesses, real files.

Nothing here is mocked. The failures this module exists to prevent are
process-shaped and transport-shaped -- a second job started on a busy GPU, a
SIGTERM that never arrives, an upload path that escapes its root -- and a
mocked socket or a fake Popen reproduces none of them.
"""

from __future__ import annotations

import json
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest
from mainspring.train.daemon import (
    DaemonError,
    JobRunner,
    load_or_create_token,
    make_handler,
    safe_relpath,
)
from mainspring.train.remote import RemoteError, RemoteTrainer


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def daemon(tmp_path):
    root = tmp_path / "traind"
    files = root / "files"
    files.mkdir(parents=True)
    token = load_or_create_token(root)
    runner = JobRunner(root)
    port = _free_port()
    httpd = ThreadingHTTPServer(("127.0.0.1", port),
                                make_handler(runner, files, token))
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    client = RemoteTrainer(f"http://127.0.0.1:{port}", token)
    try:
        yield client, root, files, token
    finally:
        runner.shutdown()
        httpd.shutdown()
        httpd.server_close()


# -- path safety ---------------------------------------------------------
@pytest.mark.parametrize("bad", [
    "../etc/passwd", "a/../../b", "/etc/passwd", "..", "a/../..",
])
def test_path_traversal_is_refused(tmp_path, bad):
    """'Upload a dataset' and 'overwrite anything on the box' must not differ
    only by how you spell the path."""
    with pytest.raises(DaemonError):
        safe_relpath(tmp_path, bad)


def test_symlink_escape_is_refused(tmp_path):
    """String-filtering '..' misses this one: a symlink inside the file root
    pointing outward is a write-anywhere primitive."""
    root = tmp_path / "files"
    root.mkdir()
    (root / "escape").symlink_to(tmp_path)
    with pytest.raises(DaemonError):
        safe_relpath(root, "escape/outside.bin")


def test_ordinary_nested_paths_are_allowed(tmp_path):
    got = safe_relpath(tmp_path, "data/packed/train.npy")
    assert got == (tmp_path / "data" / "packed" / "train.npy").resolve()


# -- auth ----------------------------------------------------------------
def test_health_needs_no_token(daemon):
    client, *_ = daemon
    assert client.health()["ok"] is True


def test_everything_else_requires_the_token(daemon):
    client, _, _, _ = daemon
    bad = RemoteTrainer(client.base_url, "not-the-token")
    with pytest.raises(RemoteError) as e:
        bad.jobs()
    assert "401" in str(e.value)


# -- jobs ----------------------------------------------------------------
def test_job_runs_and_reports_success(daemon):
    client, root, files, _ = daemon
    job = client.submit([sys.executable, "-c", "print('hello from the gpu box')"])
    done = client.wait(job["id"], poll_s=0.2, timeout_s=30)
    assert done["state"] == "done"
    assert done["returncode"] == 0
    assert "hello from the gpu box" in client.log(job["id"])


def test_failing_job_is_reported_as_failed(daemon):
    client, *_ = daemon
    job = client.submit([sys.executable, "-c", "import sys; sys.exit(3)"])
    done = client.wait(job["id"], poll_s=0.2, timeout_s=30)
    assert done["state"] == "failed"
    assert done["returncode"] == 3


def test_only_one_job_runs_at_a_time(daemon):
    """A GPU is not shareable: two jobs contend, both get slower, and the
    slowdown is silent. The second must queue, not compete."""
    client, *_ = daemon
    a = client.submit([sys.executable, "-c", "import time; time.sleep(2.5)"])
    b = client.submit([sys.executable, "-c", "print('second')"])
    time.sleep(0.8)
    assert client.job(a["id"])["state"] == "running"
    assert client.job(b["id"])["state"] == "queued"
    assert client.health()["busy"] is True
    client.wait(b["id"], poll_s=0.2, timeout_s=40)
    assert client.job(a["id"])["state"] == "done"


def test_stop_sends_sigterm_so_a_checkpointing_loop_can_save(daemon, tmp_path):
    """A loop that checkpoints on SIGTERM loses nothing when stopped. If the
    daemon sent SIGKILL, everything since the last checkpoint would be gone."""
    marker = tmp_path / "saved.txt"
    script = (
        "import signal, sys, time\n"
        f"open({str(marker)!r}, 'w').close\n"
        "def h(s, f):\n"
        f"    open({str(marker)!r}, 'w').write('checkpointed')\n"
        "    sys.exit(0)\n"
        "signal.signal(signal.SIGTERM, h)\n"
        "time.sleep(30)\n"
    )
    client, *_ = daemon
    job = client.submit([sys.executable, "-c", script])
    time.sleep(1.2)
    client.stop(job["id"])
    deadline = time.time() + 20
    while time.time() < deadline and not marker.exists():
        time.sleep(0.2)
    assert marker.exists(), "process never received SIGTERM"
    assert marker.read_text() == "checkpointed"


def test_stopping_a_queued_job_dequeues_it(daemon):
    client, *_ = daemon
    a = client.submit([sys.executable, "-c", "import time; time.sleep(2)"])
    b = client.submit([sys.executable, "-c", "print('never')"])
    time.sleep(0.5)
    client.stop(b["id"])
    assert client.job(b["id"])["state"] == "stopped"
    client.wait(a["id"], poll_s=0.2, timeout_s=30)
    assert "never" not in client.log(b["id"])


def test_empty_argv_is_rejected(daemon):
    client, *_ = daemon
    with pytest.raises(RemoteError):
        client.submit([])


def test_unknown_job_is_404(daemon):
    client, *_ = daemon
    with pytest.raises(RemoteError) as e:
        client.job("nope")
    assert "404" in str(e.value)


# -- metrics -------------------------------------------------------------
def test_metrics_stream_incrementally(daemon):
    client, root, _, _ = daemon
    job = client.submit([sys.executable, "-c", "import time; time.sleep(1.5)"])
    mp = root / "jobs" / job["id"] / "metrics.jsonl"
    mp.parent.mkdir(parents=True, exist_ok=True)
    mp.write_text(json.dumps({"step": 1}) + "\n")
    rows, nxt = client.metrics(job["id"])
    assert rows == [{"step": 1}] and nxt == 1
    with mp.open("a") as fh:
        fh.write(json.dumps({"step": 2}) + "\n")
    rows2, nxt2 = client.metrics(job["id"], since=nxt)
    assert rows2 == [{"step": 2}] and nxt2 == 2
    client.wait(job["id"], poll_s=0.2, timeout_s=30)


# -- files ---------------------------------------------------------------
def test_push_and_pull_round_trip(daemon, tmp_path):
    client, _, files, _ = daemon
    src = tmp_path / "data.bin"
    payload = bytes(range(256)) * 5000        # ~1.2MB
    src.write_bytes(payload)
    client.push(src, "data/packed/train.npy")
    assert (files / "data" / "packed" / "train.npy").read_bytes() == payload
    out = client.pull("data/packed/train.npy", tmp_path / "back.bin")
    assert out.read_bytes() == payload


def test_push_is_chunked_for_large_files(daemon, tmp_path):
    """A 2GB dataset over a home network gets interrupted; chunked transfer is
    what makes resuming possible at all."""
    client, _, files, _ = daemon
    src = tmp_path / "big.bin"
    src.write_bytes(b"x" * (20 << 20))        # 20MB -> multiple 8MB chunks
    seen: list[tuple[int, int]] = []
    client.push(src, "big.bin", on_progress=lambda a, b: seen.append((a, b)))
    assert len(seen) >= 3
    assert (files / "big.bin").stat().st_size == 20 << 20


def test_pull_of_a_missing_file_is_404(daemon, tmp_path):
    client, *_ = daemon
    with pytest.raises(RemoteError) as e:
        client.pull("no/such.bin", tmp_path / "x.bin")
    assert "404" in str(e.value)


def test_upload_cannot_escape_the_file_root(daemon, tmp_path):
    client, _, files, _ = daemon
    src = tmp_path / "evil.bin"
    src.write_bytes(b"pwn")
    with pytest.raises(RemoteError):
        client.push(src, "../escaped.bin")
    assert not (files.parent / "escaped.bin").exists()
