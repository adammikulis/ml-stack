"""The training daemon: real HTTP, real subprocesses, real files.

Nothing here is mocked. The failures this module exists to prevent are
process-shaped and transport-shaped -- a second job started on a busy GPU, a
SIGTERM that never arrives, an upload path that escapes its root -- and a
mocked socket or a fake Popen reproduces none of them.
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest
from ml_stack.train.daemon import (
    DaemonError,
    JobRunner,
    load_or_create_token,
    make_handler,
    safe_relpath,
)
from ml_stack.train.remote import RemoteError, RemoteTrainer, sha256_file


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


# -- file transfer: memory must not track file size ----------------------
def _peak_rss_mb(fn):
    """Run fn while sampling this process's RSS. Returns (result, delta_MB)."""
    import subprocess
    import threading

    def rss_mb() -> float:
        out = subprocess.run(["ps", "-o", "rss=", "-p", str(os.getpid())],
                             capture_output=True, text=True).stdout.strip()
        return int(out) / 1024

    base = rss_mb()
    peak = [base]
    stop = threading.Event()

    def watch():
        while not stop.is_set():
            peak[0] = max(peak[0], rss_mb())
            time.sleep(0.02)

    t = threading.Thread(target=watch, daemon=True)
    t.start()
    try:
        result = fn()
    finally:
        stop.set()
        t.join(timeout=2)
    return result, peak[0] - base


BIG_MB = 64


def test_downloading_does_not_load_the_file_into_memory(daemon, tmp_path):
    """A gigabyte checkpoint must not cost a gigabyte to serve.

    read_bytes() made memory a function of file size; on a resumed pull the
    extra slice made it twice the file size, on the same box that was running
    the training job the checkpoint came from.
    """
    client, root, files, _ = daemon
    big = files / "big.bin"
    big.write_bytes(os.urandom(1 << 20) * BIG_MB)

    out = tmp_path / "fresh.bin"
    _, grew = _peak_rss_mb(lambda: client.pull("big.bin", out))
    assert out.stat().st_size == BIG_MB << 20
    assert grew < BIG_MB / 2, f"a {BIG_MB}MB download grew RSS by {grew:.0f}MB"


def test_a_resumed_download_does_not_cost_twice_the_file(daemon, tmp_path):
    client, root, files, _ = daemon
    big = files / "big.bin"
    payload = os.urandom(1 << 20) * BIG_MB
    big.write_bytes(payload)

    out = tmp_path / "resumed.bin"
    # Interrupted at 1MB, exactly as a dropped transfer leaves things.
    part = out.with_suffix(out.suffix + ".part")
    part.write_bytes(payload[: 1 << 20])

    _, grew = _peak_rss_mb(lambda: client.pull("big.bin", out))
    assert out.read_bytes() == payload, "resume must reassemble the exact file"
    assert grew < BIG_MB / 2, f"a resumed {BIG_MB}MB download grew RSS by {grew:.0f}MB"


def test_progress_is_reported_while_downloading_not_after(daemon, tmp_path):
    """A callback that fires once, at the end, is not progress."""
    client, root, files, _ = daemon
    (files / "big.bin").write_bytes(os.urandom(1 << 20) * 32)
    seen: list[tuple[int, int]] = []
    client.pull("big.bin", tmp_path / "out.bin", on_progress=lambda d, t: seen.append((d, t)))
    assert len(seen) > 1, f"progress fired {len(seen)} time(s)"
    assert [d for d, _ in seen] == sorted(d for d, _ in seen), "must not go backwards"
    assert seen[-1][0] == seen[-1][1] == 32 << 20


def test_a_stale_oversized_part_is_refused_not_spliced(daemon, tmp_path):
    """A .part bigger than the remote file belongs to a different file."""
    client, root, files, _ = daemon
    (files / "small.bin").write_bytes(b"x" * 1000)
    out = tmp_path / "small.bin"
    out.with_suffix(".bin.part").write_bytes(b"y" * 5000)
    with pytest.raises(RemoteError, match="delete it and pull again"):
        client.pull("small.bin", out)
    assert not out.exists(), "a refused pull must not produce a file"


def test_a_part_that_is_already_complete_finishes_without_redownloading(daemon, tmp_path):
    client, root, files, _ = daemon
    payload = b"z" * 4096
    (files / "done.bin").write_bytes(payload)
    out = tmp_path / "done.bin"
    out.with_suffix(".bin.part").write_bytes(payload)
    assert client.pull("done.bin", out).read_bytes() == payload
    assert not out.with_suffix(".bin.part").exists()


def test_an_explicit_range_end_is_honoured(daemon, tmp_path):
    """bytes=start-end must return that window, not everything after start."""
    client, root, files, _ = daemon
    (files / "r.bin").write_bytes(bytes(range(256)))
    status, body, _ = client._request("GET", "/files/r.bin",
                                      headers={"Range": "bytes=10-19"})
    assert status == 206
    assert body == bytes(range(10, 20)), f"got {len(body)} bytes"


def test_a_suffix_range_is_refused_rather_than_mis_served(daemon):
    client, root, files, _ = daemon
    (files / "s.bin").write_bytes(b"abcdefghij")
    with pytest.raises(RemoteError, match="suffix ranges"):
        client._request("GET", "/files/s.bin", headers={"Range": "bytes=-5"})


# -- integrity -----------------------------------------------------------
def test_a_round_trip_declares_and_confirms_a_digest(daemon, tmp_path):
    client, root, files, _ = daemon
    src = tmp_path / "data.npy"
    src.write_bytes(os.urandom(1 << 20))
    reply = client.push(src, "data.npy")
    assert reply["sha256"] == sha256_file(src), "upload must report what it stored"
    out = client.pull("data.npy", tmp_path / "back.npy")
    assert out.read_bytes() == src.read_bytes()


def test_an_upload_that_does_not_match_its_digest_is_discarded(daemon, tmp_path):
    """The bytes arrived intact by luck or not at all -- either way, refuse."""
    client, root, files, _ = daemon
    payload = b"the real payload" * 64
    with pytest.raises(RemoteError, match="checksum mismatch"):
        client._request("PUT", "/files/bad.bin", data=payload, headers={
            "Content-Range": f"bytes 0-{len(payload)-1}/{len(payload)}",
            "X-ML-Stack-Complete": "1",
            "X-ML-Stack-SHA256": "00" * 32,
        })
    assert not (files / "bad.bin").exists(), "a rejected upload must leave nothing"
    assert not (files / "bad.bin.part").exists(), \
        "the partial must go too: resuming from corrupt bytes keeps them forever"


def test_a_correct_digest_on_upload_is_accepted(daemon, tmp_path):
    client, root, files, _ = daemon
    payload = b"the real payload" * 64
    status, body, _ = client._request("PUT", "/files/good.bin", data=payload, headers={
        "Content-Range": f"bytes 0-{len(payload)-1}/{len(payload)}",
        "X-ML-Stack-Complete": "1",
        "X-ML-Stack-SHA256": hashlib.sha256(payload).hexdigest().upper(),
    })
    assert status == 200, body
    assert (files / "good.bin").read_bytes() == payload


def test_a_download_whose_bytes_do_not_match_the_digest_is_discarded(daemon, tmp_path):
    """Corruption the length check cannot see: right size, wrong bytes.

    Rewritten in place with size and mtime preserved, so the daemon still
    advertises the digest of the original -- which is exactly the shape of a
    file that rotted on disk, or of bytes mangled in transit.
    """
    client, root, files, _ = daemon
    target = files / "ckpt.bin"
    original = os.urandom(4096)
    target.write_bytes(original)

    first = client.pull("ckpt.bin", tmp_path / "first.bin")
    assert first.read_bytes() == original

    st = target.stat()
    target.write_bytes(b"\x00" * len(original))       # same length, different bytes
    os.utime(target, ns=(st.st_atime_ns, st.st_mtime_ns))

    out = tmp_path / "second.bin"
    with pytest.raises(RemoteError, match="checksum mismatch"):
        client.pull("ckpt.bin", out)
    assert not out.exists(), "a corrupt download must not be promoted"
    assert not out.with_suffix(".bin.part").exists()


def test_a_resumed_download_verifies_the_bytes_it_did_not_fetch(daemon, tmp_path):
    """The prefix came from an earlier call; no single response vouches for it."""
    client, root, files, _ = daemon
    payload = os.urandom(8192)
    (files / "resume.bin").write_bytes(payload)
    out = tmp_path / "resume.bin"
    # A .part whose length is plausible but whose content is wrong.
    out.with_suffix(".bin.part").write_bytes(b"\xff" * 2048)
    with pytest.raises(RemoteError, match="checksum mismatch"):
        client.pull("resume.bin", out)
    assert not out.exists()


def test_the_digest_cache_notices_a_rewritten_file(daemon, tmp_path):
    client, root, files, _ = daemon
    target = files / "moving.bin"
    target.write_bytes(b"a" * 100)
    assert client.pull("moving.bin", tmp_path / "a.bin").read_bytes() == b"a" * 100
    time.sleep(0.01)
    target.write_bytes(b"b" * 200)           # different size and mtime
    assert client.pull("moving.bin", tmp_path / "b.bin").read_bytes() == b"b" * 200
