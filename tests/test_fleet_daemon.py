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
import tempfile
import threading
import time
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest
from ml_stack.fleet.daemon import (
    DIGEST_HEADER,
    DaemonError,
    JobRunner,
    device_report,
    load_or_create_token,
    make_handler,
    resolve_report,
    safe_relpath,
    stdlib_device_report,
)
from ml_stack.fleet.remote import PeerError, Peer, sha256_file


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
    client = Peer(f"http://127.0.0.1:{port}", token)
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
    bad = Peer(client.base_url, "not-the-token")
    with pytest.raises(PeerError) as e:
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


@pytest.mark.slow


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


@pytest.mark.slow


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
    with pytest.raises(PeerError):
        client.submit([])


def test_unknown_job_is_404(daemon):
    client, *_ = daemon
    with pytest.raises(PeerError) as e:
        client.job("nope")
    assert "404" in str(e.value)


# -- metrics -------------------------------------------------------------
def test_metrics_stream_incrementally(daemon):
    client, _root, files, _ = daemon
    job = client.submit([sys.executable, "-c", "import time; time.sleep(1.5)"])
    mp = files / "jobs" / job["id"] / "metrics.jsonl"
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
    with pytest.raises(PeerError) as e:
        client.pull("no/such.bin", tmp_path / "x.bin")
    assert "404" in str(e.value)


def test_upload_cannot_escape_the_file_root(daemon, tmp_path):
    client, _, files, _ = daemon
    src = tmp_path / "evil.bin"
    src.write_bytes(b"pwn")
    with pytest.raises(PeerError):
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
    with pytest.raises(PeerError, match="delete it and pull again"):
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
    with pytest.raises(PeerError, match="suffix ranges"):
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
    with pytest.raises(PeerError, match="checksum mismatch"):
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
    with pytest.raises(PeerError, match="checksum mismatch"):
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
    with pytest.raises(PeerError, match="checksum mismatch"):
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


# -- slots ---------------------------------------------------------------
@pytest.fixture
def multi_daemon(tmp_path):
    """A daemon with room for three jobs at once, the way a prep box is configured."""
    root = tmp_path / "traind"
    files = root / "files"
    files.mkdir(parents=True)
    token = load_or_create_token(root)
    runner = JobRunner(root, slots=3)
    port = _free_port()
    httpd = ThreadingHTTPServer(("127.0.0.1", port),
                                make_handler(runner, files, token))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    client = Peer(f"http://127.0.0.1:{port}", token)
    try:
        yield client, runner
    finally:
        runner.shutdown()
        httpd.shutdown()
        httpd.server_close()


def _await(predicate, timeout=10.0, interval=0.05):
    end = time.time() + timeout
    while time.time() < end:
        if predicate():
            return True
        time.sleep(interval)
    return False


def test_slots_let_several_jobs_run_at_once(multi_daemon):
    """The one-job rule exists for GPU contention. A box tokenizing text has none, and
    serialising four shards onto it wastes three quarters of the machine."""
    client, runner = multi_daemon
    for i in range(3):
        client.submit([sys.executable, "-c", "import time; time.sleep(3)"], name=f"j{i}")

    assert _await(lambda: runner.status()["free"] == 0), runner.status()
    status = runner.status()
    assert len(status["running"]) == 3
    assert status["slots"] == 3


@pytest.mark.slow


def test_stopping_one_job_does_not_kill_its_neighbour(multi_daemon):
    """The specific failure: with a single `_current` handle, "the running process" is
    not a thing, and stopping job B reaches for whatever ran last -- killing job A."""
    client, runner = multi_daemon
    keep = client.submit([sys.executable, "-c", "import time; time.sleep(4)"], name="keep")
    doomed = client.submit([sys.executable, "-c", "import time; time.sleep(4)"], name="doomed")

    assert _await(lambda: len(runner.status()["running"]) == 2)
    client.stop(doomed["id"])

    assert _await(lambda: client.job(doomed["id"])["state"] == "stopped")
    assert client.job(keep["id"])["state"] == "running"
    assert _await(lambda: client.job(keep["id"])["state"] == "done", timeout=15)


def test_the_default_is_still_one_job_at_a_time(daemon):
    """Raising the limit must be a decision someone made, not a default they inherited."""
    client, *_ = daemon
    assert client.health()["slots"] == 1


def test_a_slot_count_below_one_is_refused(tmp_path):
    with pytest.raises(DaemonError, match="at least 1"):
        JobRunner(tmp_path, slots=0)


def test_listing_jobs_survives_a_submit_landing_mid_poll(multi_daemon):
    """Iterating runner.jobs directly races a submit and raises 'dictionary changed size
    during iteration', which reaches the caller as a 500 on a route that cannot fail."""
    client, _ = multi_daemon
    stop = threading.Event()
    errors: list[Exception] = []

    def submit_forever():
        while not stop.is_set():
            try:
                client.submit([sys.executable, "-c", ""], name="churn")
            except Exception as exc:                  # noqa: BLE001
                errors.append(exc)
            time.sleep(0.005)

    writer = threading.Thread(target=submit_forever, daemon=True)
    writer.start()
    try:
        for _ in range(60):
            client.jobs()
    finally:
        stop.set()
        writer.join(timeout=5)
    assert not errors, errors


# -- transfer edge cases -------------------------------------------------
def test_pushing_an_empty_file_works(daemon, tmp_path):
    """A prep shard that filtered to nothing is an empty file, and it is a real result.
    The upload loop never runs for zero bytes, so the reply was read before it was set."""
    client, _root, files, _token = daemon
    empty = tmp_path / "empty.bin"
    empty.write_bytes(b"")

    client.push(empty, "empty.bin")

    assert (files / "empty.bin").exists()
    assert (files / "empty.bin").read_bytes() == b""


def test_resuming_an_upload_whose_part_vanished_is_refused(daemon, tmp_path):
    """Seeking past the end of a fresh file writes a run of zeros, which then fails the
    digest check as 'checksum mismatch' -- sending the uploader after a corruption that
    never happened. The truth is that there is nothing here to resume from."""
    client, _root, files, _token = daemon
    with pytest.raises(PeerError) as exc:
        client._request("PUT", "/files/gap.bin", data=b"tail",
                        headers={"Content-Range": "bytes 4096-4099/4100",
                                 "X-ML-Stack-Complete": "1"})
    assert "416" in str(exc.value)
    assert "resume" in str(exc.value)
    assert not (files / "gap.bin.part").exists()


def test_head_reports_size_and_digest_without_sending_the_body(daemon):
    """Data-locality scoring asks "do you already hold this exact file". Answering that
    by downloading it defeats the purpose of asking."""
    client, _root, files, _token = daemon
    (files / "known.bin").write_bytes(b"x" * 5000)

    status, body, headers = client._request("HEAD", "/files/known.bin")

    assert status == 200
    assert body == b""
    assert headers["Content-Length"] == "5000"
    assert headers[DIGEST_HEADER] == sha256_file(files / "known.bin")


# -- the device report seam ----------------------------------------------
def test_the_stdlib_report_says_nothing_about_accelerators(tmp_path):
    """A device-tier package cannot import a framework to ask. Guessing is worse than
    silence: placement reads "no GPU" as "not a training box", so an unverified negative
    would park every run on the wrong machine."""
    out = stdlib_device_report()

    assert out["cpus"] >= 1
    assert out["arch"]
    assert out["backends"] == []
    assert "cuda" not in out
    assert "gpu" not in out


def test_a_richer_probe_is_merged_over_the_stdlib_facts():
    merged = device_report(lambda: {"cuda": True, "gpu": "RTX 3090 Ti"})

    assert merged["gpu"] == "RTX 3090 Ti"
    assert merged["cpus"] >= 1, "the stdlib facts must survive the merge"


def test_a_probe_that_raises_costs_detail_not_the_daemon():
    def boom() -> dict:
        raise RuntimeError("no driver")

    out = device_report(boom)

    assert out["cpus"] >= 1


def test_a_report_spec_resolves_to_its_callable():
    fn = resolve_report("ml_stack.fleet.daemon:stdlib_device_report")
    assert fn is stdlib_device_report


@pytest.mark.parametrize("spec, why", [
    ("no_colon_here", "expected"),
    ("ml_stack.fleet.daemon:nope", "cannot load"),
    ("ml_stack.nonexistent:report", "cannot load"),
])
def test_a_bad_report_spec_says_what_is_wrong(spec, why):
    """A typo in a systemd unit must fail at boot with the reason, not silently leave a
    GPU box advertising itself as a bare CPU."""
    with pytest.raises(DaemonError, match=why):
        resolve_report(spec)


def test_the_daemon_advertises_what_the_probe_reports(tmp_path):
    """End to end: a box with a card is one that was given a probe that can see it."""
    root = tmp_path / "traind"
    (root / "files").mkdir(parents=True)
    token = load_or_create_token(root)
    runner = JobRunner(root)
    port = _free_port()
    httpd = ThreadingHTTPServer(
        ("127.0.0.1", port),
        make_handler(runner, root / "files", token, "rtx",
                     lambda: device_report(lambda: {"cuda": True, "gpu": "RTX 3090 Ti"})))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        health = Peer(f"http://127.0.0.1:{port}", token).health()
    finally:
        runner.shutdown()
        httpd.shutdown()
        httpd.server_close()

    assert health["gpu"] == "RTX 3090 Ti"
    assert health["cuda"] is True
    assert health["slots"] == 1


def test_a_job_can_write_something_the_coordinator_can_actually_pull(daemon):
    """The README's own example -- pull("jobs/<id>/ckpt/...") -- used to 404, because
    job directories sat BESIDE the file root rather than under it. A checkpoint you
    waited three hours for being unreachable is a bad way to learn that."""
    client, _root, _files, _ = daemon
    job = client.submit([sys.executable, "-c",
                         "import os, pathlib; "
                         "d = pathlib.Path(os.environ['ML_STACK_JOB_DIR']) / 'ckpt'; "
                         "d.mkdir(parents=True, exist_ok=True); "
                         "(d / 'model.safetensors').write_bytes(b'weights')"])
    client.wait(job["id"], poll_s=0.1, timeout_s=30)

    got = client.pull(f"jobs/{job['id']}/ckpt/model.safetensors",
                      Path(tempfile.mkdtemp()) / "model.safetensors")

    assert got.read_bytes() == b"weights"


def test_a_job_is_told_where_fetchable_is(daemon):
    """Without ML_STACK_FILES_ROOT a job has to guess, and a caller who passes cwd
    makes the guess wrong -- landing the artifact where nothing can see it."""
    client, _root, files, _ = daemon
    job = client.submit([sys.executable, "-c",
                         "import os, pathlib; "
                         "pathlib.Path(os.environ['ML_STACK_OUT']).mkdir(parents=True, exist_ok=True); "
                         "(pathlib.Path(os.environ['ML_STACK_OUT']) / 'where.txt')"
                         ".write_text(os.environ['ML_STACK_FILES_ROOT'])"])
    client.wait(job["id"], poll_s=0.1, timeout_s=30)

    said = (files / "jobs" / job["id"] / "out" / "where.txt").read_text()
    assert Path(said).resolve() == files.resolve()


def test_a_job_runs_in_the_file_root_by_default(daemon):
    """A job that does not name a working directory must land where pushed files are,
    or every relative path a caller gives it is wrong."""
    client, _root, files, _ = daemon
    (files / "pushed.txt").write_text("here")
    job = client.submit([sys.executable, "-c",
                         "import pathlib,os; "
                         "pathlib.Path(os.environ['ML_STACK_OUT']).mkdir(parents=True, exist_ok=True); "
                         "(pathlib.Path(os.environ['ML_STACK_OUT'])/'saw.txt')"
                         ".write_text(pathlib.Path('pushed.txt').read_text())"])
    client.wait(job["id"], poll_s=0.1, timeout_s=30)

    assert client.job(job["id"])["state"] == "done", client.log(job["id"])
    assert (files / "jobs" / job["id"] / "out" / "saw.txt").read_text() == "here"



class TestWhatAMachineIsDoing:
    """The card showed what a machine is, never what it was doing."""

    def test_the_report_carries_memory_in_use_and_how_busy_it_is(self):
        from ml_stack.fleet.daemon import stdlib_device_report

        got = stdlib_device_report()
        assert got["ram_gb"] > 0
        assert "ram_used_gb" in got, got
        assert 0 <= got["ram_used_gb"] <= got["ram_gb"]
        assert "cpu_pct" in got, got
        assert 0 <= got["cpu_pct"] <= 100

    def test_the_first_reading_does_not_come_from_psutil(self, monkeypatch):
        """psutil's first cpu_percent has nothing to compare against and answers
        0.0, which would report a working machine as asleep."""
        import sys

        import ml_stack.fleet.daemon as mod

        psutil = sys.modules.get("psutil")
        if psutil is None:
            import psutil                             # noqa: PLC0415
        monkeypatch.setattr(psutil, "cpu_percent", lambda **k: 0.0)
        monkeypatch.setattr(mod.os, "getloadavg", lambda: (8.0, 8.0, 8.0))
        monkeypatch.setattr(mod.os, "cpu_count", lambda: 16)

        was, mod._cpu_primed = mod._cpu_primed, False
        try:
            first = mod._cpu_busy_pct()
            second = mod._cpu_busy_pct()
        finally:
            mod._cpu_primed = was

        assert first == 50.0, f"the first reading was psutil's 0.0, not the load ({first})"
        assert second == 0.0, "after priming it should be psutil's own number"

    def test_it_still_answers_without_psutil(self, monkeypatch):
        """ml-stack-fleet declares no dependencies; psutil is a bonus."""
        import builtins

        import ml_stack.fleet.daemon as mod

        real = builtins.__import__

        def no_psutil(name, *a, **k):
            if name == "psutil":
                raise ImportError("not installed")
            return real(name, *a, **k)

        monkeypatch.setattr(builtins, "__import__", no_psutil)
        got = mod.stdlib_device_report()
        assert got["ram_gb"] > 0
        assert got.get("cpu_pct") is not None
        assert got.get("ram_used_gb") is not None

    def test_windows_gets_its_memory_from_the_system_call(self, monkeypatch):
        """os.sysconf does not exist there, so a Windows machine reported no memory
        at all -- not just no usage, no total either."""
        import ml_stack.fleet.daemon as mod

        def no_sysconf(*a, **k):
            raise AttributeError("no sysconf on this platform")

        import builtins

        real = builtins.__import__

        def no_psutil(name, *a, **k):
            if name == "psutil":
                raise ImportError("not installed")
            return real(name, *a, **k)

        monkeypatch.setattr(builtins, "__import__", no_psutil)
        monkeypatch.setattr(mod.os, "sysconf", no_sysconf)
        monkeypatch.setattr(mod, "_windows_memory_gb", lambda: (16.0, 6.5))

        assert mod._total_ram_gb() == 16.0
        assert mod._ram_used_gb(16.0) == 6.5

    def test_the_windows_call_is_not_made_on_anything_else(self):
        import sys

        import ml_stack.fleet.daemon as mod

        if sys.platform != "win32":
            assert mod._windows_memory_gb() is None

    def test_memory_in_use_is_never_more_than_there_is(self):
        from ml_stack.fleet.daemon import stdlib_device_report

        got = stdlib_device_report()
        assert got["ram_used_gb"] <= got["ram_gb"]
