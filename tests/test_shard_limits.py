"""Resource limits on the receiving side of a test shard: an inflation bomb and a runaway output."""

from __future__ import annotations

import gzip
import hashlib
import io
import sys
import tarfile
import time

import pytest

from ml_stack.fleet import shard_run, shard_tree


def packed(payload: bytes) -> tuple[bytes, str]:
    out = io.BytesIO()
    with tarfile.open(fileobj=out, mode="w:gz", compresslevel=9) as tar:
        info = tarfile.TarInfo("tests/test_ok.py")
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))
    data = out.getvalue()
    return data, hashlib.sha256(data).hexdigest()


def test_an_ordinary_tree_still_unpacks(tmp_path):
    data, digest = packed(b"def test_a():\n    assert 1\n")
    assert shard_tree.unpack(data, digest, tmp_path / "tree") == ["tests/test_ok.py"]


def test_a_small_archive_that_inflates_past_the_limit_is_refused_before_it_is_read(monkeypatch):
    monkeypatch.setattr(shard_tree, "MOST_UNPACKED", 1 << 20)
    monkeypatch.setattr(shard_tree, "MOST_MEMBERS", 1)
    data, digest = packed(b"\0" * (64 << 20))
    assert len(data) < 1 << 20 < 64 << 20  # a few dozen KiB of gzip that would inflate to 64 MiB
    began = time.monotonic()
    with pytest.raises(shard_tree.TreeError, match="inflates to more"):
        shard_tree.members(data, digest)
    assert time.monotonic() - began < 2


def test_two_gzip_streams_are_refused():
    data, _ = packed(b"x = 1\n")
    joined = data + gzip.compress(b"trailing")
    with pytest.raises(shard_tree.TreeError, match="more than one compressed stream"):
        shard_tree.members(joined, hashlib.sha256(joined).hexdigest())


def test_a_runner_that_floods_its_output_is_cancelled(tmp_path, monkeypatch):
    monkeypatch.setattr(shard_run, "MOST_LOG", 1 << 20)
    monkeypatch.setattr(shard_run, "POLL_S", 0.1)
    log = tmp_path / "output.log"
    flood = [sys.executable, "-c", "import sys\nwhile True:\n    sys.stdout.write('x' * 65536)\n    sys.stdout.flush()"]
    began = time.monotonic()
    code = shard_run.execute(flood, tmp_path, {}, 60, log)
    assert code == 124
    assert time.monotonic() - began < 30
    assert log.stat().st_size < 64 << 20


def test_a_runner_within_its_limits_reports_its_own_status(tmp_path):
    code = shard_run.execute([sys.executable, "-c", "raise SystemExit(3)"], tmp_path, {}, 60, tmp_path / "o.log")
    assert code == 3
