"""The peer-to-peer file fetcher refuses a remote path that leaves the files root."""

from __future__ import annotations

import pytest

from poolhouse.fleet.files import Fetcher
from poolhouse.fleet.jobs import DaemonError


@pytest.mark.parametrize("relpath", ["../../etc/passwd", "/etc/passwd", "a/../../b", "..%2f..%2fx"])
def test_a_remote_path_that_escapes_is_refused(tmp_path, relpath):
    fetcher = Fetcher(tmp_path, b"k")
    with pytest.raises(DaemonError):
        fetcher.start(source="peer", relpath=relpath, to="out.bin")
    assert fetcher.fetches == {}
