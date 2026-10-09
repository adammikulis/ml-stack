"""The fixed cost of a session stays small on this machine's real state root."""

from __future__ import annotations

import time

from conftest import file_mtimes

from ml_stack import home

STATE_FILES_AT_MOST = 5_000
SNAPSHOT_SECONDS_AT_MOST = 3.0


def test_the_real_state_root_snapshot_is_small_and_quick():
    began = time.monotonic()
    snapshot = file_mtimes(home.home(), attribute_external=False)
    elapsed = time.monotonic() - began
    assert len(snapshot) <= STATE_FILES_AT_MOST, "the real state root grew a directory the snapshot should skip"
    assert elapsed < SNAPSHOT_SECONDS_AT_MOST
