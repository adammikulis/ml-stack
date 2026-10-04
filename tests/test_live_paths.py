"""The real-state-root guard tolerates other processes' logs and nothing credential-like."""

from __future__ import annotations

import pytest
from conftest import changed_files, file_mtimes


def touch(root, rel):
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x")


@pytest.mark.parametrize("rel", ["workspace/bus.jsonl", "activity/u-501/activity.log",
                                 "activity/u-501/activity.log.head", "sentinel/events.log",
                                 "sentinel/anchor.log", "harness/abc/.tmp/x"])
def test_a_log_other_agents_append_to_is_not_a_write_by_this_run(tmp_path, rel):
    before = file_mtimes(tmp_path)
    touch(tmp_path, rel)
    assert changed_files(before, file_mtimes(tmp_path)) == []


@pytest.mark.parametrize("rel", ["keystore/blob", "sentinel/manifest.json", "sentinel/events.log.head.key",
                                 "activity/u-501/activity.log.head.key", "sentinel/honey.json",
                                 "credentials.toml", "requests/inbox.enc", "cluster.key"])
def test_anything_credential_like_is_still_guarded(tmp_path, rel):
    before = file_mtimes(tmp_path)
    touch(tmp_path, rel)
    assert changed_files(before, file_mtimes(tmp_path)) == [rel]
