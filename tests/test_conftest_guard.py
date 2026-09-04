"""The session guard over the real ml_stack cache."""

from __future__ import annotations

from conftest import truncated_logs

LOG = "llama-server-8080.log"


def test_a_log_that_grew_is_a_live_server():
    assert truncated_logs({LOG: (7, 10)}, {LOG: (7, 99)}) == []


def test_the_same_file_getting_shorter_is_a_truncation():
    assert truncated_logs({LOG: (7, 99)}, {LOG: (7, 10)}) == [LOG]


def test_a_restart_writes_a_new_file_under_the_same_name():
    assert truncated_logs({LOG: (7, 99)}, {LOG: (8, 10)}) == []


def test_a_new_log_is_another_server_starting():
    assert truncated_logs({}, {"llama-server-50085.log": (9, 4096)}) == []


def test_a_log_that_went_away_is_not_a_truncation():
    assert truncated_logs({LOG: (7, 99)}, {}) == []
