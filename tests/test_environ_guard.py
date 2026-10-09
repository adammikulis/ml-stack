"""A marker one test sets in-process is gone for the next test (tests/environ_guard.py)."""

from __future__ import annotations

import os

import pytest

from poolhouse.keystore import ENV_NONINTERACTIVE
from poolhouse.workspace.identity import Denied, Registry


def test_a_test_that_sets_the_agent_marker_in_process_sets_it():
    os.environ[ENV_NONINTERACTIVE] = "1"
    os.environ["POOLHOUSE_TEST_LEAK"] = "1"
    assert os.environ[ENV_NONINTERACTIVE] == "1"


def test_the_next_test_does_not_see_it_and_can_register_a_person(tmp_path):
    assert ENV_NONINTERACTIVE not in os.environ
    assert "POOLHOUSE_TEST_LEAK" not in os.environ
    assert Registry(tmp_path / "agents.json").init("someone")


def test_a_real_agent_marker_is_still_refused(tmp_path):
    with pytest.raises(Denied):
        Registry(tmp_path / "agents.json").init("someone", env={ENV_NONINTERACTIVE: "1"})
