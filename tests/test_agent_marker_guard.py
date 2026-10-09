"""The suite clears agent markers for every test, and the init guard still refuses a marker a test sets."""
from __future__ import annotations

import os

import pytest
from testagentenv_pytest import AGENT_VARIABLES

from ml_stack.workspace import Denied, Workspace


def test_no_agent_marker_reaches_a_test_by_default():
    assert [name for name in AGENT_VARIABLES if os.environ.get(name)] == []


def test_init_works_without_a_marker(tmp_path):
    assert Workspace(tmp_path / "ws").init("owner")


@pytest.mark.parametrize("marker", ["CLAUDECODE", "ML_STACK_AGENT", "ML_STACK_NONINTERACTIVE"])
def test_init_still_refuses_a_marker_the_test_sets(tmp_path, monkeypatch, marker):
    monkeypatch.setenv(marker, "1")
    with pytest.raises(Denied, match=marker):
        Workspace(tmp_path / "ws").init("owner")
