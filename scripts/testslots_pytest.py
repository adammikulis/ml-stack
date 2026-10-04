"""Pytest collection handoff and per-test shared CPU admission."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest
import testslots


def _control() -> Path:
    return Path(os.environ["DEV_TEST_PYTEST_CONTROL"])


def _ready() -> None:
    control = _control()
    temporary = control / "ready.tmp"
    temporary.write_text(os.environ["DEV_TEST_PYTEST_TOKEN"])
    temporary.replace(control / "ready")


def pytest_configure(config):
    if "DEV_TEST_PYTEST_CONTROL" not in os.environ:
        raise pytest.UsageError("testslots_pytest requires the testslots supervisor")
    if not hasattr(config, "workerinput"):
        workers = getattr(config.option, "numprocesses", 0) or 0
        if not isinstance(workers, int) or workers > int(os.environ["DEV_TEST_WORKERS"]):
            raise pytest.UsageError("xdist workers exceed the admitted collection budget")
        config._testslots_collected = set()


@pytest.hookimpl(optionalhook=True)
def pytest_xdist_node_collection_finished(node, ids):
    config = node.config
    config._testslots_collected.add(node.gateway.id)
    if len(config._testslots_collected) == config.option.numprocesses:
        _ready()


def pytest_collection_finish(session):
    config = session.config
    if not hasattr(config, "workerinput") and not getattr(config.option, "numprocesses", 0):
        _ready()


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_protocol(item, nextitem):
    control = _control()
    token = os.environ["DEV_TEST_PYTEST_TOKEN"]
    deadline = time.monotonic() + float(os.environ.get("DEV_TEST_WAIT_S", "3600"))
    while not (control / "admitted").exists():
        if time.monotonic() > deadline:
            raise TimeoutError("testslots: collection handoff timed out")
        time.sleep(0.02)
    if json.loads((control / "admitted").read_text())["token"] != token:
        raise RuntimeError("testslots: invalid collection handoff")
    with testslots.lease(1, 1, label=item.nodeid):
        yield
