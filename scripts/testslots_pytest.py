"""Pytest collection handoff and per-test shared CPU admission."""
from __future__ import annotations

import os
from pathlib import Path

import pytest
import testslots
import testslots_rpc


def _ready(operation: str) -> None:
    with testslots_rpc.request(operation):
        pass


def pytest_configure(config):
    if "DEV_TEST_PYTEST_ENDPOINT" not in os.environ:
        raise pytest.UsageError("testslots_pytest requires the testslots supervisor")
    if not hasattr(config, "workerinput"):
        workers = getattr(config.option, "numprocesses", 0) or 0
        if not isinstance(workers, int) or workers > int(os.environ["DEV_TEST_WORKERS"]):
            raise pytest.UsageError("xdist workers exceed the admitted pool maximum")
        config._testslots_collected = set()
        _ready("ready")


@pytest.hookimpl(optionalhook=True)
def pytest_xdist_node_collection_finished(node, ids):
    config = node.config
    config._testslots_collected.add(node.gateway.id)
    if len(config._testslots_collected) == config.option.numprocesses:
        _ready("collected")


@pytest.hookimpl(hookwrapper=True)
def pytest_collection(session):
    with testslots_rpc.request("acquire", label="pytest collection", phase="collection"):
        yield


def pytest_collection_finish(session):
    config = session.config
    if not hasattr(config, "workerinput") and not getattr(config.option, "numprocesses", 0):
        _ready("collected")


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_protocol(item, nextitem):
    heavy = Path(str(item.path)).stem in testslots.HEAVY_MODULES
    with testslots_rpc.request("acquire", label=item.nodeid, heavy=heavy):
        yield
