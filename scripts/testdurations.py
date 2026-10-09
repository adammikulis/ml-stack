"""Pytest plugin: records the CPU and wall seconds of every passing test into the shared duration history."""
from __future__ import annotations

import os
import time
from pathlib import Path

import pytest
import testhistory


def _now() -> tuple[float, float]:
    t = os.times()
    return time.perf_counter(), t.user + t.system + t.children_user + t.children_system


def _store() -> Path | None:
    raw = os.environ.get("DEV_TEST_HISTORY")
    return Path(raw) if raw else None


_STATE: dict = {"samples": {}, "bad": set(), "collected": {}}


def pytest_configure(config):
    _STATE.update(samples={}, bad=set(), collected={})
    config._durations = _STATE


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config, items):
    collected = config._durations["collected"]
    for item in items:
        collected.setdefault(item.nodeid.split("::", 1)[0], set()).add(item.nodeid)


def pytest_runtest_logreport(report):
    if report.outcome != "passed":
        _STATE["bad"].add(report.nodeid)


@pytest.hookimpl(hookwrapper=True, trylast=True)
def pytest_runtest_protocol(item, nextitem):
    wall0, cpu0 = _now()
    yield
    wall1, cpu1 = _now()
    item.config._durations["samples"][item.nodeid] = (cpu1 - cpu0, wall1 - wall0)


def pytest_sessionfinish(session):
    store, state = _store(), session.config._durations
    if store is None or not (state["samples"] or state["collected"]):
        return
    samples = {node: v for node, v in state["samples"].items() if node not in state["bad"]}
    testhistory.merge(store, Path(session.config.rootpath), samples, state["collected"])
