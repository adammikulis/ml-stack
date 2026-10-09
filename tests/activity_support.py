"""Fixtures for the activity-log tests: an isolated home, a fake keyring, no agent markers."""

from __future__ import annotations

import pytest

from ml_stack.activity import writer
from ml_stack.sentinel.human import AGENT_MARKERS
from tests import memory_keys

ring = memory_keys.ring
CANARY = "canary-subject-7f3a9c1e"


def reset() -> None:
    with writer._LOCK:
        writer._LOGS.clear()
        writer._STATE.update(session="", dropped=0, owed=0, told=set())


@pytest.fixture
def person(monkeypatch, ring):
    """A person's process: no agent marker, the log on, a fresh per-process state."""
    for name in (*AGENT_MARKERS, writer.ENV_OFF, writer.ENV_ACTOR, writer.ENV_SESSION,
                 writer.ENV_RETENTION):
        monkeypatch.delenv(name, raising=False)
    reset()
    yield ring
    reset()


def entries(log=None):
    from ml_stack.activity.schema import Entry
    held = log or writer.log()
    return [e for e in held.entries() if isinstance(e, Entry)]
