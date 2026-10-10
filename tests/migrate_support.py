"""The account fixture and helpers the migrate tests share."""

from __future__ import annotations

from pathlib import Path

import pytest

from poolhouse import home, migrate
from poolhouse.keystore import Keystore, Wires
from poolhouse.sentinel.events import Bus
from tests.onboard_support import Clock


@pytest.fixture
def account(tmp_path, monkeypatch):
    """A fresh account home, with no state root or cache root named by the environment, and no process
    of the old name alive."""
    for name in (home.ROOT_ENV, home.CACHE_ENV):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(home, "user_home", lambda: tmp_path)
    monkeypatch.setattr(migrate.process, "command_lines", lambda: [])
    monkeypatch.setattr(migrate.process, "running_within", lambda _path: [])
    return tmp_path


def old_state(account: Path) -> Path:
    root = account / ".ml-stack"
    (root / "workspace" / "tokens").mkdir(parents=True)
    (root / "workspace" / "tokens" / "claude-1").write_text("secret", encoding="utf-8")
    (root / "machine-id").write_text("abc123", encoding="utf-8")
    return root


def ring(tmp_path: Path):
    wires = Wires(clock=Clock(), say=lambda _t: None, bus=Bus(), interactive=lambda: True, sleep=lambda _s: None)
    return lambda: Keystore(directory=tmp_path / "ks", wires=wires)
