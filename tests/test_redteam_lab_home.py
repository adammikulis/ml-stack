"""The red-team lab attacks with forged traffic, plants decoys and rotates keys: none of that may reach the real home."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from poolhouse import home
from poolhouse.redteam.lab import lab, own_home

REPO = Path(__file__).resolve().parent.parent


def test_the_lab_runs_in_its_own_home_and_gives_the_old_one_back(tmp_path, monkeypatch):
    mine = tmp_path / "mine"
    monkeypatch.setenv(home.ROOT_ENV, str(mine))
    with lab() as made:
        inside = home.home()
        assert inside != mine and made.root in inside.parents
    assert home.home() == mine and os.environ[home.ROOT_ENV] == str(mine)


def test_own_home_removes_the_variable_it_set_when_there_was_none(tmp_path, monkeypatch):
    monkeypatch.delenv(home.ROOT_ENV, raising=False)
    with own_home(tmp_path / "x") as inside:
        assert os.environ[home.ROOT_ENV] == str(inside)
    assert home.ROOT_ENV not in os.environ


def test_the_red_team_command_writes_nothing_under_the_users_home(tmp_path):
    """The real regression: run the command the way a person does, with no POOLHOUSE_HOME and a throwaway HOME."""
    fake = tmp_path / "home"
    fake.mkdir()
    env = {**os.environ, "HOME": str(fake), "PYTHONPATH": str(REPO / "src")}
    env.pop(home.ROOT_ENV, None)
    done = subprocess.run([sys.executable, "-m", "poolhouse.redteam", "run", "--scenarios", "fleet", "--model", "stub"],
                          env=env, capture_output=True, text=True, timeout=300, cwd=tmp_path)
    assert done.returncode in (0, 1), done.stderr[-400:]
    assert not (fake / ".poolhouse").exists(), sorted(p.name for p in (fake / ".poolhouse").iterdir())
