"""A gate that cannot run says so and fails, naming the command that fixes it.

Found by a cold start in a clone outside $HOME: pyenv's shim for ruff exits 127 with a message
that names the versions holding it, the budgets read "3.13.5" out of that message as ruff's
version, and the run passed with five metrics uncounted.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import stat
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import gates  # noqa: E402
from gates import _pins  # noqa: E402


def budgets_script():
    loader = importlib.machinery.SourceFileLoader("budgets_script", str(REPO / "scripts" / "budgets"))
    spec = importlib.util.spec_from_loader("budgets_script", loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


def test_a_shim_with_no_version_here_is_not_the_tool(tmp_path, monkeypatch) -> None:
    shim = tmp_path / "ruff"
    shim.write_text("#!/bin/sh\necho 'pyenv: ruff: command not found; it exists in 3.13.5' >&2\nexit 127\n")
    shim.chmod(shim.stat().st_mode | stat.S_IXUSR)
    assert not _pins.runs((str(shim),))
    monkeypatch.setattr(_pins, "tool", lambda name: (str(shim),))
    _pins.version.cache_clear()
    try:
        assert _pins.version("ruff") == ""
    finally:
        _pins.version.cache_clear()


def test_a_run_that_could_not_count_a_metric_fails(monkeypatch, capsys) -> None:
    script = budgets_script()
    monkeypatch.setattr(gates, "unrunnable", lambda: {"ruff-other": "ruff is not installed; " + _pins.INSTALL})
    assert script.table({}, {"ruff-other": 3}) == 1
    out = capsys.readouterr().out
    assert "FAIL: 1 metric(s) were not counted" in out and _pins.INSTALL in out
