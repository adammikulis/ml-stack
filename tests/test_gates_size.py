"""The two size gates: a file over the limit fails, and there is no allowance to record."""

from __future__ import annotations

import importlib.machinery
import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import gates  # noqa: E402
from gates import deep_components, deep_files  # noqa: E402


def plant(root: Path, name: str, lines: int) -> Path:
    """Write a file of exactly that many lines under src/ml_stack."""
    path = root / "src" / "ml_stack" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(f"x{i} = {i}" for i in range(lines)) + "\n", encoding="utf-8")
    return path


@pytest.mark.parametrize("checker,suffix", [(deep_files, ".py"), (deep_components, ".css")],
                         ids=lambda v: getattr(v, "NAME", v))
def test_a_file_at_the_limit_is_not_a_finding(checker, suffix, tmp_path) -> None:
    plant(tmp_path, f"exactly{suffix}", checker.LIMIT)
    assert checker.find(tmp_path) == []


@pytest.mark.parametrize("checker,suffix", [(deep_files, ".py"), (deep_components, ".css")],
                         ids=lambda v: getattr(v, "NAME", v))
def test_one_line_over_the_limit_is_a_finding_that_names_the_length(checker, suffix,
                                                                   tmp_path) -> None:
    plant(tmp_path, f"over{suffix}", checker.LIMIT + 1)
    found = checker.find(tmp_path)
    assert [f.path for f in found] == [f"src/ml_stack/over{suffix}"]
    assert found[0].detail == f"{checker.LIMIT + 1} lines"


@pytest.mark.parametrize("checker,suffix", [(deep_files, ".py"), (deep_components, ".css")],
                         ids=lambda v: getattr(v, "NAME", v))
def test_every_file_over_the_limit_is_counted(checker, suffix, tmp_path) -> None:
    for i in range(3):
        plant(tmp_path, f"deep{i}{suffix}", checker.LIMIT + 100)
    plant(tmp_path, f"shallow{suffix}", 10)
    assert len(checker.find(tmp_path)) == 3


@pytest.mark.parametrize("checker,suffix", [(deep_files, ".py"), (deep_components, ".css")],
                         ids=lambda v: getattr(v, "NAME", v))
def test_a_file_that_grows_past_the_limit_becomes_a_finding(checker, suffix, tmp_path) -> None:
    plant(tmp_path, f"growing{suffix}", checker.LIMIT)
    assert checker.find(tmp_path) == []
    grown = tmp_path / "grown"
    plant(grown, f"growing{suffix}", checker.LIMIT + 1)
    assert len(checker.find(grown)) == 1


@pytest.mark.parametrize("checker", [deep_files, deep_components], ids=lambda c: c.NAME)
def test_the_size_gates_allow_nothing(checker) -> None:
    """A budget is an allowance; these two grant none, so they carry no number."""
    assert checker.NAME in gates.hard()
    recorded = json.loads((REPO / "budgets.json").read_text(encoding="utf-8"))
    assert checker.NAME not in recorded


@pytest.mark.parametrize("checker", [deep_files, deep_components], ids=lambda c: c.NAME)
def test_the_limit_a_gate_counts_is_the_limit_the_edit_guard_refuses(checker) -> None:
    guard = REPO / "scripts" / "hooks" / "claude-edit-guard"
    loader = importlib.machinery.SourceFileLoader("_ml_stack_edit_guard", str(guard))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    limits = module.line_limits()
    named = {suffix: limit for suffix, (limit, metric) in limits.items()
             if metric == checker.NAME}
    assert named, f"the edit guard refuses nothing for {checker.NAME}"
    for suffix, limit in named.items():
        assert limit == checker.LIMIT, f"{suffix} refused at {limit}, counted at {checker.LIMIT}"
