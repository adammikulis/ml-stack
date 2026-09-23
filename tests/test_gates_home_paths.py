"""What home-paths counts: a home directory resolved in place, or the state root as a literal."""

from __future__ import annotations

from pathlib import Path

import pytest
from gates import home_paths

REPO = Path(__file__).resolve().parent.parent


def _found(tmp_path: Path, source: str, name: str = "one.py") -> list[str]:
    where = tmp_path / "src" / "ml_stack" / name
    where.parent.mkdir(parents=True, exist_ok=True)
    where.write_text(source, encoding="utf-8")
    return [f.detail for f in home_paths.find(tmp_path)]


@pytest.mark.parametrize("line", [
    'ROOT = "~/.ml-stack/traind"',
    'def f(root="~/.ml-stack/bench"):\n    return root',
    'where = Path.cwd() / ".ml-stack"',
    'ap.add_argument("--root", default="~/.ml-stack")',
])
def test_a_literal_state_path_is_found(tmp_path, line: str) -> None:
    assert _found(tmp_path, line + "\n")


@pytest.mark.parametrize("line", [
    'HELP = "the key file (default: ~/.ml-stack/cluster.key)"',
    'def f():\n    """Kept under ``~/.ml-stack/bench``."""',
    'ROOT = state("traind")',
])
def test_prose_naming_the_path_is_left_alone(tmp_path, line: str) -> None:
    assert not _found(tmp_path, line + "\n")


def test_the_module_that_resolves_paths_may_name_them(tmp_path) -> None:
    assert not _found(tmp_path, 'ROOT = ".ml-stack"\n', name="home.py")


def test_the_library_holds_none() -> None:
    assert [(f.path, f.line, f.detail) for f in home_paths.find(REPO)] == []
