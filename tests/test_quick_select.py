"""`scripts/test quick` picks test files from the import graph, and runs everything when it cannot.

Each case builds a small repository on disk, so what is selected comes from real files and
real `git diff` output.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import affected  # noqa: E402

FILES = {
    "src/ml_stack/alpha.py": "def one():\n    return 1\n",
    "src/ml_stack/beta.py": "from ml_stack import alpha\n\n\ndef two():\n    return alpha.one()\n",
    "src/ml_stack/gamma.py": "def three():\n    from ml_stack.beta import two\n    return two()\n",
    "src/ml_stack/delta.py": "def four():\n    return 4\n",
    "src/ml_stack/pack/__init__.py": "",
    "src/ml_stack/pack/leaf.py": "VALUE = 1\n",
    "src/ml_stack/pack/table.json": "{}\n",
    "tests/test_alpha.py": "from ml_stack.alpha import one\n\n\ndef test_one():\n    assert one()\n",
    "tests/test_gamma.py": "from ml_stack import gamma\n\n\ndef test_three():\n    assert gamma.three()\n",
    "tests/test_delta.py": "def test_run(): ...\n\n\nARGS = ['-m', 'ml_stack.delta']\n",
    "tests/test_leaf.py": "from ml_stack.pack import leaf\n\n\ndef test_v():\n    assert leaf.VALUE\n",
    "tests/test_conftest_guard.py": "def test_nothing(): ...\n",
    "tests/conftest.py": "",
    "docs/notes.md": "notes\n",
    "README.md": "hello\n",
    "pyproject.toml": "[project]\nname = 'toy'\n",
    "contracts/shape.json": "{}\n",
}


@pytest.fixture
def toy(tmp_path: Path) -> Path:
    for rel, text in FILES.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return tmp_path


def chosen(root: Path, *changed: str, **kw) -> set[str]:
    return set(affected.select(root, list(changed), **kw).files) - {"tests/test_conftest_guard.py"}


def test_a_change_selects_the_tests_that_import_it(toy) -> None:
    assert chosen(toy, "src/ml_stack/alpha.py") >= {"tests/test_alpha.py"}


def test_a_lazy_import_inside_a_function_counts_and_the_walk_stops_at_the_depth(toy) -> None:
    assert "tests/test_gamma.py" not in chosen(toy, "src/ml_stack/beta.py")
    assert "tests/test_gamma.py" in chosen(toy, "src/ml_stack/beta.py", depth=2)
    assert "tests/test_gamma.py" not in chosen(toy, "src/ml_stack/alpha.py", depth=2)
    assert "tests/test_gamma.py" in chosen(toy, "src/ml_stack/alpha.py", depth=3)


def test_a_module_named_only_in_a_string_selects_the_test_that_names_it(toy) -> None:
    assert chosen(toy, "src/ml_stack/delta.py") == {"tests/test_delta.py"}


def test_a_change_selects_nothing_it_cannot_reach(toy) -> None:
    assert "tests/test_leaf.py" not in chosen(toy, "src/ml_stack/alpha.py", depth=3)


def test_a_data_file_selects_the_tests_of_its_package(toy) -> None:
    assert chosen(toy, "src/ml_stack/pack/table.json") == {"tests/test_leaf.py"}


def test_a_changed_test_selects_itself(toy) -> None:
    assert chosen(toy, "tests/test_alpha.py") == {"tests/test_alpha.py"}


@pytest.mark.parametrize("path", ["pyproject.toml", "tests/conftest.py", "contracts/shape.json"])
def test_a_shared_or_unplaced_path_leaves_the_caller_to_run_everything(toy, path) -> None:
    out = affected.select(toy, [path])
    assert out.unmapped == [path]


def test_prose_changes_select_nothing_and_are_not_unmapped(toy) -> None:
    out = affected.select(toy, ["README.md", "docs/notes.md"])
    assert out.unmapped == [] and not out.files and sorted(out.ignored) == [
        "README.md", "docs/notes.md"]


def test_a_deleted_module_is_unmapped(toy) -> None:
    out = affected.select(toy, ["src/ml_stack/alpha.py"], frozenset({"src/ml_stack/alpha.py"}))
    assert out.unmapped == ["src/ml_stack/alpha.py"]


def run_git(root: Path, *args: str) -> None:
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args], cwd=root,
                   check=True, capture_output=True)


def test_changed_since_sees_commits_the_working_tree_and_new_files(toy) -> None:
    run_git(toy, "init", "-q", "-b", "base")
    run_git(toy, "add", ".")
    run_git(toy, "commit", "-q", "-m", "base")
    run_git(toy, "checkout", "-q", "-b", "work")
    (toy / "src/ml_stack/alpha.py").write_text("def one():\n    return 2\n", encoding="utf-8")
    run_git(toy, "commit", "-qam", "edit")
    (toy / "src/ml_stack/delta.py").write_text("def four():\n    return 5\n", encoding="utf-8")
    (toy / "src/ml_stack/fresh.py").write_text("X = 1\n", encoding="utf-8")
    (toy / "README.md").unlink()
    changed, gone = affected.changed_since(toy, "base")
    assert changed == ["README.md", "src/ml_stack/alpha.py", "src/ml_stack/delta.py",
                       "src/ml_stack/fresh.py"]
    assert gone == {"README.md"}
