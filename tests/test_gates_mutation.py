"""The mutation checker: what it changes, what it runs, and what it counts."""

from __future__ import annotations

import ast
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from gates import _mutation, mutation_survivors  # noqa: E402
from gates._mutation import Target  # noqa: E402

SAMPLE = '''
def alive(pid):
    """Whether the pid is doing something."""
    if not pid or pid <= 0:
        return False
    return status(pid) != "zombie"
'''


def _mutants(source: str, qualname: str = "alive") -> list[_mutation.Mutant]:
    node = dict(_mutation.functions(ast.parse(source)))[qualname]
    return _mutation.mutants(source, Target("src/ml_stack/sample.py", qualname, node.lineno))


def test_every_mutation_is_a_change_that_still_parses() -> None:
    found = _mutants(SAMPLE)
    assert found
    for mutant in found:
        assert mutant.source != SAMPLE
        ast.parse(mutant.source)


def test_every_operator_reaches_this_function() -> None:
    kinds = {m.kind for m in _mutants(SAMPLE)}
    assert kinds == {"empty-body", "flip-comparison", "swap-boolop",
                     "negate-condition", "skip-branch", "constant-return"}


def test_a_comparison_is_flipped_not_removed() -> None:
    flipped = [m for m in _mutants(SAMPLE) if m.kind == "flip-comparison"]
    rendered = "\n".join(m.source for m in flipped)
    assert "pid > 0" in rendered and "status(pid) == 'zombie'" in rendered


def test_the_rest_of_the_file_is_left_alone() -> None:
    source = SAMPLE + "\n\ndef other():\n    return 1  # kept\n"
    for mutant in _mutants(source):
        assert "return 1  # kept" in mutant.source


def test_a_decorated_function_keeps_its_decorator_once() -> None:
    source = "import functools\n\n\n@functools.cache\ndef alive(pid):\n    return pid > 0\n"
    for mutant in _mutants(source):
        assert mutant.source.count("@functools.cache") == 1
        ast.parse(mutant.source)


def test_a_method_is_spliced_back_at_its_own_indentation() -> None:
    source = "class Server:\n\n    def alive(self, pid):\n        return pid > 0\n"
    for mutant in _mutants(source, "Server.alive"):
        assert "    def alive(self, pid):" in mutant.source
        ast.parse(mutant.source)


def test_a_branch_that_never_runs_is_not_a_site() -> None:
    source = ("from typing import TYPE_CHECKING\n\n\ndef alive(pid):\n"
              "    if TYPE_CHECKING:\n        pass\n    return pid > 0\n")
    assert not [m for m in _mutants(source) if m.kind in ("skip-branch", "negate-condition")]


def test_the_sample_is_the_same_for_the_same_seed() -> None:
    pool = _mutation.candidates(REPO)
    assert len(pool) > 100
    assert _mutation.sample(pool, 5, "abc") == _mutation.sample(pool, 5, "abc")
    assert _mutation.sample(pool, 5, "abc") != _mutation.sample(pool, 5, "def")


def test_the_tests_that_name_the_module_are_the_ones_that_run() -> None:
    target = Target("src/ml_stack/serve/process.py", "pid_exists", 15)
    assert _mutation.covering_tests(REPO, target, 2)[0] == "tests/test_serve.py"


def test_a_module_nothing_names_runs_nothing() -> None:
    target = Target("src/ml_stack/nowhere/absent.py", "qqzzxx", 1)
    assert _mutation.covering_tests(REPO, target, 2) == []


def _ledger(tmp_path: Path, rows: str) -> Path:
    (tmp_path / "scripts" / "gates").mkdir(parents=True)
    (tmp_path / mutation_survivors.LEDGER).write_text(rows, encoding="utf-8")
    where = tmp_path / "src" / "ml_stack"
    where.mkdir(parents=True)
    (where / "sample.py").write_text(SAMPLE, encoding="utf-8")
    return tmp_path


def test_a_recorded_survivor_is_counted_where_the_function_is(tmp_path) -> None:
    root = _ledger(tmp_path, "survivor\tsrc/ml_stack/sample.py::alive\tempty-body:0\tt.py\n")
    found = mutation_survivors.find(root)
    assert len(found) == 1
    assert found[0].path == "src/ml_stack/sample.py" and found[0].line == 2
    assert "empty-body:0" in found[0].detail


def test_a_survivor_whose_function_is_gone_is_not_counted(tmp_path) -> None:
    root = _ledger(tmp_path, "survivor\tsrc/ml_stack/sample.py::dead\tempty-body:0\tt.py\n")
    assert mutation_survivors.find(root) == []


def test_a_mutation_recorded_as_equivalent_is_not_a_survivor(tmp_path) -> None:
    root = _ledger(tmp_path,
                   "equivalent\tsrc/ml_stack/sample.py::alive\tempty-body:0\tnothing reads it\n")
    assert mutation_survivors.find(root) == []


def test_a_tree_with_no_ledger_counts_none(tmp_path) -> None:
    assert mutation_survivors.find(tmp_path) == []


def test_the_ledger_this_repository_holds_names_live_code() -> None:
    """A row nobody can act on is a row to take out."""
    for entry in mutation_survivors.entries(REPO):
        assert (REPO / entry.path).is_file(), f"{entry.site}: no such file"
        assert entry.mutation.split(":")[0] in _mutation.KINDS, entry.mutation


def _mini(root: Path, test_body: str) -> None:
    (root / "src" / "ml_stack" / "toy").mkdir(parents=True)
    (root / "src" / "ml_stack" / "toy" / "count.py").write_text(
        "def over(n):\n    if n > 3:\n        return 'many'\n    return 'few'\n", encoding="utf-8")
    (root / "tests").mkdir()
    (root / "tests" / "conftest.py").write_text(
        "import sys\nfrom pathlib import Path\n"
        "sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))\n",
        encoding="utf-8")
    (root / "tests" / "test_toy.py").write_text(test_body, encoding="utf-8")


@pytest.mark.slow
def test_a_test_that_asserts_nothing_lets_the_mutation_through(tmp_path) -> None:
    _mini(tmp_path, "from ml_stack.toy.count import over\n\n\ndef test_over():\n    over(9)\n")
    done = subprocess.run([sys.executable, str(REPO / "scripts" / "mutate"),
                           "--root", str(tmp_path), "--functions", "1", "--mutations", "9"],
                          capture_output=True, text=True, check=False)
    assert done.returncode == 1, done.stdout + done.stderr
    ran, survived = re.search(r"(\d+) mutants run, (\d+) survived", done.stdout).groups()
    assert ran == survived and int(ran) > 3, done.stdout
    assert "count.py" in done.stdout


@pytest.mark.slow
def test_a_test_that_reads_the_answer_catches_them_all(tmp_path) -> None:
    _mini(tmp_path, "from ml_stack.toy.count import over\n\n\ndef test_over():\n"
                    "    assert over(9) == 'many'\n    assert over(1) == 'few'\n")
    done = subprocess.run([sys.executable, str(REPO / "scripts" / "mutate"),
                           "--root", str(tmp_path), "--functions", "1", "--mutations", "9"],
                          capture_output=True, text=True, check=False)
    assert done.returncode == 0, done.stdout + done.stderr
    assert "0 survived" in done.stdout


@pytest.mark.slow
def test_the_tree_it_read_is_the_same_afterwards(tmp_path) -> None:
    """A run writes into a copy; src is byte for byte what it was."""
    _mini(tmp_path, "from ml_stack.toy.count import over\n\n\ndef test_over():\n    over(9)\n")
    source = tmp_path / "src" / "ml_stack" / "toy" / "count.py"
    was = source.read_bytes()
    subprocess.run([sys.executable, str(REPO / "scripts" / "mutate"),
                    "--root", str(tmp_path), "--functions", "1", "--mutations", "2"],
                   capture_output=True, text=True, check=False)
    assert source.read_bytes() == was
