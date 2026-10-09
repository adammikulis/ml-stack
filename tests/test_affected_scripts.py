"""`scripts/affected.py` places a changed script, a lowered budget and a generated file narrowly.

Each case builds a small repository on disk, so what is selected comes from real files and real
`git show` output; the last cases replay recent commits of this repository.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import affected  # noqa: E402
import affected_rules as rules  # noqa: E402

FILES = {
    "scripts/alpha.py": "def one():\n    return 1\n",
    "scripts/beta.py": "import alpha\n\n\ndef two():\n    return alpha.one()\n",
    "scripts/runner": "#!/usr/bin/env python3\nprint('run')\n",
    "scripts/hooks/guard": "#!/bin/sh\nexit 0\n",
    "scripts/wrapper.sh": "#!/bin/sh\nexec ./runner \"$@\"\n",
    "scripts/launcher": "#!/usr/bin/env python3\nimport alpha\n",
    "scripts/orphan.py": "VALUE = 1\n",
    "scripts/data.txt": "nothing reads this\n",
    "tests/kit.py": "import subprocess\nRUNNER = 'scripts/runner'\n\n\ndef go():\n    subprocess.run([RUNNER])\n",
    "tests/test_import.py": "import alpha\n\n\ndef test_it():\n    assert alpha.one()\n",
    "tests/test_from.py": "from alpha import one\n\n\ndef test_it():\n    assert one()\n",
    "tests/test_string.py": "def test_it():\n    assert 'scripts/runner' and 'scripts/alpha.py'\n",
    "tests/test_subprocess.py": (
        "import subprocess\nimport sys\nfrom pathlib import Path\nSCRIPTS = Path('x') / 'scripts'\n\n\n"
        "def test_it():\n    subprocess.run([sys.executable, str(SCRIPTS / 'runner')], check=True)\n"),
    "tests/test_guard.py": "import subprocess\n\n\ndef test_it():\n    subprocess.run(['scripts/hooks/guard'])\n",
    "tests/test_launcher.py": "def test_it():\n    assert 'scripts/launcher'\n",
    "tests/test_wrapper.py": "def test_it():\n    assert 'scripts/wrapper.sh'\n",
    "tests/test_helper.py": "import kit\n\n\ndef test_it():\n    kit.go()\n",
    "tests/test_chain.py": "import beta\n\n\ndef test_it():\n    assert beta.two()\n",
    "tests/test_doc.py": 'def test_it():\n    """Mentions scripts/runner in prose only."""\n',
    "tests/test_unrelated.py": "def test_it():\n    assert 'test' == 'test'\n",
    "tests/test_lister.py": "from pathlib import Path\n\n\ndef test_it():\n    assert list(Path('scripts').glob('*'))\n",
    "tests/test_budgets.py": "def test_it():\n    assert 1\n",
    "tests/test_reads.py": "import json\n\n\ndef test_it():\n    json.load(open('budgets.json'))\n",
    "tests/test_gen.py": "def test_it():\n    assert 'coverage.json'\n",
    "tests/conftest.py": "",
    "budgets.json": json.dumps({"a": 5, "b": 3, "floors": {"n": 10}}),
    "docs/redteam/coverage.json": "{}\n",
    "scripts/redteam_coverage.py": "print('gen')\n",
    "tests/test_cover.py": "def test_it():\n    assert 'scripts/redteam_coverage.py'\n",
}


@pytest.fixture
def toy(tmp_path: Path) -> Path:
    for rel, text in FILES.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return tmp_path


def chosen(root: Path, *changed: str, **kw) -> set[str]:
    out = affected.select(root, list(changed), **kw)
    return set(out.files) - {"tests/test_conftest_guard.py"}


def test_a_script_is_selected_by_the_tests_that_import_it_either_way(toy) -> None:
    got = chosen(toy, "scripts/alpha.py")
    assert {"tests/test_import.py", "tests/test_from.py"} <= got


def test_a_script_is_selected_by_the_tests_that_name_its_path_or_build_it(toy) -> None:
    assert "tests/test_string.py" in chosen(toy, "scripts/alpha.py")
    assert {"tests/test_string.py", "tests/test_subprocess.py"} <= chosen(toy, "scripts/runner")


def test_an_extensionless_script_in_a_folder_is_found_by_its_path(toy) -> None:
    assert "tests/test_guard.py" in chosen(toy, "scripts/hooks/guard")


def test_a_helper_that_names_the_script_carries_it_to_the_tests_that_use_the_helper(toy) -> None:
    got = chosen(toy, "scripts/runner")
    assert "tests/test_helper.py" in got and "tests/test_chain.py" not in got


def test_a_script_another_script_imports_reaches_the_tests_of_the_importer(toy) -> None:
    assert "tests/test_chain.py" in chosen(toy, "scripts/alpha.py")


def test_an_extensionless_python_script_that_imports_the_changed_one_carries_it_to_its_tests(toy) -> None:
    assert "tests/test_launcher.py" in chosen(toy, "scripts/alpha.py")


def test_a_shell_script_that_runs_the_changed_one_carries_it_to_the_tests_of_the_shell_script(toy) -> None:
    assert "tests/test_wrapper.py" in chosen(toy, "scripts/runner")


def test_tests_that_enumerate_scripts_follow_any_script_change(toy) -> None:
    assert "tests/test_lister.py" in chosen(toy, "scripts/orphan.py")


def test_a_script_selects_nothing_it_cannot_reach(toy) -> None:
    got = chosen(toy, "scripts/runner")
    assert not got & {"tests/test_import.py", "tests/test_unrelated.py", "tests/test_doc.py"}


def test_a_script_no_test_reaches_is_unmapped_unless_it_is_python(toy) -> None:
    out = affected.select(toy, ["scripts/data.txt"])
    assert out.unmapped == ["scripts/data.txt"] and "no test" in out.reasons["scripts/data.txt"]
    assert affected.select(toy, ["scripts/orphan.py"]).unmapped == []


def test_a_deleted_script_runs_everything(toy) -> None:
    out = affected.select(toy, ["scripts/runner"], frozenset({"scripts/runner"}))
    assert out.unmapped == ["scripts/runner"] and "deleted" in out.reasons["scripts/runner"]


def test_a_script_the_conftest_imports_runs_everything_and_says_why(toy) -> None:
    (toy / "tests/conftest.py").write_text("import alpha\n", encoding="utf-8")
    out = affected.select(toy, ["scripts/alpha.py"])
    assert out.unmapped == ["scripts/alpha.py"] and "every test session" in out.reasons["scripts/alpha.py"]
    assert affected.select(toy, ["scripts/runner"]).unmapped == []


def test_a_plugin_the_runner_loads_runs_everything_and_so_does_what_it_imports(toy) -> None:
    (toy / "scripts/test").write_text("ARGS = ['-p', 'plug']\n", encoding="utf-8")
    (toy / "scripts/plug.py").write_text("import beta\n", encoding="utf-8")
    assert affected.select(toy, ["scripts/plug.py"]).unmapped == ["scripts/plug.py"]
    assert affected.select(toy, ["scripts/beta.py"]).unmapped == ["scripts/beta.py"]
    assert affected.select(toy, ["scripts/orphan.py"]).unmapped == []


def test_a_name_a_helper_holds_does_not_make_a_script_session_wide(toy) -> None:
    (toy / "tests/conftest.py").write_text("import kit\n", encoding="utf-8")
    assert affected.select(toy, ["scripts/runner"]).unmapped == []


def run_git(root: Path, *args: str) -> None:
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args], cwd=root,
                   check=True, capture_output=True)


@pytest.fixture
def repo(toy: Path) -> Path:
    run_git(toy, "init", "-q", "-b", "base")
    run_git(toy, "add", ".")
    run_git(toy, "commit", "-q", "-m", "base")
    run_git(toy, "checkout", "-q", "-b", "work")
    return toy


def write_budgets(root: Path, value: object) -> None:
    (root / "budgets.json").write_text(json.dumps(value), encoding="utf-8")


def test_a_budget_that_only_fell_selects_the_budget_tests(repo) -> None:
    write_budgets(repo, {"a": 4, "b": 3, "floors": {"n": 9}})
    out = affected.select(repo, ["budgets.json"], base="base")
    assert out.unmapped == [] and {"tests/test_budgets.py", "tests/test_reads.py"} <= out.files
    assert "tests/test_import.py" not in out.files


@pytest.mark.parametrize("after", [
    {"a": 6, "b": 3, "floors": {"n": 10}},
    {"a": 5, "b": 3, "floors": {"n": 11}},
    {"a": 5, "b": 3, "floors": {"n": 10}, "c": 1},
    {"a": 5, "b": 3},
    {"a": 4.5, "b": 3, "floors": {"n": 10}},
    {"a": True, "b": 3, "floors": {"n": 10}},
    {"a": "4", "b": 3, "floors": {"n": 10}},
])
def test_a_budget_that_rose_or_changed_shape_runs_everything(repo, after) -> None:
    write_budgets(repo, after)
    out = affected.select(repo, ["budgets.json"], base="base")
    assert out.unmapped == ["budgets.json"] and "budget" in out.reasons["budgets.json"]


def test_a_budget_with_no_base_to_compare_against_runs_everything(repo) -> None:
    write_budgets(repo, {"a": 1, "b": 1, "floors": {"n": 1}})
    assert affected.select(repo, ["budgets.json"], base="nowhere").unmapped == ["budgets.json"]


def test_a_generated_file_selects_the_tests_that_check_it_and_its_generator(toy) -> None:
    got = chosen(toy, "docs/redteam/coverage.json")
    assert {"tests/test_gen.py", "tests/test_cover.py"} <= got
    assert "tests/test_import.py" not in got


def test_a_generated_file_nothing_checks_is_left_to_the_gate(toy) -> None:
    (toy / "tests/test_gen.py").unlink()
    (toy / "tests/test_cover.py").unlink()
    out = affected.select(toy, ["docs/redteam/coverage.json"])
    assert out.unmapped == [] and out.ignored == ["docs/redteam/coverage.json"]


def test_an_unknown_file_type_still_runs_everything_and_says_why(toy) -> None:
    out = affected.select(toy, ["contracts/shape.bin"])
    assert out.unmapped == ["contracts/shape.bin"] and out.reasons["contracts/shape.bin"]
    assert affected.select(toy, ["pyproject.toml"]).reasons["pyproject.toml"]


def recent_scripts(count: int) -> list[str]:
    """Scripts that still exist and were touched by the latest commits that touched scripts/."""
    done = subprocess.run(["git", "log", "-n", "400", "--no-renames", "--name-only", "--format=@@%h",
                           "--", "scripts/"], cwd=REPO, capture_output=True, text=True, check=False)
    commits = [c.splitlines()[1:] for c in done.stdout.split("@@") if c.strip()]
    sampled = commits[:: max(1, len(commits) // count)][:count]
    found = {p for c in sampled for p in c if p.startswith("scripts/") and "/gates/" not in p
             and (REPO / p).is_file() and not p.endswith((".md", ".json", ".txt"))}
    return sorted(found)


def plain_oracle(script: str) -> set[str]:
    """Test files that import the script or put its path in quotes, found with plain regexes."""
    stem = Path(script).stem if script.endswith(".py") else Path(script).name
    imports = re.compile(rf"^\s*(?:import|from)\s+{re.escape(stem)}\b", re.M)
    quoted = re.compile(rf"""["']{re.escape(script)}(?![\w.-])""")
    out = set()
    for path in sorted((REPO / "tests").glob("test_*.py")):
        text = path.read_text(encoding="utf-8", errors="replace")
        if (script.endswith(".py") and imports.search(text)) or quoted.search(text):
            out.add(f"tests/{path.name}")
    return out


def test_every_test_a_plain_search_finds_for_a_recently_changed_script_is_selected() -> None:
    scripts = recent_scripts(30)
    assert len(scripts) >= 5
    files = rules.corpus(REPO)
    missed = {}
    for script in scripts:
        why, _, shared = rules.script_tests(files, script)
        if not shared and (gap := plain_oracle(script) - set(why)):
            missed[script] = sorted(gap)
    assert not missed
