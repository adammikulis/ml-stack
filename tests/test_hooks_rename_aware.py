"""The pre-commit checks judge a renamed file against its old path, and still refuse what a rename adds."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
HOOKS = REPO / "scripts" / "hooks"
SITE = "def g():\n    try:\n        pass\n    except Exception:\n        pass\n"
GUARD_TEST = "def test_denied():\n    assert deny()\n    assert other()\n" + "".join(
    f"\n\ndef test_case_{i}():\n    assert case_{i}()\n" for i in range(8))


@pytest.fixture
def home(tmp_path):
    """A HOME and state directories of their own, so no hook reads or writes the real ones."""
    base = tmp_path / "home"
    base.mkdir()
    return {"HOME": str(base), "XDG_CONFIG_HOME": str(base / "c"), "XDG_STATE_HOME": str(base / "s"),
            "XDG_DATA_HOME": str(base / "d")}


class Repo:
    def __init__(self, root: Path, env: dict[str, str]) -> None:
        self.root = root
        self.env = {**os.environ, **env, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
                    "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
        self.env.pop("CLAUDECODE", None)
        root.mkdir()
        self.git("init", "-q", "-b", "0.3dev")

    def git(self, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run(["git", *args], cwd=self.root, env=self.env, text=True,
                              capture_output=True, check=True)

    def write(self, name: str, content: str | bytes) -> None:
        target = self.root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content if isinstance(content, bytes) else content.encode())
        self.git("add", name)

    def commit(self, message: str = "c") -> str:
        self.git("commit", "-qm", message, "--no-verify")
        return self.git("rev-parse", "HEAD").stdout.strip()

    def move(self, old: str, new: str) -> None:
        (self.root / new).parent.mkdir(parents=True, exist_ok=True)
        self.git("mv", old, new)

    def hook(self, name: str, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run([sys.executable, str(HOOKS / name), *args], cwd=self.root, env=self.env,
                              text=True, capture_output=True, check=False)


@pytest.fixture
def repo(tmp_path, home):
    return Repo(tmp_path / "r", home)


class TestNoDataFiles:
    def test_a_pure_rename_of_a_data_file_inside_a_data_directory_passes(self, repo):
        repo.write("src/ml_stack/data/fit.json", "{}")
        repo.commit()
        repo.move("src/ml_stack/data/fit.json", "src/poolhouse/data/fit.json")
        assert repo.hook("no-data-files").returncode == 0

    def test_a_pure_rename_with_an_edit_passes(self, repo):
        repo.write("src/ml_stack/data/fit.json", ''.join(f'{{"key{i}": {i}}}\n' for i in range(30)))
        repo.commit()
        repo.move("src/ml_stack/data/fit.json", "src/poolhouse/data/fit.json")
        repo.write("src/poolhouse/data/fit.json", ''.join(f'{{"key{i}": {i}}}\n' for i in range(30)) + '{"more": 1}\n')
        assert repo.hook("no-data-files").returncode == 0

    def test_a_rename_that_grows_the_file_over_the_limit_is_refused(self, repo):
        repo.write("src/ml_stack/data/fit.json", "x" * 100)
        repo.commit()
        repo.move("src/ml_stack/data/fit.json", "src/poolhouse/data/fit.json")
        repo.write("src/poolhouse/data/fit.json", "x" * 100 + "y" * ((1 << 20) + 1))
        done = repo.hook("no-data-files")
        assert done.returncode == 1 and "byte limit" in done.stderr

    def test_a_rename_into_a_data_extension_is_refused(self, repo):
        repo.write("src/ml_stack/data/fit.json", "{}")
        repo.commit()
        repo.move("src/ml_stack/data/fit.json", "src/poolhouse/data/fit.csv")
        done = repo.hook("no-data-files")
        assert done.returncode == 1 and ".csv" in done.stderr

    def test_a_new_data_file_is_refused(self, repo):
        repo.write("src/poolhouse/data/new.json", "{}")
        assert repo.hook("no-data-files").returncode == 1

    def test_a_data_file_moved_into_a_data_directory_from_elsewhere_is_refused(self, repo):
        repo.write("docs/people.json", "{}")
        repo.commit()
        repo.move("docs/people.json", "data/people.json")
        done = repo.hook("no-data-files")
        assert done.returncode == 1 and "data directory" in done.stderr

    def test_a_copy_is_not_a_rename(self, repo):
        repo.write("src/ml_stack/data/fit.json", '{"a": 1}\n' * 20)
        repo.commit()
        repo.write("src/poolhouse/data/fit.json", '{"a": 1}\n' * 20)
        assert repo.hook("no-data-files").returncode == 1

    def test_a_pushed_range_of_pure_renames_passes_and_one_with_a_new_data_file_does_not(self, repo):
        repo.write("src/ml_stack/data/fit.json", "{}")
        base = repo.commit()
        repo.move("src/ml_stack/data/fit.json", "src/poolhouse/data/fit.json")
        tip = repo.commit("rename")
        assert repo.hook("no-data-files", "--range", base, tip).returncode == 0
        repo.write("src/poolhouse/data/added.json", "{}")
        worse = repo.commit("adds")
        assert repo.hook("no-data-files", "--range", base, worse).returncode == 1

    def test_a_range_with_no_known_base_reads_each_commit_against_its_parent(self, repo):
        repo.write("src/ml_stack/data/fit.json", "{}")
        repo.commit()
        repo.move("src/ml_stack/data/fit.json", "src/poolhouse/data/fit.json")
        tip = repo.commit("rename")
        assert repo.hook("no-data-files", "--range", "0" * 40, tip).returncode == 0
        repo.move("src/poolhouse/data/fit.json", "data/fit.json")
        elsewhere = repo.commit("moves out of the package")
        assert repo.hook("no-data-files", "--range", "0" * 40, elsewhere).returncode == 0
        repo.write("docs/other.json", "{}")
        repo.commit("plain")
        repo.move("docs/other.json", "datasets/other.json")
        bad = repo.commit("moves in")
        assert repo.hook("no-data-files", "--range", "0" * 40, bad).returncode == 1


class TestBudgets:
    def start(self, repo):
        repo.write("pyproject.toml", (REPO / "pyproject.toml").read_text(encoding="utf-8"))
        repo.write("budgets.json", '{"broad-excepts": 9}\n')
        repo.write("src/ml_stack/a.py", SITE)
        repo.commit("base")

    def test_a_pure_rename_adds_no_site(self, repo):
        self.start(repo)
        repo.move("src/ml_stack/a.py", "src/poolhouse/a.py")
        done = repo.hook("budgets")
        assert done.returncode == 0, done.stderr

    def test_a_rename_that_adds_a_site_is_refused(self, repo):
        self.start(repo)
        repo.move("src/ml_stack/a.py", "src/poolhouse/a.py")
        repo.write("src/poolhouse/a.py", SITE + "\n\n" + SITE.replace("def g", "def h"))
        done = repo.hook("budgets")
        assert done.returncode == 1 and "gains 1 site" in done.stderr

    def test_a_new_file_with_a_site_is_refused(self, repo):
        self.start(repo)
        repo.write("src/poolhouse/b.py", SITE)
        assert repo.hook("budgets").returncode == 1

    def test_a_copy_with_a_site_is_refused(self, repo):
        self.start(repo)
        repo.write("src/poolhouse/a.py", SITE)
        assert repo.hook("budgets").returncode == 1

    def test_a_site_moved_in_from_a_file_that_is_not_python_is_refused(self, repo):
        self.start(repo)
        repo.write("notes.txt", SITE)
        repo.commit("text")
        repo.move("notes.txt", "src/poolhouse/c.py")
        assert repo.hook("budgets").returncode == 1

    def test_a_pure_rename_leaves_budgets_only_fall_quiet_and_a_raised_number_still_stops_it(self, repo):
        self.start(repo)
        shutil.copytree(REPO / "scripts" / "gates", repo.root / "scripts" / "gates")
        repo.move("src/ml_stack/a.py", "src/poolhouse/a.py")
        assert repo.hook("budgets-only-fall").returncode == 0
        repo.write("budgets.json", '{"broad-excepts": 10}\n')
        assert repo.hook("budgets-only-fall").returncode == 1


class TestWeakenedAssertions:
    GUARD = "tests/test_redteam_x.py"
    MOVED = "tests/test_redteam_y.py"

    def start(self, repo):
        repo.write(self.GUARD, GUARD_TEST)
        repo.commit("base")

    def run(self, repo):
        message = repo.root / ".msg"
        message.write_text("change\n")
        return repo.hook("weakened-assertions", "--staged", "--message", str(message))

    def test_a_pure_rename_of_a_guard_test_passes(self, repo):
        self.start(repo)
        repo.move(self.GUARD, self.MOVED)
        done = self.run(repo)
        assert done.returncode == 0, done.stderr

    def test_a_renamed_test_function_in_a_renamed_file_passes(self, repo):
        self.start(repo)
        repo.move(self.GUARD, self.MOVED)
        repo.write(self.MOVED, GUARD_TEST.replace("test_denied", "test_poolhouse_denied"))
        done = self.run(repo)
        assert done.returncode == 0, done.stderr

    def test_a_rename_that_drops_an_assertion_is_refused(self, repo):
        self.start(repo)
        repo.move(self.GUARD, self.MOVED)
        repo.write(self.MOVED, GUARD_TEST.replace("    assert other()\n", ""))
        done = self.run(repo)
        assert done.returncode == 1 and "fewer assertions" in done.stderr

    def test_a_rename_that_removes_a_test_is_refused(self, repo):
        self.start(repo)
        repo.move(self.GUARD, self.MOVED)
        repo.write(self.MOVED, GUARD_TEST.replace("def test_denied():", "def helper():"))
        done = self.run(repo)
        assert done.returncode == 1 and "test_denied removed" in done.stderr

    def test_a_guard_test_renamed_to_an_unguarded_name_is_still_judged(self, repo):
        self.start(repo)
        repo.move(self.GUARD, "tests/test_plain.py")
        repo.write("tests/test_plain.py", GUARD_TEST.replace("    assert other()\n", ""))
        assert self.run(repo).returncode == 1

    def test_a_renamed_name_in_a_file_that_is_not_renamed_is_still_reported(self, repo):
        self.start(repo)
        repo.write(self.GUARD, GUARD_TEST.replace("test_denied", "test_poolhouse_denied"))
        done = self.run(repo)
        assert done.returncode == 1 and "test_denied removed" in done.stderr

    def test_a_new_skip_added_in_a_rename_is_refused(self, repo):
        self.start(repo)
        repo.move(self.GUARD, self.MOVED)
        repo.write(self.MOVED, "import pytest\n\n\n" + "@" + "pytest.mark.skip" + "\n" + GUARD_TEST)
        assert self.run(repo).returncode == 1


def test_a_def_swap_without_a_hunk_of_its_own_is_not_excused_as_a_rename() -> None:
    """A merge's re-written diff has no hunks, so what it shows cannot be paired line for line."""
    import importlib.machinery
    import importlib.util

    loader = importlib.machinery.SourceFileLoader("_ml_stack_weakened_rename", str(HOOKS / "weakened-assertions"))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    hook = importlib.util.module_from_spec(spec)
    loader.exec_module(hook)
    head = ("diff --git a/tests/test_guard_a.py b/tests/test_guard_b.py\n"
            "rename from tests/test_guard_a.py\nrename to tests/test_guard_b.py\n")
    swap = "-def test_a():\n+def test_b():\n"
    assert hook.weakened(head + swap, lambda p: "")
    assert not hook.weakened(head + "@@ -1 +1 @@\n" + swap, lambda p: "")
