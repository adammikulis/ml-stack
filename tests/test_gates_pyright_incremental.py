"""The incremental pyright count equals a whole-project run after every kind of edit."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from gates import _imports, _pyright_incremental, _store, pyright_errors  # noqa: E402

PYPROJECT = """[tool.pyright]
include = ["src/poolhouse"]
typeCheckingMode = "off"
reportUndefinedVariable = "error"
reportUnusedCoroutine = "error"
"""
FILES = {
    "__init__.py": "",
    "core.py": "async def fetch():\n    return 1\n",
    "a.py": "from poolhouse.core import fetch\n\n\ndef run():\n    fetch()\n",
    "b.py": "from poolhouse import a\n\n\ndef go():\n    return a.run()\n",
    "d.py": "from . import core\n\n\ndef again():\n    core.fetch()\n",
    "leaf.py": "from poolhouse.b import go\n\nresult = go()\n",
    "c0.py": "def broken():\n    return nowhere_c0\n",
    **{f"c{n}.py": f"VALUE_{n} = {n}\n" for n in range(1, 7)},
}
PACKAGE = "src/poolhouse"


@pytest.fixture
def project(tmp_path, monkeypatch):
    if pyright_errors.skip():
        pytest.skip(pyright_errors.skip())
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path / "cache"))
    monkeypatch.delenv(_store.FORCE, raising=False)
    root = tmp_path / "project"
    (root / PACKAGE).mkdir(parents=True)
    (root / "pyproject.toml").write_text(PYPROJECT, encoding="utf-8")
    for name, text in FILES.items():
        (root / PACKAGE / name).write_text(text, encoding="utf-8")
    return root


def rows(found) -> list[tuple]:
    return sorted((f.path, f.line, f.detail) for f in found)


class Asked:
    """Runs pyright for a project and remembers which files each run was asked about."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.runs: list = []

    def findings(self) -> list:
        def run(files):
            self.runs.append(None if files is None else sorted(files))
            return pyright_errors.run(self.root, files)

        return _pyright_incremental.findings(self.root, run)

    def agrees_with_a_full_run(self) -> bool:
        return rows(self.findings()) == rows(pyright_errors.find_full(self.root))

    def files_checked(self) -> list[str]:
        return sorted({f for run in self.runs for f in (run or ["<whole project>"])})


def edit(root: Path, name: str, text: str) -> None:
    (root / PACKAGE / name).write_text(text, encoding="utf-8")


def named(*names: str) -> list[str]:
    return sorted(f"{PACKAGE}/{n}" for n in names)


def test_the_first_run_checks_everything_and_the_second_nothing(project):
    asked = Asked(project)
    first = asked.findings()
    assert asked.runs == [None]
    assert len(first) == 3
    asked.runs.clear()
    assert rows(asked.findings()) == rows(first)
    assert asked.runs == []


def test_an_edit_to_a_leaf_file_checks_only_that_file(project):
    asked = Asked(project)
    asked.findings()
    asked.runs.clear()
    edit(project, "leaf.py", FILES["leaf.py"] + "print(undefined_leaf)\n")
    assert asked.agrees_with_a_full_run()
    assert asked.files_checked() == named("leaf.py")
    assert any(f.path.endswith("leaf.py") for f in asked.findings())


def test_an_edit_to_a_widely_imported_file_checks_every_importer(project):
    asked = Asked(project)
    asked.findings()
    asked.runs.clear()
    edit(project, "core.py", "def fetch():\n    return 1\n")
    assert asked.agrees_with_a_full_run()
    assert asked.files_checked() == named("core.py", "a.py", "b.py", "d.py", "leaf.py")
    assert not any(f.path.endswith(("a.py", "d.py")) for f in asked.findings())


def test_deleting_a_file_rechecks_what_imported_it(project):
    asked = Asked(project)
    asked.findings()
    asked.runs.clear()
    (project / PACKAGE / "a.py").unlink()
    assert asked.agrees_with_a_full_run()
    assert asked.files_checked() == named("b.py", "leaf.py")


def test_a_new_file_that_an_old_one_imports_is_picked_up(project):
    asked = Asked(project)
    edit(project, "late.py", "")
    edit(project, "c1.py", "from poolhouse import late\n")
    asked.findings()
    asked.runs.clear()
    edit(project, "late.py", "async def wait():\n    return 1\n")
    edit(project, "c1.py", "from poolhouse import late\n\n\ndef f():\n    late.wait()\n")
    assert asked.agrees_with_a_full_run()
    assert any(f.path.endswith("c1.py") for f in asked.findings())


def test_a_run_forced_full_checks_the_whole_project(project, monkeypatch):
    asked = Asked(project)
    asked.findings()
    asked.runs.clear()
    monkeypatch.setenv(_store.FORCE, "1")
    asked.findings()
    assert asked.runs == [None]


@pytest.mark.parametrize("damage", ["not json", "[1, 2]", '{"environment": "other", "files": 3}'])
def test_a_damaged_or_foreign_cache_falls_back_to_a_full_run(project, damage):
    asked = Asked(project)
    wanted = rows(asked.findings())
    (kept,) = _store.directory().glob("pyright-*.json")
    kept.write_text(damage, encoding="utf-8")
    asked.runs.clear()
    assert rows(asked.findings()) == wanted
    assert asked.runs == [None]


def test_a_changed_pyright_configuration_falls_back_to_a_full_run(project):
    asked = Asked(project)
    asked.findings()
    asked.runs.clear()
    (project / "pyproject.toml").write_text(PYPROJECT + "reportUnusedExcept = \"error\"\n",
                                            encoding="utf-8")
    asked.findings()
    assert asked.runs == [None]


def test_a_configuration_the_cache_cannot_follow_runs_the_whole_project(project):
    asked = Asked(project)
    (project / "pyproject.toml").write_text(PYPROJECT + 'exclude = ["src/poolhouse/c0.py"]\n',
                                            encoding="utf-8")
    asked.findings()
    asked.findings()
    assert asked.runs == [None, None]


def test_imports_resolve_absolute_relative_and_submodule_forms(project):
    files = _imports.Resolver(project.resolve())
    importer = (project / PACKAGE / "d.py").resolve()
    here = (project / PACKAGE).resolve()
    assert files.resolve((".", 1, ("core",)), importer) == {here / "__init__.py", here / "core.py"}
    assert files.resolve(("poolhouse.core", 0, ()), importer) == {here / "__init__.py", here / "core.py"}
    assert files.resolve(("os.path", 0, ()), importer) == set()
