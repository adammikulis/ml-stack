"""What the checkout needs from a machine is declared where an install and CI read it, and a gap is reported with its fix.

The `gates` CI job failed twice because tests/conftest.py gained imports (psutil, then keyring and
cryptography) that the job's install line did not name; a fresh `pip install '.[test]'` could not
collect at all because conftest imports numpy. Both are the same drift: the files a purpose runs
import something nothing installs. These tests derive what is imported and compare it with what
pyproject and the workflow install.
"""

from __future__ import annotations

import re
import subprocess
import sys
from importlib import metadata
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import envcheck  # noqa: E402

GATE_JOB_ENTRIES = ("tests/conftest.py", "tests/test_budgets.py", "tests/test_wiring.py", "tests/test_layers.py",
                    "scripts/budgets")


def requires_of(name: str, seen: set[str]) -> set[str]:
    """``name`` and everything it requires (extras left out), by canonical name, for what is installed here."""
    key = envcheck.canonical(name)
    if key in seen:
        return seen
    seen.add(key)
    try:
        needs = metadata.requires(name) or []
    except metadata.PackageNotFoundError:
        return seen
    for text in needs:
        if "extra ==" not in text:
            requires_of(re.match(r"[A-Za-z0-9._-]+", text).group(0), seen)
    return seen


def installable(requirements: list[str]) -> set[str]:
    """Every distribution a pip install of ``requirements`` brings, as canonical names."""
    seen: set[str] = set()
    for text in requirements:
        found = envcheck.parsed(text)
        if found is not None:
            requires_of(found[0], seen)
    return seen


def needed(entries: tuple[str, ...]) -> dict[str, str]:
    """Distribution -> the file that imports it, for the third-party modules the entries reach."""
    return {envcheck.canonical(envcheck.distribution_of(name)): by for name, by in envcheck.third_party(entries, REPO).items()}


def gates_job_installs() -> list[str]:
    """The requirement words on the `pip install` lines of the CI `gates` job, `-r` files expanded."""
    workflow = (REPO / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    job = re.search(r"^  gates:\n(.*?)^  \w[\w-]*:\n", workflow + "\n  end:\n", re.S | re.M).group(1)
    words: list[str] = []
    for line in re.findall(r"pip install (.+)", job):
        tokens = line.split()
        if "-r" in tokens:
            path = REPO / tokens[tokens.index("-r") + 1]
            words += [t.split("#")[0].strip() for t in path.read_text(encoding="utf-8").splitlines() if t.split("#")[0].strip()]
        words += [t for t in tokens if not t.startswith("-") and not t.endswith(".txt") and t != "-r"]
    return words


def test_the_test_extra_installs_everything_the_suite_imports_at_collection() -> None:
    brought = installable(envcheck.pyproject(REPO)["project"]["dependencies"] + envcheck.extra_requirements("test", REPO))
    missing = {dist: by for dist, by in needed(envcheck.ENTRY_FILES["dev"]).items() if dist not in brought}
    assert not missing, f"pip install '.[test]' does not bring {missing}: add them to the test extra in pyproject.toml"


def test_the_ci_gates_job_installs_everything_the_gate_tests_import() -> None:
    brought = installable(gates_job_installs())
    missing = {dist: by for dist, by in needed(GATE_JOB_ENTRIES).items() if dist not in brought}
    assert not missing, f"the `gates` job in .github/workflows/ci.yml installs nothing that provides {missing}"


def test_the_gates_job_installs_the_pytest_plugin_pyproject_asks_for() -> None:
    addopts = envcheck.pyproject(REPO)["tool"]["pytest"]["ini_options"]["addopts"]
    if re.search(r"(^|\s)-n\b", addopts if isinstance(addopts, str) else " ".join(addopts)):
        assert "pytest-xdist" in installable(gates_job_installs())


def small_checkout(tmp_path: Path, conftest: str, python: str = ">=3.12") -> Path:
    (tmp_path / "scripts" / "gates").mkdir(parents=True)
    (tmp_path / "tests").mkdir()
    (tmp_path / "scripts" / "test").write_text("import sys\n")
    (tmp_path / "scripts" / "gates" / "pinned.txt").write_text("# none\n")
    (tmp_path / "tests" / "conftest.py").write_text(conftest)
    (tmp_path / "pyproject.toml").write_text(
        f'[project]\nname = "x"\nrequires-python = "{python}"\ndependencies = ["psutil>=1"]\n'
        '[project.optional-dependencies]\ntest = ["pytest>=1"]\n')
    return tmp_path


def test_an_import_nothing_installs_is_reported_with_the_install_command(tmp_path: Path) -> None:
    root = small_checkout(tmp_path, "import zzz_not_a_module\ntry:\n    import yyy_optional\nexcept ImportError:\n    pass\n"
                                   "def late():\n    import xxx_inside_a_function\n")
    status, text = envcheck.report(["dev"], root)
    assert status == 1
    assert "zzz_not_a_module (imported by tests/conftest.py) is not importable" in text
    assert "-m pip install 'zzz_not_a_module'" in text
    assert "yyy_optional" not in text and "xxx_inside_a_function" not in text


def test_a_checkout_whose_imports_are_installed_is_ready(tmp_path: Path) -> None:
    status, text = envcheck.report(["dev"], small_checkout(tmp_path, "import json\nimport psutil\n"))
    assert status == 0 and text.startswith("preflight: dev ready")


def test_a_python_older_than_the_project_asks_for_is_reported(tmp_path: Path) -> None:
    status, text = envcheck.report(["runtime"], small_checkout(tmp_path, "", python=">=99.0"))
    assert status == 1 and "pyenv install" in text


def test_a_requirement_older_than_asked_names_the_version_that_is_there(tmp_path: Path) -> None:
    (gap,) = envcheck.requirement_gaps(["psutil>=999"])
    assert "is installed and >=999 is wanted" in gap.what and gap.pip == "psutil>=999"


def test_scripts_test_stops_with_the_fix_before_importing_what_is_missing() -> None:
    hide_numpy = ("import sys, runpy; sys.modules['numpy'] = None; sys.argv = ['scripts/test', 'fast']; "
                  "runpy.run_path('scripts/test', run_name='__main__')")
    done = subprocess.run([sys.executable, "-c", hide_numpy], cwd=REPO, capture_output=True, text=True, timeout=60, check=False)
    assert done.returncode == 1
    assert "numpy" in done.stderr and "pip install" in done.stderr and "Traceback" not in done.stderr


@pytest.mark.parametrize("purpose", envcheck.PURPOSES)
def test_every_purpose_is_known_to_gaps(purpose: str) -> None:
    assert isinstance(envcheck.gaps(purpose, REPO), list)
