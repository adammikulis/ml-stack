"""The interpreter the app runs on, and the range the library supports.

The two installers, the Linux container, the release workflow and
`ml_stack.fleet.environment` name one interpreter, `PYTHON`: the app builds its own
environment and chooses it. `pyproject.toml` and the CI test matrix name the range the
library is imported under, which is every release the suite runs on. A release newer than the
newest tested one is run as an experiment that may fail.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest
import yaml

from ml_stack.fleet.environment import PYTHON

REPO = Path(__file__).resolve().parents[1]
PYPROJECT = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
SH = (REPO / "packaging" / "install.sh").read_text(encoding="utf-8")
PS1 = (REPO / "packaging" / "install.ps1").read_text(encoding="utf-8")
RUNNER = (REPO / "scripts" / "test-on-linux").read_text(encoding="utf-8")
MAJOR, MINOR = (int(part) for part in PYTHON.split("."))
OLDEST = "3.12"
NEWEST = "3.14"
SUPPORTED = {f"3.{minor}" for minor in range(int(OLDEST.split(".")[1]), int(NEWEST.split(".")[1]) + 1)}


def workflow(name: str) -> dict:
    return yaml.safe_load((REPO / ".github/workflows" / name).read_text(encoding="utf-8"))


def test_requires_python_starts_at_the_oldest_release_tested_and_has_no_ceiling():
    assert PYPROJECT["project"]["requires-python"] == f">={OLDEST}"
    assert PYTHON in SUPPORTED


def test_one_classifier_per_supported_python():
    named = [c for c in PYPROJECT["project"]["classifiers"]
             if c.startswith("Programming Language :: Python :: 3")]
    assert named == [f"Programming Language :: Python :: {v}" for v in sorted(SUPPORTED)]


def test_ruff_targets_the_oldest_supported_release():
    assert PYPROJECT["tool"]["ruff"]["target-version"] == "py" + OLDEST.replace(".", "")


def test_the_shell_installer_looks_for_that_python_and_no_other():
    assert re.search(rf'^PYTHON="{re.escape(PYTHON)}"$', SH, re.M)
    assert 'python$PYTHON' in SH
    assert not re.search(r"python3\.\d+", SH), "a version spelled out beside $PYTHON"


def test_the_windows_installer_looks_for_that_python_and_no_other():
    assert re.search(rf'^\$python\s+= "{re.escape(PYTHON)}"$', PS1, re.M)
    assert '"-$python"' in PS1 and '"python$python"' in PS1
    assert not re.search(r"python3\.\d+", PS1)


@pytest.mark.parametrize("script", ["install.sh", "install.ps1"], ids=["sh", "ps1"])
def test_neither_installer_offers_to_run_on_something_older(script):
    body = SH if script == "install.sh" else PS1
    assert "or newer" not in body
    assert "version_info" not in body, "an installer that probes a version accepts a range"


def test_the_ci_test_matrix_runs_every_supported_python_and_other_jobs_the_app_python():
    ci = workflow("ci.yml")
    entries = ci["jobs"]["test"]["strategy"]["matrix"]["include"]
    assert {e["python"] for e in entries if not e.get("experimental")} == SUPPORTED
    experimental = {e["python"] for e in entries if e.get("experimental")}
    assert experimental == {f"3.{int(NEWEST.split('.')[1]) + 1}"}
    assert ci["jobs"]["test"]["continue-on-error"] == "${{ matrix.experimental || false }}"
    pinned = [str(step.get("with", {}).get("python-version", ""))
              for name, job in ci["jobs"].items() if name != "test" for step in job["steps"]]
    assert {v for v in pinned if v and not v.startswith("${{")} == {PYTHON}


def test_the_release_workflow_builds_on_the_one_python():
    pinned = {str(step.get("with", {}).get("python-version", ""))
              for name in ("release.yml", "release-build.yml")
              for job in workflow(name)["jobs"].values() for step in job.get("steps", [])}
    assert {v for v in pinned if v} == {PYTHON}


def test_the_linux_container_is_that_python():
    assert re.search(rf'^PYTHON="{re.escape(PYTHON)}"$', RUNNER, re.M)
    assert 'python:$PYTHON' in RUNNER
    assert not re.search(r"python:3\.\d+", RUNNER), "a version buried in the docker line"
