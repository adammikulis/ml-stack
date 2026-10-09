"""What makes a branch's green mean something on a platform nobody here runs.

A macOS-only suite cannot see a graph read that returns a different order on Linux, or a
process that is reapable a couple of milliseconds later there. `scripts/test-on-linux`
runs the suite in a container, one matrix entry runs it in a single process, and CI runs
`docs/verify_release.py`, which nothing ran for nine days.
"""

from __future__ import annotations

import os
import re
import stat
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
RUNNER = REPO / "scripts" / "test-on-linux"
VERIFIER = REPO / "docs" / "verify_release.py"


def workflows() -> dict:
    loaded = yaml.safe_load((REPO / ".github/workflows/ci.yml").read_text(encoding="utf-8"))
    loaded["on"] = loaded.pop(True, loaded.get("on"))
    return loaded


def ci() -> dict:
    return workflows()["jobs"]["test"]


def entries() -> list[dict]:
    return ci()["strategy"]["matrix"]["include"]


def steps() -> list[dict]:
    return ci()["steps"]


def test_the_runner_is_executable():
    assert RUNNER.exists(), "there is no way to run the suite on Linux from a Mac"
    assert RUNNER.stat().st_mode & stat.S_IXUSR


def test_the_runner_never_lets_the_credential_helper_be_consulted():
    """A pull through the macOS desktop helper can hang until it is killed."""
    assert "credsStore" in RUNNER.read_text(encoding="utf-8"), "--help has to say why"
    launch = (REPO / "scripts" / "test_container_launch.py").read_text(encoding="utf-8")
    assert '{"auths":{}}' in launch, "the config directory holds no credential helper"
    assert "--config" in launch


def test_the_runner_installs_what_ci_installs():
    text = (RUNNER.parent / "test-on-linux-setup").read_text(encoding="utf-8")
    install = next(s["run"] for s in steps() if s.get("name") == "install test dependencies")
    for package in ("pytest-xdist", "numpy", "psutil", "pillow", "networkx", "gguf",
                    "safetensors"):
        assert package in text, f"the container does not install {package}, and CI does"
    extras = re.search(r'-e "(\.\[[^\]]*\])"', install).group(1)
    assert f"EXTRAS='{extras[1:]}'" in text, f"the container does not install {extras}, and CI does"
    assert "spacy download en_core_web_sm" in text and "spacy download en_core_web_sm" in install


def test_a_machine_without_docker_is_refused_before_anything_starts(monkeypatch):
    import pytest

    monkeypatch.syspath_prepend(str(REPO / "scripts"))
    monkeypatch.setenv("PATH", os.defpath)
    monkeypatch.setattr("shutil.which", lambda name: None)
    from test_container_launch import ContainerRun

    with pytest.raises(RuntimeError, match="Docker is required"):
        ContainerRun([sys.executable, "-m", "pytest", "tests/test_layers.py"], {})


def test_the_runner_passes_pytest_arguments_through():
    text = RUNNER.read_text(encoding="utf-8")
    assert 'PYTEST_ARGS=("$@")' in text
    assert '[ -n "$SINGLE" ] && WANT=1' in text, "--single has to reach the scheduler as one process"
    assert 'ML_STACK_LINUX_SINGLE="$SINGLE"' in text


def test_every_matrix_entry_runs_the_slow_tests_on_ubuntu():
    assert {e.get("pytest") for e in entries()} == {"--slow"}
    assert {e["os"] for e in entries()} == {"ubuntu-latest"}


def test_no_push_waits_on_macos():
    """A macOS run outlasts the gap between two pushes, so one on the push path is
    cancelled by the next push and never reports."""
    assert {e["os"] for e in entries()} == {"ubuntu-latest"}
    when = workflows()["jobs"]["macos"]["if"]
    assert "github.event_name == 'schedule'" in when
    assert "github.event_name == 'workflow_dispatch'" in when
    assert "inputs.ref" not in when, "a caller's ref never reaches a run that can be dispatched"


def test_macos_has_a_nightly_to_run_on():
    schedule = workflows()["on"]["schedule"]
    assert [entry["cron"] for entry in schedule] == ["30 4 * * *"]


def test_macos_runs_the_slow_tests_on_the_wheels_it_built():
    steps = workflows()["jobs"]["macos"]["steps"]
    assert any("packaging/build.py" in str(s.get("run", "")) for s in steps)
    tests = [s for s in steps if s.get("name") == "tests"]
    assert tests and "--slow" in str(tests[0]["env"]["MACOS_PYTEST"])
    assert "--slow" in workflows()["on"]["workflow_dispatch"]["inputs"]["macos-pytest"]["default"]


def test_ci_runs_the_release_verifier():
    named = [s for s in steps() if "verify_release.py" in str(s.get("run", ""))]
    assert named, "FEATURES.md points at a file nothing runs"
    assert "--offline" in named[0]["run"], "CI cannot wait on Hugging Face"


def test_offline_skips_the_checks_that_reach_the_internet():
    text = VERIFIER.read_text(encoding="utf-8")
    assert text.count(", network=True)") == 3, (
        "a check that reaches a host outside this machine has to say so, or CI runs it")
    assert 'OFFLINE = "--offline" in sys.argv' in text
