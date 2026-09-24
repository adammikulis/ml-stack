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
import subprocess
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
    text = RUNNER.read_text(encoding="utf-8")
    assert '--config "$DCFG"' in text
    assert '{"auths":{}}' in text
    assert "credsStore" in text, "--help has to say why the config directory is there"


def test_the_runner_installs_what_ci_installs():
    text = RUNNER.read_text(encoding="utf-8")
    install = next(s["run"] for s in steps() if s.get("name") == "install test dependencies")
    for package in ("pytest-xdist", "numpy", "psutil", "pillow", "networkx", "gguf",
                    "safetensors"):
        assert package in text, f"the container does not install {package}, and CI does"
    extras = re.search(r'-e "(\.\[[^\]]*\])"', install).group(1)
    assert f"EXTRAS='{extras}'" in text, f"the container does not install {extras}, and CI does"
    assert "spacy download en_core_web_sm" in text and "spacy download en_core_web_sm" in install


def test_the_runner_says_which_command_told_it_docker_is_missing(tmp_path):
    fake = tmp_path / "bin"
    fake.mkdir()
    (fake / "docker").write_text("#!/bin/sh\nexit 1\n")
    (fake / "docker").chmod(0o755)
    done = subprocess.run([str(RUNNER), "-q"], capture_output=True, text=True,
                          env={**os.environ, "PATH": f"{fake}:{os.environ['PATH']}"})
    assert done.returncode == 2, done.stdout
    assert "`docker info` failed" in done.stderr, done.stderr


def test_the_runner_passes_pytest_arguments_through():
    text = RUNNER.read_text(encoding="utf-8")
    assert 'PYTEST_ARGS=("$@")' in text
    assert 'PYTEST_ARGS+=(-n 0)' in text, "--single has to reach pytest"


def test_one_matrix_entry_runs_the_suite_in_a_single_process():
    single = [e for e in entries() if e.get("pytest") == "-n 0"]
    assert single, "every job passes -n, so no run sees an order-dependent test"
    assert single[0]["os"] == "ubuntu-latest"


def test_the_other_entries_still_run_the_slow_tests():
    slow = [e for e in entries() if e.get("pytest") == "--slow"]
    assert len(slow) == len(entries()) - 1
    assert {e["os"] for e in slow} == {"ubuntu-latest"}


def test_no_push_waits_on_macos():
    """A macOS run outlasts the gap between two pushes, so one on the push path is
    cancelled by the next push and never reports."""
    assert {e["os"] for e in entries()} == {"ubuntu-latest"}
    when = workflows()["jobs"]["macos"]["if"]
    assert "github.event_name == 'schedule'" in when
    assert "github.event_name == 'workflow_dispatch'" in when
    assert "inputs.ref != ''" in when, "the release branch ships the macOS app"


def test_macos_has_a_nightly_to_run_on():
    schedule = workflows()["on"]["schedule"]
    assert [entry["cron"] for entry in schedule] == ["30 4 * * *"]


def test_macos_runs_the_slow_tests_on_the_wheels_it_built():
    steps = workflows()["jobs"]["macos"]["steps"]
    assert any("packaging/build.py" in str(s.get("run", "")) for s in steps)
    tests = [s for s in steps if s.get("name") == "tests"]
    assert tests and "--slow" in str(tests[0]["env"]["MACOS_PYTEST"])
    for trigger in ("workflow_dispatch", "workflow_call"):
        assert "--slow" in workflows()["on"][trigger]["inputs"]["macos-pytest"]["default"]


def test_ci_runs_the_release_verifier():
    named = [s for s in steps() if "verify_release.py" in str(s.get("run", ""))]
    assert named, "FEATURES.md points at a file nothing runs"
    assert "--offline" in named[0]["run"], "CI cannot wait on Hugging Face"


def test_offline_skips_the_checks_that_reach_the_internet():
    text = VERIFIER.read_text(encoding="utf-8")
    assert text.count(", network=True)") == 3, (
        "a check that reaches a host outside this machine has to say so, or CI runs it")
    assert 'OFFLINE = "--offline" in sys.argv' in text
