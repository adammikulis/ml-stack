"""What the release workflows must keep doing, since nothing else runs them here.

release-please opens its pull request with the action's own token, and a workflow run on
such a pull request waits for approval that nobody gives -- every one since 2026-09-03 sat
`action_required`. So the release branch is tested by release-please.yml calling ci.yml,
not by the pull request triggering it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

WORKFLOWS = Path(__file__).resolve().parents[1] / ".github" / "workflows"


def read(name: str) -> dict[str, Any]:
    """A workflow file as a mapping. ``on:`` parses as the boolean True in YAML 1.1."""
    loaded = yaml.safe_load((WORKFLOWS / name).read_text(encoding="utf-8"))
    loaded["on"] = loaded.pop(True, loaded.get("on"))
    return loaded


CI = read("ci.yml")
PLEASE = read("release-please.yml")
RELEASE_FILE = "release.yml"
RELEASE = read(RELEASE_FILE)


def checkouts() -> list[dict[str, Any]]:
    return [step for job in CI["jobs"].values() for step in job["steps"]
            if str(step.get("uses", "")).startswith("actions/checkout@")]


def test_ci_can_be_called_by_another_workflow():
    assert "workflow_call" in CI["on"], "release-please.yml cannot call an uncallable workflow"


def test_ci_takes_the_ref_to_check_out():
    ref = CI["on"]["workflow_call"]["inputs"]["ref"]
    assert ref["type"] == "string"
    assert ref["default"] == "", "an empty default is the caller's own ref"


def test_there_is_a_checkout_to_check():
    assert checkouts(), "ci.yml stopped checking anything out"


@pytest.mark.parametrize("step", checkouts(), ids=lambda s: s.get("uses", "?"))
def test_every_checkout_honours_the_ref_it_was_given(step):
    """A called run that checks out its caller's ref tests the wrong commit and says green."""
    assert (step.get("with") or {}).get("ref") == "${{ inputs.ref }}"


def test_a_called_run_does_not_cancel_the_run_that_called_it():
    """Both would sit in group ci-refs/heads/main, and cancel-in-progress kills one."""
    assert CI["concurrency"]["group"] == "ci-${{ github.ref }}-${{ inputs.ref }}"
    assert CI["concurrency"]["cancel-in-progress"] is True


def test_release_please_tests_the_branch_it_opened():
    job = PLEASE["jobs"]["release-pr"]
    assert job["uses"] == "./.github/workflows/ci.yml"
    assert job["with"]["ref"] == "${{ needs.propose.outputs.pr_branch }}"
    assert job["if"] == "needs.propose.outputs.pr_branch != ''", "no pull request, no run"


def test_the_release_branch_is_read_from_the_action_and_not_guessed():
    steps = PLEASE["jobs"]["propose"]["steps"]
    named = [s for s in steps if s.get("id") == "pr"]
    assert named and "jq -r .headBranchName" in named[0]["run"]
    assert PLEASE["jobs"]["propose"]["outputs"]["pr_branch"] == "${{ steps.pr.outputs.branch }}"


def test_the_build_still_only_runs_for_a_release():
    assert PLEASE["jobs"]["build"]["if"] == "needs.propose.outputs.released == 'true'"


def test_the_test_extra_carries_what_these_tests_read():
    text = (WORKFLOWS.parents[1] / "pyproject.toml").read_text(encoding="utf-8")
    assert "pyyaml" in text, "this file parses YAML; the extra has to say so"


def upload_step() -> dict[str, Any]:
    steps = RELEASE["jobs"]["pypi"]["steps"]
    found = [s for s in steps
             if str(s.get("uses", "")).startswith("pypa/gh-action-pypi-publish@")]
    assert len(found) == 1, "one upload step, or the job is not what this file describes"
    return found[0]


def test_the_release_publishes_to_pypi():
    assert "pypi" in RELEASE["jobs"], "nothing uploads the wheels anywhere"


def test_the_workflow_filename_is_the_one_the_publisher_names():
    """PyPI matches a trusted publisher on owner, repository and workflow filename."""
    assert (WORKFLOWS / RELEASE_FILE).is_file()


def test_the_upload_mints_an_oidc_token():
    assert RELEASE["jobs"]["pypi"]["permissions"]["id-token"] == "write"


def id_token_jobs(workflow: dict[str, Any]) -> list[str]:
    """The jobs of ``workflow`` whose permissions name ``id-token``."""
    return [name for name, job in workflow["jobs"].items()
            if "id-token" in (job.get("permissions") or {})]


def test_only_the_upload_asks_for_the_oidc_token():
    assert id_token_jobs(RELEASE) == ["pypi"]
    assert "id-token" not in (RELEASE.get("permissions") or {})


def test_the_caller_grants_the_oidc_token_only_to_the_job_that_calls_release():
    """A called workflow cannot ask for more than the calling job grants; without the grant
    release-please.yml fails at startup."""
    calls = [name for name, job in PLEASE["jobs"].items()
             if job.get("uses") == f"./.github/workflows/{RELEASE_FILE}"]
    assert calls == ["build"]
    assert id_token_jobs(PLEASE) == calls
    assert PLEASE["jobs"]["build"]["permissions"]["id-token"] == "write"
    assert "id-token" not in (PLEASE.get("permissions") or {})


def test_the_upload_passes_no_password():
    """A password is used instead of the OIDC token, so trusted publishing never runs."""
    with_ = upload_step().get("with") or {}
    assert "password" not in with_
    assert "user" not in with_
    assert not [k for k in with_ if "token" in k or "secret" in k]


def test_no_secret_reaches_the_upload_job():
    assert "secrets." not in yaml.safe_dump(RELEASE["jobs"]["pypi"])


def test_the_upload_declares_no_environment():
    """The registered publisher names no environment; one here would stop it matching."""
    assert "environment" not in RELEASE["jobs"]["pypi"]


def test_a_rerun_does_not_fail_on_a_version_already_there():
    assert (upload_step().get("with") or {})["skip-existing"] is True


def test_the_upload_waits_for_wheels_that_built():
    job = RELEASE["jobs"]["pypi"]
    assert job["needs"] == "wheels" or "wheels" in job["needs"]
    assert "needs.wheels.result == 'success'" in job["if"], "always() uploads nothing"
    assert "always()" not in job["if"]


def test_the_github_release_says_pip_only_when_the_upload_succeeded():
    publish = RELEASE["jobs"]["publish"]
    assert "pypi" in publish["needs"]
    step = next(s for s in publish["steps"] if s.get("id") == "downloads")
    assert step["env"]["PYPI"] == "${{ needs.pypi.result }}"
    assert '[ "$PYPI" = "success" ]' in step["run"]


def test_a_failed_upload_still_releases_on_github():
    """A PyPI outage must not cost the bundles their release page."""
    assert "needs.pypi.result == 'success'" not in RELEASE["jobs"]["publish"]["if"]


def setups() -> list[dict[str, Any]]:
    return [step for job in CI["jobs"].values() for step in job["steps"]
            if str(step.get("uses", "")).startswith("actions/setup-python@")]


@pytest.mark.parametrize("step", setups(), ids=lambda s: str(s.get("with", {}).get(
    "python-version", "?")))
def test_every_job_caches_the_packages_it_installs(step):
    """Every job installs the same set; without a cache each one downloads torch again."""
    with_ = step.get("with") or {}
    assert with_.get("cache") == "pip"
    assert with_.get("cache-dependency-path") == "pyproject.toml"
