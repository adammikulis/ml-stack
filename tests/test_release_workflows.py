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
