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
BUILD_FILE = "release-build.yml"
BUILD = read(BUILD_FILE)
DRY_RUN_FILE = "release-dry-run.yml"
DRY_RUN = read(DRY_RUN_FILE)


def checkouts() -> list[dict[str, Any]]:
    """The checkouts of the tree under test; the base commit's checker copy is separate."""
    return [step for job in CI["jobs"].values() for step in job["steps"]
            if str(step.get("uses", "")).startswith("actions/checkout@")
            and (step.get("with") or {}).get("path") != "trusted"]


def trusted_checkouts() -> list[dict[str, Any]]:
    return [step for job in CI["jobs"].values() for step in job["steps"]
            if (step.get("with") or {}).get("path") == "trusted"]


def test_the_checkers_of_a_pull_request_come_from_its_base():
    found = trusted_checkouts()
    assert len(found) == 2, "the gates and privacy jobs each check out the base"
    for step in found:
        assert step["with"]["ref"] == "${{ github.event.pull_request.base.sha }}"
        assert step["if"] == "github.event_name == 'pull_request'"


def test_the_gate_scripts_run_from_the_base_not_the_pull_request():
    runs = [str(s.get("run", "")) for s in CI["jobs"]["gates"]["steps"]]
    overlay = next(i for i, run in enumerate(runs) if "cp -R trusted/scripts/gates" in run)
    for needle in ("python scripts/budgets", "budgets-only-fall"):
        at = next(i for i, run in enumerate(runs) if needle in run)
        assert at > overlay, f"{needle} ran before the base commit's copy was in place"
    pins = next(s for s in CI["jobs"]["gates"]["steps"]
                if "pinned.txt" in str(s.get("env", {})))
    assert "trusted/" in pins["env"]["PINS"]


def test_the_name_hooks_run_from_the_base_not_the_pull_request():
    runs = [str(s.get("run", "")) for s in CI["jobs"]["privacy"]["steps"]]
    overlay = next(i for i, run in enumerate(runs) if "cp -R trusted/scripts/hooks" in run)
    for needle in ("no-real-names", "commit-msg"):
        at = next(i for i, run in enumerate(runs) if needle in run)
        assert at > overlay


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


def test_the_caller_grants_the_oidc_token_only_to_jobs_that_call_release():
    """A called workflow cannot ask for more than the calling job grants; without the grant
    release-please.yml fails at startup."""
    calls = [name for name, job in PLEASE["jobs"].items()
             if job.get("uses") == f"./.github/workflows/{RELEASE_FILE}"]
    assert calls == ["build"]
    assert id_token_jobs(PLEASE) == calls
    assert PLEASE["jobs"]["build"]["permissions"]["id-token"] == "write"
    assert "id-token" not in (PLEASE.get("permissions") or {})


def test_no_workflow_passes_every_secret_to_a_called_one():
    for name in ("release-please.yml", "release.yml", "release-dry-run.yml", "ci.yml"):
        assert "secrets: inherit" not in (WORKFLOWS / name).read_text(encoding="utf-8")
    assert "secrets" not in RELEASE["on"]["workflow_call"]


def test_the_signing_key_is_read_only_by_a_job_in_the_release_environment():
    jobs = {n: j for n, j in RELEASE["jobs"].items()
            if "secrets." in yaml.safe_dump(j)}
    assert list(jobs) == ["publish"]
    assert jobs["publish"]["environment"] == "release"
    assert "secrets." not in yaml.safe_dump(BUILD)


def test_the_build_and_the_dry_run_hold_no_write_grant():
    called = {
        "release-dry-run.yml": list(DRY_RUN["jobs"].values()),
        "release-please.yml": [PLEASE["jobs"]["release-pr-bundles"],
                               PLEASE["jobs"]["release-pr"]],
    }
    for name, jobs in called.items():
        for job in jobs:
            assert job["permissions"] == {"contents": "read"}, name
    assert BUILD["permissions"] == {"contents": "read"}
    assert all("permissions" not in job for job in BUILD["jobs"].values())


def test_a_tag_publishes_only_a_commit_that_is_on_main():
    guard = RELEASE["jobs"]["guard"]
    run = next(s["run"] for s in guard["steps"] if "run" in s)
    assert "merge-base --is-ancestor" in run and "origin main" in run
    assert guard["steps"][0]["with"]["fetch-depth"] == 0
    for name in ("pypi", "publish"):
        assert "guard" in RELEASE["jobs"][name]["needs"]
        assert "needs.guard.result == 'success'" in RELEASE["jobs"][name]["if"]


def test_the_upload_passes_no_password():
    """A password is used instead of the OIDC token, so trusted publishing never runs."""
    with_ = upload_step().get("with") or {}
    assert "password" not in with_
    assert "user" not in with_
    assert not [k for k in with_ if "token" in k or "secret" in k]


def test_no_secret_reaches_the_upload_job():
    assert "secrets." not in yaml.safe_dump(RELEASE["jobs"]["pypi"])


def test_the_upload_runs_in_the_release_environment():
    """The registered publisher names the `release` environment, whose deployment rules
    keep a branch's copy of this file from minting the token."""
    assert RELEASE["jobs"]["pypi"]["environment"] == "release"


def test_a_rerun_does_not_fail_on_a_version_already_there():
    assert (upload_step().get("with") or {})["skip-existing"] is True


def test_the_upload_waits_for_wheels_that_built():
    job = RELEASE["jobs"]["pypi"]
    assert "build" in job["needs"]
    assert "needs.build.outputs.wheels == 'success'" in job["if"], "always() uploads nothing"
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


def bundle_runs() -> list[str]:
    """Each shell block of the bundle job, with its continuation lines joined."""
    return [str(s["run"]).replace("\\\n", " ") for s in BUILD["jobs"]["bundle"]["steps"]
            if "run" in s]


def test_the_bundle_is_set_up_and_opened_in_a_browser():
    assert any("packaging/smoke.py" in run for run in bundle_runs())


def test_no_curl_is_piped_into_a_reader_that_stops_early():
    """`grep -q` closes the pipe on its first match; under pipefail curl then fails 23."""
    for run in bundle_runs():
        for line in run.splitlines():
            assert not ("curl" in line and "| grep -q" in line), line


def test_the_window_reopens_only_on_the_platform_that_has_the_event():
    """tauri defines `RunEvent::Reopen` on macOS alone."""
    main = (WORKFLOWS.parents[1] / "app" / "src-tauri" / "src" / "main.rs").read_text(
        encoding="utf-8").splitlines()
    at = [i for i, line in enumerate(main) if "RunEvent::Reopen" in line]
    assert at, "the window no longer handles a reopen"
    for i in at:
        assert main[i - 1].strip() == '#[cfg(target_os = "macos")]'


def release_checkouts() -> list[dict[str, Any]]:
    return [step for workflow in (RELEASE, BUILD) for job in workflow["jobs"].values()
            for step in job.get("steps", [])
            if str(step.get("uses", "")).startswith("actions/checkout@")]


def test_release_can_be_called_by_another_workflow():
    assert "workflow_call" in RELEASE["on"], "release-please.yml cannot call an uncallable workflow"


def test_release_takes_the_ref_to_check_out():
    ref = RELEASE["on"]["workflow_call"]["inputs"]["ref"]
    assert ref["type"] == "string"
    assert ref["default"] == "", "an empty default is the caller's own ref"


@pytest.mark.parametrize("step", release_checkouts(), ids=lambda s: s.get("uses", "?"))
def test_every_release_checkout_honours_the_ref_it_was_given(step):
    assert (step.get("with") or {}).get("ref") == "${{ inputs.ref }}"


def test_a_called_release_run_does_not_cancel_the_run_that_called_it():
    """Two calls sharing github.ref (release-please's own push, and a bare CI run) must
    not sit in the same concurrency group."""
    assert RELEASE["concurrency"]["group"] == "release-${{ github.ref }}-${{ inputs.ref }}"
    assert RELEASE["concurrency"]["cancel-in-progress"] is True


def test_the_release_tag_is_optional_so_a_dry_build_can_omit_it():
    tag = RELEASE["on"]["workflow_call"]["inputs"]["tag"]
    assert tag["required"] is False
    assert tag["default"] == ""


def test_pypi_only_runs_once_the_project_is_registered():
    assert "vars.PYPI_ENABLED == 'true'" in RELEASE["jobs"]["pypi"]["if"]


def test_release_please_builds_the_pr_branch_without_publishing():
    """The release pull request is green only once every bundle builds and passes the
    smoke check, which release.yml alone runs."""
    job = PLEASE["jobs"]["release-pr-bundles"]
    assert job["uses"] == f"./.github/workflows/{BUILD_FILE}"
    assert job["if"] == "needs.propose.outputs.pr_branch != ''"
    assert job["with"]["ref"] == "${{ needs.propose.outputs.pr_branch }}"
    assert "tag" not in job["with"], "a tag here would try to publish the PR branch"


def test_the_dry_run_watches_the_development_branches():
    assert DRY_RUN["on"]["push"]["branches"] == ["*dev"]


def test_the_dry_run_path_filter_covers_what_ships_the_app():
    paths = DRY_RUN["on"]["pull_request"]["paths"]
    for expected in (
        "app/**",
        "packaging/**",
        "src/ml_stack/fleet/**",
        "src/ml_stack/ui/**",
        ".github/workflows/release*.yml",
    ):
        assert expected in paths


def test_the_dry_run_calls_release_without_publishing():
    jobs = list(DRY_RUN["jobs"].values())
    assert len(jobs) == 1
    job = jobs[0]
    assert job["uses"] == f"./.github/workflows/{BUILD_FILE}"
    assert "with" not in job or "tag" not in job["with"]


def test_the_smoke_screenshots_are_not_published():
    publish = RELEASE["jobs"]["publish"]["steps"]
    runs = [step.get("run", "") for step in publish]
    cleared = next(i for i, run in enumerate(runs) if "-smoke" in run and "rm " in run)
    uploads = [i for i, step in enumerate(publish) if "gh release" in step.get("run", "")
               or "release" in str(step.get("uses", ""))]
    assert uploads and all(cleared < i for i in uploads)
