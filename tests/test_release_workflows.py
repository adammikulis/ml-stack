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
    overlay = next(i for i, run in enumerate(runs) if "trusted/scripts/gates/*.py" in run)
    for needle in ("python scripts/budgets", "budgets-only-fall"):
        at = next(i for i, run in enumerate(runs) if needle in run)
        assert at > overlay, f"{needle} ran before the base commit's copy was in place"


def test_the_data_beside_the_checkers_is_the_pull_requests_own():
    """A pull request that moves a pin or a survivor row is measured against its own data."""
    overlay = next(s["run"] for s in CI["jobs"]["gates"]["steps"]
                   if "trusted/scripts/gates/*.py" in str(s.get("run", "")))
    assert "pinned.txt" not in overlay and "survivors.txt" not in overlay
    assert "rm -rf scripts/gates " not in overlay and "scripts/gates/*.py" in overlay


def test_the_name_hooks_run_from_the_base_not_the_pull_request():
    runs = [str(s.get("run", "")) for s in CI["jobs"]["privacy"]["steps"]]
    overlay = next(i for i, run in enumerate(runs) if "cp -R trusted/scripts/hooks" in run)
    for needle in ("no-real-names", "commit-msg"):
        at = next(i for i, run in enumerate(runs) if needle in run)
        assert at > overlay


def test_ci_names_no_caller_ref():
    """CodeQL's cache-poisoning rule: a job that runs under workflow_dispatch or schedule
    and checks out a ref an input names runs that ref in the default branch's cache scope.
    ci.yml is dispatched on a branch instead (`gh workflow run ci.yml --ref`)."""
    assert "workflow_call" not in CI["on"]
    assert "inputs.ref" not in yaml.safe_dump(CI)


def test_there_is_a_checkout_to_check():
    assert checkouts(), "ci.yml stopped checking anything out"


@pytest.mark.parametrize("step", checkouts(), ids=lambda s: s.get("uses", "?"))
def test_no_checkout_takes_its_ref_from_an_input(step):
    assert "inputs." not in str((step.get("with") or {}).get("ref", ""))


def test_concurrent_runs_of_one_ref_cancel_each_other():
    assert CI["concurrency"]["group"] == "ci-${{ github.ref }}"
    assert CI["concurrency"]["cancel-in-progress"] is True


def test_release_please_tests_the_branch_it_opened():
    job = PLEASE["jobs"]["release-pr"]
    assert job["if"] == "needs.propose.outputs.pr_branch != ''", "no pull request, no run"
    assert job["permissions"] == {"actions": "write", "contents": "read"}
    script = "\n".join(str(step.get("run", "")) for step in job["steps"])
    assert 'gh workflow run ci.yml --ref "$BRANCH"' in script
    assert 'gh workflow run release-dry-run.yml --ref "$BRANCH"' in script
    assert job["steps"][0]["env"]["BRANCH"] == "${{ needs.propose.outputs.pr_branch }}"


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
    assert sorted(jobs) == ["publish", "pypi"]
    assert all(job["environment"] == "release" for job in jobs.values())
    assert "secrets." not in yaml.safe_dump(BUILD)


def test_the_build_and_the_dry_run_hold_no_write_grant():
    called = {
        "release-dry-run.yml": list(DRY_RUN["jobs"].values()),
    }
    for name, jobs in called.items():
        for job in jobs:
            assert job["permissions"] == {"contents": "read"}, name
    assert BUILD["permissions"] == {"contents": "read"}
    assert all("permissions" not in job for job in BUILD["jobs"].values())


def test_a_tag_publishes_only_a_commit_that_is_on_main():
    guard = RELEASE["jobs"]["guard"]
    run = next(s["run"] for s in guard["steps"] if "run" in s)
    assert "merge-base --is-ancestor" in run and "origin/main" in run
    assert "git fetch" not in run
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


def test_the_signing_key_reaches_the_upload_job_only_as_a_presence_check():
    steps = RELEASE["jobs"]["pypi"]["steps"]
    holders = [s for s in steps if "secrets." in yaml.safe_dump(s)]
    assert len(holders) == 1 and steps.index(holders[0]) == 0
    assert "test -n" in holders[0]["run"]
    assert "secrets." not in yaml.safe_dump(upload_step())


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
def test_no_job_that_checks_out_a_caller_ref_shares_a_package_cache(step):
    """CodeQL's cache-poisoning rule: ``ref: inputs.ref`` on a dispatch runs untrusted code in
    the default branch's context, and a cache it saved would be restored by trusted jobs."""
    with_ = step.get("with") or {}
    assert "cache" not in with_
    assert "cache-dependency-path" not in with_


def test_the_bundle_job_keeps_no_rust_cache():
    uses = [str(s.get("uses", "")) for s in BUILD["jobs"]["bundle"]["steps"]]
    assert not any(u.startswith("Swatinem/rust-cache@") for u in uses)


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


def test_release_and_build_take_no_ref_input():
    for workflow in (RELEASE, BUILD):
        assert "ref" not in workflow["on"]["workflow_call"]["inputs"]
        assert "inputs.ref" not in yaml.safe_dump(workflow)


@pytest.mark.parametrize("step", release_checkouts(), ids=lambda s: s.get("uses", "?"))
def test_every_release_checkout_uses_the_callers_own_ref(step):
    assert "ref" not in (step.get("with") or {}) or "inputs." not in step["with"]["ref"]


def test_concurrent_release_runs_of_one_ref_cancel_each_other():
    assert RELEASE["concurrency"]["group"] == "release-${{ github.ref }}"
    assert RELEASE["concurrency"]["cancel-in-progress"] is True


def test_the_release_tag_is_optional_so_a_dry_build_can_omit_it():
    tag = RELEASE["on"]["workflow_call"]["inputs"]["tag"]
    assert tag["required"] is False
    assert tag["default"] == ""


def test_pypi_only_runs_once_the_project_is_registered():
    assert "vars.PYPI_ENABLED == 'true'" in RELEASE["jobs"]["pypi"]["if"]


def test_the_dry_run_can_be_dispatched_on_the_release_branch():
    """The release pull request is green only once every bundle builds and passes the smoke
    check; release-please.yml dispatches this on the branch, with no tag, so nothing publishes."""
    assert "workflow_dispatch" in DRY_RUN["on"]
    assert "tag" not in (DRY_RUN["jobs"]["build"].get("with") or {})


def test_the_dry_run_watches_the_development_branches():
    assert DRY_RUN["on"]["push"]["branches"] == ["*dev"]


def test_the_dry_run_path_filter_covers_what_ships_the_app():
    paths = DRY_RUN["on"]["pull_request"]["paths"]
    for expected in (
        "app/**",
        "packaging/**",
        "src/poolhouse/fleet/**",
        "src/poolhouse/ui/**",
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
