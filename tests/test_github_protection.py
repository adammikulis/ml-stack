"""scripts/github-protection: drift found from canned `gh api` replies, and no call that writes."""

from __future__ import annotations

import copy
import importlib.machinery
import importlib.util
import re
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "github-protection"
from tests.github_support import GOOD, REPO, serve  # noqa: E402


@pytest.fixture(scope="module")
def gp():
    loader = importlib.machinery.SourceFileLoader("github_protection", str(SCRIPT))
    spec = importlib.util.spec_from_loader("github_protection", loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


def run(gp, capsys, *flags):
    code = gp.main(["--check", "--repo", REPO, *flags])
    return code, capsys.readouterr().out


def codes(out):
    return {line.split()[0] for line in out.splitlines() if "->" in line}


def anchors():
    """The anchors GitHub gives the headings of docs/github-protection.md."""
    text = (SCRIPT.parent.parent / "docs" / "github-protection.md").read_text(encoding="utf-8")
    return {re.sub(r"[^a-z0-9 -]", "", h.lower()).replace(" ", "-")
            for h in re.findall(r"^#{2,3} (.+)$", text, re.M)}


SECTIONS = ("rulesets", "codeowners", "workflow-rules", "agent-identity", "who-merges",
            "actions-settings", "code-security-settings", "the-release-environment",
            "main-branch-ruleset", "development-branch-ruleset", "release-tag-ruleset")


@pytest.mark.parametrize("section", SECTIONS)
def test_every_section_the_script_points_at_is_a_heading(section):
    assert f'"{section}"' in SCRIPT.read_text(encoding="utf-8")
    assert section in anchors()


def changed(mutate):
    replies = copy.deepcopy(GOOD)
    mutate(replies)
    return replies


def test_the_script_contains_no_call_that_writes():
    text = SCRIPT.read_text(encoding="utf-8")
    assert not re.search(r"-X\s*(POST|PUT|PATCH|DELETE)", text)
    for forbidden in ("gh secret", "gh workflow run", "--method", "--input", "--field", "--raw-field"):
        assert forbidden not in text, forbidden
    assert not re.search(r"""["']-[fF]["']""", text)


def test_a_repository_that_matches_the_target_has_no_findings(gp, tmp_path, monkeypatch, capsys):
    log = serve(tmp_path, monkeypatch, GOOD)
    code, out = run(gp, capsys)
    assert (code, out.strip().splitlines()[-1]) == (0, "0 findings"), out
    calls = log.read_text().splitlines()
    assert calls and all(re.fullmatch(r"GET \S+", line) for line in calls), calls


def test_every_call_is_a_plain_get(gp, tmp_path, monkeypatch, capsys):
    log = serve(tmp_path, monkeypatch, GOOD)
    run(gp, capsys, "--agent")
    for line in log.read_text().splitlines():
        words = line.split()
        assert words[0] == "GET" and len(words) == 2, line


def test_the_repository_comes_from_gh_when_none_is_named(gp, tmp_path, monkeypatch, capsys):
    serve(tmp_path, monkeypatch, GOOD)
    assert gp.main(["--check"]) == 0
    assert capsys.readouterr().out.strip().endswith("0 findings")


def test_local_mode_makes_no_request(gp, tmp_path, monkeypatch, capsys):
    log = serve(tmp_path, monkeypatch, {})
    assert gp.main(["--check", "--local"]) == 0
    assert "(local only)" in capsys.readouterr().out
    assert log.read_text() == ""


CASES = {
    "bypass actor on main": (
        lambda r: r[f"repos/{REPO}/rulesets/1"]["bypass_actors"].append(
            {"actor_id": 5, "actor_type": "RepositoryRole", "bypass_mode": "always"}),
        "ruleset.main.bypass"),
    "no pull request rule": (
        lambda r: r[f"repos/{REPO}/rulesets/1"]["rules"].pop(2), "ruleset.main.rule"),
    "gates not required": (
        lambda r: r[f"repos/{REPO}/rulesets/1"]["rules"][3]["parameters"][
            "required_status_checks"].pop(0), "ruleset.main.checks"),
    "dev ruleset blocks pushes": (
        lambda r: r[f"repos/{REPO}/rulesets/2"]["rules"].append({"type": "update"}),
        "ruleset.dev.extra"),
    "tag ruleset missing": (
        lambda r: r[f"repos/{REPO}/rulesets"].pop(), "ruleset.tags.missing"),
    "ruleset only evaluating": (
        lambda r: r[f"repos/{REPO}/rulesets/3"].update(enforcement="evaluate"),
        "ruleset.tags.enforcement"),
    "no reviewer": (
        lambda r: r[f"repos/{REPO}/environments/release"].update(protection_rules=[]),
        "env.reviewer"),
    "admins bypass the environment": (
        lambda r: r[f"repos/{REPO}/environments/release"].update(can_admins_bypass=True),
        "env.admin-bypass"),
    "environment open to every branch": (
        lambda r: r[f"repos/{REPO}/environments/release"].update(
            deployment_branch_policy={"custom_branch_policies": False}), "env.deployment"),
    "environment allows another branch": (
        lambda r: r[f"repos/{REPO}/environments/release/deployment-branch-policies"][
            "branch_policies"].append({"name": "v*", "type": "tag"}), "env.deployment"),
    "key held as a repository secret": (
        lambda r: r[f"repos/{REPO}/actions/secrets"]["secrets"].append(
            {"name": "RELEASE_SIGNING_KEY"}), "secret.repository"),
    "key missing from the environment": (
        lambda r: r[f"repos/{REPO}/environments/release/secrets"].update(secrets=[]),
        "secret.environment"),
    "token can write": (
        lambda r: r[f"repos/{REPO}/actions/permissions/workflow"].update(
            default_workflow_permissions="write"), "actions.default-token"),
    "every action allowed": (
        lambda r: r[f"repos/{REPO}/actions/permissions"].update(allowed_actions="all"),
        "actions.allowed"),
    "first-time contributors only": (
        lambda r: r[f"repos/{REPO}/actions/permissions/fork-pr-contributor-approval"].update(
            approval_policy="first_time_contributors"), "actions.fork-approval"),
    "push protection off": (
        lambda r: r[f"repos/{REPO}"]["security_and_analysis"].update(
            secret_scanning_push_protection={"status": "disabled"}),
        "security.secret_scanning_push_protection"),
    "auto-merge on": (
        lambda r: r[f"repos/{REPO}"].update(allow_auto_merge=True), "repo.auto-merge"),
}


@pytest.mark.parametrize("what", CASES)
def test_drift_is_reported_with_its_fix(gp, tmp_path, monkeypatch, capsys, what):
    mutate, code = CASES[what]
    serve(tmp_path, monkeypatch, changed(mutate))
    status, out = run(gp, capsys)
    assert status == 1
    assert code in codes(out), out
    line = next(ln for ln in out.splitlines() if ln.startswith(code))
    assert re.search(r"-> docs/github-protection\.md#[a-z0-9-]+$", line), line
    assert line.rsplit("#", 1)[1] in anchors()


def test_an_unreadable_endpoint_is_a_finding(gp, tmp_path, monkeypatch, capsys):
    replies = changed(lambda r: r.pop(f"repos/{REPO}/environments/release"))
    serve(tmp_path, monkeypatch, replies)
    status, out = run(gp, capsys)
    assert status == 1 and "env.read" in codes(out)


def test_an_agent_run_fails_on_owner_or_admin_credentials(gp, tmp_path, monkeypatch, capsys):
    def owner(r):
        r["user"] = {"login": "ownerlogin"}
        r[f"repos/{REPO}"]["permissions"]["admin"] = True

    serve(tmp_path, monkeypatch, changed(owner))
    assert run(gp, capsys)[0] == 0
    status, out = run(gp, capsys, "--agent")
    assert status == 1 and {"identity.owner", "identity.admin"} <= codes(out)


def test_the_owner_review_is_required_only_when_asked(gp, tmp_path, monkeypatch, capsys):
    serve(tmp_path, monkeypatch, GOOD)
    assert run(gp, capsys)[0] == 0
    status, out = run(gp, capsys, "--require-owner-review")
    assert status == 1 and "ruleset.main.pull-request" in codes(out)


def test_any_repository_secret_is_a_finding(gp, tmp_path, monkeypatch, capsys):
    replies = changed(lambda r: r[f"repos/{REPO}/actions/secrets"]["secrets"].append(
        {"name": "OTHER_TOKEN"}))
    serve(tmp_path, monkeypatch, replies)
    status, out = run(gp, capsys)
    assert status == 1 and "OTHER_TOKEN" in out


def test_tag_creation_is_blocked_only_when_asked(gp, tmp_path, monkeypatch, capsys):
    replies = changed(lambda r: r[f"repos/{REPO}/rulesets/3"]["rules"].pop())
    serve(tmp_path, monkeypatch, replies)
    assert run(gp, capsys)[0] == 0
    status, out = run(gp, capsys, "--strict-tags")
    assert status == 1 and "ruleset.tags.rule" in codes(out)


def test_signed_commits_are_required_only_when_asked(gp, tmp_path, monkeypatch, capsys):
    serve(tmp_path, monkeypatch, GOOD)
    assert "ruleset.main.rule" in codes(run(gp, capsys, "--require-signatures")[1])


def test_codeowners_rules_cover_the_paths_that_decide_what_the_gates_accept(gp):
    rules = gp.owners_rules((SCRIPT.parent.parent / ".github" / "CODEOWNERS").read_text())
    for path in ("scripts/hooks/pre-push", "scripts/gates/ruff.py", "budgets.json",
                 "src/ml_stack/fleet/signing.py", "src/ml_stack/guard/x.py",
                 "src/ml_stack/workspace/person_session.py", ".github/workflows/ci.yml",
                 "scripts/land_run.py", "scripts/test", "pyproject.toml", ".claude/settings.json"):
        rule, owners = gp.owners_of(path, rules)
        assert rule != "*" and owners, path
    assert gp.owners_of("src/ml_stack/serve/x.py", rules)[0] == "*"


def test_codeowners_coverage_names_each_path_left_to_the_default_rule(gp, tmp_path, monkeypatch):
    (tmp_path / ".github").mkdir()
    (tmp_path / ".github" / "CODEOWNERS").write_text("* @someone\n/scripts/hooks/ @someone\n")
    monkeypatch.setattr(gp, "ROOT", tmp_path)
    found = gp.check_codeowners(["scripts/hooks/a", "scripts/gates/b", "budgets.json"])
    uncovered = [f for f in found if f.startswith("codeowners.uncovered")]
    assert any("scripts/gates/b" in f for f in uncovered)
    assert any("budgets.json" in f for f in uncovered)
    assert not any("scripts/hooks/a" in f for f in uncovered)


def test_a_missing_codeowners_file_is_a_finding(gp, tmp_path, monkeypatch):
    monkeypatch.setattr(gp, "ROOT", tmp_path)
    assert gp.check_codeowners([])[0].startswith("codeowners.file")


def test_this_repository_covers_every_protected_path(gp):
    assert gp.check_codeowners() == []


BAD_WORKFLOW = """\
on:
  pull_request_target:
permissions:
  contents: write
jobs:
  build:
    runs-on: ubuntu-latest
    permissions:
      id-token: write
      contents: write
    steps:
      - uses: actions/checkout@v4
      - run: echo ${{ secrets.SIGNING }}
  call:
    uses: ./.github/workflows/other.yml
    secrets: inherit
"""


def test_workflow_rules_name_each_unsafe_construct(gp, tmp_path, monkeypatch):
    (tmp_path / "bad.yml").write_text(BAD_WORKFLOW)
    monkeypatch.setattr(gp, "WORKFLOWS", tmp_path)
    found = {line.split()[0] for line in gp.check_workflows()}
    assert {"workflow.unpinned", "workflow.pr-target", "workflow.top-write", "workflow.id-token",
            "workflow.pr-write", "workflow.secret-scope", "workflow.secrets-inherit",
            "workflow.tag-guard", "workflow.base-gate"} <= found


def test_this_repository_workflows_follow_the_rules(gp):
    assert gp.check_workflows() == []
