"""A `gh` on PATH that answers GET requests from canned JSON and records every call."""

from __future__ import annotations

import json
import os

REPO = "ownerlogin/poolhouse-test"


def ruleset(name, target, include, rules, bypass=()):
    return {"name": name, "target": target, "enforcement": "active", "bypass_actors": list(bypass),
            "conditions": {"ref_name": {"include": include, "exclude": []}}, "rules": rules}


GOOD = {
    f"repos/{REPO}/rulesets": [{"id": 1, "name": "main"}, {"id": 2, "name": "development"},
                                  {"id": 3, "name": "release-tags"}],
    f"repos/{REPO}/rulesets/1": ruleset("main", "branch", ["~DEFAULT_BRANCH"], [
        {"type": "deletion"}, {"type": "non_fast_forward"},
        {"type": "pull_request", "parameters": {
            "required_approving_review_count": 0, "require_code_owner_review": False,
            "dismiss_stale_reviews_on_push": False, "require_last_push_approval": False}},
        {"type": "required_status_checks", "parameters": {"required_status_checks": [
            {"context": "gates"}, {"context": "privacy"}, {"context": "licenses"},
            {"context": "test (ubuntu-latest, 3.13, --slow)"}]}}]),
    f"repos/{REPO}/rulesets/2": ruleset("development", "branch", ["refs/heads/*dev"], [
        {"type": "deletion"}, {"type": "non_fast_forward"}]),
    f"repos/{REPO}/rulesets/3": ruleset("release-tags", "tag", ["refs/tags/v*"], [
        {"type": "deletion"}, {"type": "non_fast_forward"}, {"type": "update"},
        {"type": "creation"}]),
    f"repos/{REPO}/environments/release": {
        "can_admins_bypass": False,
        "protection_rules": [{"type": "required_reviewers", "reviewers": [
            {"type": "User", "reviewer": {"login": "ownerlogin"}}]}],
        "deployment_branch_policy": {"protected_branches": False, "custom_branch_policies": True}},
    f"repos/{REPO}/environments/release/deployment-branch-policies": {"branch_policies": [
        {"name": "main", "type": "branch"}]},
    f"repos/{REPO}/actions/secrets": {"secrets": []},
    f"repos/{REPO}/environments/release/secrets": {"secrets": [{"name": "RELEASE_SIGNING_KEY"}]},
    f"repos/{REPO}/actions/permissions/workflow": {
        "default_workflow_permissions": "read", "can_approve_pull_request_reviews": False},
    f"repos/{REPO}/actions/permissions": {"allowed_actions": "selected", "sha_pinning_required": True},
    f"repos/{REPO}/actions/permissions/fork-pr-contributor-approval": {
        "approval_policy": "all_external_contributors"},
    f"repos/{REPO}": {
        "allow_auto_merge": False, "permissions": {"admin": False, "push": True},
        "security_and_analysis": {k: {"status": "enabled"} for k in (
            "secret_scanning", "secret_scanning_push_protection", "dependabot_security_updates")}},
    "user": {"login": "agent-app", "id": 7},
}

GH = """#!/bin/sh
if [ "$2" = -X ]; then
    method=$3; path=$4; body=$(cat)
    printf '%s %s %s\\n' "$method" "$path" "$body" >> "$GH_LOG"
    case " $GH_FAIL " in *" $path "*) echo "gh: HTTP 422" >&2; exit 1;; esac
    echo '{}'; exit 0
fi
printf 'GET %s\\n' "$2" >> "$GH_LOG"
if [ "$1" = repo ]; then echo "$GH_REPO"; exit 0; fi
name=$(printf '%s' "$2" | tr '/' '_')
if [ -f "$GH_REPLIES/$name.json" ]; then cat "$GH_REPLIES/$name.json"; exit 0; fi
echo "gh: Not Found (HTTP 404)" >&2
exit 1
"""


def serve(tmp_path, monkeypatch, replies, fail=""):
    """Put a `gh` on PATH that answers each GET from ``replies`` and logs its arguments."""
    bin_dir, store = tmp_path / "bin", tmp_path / "replies"
    bin_dir.mkdir()
    store.mkdir()
    gh = bin_dir / "gh"
    gh.write_text(GH)
    gh.chmod(0o755)
    for path, body in replies.items():
        (store / f"{path.replace('/', '_')}.json").write_text(json.dumps(body))
    log = tmp_path / "gh.log"
    log.write_text("")
    monkeypatch.setenv("GH_LOG", str(log))
    monkeypatch.setenv("GH_REPLIES", str(store))
    monkeypatch.setenv("GH_REPO", REPO)
    monkeypatch.setenv("GH_FAIL", fail)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    return log


