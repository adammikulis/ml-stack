"""scripts/github-protection-apply: a dry run by default, the order of the changes, the backup, and
the person-only refusal."""

from __future__ import annotations

import copy
import importlib.machinery
import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

from tests.github_support import GOOD, REPO, ruleset, serve

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "github-protection-apply"
TERMINAL = (True, True)
APPLY = ["--repo", REPO, "--apply", "--identity-ready"]


@pytest.fixture
def ap(tmp_path, monkeypatch):
    for name in ("CLAUDECODE", "POOLHOUSE_AGENT", "POOLHOUSE_NONINTERACTIVE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("POOLHOUSE_HOME", str(tmp_path / "home"))
    loader = importlib.machinery.SourceFileLoader("github_protection_apply", str(SCRIPT))
    module = importlib.util.module_from_spec(importlib.util.spec_from_loader(loader.name, loader))
    monkeypatch.setitem(sys.modules, loader.name, module)
    loader.exec_module(module)
    return module


def before_state():
    """A repository as it is today: the old ruleset, a repository secret, nothing on the environment."""
    replies = copy.deepcopy(GOOD)
    replies[f"repos/{REPO}/rulesets"].append({"id": 4, "name": "default"})
    replies[f"repos/{REPO}/rulesets/4"] = ruleset(
        "default", "branch", ["~DEFAULT_BRANCH", "refs/heads/*dev"],
        [{"type": "update"}], [{"actor_id": 5, "actor_type": "RepositoryRole", "bypass_mode": "always"}])
    replies[f"repos/{REPO}/actions/secrets"] = {"secrets": [{"name": "RELEASE_SIGNING_KEY"}]}
    replies[f"repos/{REPO}/environments/release/deployment-branch-policies"] = {"branch_policies": []}
    replies[f"repos/{REPO}"]["allow_auto_merge"] = True
    return replies


def calls(log):
    return [line.split(" ", 2) for line in log.read_text().splitlines()]


def writes(log):
    return [(m, p) for m, p, *_ in calls(log) if m != "GET"]


def test_without_apply_nothing_is_changed_and_the_diff_is_printed(ap, tmp_path, monkeypatch, capsys):
    log = serve(tmp_path, monkeypatch, before_state())
    assert ap.main(["--repo", REPO], TERMINAL) == 0
    out = capsys.readouterr().out
    assert writes(log) == []
    assert "== release environment" in out and "== remove the ruleset named default" in out
    assert re.search(r'^\s+-\s+"allow_auto_merge": true', out, re.M)
    assert re.search(r'^\s+\+\s+"allow_auto_merge": false', out, re.M)
    assert "agent identity comes first" in out
    assert not (tmp_path / "home" / "github-protection").exists()


def test_apply_without_the_identity_confirmed_changes_nothing(ap, tmp_path, monkeypatch, capsys):
    log = serve(tmp_path, monkeypatch, before_state())
    assert ap.main(["--repo", REPO, "--apply"], TERMINAL) == 1
    assert writes(log) == []
    assert "identity" in capsys.readouterr().err


def test_the_changes_are_made_in_order_after_a_backup(ap, tmp_path, monkeypatch, capsys):
    log = serve(tmp_path, monkeypatch, before_state())
    assert ap.main(APPLY, TERMINAL) == 1
    base = f"repos/{REPO}"
    assert writes(log) == [
        ("PUT", f"{base}/environments/release"),
        ("POST", f"{base}/environments/release/deployment-branch-policies"),
        ("DELETE", f"{base}/actions/secrets/RELEASE_SIGNING_KEY"),
        ("PUT", f"{base}/rulesets/1"), ("PUT", f"{base}/rulesets/2"), ("PUT", f"{base}/rulesets/3"),
        ("DELETE", f"{base}/rulesets/4"),
        ("PUT", f"{base}/actions/permissions/workflow"), ("PUT", f"{base}/actions/permissions"),
        ("PUT", f"{base}/actions/permissions/selected-actions"),
        ("PUT", f"{base}/actions/permissions/fork-pr-contributor-approval"),
        ("PATCH", base)]
    first_write = next(i for i, c in enumerate(calls(log)) if c[0] != "GET")
    assert all(c[0] == "GET" for c in calls(log)[:first_write])
    saved = list((tmp_path / "home" / "github-protection").glob("backup-*.json"))
    assert len(saved) == 1
    body = json.loads(saved[0].read_text())
    assert body["repo"] == REPO and body["reads"][f"{base}/rulesets/4"]["name"] == "default"
    out = capsys.readouterr().out
    assert f"--restore {saved[0]}" in out
    assert "finding" in out


def test_the_environment_body_names_the_owner_and_main_only(ap, tmp_path, monkeypatch):
    log = serve(tmp_path, monkeypatch, before_state())
    ap.main(APPLY, TERMINAL)
    put = next(c for c in calls(log) if c[0] == "PUT" and c[1].endswith("environments/release"))
    assert json.loads(put[2]) == {
        "reviewers": [{"type": "User", "id": 7}], "can_admins_bypass": False,
        "deployment_branch_policy": {"protected_branches": False, "custom_branch_policies": True}}
    post = next(c for c in calls(log) if c[0] == "POST")
    assert json.loads(post[2]) == {"name": "main", "type": "branch"}


def test_a_repository_already_at_the_target_ends_with_no_findings(ap, tmp_path, monkeypatch, capsys):
    serve(tmp_path, monkeypatch, GOOD)
    assert ap.main(APPLY, TERMINAL) == 0
    assert capsys.readouterr().out.strip().endswith("0 findings")


def test_it_stops_before_the_rulesets_when_the_key_is_not_in_the_environment(
        ap, tmp_path, monkeypatch, capsys):
    replies = before_state()
    replies[f"repos/{REPO}/environments/release/secrets"] = {"secrets": []}
    log = serve(tmp_path, monkeypatch, replies)
    assert ap.main(APPLY, TERMINAL) == 1
    assert all("/environments/" in p for _m, p in writes(log))
    assert "release-key" in capsys.readouterr().err


def test_it_stops_at_the_first_error(ap, tmp_path, monkeypatch, capsys):
    base = f"repos/{REPO}"
    log = serve(tmp_path, monkeypatch, before_state(), fail=f"{base}/rulesets/1")
    assert ap.main(APPLY, TERMINAL) == 1
    assert writes(log)[-1] == ("PUT", f"{base}/rulesets/1")
    assert "422" in capsys.readouterr().err


def test_the_owner_review_switch_changes_only_the_main_ruleset(ap):
    plain, review = ap.main_ruleset(False), ap.main_ruleset(True)
    assert plain["rules"][2]["parameters"]["require_code_owner_review"] is False
    assert review["rules"][2]["parameters"]["require_code_owner_review"] is True
    assert plain["rules"][3] == review["rules"][3]


def test_restore_puts_back_the_old_ruleset_and_removes_the_new_ones(ap, tmp_path, monkeypatch, capsys):
    before = before_state()
    serve(tmp_path, monkeypatch, before)
    ap.main(APPLY, TERMINAL)
    saved = next((tmp_path / "home" / "github-protection").glob("backup-*.json"))
    after = copy.deepcopy(GOOD)
    after[f"repos/{REPO}/rulesets"] = [{"id": 1, "name": "main"}, {"id": 5, "name": "extra"}]
    after[f"repos/{REPO}/rulesets/5"] = ruleset("extra", "branch", ["x"], [])
    (tmp_path / "again").mkdir()
    log = serve(tmp_path / "again", monkeypatch, after)
    assert ap.main(["--restore", str(saved), "--apply"], TERMINAL) == 0
    done = writes(log)
    base = f"repos/{REPO}"
    assert ("DELETE", f"{base}/rulesets/5") in done
    assert ("POST", f"{base}/rulesets") in done
    assert done[-1] == ("PATCH", base)


def test_an_agent_marker_refuses_before_any_request(ap, tmp_path, monkeypatch, capsys):
    log = serve(tmp_path, monkeypatch, GOOD)
    monkeypatch.setenv("CLAUDECODE", "1")
    assert ap.main(APPLY, TERMINAL) == 1
    assert log.read_text() == ""
    assert "agent" in capsys.readouterr().err
    assert not (tmp_path / "home" / "github-protection").exists()


def test_no_terminal_refuses_before_any_request(ap, tmp_path, monkeypatch, capsys):
    log = serve(tmp_path, monkeypatch, GOOD)
    assert ap.main(APPLY, (False, False)) == 1
    assert log.read_text() == ""
    assert "terminal" in capsys.readouterr().err


def test_no_library_code_reaches_the_apply_script():
    for path in (ROOT / "src").rglob("*.py"):
        assert "github-protection-apply" not in path.read_text(encoding="utf-8"), path


def test_the_commands_in_the_document_are_the_bodies_the_script_applies(ap):
    text = (ROOT / "docs" / "github-protection.md").read_text(encoding="utf-8")
    blocks = [json.loads(b) for b in re.findall(r"<<'JSON'\n(.*?)\nJSON", text, re.S)]
    named = {b["name"]: b for b in blocks if "name" in b and "rules" in b}
    assert named["main"] == ap.main_ruleset(False)
    assert named["development"] == ap.dev_ruleset()
    assert named["release-tags"] == ap.tag_ruleset(False)
    assert any(b == ap.REPOSITORY for b in blocks)
    selected = next(b for b in blocks if "patterns_allowed" in b)
    assert selected == ap.ACTIONS[2][3]
