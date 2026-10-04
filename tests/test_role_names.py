"""The three role names are defined once and the old ones are not used as role names; saved rules
written under the old names are read under the new ones."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from ml_stack import chatpolicy, roles, rules as saved

ROOT = Path(__file__).resolve().parents[1]
OLD = ("reader", "operator", "runner")
SCANNED = ["src/ml_stack/roles.py", "src/ml_stack/rules.py", "src/ml_stack/chat.py",
           "docs/agent-roles.md", "docs/assistant-security.md", "README.md", "docs/FEATURES.md",
           "docs/memory.md", "docs/commands.md", "src/ml_stack/cli/reference.py",
           "docs/notes/agent-control-plane.md", "docs/notes/undo-mode.md"]
SCANNED += [str(p.relative_to(ROOT)) for pat in ("tests/test_roles.py", "tests/test_redteam_*.py",
                                                 "tests/test_requests*.py", "tests/test_chat*.py")
            for p in ROOT.glob(pat)]
ROLE_USE = re.compile(
    r"`(?:reader|operator|runner)`|[\"'](?:reader|operator|runner)[\"']|roles?\W+(?:reader|operator|runner)\b"
    r"|\b(?:reader|operator|runner) role\b|/role (?:reader|operator|runner)")
ALLOWED = {"tests/test_role_names.py"}


def test_the_table_uses_the_three_new_names():
    assert tuple(roles.ROLES) == ("read-only", "approve-first", "plan-and-go")
    assert (roles.DEFAULT, roles.TASK_DEFAULT) == ("approve-first", "plan-and-go")
    assert tuple(chatpolicy.LEGACY_ROLE_NAMES) == OLD


@pytest.mark.parametrize("rel", sorted(set(SCANNED) - ALLOWED))
def test_no_file_uses_an_old_word_as_a_role_name(rel):
    path = ROOT / rel
    if not path.exists():
        pytest.skip(rel)
    hits = [f"{rel}:{n}: {ln.strip()[:100]}" for n, ln in enumerate(path.read_text().splitlines(), 1)
            if ROLE_USE.search(ln)]
    assert not hits, "\n".join(hits)


def _file(tmp_path, version, role):
    path = tmp_path / "r.json"
    path.write_text(json.dumps({"schema_version": version, "rules": [
        {"tool": "serve_up", "match": {"model": "quince-2b.gguf"}, "verdict": "always", "role": role}]}))
    path.chmod(0o600)
    return path


@pytest.mark.parametrize("old,new", list(chatpolicy.LEGACY_ROLE_NAMES.items()))
def test_a_rule_saved_under_an_old_role_name_applies_to_the_new_one_and_is_rewritten(tmp_path, old, new):
    path = _file(tmp_path, 1, old)
    rules = saved.Rules(path)
    args = {"model": "quince-2b.gguf"}
    assert not rules.broken and rules.rules[0].role == new
    assert rules.covers("serve_up", args, new, False).verdict == "always"
    rules.add("serve_up", {"model": "other.gguf"}, "never", new)
    data = json.loads(path.read_text())
    assert data["schema_version"] == saved.SCHEMA_VERSION
    assert [r["role"] for r in data["rules"]] == [new, new]


def test_an_unknown_role_name_in_a_saved_rule_is_refused_not_dropped(tmp_path):
    rules = saved.Rules(_file(tmp_path, 2, "root"))
    assert "role 'root'" in rules.broken and rules.rules == []
    with pytest.raises(ValueError, match="no role"):
        roles.get("operator")
