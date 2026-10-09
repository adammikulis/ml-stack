#!/usr/bin/env python3
"""Rename the three agent roles: reader -> read-only, operator -> approve-first, runner -> plan-and-go.

Run from the repo root: python3 rename_roles.py [--check]. Idempotent. It edits only places where
the words name a role: every word in WHOLE files, and only lines matching ROLE_LINE in LINES
files. Other uses (PDF/page readers, job runners, policy operators) are never touched. Files
matching tests/test_*.py that pass a role are found by TEST_PATTERNS.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

MAP = {"reader": "read-only", "operator": "approve-first", "runner": "plan-and-go"}
WORD = re.compile(r"\b(reader|operator|runner)\b(?!\.\w)(?!, model)")

# every occurrence of the three words in these files is a role name
WHOLE = [
    "src/poolhouse/chat.py", "docs/agent-roles.md", "tests/test_roles.py", "tests/test_redteam_chat.py",
    "tests/test_chat_task.py", "tests/test_redteam_human_floor.py", "tests/test_memory_chat.py",
    "tests/test_reputation_safety.py", "tests/test_activity_feeds.py", "tests/test_redteam_memory.py",
    "tests/test_activity_viewer.py", "tests/test_requests_floor.py",
]
# only lines that name a role in these files
ROLE_LINE = re.compile(
    r"`(reader|operator|runner)`|roles? \(?(reader|operator|runner)|\b(reader|operator|runner),? (operator|runner)\b"
    r"|\b(?:operator|runner) role|^runner\), a project|role=\"(reader|operator|runner)\"|subject=\"reader\"")
LINES = [
    "README.md", "HANDOFF.md", "docs/FEATURES.md", "docs/memory.md", "docs/assistant-security.md",
    "docs/commands.md", "src/poolhouse/cli/reference.py", "docs/notes/agent-control-plane.md",
    "scripts/redteam_coverage.py", "tests/test_activity_log.py",
]
# literal rewrites for prose that names the role without the word "role"
EXTRA = [("an approved runner plan", "an approved plan under plan-and-go"),
         ("inside a plan-and-go's\nplan", "inside a plan-and-go\nplan")]
KEEP = "earlier role names"  # lines that document the old names are left as written
TEST_PATTERNS = ["tests/test_requests*.py", "tests/test_redteam_*.py", "tests/test_chat*.py"]


def swap(text: str) -> str:
    return WORD.sub(lambda m: MAP[m.group(1)], text)


def apply(path: Path, whole: bool) -> bool:
    old = path.read_text(encoding="utf-8")
    if whole:
        new = "".join(ln if KEEP in ln else swap(ln) for ln in old.splitlines(keepends=True))
    else:
        new = "".join(swap(ln) if ROLE_LINE.search(ln) else ln for ln in old.splitlines(keepends=True))
    for a, b in EXTRA:
        new = new.replace(a, b)
    if new != old:
        path.write_text(new, encoding="utf-8")
    return new != old


def main() -> int:
    root = Path.cwd()
    changed = []
    whole = set(WHOLE)
    for pat in TEST_PATTERNS:
        for p in root.glob(pat):
            # a test file that passes a role name, via role=/--role/ROLES[/"/role X"
            if re.search(r"role=\"(?:reader|operator|runner)\"|ROLES\[\"(?:reader|operator|runner)\"\]|/role (?:reader|operator|runner)",
                         p.read_text(encoding="utf-8")):
                whole.add(str(p.relative_to(root)))
    for rel in sorted(whole):
        if (root / rel).exists() and apply(root / rel, True):
            changed.append(rel)
    for rel in LINES:
        if (root / rel).exists() and apply(root / rel, False):
            changed.append(rel)
    docs = root / "docs/notes/undo-mode.md"
    if docs.exists() and apply(docs, False):
        changed.append(str(docs.relative_to(root)))
    print("changed:", *changed, sep="\n  " if changed else " ")
    return 1 if changed and "--check" in sys.argv else 0


if __name__ == "__main__":
    sys.exit(main())
