"""Every ML_STACK_* variable the docs name is read (or set) somewhere in the shipped source.

A documented switch that the code does not know silently does nothing; this caught CHANGELOG.md naming
``ML_STACK_UNMANAGED`` for the variable the code reads as ``ML_STACK_ADOPT_UNMANAGED``.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = [ROOT / "CHANGELOG.md", ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md"))]
SOURCE_DIRS = ["src", "scripts", "packaging", "app/src", "app/src-tauri/src", "contracts"]
SOURCE_FILES = ["pyproject.toml"]
TEXT = {".py", ".rs", ".ts", ".tsx", ".js", ".sh", ".ps1", ".toml", ".json", ".yml", ".yaml", ".nsi", ".cmd", ".md", ""}
NAME = re.compile(r"\bML_STACK_[A-Z0-9_]*[A-Z0-9]\b")

EXTERNAL: dict[str, str] = {
    "ML_STACK_FROZEN_BINARY": "standalone daemon input for packaging conversation tests",
    "ML_STACK_FROZEN_CODING_BINARY": "standalone daemon input for packaging coding tests",
    "ML_STACK_ACCEPTANCE_DECIDE_CACHE": "installed decide cache input for native gym acceptance tests",
    "ML_STACK_ACCEPTANCE_DECISION_DEVICE": "accelerator selection for native gym acceptance tests",
    "ML_STACK_ACCEPTANCE_VISION": "opt-in native vision gym acceptance test",
    "ML_STACK_TEST_GGUF": "opt-in input of the real-model tests, read by tests/ only",
    "ML_STACK_TEST_SSHD": "opt-in switch of the localhost-sshd onboarding test, read by tests/ only",
    "ML_STACK_PUSH_MAIN": "documented as inert: the Bash guard no longer reads it; release-main approval replaces it",
    "ML_STACK_FLEET_TLS": "documented as removed: the pool has no TLS-off switch (docs/pool-encryption.md); tests/ sets it to prove it is ignored",
    "ML_STACK_MANUAL_DIALOG": "opt-in switch of the manual notification-dialog test, read by tests/ only",
}
"""Documented variables that are intentionally not in the shipped source, each with the reason."""


def source_text() -> str:
    parts = []
    files = [ROOT / f for f in SOURCE_FILES]
    for d in SOURCE_DIRS:
        files += [p for p in (ROOT / d).rglob("*") if p.is_file() and p.suffix in TEXT and "node_modules" not in p.parts]
    for p in files:
        try:
            parts.append(p.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError):
            continue
    return "\n".join(parts)


def documented() -> dict[str, list[str]]:
    found: dict[str, list[str]] = {}
    for doc in DOCS:
        for name in sorted(set(NAME.findall(doc.read_text(encoding="utf-8")))):
            found.setdefault(name, []).append(doc.relative_to(ROOT).as_posix())
    return found


def test_every_documented_ml_stack_variable_exists_in_the_source():
    source = source_text()
    known = set(NAME.findall(source))
    docs = documented()
    assert len(docs) > 20, "the scan found almost nothing; the pattern or the paths are wrong"
    assert len(known) > 20
    missing = {n: where for n, where in docs.items() if n not in known and n not in EXTERNAL}
    assert not missing, f"documented but never read or set in the source: {missing}"


def test_every_allow_listed_external_variable_has_a_reason_and_is_still_documented():
    docs = documented()
    for name, reason in EXTERNAL.items():
        assert reason.strip()
        assert name in docs, f"{name} is no longer documented: drop it from EXTERNAL"


def test_the_external_variables_are_really_read_by_a_test():
    tests = "\n".join(p.read_text(encoding="utf-8") for p in (ROOT / "tests").glob("test_*.py")
                      if p.name != Path(__file__).name)
    for name in EXTERNAL:
        assert name in tests, f"{name} is read by no test: it is documented for nothing"
