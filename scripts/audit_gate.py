"""Fail on a known vulnerability in the installed packages, minus the accepted findings.

``pip-audit --format json`` output is read from a file; ``.github/pip-audit-allow.json`` lists the findings the
owner has accepted, each with a reason and an expiry date. An entry past its expiry stops suppressing its
finding (and is reported), so an acceptance has to be renewed on purpose.

    pip-audit --format json --output pip-audit.json || true
    python scripts/audit_gate.py pip-audit.json
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

ALLOW = Path(__file__).resolve().parent.parent / ".github" / "pip-audit-allow.json"


def load_allow(path: Path = ALLOW) -> list[dict[str, str]]:
    entries = json.loads(path.read_text(encoding="utf-8")).get("allow", [])
    for e in entries:
        missing = {"id", "package", "reason", "expires"} - set(e)
        if missing:
            raise ValueError(f"{path.name}: entry {e} lacks {sorted(missing)}")
        date.fromisoformat(e["expires"])
    return entries


def findings(report: dict) -> list[tuple[str, str, str]]:
    """(package, version, vulnerability id) for every vulnerability in a pip-audit JSON report."""
    return [
        (dep["name"], dep.get("version", ""), vuln["id"])
        for dep in report.get("dependencies", [])
        for vuln in dep.get("vulns", [])
    ]


def gate(report: dict, allow: list[dict[str, str]], today: date) -> list[str]:
    """The problems that fail the job: unaccepted findings and lapsed acceptances that still match one."""
    problems: list[str] = []
    live = {(e["package"].lower(), e["id"]) for e in allow if date.fromisoformat(e["expires"]) >= today}
    lapsed = {(e["package"].lower(), e["id"]): e["expires"] for e in allow if date.fromisoformat(e["expires"]) < today}
    for name, version, vid in findings(report):
        key = (name.lower(), vid)
        if key in live:
            continue
        why = f" (accepted until {lapsed[key]}, expired)" if key in lapsed else ""
        problems.append(f"{name} {version}: {vid}{why}")
    return problems


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        sys.stderr.write("usage: audit_gate.py pip-audit.json\n")
        return 2
    problems = gate(json.loads(Path(argv[0]).read_text(encoding="utf-8")), load_allow(), date.today())
    for p in problems:
        sys.stdout.write(f"vulnerable: {p}\n")
    sys.stdout.write(f"{len(problems)} unaccepted vulnerabilities\n")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
