"""Fail on a known vulnerability in the shipped dependencies, minus the accepted findings.

Reads the JSON report of ``pip-audit --format json``, ``npm audit --json`` or ``cargo audit --json`` from a
file. ``.github/pip-audit-allow.json`` lists the findings the owner has accepted, each with a reason and an
expiry date (and an optional ``ecosystem``: ``pip`` when absent, ``npm`` or ``cargo``). An entry past its expiry
stops suppressing its finding (and is reported), so an acceptance has to be renewed on purpose. An advisory is
matched by its own id or by any alias it lists (a CVE for a GHSA).

    pip-audit --format json --output pip-audit.json || true
    python scripts/audit_gate.py pip-audit.json
    npm audit --json > npm-audit.json || true
    python scripts/audit_gate.py --kind npm npm-audit.json
    cargo audit --json > cargo-audit.json || true
    python scripts/audit_gate.py --kind cargo cargo-audit.json

A report that is not the tool's JSON (a crash, an empty file) fails the gate rather than passing it.
"""

from __future__ import annotations

import json
import re
import sys
from datetime import date
from pathlib import Path

ALLOW = Path(__file__).resolve().parent.parent / ".github" / "pip-audit-allow.json"
KINDS = ("pip", "npm", "cargo")
Finding = tuple[str, str, tuple[str, ...]]
"""(package, version, (advisory id, *aliases))."""


def load_allow(path: Path = ALLOW) -> list[dict[str, str]]:
    entries = json.loads(path.read_text(encoding="utf-8")).get("allow", [])
    for e in entries:
        missing = {"id", "package", "reason", "expires"} - set(e)
        if missing:
            raise ValueError(f"{path.name}: entry {e} lacks {sorted(missing)}")
        date.fromisoformat(e["expires"])
        if e.get("ecosystem", "pip") not in KINDS:
            raise ValueError(f"{path.name}: entry {e['id']} has an unknown ecosystem")
    return entries


def pip_findings(report: dict) -> list[Finding]:
    """Every vulnerability in a pip-audit JSON report."""
    return [
        (dep["name"], dep.get("version", ""), (vuln["id"], *vuln.get("aliases", [])))
        for dep in report["dependencies"]
        for vuln in dep.get("vulns", [])
    ]


def npm_findings(report: dict) -> list[Finding]:
    """Every advisory of an ``npm audit --json`` report (npm 7+): the object entries of each ``via`` list; a
    string entry only says the package is vulnerable because of another, which has its own entry."""
    if "vulnerabilities" not in report:
        raise ValueError("no 'vulnerabilities' key")
    out: list[Finding] = []
    for name, row in report["vulnerabilities"].items():
        for via in row.get("via", []):
            if not isinstance(via, dict):
                continue
            advisory = re.findall(r"GHSA-[0-9a-z-]+", str(via.get("url", "")))
            ids = tuple(advisory) or (str(via.get("source") or via.get("title") or "unknown"),)
            out.append((str(via.get("name") or name), str(row.get("range", "")), ids))
    return out


def cargo_findings(report: dict) -> list[Finding]:
    """Every vulnerability of a ``cargo audit --json`` report (warnings such as unmaintained crates are not
    vulnerabilities and are not gated here)."""
    return [
        (v["package"]["name"], v["package"].get("version", ""), (v["advisory"]["id"], *v["advisory"].get("aliases", [])))
        for v in report["vulnerabilities"]["list"]
    ]


READERS = {"pip": pip_findings, "npm": npm_findings, "cargo": cargo_findings}


def findings(report: dict, kind: str = "pip") -> list[Finding]:
    try:
        return READERS[kind](report)
    except (KeyError, TypeError, AttributeError, ValueError) as exc:
        raise ValueError(f"not a {kind} audit report: {type(exc).__name__}: {exc}") from None


def gate(report: dict, allow: list[dict[str, str]], today: date, kind: str = "pip") -> list[str]:
    """The problems that fail the job: unaccepted findings and lapsed acceptances that still match one."""
    problems: list[str] = []
    mine = [e for e in allow if e.get("ecosystem", "pip") == kind]
    live = {(e["package"].lower(), e["id"]) for e in mine if date.fromisoformat(e["expires"]) >= today}
    lapsed = {(e["package"].lower(), e["id"]): e["expires"] for e in mine if date.fromisoformat(e["expires"]) < today}
    for name, version, ids in findings(report, kind):
        if any((name.lower(), i) in live for i in ids):
            continue
        late = next((lapsed[(name.lower(), i)] for i in ids if (name.lower(), i) in lapsed), "")
        why = f" (accepted until {late}, expired)" if late else ""
        problems.append(f"{name} {version}: {ids[0]}{why}")
    return problems


def main(argv: list[str]) -> int:
    kind = "pip"
    if len(argv) == 3 and argv[0] == "--kind" and argv[1] in KINDS:
        kind, argv = argv[1], argv[2:]
    if len(argv) != 1:
        sys.stderr.write("usage: audit_gate.py [--kind pip|npm|cargo] audit.json\n")
        return 2
    try:
        report = json.loads(Path(argv[0]).read_text(encoding="utf-8"))
        problems = gate(report, load_allow(), date.today(), kind)
    except (OSError, ValueError) as exc:
        sys.stdout.write(f"audit report unusable, failing: {exc}\n")
        return 1
    for p in problems:
        sys.stdout.write(f"vulnerable: {p}\n")
    sys.stdout.write(f"{len(problems)} unaccepted vulnerabilities ({kind})\n")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
