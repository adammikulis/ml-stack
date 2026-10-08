"""Checks a landing batch is verified with: choosing them from the diff, running them through
``scripts/test``, reusing recorded passes, and the full-tier background lane."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import affected
import land_git as lg
import testslots

from ml_stack.activity.gate import Run, evidence, record_run, tree_hash

FULL_FILES = 120
FULL_LINES = 4000
INFRA = re.compile(r"^(tests/conftest\.py|scripts/test[^/]*|scripts/testslots[^/]*|pyproject\.toml|packaging/.*)$")
DOC = re.compile(r"^(HANDOFF\.md|.*\.md|docs/.*)$")
FAILED = re.compile(r"^(?:FAILED|ERROR) (\S+?)(?:::|\s|$)", re.M)
TAIL = 12


@dataclass
class Check:
    """One verification step: a name, the tier it is recorded under and the command."""

    name: str
    tier: str
    argv: list[str]
    files: list[str] = field(default_factory=list)


@dataclass
class Result:
    """How a check ended: pass, fail, reused, baseline or skipped."""

    name: str
    status: str
    seconds: float = 0.0
    tail: str = ""
    failed_files: list[str] = field(default_factory=list)
    detail: str = ""

    def view(self) -> dict:
        """A JSON-ready summary."""
        return {"check": self.name, "status": self.status, "seconds": round(self.seconds, 1),
                "detail": self.detail}


def classify(paths: list[str]) -> str:
    """``none``, ``docs`` or ``code`` for a set of changed paths."""
    if not paths:
        return "none"
    return "docs" if all(DOC.match(p) for p in paths) else "code"


def needs_full(root: Path, base: str, paths: list[str], unmapped: list[str]) -> str:
    """Why a background full run is wanted for this diff, or an empty string."""
    infra = [p for p in paths if INFRA.match(p)]
    if infra:
        return f"shared infrastructure touched: {infra[0]}"
    if unmapped:
        return f"no selector maps {unmapped[0]}"
    stat = lg.lines(root, "diff", "--numstat", base, "HEAD")
    lines = sum(int(a) + int(b) for a, b, *_ in (r.split("\t") for r in stat) if a.isdigit() and b.isdigit())
    if len(paths) > FULL_FILES or lines > FULL_LINES:
        return f"{len(paths)} files, {lines} lines exceed the combined-diff threshold"
    return ""


def preview(root: Path, files: list[str]) -> tuple[list[str], str, str]:
    """Check names, diff kind and full-run reason predicted from the planned branches' files."""
    kind = classify(files)
    if kind == "docs":
        return ["budgets", "clean-diff"], kind, ""
    if kind == "none":
        return [], kind, ""
    pick = affected.select(root, files, frozenset())
    names = ["gate", *(["affected"] if pick.files else [])]
    infra = [p for p in files if INFRA.match(p)]
    reason = (f"shared infrastructure touched: {infra[0]}" if infra else
              f"no selector maps {pick.unmapped[0]}" if pick.unmapped else
              f"{len(files)} files exceed the combined-diff threshold" if len(files) > FULL_FILES else "")
    return names, kind, reason


def build_checks(wt: Path, base: str, workers: int) -> tuple[list[Check], str, str]:
    """The checks for the diff of ``wt`` against ``base``, the diff kind and the reason for a full run."""
    changed = lg.lines(wt, "diff", "--name-only", "--no-renames", base, "HEAD")
    kind = classify(changed)
    py = sys.executable
    if kind == "none":
        return [], kind, ""
    if kind == "docs":
        return [Check("budgets", "land:budgets", [py, "scripts/budgets"]),
                Check("clean-diff", "land:clean-diff", ["git", "diff", "--check", base, "HEAD"])], kind, ""
    gone = frozenset(p for p in changed if not (wt / p).exists())
    pick = affected.select(wt, changed, gone)
    files = sorted(pick.files)
    checks = [Check("gate", "land:gate", [py, "scripts/test", "gate"])]
    if files:
        digest = hashlib.sha256(" ".join(files).encode()).hexdigest()[:12]
        checks.append(Check("affected", f"land:affected:{digest}",
                            [py, "scripts/test", "all", "-n", str(workers), *files], files))
    return checks, kind, needs_full(wt, base, changed, pick.unmapped)


def run_command(wt: Path, argv: list[str]) -> tuple[int, str]:
    """Run ``argv`` in ``wt`` and return its status and combined output."""
    done = subprocess.run(argv, cwd=wt, capture_output=True, text=True, check=False,
                          env={**os.environ, "PYTHONPATH": os.pathsep.join(
                              (str(wt / "src"), str(wt / "scripts"), os.environ.get("PYTHONPATH", "")))})
    return done.returncode, done.stdout + done.stderr


def execute(wt: Path, check: Check, tree: str) -> Result:
    """Run one check unless this tree already passed it; record a pass as evidence."""
    prior = evidence(tree, check.tier) if tree else None
    if prior is not None:
        return Result(check.name, "reused", detail=f"reused from {prior.id}")
    began = time.monotonic()
    status, text = run_command(wt, check.argv)
    seconds = time.monotonic() - began
    if status == 0:
        record_run(Run(wt, check.tier, check.argv, wt / ".land-no-junit", 0, seconds, tree))
        return Result(check.name, "pass", seconds)
    tail = "\n".join(text.strip().splitlines()[-TAIL:])
    return Result(check.name, "fail", seconds, tail, sorted(set(FAILED.findall(text))))


def restrict(check: Check, failed: list[str]) -> Check:
    """The check narrowed to the failing test files when it runs tests."""
    if not check.files or not failed:
        return check
    head = check.argv[: check.argv.index("all") + 3]
    return Check(check.name, check.tier, [*head, *failed], failed)


def probe(wt: Path, check: Check) -> bool:
    """Whether ``check`` passes in ``wt``; test files missing there count as nothing to run."""
    if check.files:
        here = [f for f in check.files if (wt / f).exists()]
        if not here:
            return True
        check = Check(check.name, check.tier, [*check.argv[: len(check.argv) - len(check.files)], *here], here)
    return run_command(wt, check.argv)[0] == 0


def other_full_run(common: Path) -> str:
    """A description of a full run already in flight, or an empty string."""
    lock = common / "land" / "full.lock"
    try:
        pid = int(lock.read_text(encoding="utf-8").split()[0])
        os.kill(pid, 0)
        return f"land full run pid {pid}"
    except (OSError, ValueError, IndexError):
        pass
    try:
        busy = testslots.status()
    except (OSError, ValueError):
        return ""
    for slot in [*busy.get("running", []), *busy.get("waiting", [])]:
        if "full" in str(slot.get("label", "")):
            return f"broker: {slot['label']}"
    return ""


def start_full(wt: Path, common: Path) -> dict:
    """Start one full-tier run in the background through ``scripts/test``; report where its log is."""
    held = other_full_run(common)
    if held:
        return {"status": "refused", "reason": held}
    folder = common / "land"
    folder.mkdir(exist_ok=True)
    log = folder / f"full-{int(time.time())}.log"
    child = subprocess.Popen([sys.executable, "scripts/test", "full"], cwd=wt, stdin=subprocess.DEVNULL,
                             stdout=log.open("ab"), stderr=subprocess.STDOUT, start_new_session=True)
    (folder / "full.lock").write_text(f"{child.pid}\n", encoding="utf-8")
    return {"status": "started", "pid": child.pid, "log": str(log), "tree": tree_hash(wt)}


def summary_json(data: dict) -> str:
    """The one-line machine-readable summary."""
    return json.dumps(data, sort_keys=True, separators=(",", ":"))
