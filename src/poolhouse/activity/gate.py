"""Test results as evidence: a run is recorded against the hash of the tree it ran on, so
"the full tier is green on tree X" is something the tool wrote, not something said."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import psutil

from poolhouse.activity import writer
from poolhouse.activity.schema import Entry, Said, build
from poolhouse.files import write_json

__all__ = ["Run", "evidence", "record_run", "tree_hash"]


def tree_hash(root: Path) -> str:
    """The git tree hash of everything in ``root`` that is not ignored, working tree
    included, computed in a throwaway index; empty when ``root`` is not a git checkout."""
    with tempfile.TemporaryDirectory() as scratch:
        env = {**os.environ, "GIT_INDEX_FILE": str(Path(scratch) / "index")}
        for step in (["add", "-A"], ["write-tree"]):
            done = subprocess.run(["git", "-C", str(root), *step], env=env, capture_output=True,
                                  text=True, check=False)
            if done.returncode:
                return ""
        return done.stdout.strip()


def _counts(junit: Path) -> dict[str, int | float]:
    node = ET.parse(junit).getroot()  # noqa: S314 - the file pytest just wrote
    suite = node if node.tag == "testsuite" else node.find("testsuite")
    attrs = suite.attrib if suite is not None else {}
    total, bad = int(attrs.get("tests", 0)), int(attrs.get("failures", 0))
    errors, skipped = int(attrs.get("errors", 0)), int(attrs.get("skipped", 0))
    return {"passed": total - bad - errors - skipped, "failed": bad, "errors": errors,
            "skipped": skipped}


@dataclass(frozen=True, slots=True)
class Run:
    """One pytest run: where, which tier, the command, the junit file it wrote, how it ended."""

    root: Path
    tier: str
    command: list[str]
    junit: Path
    exit_code: int
    seconds: float
    tree: str = ""


def record_run(run: Run, artifact: Path | None = None, *, destination=None, boundary: dict | None = None) -> bool:
    """Record one pytest run: tree, tier, a hash of the command, counts and duration."""
    root, tier, exit_code = run.root, run.tier, run.exit_code
    try:
        counts = _counts(run.junit)
    except (OSError, ET.ParseError, ValueError):
        counts = {}
    tree = run.tree or tree_hash(root)
    words = [w for w in run.command if not w.startswith("--junitxml") and not w.startswith("-n")]
    digest = hashlib.sha256(" ".join(words).encode()).hexdigest()[:16]
    said: Said = {"subject": f"tree:{tree[:12]}", "outcome": "pass" if exit_code == 0 else "fail",
                  "refs": {"tree": tree, "command": digest, "tier": tier},
                  "meta": {**counts, "seconds": round(run.seconds, 1), "exit": exit_code}}
    if artifact is not None:
        payload = build("test.result", ts=time.time(), actor="system",
                        session=f"test-{os.getpid()}", **said)
        payload["run"] = {"requested_command": list(run.command), "runner_interpreter": sys.executable,
                          "executed_interpreter": run.command[0] if run.command else "",
                          "root": str(root.resolve()), "tree_after": tree_hash(root)}
        payload["process"] = {"pid": os.getpid(),
                              "started": psutil.Process().create_time()}
        if boundary is not None:
            payload["boundary"] = boundary
        if destination is None:
            write_json(artifact, payload)
        else:
            destination.write(json.dumps(payload, sort_keys=True).encode() + b"\n")
        return True
    return writer.record("test.result", actor="system", **said)


def evidence(tree: str, tier: str = "") -> Entry | None:
    """The latest passing record for ``tree`` (and ``tier``), else None."""
    found = None
    for e in writer.log().entries():
        if isinstance(e, Entry) and e.kind == "test.result" and e.refs.get("tree") == tree \
                and e.outcome == "pass" and tier in ("", e.refs.get("tier")):
            found = e
    return found
