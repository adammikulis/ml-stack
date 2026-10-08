"""Run explicit test files with per-file reuse: hits, single-flight, canaries and honest reporting."""

from __future__ import annotations

import json
import os
import random
import shutil
import tempfile
import time
import xml.etree.ElementTree as ET
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import testreuse_key as keys
import testreuse_store as storage

CANARY_RATE = 0.05
WAIT_S = 900.0
POLL_S = 0.5
TIERS = ("fast", "full", "slow", "all")
Launch = Callable[[list[str]], int]


class Events:
    """The notifications a run raises; the default does nothing."""

    def waiting(self, file: str, owner: dict) -> None: ...
    def wait_failed(self, file: str, owner: dict) -> None: ...
    def canary_mismatch(self, file: str, detail: str) -> None: ...


@dataclass
class Outcome:
    """What happened to one file: how it was satisfied and, when run, how it ended."""

    file: str
    how: str = "ran"
    detail: str = ""
    passed: bool = False
    stored: str = ""


@dataclass
class Report:
    """The outcomes of a run and the exit status they imply."""

    outcomes: list[Outcome] = field(default_factory=list)
    status: int = 0

    def lines(self) -> list[str]:
        """One line per file and a summary that counts both kinds."""
        rows = [f"  {o.how}  {o.file}" + (f"  {o.detail}" if o.detail else "") for o in self.outcomes]
        reused = sum(o.how.startswith("reused") for o in self.outcomes)
        return [*rows, f"test: {len(self.outcomes) - reused} file(s) ran, {reused} reused"]


def selectors(command: list[str], root: Path) -> list[str] | None:
    """The test files a command names, or None when it names a node, a directory or nothing."""
    files = []
    for word in command[command.index("pytest") + 1:] if "pytest" in command else []:
        if "::" in word and not word.startswith("-"):
            return None
        target = root / word
        if not word.startswith("-") and target.is_dir():
            return None
        if not word.startswith("-") and target.is_file() and word.endswith(".py"):
            files.append(word)
    return files or None


def results(junit: Path, root: Path) -> dict[str, dict[str, int]]:
    """Per test file: tests, failures, errors and skips counted from a junit file."""
    rows: dict[str, dict[str, int]] = {}
    for case in ET.parse(junit).getroot().iter("testcase"):  # noqa: S314 - the file pytest just wrote
        parts = case.get("classname", "").split(".")
        path = next(("/".join(parts[:i]) + ".py" for i in range(len(parts), 0, -1)
                     if (root / ("/".join(parts[:i]) + ".py")).is_file()), "")
        if not path:
            continue
        row = rows.setdefault(path, {"tests": 0, "failed": 0, "skipped": 0})
        row["tests"] += 1
        row["failed"] += any(child.tag in ("failure", "error") for child in case)
        row["skipped"] += any(child.tag == "skipped" for child in case)
    return rows


def recorded(folder: Path) -> dict[str, dict]:
    """The plugin's per-file records from every process, merged."""
    merged: dict[str, dict] = {}
    for path in sorted(folder.glob("*.json")):
        for rel, row in json.loads(path.read_text(encoding="utf-8")).items():
            into = merged.setdefault(rel, {"reads": set(), "dirs": set(), "marks": set(),
                                           "violations": set(), "skipped": False})
            for name in ("reads", "dirs", "marks", "violations"):
                into[name] |= set(row[name])
            into["skipped"] = into["skipped"] or row["skipped"]
    return merged


def age(seconds: float) -> str:
    """A short age."""
    return (f"{seconds:.0f}s" if seconds < 90 else f"{seconds / 60:.0f}m" if seconds < 5400
            else f"{seconds / 3600:.1f}h" if seconds < 172800 else f"{seconds / 86400:.1f}d")


class Session:
    """One run of explicit test files under reuse."""

    def __init__(self, root: Path, command: list[str], files: list[str], store: storage.Store,
                 agent: dict, events: Events | None = None) -> None:
        self.root, self.command, self.files, self.store = root, command, files, store
        self.agent, self.events = agent, events or Events()
        self.closures = keys.Closures(root)
        self.rate = float(os.environ.get("DEV_TEST_REUSE_CANARY", CANARY_RATE))
        self.draw: Callable[[], float] = random.random
        self.tree = ""
        self.report = Report()
        self.junit_sha = ""

    def run(self, launch: Launch, tree: str) -> Report:
        """Reuse what can be reused, execute the rest through ``launch``, record what passed."""
        self.tree = tree
        outcomes: dict[str, Outcome] = {f: Outcome(f) for f in self.files}
        lookups = {f: keys.lookup(self.root, f, self.command) for f in self.files}
        run, claimed, canary, deferred = [], [], {}, {}
        for file in self.files:
            look = lookups[file]
            if look.barred:
                outcomes[file].detail = f"(not reusable: {look.barred})"
                run.append(file)
                continue
            hit, why = self.store.hit(look.key, self.root, self.closures)
            if hit and self.draw() >= self.rate:
                outcomes[file].how = f"reused from {hit.id}"
                outcomes[file].passed = True
                outcomes[file].detail = (f"(tree {hit.entry['tree'][:10]}, "
                                         f"{age(time.time() - hit.entry['created'])}, by {hit.entry['agent'].get('id') or '?'})")
                continue
            if hit:
                canary[file] = hit
                outcomes[file].detail = f"(canary of {hit.id})"
            elif why not in ("no earlier run",):
                outcomes[file].detail = f"({why})"
            owner = self.store.claim(look.key, {"agent": self.agent, "files": [file]})
            if owner is not None and hit:
                outcomes[file].how = f"reused from {hit.id}"
                outcomes[file].passed = True
                outcomes[file].detail = f"(tree {hit.entry['tree'][:10]}, canary skipped: key in flight)"
                canary.pop(file)
            elif owner is None:
                claimed.append(file)
                run.append(file)
            else:
                deferred[file] = owner
                self.events.waiting(file, owner)
                outcomes[file].how = "waited"
        try:
            self.execute(run, launch, outcomes, lookups, canary, claimed)
        finally:
            for file in claimed:
                self.store.release(lookups[file].key)
        again = self.wait_for(deferred, outcomes, lookups)
        if again:
            self.execute(again, launch, outcomes, lookups, {}, again)
        self.report.outcomes = [outcomes[f] for f in self.files]
        failed = [o for o in self.report.outcomes if not o.passed]
        self.report.status = self.report.status or (1 if failed else 0)
        return self.report

    def execute(self, run: list[str], launch: Launch, outcomes: dict[str, Outcome], lookups: dict,
                canary: dict, claimed: list[str]) -> None:
        """Launch pytest for ``run`` and store the files that passed cleanly."""
        if not run:
            return
        scratch = Path(tempfile.mkdtemp(prefix="reuse-"))
        junit = scratch / "run.xml"
        os.environ["DEV_TEST_REUSE_RECORD"] = str(scratch / "record")
        os.environ["DEV_TEST_REUSE_ROOT"] = str(self.root)
        chosen = [w for w in self.command if w not in self.files]
        command = [*chosen, *run, f"--junitxml={junit}"]
        try:
            status = launch(command)
            self.report.status = max(self.report.status, status)
            counts = results(junit, self.root) if junit.is_file() else {}
            seen = recorded(scratch / "record") if (scratch / "record").is_dir() else {}
            self.junit_sha = keys.sha(junit.read_bytes()) if junit.is_file() else ""
            for file in run:
                self.settle(file, counts.get(file), seen.get(file), outcomes[file], lookups[file],
                            canary.get(file), file in claimed or file in canary, command)
        finally:
            os.environ.pop("DEV_TEST_REUSE_RECORD", None)
            os.environ.pop("DEV_TEST_REUSE_ROOT", None)
            shutil.rmtree(scratch, ignore_errors=True)

    def settle(self, file: str, count: dict | None, seen: dict | None, outcome: Outcome, look: keys.Lookup,
               hit: storage.Hit | None, store: bool, command: list[str]) -> None:
        """Record one executed file's outcome, store a clean pass, and compare it with a canary."""
        outcome.passed = bool(count) and not count["failed"]
        clean = outcome.passed and seen is not None and not seen["violations"] and not seen["skipped"] \
            and not (seen["marks"] & keys.NEVER_MARKS) and not look.barred and count["skipped"] == 0
        if hit is not None and not outcome.passed:
            detail = f"cached pass from {hit.id} but a fresh run failed"
            self.store.disable(look.key, hit.entry, detail)
            self.events.canary_mismatch(file, detail)
            outcome.how = "ran  CANARY MISMATCH"
            outcome.detail = f"({detail}; reuse disabled for this file)"
            self.report.status = self.report.status or 1
            return
        if hit is not None and outcome.passed:
            outcome.how = "ran  canary agrees"
        if not store or look.barred or not count:
            return
        if not outcome.passed:
            self.store_entry(file, look, count, seen, command, "fail")
        elif clean:
            outcome.stored = self.store_entry(file, look, count, seen, command, "pass")
        elif seen and (seen["violations"] or seen["skipped"]):
            outcome.detail = (outcome.detail + " (result not stored: " + (
                sorted(seen["violations"])[0] if seen["violations"] else "a test was skipped") + ")").strip()

    def store_entry(self, file: str, look: keys.Lookup, count: dict, seen: dict | None, command: list[str],
                    kind: str) -> str:
        """Write the entry for an executed file."""
        seen = seen or {"reads": set(), "dirs": set()}
        manifest = keys.build_manifest(self.root, file, set(seen["reads"]), set(seen["dirs"]), self.closures)
        return self.store.put({
            "lookup": look.key, "file": file, "outcome": kind, "manifest": manifest,
            "manifest_digest": keys.manifest_digest(manifest), "command": command,
            "junit_sha256": self.junit_sha, "counts": count, "tree": self.tree,
            "runner": {"pid": os.getpid(), "started": storage.process_start(os.getpid())},
            "agent": self.agent}, kind)

    def wait_for(self, deferred: dict[str, dict], outcomes: dict[str, Outcome], lookups: dict) -> list[str]:
        """Wait for runs of the same keys by other agents; the files whose run did not land must be run here."""
        again, until = [], time.monotonic() + WAIT_S
        for file, owner in deferred.items():
            key = lookups[file].key
            while self.store.claimed(key) is not None and time.monotonic() < until:
                time.sleep(POLL_S)
            hit, _ = self.store.hit(key, self.root, self.closures)
            if hit:
                outcomes[file].how = f"reused from {hit.id}"
                outcomes[file].passed = True
                outcomes[file].detail = (f"(tree {hit.entry['tree'][:10]}, run by "
                                         f"{hit.entry['agent'].get('id') or '?'} while this request waited)")
            else:
                self.events.wait_failed(file, owner)
                outcomes[file].how = "ran"
                outcomes[file].detail = "(the run it waited on did not pass)"
                again.append(file)
        return again
