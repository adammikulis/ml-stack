"""Run test files with per-file reuse: hits, single-flight, canaries and honest reporting."""

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
TIERS = ("fast", "full", "slow", "all", "gate")
Launch = Callable[[list[str]], int]


class Events:
    """The notifications a run raises; the default does nothing."""

    def waiting(self, file: str, owner: dict) -> None:
        """A request found its key in flight under ``owner``."""

    def wait_failed(self, file: str, owner: dict) -> None:
        """The run a request waited on did not leave a passing result."""

    def canary_mismatch(self, file: str, detail: str) -> None:
        """A re-executed hit failed."""

    def claimed(self, file: str, key: str) -> int:
        """This run took the key for ``file``; returns the board thread it opened, or 0."""
        return 0

    def job_started(self, job: str, spec: dict) -> int:
        """A background job began; returns the board thread it opened, or 0."""
        return 0

    def job_done(self, job: str, spec: dict, status: dict, thread: int = 0) -> None:
        """A background job ended."""

    def finished(self, file: str, key: str, outcome: str, entry: str) -> None:
        """This run finished the key for ``file``."""


@dataclass
class Outcome:
    """What happened to one file: how it was satisfied and, when run, how it ended."""

    file: str
    how: str = "ran"
    detail: str = ""
    passed: bool = False
    stored: str = ""
    key: str = ""


@dataclass
class Report:
    """The outcomes of a run and the exit status they imply."""

    outcomes: list[Outcome] = field(default_factory=list)
    status: int = 0

    def counts(self) -> tuple[int, int]:
        """Files that ran and files that were reused."""
        reused = sum(o.how.startswith("reused") for o in self.outcomes)
        return len(self.outcomes) - reused, reused

    def lines(self) -> list[str]:
        """One line per file and a summary that counts both kinds."""
        rows = [f"  {o.how}  {o.file}  key {o.key[:12]}" + (f"  {o.detail}" if o.detail else "") for o in self.outcomes]
        ran, reused = self.counts()
        return [*rows, f"test: {ran} file(s) ran, {reused} reused"]


def selectors(command: list[str], root: Path) -> list[str] | None:
    """The test files a command names, or None when it names a node, a directory or nothing."""
    files = []
    for word in command[command.index("pytest") + 1:] if "pytest" in command else []:
        if word.startswith("-"):
            continue
        if "::" in word or (root / word).is_dir():
            return None
        if (root / word).is_file() and word.endswith(".py"):
            files.append(word)
    return files or None


def file_of(classname: str, root: Path) -> str:
    """The test file a junit ``classname`` belongs to, or empty."""
    parts = classname.split(".")
    return next(("/".join(parts[:i]) + ".py" for i in range(len(parts), 0, -1)
                 if (root / ("/".join(parts[:i]) + ".py")).is_file()), "")


def results(junit: Path, root: Path) -> dict[str, dict[str, int]]:
    """Per test file: tests, failures and skips counted from a junit file."""
    rows: dict[str, dict[str, int]] = {}
    for case in ET.parse(junit).getroot().iter("testcase"):  # noqa: S314 - the file pytest just wrote
        path = file_of(case.get("classname", ""), root)
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
    """One run of test files under reuse, or a run whose passes refresh the store."""

    def __init__(self, root: Path, command: list[str], files: list[str] | None, store: storage.Store,
                 agent: dict, events: Events | None = None) -> None:
        self.root, self.command, self.files, self.store = root, command, files, store
        self.agent, self.events = agent, events or Events()
        self.closures = keys.Closures(root)
        self.rate = float(os.environ.get("DEV_TEST_REUSE_CANARY", CANARY_RATE))
        self.draw: Callable[[], float] = random.random
        self.tree, self.refresh = "", False
        self.report = Report()
        self.junit_sha = ""
        self.outcomes: dict[str, Outcome] = {}
        self.looked: dict[str, keys.Lookup] = {}

    def lookup(self, file: str) -> keys.Lookup:
        """The lookup key of ``file`` under this run's command."""
        if file not in self.looked:
            self.looked[file] = keys.lookup(self.root, file, self.command)
        return self.looked[file]

    def outcome(self, file: str) -> Outcome:
        """The outcome record for ``file``."""
        return self.outcomes.setdefault(file, Outcome(file, key=self.lookup(file).key))

    def run(self, launch: Launch, tree: str, reuse: bool = True) -> Report:
        """Reuse what can be reused, execute the rest through ``launch``, record what passed."""
        self.tree, self.refresh = tree, not reuse
        run, claimed, canary, deferred = list(self.files or []), [], {}, {}
        if reuse and self.files:
            run = []
            for file in self.files:
                self.consider(file, run, claimed, canary, deferred)
        try:
            self.execute(run, launch, canary, claimed, not self.files or bool(run))
        finally:
            for file in claimed:
                self.store.release(self.lookup(file).key)
        again = self.wait_for(deferred)
        if again:
            self.execute(again, launch, {}, again, True)
        self.report.outcomes = [self.outcome(f) for f in (self.files or sorted(self.outcomes))]
        failed = any(not o.passed for o in self.report.outcomes)
        self.report.status = self.report.status or (1 if failed else 0)
        return self.report

    def consider(self, file: str, run: list[str], claimed: list[str], canary: dict, deferred: dict) -> None:
        """Decide whether ``file`` is reused, executed, executed as a canary or waited for."""
        look, outcome = self.lookup(file), self.outcome(file)
        if look.barred:
            outcome.detail = f"(not reusable: {look.barred})"
            run.append(file)
            return
        hit, why = self.store.hit(look.key, self.root, self.closures)
        if hit and self.draw() >= self.rate:
            self.reuse(outcome, hit, f"(tree {hit.entry['tree'][:10]}, {age(time.time() - hit.entry['created'])}, "
                                     f"by {hit.entry['agent'].get('id') or '?'})")
            return
        if hit:
            canary[file] = hit
            outcome.detail = f"(canary of {hit.id})"
        elif why != "no earlier run":
            outcome.detail = f"({why})"
        owner = self.store.claim(look.key, {"agent": self.agent, "files": [file]})
        if owner is None:
            seq = self.events.claimed(file, look.key)
            if seq:
                self.store.note_thread(look.key, seq)
            claimed.append(file)
            run.append(file)
        elif hit:
            canary.pop(file)
            self.reuse(outcome, hit, f"(tree {hit.entry['tree'][:10]}, canary skipped: key in flight)")
        else:
            deferred[file] = owner
            print(f"test: {file} is being run by {owner.get('agent', {}).get('id') or 'another run'}; waiting", flush=True)
            self.events.waiting(file, owner)
            outcome.how = "waited"

    @staticmethod
    def reuse(outcome: Outcome, hit: storage.Hit, detail: str) -> None:
        """Mark ``outcome`` satisfied by a stored pass."""
        outcome.how, outcome.passed, outcome.detail = f"reused from {hit.id}", True, detail

    def execute(self, run: list[str], launch: Launch, canary: dict, claimed: list[str], go: bool) -> None:
        """Launch pytest for ``run`` (the whole command when ``run`` is empty) and store clean passes."""
        if not go:
            return
        scratch = Path(tempfile.mkdtemp(prefix="reuse-"))
        junit = scratch / "run.xml"
        os.environ["DEV_TEST_REUSE_RECORD"] = str(scratch / "record")
        os.environ["DEV_TEST_REUSE_ROOT"] = str(self.root)
        command = [*[w for w in self.command if w not in (self.files or [])], *run, f"--junitxml={junit}"]
        try:
            self.report.status = max(self.report.status, launch(command))
            counts = results(junit, self.root) if junit.is_file() else {}
            seen = recorded(scratch / "record") if (scratch / "record").is_dir() else {}
            self.junit_sha = keys.sha(junit.read_bytes()) if junit.is_file() else ""
            for file in run or sorted(counts):
                self.settle(file, counts.get(file), seen.get(file), canary.get(file),
                            self.refresh or not run or file in claimed or file in canary, command)
        finally:
            os.environ.pop("DEV_TEST_REUSE_RECORD", None)
            os.environ.pop("DEV_TEST_REUSE_ROOT", None)
            shutil.rmtree(scratch, ignore_errors=True)

    def settle(self, file: str, count: dict | None, seen: dict | None, hit: storage.Hit | None,
               store: bool, command: list[str]) -> None:
        """Record one executed file's outcome, store a clean pass, and compare it with a canary."""
        outcome, look = self.outcome(file), self.lookup(file)
        outcome.passed = bool(count) and not count["failed"]
        if hit is not None and not outcome.passed:
            detail = f"cached pass from {hit.id} but a fresh run failed"
            self.store.disable(look.key, hit.entry, detail)
            self.events.canary_mismatch(file, detail)
            outcome.how, outcome.detail = "ran  CANARY MISMATCH", f"({detail}; reuse disabled for this file)"
            self.report.status = self.report.status or 1
            return
        if hit is not None:
            outcome.how = "ran  canary agrees"
        if not store or look.barred or not count:
            return
        if not outcome.passed:
            self.events.finished(file, look.key, "fail", self.store_entry(file, count, seen, command, "fail"))
            return
        why = self.refusal(seen, count)
        if why:
            outcome.detail = f"{outcome.detail} (result not stored: {why})".strip()
            self.events.finished(file, look.key, "pass-unstored", "")
        else:
            outcome.stored = self.store_entry(file, count, seen, command, "pass")
            self.events.finished(file, look.key, "pass", outcome.stored)

    @staticmethod
    def refusal(seen: dict | None, count: dict) -> str:
        """Why a passing run may not be stored, or empty."""
        if seen is None:
            return "no record of what it read"
        if seen["violations"]:
            return sorted(seen["violations"])[0]
        if seen["skipped"] or count["skipped"]:
            return "a test was skipped"
        return f"marked {sorted(seen['marks'] & keys.NEVER_MARKS)[0]}" if seen["marks"] & keys.NEVER_MARKS else ""

    def store_entry(self, file: str, count: dict, seen: dict | None, command: list[str], kind: str) -> str:
        """Write the entry for an executed file."""
        seen = seen or {"reads": set(), "dirs": set()}
        manifest = keys.build_manifest(self.root, file, set(seen["reads"]), set(seen["dirs"]), self.closures)
        return self.store.put({
            "lookup": self.lookup(file).key, "file": file, "outcome": kind, "manifest": manifest,
            "manifest_digest": keys.manifest_digest(manifest), "command": command,
            "junit_sha256": self.junit_sha, "counts": count, "tree": self.tree,
            "runner": {"pid": os.getpid(), "started": storage.process_start(os.getpid())},
            "agent": self.agent}, kind)

    def wait_for(self, deferred: dict[str, dict]) -> list[str]:
        """Wait for runs of the same keys by other agents; the files whose run did not land run here."""
        again, until = [], time.monotonic() + WAIT_S
        for file, owner in deferred.items():
            key, outcome = self.lookup(file).key, self.outcome(file)
            while self.store.claimed(key) is not None and time.monotonic() < until:
                time.sleep(POLL_S)
            hit, _ = self.store.hit(key, self.root, self.closures)
            if hit:
                self.reuse(outcome, hit, f"(tree {hit.entry['tree'][:10]}, run by "
                                         f"{hit.entry['agent'].get('id') or '?'} while this request waited)")
            else:
                self.events.wait_failed(file, owner)
                outcome.how, outcome.detail = "ran", "(the run it waited on did not pass)"
                again.append(file)
        return again
