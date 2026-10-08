"""Run test files with per-file reuse: hits, single-flight, canaries and honest reporting."""

from __future__ import annotations

import json
import os
import random
import shutil
import subprocess
import tempfile
import time
import xml.etree.ElementTree as ET
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import testreuse_key as keys
import testreuse_store as storage

from ml_stack.log import warn

CANARY_RATE = 0.05
WAIT_S = 900.0
POLL_S = 0.5
TIERS = ("fast", "full", "slow", "all", "gate")
Launch = Callable[[list[str]], int]


class Events:
    """The notifications a run raises, and the agent that raises them; the default does nothing."""

    def __init__(self) -> None:
        self.agent: dict = {"id": "", "source": "none"}

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
class Plan:
    """What a run will do with its files: execute, claim, re-check as a canary or wait."""

    run: list[str] = field(default_factory=list)
    claimed: list[str] = field(default_factory=list)
    canary: dict[str, storage.Hit] = field(default_factory=dict)
    deferred: dict[str, dict] = field(default_factory=dict)


@dataclass
class Ran:
    """One executed file's results: junit counts, the plugin's record, the command, whether to store, a canary hit."""

    count: dict | None
    seen: dict | None
    command: list[str]
    store: bool
    hit: storage.Hit | None = None
    problem: str = ""
    status: int = 0


@dataclass
class Report:
    """The outcomes of a run and the exit status they imply."""

    outcomes: list[Outcome] = field(default_factory=list)
    status: int = 0
    canary: float = CANARY_RATE

    def counts(self) -> tuple[int, int]:
        """Files that ran and files that were reused."""
        reused = sum(o.how.startswith("reused") for o in self.outcomes)
        return len(self.outcomes) - reused, reused

    def lines(self) -> list[str]:
        """One line per file and a summary that counts both kinds."""
        rows = [f"  {o.how}  {o.file}  key {o.key[:12]}" + (f"  {o.detail}" if o.detail else "") for o in self.outcomes]
        ran, reused = self.counts()
        sampling = "canary off" if self.canary <= 0 else f"canary {self.canary:.0%}"
        return [*rows, f"test: {ran} file(s) ran, {reused} reused ({sampling})"]


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
    """Per test file: tests, failures and skips counted from a junit file; ``""`` holds cases that name no file."""
    rows: dict[str, dict[str, int]] = {}
    for case in ET.parse(junit).getroot().iter("testcase"):  # noqa: S314 - the file pytest just wrote
        path = file_of(case.get("classname", "") or case.get("name", ""), root)
        row = rows.setdefault(path, {"tests": 0, "failed": 0, "skipped": 0})
        row["tests"] += 1
        row["failed"] += any(child.tag in ("failure", "error") for child in case)
        row["skipped"] += any(child.tag == "skipped" for child in case)
    return rows


def recorded(folder: Path) -> tuple[dict[str, dict], str]:
    """The plugin's per-file records from every process, merged, and why they cannot be trusted, if so."""
    merged: dict[str, dict] = {}
    nodes: set[str] = set()
    workers: set[str] = set()
    died: list[str] = []
    for path in sorted(folder.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        meta = data["meta"]
        if meta.get("worker"):
            workers.add(meta["worker"])
        else:
            nodes |= set(meta.get("nodes", ()))
            died += meta.get("died", ())
        for rel, row in data["files"].items():
            into = merged.setdefault(rel, {"reads": set(), "dirs": set(), "stats": set(), "links": {}, "marks": set(),
                                           "violations": set(), "skipped": False, "collected": 0})
            for name in ("reads", "dirs", "stats", "marks", "violations"):
                into[name] |= set(row[name])
            into["links"].update(row["links"])
            into["skipped"] = into["skipped"] or row["skipped"]
            into["collected"] = max(into["collected"], row["collected"])
    shared = merged.pop("*", None)
    for row in merged.values():
        for name in ("reads", "dirs", "stats", "violations"):
            row[name] |= shared[name] if shared else set()
        row["links"].update(shared["links"] if shared else {})
    problem = f"worker {died[0]} died" if died else (
        f"worker {sorted(nodes - workers)[0]} did not report" if nodes - workers else "")
    return merged, problem


def canary_rate() -> float:
    """The sampling rate from ``DEV_TEST_REUSE_CANARY``, the default when it is absent or not a number in [0, 1]."""
    try:
        rate = float(os.environ.get("DEV_TEST_REUSE_CANARY", CANARY_RATE))
    except ValueError:
        return CANARY_RATE
    return rate if 0.0 <= rate <= 1.0 else CANARY_RATE


def is_clean(root: Path) -> bool:
    """Whether the checkout has no uncommitted or untracked changes; false when git cannot say."""
    done = subprocess.run(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all"],
                          capture_output=True, text=True, check=False)
    return done.returncode == 0 and not done.stdout.strip()


def head(root: Path) -> str:
    """The commit the checkout is on, or empty."""
    done = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    return done.stdout.strip() if done.returncode == 0 else ""


def age(seconds: float) -> str:
    """A short age."""
    return (f"{seconds:.0f}s" if seconds < 90 else f"{seconds / 60:.0f}m" if seconds < 5400
            else f"{seconds / 3600:.1f}h" if seconds < 172800 else f"{seconds / 86400:.1f}d")


class Session:
    """One run of test files under reuse, or a run whose passes refresh the store."""

    def __init__(self, root: Path, command: list[str], files: list[str] | None, store: storage.Store,
                 events: Events | None = None) -> None:
        self.root, self.command, self.files, self.store = root, command, files, store
        self.events = events or Events()
        self.closures = keys.Closures(root)
        self.rate = canary_rate()
        self.commit = ""
        self.tree_of: Callable[[], str] = lambda: ""
        self.draw: Callable[[], float] = random.random
        self.tree, self.refresh = "", False
        self.report = Report(canary=self.rate)
        self.junit_sha = ""
        self.outcomes: dict[str, Outcome] = {}
        self.looked: dict[str, keys.Lookup] = {}
        self.checked: dict[str, str] = {}
        self.claims: dict[str, str] = {}
        self.checked_tree = ""
        self.clean = False

    def lookup(self, file: str) -> keys.Lookup:
        """The lookup key of ``file`` under this run's command."""
        if file not in self.looked:
            self.looked[file] = keys.lookup(self.root, file, self.command)
        return self.looked[file]

    def outcome(self, file: str) -> Outcome:
        """The outcome record for ``file``."""
        return self.outcomes.setdefault(file, Outcome(file, key=self.lookup(file).key))

    def run(self, launch: Launch, tree: str | Callable[[], str], reuse: bool = True) -> Report:
        """Reuse what can be reused, execute the rest through ``launch``, record what passed.

        ``tree`` is the tree hash, or a function returning it; it is read before and after each launch.
        """
        self.tree_of = tree if callable(tree) else (lambda: tree)
        self.refresh = not reuse
        self.checked_tree = self.tree_of()
        plan = Plan(run=list(self.files or []))
        try:
            if reuse and self.files:
                plan.run = []
                for file in self.files:
                    self.consider(file, plan)
            self.execute(plan, launch)
        finally:
            for file in plan.claimed:
                self.store.release(self.claims.get(file) or self.lookup(file).key)
        again = self.wait_for(plan.deferred)
        if again:
            self.execute(Plan(run=again, claimed=again), launch)
        self.report.outcomes = [self.outcome(f) for f in (self.files or sorted(self.outcomes))]
        failed = any(not o.passed for o in self.report.outcomes)
        self.report.status = self.report.status or (1 if failed else 0)
        return self.report

    def consider(self, file: str, plan: Plan) -> None:
        """Decide whether ``file`` is reused, executed, executed as a canary or waited for."""
        look, outcome = self.lookup(file), self.outcome(file)
        self.checked[file] = look.key
        if look.barred:
            outcome.detail = f"(not reusable: {look.barred})"
            plan.run.append(file)
            return
        hit, why = self.store.hit(look.key, self.root, self.closures)
        if hit and self.draw() >= self.rate:
            self.reuse(outcome, hit, f"(tree {hit.entry['tree'][:10]}, {age(time.time() - hit.entry['created'])}, "
                                     f"by {hit.entry['agent'].get('id') or '?'})")
            return
        if hit:
            plan.canary[file] = hit
            outcome.detail = f"(canary of {hit.id})"
        elif why != "no earlier run":
            outcome.detail = f"({why})"
        owner = self.store.claim(look.key, {"agent": self.events.agent, "files": [file]})
        if owner is None:
            self.claims[file] = look.key
            seq = self.events.claimed(file, look.key)
            if seq:
                self.store.note_thread(look.key, seq)
            plan.claimed.append(file)
            plan.run.append(file)
        elif hit:
            plan.canary.pop(file)
            self.reuse(outcome, hit, f"(tree {hit.entry['tree'][:10]}, canary skipped: key in flight)")
        else:
            plan.deferred[file] = owner
            print(f"test: {file} is being run by {owner.get('agent', {}).get('id') or 'another run'}; waiting", flush=True)
            self.events.waiting(file, owner)
            outcome.how = "waited"

    @staticmethod
    def reuse(outcome: Outcome, hit: storage.Hit, detail: str) -> None:
        """Mark ``outcome`` satisfied by a stored pass."""
        outcome.how, outcome.passed, outcome.detail = f"reused from {hit.id}", True, detail

    def execute(self, plan: Plan, launch: Launch) -> None:
        """Launch pytest for the plan's files (the whole command when it names none) and store clean passes."""
        run = plan.run
        if self.files and not run:
            return
        scratch = Path(tempfile.mkdtemp(prefix="reuse-"))
        junit = scratch / "run.xml"
        os.environ["DEV_TEST_REUSE_RECORD"] = str(scratch / "record")
        os.environ["DEV_TEST_REUSE_ROOT"] = str(self.root)
        command = [*[w for w in self.command if w not in (self.files or [])], *run, f"--junitxml={junit}"]
        try:
            self.looked.clear()
            before = {f: self.lookup(f).key for f in run}
            self.tree, self.commit, self.clean = self.tree_of(), head(self.root), is_clean(self.root)
            stamps = keys.snapshot(self.root)
            status = launch(command)
            moved = (stamps is not None and (keys.snapshot(self.root) != stamps)) or (
                bool(self.tree) and self.tree_of() != self.tree)
            self.looked.clear()
            stale = {f for f in run if self.checked.get(f, before[f]) != before[f] or self.lookup(f).key != before[f]}
            stale |= set(run) if self.checked_tree and self.checked_tree != self.tree else set()
            try:
                counts = results(junit, self.root) if junit.is_file() else {}
            except (ET.ParseError, OSError):
                counts, unreadable = {}, True
            else:
                unreadable = False
            unattributed = counts.pop("", None)
            ended = 0 if status == 5 and counts and all(counts.get(f) for f in (run or counts)) else status
            self.report.status = max(self.report.status, 4 if moved else ended)
            seen, trouble = recorded(scratch / "record") if (scratch / "record").is_dir() else ({}, "")
            self.junit_sha = keys.sha(junit.read_bytes()) if junit.is_file() else ""
            problem = "the tree changed during the run" if moved else (
                "the junit file is unreadable" if unreadable else
                "the checkout is too large to watch" if stamps is None else
                f"pytest exited {status}" if status not in (0, 1, 5) else (
                    "an error belongs to no test file" if status == 1 and unattributed and unattributed["failed"]
                    else "the tree hash is unknown" if not self.tree else trouble))
            for file in run or sorted(counts):
                keep = self.refresh or not run or file in plan.claimed or file in plan.canary
                why = problem or ("its inputs changed between the hit check and the run" if file in stale else "")
                self.settle(file, Ran(counts.get(file), seen.get(file), command, keep, plan.canary.get(file),
                                      why, status))
        finally:
            os.environ.pop("DEV_TEST_REUSE_RECORD", None)
            os.environ.pop("DEV_TEST_REUSE_ROOT", None)
            shutil.rmtree(scratch, ignore_errors=True)

    def settle(self, file: str, ran: Ran) -> None:
        """Record one executed file's outcome, store a clean pass, and compare it with a canary."""
        outcome, look, count, hit = self.outcome(file), self.lookup(file), ran.count, ran.hit
        outcome.passed = bool(count) and not count["failed"]
        if count is None and not ran.problem:
            outcome.detail = f"{outcome.detail} (no tests ran)".strip()
            return
        if ran.problem:
            outcome.passed = outcome.passed and not ran.problem.startswith(("the tree", "pytest exited"))
            outcome.detail = f"{outcome.detail} (not stored: {ran.problem})".strip()
            return
        if hit is not None and not outcome.passed:
            detail = f"cached pass from {hit.id} but a fresh run failed"
            self.store.disable(look.key, hit.entry, detail)
            self.events.canary_mismatch(file, detail)
            outcome.how, outcome.detail = "ran  canary mismatch", f"({detail}; reuse disabled for this file)"
            self.report.status = self.report.status or 1
            return
        if hit is not None:
            outcome.how = "ran  canary agrees"
        if not ran.store or look.barred or not count:
            return
        if not outcome.passed:
            self.events.finished(file, look.key, "fail", self.store_entry(file, ran, "fail"))
            return
        manifest = keys.build_manifest(self.root, file, ran.seen or {}, self.closures, tuple(self.command))
        why = self.refusal(ran.seen, count) or (
            f"its closure uses {manifest['unobserved']}, which the runner cannot observe" if manifest["unobserved"]
            else "")
        if why:
            outcome.detail = f"{outcome.detail} (result not stored: {why})".strip()
            self.events.finished(file, look.key, "pass-unstored", "")
        else:
            outcome.stored = self.store_entry(file, ran, "pass", manifest)
            if not outcome.stored:
                outcome.detail = f"{outcome.detail} (not stored: the store is unavailable or an input changed)".strip()
            self.events.finished(file, look.key, "pass" if outcome.stored else "pass-unstored", outcome.stored)

    @staticmethod
    def refusal(seen: dict | None, count: dict) -> str:
        """Why a passing run may not be stored, or empty."""
        if seen is None:
            return "no record of what it read"
        if seen["violations"]:
            return sorted(seen["violations"])[0]
        if seen["skipped"] or count["skipped"]:
            return "a test was skipped"
        if seen["collected"] != count["tests"]:
            return f"{count['tests']} of {seen['collected']} collected tests ran"
        return f"marked {sorted(seen['marks'] & keys.NEVER_MARKS)[0]}" if seen["marks"] & keys.NEVER_MARKS else ""

    def store_entry(self, file: str, ran: Ran, kind: str, manifest: dict | None = None) -> str:
        """Write the entry for an executed file; empty when an input changed or the store could not be written."""
        manifest = manifest or keys.build_manifest(self.root, file, ran.seen or {}, self.closures, tuple(self.command))
        if kind == "pass" and keys.manifest_holds(self.root, manifest, self.closures):
            return ""
        try:
            return self.put_entry(file, ran, kind, manifest)
        except OSError as error:
            warn(f"test: {file}: result not stored ({error})")
            return ""

    def put_entry(self, file: str, ran: Ran, kind: str, manifest: dict) -> str:
        """Hand the entry's fields to the store."""
        count, command = ran.count, ran.command
        return self.store.put({
            "lookup": self.lookup(file).key, "file": file, "outcome": kind, "manifest": manifest,
            "manifest_digest": keys.manifest_digest(manifest), "command": command,
            "junit_sha256": self.junit_sha, "counts": count, "tree": self.tree, "commit": self.commit, "clean": self.clean,
            "runner": {"pid": os.getpid(), "started": storage.process_start(os.getpid())},
            "agent": self.events.agent}, kind)

    def wait_for(self, deferred: dict[str, dict]) -> list[str]:
        """Wait for runs of the same keys by other agents; the files whose run did not land run here."""
        again, until = [], time.monotonic() + WAIT_S
        for file, owner in deferred.items():
            key, outcome = self.lookup(file).key, self.outcome(file)
            while self.store.claimed(key) is not None and time.monotonic() < until:
                time.sleep(POLL_S)
            self.looked.pop(file, None)
            fresh = self.lookup(file).key
            hit, _ = self.store.hit(fresh, self.root, self.closures) if fresh == key else (None, "")
            if hit:
                self.reuse(outcome, hit, f"(tree {hit.entry['tree'][:10]}, run by "
                                         f"{hit.entry['agent'].get('id') or '?'} while this request waited)")
            else:
                self.events.wait_failed(file, owner)
                outcome.how, outcome.detail = "ran", "(the run it waited on did not pass)"
                self.checked[file], self.checked_tree = fresh, self.tree_of()
                again.append(file)
        return again
