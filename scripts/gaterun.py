"""Run the gate's steps: independent ones side by side, and a passed one is reused while nothing it reads moved.

Every gate step is a function of the tree it judges (source, tests, generated files, data beside the
checkers) and of the tools that read it, so a step that passed on a tree is not run again on the same
tree. The key is the git tree hash of everything not ignored, working tree included, plus the step's own
identity, the interpreter, the base branch tip and the version of each external tool. A change to any
tracked or untracked file changes the key; a failure is never stored, so a violation is always found.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from ml_stack.activity.gate import tree_hash

KEEP = 400
"""Stored passes kept; the oldest go first."""
TOOLS = ("ruff", "pyright")


@dataclass(frozen=True)
class Step:
    """One gate step: its name, what it runs, and the words that identify it for the key."""

    name: str
    run: Callable[[], int]
    identity: Sequence[str] = ()


@dataclass
class Outcome:
    """How a step ended: its status, how long it took, and whether a stored pass answered for it."""

    name: str
    status: int = 0
    seconds: float = 0.0
    reused: str = ""


@dataclass
class Gate:
    """The steps of a gate, the folder that remembers passes, and whether to consult it."""

    root: Path
    base: str
    folder: Path | None = None
    reuse: bool = True
    lock: threading.Lock = field(default_factory=threading.Lock)

    def key(self, step: Step) -> str:
        """The content key of ``step`` on the tree as it is now; empty when the tree cannot be hashed."""
        tree = tree_hash(self.root)
        if not tree:
            return ""
        tools = {t: _stamp(t) for t in TOOLS}
        parts = [step.name, list(step.identity), tree, sys.version, sys.executable, self.tip(), tools]
        return hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()

    def tip(self) -> str:
        """The commit the base branch is at, which the structure tests may compare the tree against."""
        done = subprocess.run(["git", "-C", str(self.root), "rev-parse", "--verify", "--quiet", self.base],
                              capture_output=True, text=True, check=False)
        return done.stdout.strip()

    def remembered(self, key: str) -> bool:
        """Whether a pass is stored under ``key``."""
        return bool(key) and self.folder is not None and (self.folder / key).is_file()

    def remember(self, key: str, name: str) -> None:
        """Store a pass under ``key``; a store that cannot be written only costs the next run its reuse."""
        if not key or self.folder is None:
            return
        try:
            self.folder.mkdir(parents=True, exist_ok=True)
            marker = self.folder / key
            marker.write_text(json.dumps({"step": name, "created": time.time()}), encoding="utf-8")
            old = sorted(self.folder.iterdir(), key=lambda p: p.stat().st_mtime)
            for stale in old[:-KEEP]:
                stale.unlink(missing_ok=True)
        except OSError:
            return

    def execute(self, step: Step) -> Outcome:
        """Run one step, or answer from a stored pass of the same key."""
        began = time.monotonic()
        key = self.key(step) if self.reuse else ""
        if self.remembered(key):
            self.say(f"gate: {step.name} reused: it passed on this tree ({key[:12]})")
            return Outcome(step.name, 0, time.monotonic() - began, key)
        status = step.run()
        if status == 0:
            self.remember(key, step.name)
        return Outcome(step.name, status, time.monotonic() - began)

    def say(self, text: str) -> None:
        """Print one block without another thread's lines cutting into it."""
        with self.lock:
            print(text, flush=True)

    def chain(self, steps: Sequence[Step]) -> list[Outcome]:
        """Run steps one after another, as one lane."""
        out = []
        for step in steps:
            self.say(f"gate: {step.name}")
            out.append(self.execute(step))
            self.say(f"gate: {step.name} {'ok' if not out[-1].status else 'FAILED'} in {out[-1].seconds:.0f} s")
        return out

    def run(self, lanes: Sequence[Sequence[Step]]) -> list[Outcome]:
        """Run each lane at the same time, the first lane on this thread, and return outcomes in lane order.

        A lane's steps stay in order. The first lane holds the pytest step, which installs signal handlers
        and so must be on the main thread; the others are the lighter scripts.
        """
        if len(lanes) < 2:
            return [o for lane in lanes for o in self.chain(lane)]
        with ThreadPoolExecutor(max_workers=len(lanes) - 1) as pool:
            side = [pool.submit(self.chain, lane) for lane in lanes[1:]]
            first = self.chain(lanes[0])
            return [*first, *(o for future in side for o in future.result())]


def _stamp(tool: str) -> str:
    """Where a tool is and when it was installed, empty when it is absent."""
    path = shutil.which(tool)
    return "" if path is None else f"{path}@{Path(path).stat().st_mtime_ns}"
