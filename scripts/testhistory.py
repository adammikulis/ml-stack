"""Measured per-test durations shared by every worktree of a repository, and the run-length estimate built on them."""
from __future__ import annotations

import contextlib
import json
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from ml_stack.lock import release, take

ALPHA = 0.3
OUTLIER_FACTOR = 3.0
WALL_SHARE = 0.25
DEFAULT_FILE_S = 30.0
DEFAULT_NODE_S = 5.0
THRESHOLD_S = 180.0
MIN_RECORDED = 50


def history_path(root: Path) -> Path:
    """<git-common-dir>/test-history/durations.json for the checkout at ``root``."""
    out = subprocess.run(["git", "rev-parse", "--git-common-dir"], cwd=root, capture_output=True, text=True, check=True)
    common = Path(out.stdout.strip())
    return (common if common.is_absolute() else root / common).resolve() / "test-history" / "durations.json"


def threshold() -> float:
    """Estimated seconds at which a run counts as background (DEV_TEST_BACKGROUND_S)."""
    raw = os.environ.get("DEV_TEST_BACKGROUND_S", "")
    return float(raw) if raw.replace(".", "", 1).isdigit() else THRESHOLD_S


def load(path: Path) -> dict[str, dict]:
    """The recorded tests; empty when the store is missing or unreadable."""
    try:
        data = json.loads(path.read_text())
        tests = data["tests"]
        return tests if isinstance(tests, dict) else {}
    except (OSError, ValueError, KeyError, TypeError):
        return {}


def seconds(entry: dict) -> float:
    """A test's estimated seconds: its CPU seconds, but never under a quarter of its wall seconds."""
    return max(float(entry.get("cpu", 0.0)), WALL_SHARE * float(entry.get("wall", 0.0)))


def _blend(old: float | None, sample: float, n: int) -> float:
    if old is None:
        return sample
    if n >= 3:
        sample = min(sample, OUTLIER_FACTOR * max(old, 0.01))
    return (1 - ALPHA) * old + ALPHA * sample


def update(entry: dict | None, cpu: float, wall: float) -> dict:
    """The entry after one more sample: an exponentially weighted value that one loaded outlier cannot move far."""
    n = int(entry["n"]) if entry else 0
    return {"cpu": _blend(entry["cpu"] if entry else None, cpu, n),
            "wall": _blend(entry["wall"] if entry else None, wall, n), "n": n + 1}


@contextlib.contextmanager
def _locked(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with (path.parent / "durations.lock").open("a+") as handle:
        while not take(handle):
            time.sleep(0.01)
        try:
            yield
        finally:
            release(handle)


def merge(path: Path, root: Path, samples: dict[str, tuple[float, float]],
          collected: dict[str, set[str]] | None = None) -> None:
    """Merge (cpu, wall) samples into the store atomically; drop tests whose file is gone or no longer defines them."""
    with _locked(path):
        tests = load(path)
        for node, (cpu, wall) in samples.items():
            tests[node] = update(tests.get(node), cpu, wall)
        gone = {node for node in tests if not (root / node.split("::", 1)[0]).exists()}
        for file, ids in (collected or {}).items():
            gone |= {node for node in tests if node.split("::", 1)[0] == file and node not in ids}
        for node in gone:
            del tests[node]
        descriptor, name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        with os.fdopen(descriptor, "w") as stream:
            json.dump({"version": 1, "updated": time.time(), "tests": tests}, stream)
        Path(name).replace(path)


@dataclass
class Estimate:
    seconds: float
    recorded: int
    unknown: int

    def wall(self, workers: int) -> float:
        """Estimated wall seconds on ``workers`` parallel workers."""
        return self.seconds / max(1, workers)


def _files(root: Path, selectors: list[str]) -> list[str]:
    tests = sorted(p.relative_to(root).as_posix() for p in (root / "tests").rglob("test_*.py"))
    chosen = [s.split("::", 1)[0].rstrip("/") for s in selectors if "::" not in s]
    if not selectors:
        return tests
    return [t for t in tests if any(t == c or t.startswith(c + "/") for c in chosen)]


def _matches(node: str, selector: str) -> bool:
    return node == selector or (node.startswith(selector) and node[len(selector)] in "[:")


def estimate(tests: dict[str, dict], root: Path, selectors: list[str]) -> Estimate:
    """Seconds the selected tests are expected to take, summed per file from the history without collecting."""
    by_file: dict[str, list[tuple[str, float]]] = {}
    for node, entry in tests.items():
        by_file.setdefault(node.split("::", 1)[0], []).append((node, seconds(entry)))
    total, recorded, unknown = 0.0, 0, 0
    for file in _files(root, selectors):
        rows = by_file.get(file, [])
        total += sum(s for _, s in rows) if rows else DEFAULT_FILE_S
        recorded += len(rows)
        unknown += not rows
    for selector in (s for s in selectors if "::" in s):
        rows = [s for node, s in by_file.get(selector.split("::", 1)[0], []) if _matches(node, selector)]
        total += sum(rows) if rows else DEFAULT_NODE_S
        recorded += len(rows)
        unknown += not rows
    return Estimate(total, recorded, unknown)
