"""Dividing test files between devices: by recorded duration, weighted by free workers; Mac-only files stay."""

from __future__ import annotations

import json
import re
import statistics
from dataclasses import dataclass
from pathlib import Path

DEFAULT_S = 5.0
LOCAL_ONLY = re.compile(r"""["']darwin["']|sandbox-exec|seatbelt|launchctl|osascript""", re.IGNORECASE)


@dataclass(frozen=True)
class Target:
    """One device a run may use: its name, the workers it can spare and the fixed cost of using it."""

    name: str
    workers: int
    setup_s: float = 0.0


def local_only(root: Path, name: str) -> bool:
    """Whether the test file mentions something only a Mac has (read from the file itself)."""
    try:
        return bool(LOCAL_ONLY.search((root / name).read_text(errors="replace")))
    except OSError:
        return True


def load(path: Path) -> dict[str, float]:
    """The recorded seconds per test file, or nothing when there is no history yet."""
    try:
        raw = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    return {k: float(v) for k, v in raw.items() if isinstance(k, str) and isinstance(v, (int, float))}


def record(path: Path, result: dict) -> None:
    """Fold a result's per-file wall seconds into the history at ``path`` (an average of old and new)."""
    seen = load(path)
    for name, counts in result.get("files", {}).items():
        seconds = float(counts.get("wall_s", 0.0))
        if seconds > 0:
            seen[name] = round((seen[name] + seconds) / 2 if name in seen else seconds, 3)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(seen, sort_keys=True))


def estimate(history: dict[str, float], name: str) -> float:
    """Seconds a file is expected to take: its history, else the median of the history, else a default."""
    if name in history:
        return history[name]
    return statistics.median(history.values()) if history else DEFAULT_S


def split(files: list[str], targets: list[Target], history: dict[str, float],
          stay: set[str] = frozenset()) -> dict[str, list[str]]:
    """Give each file to the target that would finish it soonest, longest files first.

    ``targets[0]`` is this machine; files in ``stay`` go there and still count toward its load.
    """
    plan: dict[str, list[str]] = {t.name: [] for t in targets}
    load_s = {t.name: t.setup_s for t in targets}
    here = targets[0]
    for name in files:
        if name in stay:
            plan[here.name].append(name)
            load_s[here.name] += estimate(history, name) / max(1, here.workers)
    for name in sorted((f for f in files if f not in stay), key=lambda f: -estimate(history, f)):
        best = min(targets, key=lambda t: load_s[t.name] + estimate(history, name) / max(1, t.workers))
        plan[best.name].append(name)
        load_s[best.name] += estimate(history, name) / max(1, best.workers)
    return {name: sorted(got) for name, got in plan.items() if got}
