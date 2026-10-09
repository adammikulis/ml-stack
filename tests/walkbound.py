"""The walk behind the real-state snapshot, bounded so it cannot quietly grow expensive."""

from __future__ import annotations

import os
from collections import Counter
from collections.abc import Callable
from pathlib import Path

LIMIT = 20_000


class WalkTooLarge(RuntimeError):
    """The snapshot walked more files than ``LIMIT``."""


def mtimes(root: Path, skip: frozenset[str], ignored: Callable[[Path], bool],
           limit: int | None = None) -> dict[str, int]:
    """Every file under ``root`` by relative path with its mtime in ns, leaving out the
    top-level names in ``skip``, atomic-write temporaries and files ``ignored`` names; empty
    when ``root`` is absent. Raises ``WalkTooLarge`` with a count per top-level directory when
    more than ``limit`` files are walked."""
    limit = LIMIT if limit is None else limit
    out: dict[str, int] = {}
    seen: Counter[str] = Counter()
    for dirpath, dirnames, filenames in os.walk(root):
        rel = Path(dirpath).relative_to(root)
        dirnames[:] = [d for d in dirnames if (rel / d).as_posix() not in skip]
        if not rel.parts:
            filenames = [f for f in filenames if f not in skip]
        seen[rel.parts[0] if rel.parts else "."] += len(filenames)
        if sum(seen.values()) > limit:
            raise WalkTooLarge(f"the snapshot of {root} walked more than {limit} files: " + ", ".join(
                f"{name} {count}" for name, count in seen.most_common()) + "; add the large directories to LIVE_WRITERS")
        for name in filenames:
            if name.endswith(".tmp") or ignored(rel / name):
                continue
            try:
                out[(rel / name).as_posix()] = (Path(dirpath) / name).lstat().st_mtime_ns
            except OSError:
                continue
    return out
