"""The commands a peer may be asked to run, and the form the daemon runs them in."""

from __future__ import annotations

import shutil
import sys
from collections.abc import Sequence
from pathlib import Path

__all__ = ["CONSOLE", "INTERPRETERS", "MODULES", "allowed"]

CONSOLE = frozenset({"poolhouse-bench"})
"""Console scripts, named without a path."""
MODULES = frozenset({"poolhouse.fleet.calibration"})
"""Modules run as ``python -m MODULE``."""
INTERPRETERS = frozenset({"python", "python3"})


def allowed(argv: Sequence[str]) -> list[str]:
    """``argv`` with its program resolved to this machine's install; ValueError names what is refused."""
    if not argv:
        raise ValueError("empty argv")
    program = str(argv[0])
    if program in CONSOLE:
        found = shutil.which(program) or str(Path(sys.executable).parent / program)
        return [found, *map(str, argv[1:])]
    if program in INTERPRETERS and len(argv) >= 3 and argv[1] == "-m" and argv[2] in MODULES:
        return [sys.executable, *map(str, argv[1:])]
    raise ValueError(f"{program!r} is not a command a peer may be asked to run "
                     f"(allowed: {', '.join(sorted(CONSOLE))}, python -m "
                     f"{', '.join(sorted(MODULES))})")
