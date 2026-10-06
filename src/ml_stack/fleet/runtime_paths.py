"""State and cache roots for a Fleet daemon process."""

from __future__ import annotations

import os
from pathlib import Path

from ml_stack import home


def configure(root: Path) -> None:
    """Route custom daemon installations to their own process state and cache."""
    if home.ROOT_ENV in os.environ or root.resolve() == home.state("traind").resolve():
        return
    state = root.resolve() / "state"
    os.environ[home.ROOT_ENV] = str(state)
    os.environ.setdefault(home.CACHE_ENV, str(state / "cache"))
