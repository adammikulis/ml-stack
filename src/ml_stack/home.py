"""Where this machine keeps ml-stack's state and its cache.

`state()` names anything under the state root; `cache()` names anything under the cache
root. Both read the environment when they are called, so a caller that moves a root sees
the move without reloading a module.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = ["CACHE_ENV", "OVERRIDES", "ROOT_ENV", "cache", "expand", "home", "state",
           "user_home"]

ROOT_ENV = "ML_STACK_HOME"
"""Moves the state root."""

CACHE_ENV = "ML_STACK_CACHE"
"""Moves the cache root."""

OVERRIDES = {
    "bench": "MLSTACK_BENCH_HOME",
    "ingest": "MLSTACK_INGEST_HOME",
    "jobs": "MLSTACK_JOBS_HOME",
    "train": "MLSTACK_TRAIN_HOME",
    "web": "MLSTACK_WEB_PROFILE",
    "fit.json": "MLSTACK_FIT_FILE",
    "profiles.json": "MLSTACK_PROFILES_FILE",
    "limits.json": "MLSTACK_LIMITS_FILE",
    "rates.json": "ML_STACK_RATES",
}
"""The variable that moves one name out of the state root, by that name."""


def user_home() -> Path:
    """The account's home directory."""
    return Path.home()


def expand(path: str | Path) -> Path:
    """A path somebody named, with a leading ``~`` resolved."""
    return Path(path).expanduser()


def home() -> Path:
    """The directory this machine keeps ml-stack's state in."""
    named = os.environ.get(ROOT_ENV)
    return expand(named) if named else user_home() / ".ml-stack"


def state(*parts: str) -> Path:
    """A path under the state root, honouring the variable that moves its first part."""
    named = OVERRIDES.get(parts[0]) if parts else None
    moved = os.environ.get(named) if named else None
    if moved:
        return expand(moved).joinpath(*parts[1:])
    return home().joinpath(*parts)


def cache(*parts: str) -> Path:
    """A path under this machine's ml-stack cache directory."""
    named = os.environ.get(CACHE_ENV)
    root = expand(named) if named else user_home() / ".cache" / "ml_stack"
    return root.joinpath(*parts)
