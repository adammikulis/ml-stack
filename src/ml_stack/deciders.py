"""Which trained deciders this machine has registered, and where their files are.

Below both ``hub`` (which labels a model ``decision`` when it sits in one of these directories)
and ``decide`` (which trains, registers and runs them), so neither has to import the other.
Reading the registry grants nothing: it is a list of directories.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ml_stack import home
from ml_stack.files import read_json
from ml_stack.home import expand

CONFIG = "decider.json"
"""The file that marks a directory as a decider."""


def path() -> Path:
    """The registry file."""
    return home.state("decide", "registry.json")


def listing() -> list[dict[str, Any]]:
    """Every registered decider whose directory still exists."""
    rows = read_json(path(), [])
    return [r for r in rows if isinstance(r, dict) and (Path(r.get("path", "")) / CONFIG).is_file()]


def local_bases(root: Path) -> list[Path]:
    """The local base-model directories a registered decider's config names (none when it
    is pinned to a Hub revision)."""
    try:
        base = json.loads((root / CONFIG).read_text()).get("base", {})
        return [expand(base["path"]).resolve()] if "path" in base else []
    except (OSError, ValueError, TypeError, AttributeError):
        return []


def roots() -> list[Path]:
    """The directory of every registered decider, and the local base models they sit on."""
    out: list[Path] = []
    for row in listing():
        root = Path(row["path"])
        out.append(root.resolve())
        out.extend(local_bases(root))
    return out
