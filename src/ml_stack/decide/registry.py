"""The trained deciders this machine knows about, by name."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from ml_stack.decide.sources import FORMAT
from ml_stack.decide.types import DecideError
from ml_stack.deciders import (  # noqa: F401  (the readers live below hub; re-exported here)
    CONFIG,
    listing,
    local_bases,
    path,
)
from ml_stack.files import read_json, write_json
from ml_stack.home import expand


def register(directory: Path | str) -> dict[str, Any]:
    """Record a directory written by `ml_stack.train.decider` under the name in its config."""
    root = expand(directory).resolve()
    cfg = json.loads((root / CONFIG).read_text())
    if cfg.get("format") != FORMAT:
        raise DecideError(f"{root / CONFIG} is not a {FORMAT} config")
    manifest = read_json(root / "manifest.json", {})
    entry = {"name": cfg["name"], "path": str(root), "base": cfg["base"]["repo"],
             "data_hash": manifest.get("data_hash", ""), "metrics": manifest.get("metrics", {}),
             "registered": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    held = {e["name"]: e for e in listing()}
    held[entry["name"]] = entry
    write_json(path(), sorted(held.values(), key=lambda e: e["name"]))
    return entry


def find(name: str) -> Path:
    """The directory registered as ``name``."""
    for row in listing():
        if row["name"] == name:
            return Path(row["path"])
    raise DecideError(f"no decider called {name!r}; registered: "
                      f"{[r['name'] for r in listing()]}")
