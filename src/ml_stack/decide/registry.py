"""The trained deciders this machine knows about, by name."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

from ml_stack import home
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

NAME = re.compile(r"[a-z0-9][a-z0-9._-]{0,62}")
PREVIOUS = ".prev"
"""Appended to the name a replaced decider keeps."""


def check_name(name: str) -> str:
    """``name`` if it is a plain lower-case name (letters, digits, ``.``, ``_``, ``-``, no
    path part); else `DecideError`."""
    if not NAME.fullmatch(name) or ".." in name or name.endswith(PREVIOUS):
        raise DecideError(f"{name!r} is not a decider name: lower-case letters, digits, '.', "
                          f"'_' and '-', at most 63 characters, not ending in {PREVIOUS!r}")
    return name


def models_dir() -> Path:
    """Where trained deciders are written unless a directory is given."""
    return home.state("decide", "models")


def taken(name: str) -> bool:
    """Whether a decider is registered as ``name``."""
    return any(r["name"] == name for r in listing())


def register(directory: Path | str, *, replace: bool = False) -> dict[str, Any]:
    """Record a directory written by `ml_stack.train.decider` under the name in its config.

    A name already registered for another directory is refused unless ``replace``, which
    keeps the old entry (and its directory) as ``<name>.prev``.
    """
    root = expand(directory).resolve()
    cfg = json.loads((root / CONFIG).read_text())
    if cfg.get("format") != FORMAT:
        raise DecideError(f"{root / CONFIG} is not a {FORMAT} config")
    check_name(str(cfg.get("name", "")))
    manifest = read_json(root / "manifest.json", {})
    entry = {"name": cfg["name"], "path": str(root), "base": cfg["base"]["repo"],
             "data_hash": manifest.get("data_hash", ""), "metrics": manifest.get("metrics", {}),
             "registered": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    held = {e["name"]: e for e in listing()}
    old = held.get(entry["name"])
    if old is not None and Path(old["path"]).resolve() != root:
        if not replace:
            raise DecideError(f"a decider called {entry['name']!r} is already registered at "
                              f"{old['path']}; pass replace to register this one instead (the "
                              f"old one is kept as {entry['name'] + PREVIOUS!r})")
        held[entry["name"] + PREVIOUS] = {**old, "name": entry["name"] + PREVIOUS}
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
