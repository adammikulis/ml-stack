"""The builds `ml-stack-serve llama-cpp` keeps: which is active, which came before, which is pinned."""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from ml_stack import files
from ml_stack.serve import llamacpp_trust
from ml_stack.serve.build_paths import builds_dir, current_link, root
from ml_stack.serve.build_platform import server_name
from ml_stack.serve.build_verify import point_current

__all__ = ["KEEP", "STATE_VERSION", "Build", "activate", "active", "builds", "failed_dir",
           "load", "pin", "prune_candidates", "remove", "rollback", "unpin"]

STATE_VERSION = 1
KEEP = 3
"""How many good tracked builds are kept besides the active one before `prune` offers the rest."""


@dataclass(frozen=True, slots=True)
class Build:
    """One build directory under ``builds/``."""

    name: str
    path: Path
    info: dict

    @property
    def tracked(self) -> bool:
        return bool(self.info.get("track"))

    @property
    def good(self) -> bool:
        return self.tracked and bool((self.info.get("smoke") or {}).get("passed"))

    @property
    def binary(self) -> Path:
        return self.path / server_name()


def _state_file() -> Path:
    return root() / "track.json"


def failed_dir() -> Path:
    """Where a build that failed its smoke test is kept for inspection."""
    return root() / "failed"


def load() -> dict:
    """The tracking state: ``previous`` (oldest first) and ``pinned`` (a build name or empty)."""
    state = files.read_json(_state_file(), {})
    if not isinstance(state, dict) or files.version_of(state) != STATE_VERSION:
        state = {}
    return {"version": STATE_VERSION, "previous": list(state.get("previous") or []),
            "pinned": str(state.get("pinned") or "")}


def _save(state: dict) -> None:
    root().mkdir(parents=True, exist_ok=True)
    files.write_json(_state_file(), state)


def builds() -> list[Build]:
    """Every build under ``builds/``, oldest first."""
    found = []
    for manifest in sorted(builds_dir().glob("*/BUILD.json")):
        try:
            info = json.loads(manifest.read_text())
        except (OSError, ValueError):
            info = {}
        if isinstance(info, dict) and manifest.parent.joinpath(server_name()).exists():
            found.append(Build(manifest.parent.name, manifest.parent, info))
    return sorted(found, key=lambda b: (str(b.info.get("built_at", "")), b.name))


def active() -> Build | None:
    """The build ``current`` points at."""
    link = current_link()
    if not (link.is_symlink() or link.exists()):
        return None
    target = Path(os.path.realpath(link))
    return next((b for b in builds() if b.path == target), None)


def activate(build: Build) -> None:
    """Make ``build`` the active one, remembering the one it replaces."""
    state, before = load(), active()
    if before is not None and before.name != build.name:
        state["previous"] = [n for n in state["previous"] if n not in (before.name, build.name)]
        state["previous"].append(before.name)
    state["previous"] = [n for n in state["previous"] if n != build.name]
    point_current(build.path)
    _save(state)


def rollback() -> Build:
    """Switch to the most recent earlier build that is still installed and passes its pin.
    `LookupError` when there is none."""
    state, here = load(), active()
    names = {b.name: b for b in builds()}
    while state["previous"]:
        name = state["previous"].pop()
        found = names.get(name)
        if found is None or (here and found.name == here.name):
            continue
        if llamacpp_trust.problem(found.binary):
            continue
        point_current(found.path)
        state["pinned"] = ""
        _save(state)
        return found
    _save(state)
    raise LookupError("no earlier build to roll back to")


def pin(name: str) -> Build:
    """Stop tracking upstream: make build ``name`` active and remember it as pinned."""
    found = next((b for b in builds() if b.name == name), None)
    if found is None:
        raise LookupError(f"no build {name!r} under {builds_dir()}")
    if problem := llamacpp_trust.problem(found.binary):
        raise LookupError(problem)
    activate(found)
    state = load()
    state["pinned"] = found.name
    _save(state)
    return found


def unpin() -> None:
    state = load()
    state["pinned"] = ""
    _save(state)


def prune_candidates(keep: int = KEEP) -> list[Build]:
    """Good tracked builds beyond the newest ``keep``, never the active, pinned or previous one."""
    state, here = load(), active()
    protected = {state["pinned"], *state["previous"][-1:], here.name if here else ""}
    good = [b for b in builds() if b.good]
    return [b for b in good[:-keep] if b.name not in protected] if keep else []


def remove(build: Build) -> None:
    """Delete one build directory and its pin."""
    llamacpp_trust.unpin(build.binary)
    shutil.rmtree(build.path)
    state = load()
    state["previous"] = [n for n in state["previous"] if n != build.name]
    _save(state)

