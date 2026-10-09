"""``land run --entries``: branches named with the exact commit they were requested at."""

from __future__ import annotations

import json
from pathlib import Path

import land_git as lg


def check(root: Path, target: str, branch: str, tip: str) -> None:
    """Raise ValueError unless ``branch`` is at exactly ``tip`` and shares history with the target.

    A branch whose tip moved since the request is never merged: the requester asks again at the new
    commit, which needs a new review. A branch with no common history with the target is refused.
    """
    lg.refuse_protected(branch, target)
    if not lg.exists(root, branch):
        raise ValueError(f"{branch} does not exist")
    now = lg.out(root, "rev-parse", f"{branch}^{{commit}}")
    if now != tip:
        raise ValueError(f"{branch} is at {now[:12]}, not the requested {tip[:12]}; request it again")
    base = f"origin/{target}" if lg.exists(root, f"origin/{target}") else target
    if lg.git(root, "merge-base", base, branch, check=False).returncode:
        raise ValueError(f"{branch} shares no history with {base}")


def load(root: Path, target: str, path: Path) -> list[str]:
    """The branch names of an entries file, a JSON list of ``{"branch": name, "tip": full sha}``."""
    entries = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(entries, list) or not entries:
        raise ValueError("an entries file is a non-empty JSON list of {branch, tip}")
    names = []
    for entry in entries:
        branch, tip = str(entry.get("branch", "")), str(entry.get("tip", ""))
        check(root, target, branch, tip)
        names.append(branch)
    return names
