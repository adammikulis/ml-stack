"""Who owns each worktree, and which ones were forgotten.

Every worktree of a repository has an owner and a state in one registry file kept in the shared
git directory (``ml-stack-trees.json``). A tree is *active* while its owner is alive, *landed*
when its owner is gone and every commit is in the development branch with nothing dirty, and an
*orphan* otherwise: its owner stopped and it still holds work nobody decided about. An orphan is
shown at once; only the landing gate waits out a grace period. ``close`` is the one removal path.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ml_stack import lock
from ml_stack.files import read_json, write_json
from ml_stack.workspace import integration_git as repo

UNKNOWN = "unknown"
DECIDED = ("bundled", "abandoned")
PROTECTED = ("main", "master")
DEFAULTS = {"grace_h": 2.0, "claim_h": 1.0, "ttl_h": 12.0, "max_ahead": 10, "max_behind": 20,
            "max_age_h": 4.0, "repeat_min": 30.0}
"""The policy: hours of grace for the gate, hours an unclaimed tree is assumed to have a live
creator, hours without a sign of life before an owner with no process is taken to have stopped,
the commits ahead, commits behind and hours with unlanded commits that raise a notice, and the
minutes between repeats of one notice."""


ENVIRONMENT = {"grace_h": "ML_STACK_TREES_GRACE_H", "claim_h": "ML_STACK_TREES_CLAIM_H",
               "ttl_h": "ML_STACK_TREES_TTL_H", "max_ahead": "ML_STACK_TREES_MAX_AHEAD",
               "max_behind": "ML_STACK_TREES_MAX_BEHIND", "max_age_h": "ML_STACK_TREES_MAX_AGE_H",
               "repeat_min": "ML_STACK_TREES_REPEAT_MIN"}
"""The environment variable that overrides each policy key; ML_STACK_TREES_LEAD names the coordinator."""


class Refused(Exception):
    """A removal or decision the registry will not make."""


def registry_path(root: Path | str) -> Path:
    """The registry file, shared by every worktree of the repository holding ``root``."""
    common = repo.git(Path(root), "rev-parse", "--path-format=absolute", "--git-common-dir")
    return Path(common) / "ml-stack-trees.json"


def listing(root: Path | str) -> list[tuple[Path, str, str]]:
    """Every registered worktree as (path, branch, head); the first is the primary checkout."""
    found, path, head, branch = [], "", "", ""
    for line in [*repo.git(Path(root), "worktree", "list", "--porcelain").splitlines(), ""]:
        if line.startswith("worktree "):
            path = line[9:]
        elif line.startswith("HEAD "):
            head = line[5:]
        elif line.startswith("branch refs/heads/"):
            branch = line[18:]
        elif not line and path:
            found.append((Path(path).resolve(), branch, head))
            path, head, branch = "", "", ""
    return found


def development(primary: Path) -> str:
    """The branch the primary checkout is on, the development branch."""
    return repo.git(primary, "branch", "--show-current") or "HEAD"


def update(root: Path | str, change: Callable[[dict], Any]) -> Any:
    """Apply ``change`` to the registry under its lock and save it; returns what ``change`` returns."""
    path = registry_path(root)
    with lock.rewriting(path):
        data = read_json(path, {})
        data.setdefault("trees", {})
        data.setdefault("history", [])
        before = json.dumps(data, sort_keys=True)
        result = change(data)
        if json.dumps(data, sort_keys=True) != before:
            write_json(path, data)
    return result


def policy(data: dict) -> dict:
    """The thresholds: defaults, then the registry's own, then ``ML_STACK_TREES_<NAME>`` in the environment."""
    merged = {**DEFAULTS, **{k: v for k, v in data.get("policy", {}).items() if k in DEFAULTS}}
    for key, name in ENVIRONMENT.items():
        text = os.environ.get(name, "")
        if text:
            try:
                merged[key] = float(text)
            except ValueError:
                continue
    return merged


def recorded(root: Path | str) -> dict:
    """The registry as saved, without scanning or writing anything."""
    return read_json(registry_path(root), {})


def current_policy(root: Path | str) -> dict:
    """The policy in force for the repository holding ``root``."""
    return policy(recorded(root))


def set_lead(root: Path | str, name: str) -> None:
    """Record the coordinator that escalations go to."""
    update(root, lambda data: data.__setitem__("lead", name))


def new_entry(owner: str, purpose: str, base: str, now: float, pid: int = 0) -> dict:
    """A freshly created tree owned by ``owner``."""
    return {"owner": owner, "purpose": purpose, "base": base, "created": now, "seen": now, "pid": pid,
            "state": "active", "stopped": 0.0, "notified": {}}


def owner_alive(entry: dict, now: float, pol: dict) -> bool:
    """Whether the tree's owner is still running: its process when one is recorded, else a recent sign of life."""
    if entry["state"] != "active":
        return False
    if entry.get("pid"):
        return lock.pid_alive(int(entry["pid"]))
    window = pol["claim_h"] if entry["owner"] == UNKNOWN else pol["ttl_h"]
    return now - entry["seen"] < window * 3600


def standing(tree: tuple[Path, str, str], entry: dict, dev: str, clock: tuple[float, dict]) -> dict:
    """One tree's standing: counts against the development branch, dirty files, owner and status."""
    (path, branch, head), (now, pol) = tree, clock
    unique = [x for x in repo.git(path, "cherry", dev, head).splitlines() if x.startswith("+")]
    stamps = repo.git(path, "log", "--format=%ct", f"{dev}..{head}").split()
    dirty = len([x for x in repo.git(path, "status", "--porcelain").splitlines() if x])
    alive = owner_alive(entry, now, pol)
    landed = not unique and not dirty
    status = "active" if alive else entry["state"] if entry["state"] in DECIDED else "landed" if landed else "orphan"
    since = entry["stopped"] or entry["created"] + (pol["claim_h"] * 3600 if entry["owner"] == UNKNOWN else 0)
    return {"path": str(path), "branch": branch or "(detached)", "head": head, "owner": entry["owner"],
            "purpose": entry["purpose"], "status": status, "alive": alive, "landed": landed,
            "ahead": int(repo.git(path, "rev-list", "--count", f"{dev}..{head}")),
            "behind": int(repo.git(path, "rev-list", "--count", f"{head}..{dev}")),
            "unlanded": len(unique), "dirty": dirty, "since": since,
            "unlanded_age_h": max(0.0, (now - min(int(s) for s in stamps)) / 3600) if unique and stamps else 0.0,
            "age_h": max(0.0, (now - entry["created"]) / 3600)}


def scan(root: Path | str, now: float, claimed: tuple[str, Path] | None = None) -> dict:
    """Claim every unregistered worktree as owned by nobody known, drop the ones that are gone,
    mark owners whose process ended as stopped, and give ``claimed`` (owner, tree) its tree."""
    owner, here = claimed or ("", None)
    trees = listing(root)
    live = {str(p): (b, h) for p, b, h in trees[1:] if p.exists()}
    mine = str(here.resolve()) if here else ""

    def change(data: dict) -> dict:
        pol = policy(data)
        for path, (_branch, head) in live.items():
            entry = data["trees"].get(path)
            if entry is None:
                base = repo.git(Path(path), "merge-base", head, development(trees[0][0]))
                entry = data["trees"][path] = new_entry(UNKNOWN, "", base, now)
            if owner and path == mine and entry["state"] == "active" and entry["owner"] in (UNKNOWN, owner):
                entry.update(owner=owner, seen=now)
            if entry["state"] == "active" and entry.get("pid") and not owner_alive(entry, now, pol):
                entry.update(state="finished", stopped=now)
        for path in [p for p in data["trees"] if p not in live]:
            del data["trees"][path]
        return data

    return update(root, change)


def claim(root: Path | str, path: Path | str, owner: str, now: float, details: dict | None = None) -> None:
    """Register the worktree at ``path`` as created by ``owner`` (harness, land, by hand).

    ``details`` may hold the ``purpose`` and the ``pid`` whose life stands for the owner's.
    """
    extra = details or {}
    target = str(Path(path).resolve())
    trees = listing(root)
    if not any(str(p) == target for p, _, _ in trees[1:]):
        return
    base = repo.git(Path(target), "merge-base", "HEAD", development(trees[0][0]))
    update(root, lambda data: data["trees"].__setitem__(target, new_entry(owner, extra.get("purpose", ""), base, now, int(extra.get("pid", 0)))))


def finish(root: Path | str, owner: str, now: float) -> list[dict]:
    """Mark every tree ``owner`` has as finished; returns the ones that still hold work (debt)."""
    def change(data: dict) -> None:
        for entry in data["trees"].values():
            if entry["owner"] == owner and entry["state"] == "active":
                entry.update(state="finished", stopped=now)

    scan(root, now)
    update(root, change)
    return [r for r in rows(root, now) if r["owner"] == owner and r["status"] == "orphan"]


def rows(root: Path | str, now: float, claimed: tuple[str, Path] | None = None, only: Path | None = None) -> list[dict]:
    """The standing of every tree other than the primary checkout, oldest first; of just ``only`` when given."""
    data = scan(root, now, claimed)
    pol = policy(data)
    trees = listing(root)
    dev = development(trees[0][0])
    found = [standing((p, b, h), data["trees"][str(p)], dev, (now, pol)) for p, b, h in trees[1:]
             if str(p) in data["trees"] and (only is None or p == only.resolve())]
    return sorted(found, key=lambda r: data["trees"][r["path"]]["created"])


def orphans(found: list[dict]) -> list[dict]:
    """The rows whose owner stopped with work nobody decided about."""
    return [r for r in found if r["status"] == "orphan"]


def past_grace(found: list[dict], pol: dict, now: float) -> list[dict]:
    """The orphans that have been orphans for longer than the gate's grace."""
    return [r for r in orphans(found) if now - r["since"] > pol["grace_h"] * 3600]


def line(row: dict, now: float) -> str:
    """One orphan as a line naming the tree, branch, work at stake, owner, age and the three actions."""
    hours = max(0.0, (now - row["since"]) / 3600)
    return (f"ORPHAN {row['branch']} {row['path']} +{row['ahead']} unlanded={row['unlanded']} dirty={row['dirty']} "
            f"owner={row['owner']} stopped {hours:.1f}h ago; scripts/worktrees close {row['path']} "
            "--landed | --bundle | --abandon 'reason'")


def lines(root: Path | str, now: float, limit: int = 8) -> list[str]:
    """Orphan lines for a status page or a hook, at most ``limit`` and a count of the rest."""
    found = orphans(rows(root, now))
    out = [line(r, now) for r in found[:limit]]
    return out + ([f"... and {len(found) - limit} more orphans"] if len(found) > limit else [])


def _bundle(primary: Path, row: dict, dev: str) -> Path:
    """Write the tree's unlanded commits to a bundle and verify that it holds the tip."""
    folder = Path(registry_path(primary)).parent / "ml-stack-bundles"
    folder.mkdir(exist_ok=True)
    name = re.sub(r"[^A-Za-z0-9._-]", "_", row["branch"]) + "-" + row["head"][:8]
    target, ref = folder / (name + ".bundle"), "refs/ml-stack/bundle/" + row["head"][:12]
    repo.git(primary, "update-ref", ref, row["head"])
    try:
        repo.git(primary, "bundle", "create", str(target), f"{dev}..{ref}")
        repo.git(primary, "bundle", "verify", str(target))
        if row["head"] not in repo.git(primary, "bundle", "list-heads", str(target)):
            raise Refused(f"the bundle {target} does not hold {row['head'][:8]}")
    finally:
        repo.git(primary, "update-ref", "-d", ref)
    return target


def _preserve_dirty(primary: Path, row: dict) -> str:
    """Save the tree's uncommitted diff beside the bundles before an abandon deletes it."""
    folder = registry_path(primary).parent / "ml-stack-bundles"
    folder.mkdir(exist_ok=True)
    patch = folder / (re.sub(r"[^A-Za-z0-9._-]", "_", row["branch"]) + "-" + row["head"][:8] + ".dirty.patch")
    patch.write_text(repo.git(Path(row["path"]), "diff", "--binary", "HEAD"))
    return str(patch)


def _check(row: dict, mode: str, reason: str, caller: str) -> None:
    """Raise Refused unless this decision may be made about this tree."""
    if row["alive"] and caller != row["owner"]:
        raise Refused(f"{row['path']} belongs to {row['owner']}, who is still active")
    if mode == "abandon" and not reason.strip():
        raise Refused("--abandon needs a reason")
    if mode == "landed" and not row["landed"]:
        raise Refused(f"{row['path']} is not landed: {row['unlanded']} unlanded commits, {row['dirty']} dirty files; "
                      "land it, or --bundle / --abandon 'reason'")
    if mode == "bundle" and row["dirty"]:
        raise Refused(f"{row['path']} has {row['dirty']} dirty files a bundle cannot hold; commit them or --abandon")
    if mode == "bundle" and not row["unlanded"]:
        raise Refused(f"{row['path']} has no unlanded commits to bundle; use --landed")


def close(root: Path | str, path: Path | str, decision: tuple[str, str], now: float, caller: str = "") -> dict:
    """Remove one tree on a recorded decision, (mode, reason): ``landed``, ``bundle`` or ``abandon``.

    Refuses a tree whose owner is active (unless the owner itself asks), a landed claim that is
    not true, a bundle that cannot be verified, an abandon with no reason, the primary checkout
    and a tree the current directory is inside.
    """
    mode, reason = decision
    target = Path(path).resolve()
    primary = listing(root)[0][0]
    here = Path.cwd().resolve()
    if target in (primary, here) or target in here.parents:
        raise Refused("close runs from outside the tree and never on the primary checkout")
    row = next((r for r in rows(root, now) if r["path"] == str(target)), None)
    if row is None:
        raise Refused(f"{target} is not a registered worktree")
    _check(row, mode, reason, caller)
    dev = development(primary)
    saved = _bundle(primary, row, dev) if mode == "bundle" else ""
    patch = _preserve_dirty(primary, row) if mode == "abandon" and row["dirty"] else ""
    repo.git(primary, "worktree", "remove", "--force", str(target))
    if row["branch"] != "(detached)" and row["branch"] not in (*PROTECTED, dev):
        repo.git(primary, "branch", "-D", row["branch"])
    repo.git(primary, "worktree", "prune")
    done = {"path": str(target), "branch": row["branch"], "head": row["head"], "owner": row["owner"],
            "mode": mode, "reason": reason, "at": now, "bundle": str(saved), "dirty_patch": patch}
    update(root, lambda data: _forget(data, str(target), done))
    return done


def _forget(data: dict, path: str, done: dict) -> None:
    """Drop a removed tree from the registry and keep the decision in the history."""
    data["trees"].pop(path, None)
    data["history"] = [*data["history"], done][-100:]


def sweep(root: Path | str, now: float) -> list[dict]:
    """Remove every tree that is landed and whose owner is not active; nothing else is touched."""
    return [close(root, r["path"], ("landed", ""), now, "sweep")
            for r in rows(root, now) if r["status"] == "landed" and not r["alive"]]
