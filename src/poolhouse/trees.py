"""Who owns each worktree, and which ones were forgotten.

Every worktree of a repository has an owner and a state in one registry file kept in the shared
git directory (``poolhouse-trees.json``). A tree is *active* while its owner is alive and *landed*
when its owner is gone and every commit is in the development branch with nothing dirty. A tree
whose owner stopped and which still holds work is *waiting* (finished, waiting to land) for the
grace, then an *orphan*. Both are shown at once; only the notices' escalation and the landing gate
use the grace. ``close`` is the one removal path.

Cost: one ``git worktree list`` and, per tree whose head or the development tip moved, three git
calls cached in the registry; ``git status`` runs only for trees whose owner is not active. A
registry that is locked, missing or garbled never stops anything: callers fail open.
"""

from __future__ import annotations

import contextlib
import copy
import json
import os
import re
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

from poolhouse import lock
from poolhouse.files import read_json, write_json

UNKNOWN = "unknown"
DECIDED = ("bundled", "abandoned")
PROTECTED = ("main", "master")
LOCK_TIMEOUT = 2.0
DEFAULTS = {"grace_h": 2.0, "claim_h": 1.0, "ttl_h": 12.0, "max_ahead": 10, "max_behind": 20,
            "max_age_h": 4.0, "repeat_min": 30.0, "cap_per_hour": 6}
ENVIRONMENT = {"grace_h": "POOLHOUSE_TREES_GRACE_H", "claim_h": "POOLHOUSE_TREES_CLAIM_H",
               "ttl_h": "POOLHOUSE_TREES_TTL_H", "max_ahead": "POOLHOUSE_TREES_MAX_AHEAD",
               "max_behind": "POOLHOUSE_TREES_MAX_BEHIND", "max_age_h": "POOLHOUSE_TREES_MAX_AGE_H",
               "repeat_min": "POOLHOUSE_TREES_REPEAT_MIN", "cap_per_hour": "POOLHOUSE_TREES_CAP_PER_HOUR"}
"""The policy: hours of grace before waiting becomes orphan, hours an unclaimed tree is assumed to
have a live creator, hours without a sign of life before an owner with no process is taken to have
stopped, the commits ahead, commits behind and hours with unlanded commits that raise a notice,
the minutes between repeats of one notice and the most messages one recipient gets per hour. The
environment variable that overrides each key is in ``ENVIRONMENT``; POOLHOUSE_TREES_LEAD names the
coordinator."""


def git(root: Path | str, *args: str) -> str:
    """Stripped standard output of ``git -C root args``; a failure or a stall is a RuntimeError."""
    try:
        done = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, check=False,
                              timeout=20, env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"})
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(f"git {args[0]} did not answer in 20 seconds") from error
    if done.returncode:
        raise RuntimeError(f"git {args[0]} failed (exit {done.returncode}): {done.stderr.strip()[-500:]}")
    return done.stdout.strip()


class Refused(Exception):
    """A removal or decision the registry will not make."""


def registry_path(root: Path | str) -> Path:
    """The registry file, shared by every worktree of the repository holding ``root``."""
    common = git(Path(root), "rev-parse", "--path-format=absolute", "--git-common-dir")
    return Path(common) / "poolhouse-trees.json"


def listing(root: Path | str) -> list[tuple[Path, str, str]]:
    """Every registered worktree as (path, branch, head); the first is the primary checkout."""
    found, path, head, branch = [], "", "", ""
    for line in [*git(Path(root), "worktree", "list", "--porcelain").splitlines(), ""]:
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
    return git(primary, "branch", "--show-current") or "HEAD"


def clean(data: Any) -> dict:
    """A registry value made usable: a garbled or half-written file loses its owners, not its trees."""
    data = data if isinstance(data, dict) else {}
    data["trees"] = {k: v for k, v in data.get("trees", {}).items() if isinstance(v, dict)} \
        if isinstance(data.get("trees"), dict) else {}
    data["history"] = data["history"] if isinstance(data.get("history"), list) else []
    for name in ("policy", "sent"):
        data[name] = data[name] if isinstance(data.get(name), dict) else {}
    return data


def recorded(root: Path | str) -> dict:
    """The registry as saved, without scanning or writing anything."""
    return clean(read_json(registry_path(root), {}))


def update(root: Path | str, change: Callable[[dict], Any]) -> Any:
    """Apply ``change`` to the registry under its lock and save it if it changed.

    Waits at most ``LOCK_TIMEOUT`` seconds for the lock (raises `lock.Busy`); the write replaces
    the file atomically; a garbled file is read as empty and rebuilt by the next scan.
    """
    path = registry_path(root)
    with lock.rewriting(path, timeout=LOCK_TIMEOUT):
        data = clean(read_json(path, {}))
        before = json.dumps(data, sort_keys=True)
        result = change(data)
        if json.dumps(data, sort_keys=True) != before:
            write_json(path, data)
    return result


def policy(data: dict) -> dict:
    """The thresholds: defaults, then the registry's own, then the environment."""
    merged = {**DEFAULTS, **{k: v for k, v in data.get("policy", {}).items() if k in DEFAULTS}}
    for key, name in ENVIRONMENT.items():
        text = os.environ.get(name, "")
        if text:
            try:
                merged[key] = float(text)
            except ValueError:
                continue
    return merged


def current_policy(root: Path | str) -> dict:
    """The policy in force for the repository holding ``root``."""
    return policy(recorded(root))


def set_lead(root: Path | str, name: str) -> None:
    """Record the coordinator that escalations go to."""
    update(root, lambda data: data.__setitem__("lead", name))


def new_entry(owner: str, purpose: str, base: str, now: float, pid: int = 0) -> dict:
    """A freshly created tree owned by ``owner``."""
    return {"owner": owner, "purpose": purpose, "base": base, "created": now, "seen": now, "pid": pid,
            "state": "active", "stopped": 0.0, "notified": {}, "kind": "work"}


def window(entry: dict, pol: dict) -> float:
    """Seconds an owner with no process is taken to be alive after its last sign of life."""
    return (pol["claim_h"] if entry["owner"] == UNKNOWN else pol["ttl_h"]) * 3600


def owner_alive(entry: dict, now: float, pol: dict) -> bool:
    """Whether the tree's owner is still running: its process when one is recorded, else a recent sign of life."""
    if entry["state"] != "active":
        return False
    if entry.get("pid"):
        return lock.pid_alive(int(entry["pid"]))
    return now - entry["seen"] < window(entry, pol)


def facts(path: Path, head: str, tip: tuple[str, str], entry: dict) -> dict:
    """Counts of ``head`` against the development tip; recomputed only when either moved."""
    snap = entry.get("snap") or {}
    if snap.get("head") == head and snap.get("dev") == tip[1]:
        return snap
    behind, ahead = (int(x) for x in git(path, "rev-list", "--left-right", "--count", f"{tip[1]}...{head}").split())
    unique = len([x for x in git(path, "cherry", tip[1], head).splitlines() if x.startswith("+")]) if ahead else 0
    stamps = git(path, "log", "--format=%ct", f"{tip[1]}..{head}").split() if unique else []
    return {"head": head, "dev": tip[1], "ahead": ahead, "behind": behind, "unique": unique,
            "oldest": min((int(s) for s in stamps), default=0),
            "tip_ts": int(git(path, "show", "-s", "--format=%ct", head))}


def standing(tree: tuple[Path, str, str], entry: dict, tip: tuple[str, str], clock: tuple[float, dict]) -> dict:
    """One tree's standing: counts against the development branch, dirty files (asked only when the
    owner is not active), owner and status."""
    (path, branch, head), (now, pol) = tree, clock
    snap = facts(path, head, tip, entry)
    alive = owner_alive(entry, now, pol)
    scratch = entry.get("kind") == "scratch"
    dirty = 0 if alive else len([x for x in git(path, "status", "--porcelain").splitlines() if x])
    landed = not snap["unique"] and not dirty
    since = entry["stopped"] or entry["seen"] + window(entry, pol)
    within = alive or now - since <= pol["grace_h"] * 3600
    status = ("scratch" if scratch else "active" if alive else entry["state"] if entry["state"] in DECIDED
              else "landed" if landed else "waiting" if within else "orphan")
    return {"path": str(path), "branch": branch or "(detached)", "head": head, "owner": entry["owner"],
            "purpose": entry["purpose"], "status": status, "alive": alive, "landed": landed,
            "ahead": snap["ahead"], "behind": snap["behind"], "unlanded": snap["unique"], "dirty": dirty,
            "since": since, "unlanded_age_h": max(0.0, (now - snap["oldest"]) / 3600) if snap["unique"] else 0.0,
            "age_h": max(0.0, (now - entry["created"]) / 3600), "tip_age_d": max(0.0, (now - snap["tip_ts"]) / 86400),
            "snap": snap}


def _apply(data: dict, live: dict, now: float, claimed: tuple[str, Path] | None) -> dict:
    """Register unseen trees (owner unknown), complete half-written entries, drop trees git no longer
    lists (kept in the history), mark owners whose process ended as stopped and refresh ``claimed``."""
    owner, here = claimed or ("", None)
    mine, pol = (str(here.resolve()) if here else ""), policy(data)
    for path, (_p, _branch, head) in live.items():
        entry = data["trees"].setdefault(path, new_entry(UNKNOWN, "", head, now))
        for key, value in new_entry(UNKNOWN, "", head, now).items():
            entry.setdefault(key, value)
        if owner and path == mine and entry["state"] == "active" and entry["owner"] in (UNKNOWN, owner):
            entry.update(owner=owner, seen=now)
        if entry["state"] == "active" and entry.get("pid") and not owner_alive(entry, now, pol):
            entry.update(state="finished", stopped=now)
    for path in [p for p in data["trees"] if p not in live]:
        gone = data["trees"].pop(path)
        data["history"] = [*data["history"], {"path": path, "owner": gone.get("owner", UNKNOWN),
                                              "mode": "removed-outside-the-tool", "at": now}][-100:]
    return data


def scan(root: Path | str, now: float, claimed: tuple[str, Path] | None = None) -> dict:
    """Reconcile the registry with ``git worktree list`` and give ``claimed`` (owner, tree) its tree."""
    live = {str(p): (p, b, h) for p, b, h in listing(root)[1:] if p.exists()}
    return update(root, lambda data: _apply(data, live, now, claimed))


def claim(root: Path | str, path: Path | str, owner: str, now: float, details: dict | None = None) -> None:
    """Register the worktree at ``path`` as created by ``owner`` (harness, land, by hand).

    ``details`` may hold the ``purpose``, the ``pid`` whose life stands for the owner's and a
    ``kind`` (``scratch`` for a disposable tree that never counts).
    """
    extra = details or {}
    target = str(Path(path).resolve())
    found = next(((p, h) for p, _b, h in listing(root)[1:] if str(p) == target), None)
    if found is None:
        return
    entry = new_entry(owner, extra.get("purpose", ""), found[1], now, int(extra.get("pid", 0)))
    entry["kind"] = extra.get("kind", "work")
    update(root, lambda data: data["trees"].__setitem__(target, entry))


def finish(root: Path | str, owner: str, now: float) -> list[dict]:
    """Mark every tree ``owner`` has as finished; returns the ones that still hold work."""
    def change(data: dict) -> None:
        for entry in data["trees"].values():
            if entry.get("owner") == owner and entry.get("state") == "active":
                entry.update(state="finished", stopped=now)

    scan(root, now)
    update(root, change)
    return [r for r in rows(root, now) if r["owner"] == owner and r["status"] in ("waiting", "orphan")]


def rows(root: Path | str, now: float, claimed: tuple[str, Path] | None = None, only: Path | None = None) -> list[dict]:
    """The standing of every tree other than the primary checkout, oldest first; of just ``only`` when given.

    Git is asked outside the lock; the registry is then reconciled and the cached counts saved
    best effort, so a locked registry still answers.
    """
    found = listing(root)
    tip = (found[0][1] or "HEAD", found[0][2])
    live = {str(p): (p, b, h) for p, b, h in found[1:] if p.exists()}
    data = _apply(copy.deepcopy(recorded(root)), live, now, claimed)
    pol = policy(data)
    done = [standing(live[path], data["trees"][path], tip, (now, pol)) for path in live
            if only is None or live[path][0] == only.resolve()]

    def save(fresh: dict) -> None:
        _apply(fresh, live, now, claimed)
        for row in done:
            fresh["trees"][row["path"]]["snap"] = row["snap"]

    with contextlib.suppress(lock.Busy, OSError):
        update(root, save)
    return sorted(done, key=lambda r: data["trees"][r["path"]]["created"])


def orphans(found: list[dict]) -> list[dict]:
    """The rows whose owner stopped longer than the grace ago with work nobody decided about."""
    return [r for r in found if r["status"] == "orphan"]


def waiting(found: list[dict]) -> list[dict]:
    """The rows whose owner stopped within the grace: finished, waiting to land."""
    return [r for r in found if r["status"] == "waiting" and not r["alive"]]


def past_grace(found: list[dict], pol: dict, now: float) -> list[dict]:
    """The orphans the landing gate counts: stopped longer than the grace ago."""
    return [r for r in orphans(found) if now - r["since"] > pol["grace_h"] * 3600]


def line(row: dict, now: float) -> str:
    """One tree as a line naming the tree, branch, work at stake, owner, age and the actions."""
    hours = max(0.0, (now - row["since"]) / 3600)
    label = "ORPHAN" if row["status"] == "orphan" else "FINISHED, waiting to land"
    return (f"{label} {row['branch']} {row['path']} +{row['ahead']} unlanded={row['unlanded']} dirty={row['dirty']} "
            f"owner={row['owner']} stopped {hours:.1f}h ago; scripts/worktrees close {row['path']} "
            "--landed | --bundle | --abandon 'reason'")


def suggested(row: dict) -> str:
    """The exact close command that fits this tree: land-by-hand first, else bundle, else abandon."""
    how = "--abandon 'reason'" if row["dirty"] else "--bundle" if row["unlanded"] else "--landed"
    return f"scripts/worktrees close {row['path']} {how}"


def lines(root: Path | str, now: float, limit: int = 8, found: list[dict] | None = None) -> list[str]:
    """Lines for the trees needing a decision, at most ``limit`` and a count of the rest."""
    found = rows(root, now) if found is None else found
    found = [r for r in found if r["status"] in ("waiting", "orphan") and not r["alive"]]
    out = [line(r, now) for r in found[:limit]]
    return out + ([f"... and {len(found) - limit} more trees needing a decision"] if len(found) > limit else [])


def _folder(primary: Path) -> Path:
    folder = registry_path(primary).parent / "poolhouse-bundles"
    folder.mkdir(exist_ok=True)
    return folder


def _bundle(primary: Path, row: dict, dev: str) -> Path:
    """Write the tree's unlanded commits to a bundle and verify that it holds the tip."""
    name = re.sub(r"[^A-Za-z0-9._-]", "_", row["branch"]) + "-" + row["head"][:8]
    target, ref = _folder(primary) / (name + ".bundle"), "refs/poolhouse/bundle/" + row["head"][:12]
    git(primary, "update-ref", ref, row["head"])
    try:
        git(primary, "bundle", "create", str(target), f"{dev}..{ref}")
        git(primary, "bundle", "verify", str(target))
        if row["head"] not in git(primary, "bundle", "list-heads", str(target)):
            raise Refused(f"the bundle {target} does not hold {row['head'][:8]}")
    finally:
        git(primary, "update-ref", "-d", ref)
    return target


def _preserve_dirty(primary: Path, row: dict) -> str:
    """Save the tree's uncommitted diff beside the bundles before an abandon deletes it."""
    patch = _folder(primary) / (re.sub(r"[^A-Za-z0-9._-]", "_", row["branch"]) + "-" + row["head"][:8] + ".dirty.patch")
    patch.write_text(git(Path(row["path"]), "diff", "--binary", "HEAD"))
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
        raise Refused(f"{target} is not a registered worktree (already removed? scripts/worktrees scan reconciles)")
    _check(row, mode, reason, caller)
    dev = development(primary)
    saved = _bundle(primary, row, dev) if mode == "bundle" else ""
    patch = _preserve_dirty(primary, row) if mode == "abandon" and row["dirty"] else ""
    git(primary, "worktree", "remove", "--force", str(target))
    if row["branch"] != "(detached)" and row["branch"] not in (*PROTECTED, dev):
        git(primary, "branch", "-D", row["branch"])
    git(primary, "worktree", "prune")
    done = {"path": str(target), "branch": row["branch"], "head": row["head"], "owner": row["owner"],
            "mode": mode, "reason": reason, "at": now, "bundle": str(saved), "dirty_patch": patch}
    update(root, lambda data: _forget(data, str(target), done))
    return done


def _forget(data: dict, path: str, done: dict) -> None:
    """Drop a removed tree from the registry and keep the decision in the history."""
    data["trees"].pop(path, None)
    data["history"] = [*data["history"], done][-100:]


def sweep(root: Path | str, now: float) -> list[dict]:
    """Remove every landed tree, and every clean scratch tree, whose owner is not active."""
    found = [r for r in rows(root, now) if not r["alive"]]
    done = [close(root, r["path"], ("landed", ""), now, "sweep") for r in found if r["status"] == "landed"]
    return done + [close(root, r["path"], ("abandon", "scratch tree"), now, "sweep")
                   for r in found if r["status"] == "scratch" and not r["dirty"]]
