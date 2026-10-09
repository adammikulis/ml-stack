"""``land run``: merge the planned branches in an integration worktree and verify them once."""

from __future__ import annotations

import contextlib
import json
import tempfile
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import land_check as lc
import land_git as lg
import land_plan

from ml_stack.activity.gate import tree_hash

UNION_FILES = {"HANDOFF.md"}
OWNER_KEYS = ("Agent-Label", "Agent", "Label")


@dataclass
class Batch:
    """An integration worktree and the merges made in it."""

    root: Path
    target: str
    base: str
    wt: Path
    branch: str
    order: list[str]
    merged: list[tuple[str, str]] = field(default_factory=list)
    ejected: list[dict] = field(default_factory=list)


def say(message: str) -> None:
    """Print one line of progress."""
    print(message, flush=True)


def state_file(root: Path) -> Path:
    """Where the last batch is remembered."""
    folder = lg.common_dir(root) / "land"
    folder.mkdir(exist_ok=True)
    return folder / "state.json"


def save(batch: Batch, status: str, extra: dict) -> None:
    """Write the batch state for ``land finish``."""
    data = {"integration": batch.branch, "worktree": str(batch.wt), "target": batch.target,
            "base": batch.base, "sources": batch.order, "merged": [b for b, _ in batch.merged],
            "ejected": batch.ejected, "status": status, "tree": tree_hash(batch.wt), **extra}
    state_file(batch.root).write_text(json.dumps(data, indent=1), encoding="utf-8")


def resolve_base(root: Path, target: str) -> str:
    """The commit to build on: the fetched upstream target when it is ahead of the local one."""
    if "origin" in lg.lines(root, "remote"):
        lg.git(root, "fetch", "origin", target, check=False)
    remote = f"origin/{target}"
    if lg.exists(root, remote):
        mine = lg.out(root, "rev-parse", target)
        if lg.git(root, "merge-base", "--is-ancestor", mine, remote, check=False).returncode == 0:
            return lg.out(root, "rev-parse", remote)
        if lg.git(root, "merge-base", "--is-ancestor", remote, mine, check=False).returncode:
            raise RuntimeError(f"{target} and {remote} have diverged; reconcile before landing")
    return lg.out(root, "rev-parse", target)


def union_resolve(wt: Path, path: str) -> None:
    """Resolve a conflicted file by keeping both sides."""
    with tempfile.TemporaryDirectory() as scratch:
        parts = []
        for stage in (2, 1, 3):
            show = lg.git(wt, "show", f":{stage}:{path}", check=False)
            part = Path(scratch) / f"s{stage}"
            part.write_text(show.stdout if show.returncode == 0 else "", encoding="utf-8")
            parts.append(str(part))
        merged = lg.git(wt, "merge-file", "--union", "-p", *parts, check=False).stdout
    (wt / path).write_text(merged, encoding="utf-8")
    lg.git(wt, "add", path)


def merge_one(wt: Path, branch: str) -> tuple[bool, list[str]]:
    """Merge ``branch`` with --no-ff; the conflicted paths when it cannot be completed."""
    done = lg.git(wt, "-c", "rerere.enabled=true", "-c", "rerere.autoUpdate=true", "merge", "--no-ff",
                  "-m", f"chore: merge {branch}", branch, check=False)
    if done.returncode == 0:
        return True, []
    left = lg.lines(wt, "diff", "--name-only", "--diff-filter=U")
    for path in [p for p in left if p in UNION_FILES]:
        union_resolve(wt, path)
    left = [p for p in left if p not in UNION_FILES]
    if not left and lg.git(wt, "commit", "--no-edit", "-m", f"chore: merge {branch}", check=False).returncode == 0:
        return True, []
    lg.git(wt, "merge", "--abort", check=False)
    return False, left or ["(merge failed)"]


def owner_of(root: Path, branch: str) -> str:
    """The agent label in the branch tip's trailers, else its worktree's directory name."""
    for key in OWNER_KEYS:
        value = lg.out(root, "log", "-1", f"--format=%(trailers:key={key},valueonly)", branch).strip()
        if value:
            return value
    tree = lg.tree_of(root, branch)
    return tree.path.name if tree else branch


def eject(batch: Batch, branch: str, check: str, evidence: str, sha: str = "") -> dict:
    """Record a branch as ejected and print the board-ready report."""
    count = len(lg.unique(batch.root, batch.target, branch))
    row = {"branch": branch, "check": check, "evidence": evidence, "first_bad": sha,
           "owner": owner_of(batch.root, branch), "commits": count}
    batch.ejected.append(row)
    say(f"EJECTED {branch} check={check} owner={row['owner']} commits={count}")
    say(f"  evidence: {evidence}")
    return row


def merge_all(batch: Batch, conflicts: str) -> bool:
    """Merge the remaining branches in order; false when ``conflicts`` is ``stop`` and one conflicts."""
    gone = {e["branch"] for e in batch.ejected}
    batch.merged = []
    for name in [n for n in batch.order if n not in gone]:
        before = lg.out(batch.wt, "rev-parse", "HEAD")
        try:
            ok, files = merge_one(batch.wt, name)
            why = "conflicts: " + ", ".join(files[:6])
        except (RuntimeError, OSError) as error:
            lg.git(batch.wt, "merge", "--abort", check=False)
            lg.git(batch.wt, "reset", "--hard", "-q", before, check=False)
            ok, why = False, f"the merge step crashed: {str(error)[:200]}"
        if ok:
            batch.merged.append((name, lg.out(batch.wt, "rev-parse", "HEAD")))
            continue
        eject(batch, name, "merge", why)
        if conflicts == "stop":
            return False
    return True


@contextlib.contextmanager
def probe_tree(batch: Batch) -> Iterator[Path]:
    """A throwaway detached worktree beside the integration one."""
    path = batch.wt.with_name(batch.wt.name + "-probe")
    lg.git(batch.root, "worktree", "add", "--detach", str(path), batch.base)
    try:
        yield path
    finally:
        lg.git(path, "reset", "--hard", "-q", check=False)
        lg.git(path, "clean", "-fdq", check=False)
        lg.git(batch.root, "worktree", "remove", str(path), check=False)


def first_bad(batch: Batch, check: lc.Check, probe: Path) -> int:
    """The 1-based index of the first merge after which ``check`` fails."""
    low, high = 0, len(batch.merged)
    while high - low > 1:
        mid = (low + high) // 2
        lg.git(probe, "checkout", "-q", "--detach", batch.merged[mid - 1][1])
        if lc.probe(probe, check):
            low = mid
        else:
            high = mid
    return high


def blame(batch: Batch, check: lc.Check, result: lc.Result) -> str:
    """Judge one failure: ``baseline`` or the ejected branch's name."""
    narrow = lc.restrict(check, result.failed_files)
    with probe_tree(batch) as probe:
        if not lc.probe(probe, narrow):
            return "baseline"
        index = first_bad(batch, narrow, probe) if len(batch.merged) > 1 else 1
    branch, sha = batch.merged[index - 1]
    what = ", ".join(result.failed_files[:4]) or (result.tail.splitlines() or [check.name])[-1]
    eject(batch, branch, check.name, what, sha)
    return branch


def verify(batch: Batch, workers: int) -> tuple[list[lc.Result], str, str]:
    """Run the checks for the diff; eject and re-verify until nothing more fails."""
    passed_files: set[str] = set()
    results: list[lc.Result] = []
    kind, reason = "none", ""
    for _ in range(len(batch.order) + 1):
        checks, kind, reason = lc.build_checks(batch.wt, batch.base, workers)
        tree = tree_hash(batch.wt)
        results, bad = [], []
        for check in checks:
            if check.files:
                check = lc.Check(check.name, check.tier, [*check.argv[:check.argv.index("all") + 3],
                                 *[f for f in check.files if f not in passed_files]],
                                 [f for f in check.files if f not in passed_files])
                if not check.files:
                    results.append(lc.Result(check.name, "reused", detail="passed earlier on a superset"))
                    continue
            res = lc.execute(batch.wt, check, tree)
            say(f"check {check.name}: {res.status} {res.detail}".rstrip())
            results.append(res)
            if res.status == "fail":
                bad.append((check, res))
            elif check.files:
                passed_files.update(check.files)
        if not bad:
            break
        removed = False
        for check, res in bad:
            who = blame(batch, check, res)
            res.status = "baseline" if who == "baseline" else "ejected"
            res.detail = who if who != "baseline" else "also fails on the clean target"
            removed = removed or who != "baseline"
        if not removed:
            break
        lg.git(batch.wt, "reset", "--hard", "-q", batch.base)
        if not merge_all(batch, "eject"):
            break
    return results, kind, reason


def create(root: Path, target: str, order: list[str]) -> Batch:
    """The integration worktree and branch cut from the fetched target."""
    base = resolve_base(root, target)
    stamp = f"{time.strftime('%Y%m%d-%H%M%S')}-{time.time_ns() % 1_000_000:06d}"
    wt = root.parent / f"{root.name}-land-{stamp}"
    branch = f"land/{stamp}"
    lg.git(root, "worktree", "add", "-b", branch, str(wt), base)
    return Batch(root, target, base, wt, branch, order)


def run(root: Path, target: str, names: list[str], flags: dict) -> tuple[int, dict]:
    """Plan, merge and verify; the exit status and the summary."""
    plan = land_plan.build(root, target, names)
    order = [e.branch for e in plan.order]
    candidates = [*order, *(e.branch for e in plan.covered), *plan.empty]
    summary: dict = {"command": "run", "target": target, "planned": order, "empty": plan.empty,
                     "dirty": plan.dirty}
    if not order:
        say("nothing to land")
        return 1, {**summary, "status": "empty"}
    if flags["dry_run"]:
        files = sorted({f for e in plan.order for f in e.files})
        checks, kind, reason = lc.preview(root, files)
        say(f"dry run: merge {', '.join(order)}; {kind} diff; checks: {', '.join(checks) or 'none'}"
            + (f"; full run: {reason}" if reason else ""))
        return 0, {**summary, "status": "dry-run", "kind": kind, "checks": checks, "full": reason}
    batch = create(root, target, order)
    say(f"integration {batch.branch} at {batch.wt}")
    if not merge_all(batch, flags["conflicts"]):
        save(batch, "stopped", {"candidates": candidates})
        return 2, {**summary, "status": "stopped", "ejected": batch.ejected}
    results, kind, reason = verify(batch, flags["workers"])
    full = lc.start_full(batch.wt, lg.common_dir(root)) if reason and batch.merged else {}
    blocked = [r.name for r in results if r.status in ("fail", "ejected")]
    status = "verified" if batch.merged and not [r for r in results if r.status == "fail"] else "failed"
    save(batch, status, {"kind": kind, "full": full, "candidates": candidates})
    done = {**summary, "status": status, "integration": batch.branch, "worktree": str(batch.wt),
            "merged": [b for b, _ in batch.merged], "ejected": batch.ejected, "kind": kind,
            "checks": [r.view() for r in results], "baseline": [r.name for r in results
                                                                   if r.status == "baseline"],
            "full": full or reason, "blocked": blocked, "tree": tree_hash(batch.wt)}
    return (0 if status == "verified" else 1), done
