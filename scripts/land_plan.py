"""Landing plan: unique patches per branch, the minimal cover, merge order, predicted conflicts."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import land_git as lg

MAX_COMMITS = 50
MAX_BEHIND = 20


@dataclass
class Entry:
    """One branch as the plan sees it."""

    branch: str
    patches: dict[str, str]
    files: list[str]
    behind: int
    worktree: str = ""
    dirty: bool = False
    contained_in: str = ""
    conflicts_with_target: list[str] = field(default_factory=list)
    conflicts_with_plan: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ids(self) -> set[str]:
        """The patch-ids this branch carries beyond the target."""
        return set(self.patches.values())

    def view(self) -> dict:
        """A JSON-ready summary."""
        return {"branch": self.branch, "commits": len(self.patches), "behind": self.behind,
                "files": self.files, "worktree": self.worktree, "dirty": self.dirty,
                "contained_in": self.contained_in, "conflicts_with_target": self.conflicts_with_target,
                "conflicts_with_plan": self.conflicts_with_plan, "warnings": self.warnings}


@dataclass
class Plan:
    """The ordered merge list, branches the cover dropped, and branches with nothing to land."""

    target: str
    order: list[Entry] = field(default_factory=list)
    covered: list[Entry] = field(default_factory=list)
    empty: list[str] = field(default_factory=list)
    dirty: list[str] = field(default_factory=list)


def cover_branches(root: Path, target: str) -> list[str]:
    """Local branches that have a worktree and are neither the target nor protected."""
    return sorted({t.branch for t in lg.trees(root) if t.branch and t.branch != target
                   and t.branch not in lg.PROTECTED and not t.branch.startswith("land/")})


def describe(root: Path, target: str, branch: str) -> Entry:
    """The unique patches, files, behind count and worktree state of one branch."""
    patches = lg.unique(root, target, branch)
    tree = lg.tree_of(root, branch)
    entry = Entry(branch, patches, lg.files_of(root, list(patches)), lg.behind(root, target, branch),
                  str(tree.path) if tree else "", lg.dirty(tree.path) if tree else False)
    if len(patches) > MAX_COMMITS:
        entry.warnings.append(f"{len(patches)} commits (over {MAX_COMMITS}): split into batches")
    if entry.behind > MAX_BEHIND:
        entry.warnings.append(f"{entry.behind} behind (over {MAX_BEHIND}): split or rebase")
    return entry


def minimal_cover(entries: list[Entry]) -> tuple[list[Entry], list[Entry]]:
    """Greedy set cover over patch-ids: the branches to merge and those a chosen one contains."""
    need = set().union(*(e.ids for e in entries)) if entries else set()
    chosen: list[Entry] = []
    left = list(entries)
    while need:
        best = max(left, key=lambda e: (len(e.ids & need), -e.behind, -len(e.ids), e.branch))
        chosen.append(best)
        need -= best.ids
        left.remove(best)
    dropped = []
    for e in left:
        holder = next((c for c in chosen if e.ids <= c.ids), None)
        e.contained_in = holder.branch if holder else ""
        dropped.append(e)
    return chosen, dropped


def order_and_predict(root: Path, target: str, chosen: list[Entry]) -> list[Entry]:
    """Order the chosen branches so each merges cleanly onto the plan so far where one can."""
    acc = lg.out(root, "rev-parse", target)
    pending = sorted(chosen, key=lambda e: (len(e.patches), e.branch))
    ordered: list[Entry] = []
    while pending:
        pick, outcome = pending[0], None
        for entry in pending:
            tree, bad = lg.merge_tree(root, acc, entry.branch)
            if not bad:
                pick, outcome = entry, tree
                break
        pending.remove(pick)
        if outcome is None:
            outcome, pick.conflicts_with_plan = lg.merge_tree(root, acc, pick.branch)
        pick.conflicts_with_target = lg.merge_tree(root, target, pick.branch)[1]
        if outcome and not pick.conflicts_with_plan:
            acc = lg.synthetic_merge(root, outcome, acc, pick.branch)
        ordered.append(pick)
    return ordered


def build(root: Path, target: str, branches: list[str]) -> Plan:
    """The plan for ``branches`` against ``target``."""
    lg.refuse_protected(target, *branches)
    plan = Plan(target)
    live = []
    for name in dict.fromkeys(branches):
        if name == target:
            continue
        if not lg.exists(root, name):
            raise ValueError(f"no such branch: {name}")
        entry = describe(root, target, name)
        if entry.dirty:
            plan.dirty.append(name)
        if entry.patches:
            live.append(entry)
        else:
            plan.empty.append(name)
    chosen, plan.covered = minimal_cover(live)
    plan.order = order_and_predict(root, target, chosen)
    return plan
