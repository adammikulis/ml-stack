"""When a tree has grown too big, drifted too far or been forgotten, who is told and how often.

Conditions come from the policy in ``trees`` (all configurable). Each condition on each tree is
told at most once per ``repeat_min`` minutes. The board always gets the note. The owner is told
while it is alive; the coordinator is told as well when the owner has stopped, is unknown, or was
already told this same condition and has not fixed it (silent).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from ml_stack.workspace import trees


@dataclass
class Notice:
    """One message to deliver about one tree."""

    path: str
    branch: str
    owner: str
    keys: list[str]
    text: str
    to: list[str] = field(default_factory=list)


def conditions(row: dict, pol: dict) -> dict[str, str]:
    """The thresholds this tree is over, by key, each with its words."""
    found = {}
    if row["status"] in ("landed", "bundled", "abandoned"):
        return found
    if row["ahead"] >= pol["max_ahead"]:
        found["ahead"] = f"{row['ahead']} commits ahead (limit {pol['max_ahead']:g})"
    if row["behind"] > pol["max_behind"]:
        found["behind"] = f"{row['behind']} behind (limit {pol['max_behind']:g})"
    if row["unlanded"] and row["unlanded_age_h"] > pol["max_age_h"]:
        found["age"] = f"unlanded commits {row['unlanded_age_h']:.1f}h old (limit {pol['max_age_h']:g}h)"
    if row["status"] == "orphan":
        found["orphan"] = f"owner stopped, {row['unlanded']} unlanded, {row['dirty']} dirty"
    return found


def lead_of(data: dict) -> str:
    """The coordinator to escalate to: ``ML_STACK_TREES_LEAD``, else the one the registry recorded."""
    return os.environ.get("ML_STACK_TREES_LEAD") or data.get("lead", "")


def plan(found: list[dict], data: dict, now: float) -> list[Notice]:
    """The notices due now, recording each as told so the next call within the interval stays quiet."""
    pol, lead, out = trees.policy(data), lead_of(data), []
    for row in found:
        entry = data["trees"].get(row["path"])
        if entry is None:
            continue
        told = entry.setdefault("notified", {})
        due = {k: v for k, v in conditions(row, pol).items() if k not in told or now - told[k] >= pol["repeat_min"] * 60}
        if not due:
            continue
        silent = any(k in told for k in due)
        names = []
        if row["alive"] and row["owner"] != trees.UNKNOWN:
            names.append(row["owner"])
        if lead and (not row["alive"] or silent or row["owner"] == trees.UNKNOWN):
            names.append(lead)
        for key in due:
            told[key] = now
        words = "; ".join(due.values())
        out.append(Notice(row["path"], row["branch"], row["owner"], sorted(due),
                          f"tree {row['branch']} ({Path(row['path']).name}, owner {row['owner']}): {words}",
                          list(dict.fromkeys(names))))
    return out


def notify(root: Path | str, now: float, claimed: tuple[str, Path] | None = None,
           only_claimed: bool = False) -> list[Notice]:
    """The notices due now for every tree (or only the ``claimed`` one), recorded as told."""
    found = trees.rows(root, now, claimed, claimed[1] if only_claimed and claimed else None)
    return trees.update(root, lambda data: plan(found, data, now))


def status_lines(root: Path | str, now: float) -> list[str]:
    """Every tree over a threshold, one line each, for the status page and the report."""
    out, pol = [], trees.current_policy(root)
    for row in trees.rows(root, now):
        said = {k: v for k, v in conditions(row, pol).items() if k != "orphan"}   # orphans have their own line
        if said:
            out.append(f"{row['branch']} ({row['owner']}, {row['status']}): " + "; ".join(said.values()))
    return out
