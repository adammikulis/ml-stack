"""When a tree has grown too big, drifted too far or been left, who is told and how often.

Conditions come from the policy in ``trees`` (all configurable). A condition is told when it first
appears and again only when its value changed, no sooner than ``repeat_min`` minutes after the last
telling; an unchanged condition goes once more to the coordinator when the owner has not acted. The
board gets the note; the owner is told while alive; the coordinator is told when the owner has
stopped, is unknown, or stayed silent. Every recipient (and the board) gets at most
``cap_per_hour`` messages an hour: the last one is a roll-up line and the rest are held.
A tree whose owner just finished is "finished, waiting to land", not an orphan, until the grace passes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from poolhouse import trees


@dataclass
class Notice:
    """One message to deliver about one tree (or a roll-up when ``path`` is empty)."""

    path: str
    branch: str
    owner: str
    keys: list[str]
    text: str
    to: list[str] = field(default_factory=list)
    board: bool = True


def conditions(row: dict, pol: dict) -> dict[str, tuple[object, str]]:
    """The thresholds this tree is over, by key, each as (value, words)."""
    found: dict[str, tuple[object, str]] = {}
    if row["status"] in ("landed", "bundled", "abandoned", "scratch"):
        return found
    if row["ahead"] >= pol["max_ahead"]:
        found["ahead"] = (row["ahead"], f"{row['ahead']} commits ahead (limit {pol['max_ahead']:g})")
    if row["behind"] > pol["max_behind"]:
        found["behind"] = (row["behind"], f"{row['behind']} behind (limit {pol['max_behind']:g})")
    if row["unlanded"] and row["unlanded_age_h"] > pol["max_age_h"]:
        found["age"] = (int(row["unlanded_age_h"]),
                        f"unlanded commits {row['unlanded_age_h']:.1f}h old (limit {pol['max_age_h']:g}h)")
    detail = f"{row['unlanded']} unlanded, {row['dirty']} dirty"
    if row["status"] == "waiting" and not row["alive"]:
        found["waiting"] = (detail, f"finished, waiting to land: {detail}")
    if row["status"] == "orphan":
        found["orphan"] = (detail, f"ORPHAN, owner stopped past the grace: {detail}")
    return found


def lead_of(data: dict) -> str:
    """The coordinator to escalate to: ``POOLHOUSE_TREES_LEAD``, else the one the registry recorded."""
    return os.environ.get("POOLHOUSE_TREES_LEAD") or data.get("lead", "")


def _admit(data: dict, who: str, now: float, pol: dict) -> str:
    """``send``, ``rollup`` (the last message of the hour) or ``hold`` for one more message to ``who``."""
    sent = [t for t in data["sent"].get(who, []) if now - t < 3600]
    data["sent"][who] = sent
    if len(sent) >= pol["cap_per_hour"]:
        return "hold"
    sent.append(now)
    return "rollup" if len(sent) == pol["cap_per_hour"] else "send"


def _recipients(row: dict, prior: dict | None, changed: bool, lead: str) -> list[str]:
    """Who is told about a condition: the live owner first, the coordinator when the owner is gone or silent."""
    names = []
    known = row["alive"] and row["owner"] != trees.UNKNOWN
    if known and changed:
        names.append(row["owner"])
    silent = prior is not None and not changed
    if lead and (not known or silent) and (changed or lead not in prior["to"]):
        names.append(lead)
    return names


def _due(row: dict, entry: dict, pol: dict, lead: str, now: float) -> tuple[list[str], list[str], str]:
    """The keys of this tree told now, who to tell and the words; records them as told."""
    told, names, said, keys = entry.setdefault("notified", {}), [], [], []
    for key, (value, words) in conditions(row, pol).items():
        prior = told.get(key) if isinstance(told.get(key), dict) else None
        if prior and now - prior["at"] < pol["repeat_min"] * 60:
            continue
        changed = prior is None or prior["value"] != value
        who = _recipients(row, prior, changed, lead)
        if not changed and not who:
            continue
        told[key] = {"value": value, "at": now, "to": sorted(set(prior["to"] if prior else []) | set(who))}
        names += who
        keys.append(key)
        said.append(words if changed else words + " (still)")
    return keys, list(dict.fromkeys(names)), "; ".join(said)


def plan(found: list[dict], data: dict, now: float) -> list[Notice]:
    """The notices due now, recorded as told and shaped by the per-recipient hourly cap."""
    pol, lead, out, held = trees.policy(data), lead_of(data), [], {}
    for row in found:
        entry = data["trees"].get(row["path"])
        if entry is None:
            continue
        keys, names, words = _due(row, entry, pol, lead, now)
        if not keys:
            continue
        note = Notice(row["path"], row["branch"], row["owner"], sorted(keys),
                      f"tree {row['branch']} ({Path(row['path']).name}, owner {row['owner']}): {words}", [])
        for who in names:
            verdict = _admit(data, who, now, pol)
            if verdict == "send":
                note.to.append(who)
            else:
                held.setdefault(who, verdict)
        note.board = _admit(data, "board", now, pol) == "send"
        out.append(note)
    return out + [Notice("", "", "", ["rollup"], f"tree notices for {who} are capped at "
                         f"{pol['cap_per_hour']:g} an hour; the rest are held: see scripts/worktrees", [who], False)
                  for who, verdict in held.items() if verdict == "rollup"]


def notify(root: Path | str, now: float, claimed: tuple[str, Path] | None = None,
           only_claimed: bool = False) -> list[Notice]:
    """The notices due now for every tree (or only the ``claimed`` one), recorded as told."""
    found = trees.rows(root, now, claimed, claimed[1] if only_claimed and claimed else None)
    return trees.update(root, lambda data: plan(found, data, now))


def status_lines(root: Path | str, now: float, found: list[dict] | None = None) -> list[str]:
    """Every tree over a threshold other than the waiting and orphan ones (which have their own line)."""
    out, pol = [], trees.current_policy(root)
    for row in trees.rows(root, now) if found is None else found:
        said = {k: v[1] for k, v in conditions(row, pol).items() if k not in ("waiting", "orphan")}
        if said:
            out.append(f"{row['branch']} ({row['owner']}, {row['status']}): " + "; ".join(said.values()))
    return out
