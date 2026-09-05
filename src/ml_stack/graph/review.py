"""A queue of proposals waiting for a person, and what accepting one does.

A proposal is what `ml_stack.graph.requests.propose` wrote: a request, the edits a model
read out of it, and a ``status``. `Queue` lists them for a page and acts on one: refusing
marks it, accepting lands its edits in the store through `ml_stack.graph.propose.apply`,
undoing takes an accepted one back to proposed. A caller that keeps its own journal of
accepted edits -- something a rebuild applies over fresh extractions -- hands the queue a
``journal`` pair, ``(fold, unfold)``, called with each edit and the journal's mapping.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from ml_stack.files import read_json, write_json

ACTIONS = ("accept", "refuse", "undo")

# what an edit is called in ml-stack's change vocabulary, where a proposal's name differs
AS_CHANGE = {"remove": "remove_node", "set_attr": "set_attribute"}
#: the reason a change carries when the proposal had none
ACCEPTED = "accepted in review"


def listed(proposals: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The proposals in key order, each carrying its key as ``id`` and its place as ``index``."""
    return [{"id": k, "index": i, **p} for i, (k, p) in enumerate(sorted(proposals.items()), 1)]


def as_change(edit: Mapping[str, Any]) -> Any:
    """One edit as the `Change` the store applies.

    ``remove_relation`` names an edge as one ``source|relation|target`` id, and ``set_attr``
    and ``remove`` are short forms; the store's vocabulary is `ml_stack.graph.propose`'s.
    """
    from ml_stack.graph.propose import Change

    op, target = str(edit.get("op") or ""), str(edit.get("target") or "")
    reason = str(edit.get("reason") or "") or ACCEPTED
    if op == "remove_relation":
        ends = target.split("|")
        if len(ends) != 3:
            return Change(op=op, target=target, reason=reason,
                          problems=[f"{target!r} is not source|relation|target"])
        return Change(op="remove_edge", target=ends[0], other=ends[2], name=ends[1],
                      reason=reason)
    return Change(op=AS_CHANGE.get(op, op), target=target, other=str(edit.get("other") or ""),
                  name=str(edit.get("name") or ""), value=str(edit.get("value") or ""),
                  reason=reason)


class Queue:
    """The proposals in ``path``, listed and acted on.

    ``store`` is the GraphStore path accepted edits land in; without one, or while it does
    not exist yet, accepting only marks the proposal and journals it. ``journal`` is
    ``(path, fold, unfold)``: a JSON mapping and the two functions that put one edit into it
    and take it out again, ``fold`` returning a reason when it cannot. ``exported`` runs
    after edits have landed in the store, for a caller that keeps a copy of the graph
    beside it; it returns a reason when that failed, or None.
    """

    def __init__(self, path: Path | str, *, store: Path | str | None = None,
                 journal: tuple[Path | str, Callable[..., str | None], Callable[..., Any]]
                 | None = None,
                 exported: Callable[[], str | None] | None = None,
                 log: Callable[[str], Any] = lambda _: None) -> None:
        self.path = Path(path)
        self.store = Path(store) if store else None
        self.journal = journal
        self.exported = exported
        self.log = log

    def read(self) -> dict[str, Any]:
        return read_json(self.path, {})

    def listed(self) -> list[dict[str, Any]]:
        """Every proposal, for a page: `listed` over what is on disk."""
        return listed(self.read())

    def act(self, key: str, action: str) -> list[str]:
        """Accept, refuse or undo one proposal on disk. Returns what could not be applied.

        Accepting journals first, then lands in the store, then exports: the journal is what
        a rebuild reads, so an edit the store would not take is still not lost. ``KeyError``
        for a key not in the queue, ``ValueError`` for an action that is not one of `ACTIONS`.
        """
        if action not in ACTIONS:
            raise ValueError(f"no such action: {action}")
        proposals = self.read()
        if key not in proposals:
            raise KeyError(key)
        problems: list[str] = []
        edits = proposals[key].get("edits") or []
        if action == "refuse":
            proposals[key]["status"] = "refused"
        elif action == "undo":
            self._journal(edits, taking_back=True)
            proposals[key]["status"] = "proposed"
        else:
            taken = self._journal(edits, taking_back=False, problems=problems)
            problems += self.into_store(taken)
            proposals[key]["status"] = "accepted"
        write_json(self.path, proposals)
        return problems

    def _journal(self, edits: list[dict[str, Any]], *, taking_back: bool,
                 problems: list[str] | None = None) -> list[dict[str, Any]]:
        """Fold every edit into the journal (or unfold it); returns the edits that folded."""
        if self.journal is None:
            return list(edits)
        path, fold, unfold = self.journal
        held = read_json(Path(path), {})
        taken: list[dict[str, Any]] = []
        for edit in edits:
            if taking_back:
                unfold(edit, held)
                continue
            said = fold(edit, held)
            if said:
                if problems is not None:
                    problems.append(said)
            else:
                taken.append(edit)
        write_json(Path(path), held)
        return taken

    def into_store(self, edits: list[dict[str, Any]]) -> list[str]:
        """Apply edits to the store and run ``exported``. Returns the problems.

        A store that is not there yet is left alone: the journal has the edits, and opening
        a store for writing would create an empty one.
        """
        if self.store is None or not edits:
            return []
        if not self.store.exists():
            self.log("store: not there yet, so the edit waits for the rebuild")
            return []
        try:
            from ml_stack.graph.propose import apply
            from ml_stack.graph.store import GraphStore

            changes = [as_change(e) for e in edits]
            problems = [f"{c.op} {c.target}: {'; '.join(c.problems)}"
                        for c in changes if c.problems]
            with GraphStore(self.store) as store:
                landed = apply(store, [c for c in changes if c.sound])
        except Exception as exc:  # noqa: BLE001 - the journal has it; the rebuild applies it
            return [f"store: not applied ({type(exc).__name__}: {str(exc)[:120]})"]
        problems += [f"{c.op} {c.target}: {'; '.join(c.problems) or 'the store would not take it'}"
                     for c in landed["skipped"]]
        if self.exported is not None:
            said = self.exported()
            if said:
                problems.append(said)
        return problems
