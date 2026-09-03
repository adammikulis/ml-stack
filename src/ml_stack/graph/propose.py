"""Letting a model propose changes to a graph, without letting it make them.

Reading a graph and changing one are different privileges. A model that can look things up is
useful; a model that can quietly rewrite what a community said about itself is a liability. So
these tools return proposals — each one checked against the graph as it stands, each one
carrying the reason it was made — and something else decides.

The checking is the part worth having. A proposal to join two nodes where one does not exist
is a mistake, not an instruction; a proposal to add a node that is already there is a
duplicate; a proposal to remove something the asker has nothing to do with is the question a
reviewer most needs asked. All of that is settled here, before a person reads a list.

    tools, gather = proposing(graph)
    reply = client.chat(messages, tools=[*ASK_TOOLS, *tools])
    for change in gather(reply.tool_calls):
        ...                                  # review, then apply
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

KINDS_SHOWN = 12


@dataclass
class Change:
    """One proposed change, and whether it can be made at all."""

    op: str
    target: str = ""
    other: str = ""
    name: str = ""
    value: str = ""
    reason: str = ""
    problems: list[str] = field(default_factory=list)

    @property
    def sound(self) -> bool:
        """Whether the graph would accept it. Not whether anybody should."""
        return not self.problems

    def ids(self) -> str:
        """The op and what it names, as one line: ``remove_edge a -rel-> b``."""
        if self.op == "add_node":
            what = self.value
        elif self.op in ("add_edge", "remove_edge"):
            what = f"{self.target} -{self.name}-> {self.other}"
        elif self.op == "merge_nodes":
            what = f"{self.other} into {self.target}"
        else:
            what = self.target
        return f"{self.op} {what}".strip()

    def describe(self) -> str:
        parts = {"add_node": f"add {self.value or self.target}",
                 "add_edge": f"join {self.target} -{self.name}-> {self.other}",
                 "rename": f"rename {self.target} to {self.value}",
                 "set_attribute": f"set {self.name} of {self.target} to {self.value}",
                 "unset_attribute": f"unset {self.name} of {self.target}",
                 "remove_node": f"remove {self.target}",
                 "remove_edge": f"unjoin {self.target} -{self.name}-> {self.other}",
                 "merge_nodes": f"fold {self.other} into {self.target}"}
        said = parts.get(self.op, f"{self.op} {self.target}")
        return said + (f" — {self.reason}" if self.reason else "")


def tools_for(graph: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The change tools, described with this graph's own vocabulary rather than a fixed one."""
    kinds = sorted({str(n.get("kind") or "") for n in (graph.get("nodes") or ()) if n.get("kind")})
    rels = sorted({str(e.get("rel") or "") for e in (graph.get("edges") or ()) if e.get("rel")})
    kind_hint = ("what kind of entry it is"
                 + (f". Ones already in use: {', '.join(kinds[:KINDS_SHOWN])}" if kinds else ""))
    rel_hint = ("what one is to the other, as a short lower-case verb phrase"
                + (f". Ones already in use: {', '.join(rels[:KINDS_SHOWN])}" if rels else ""))
    reason = {"type": "string", "description": "why, in one sentence, from what you were told"}
    return [
        {"type": "function", "function": {
            "name": "add_node", "description": "Propose an entry the graph does not have yet.",
            "parameters": {"type": "object", "properties": {
                "label": {"type": "string", "description": "what it is called"},
                "kind": {"type": "string", "description": kind_hint},
                "reason": reason}, "required": ["label", "kind", "reason"]}}},
        {"type": "function", "function": {
            "name": "add_edge", "description": "Propose joining two entries that are not joined.",
            "parameters": {"type": "object", "properties": {
                "from_id": {"type": "string"}, "to_id": {"type": "string"},
                "rel": {"type": "string", "description": rel_hint},
                "reason": reason}, "required": ["from_id", "to_id", "rel", "reason"]}}},
        {"type": "function", "function": {
            "name": "rename", "description": "Propose a different name for an entry.",
            "parameters": {"type": "object", "properties": {
                "id": {"type": "string"}, "label": {"type": "string"}, "reason": reason},
                "required": ["id", "label", "reason"]}}},
        {"type": "function", "function": {
            "name": "set_attribute",
            "description": "Propose a value for one attribute of an entry.",
            "parameters": {"type": "object", "properties": {
                "id": {"type": "string"}, "name": {"type": "string"},
                "value": {"type": "string"}, "reason": reason},
                "required": ["id", "name", "value", "reason"]}}},
        {"type": "function", "function": {
            "name": "unset_attribute",
            "description": "Propose taking one attribute off an entry.",
            "parameters": {"type": "object", "properties": {
                "id": {"type": "string"}, "name": {"type": "string"}, "reason": reason},
                "required": ["id", "name", "reason"]}}},
        {"type": "function", "function": {
            "name": "remove_node", "description": "Propose taking an entry out of the graph.",
            "parameters": {"type": "object", "properties": {
                "id": {"type": "string"}, "reason": reason}, "required": ["id", "reason"]}}},
        {"type": "function", "function": {
            "name": "remove_edge", "description": "Propose unjoining two entries.",
            "parameters": {"type": "object", "properties": {
                "from_id": {"type": "string"}, "to_id": {"type": "string"},
                "rel": {"type": "string"}, "reason": reason},
                "required": ["from_id", "to_id", "rel", "reason"]}}},
        {"type": "function", "function": {
            "name": "merge_nodes",
            "description": "Propose that two entries are the same thing and should be one.",
            "parameters": {"type": "object", "properties": {
                "keep_id": {"type": "string"}, "remove_id": {"type": "string"},
                "reason": reason}, "required": ["keep_id", "remove_id", "reason"]}}},
    ]


NAMES = {t["function"]["name"] for t in tools_for({})}


def check(graph: Mapping[str, Any], change: Change) -> Change:
    """Fill in what the graph says is wrong with a proposal. It is not applied either way."""
    nodes = {str(n["id"]): n for n in (graph.get("nodes") or ())}
    labels = {str(n.get("label") or "").casefold() for n in nodes.values()}
    joined = {(str(e.get("source")), str(e.get("rel") or ""), str(e.get("target")))
              for e in (graph.get("edges") or ())}
    say = change.problems.append

    def known(node_id: str, what: str) -> bool:
        if node_id not in nodes:
            say(f"{what} {node_id!r} is not in the graph")
            return False
        return True

    if change.op == "add_node":
        if not change.value:
            say("no name given")
        elif change.value.casefold() in labels:
            say(f"{change.value!r} is already in the graph")
        if not change.name:
            say("no kind given")
    elif change.op == "add_edge":
        if known(change.target, "the first entry") and known(change.other, "the second entry"):
            if change.target == change.other:
                say("an entry cannot be joined to itself")
            elif (change.target, change.name, change.other) in joined:
                say("they are already joined that way")
        if not change.name:
            say("no relation given")
    elif change.op == "rename":
        if known(change.target, "the entry") and not change.value:
            say("no new name given")
    elif change.op in ("set_attribute", "unset_attribute"):
        known(change.target, "the entry")
        if not change.name:
            say("no attribute named")
    elif change.op == "remove_node":
        known(change.target, "the entry")
    elif change.op == "remove_edge":
        if (change.target, change.name, change.other) not in joined:
            say("they are not joined that way")
    elif change.op == "merge_nodes":
        if known(change.target, "the entry to keep") and known(change.other, "the entry to fold"):
            if change.target == change.other:
                say("those are the same entry")
            elif nodes[change.target].get("kind") != nodes[change.other].get("kind"):
                say("those are different kinds of thing")
    else:
        say(f"no such change: {change.op}")
    if not change.reason:
        say(f"no reason given for {change.ids()}")
    return change


def _read(name: str, args: Mapping[str, Any]) -> Change:
    text = lambda key: str(args.get(key) or "").strip()   # noqa: E731
    if name == "add_node":
        return Change(op=name, value=text("label"), name=text("kind"), reason=text("reason"))
    if name == "add_edge":
        return Change(op=name, target=text("from_id"), other=text("to_id"),
                      name=text("rel"), reason=text("reason"))
    if name == "rename":
        return Change(op=name, target=text("id"), value=text("label"), reason=text("reason"))
    if name == "set_attribute":
        return Change(op=name, target=text("id"), name=text("name"), value=text("value"),
                      reason=text("reason"))
    if name == "unset_attribute":
        return Change(op=name, target=text("id"), name=text("name"), reason=text("reason"))
    if name == "remove_node":
        return Change(op=name, target=text("id"), reason=text("reason"))
    if name == "remove_edge":
        return Change(op=name, target=text("from_id"), other=text("to_id"), name=text("rel"),
                      reason=text("reason"))
    if name == "merge_nodes":
        return Change(op=name, target=text("keep_id"), other=text("remove_id"),
                      reason=text("reason"))
    return Change(op=name, reason=text("reason"))


def proposing(graph: Mapping[str, Any]) -> tuple[list[dict[str, Any]],
                                                 Callable[[Sequence[Mapping[str, Any]]], list[Change]]]:
    """The change tools for this graph, and something to read what came back.

    Returns proposals, sound and unsound alike: a reviewer is better served by "this asks to
    remove something that is not there" than by silence.
    """
    def gather(calls: Sequence[Mapping[str, Any]]) -> list[Change]:
        out: list[Change] = []
        for call in calls or ():
            fn = call.get("function") or {}
            name = str(fn.get("name") or "")
            if name not in NAMES:
                continue
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except ValueError:
                args = {}
            out.append(check(graph, _read(name, args if isinstance(args, Mapping) else {})))
        return out

    return tools_for(graph), gather


def _ident(change: Change) -> str:
    return f"{change.name}:{change.value.strip().casefold().replace(' ', '-')}"


def _land(store: Any, change: Change, ident: Callable[[Change], str]) -> bool:
    if change.op == "add_node":
        store.upsert_node({"id": ident(change), "kind": change.name, "label": change.value,
                           "mentions": 0, "attrs": {}})
        return True
    if change.op == "add_edge":
        return store.upsert_edge({"source": change.target, "target": change.other,
                                  "rel": change.name, "weight": 1})
    if change.op == "rename":
        return store.rename(change.target, change.value)
    if change.op == "set_attribute":
        return store.set_attribute(change.target, change.name, change.value)
    if change.op == "unset_attribute":
        return store.unset_attribute(change.target, change.name)
    if change.op == "remove_node":
        return bool(store.drop([change.target]))
    if change.op == "remove_edge":
        return store.remove_edge(change.target, change.name, change.other)
    if change.op == "merge_nodes":
        try:
            store.merge_nodes(change.target, change.other)
        except KeyError as exc:
            change.problems.append(f"{exc.args[0]!r} is no longer in the store")
            return False
        return True
    return False


def apply(store: Any, changes: Iterable[Change], *,
          ident: Callable[[Change], str] | None = None) -> dict[str, list[Change]]:
    """Make the sound changes, together or not at all. Returns what landed and what did not.

    Each change is checked again against the store as it stands, so a change that went stale
    between proposing and applying is skipped with its problems filled in, and an earlier
    change in the batch can make a later one sound.
    """
    ident = ident or _ident
    applied: list[Change] = []
    skipped: list[Change] = []
    with store.transaction():
        for change in changes:
            checked = check(store.read(), dataclasses.replace(change, problems=[]))
            if not checked.sound:
                skipped.append(checked)
                continue
            (applied if _land(store, checked, ident) else skipped).append(checked)
    return {"applied": applied, "skipped": skipped}
