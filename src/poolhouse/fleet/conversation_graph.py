"""Conversation, message, model and project relationships in GraphStore."""
from __future__ import annotations

import hashlib
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from poolhouse import lock
from poolhouse.fleet.conversation_legacy import read_legacy
from poolhouse.fleet.conversation_settings import checked
from poolhouse.fleet.conversation_types import Conversation, Message
from poolhouse.graph.columns import column
from poolhouse.graph.store import GraphStore

CONVERSATION = "conversation"
MESSAGE = "conversation-message"
SCHEMA = "_conversations"


def node_id(cid: str) -> str:
    return f"conversation:{cid}"


@contextmanager
def opened(root: Path) -> Iterator[GraphStore]:
    with (lock.only_one(root / "conversations.lock", timeout=30, announce=lambda text: None),
          GraphStore(root / "conversations.db", buffer_pool_size=32 << 20, max_db_size=1 << 30) as graph):
        if graph.get_doc(SCHEMA, {}).get("version") != 1:
            with graph.transaction():
                imported = 0
                for path in root.glob("*.json"):
                    conversation = read_legacy(path)
                    if conversation is not None:
                        save(graph, conversation)
                        for sequence, message in enumerate(conversation.messages):
                            append(graph, conversation, message, sequence)
                        imported += 1
                graph.put_doc(SCHEMA, {"version": 1, "imported": imported})
        yield graph


def get(graph: GraphStore, cid: str) -> Conversation | None:
    rows = graph.query("MATCH (c:Node {id:$id}) RETURN c.label AS title, c.attrs AS attrs", {"id": node_id(cid)})
    if not rows:
        return None
    attrs = column(rows[0]["attrs"], cid)
    messages = []
    found = graph.query("MATCH (c:Node {id:$id})-[:Edge {rel:'contains'}]->(m:Node) "
                        "RETURN m.label AS content, m.attrs AS attrs", {"id": node_id(cid)})
    ordered = [(column(row["attrs"], cid), row["content"]) for row in found]
    for info, content in sorted(ordered, key=lambda pair: pair[0]["sequence"]):
        messages.append(Message(role=info["role"], content=content, at=info["at"],
                                reasoning=info.get("reasoning", ""), status=info.get("status", "complete")))
    return Conversation(id=cid, title=rows[0]["title"], model=attrs.get("model", ""),
                        created=attrs["created"], messages=messages, settings=checked(attrs.get("settings")))


def all_ids(graph: GraphStore) -> list[str]:
    return [row["id"].removeprefix("conversation:") for row in graph.query(
        "MATCH (c:Node) WHERE c.kind=$kind RETURN c.id AS id", {"kind": CONVERSATION})]


def save(graph: GraphStore, conversation: Conversation) -> None:
    source = node_id(conversation.id)
    graph.upsert_node({"id": source, "kind": CONVERSATION, "label": conversation.title,
                       "attrs": {"version": 1, "created": conversation.created, "model": conversation.model,
                                 "settings": conversation.settings}})
    _relation(graph, source, "uses-model", "model", conversation.model)
    _relation(graph, source, "in-project", "project", conversation.settings["project"])


def _relation(graph: GraphStore, source: str, relation: str, kind: str, name: str) -> None:
    graph.query("MATCH (c:Node {id:$id})-[e:Edge {rel:$rel}]->() DELETE e", {"id": source, "rel": relation})
    if not name:
        return
    target = f"{kind}:{hashlib.sha256(name.encode()).hexdigest()[:24]}"
    graph.upsert_node({"id": target, "kind": kind, "label": name, "attrs": {"version": 1}})
    graph.upsert_edge({"source": source, "target": target, "rel": relation})


def append(graph: GraphStore, conversation: Conversation, message: Message, sequence: int) -> None:
    ident = f"message:{conversation.id}:{sequence}"
    graph.upsert_node({"id": ident, "kind": MESSAGE, "label": message.content,
                       "attrs": {"version": 1, "role": message.role, "at": message.at, "sequence": sequence,
                                 "reasoning": message.reasoning, "status": message.status}})
    graph.upsert_edge({"source": node_id(conversation.id), "target": ident, "rel": "contains"})


def remove(graph: GraphStore, cid: str) -> bool:
    if get(graph, cid) is None:
        return False
    graph.query("MATCH (c:Node {id:$id})-[:Edge {rel:'contains'}]->(m:Node) DETACH DELETE m", {"id": node_id(cid)})
    graph.query("MATCH (c:Node {id:$id}) DETACH DELETE c", {"id": node_id(cid)})
    return True
