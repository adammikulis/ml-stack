"""A conversation as part of the graph, rather than beside it.

A chat about a graph is usually kept somewhere else: a list in the browser's memory, or a
line appended to a log. Both lose the thing worth keeping — that a turn *drew on* particular
entries, and which ones. The answer knows: it read some, traced a path through others, and
said its answer was about a few. That is an edge, and edges belong in the graph.

So a turn is a node, turns are chained newest-after-oldest, and each is joined to the
entries it drew on with *how* it drew on them. What that buys:

* history survives the tab, and can be reopened on another machine;
* "what have we said about this person?" is a query rather than a grep;
* an answer can be shown with its own working, or without it, because the working is
  attached rather than baked into the prose;
* a stale answer is findable when the entries under it change.

Nothing here decides what a reader sees. It records what happened; showing or hiding the
working is the caller's choice, and both are one query away.
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

__all__ = ["DREW", "HOW", "Turn", "drew_on", "follow", "forget_thread", "of_node", "recent",
           "remember_turn", "threads", "turn_of"]

TURN_TABLE = """CREATE NODE TABLE IF NOT EXISTS Turn(
    id STRING, thread STRING, seq INT64, at STRING, role STRING, text STRING,
    meta STRING, PRIMARY KEY (id))"""
AFTER_TABLE = """CREATE REL TABLE IF NOT EXISTS After(FROM Turn TO Turn)"""
DREW_TABLE = """CREATE REL TABLE IF NOT EXISTS Drew(FROM Turn TO Node, how STRING)"""

# How a turn touched an entry, weakest to strongest. `found` is a search result the model was
# shown; `read` is one it opened; `path` is one it travelled through; `shown` is one it said
# its answer was actually about. Only the last is what the answer *means* -- the rest are
# working, which is exactly why they are worth keeping separately rather than merged.
HOW = ("found", "read", "path", "shown")
DREW = "Drew"


class Turn:
    """One thing said, and what it drew on. A plain record; the store holds the truth."""

    __slots__ = ("id", "thread", "seq", "at", "role", "text", "meta", "drew")

    def __init__(self, *, id: str = "", thread: str = "", seq: int = 0, at: str = "",
                 role: str = "user", text: str = "", meta: Mapping[str, Any] | None = None,
                 drew: Mapping[str, Sequence[str]] | None = None) -> None:
        self.id = id or uuid.uuid4().hex[:16]
        self.thread = thread
        self.seq = int(seq)
        self.at = at or time.strftime("%FT%T")
        self.role = role
        self.text = text
        self.meta = dict(meta or {})
        self.drew = {k: list(v) for k, v in (drew or {}).items() if v}

    def __repr__(self) -> str:
        return f"Turn({self.role} {self.id[:6]} seq={self.seq} drew={sorted(self.drew)})"

    def as_dict(self) -> dict[str, Any]:
        return {"id": self.id, "thread": self.thread, "seq": self.seq, "at": self.at,
                "role": self.role, "text": self.text, "meta": self.meta, "drew": self.drew}


def _tables(store: Any) -> None:
    """Make the conversation tables, once. A reader cannot, and does not need to."""
    for statement in (TURN_TABLE, AFTER_TABLE, DREW_TABLE):
        try:
            store.query(statement)
        except Exception:  # noqa: BLE001 - a read-only store already has them, or has none
            return


def _rows(store: Any, cypher: str, params: Mapping[str, Any] | None = None) -> list[dict]:
    """Query, treating "no such table" as no rows.

    A graph that has never been talked to has no Turn table, and a reader cannot make one.
    Asking it for a conversation should come back empty rather than raise: history is an
    addition to a graph, not a part of every graph.
    """
    try:
        return store.query(cypher, params or {})
    except Exception:  # noqa: BLE001 - a graph with no conversation in it has no conversation
        return []


def drew_on(answer: Any) -> dict[str, list[str]]:
    """What an ``Answer`` touched, by how it touched it.

    Reads the four lists an answer already carries. Ids appear under every way they were
    reached: an entry that was found, read *and* named is all three, and flattening that to
    one relation would lose the only distinction a reader cares about.
    """
    # `Answer` calls it `show` -- the ids the model said its answer was about. The relation
    # is `shown`, which reads better on an edge. Neither name is worth changing for the other.
    spelt = {"shown": ("show", "shown")}
    out: dict[str, list[str]] = {}
    for how in HOW:
        names = spelt.get(how, (how,))
        got = None
        for name in names:
            got = getattr(answer, name, None)
            if got is None and isinstance(answer, Mapping):
                got = answer.get(name)
            if got:
                break
        ids = [str(i) for i in (got or ()) if str(i).strip()]
        if ids:
            out[how] = list(dict.fromkeys(ids))
    return out


def remember_turn(store: Any, *, thread: str, role: str, text: str,
                  drew: Mapping[str, Sequence[str]] | None = None,
                  meta: Mapping[str, Any] | None = None, after: str = "") -> Turn:
    """Write one turn, chained after the last in its thread, joined to what it drew on.

    ``after`` names the turn this follows; left empty it follows whatever is last in the
    thread, which is what a linear conversation wants. Passing it explicitly is how a
    conversation branches -- two answers to the same question, kept side by side.

    An id the graph does not hold is not joined. A conversation must not be able to invent
    entries, which is the same rule the tool loop follows.
    """
    _tables(store)
    rows = _rows(store, "MATCH (t:Turn) WHERE t.thread = $th RETURN max(t.seq) AS n",
                 {"th": str(thread)})
    last = int(rows[0]["n"] or 0) if rows and rows[0].get("n") is not None else 0
    turn = Turn(thread=str(thread), seq=last + 1, role=role, text=text, meta=meta, drew=drew)

    store.query(
        "CREATE (t:Turn {id:$id, thread:$th, seq:$seq, at:$at, role:$role, text:$text, "
        "meta:$meta})",
        {"id": turn.id, "th": turn.thread, "seq": turn.seq, "at": turn.at,
         "role": turn.role, "text": turn.text,
         "meta": json.dumps(turn.meta, ensure_ascii=False)})

    previous = after or _last_before(store, turn.thread, turn.seq)
    if previous:
        store.query("MATCH (a:Turn {id:$a}), (b:Turn {id:$b}) CREATE (a)-[:After]->(b)",
                    {"a": turn.id, "b": previous})

    kept: dict[str, list[str]] = {}
    for how, ids in turn.drew.items():
        for node_id in ids:
            done = store.query(
                "MATCH (t:Turn {id:$t}), (n:Node {id:$n}) CREATE (t)-[:Drew {how:$how}]->(n) "
                "RETURN 1 AS ok", {"t": turn.id, "n": str(node_id), "how": str(how)})
            if done:
                kept.setdefault(how, []).append(str(node_id))
    turn.drew = kept
    return turn


def _last_before(store: Any, thread: str, seq: int) -> str:
    rows = _rows(
        store, "MATCH (t:Turn) WHERE t.thread = $th AND t.seq < $seq RETURN t.id AS id, t.seq AS seq "
        "ORDER BY t.seq DESC LIMIT 1", {"th": str(thread), "seq": int(seq)})
    return str(rows[0]["id"]) if rows else ""


def _read(row: Mapping[str, Any]) -> Turn:
    try:
        meta = json.loads(row.get("meta") or "{}")
    except (ValueError, TypeError):
        meta = {}
    return Turn(id=str(row.get("id") or ""), thread=str(row.get("thread") or ""),
                seq=int(row.get("seq") or 0), at=str(row.get("at") or ""),
                role=str(row.get("role") or ""), text=str(row.get("text") or ""), meta=meta)


def follow(store: Any, thread: str, *, limit: int = 0, working: bool = True) -> list[Turn]:
    """A thread in the order it was said, oldest first.

    ``working`` carries what each turn drew on. Off, the turns come back as plain words --
    which is the same history a reader sees when the working is hidden, and cheaper.
    ``limit`` keeps the last N turns, because a long conversation is usually read from
    its end.
    """
    rows = _rows(
        store, "MATCH (t:Turn) WHERE t.thread = $th "
        "RETURN t.id AS id, t.thread AS thread, t.seq AS seq, t.at AS at, t.role AS role, "
        "t.text AS text, t.meta AS meta ORDER BY t.seq", {"th": str(thread)})
    turns = [_read(r) for r in rows]
    if limit > 0:
        turns = turns[-limit:]
    if working and turns:
        held = _drew_for(store, [t.id for t in turns])
        for one in turns:
            one.drew = held.get(one.id, {})
    return turns


def _drew_for(store: Any, ids: Sequence[str]) -> dict[str, dict[str, list[str]]]:
    """What each of those turns drew on, in one query rather than one per turn."""
    if not ids:
        return {}
    rows = _rows(
        store, "MATCH (t:Turn)-[d:Drew]->(n:Node) WHERE list_contains($ids, t.id) "
        "RETURN t.id AS turn, d.how AS how, n.id AS node", {"ids": [str(i) for i in ids]})
    out: dict[str, dict[str, list[str]]] = {}
    for row in rows:
        out.setdefault(str(row["turn"]), {}).setdefault(str(row["how"]), []).append(
            str(row["node"]))
    return out


def turn_of(store: Any, turn_id: str, *, working: bool = True) -> Turn | None:
    """One turn by id, with what it drew on."""
    rows = _rows(
        store, "MATCH (t:Turn {id:$id}) RETURN t.id AS id, t.thread AS thread, t.seq AS seq, "
        "t.at AS at, t.role AS role, t.text AS text, t.meta AS meta", {"id": str(turn_id)})
    if not rows:
        return None
    one = _read(rows[0])
    if working:
        one.drew = _drew_for(store, [one.id]).get(one.id, {})
    return one


def of_node(store: Any, node_id: str, *, how: Iterable[str] = ()) -> list[Turn]:
    """Every turn that drew on this entry, newest first.

    This is the question a flat log cannot answer: "what have we said about this person?"
    ``how`` narrows it -- ``how=("shown",)`` finds the turns an entry was actually the
    subject of, rather than every turn that happened to open it on the way past.
    """
    wanted = [str(h) for h in how]
    rows = _rows(
        store, "MATCH (t:Turn)-[d:Drew]->(n:Node {id:$n}) "
        "WHERE size($how) = 0 OR list_contains($how, d.how) "
        "RETURN DISTINCT t.id AS id, t.thread AS thread, t.seq AS seq, t.at AS at, "
        "t.role AS role, t.text AS text, t.meta AS meta ORDER BY at DESC, seq DESC",
        {"n": str(node_id), "how": wanted})
    return [_read(r) for r in rows]


def threads(store: Any) -> list[dict[str, Any]]:
    """Every conversation held, newest first: its name, how long, and when it was last said to."""
    rows = _rows(
        store, "MATCH (t:Turn) RETURN t.thread AS thread, count(t) AS turns, max(t.at) AS last "
        "ORDER BY last DESC")
    return [{"thread": str(r["thread"]), "turns": int(r["turns"]), "last": str(r["last"])}
            for r in rows]


def recent(store: Any, thread: str, *, turns: int = 6) -> list[dict[str, str]]:
    """The last few turns as chat messages, ready to send back to a model.

    Only role and content: the working is for the reader, not for the next prompt, and
    sending it back would spend the context on a transcript of searching.
    """
    return [{"role": t.role, "content": t.text}
            for t in follow(store, thread, limit=turns, working=False) if t.text.strip()]


def forget_thread(store: Any, thread: str) -> int:
    """Delete a conversation and its joins. Returns how many turns went."""
    held = _rows(store, "MATCH (t:Turn) WHERE t.thread = $th RETURN count(t) AS n",
                 {"th": str(thread)})
    n = int(held[0]["n"]) if held else 0
    if n:
        store.query("MATCH (t:Turn)-[d:Drew]->(:Node) WHERE t.thread = $th DELETE d",
                    {"th": str(thread)})
        store.query("MATCH (a:Turn)-[d:After]->(b:Turn) WHERE a.thread = $th "
                    "OR b.thread = $th DELETE d", {"th": str(thread)})
        store.query("MATCH (t:Turn) WHERE t.thread = $th DELETE t", {"th": str(thread)})
    return n
