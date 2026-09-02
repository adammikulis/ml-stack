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

A conversation of any length
----------------------------

What goes back to the model with a question is three things, in this order, and only the
last is chosen by recency:

* the latest *summary* -- one paragraph the small model writes every ``EVERY`` turns
  (``summarise``): what is established, what the asker wants, what is open, the entry ids it
  rests on. It is a ``Turn`` of role ``"summary"``, joined to those ids, kept out of the
  ordinary window, and it changes rarely, so it sits inside the model's cached prefix;
* what ``recall`` finds -- the two or three earlier turns, outside the window, whose words or
  meaning match the question, each with what it drew on;
* the last ``WINDOW`` ordinary turns, always, chosen by recency alone. A follow-up ("and
  where is she based?") resolves from these with nothing else; ``recall`` and the summary
  are additions ahead of them and never displace one.

A fact stated in conversation reaches the *graph* through the change-request path, not
through any of this: the summary and the recall keep it in the model's view; only an entry
makes the tools find it.
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

__all__ = ["DREW", "EVERY", "HOW", "SUMMARY", "Turn", "WINDOW", "drew_on", "follow",
           "forget_thread", "latest_summary", "of_node", "recall", "recent", "remember_turn",
           "summarise", "threads", "turn_of", "write_summary"]

TURN_TABLE = """CREATE NODE TABLE IF NOT EXISTS Turn(
    id STRING, thread STRING, seq INT64, at STRING, role STRING, text STRING,
    meta STRING, PRIMARY KEY (id))"""
AFTER_TABLE = """CREATE REL TABLE IF NOT EXISTS After(FROM Turn TO Turn)"""
DREW_TABLE = """CREATE REL TABLE IF NOT EXISTS Drew(FROM Turn TO Node, how STRING)"""
# The word index over what was said, so an earlier turn is findable by its words the way an
# entry is findable by its label. Built by whoever writes turns: a reader cannot build one.
TURN_INDEX = "CALL CREATE_FTS_INDEX('Turn', 'turn_index', ['text'])"

# How many ordinary turns always go back with a question, newest last, chosen by recency and
# nothing else. Raised from six when the summary took over carrying what is older: the window
# is for consistency and follow-ups, and is never trimmed to make room for what recall found.
WINDOW = 10
# How many ordinary turns pass between one summary and the next.
EVERY = 8
# The role a summary turn is kept under. Never in the ordinary window; `latest_summary` reads it.
SUMMARY = "summary"

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
    _words_index(store)


def _words_index(store: Any) -> bool:
    """The word index over turns, built once per writing handle; False when it cannot be.

    Measured: an index built before any turn was said sees every turn said after it, so it
    is built with the tables and never rebuilt. Rebuilding raises "already exists", which
    the store's own once-per-handle guard swallows. A store that cannot index -- no
    extension, a reader -- still keeps the conversation; recall simply has one voter fewer.
    """
    try:
        store._extension("fts")                       # noqa: SLF001 - same package
        store._index("turn_fts", TURN_INDEX)          # noqa: SLF001
        return True
    except Exception:  # noqa: BLE001
        return False


def _tag(thread: str) -> str:
    """The name a thread's turn vectors are kept under, so `similar` finds only this thread."""
    return "thread:" + str(thread)


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
                  meta: Mapping[str, Any] | None = None, after: str = "",
                  embedder: Callable[[Sequence[str]], Sequence[Sequence[float]]] | None = None
                  ) -> Turn:
    """Write one turn, chained after the last in its thread, joined to what it drew on.

    ``after`` names the turn this follows; left empty it follows whatever is last in the
    thread, which is what a linear conversation wants. Passing it explicitly is how a
    conversation branches -- two answers to the same question, kept side by side.

    An id the graph does not hold is not joined. A conversation must not be able to invent
    entries, which is the same rule the tool loop follows.

    ``embedder`` -- ``texts -> vectors`` -- makes the turn findable by meaning as well as by
    its words: the vector is kept under the thread's own name, so ``recall`` with the same
    embedder finds it and nothing else's. It is called before anything is written, so an
    embedder that fails writes nothing rather than half a turn. A summary is not embedded:
    it is sent every time, and has nothing to be recalled for.
    """
    _tables(store)
    vector: Sequence[float] | None = None
    if embedder is not None and role != SUMMARY and text.strip():
        vector = list(embedder([text])[0])
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
    if vector is not None:
        store.set_embedding(turn.id, vector, model=_tag(turn.thread))
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


def follow(store: Any, thread: str, *, limit: int = 0, working: bool = True,
           summaries: bool = False) -> list[Turn]:
    """A thread in the order it was said, oldest first.

    ``working`` carries what each turn drew on. Off, the turns come back as plain words --
    which is the same history a reader sees when the working is hidden, and cheaper.
    ``limit`` keeps the last N turns, because a long conversation is usually read from
    its end. Summaries are not part of the conversation and stay out unless ``summaries``
    asks for them: the window is the last N things *said*, and ``latest_summary`` is the
    way to the summary.
    """
    only = "" if summaries else "AND t.role <> $summary "
    rows = _rows(
        store, "MATCH (t:Turn) WHERE t.thread = $th " + only +
        "RETURN t.id AS id, t.thread AS thread, t.seq AS seq, t.at AS at, t.role AS role, "
        "t.text AS text, t.meta AS meta ORDER BY t.seq",
        {"th": str(thread), "summary": SUMMARY})
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


def recent(store: Any, thread: str, *, turns: int = WINDOW) -> list[dict[str, str]]:
    """The last few turns as chat messages, ready to send back to a model.

    Only role and content: the working is for the reader, not for the next prompt, and
    sending it back would spend the context on a transcript of searching. Chosen by
    recency alone -- this is the window, and nothing found by ``recall`` displaces a turn
    in it.
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
        # the turn vectors too, kept under the thread's name; a store that never embedded
        # a turn has no table to delete from, which is nothing to delete
        _rows(store, "MATCH (e:Embedding) WHERE e.model = $m DELETE e", {"m": _tag(thread)})
    return n


# ---------------------------------------------------------- of any length


def _ordinary(store: Any, thread: str) -> list[Turn]:
    """Every turn actually said in a thread, oldest first, without the working."""
    return follow(store, thread, working=False)


def recall(store: Any, thread: str, question: str, *,
           embedder: Callable[[Sequence[str]], Sequence[Sequence[float]]] | None = None,
           limit: int = 3, window: int = WINDOW) -> list[Turn]:
    """The earlier turns of this thread, outside the window, that best match the question.

    Two voters, fused the way ``search.hybrid`` fuses: the word index over what was said,
    and -- when ``embedder`` is given, the same one the turns were remembered with -- the
    vectors kept under the thread's name. Whatever is unavailable does not vote. Never a
    turn inside the last ``window`` ordinary turns, which the caller is already sending,
    and never a summary. Each comes with what it drew on, so the ids an old answer rested on
    come back with its words.

    Chosen by how well they match; returned in the order they were said, oldest first,
    because that is the order they are read in.
    """
    want = " ".join(str(question or "").split())
    if not want or limit <= 0:
        return []
    said = _ordinary(store, thread)
    earlier = said[:-window] if window > 0 else said
    if not earlier:
        return []
    allowed = {t.id: t for t in earlier}
    # ask for enough that the window, the other threads and the summaries do not crowd out
    # what is wanted: the index ranks every turn in the store, not this thread's
    k = len(said) + 4 * max(limit, window) + 16

    from ml_stack.graph.search import rrf_scored

    rankings: list[list[str]] = []
    words = _by_words(store, want, k)
    if words:
        rankings.append([i for i in words if i in allowed])
    if embedder is not None:
        try:
            vector = list(embedder([want])[0])
            near = store.similar(vector, model=_tag(thread), limit=k)
            rankings.append([str(r["id"]) for r in near if str(r["id"]) in allowed])
        except Exception:  # noqa: BLE001 - nothing embedded yet is one voter fewer, not an error
            pass
    chosen = [turn_id for turn_id, _ in rrf_scored(*rankings, limit=limit)] if rankings else []
    if not chosen:
        return []
    picked = sorted((allowed[i] for i in chosen), key=lambda t: t.seq)
    held = _drew_for(store, [t.id for t in picked])
    for one in picked:
        one.drew = held.get(one.id, {})
    return picked


def _by_words(store: Any, text: str, k: int) -> list[str]:
    """Turn ids whose words match, best first; empty when the store has no word index."""
    try:
        store._extension("fts")                       # noqa: SLF001 - same package
    except Exception:  # noqa: BLE001
        return []
    rows = _rows(
        store, "CALL QUERY_FTS_INDEX('Turn', 'turn_index', $q, TOP := $k) "
        "RETURN node.id AS id, score AS score", {"q": text, "k": int(k)})
    rows.sort(key=lambda r: -float(r.get("score") or 0.0))
    return [str(r["id"]) for r in rows]


def latest_summary(store: Any, thread: str) -> Turn | None:
    """The newest summary of a thread, with the ids it rests on; None before the first."""
    rows = _rows(
        store, "MATCH (t:Turn) WHERE t.thread = $th AND t.role = $summary "
        "RETURN t.id AS id, t.thread AS thread, t.seq AS seq, t.at AS at, t.role AS role, "
        "t.text AS text, t.meta AS meta ORDER BY t.seq DESC LIMIT 1",
        {"th": str(thread), "summary": SUMMARY})
    if not rows:
        return None
    one = _read(rows[0])
    one.drew = _drew_for(store, [one.id]).get(one.id, {})
    return one


def summarise(store: Any, thread: str, writer: Callable[[Sequence[Turn]], str], *,
              every: int = EVERY) -> Turn | None:
    """Roll the summary forward when ``every`` ordinary turns have been said since the last.

    ``writer(turns) -> str`` is the small model, or anything scripted to stand in for it:
    it is handed the previous summary (first, when there is one) and every ordinary turn
    since, each with what it drew on, and returns one paragraph -- what is established,
    what the asker wants, what is open, the entry ids it rests on. That paragraph is kept as
    a ``Turn`` of role ``"summary"``, joined to every id it names that a summarised turn or
    the previous summary drew on: an id the writer drops is an id the summary no longer
    rests on, and one it invents is not in that set to begin with.

    Returns the new summary, or None when it is not yet time or the writer said nothing.
    Costs one model call every ``every`` turns, over roughly ``every`` turns of text plus
    the previous paragraph; the answer it is written after has already gone out.
    """
    last = latest_summary(store, thread)
    since = int(last.seq) if last is not None else 0
    fresh = [t for t in follow(store, thread) if t.seq > since]
    if every <= 0 or len(fresh) < every:
        return None
    given: list[Turn] = ([last] if last is not None else []) + fresh
    text = str(writer(given) or "").strip()
    if not text:
        return None
    candidates: list[str] = []
    for turn in given:
        for how in HOW:
            for node_id in turn.drew.get(how, ()):
                if node_id not in candidates:
                    candidates.append(node_id)
    rests = [i for i in candidates if i in text]
    return remember_turn(store, thread=thread, role=SUMMARY, text=text,
                         drew={"shown": rests} if rests else None,
                         meta={"over": [fresh[0].seq, fresh[-1].seq]})


SUMMARY_SYSTEM = (
    "You keep the running notes of a conversation about a knowledge graph. Write one "
    "paragraph and nothing else: what has been established, what the asker wants, what is "
    "still open, and the entry ids (like person:iris) that it rests on -- keep every id you "
    "are given that still matters, written exactly as given. Fold the previous notes in "
    "rather than repeating them. No preamble, no headings.")


def write_summary(client: Any, turns: Sequence[Turn]) -> str:
    """A summary writer made from a chat client: ``partial(write_summary, client)``.

    One short call with no tools, ``think=False``; what the model says is the paragraph.
    The turns are laid out one per line as ``role: text`` with the ids each rests on, so
    the ids are in front of the model in the form it is asked to keep them in.
    """
    lines = []
    for turn in turns:
        ids = [i for how in HOW for i in turn.drew.get(how, ())]
        rests = f"  [rests on: {', '.join(dict.fromkeys(ids))}]" if ids else ""
        what = "previous notes" if turn.role == SUMMARY else turn.role
        lines.append(f"{what}: {' '.join(turn.text.split())[:2000]}{rests}")
    reply = client.chat([{"role": "system", "content": SUMMARY_SYSTEM},
                         {"role": "user", "content": "\n".join(lines)}], think=False)
    return str(getattr(reply, "content", "") or "").strip()
