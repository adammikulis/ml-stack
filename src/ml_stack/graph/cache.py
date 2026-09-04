"""An answer already given, given again.

Asking a graph a question costs seconds of a large model's time and a few thousand tokens.
Asking it the *same* question, of the same graph, with the same model and the same tools,
costs exactly the same again and produces the same answer — so it is worth keeping.

What makes this safe is the fingerprint. An answer depends on more than the question: the
graph it was read from, the model that wrote it, the system prompt it was written under, the
tools it could call and how they were described, and any shortlist handed over before it
started. All of them go into the key, so changing any one of them misses the cache rather
than serving an answer that was true of the old arrangement. Editing a tool's description
invalidates every answer that description could have shaped, which is exactly what should
happen — that edit changed the answers when it was measured.

Entries are kept under keys beginning with an underscore, which `GraphStore.docs` skips, so
a cache in the same store as a graph never leaks into `read()`.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

__all__ = ["PREFIX", "digest", "fingerprint", "forget", "kept", "recall", "remember", "asked"]

PREFIX = "_answer:"


def _hash(*parts: Any) -> str:
    blob = json.dumps(parts, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:24]


def digest(graph: Mapping[str, Any]) -> str:
    """A short hash of everything in a graph an answer could have been drawn from.

    Nodes, edges and the messages behind them. Not the stats or the metadata: a rebuild that
    changes only a count has not changed what is true about anybody, and throwing away every
    answer for it would mean the cache never survived a pipeline run.
    """
    nodes = sorted(((str(n.get("id")), str(n.get("label") or ""), str(n.get("kind") or ""),
                     n.get("attrs") or {}, sorted(map(str, n.get("messages") or ())))
                    for n in (graph.get("nodes") or ())), key=lambda n: n[0])
    edges = sorted((str(e.get("source")), str(e.get("rel") or ""), str(e.get("target")))
                   for e in (graph.get("edges") or ()))
    said = sorted((str(k), str((v or {}).get("text") or ""))
                  for k, v in (graph.get("messages") or {}).items())
    return _hash(nodes, edges, said)


def fingerprint(question: str, *, graph: Mapping[str, Any] | None = None, on: str = "",
                model: str = "", system: str = "", tools: Sequence[Any] = (),
                opening: Sequence[str] = (), limit: int = 0, context: Any = (),
                ways: Mapping[str, Any] | None = None) -> str:
    """The key under which this question's answer is the same answer.

    ``ways`` is how the question is asked -- `Asking.said()`, or any mapping of the flags
    `converse` was handed -- so an answer under one asking is never served under another.

    ``on`` is a graph digest already worked out; pass it rather than ``graph`` when asking
    many questions of one graph, since hashing a graph is the expensive part of this.

    ``tools`` are hashed by name *and description*: the words a tool is described in change
    what the model does with it, so an edit to them has to miss.

    ``context`` is whatever else the caller knows changes the answer — the turns before this
    one, most obviously. "And where is she based?" is a different question after a different
    question, and a cache that ignored what came before would answer it about the wrong
    person.
    """
    if not on:
        on = digest(graph or {})
    named = []
    for one in tools or ():
        fn = (one.get("function") if isinstance(one, Mapping) else None) or {}
        named.append((str(fn.get("name") or ""), str(fn.get("description") or ""),
                      fn.get("parameters") or {}))
    asked = sorted((str(k), v) for k, v in dict(ways or {}).items())
    return PREFIX + _hash(" ".join(str(question).split()).casefold(), on, model, system,
                          sorted(named), sorted(map(str, opening)), int(limit), context,
                          *([asked] if asked else []))


def remember(store: Any, key: str, answer: Any, *, question: str = "", on: str = "") -> None:
    """Keep an answer under its fingerprint. Anything unstorable is simply not kept.

    ``on`` is the graph digest this answer was drawn from. It is already inside the key, but
    a key is a hash and cannot be read back — so it is written beside the answer as well,
    which is the only way to later tell which entries belong to a graph that no longer
    exists. Without it a cache grows for the life of the machine.
    """
    from dataclasses import asdict, is_dataclass

    try:
        body = asdict(answer) if is_dataclass(answer) else dict(answer)
        store.put_doc(key, {"at": time.strftime("%FT%T"), "question": question,
                            "on": str(on), "answer": body})
    except Exception:  # noqa: BLE001 - a cache that cannot write is a cache that is slow
        pass


def recall(store: Any, key: str, *, kind: Callable[..., Any] | None = None,
           older_than: float = 0.0) -> Any | None:
    """The answer kept under ``key``, rebuilt as ``kind``, or None.

    ``older_than`` in seconds forgets anything staler; 0 keeps an answer until something it
    was made from changes, which is what the fingerprint is for.
    """
    try:
        held = store.get_doc(key)
    except Exception:  # noqa: BLE001 - an unreadable cache is a miss, not a failure
        return None
    if not isinstance(held, Mapping) or not isinstance(held.get("answer"), Mapping):
        return None
    if older_than > 0:
        try:
            made = time.mktime(time.strptime(str(held.get("at")), "%Y-%m-%dT%H:%M:%S"))
            if time.time() - made > older_than:
                return None
        except (ValueError, TypeError):
            return None
    body = dict(held["answer"])
    if kind is None:
        return body
    try:
        return kind(**body)
    except TypeError:
        return None            # the shape changed under us, which is a miss


def kept(store: Any) -> dict[str, Any]:
    """Every answer this store is holding, by key. For counting and for clearing."""
    try:
        rows = store.query("MATCH (d:Doc) WHERE d.key STARTS WITH $p "
                           "RETURN d.key AS key, d.value AS value", {"p": PREFIX})
    except Exception:  # noqa: BLE001
        return {}
    out = {}
    for row in rows:
        try:
            out[row["key"]] = json.loads(row["value"])
        except (ValueError, TypeError, KeyError):
            continue
    return out


def forget(store: Any, *, keeping: str = "") -> int:
    """Throw kept answers away, returning how many went.

    With ``keeping`` set to a graph digest, only answers drawn from *other* graphs go — which
    is how a rebuilt graph sweeps up after itself instead of leaving every answer it ever
    gave behind. An entry written before digests were recorded has no ``on`` and is swept,
    since there is no way to know it is still true.
    """
    held = kept(store)
    if not held:
        return 0
    if not keeping:
        store.query("MATCH (d:Doc) WHERE d.key STARTS WITH $p DELETE d", {"p": PREFIX})
        return len(held)
    doomed = [k for k, v in held.items()
              if not isinstance(v, Mapping) or str(v.get("on") or "") != keeping]
    for key in doomed:
        store.query("MATCH (d:Doc {key:$k}) DELETE d", {"k": key})
    return len(doomed)


def asked(store: Any, question: str, fresh: Callable[[], Any], *, key: str = "",
          kind: Callable[..., Any] | None = None, on_event: Any = None,
          older_than: float = 0.0, keep: Callable[[Any], bool] | None = None,
          **making: Any) -> tuple[Any, bool]:
    """The answer to ``question``, from the cache when nothing has changed, else asked.

    Returns ``(answer, remembered)``. ``fresh`` is called only on a miss, so the model is
    never troubled for a question already answered. A caller streaming to ``on_event`` gets
    the answer replayed as one `answer` event and a `done`, because a reader watching an
    empty panel cannot tell a fast answer from a broken one.

    ``keep`` decides whether an answer is worth remembering, and is asked *after* the model
    has spoken so it can judge what the turn did as well as what it said. Some answers must
    not be kept however cacheable they look: an answer that also filed a request is a receipt
    for something that happened once, and serving it again would tell the next person their
    request was filed when it was not.
    """
    on = str(making.get("on") or "")
    if not on and making.get("graph") is not None:
        on = digest(making["graph"])
        making = {**making, "on": on, "graph": None}
    if not key:
        key = fingerprint(question, **making)
    found = recall(store, key, kind=kind, older_than=older_than) if store is not None else None
    if found is not None:
        if on_event is not None:
            said = getattr(found, "content", None)
            if said is None and isinstance(found, Mapping):
                said = found.get("content")
            on_event({"event": "answer", "text": said or "", "remembered": True})
            on_event({"event": "done", "remembered": True})
        return found, True
    out = fresh()
    worth = (store is not None and out is not None
             and bool((getattr(out, "content", "") or "").strip())
             and (keep is None or bool(keep(out))))
    if worth:
        remember(store, key, out, question=question, on=on)
    return out, False
