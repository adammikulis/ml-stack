"""A persona speaking through a model over the subgraph it knows.

`ModelWriter` is one `converse` a message: the persona's system prompt, its memory of
earlier threads in the same arc, the thread so far as turns, the tools over its subgraph
and an opening it is grounded in. `_Counting` wraps the client so a run can say what a
message cost in round trips.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ml_stack.world import World

__all__ = ["ROUNDS", "ModelWriter", "model_writer"]

ROUNDS = 4                            # tool rounds a persona gets to look something up


class _Counting:
    """A client that counts what it is asked, so a run can say what a message cost."""

    def __init__(self, client: Any) -> None:
        self.client = client
        self.calls = 0

    def chat(self, *args: Any, **kwargs: Any) -> Any:
        self.calls += 1
        return self.client.chat(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.client, name)


def _subgraph(graph: Mapping[str, Any], ids: Sequence[str]) -> dict[str, Any]:
    """The part of ``graph`` a persona knows: those nodes, the edges among them, their messages."""
    keep = {str(i) for i in ids}
    nodes = [n for n in (graph.get("nodes") or ()) if str(n.get("id")) in keep]
    edges = [e for e in (graph.get("edges") or ())
             if str(e.get("source")) in keep and str(e.get("target")) in keep]
    held = graph.get("messages") or {}
    wanted = {m for n in nodes for m in (n.get("messages") or ())}
    return {"nodes": nodes, "edges": edges,
            "messages": {m: held[m] for m in wanted if m in held}}


class ModelWriter:
    """A persona speaking through the tool loop over the subgraph it knows.

    Each call is one `converse`: the persona's system prompt, its memory of earlier threads
    in the same arc (read from ``store``) and the thread so far as turns, the tools over its
    subgraph, what the thread is about handed over as the ``opening`` -- its own entry, the
    project, the place, the people in the thread -- and the prompt asking for one or two
    sentences in its voice. The reply is stripped of planning and of any written-out ``show``
    call; an empty reply is returned empty and the simulation falls back to a template.

    ``last`` is the second call, whose ids are the ``Drew`` edges the thread memory wants.
    ``calls`` counts round trips and ``messages`` what they produced.
    """

    def __init__(self, client: Any, world: World, store: Any = None, *,
                 rounds: int = ROUNDS, memory: int = 2) -> None:
        self.client = _Counting(client)
        self.world = world
        self.store = store
        self.rounds = rounds
        self.memory = memory
        self.messages = 0
        self.last: Any = None
        # what the last message asserts, as far as is known: the opening it was grounded
        # in and every id its answer drew on -- a lower bound, since the persona may have
        # named more than it looked up
        self.last_ids: list[str] = []
        self._subgraphs: dict[tuple[str, int, int], dict[str, Any]] = {}

    @property
    def calls(self) -> int:
        return self.client.calls

    def known(self, persona: Mapping[str, Any]) -> dict[str, Any]:
        """The subgraph this persona knows, recomputed when the graph has grown."""
        graph = self.world.graph
        key = (str(persona.get("id") or ""), len(graph.get("nodes") or ()),
               len(graph.get("edges") or ()))
        if key not in self._subgraphs:
            ids = list(persona.get("knows") or [])
            if persona.get("id") and persona["id"] not in ids:
                ids.append(persona["id"])
            self._subgraphs.clear()
            self._subgraphs[key] = _subgraph(graph, ids)
        return self._subgraphs[key]

    def remembered(self, persona: Mapping[str, Any], context: Mapping[str, Any]) -> list[dict[str, str]]:
        """Turns from this persona's earlier threads in the same arc, oldest first."""
        arc_key = str(context.get("arc_key") or "")
        if self.store is None or not arc_key:
            return []
        from ml_stack.graph.thread import follow, threads

        me = str(persona.get("id") or "")
        current = str(context.get("thread") or "")
        names = [t["thread"] for t in threads(self.store)
                 if t["thread"].startswith(arc_key + "/") and not t["thread"].endswith("/" + current)]
        out: list[dict[str, str]] = []
        for name in sorted(names)[-self.memory:]:
            turns = follow(self.store, name, working=False)
            if not any(me in (t.meta.get("who") or ()) for t in turns):
                continue
            out.extend({"role": "assistant" if t.meta.get("speaker") == me else "user",
                        "content": t.text} for t in turns)
        return out

    def __call__(self, persona: Mapping[str, Any], prompt: str, context: Mapping[str, Any]) -> str:
        from ml_stack.asking import Asking
        from ml_stack.graph.conversation import converse
        from ml_stack.graph.looking import tools_for
        from ml_stack.graph.prompts import SYSTEM
        from ml_stack.graph.replies import spoken_show, without_notes

        graph = self.known(persona)
        me = str(persona.get("id") or "")
        labels = context.get("labels") or {}
        turns = self.remembered(persona, context)
        turns.extend({"role": "assistant" if speaker == me else "user",
                      "content": f"{labels.get(speaker, speaker)}: {text}"}
                     for speaker, text in (context.get("said") or ()))
        known = {str(n.get("id")) for n in graph.get("nodes") or ()}
        facts = context.get("facts") or {}
        opening = [me, *(str(facts.get(k) or "") for k in ("project_id", "place_id",
                                                             "topic_id", "group_id")),
                   *(str(p) for p in context.get("others") or ())]
        opening = list(dict.fromkeys(i for i in opening if i in known))
        answer = converse(prompt, graph, self.client, turns=turns,
                          system=str(persona.get("system") or SYSTEM),
                          tools=tools_for(graph), asking=Asking(rounds=self.rounds),
                          opening=opening)
        self.last = answer
        from ml_stack.graph.thread import drew_on

        drawn = [i for ids in drew_on(answer).values() for i in ids]
        self.last_ids = list(dict.fromkeys(i for i in (*opening, *drawn) if i in known))
        text, _ids = spoken_show(without_notes(answer.content))
        text = " ".join(text.split())
        if text:
            self.messages += 1
        return text


def model_writer(client: Any, world: World, store: Any = None, *,
                 rounds: int = ROUNDS) -> ModelWriter:
    """`ModelWriter` over ``client`` for that world; ``store`` is where memory is read from."""
    return ModelWriter(client, world, store, rounds=rounds)
