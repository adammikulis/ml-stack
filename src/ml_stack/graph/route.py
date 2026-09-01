"""Which tool a question wants, decided by an embedder rather than by the model.

A model offered four tools reads four descriptions before it reads the question, every turn.
That is cheap at four and not at fourteen, and the descriptions are the part measured as
already too long: 721 characters each, which took a 2B model from 17% to 70% recall and
cost a 120B twenty points over the same questions.

So this asks a small embedder first. It compares the question to *example questions* for
each tool -- `ask.TOOL_PROMPTS` -- rather than to prose about what the tool does, because a
question against questions is like-to-like and that is where the signal is. Comparing a
question to a capability description is the same mismatch that made the DOCUMENT and QUERY
prefixes necessary in `graph.vectors`.

Nothing here narrows anything by itself. It returns an order and a confidence, and the
caller decides whether to act on it -- because a router that quietly hides a tool the model
needed produces an answer that is wrong for a reason nobody can see.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

__all__ = ["MARGIN", "Routed", "chatty", "narrow", "rank"]

# How far the best tool must stand above the rest before a routing is worth acting on. The
# same shape of test as `vectors.stands_out`, and for the same reason: a greeting matches
# everything a little and nothing much, so the height of the best score says nothing and the
# gap between it and the field says everything.
MARGIN = 0.05


class Routed:
    """An order over tools, and whether it is confident enough to act on."""

    __slots__ = ("order", "scores", "clear")

    @property
    def chat(self) -> bool:
        """Whether this wants no graph at all -- a greeting, a joke, an aside.

        Only when the router was sure. A question mistaken for small talk is answered
        without ever looking anything up, which is the worst failure available here: it
        reads as a confident answer and is about nothing.
        """
        from ml_stack.graph.ask import CHAT

        return bool(self.clear and self.order and self.order[0] == CHAT)

    def __init__(self, order: list[str], scores: dict[str, float], clear: bool) -> None:
        self.order, self.scores, self.clear = order, scores, clear

    def __repr__(self) -> str:
        best = self.order[0] if self.order else "-"
        return f"Routed({best} {self.scores.get(best, 0):.3f}, clear={self.clear})"


def rank(question: str, prompts: Mapping[str, Sequence[str]], *, base_url: str,
         model: str, margin: float = MARGIN,
         embedder: Callable[..., list[list[float]]] | None = None) -> Routed:
    """Order the tools by how much this question looks like the questions they answer.

    A tool scores as its *best* matching example, not its average: a tool with one example
    that fits exactly and four that do not is the right tool, and averaging would bury it
    under a tool whose examples are all vaguely close.
    """
    from ml_stack.client.embed import cosine
    from ml_stack.graph.vectors import TASK

    if embedder is None:
        from ml_stack.client.embed import embed as embedder

    named = [(name, text) for name, texts in prompts.items() for text in texts]
    if not named or not str(question).strip():
        return Routed([], {}, False)

    try:
        # Both sides are questions, so both carry the *same* prefix. The asymmetric
        # QUERY/DOCUMENT pair is for a question against a document and is wrong here --
        # measured, it scored "tell me about Otto Vance" at 0.409 against an example
        # reading "tell me about Iris Bellweather", which are the same sentence.
        vectors = embedder([TASK + str(question)] + [TASK + t for _n, t in named],
                           base_url=base_url, model=model, timeout=60)
    except Exception:  # noqa: BLE001 - a router that cannot embed simply does not route
        return Routed([], {}, False)

    asked, rest = vectors[0], vectors[1:]
    best: dict[str, float] = {}
    for (name, _text), vector in zip(named, rest, strict=False):
        score = cosine(asked, vector)
        if score > best.get(name, -1.0):
            best[name] = score

    order = sorted(best, key=lambda n: -best[n])
    clear = False
    if len(order) > 1 and margin > 0:
        others = [best[n] for n in order[1:]]
        clear = best[order[0]] - (sum(others) / len(others)) >= margin
    return Routed(order, best, clear)


def chatty(question: str, prompts: Mapping[str, Sequence[str]], **kw: Any) -> bool:
    """Whether this message wants no graph at all. Convenience over :func:`rank`."""
    return rank(question, prompts, **kw).chat


def narrow(tools: Sequence[tuple[dict[str, Any], Any]], routed: Routed, *,
           keep: int = 2) -> list[tuple[dict[str, Any], Any]]:
    """The tools worth offering, in the router's order, or all of them.

    Nothing is dropped unless the routing was clear: a tool hidden from a model that needed
    it produces a wrong answer with no visible cause, which is worse than a longer prompt.
    `show` is never dropped -- a turn that cannot say what its answer is about lights
    nothing, whatever else it got right.

    A message routed to chat gets no tools whatsoever. That is the point of it: "tell me a
    joke" spends six model calls searching a graph for a joke otherwise.
    """
    from ml_stack.graph.ask import CHAT

    if routed.chat:
        return []
    if not routed.clear or keep <= 0:
        return list(tools)
    wanted = (set(routed.order[:keep]) | {"show"}) - {CHAT}
    kept = [pair for pair in tools
            if str((pair[0].get("function") or {}).get("name") or "") in wanted]
    return kept or list(tools)
