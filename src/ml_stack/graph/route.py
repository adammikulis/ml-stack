"""Which tool a question wants, decided by an embedder rather than by the model.

A model offered four tools reads four descriptions before it reads the question, every turn.
That is cheap at four and not at fourteen, and the descriptions are the part measured as
already too long: 721 characters each, which took a 2B model from 17% to 70% recall and
cost a 120B twenty points over the same questions.

So this asks a small embedder first, comparing the question to *example questions* for each
tool -- `ask.TOOL_PROMPTS` -- rather than to prose about what the tool does. The ranking is
`client.select`; what is here is this repository's own data and its rule for acting on one.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

__all__ = ["MARGIN", "Routed", "chatty", "narrow", "rank"]

# How far the best tool must stand above the rest before a routing is worth acting on. A
# greeting matches everything a little and nothing much, so the height of the best score
# says nothing and the gap between it and the field says everything.
MARGIN = 0.05


class Routed:
    """An order over tools, and whether it is confident enough to act on."""

    __slots__ = ("order", "scores", "clear")

    @property
    def chat(self) -> bool:
        """Whether this wants no graph at all -- a greeting, a joke, an aside.

        Only when the router was sure. A question mistaken for small talk is answered
        without ever looking anything up, which reads as a confident answer about nothing.
        """
        from ml_stack.graph.prompts import CHAT

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

    A tool scores as its *best* matching example, not its average. A router that cannot
    embed does not route: it returns an empty order, and `narrow` then offers every tool.
    """
    from ml_stack.client.select import Selector
    from ml_stack.http import ServerError

    selector = Selector(prompts, model=model, margin=margin)
    try:
        ranked = selector.rank(question, base_url=base_url, embedder=embedder)
    except (ServerError, OSError):
        # An embedder that did not answer does not route. A VectorMismatch is not caught:
        # it is a bug, and swallowed it reads exactly like a flat ranking.
        return Routed([], {}, False)
    return Routed(ranked.order, ranked.scores, ranked.clear)


def chatty(question: str, prompts: Mapping[str, Sequence[str]], **kw: Any) -> bool:
    """Whether this message wants no graph at all. Convenience over :func:`rank`."""
    return rank(question, prompts, **kw).chat


def narrow(tools: Sequence[tuple[dict[str, Any], Any]], routed: Routed, *,
           keep: int = 2) -> list[tuple[dict[str, Any], Any]]:
    """The tools worth offering, in the router's order, or all of them.

    Nothing is dropped unless the routing was clear: a tool hidden from a model that needed
    it produces a wrong answer with no visible cause, which is worse than a longer prompt.
    `show` is never dropped. A message routed to chat gets no tools whatsoever.
    """
    from ml_stack.graph.prompts import CHAT

    if routed.chat:
        return []
    if not routed.clear or keep <= 0:
        return list(tools)
    wanted = (set(routed.order[:keep]) | {"show"}) - {CHAT}
    kept = [pair for pair in tools
            if str((pair[0].get("function") or {}).get("name") or "") in wanted]
    return kept or list(tools)
