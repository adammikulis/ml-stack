"""Which of a set of named documents a query asks for, decided by an embedder.

The documents are anything a caller wants chosen between -- tools, categories, routes.
Each name carries one or more texts and scores as its best matching text. One name may be
marked as the abstain document, so "none of these" is a candidate rather than a threshold.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence

from ml_stack.client.embed import MARGIN, TASK, cosine, stands_out

__all__ = ["Ranking", "Selector", "fingerprint"]

Embedder = Callable[..., list[list[float]]]

_VECTORS: dict[str, list[list[float]]] = {}


def _texts_of(docs: Mapping[str, str | Sequence[str]]) -> list[tuple[str, str]]:
    """``(name, text)`` for every document, in name order."""
    out: list[tuple[str, str]] = []
    for name in sorted(docs):
        value = docs[name]
        texts = [value] if isinstance(value, str) else list(value)
        out += [(name, str(t)) for t in texts if str(t).strip()]
    return out


def fingerprint(docs: Mapping[str, str | Sequence[str]], *, model: str,
                query_prefix: str, document_prefix: str, base_url: str = "") -> str:
    """The cache key for one embedded document set: the model, the prefixes and the texts."""
    digest = hashlib.sha256()
    for part in (model, query_prefix, document_prefix, base_url):
        digest.update(str(part).encode("utf-8"))
        digest.update(b"\x00")
    for name, text in _texts_of(docs):
        digest.update(name.encode("utf-8"))
        digest.update(b"\x1f")
        digest.update(text.encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()


class Ranking:
    """An order over document names, their scores, and whether it is worth acting on."""

    __slots__ = ("abstain", "clear", "order", "scores")

    def __init__(self, order: list[str], scores: dict[str, float], clear: bool,
                 abstain: str | None = None) -> None:
        self.order, self.scores, self.clear, self.abstain = order, scores, clear, abstain

    @property
    def best(self) -> str | None:
        """The highest-scoring name, abstain included."""
        return self.order[0] if self.order else None

    @property
    def abstained(self) -> bool:
        """Whether the abstain document won."""
        return self.abstain is not None and self.best == self.abstain

    def chose(self, *, require_clear: bool = False) -> str | None:
        """The name to act on, or None to abstain.

        ``require_clear`` also abstains when the winner does not stand out from the field.
        """
        if not self.order or self.abstained:
            return None
        return None if require_clear and not self.clear else self.order[0]

    def top(self, k: int) -> list[str]:
        """The ``k`` highest-scoring names, abstain removed."""
        return [n for n in self.order if n != self.abstain][:k]

    def __repr__(self) -> str:
        best = self.best or "-"
        return f"Ranking({best} {self.scores.get(best, 0.0):.3f}, clear={self.clear})"


class Selector:
    """A set of named documents, embedded once, ranked against a query.

    The vectors are cached on the model, the prefixes and the texts, so a description edit
    re-embeds and a steady process embeds the set once.
    """

    def __init__(self, docs: Mapping[str, str | Sequence[str]], *, model: str,
                 prefixes: tuple[str, str] = (TASK, TASK),
                 abstain: str | None = None, margin: float = MARGIN) -> None:
        self.docs = dict(docs)
        self.model = model
        self.query_prefix, self.document_prefix = prefixes
        self.abstain = abstain
        self.margin = margin

    def key(self, base_url: str = "") -> str:
        """This set's cache key."""
        return fingerprint(self.docs, model=self.model, query_prefix=self.query_prefix,
                           document_prefix=self.document_prefix, base_url=base_url)

    def vectors(self, *, base_url: str,
                embedder: Embedder | None = None) -> list[list[float]]:
        """The document vectors, embedded on first use and cached afterwards."""
        cached = _VECTORS.get(self.key(base_url))
        if cached is not None:
            return cached
        if embedder is None:
            from ml_stack.client.embed import embed as embedder
        named = _texts_of(self.docs)
        built = embedder([self.document_prefix + text for _n, text in named],
                         base_url=base_url, model=self.model, timeout=60)
        _VECTORS[self.key(base_url)] = built
        return built

    def rank(self, query: str, *, base_url: str, embedder: Embedder | None = None,
             among: Sequence[str] | None = None) -> Ranking:
        """Order the documents by how much the query looks like them.

        ``among`` keeps only those names; the abstain document is always scored. Raises
        whatever the embedder raises -- a caller that cannot embed decides for itself what
        an unranked turn means.
        """
        named = _texts_of(self.docs)
        if not named or not str(query).strip():
            return Ranking([], {}, False, self.abstain)
        if embedder is None:
            from ml_stack.client.embed import embed as embedder

        held = self.vectors(base_url=base_url, embedder=embedder)
        (asked,) = embedder([self.query_prefix + str(query)], base_url=base_url,
                            model=self.model, timeout=60)

        allowed = None if among is None else set(among) | ({self.abstain} - {None})
        best: dict[str, float] = {}
        for (name, _text), vector in zip(named, held, strict=False):
            if allowed is not None and name not in allowed:
                continue
            score = cosine(asked, vector)
            if score > best.get(name, -1.0):
                best[name] = score

        order = sorted(best, key=lambda n: -best[n])
        # A margin of 0 or less turns the gate off, and an ungated ranking is never
        # "clear": nothing was tested, so nothing may be narrowed on the strength of it.
        clear = self.margin > 0 and stands_out([best[n] for n in order], margin=self.margin)
        return Ranking(order, best, clear, self.abstain)
