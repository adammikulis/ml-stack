"""An embedder standing in for a real one, so a test ranks without a server."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

__all__ = ["bag_of_words_embedder", "words_in"]


def words_in(text: str) -> set[str]:
    """The words of one text, lowercased and stripped of trailing punctuation."""
    return {w.strip(".,?!") for w in str(text).casefold().split()}


def bag_of_words_embedder(*texts: Iterable[str]) -> Callable[..., list[list[float]]]:
    """An embedder standing in for a real one over a fixed vocabulary.

    Crude -- a text is its set of words -- but it prefers the same sentence to a different
    one, and it maps one text to one vector whatever else is in the batch, which is the
    property a real embedder has and a vocabulary built per call does not.
    """
    vocab = sorted({w for group in texts for t in group for w in words_in(t)})

    def embed(batch: Iterable[str], **_kw: Any) -> list[list[float]]:
        return [[1.0 if w in words_in(t) else 0.0 for w in vocab] for t in batch]

    return embed
