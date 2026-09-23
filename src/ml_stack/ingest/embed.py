"""The text an ingest store's nodes are embedded from, and the store filled with the
vectors of it."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ml_stack.client.embed import embed as _embed
from ml_stack.graph.store import GraphStore
from ml_stack.graph.vectors import (
    MOST_CHARS as MOST_CHARS,
    RememberOptions,
    remember,
    texts_for as texts_for,
    to_embed,
)


@dataclass(frozen=True)
class Embedded:
    """How an embedding pass over a store went: what it wrote, out of what it should."""

    written: int
    total: int


def embed_store(out: str | Path, *, base_url: str, model: str, smooth_hops: int = 0,
                log: Callable[[str], None] | None = None) -> Embedded:
    """Embed every node an ingest store holds, and write the vectors back into it.

    Opens the store writable, never read-only: a read-only handle cannot build the vector
    index a search needs, and would leave one silently missing. Every batch is embedded
    with ``expect_dim`` held to the store's own width for ``model`` -- its existing vectors'
    width if it has any, the first batch's otherwise -- so a server answering with a
    different model's width is refused a batch at a time rather than written in.
    """
    with GraphStore(out) as store:
        existing = store.embeddings(model=model)
        width = [len(next(iter(existing.values())))] if existing else []

        def fixed(texts: list[str], **kw: Any) -> list[list[float]]:
            if width:
                kw["expect_dim"] = width[0]
            vectors = _embed(texts, **kw)
            if not width and vectors:
                width.append(len(vectors[0]))
            return vectors

        graph = store.read()
        texts = texts_for(graph)
        written = remember(store, texts, base_url=base_url, model=model,
                           options=RememberOptions(embedder=fixed, smooth_hops=smooth_hops,
                                                   graph=graph, log=log))
        store.index()
    return Embedded(written=written, total=len(to_embed(texts)))
