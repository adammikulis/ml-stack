"""The text an ingest store's nodes are embedded from, and the store filled with the
vectors of it."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from ml_stack.client.embed import embed as _embed
from ml_stack.graph.store import GraphStore
from ml_stack.graph.vectors import remember

MOST_CHARS = 1400


def texts_for(graph: Mapping[str, Any], *, most: int = MOST_CHARS) -> dict[str, str]:
    """Node id -> its label and the sentences it was read from, capped at ``most`` characters."""
    messages = graph.get("messages") or {}
    out: dict[str, str] = {}
    for node in graph.get("nodes") or ():
        said = " ".join((messages.get(m) or {}).get("text", "")
                        for m in (node.get("messages") or ()))
        out[str(node["id"])] = (str(node.get("label", "")) + " — " + said)[:most]
    return out


def embed_store(out: str | Path, *, base_url: str, model: str, smooth_hops: int = 0,
                log: Callable[[str], None] | None = None) -> int:
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
        written = remember(store, texts_for(graph), base_url=base_url, model=model,
                           embedder=fixed, smooth_hops=smooth_hops, graph=graph, log=log)
        store.index()
    return written
