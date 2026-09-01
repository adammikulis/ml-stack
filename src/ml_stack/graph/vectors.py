"""What a graph means, as vectors a reader can search.

The word index finds a name typed nearly right; vectors find a thing described another way
— "who fixes machines" reaching a robotics technician who never wrote either word. Both are
built here while there is write access, because a reader has none.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

# embeddinggemma is trained to be told what the embedding is for, and without it short
# business prose all lands above 0.9 against everything else — measured over 63 topic
# labels, where "hiring" and "retail" scored 0.922 and no threshold existed at all.
#
# Searching is asymmetric: a three-word question and a paragraph about somebody are not the
# same kind of text, and saying so is what stops the longest, most generic entry winning
# every query. Measured on a real graph of 290 nodes, ranking six entries against three
# questions: told they were all one kind, a generic "help" opportunity came first for two of
# the three and the robotics technician did not place at all for "who fixes machines". Told
# which was the question and which the document, "help" left the results entirely and the
# technician came second. The scores are lower that way (0.37-0.60 against 0.62-0.80) and it
# does not matter — nothing compares them across queries, only within one.
DOCUMENT = "title: none | text: "
QUERY = "task: search result | query: "

# Grouping things by how alike they are is a different task, and asks to be named as one.
TASK = "task: clustering | query: "

# One request per node is a round trip per node. Big enough to be worth batching, small
# enough that a failure loses a little work rather than all of it.
BATCH = 32

# How far the best match must stand above the rest before a search is worth acting on.
# Measured over the invented community: the score alone cannot tell a greeting from a
# question, because a greeting scores as high as one — "hi" 0.754 against "someone who can
# sell things" 0.740. What separates them is the gap between the best match and the mean of
# the rest: greetings 0.015-0.038, questions 0.029-0.196. So the shape of the results is the
# test, not their height.
MARGIN = 0.04


def stands_out(scores: Sequence[float], *, margin: float = MARGIN) -> bool:
    """Whether these similarities point somewhere, or are flat enough to mean nothing.

    A greeting embeds as well as a question does, so asking whether the best score is high
    fails: "hi" outscores "someone who can sell things" against the same graph. What differs
    is whether one entry stands apart from the field. When nothing does, there is nothing
    worth handing to a model, and it should go looking for itself.

    A ``margin`` of 0 or less turns the test off and everything stands out.
    """
    if margin <= 0:
        return True
    kept = [float(s) for s in scores]
    if not kept:
        return False
    return kept[0] - (sum(kept) / len(kept)) >= margin


def remember(store: Any, texts: Mapping[str, str], *, base_url: str, model: str,
             prefix: str = DOCUMENT, batch: int = BATCH,
             embedder: Callable[..., list[list[float]]] | None = None,
             log: Callable[[str], None] | None = None) -> int:
    """Embed what each node was read from, and store it against that node.

    ``texts`` is node id -> the words that stand for it, and the caller chooses them because
    only it knows what its graph is made of. Give it more than a label: a name is two or
    three words and every two-word phrase embeds within a whisker of every other, so a graph
    embedded by its labels retrieves noise. The sentences a node came from are what separate
    it.

    These are the document side of a search, so they carry :data:`DOCUMENT`; whoever embeds
    the question has to use :data:`QUERY`, or the two are not being compared as a question
    against an answer and the longest entry wins everything.

    Returns how many were written. Anything the server refuses is skipped rather than
    raised: a graph with most of its vectors is better than a rebuild that failed.
    """
    if embedder is None:
        from ml_stack.client.embed import embed as embedder
    ids = [i for i in sorted(texts) if str(texts[i]).strip()]
    written = 0
    for at in range(0, len(ids), max(1, batch)):
        chunk = ids[at:at + max(1, batch)]
        try:
            vectors = embedder([prefix + texts[i] for i in chunk],
                               base_url=base_url, model=model, timeout=300)
        except Exception as exc:  # noqa: BLE001 - one bad batch is not the whole graph
            if log:
                log(f"vectors: {len(chunk)} could not be embedded: {exc}")
            continue
        for node_id, vector in zip(chunk, vectors, strict=False):
            store.set_embedding(node_id, vector, model=model)
            written += 1
    if log:
        log(f"vectors: {written} of {len(ids)} nodes embedded with {model}")
    return written


def embedded(store: Any, *, model: str = "") -> int:
    """How many vectors the store holds, for the model that built them."""
    try:
        rows = store.query(
            "MATCH (e:Embedding) WHERE $m = '' OR e.model = $m RETURN count(e) AS n",
            {"m": str(model)})
    except RuntimeError:
        return 0                      # no Embedding table yet, which is none of them
    return int(rows[0]["n"]) if rows else 0
