"""Storing what a graph means, and finding it again by meaning.

The embedding server is a real HTTP server answering /v1/embeddings, and the store is a real
store reopened on a fresh read-only handle — the shape retrieval actually uses.
"""

from __future__ import annotations

import json

import pytest
from conftest import json_reply
from ml_stack.graph.store import GraphStore
from ml_stack.graph.vectors import DOCUMENT, QUERY, embedded, remember

pytest.importorskip("ladybug", reason="the store needs ml-stack[store]")

# a toy space where the first number is "about machines" and the second "about selling"
SAID = {
    "person:ada": "I fix robots and machines all day",
    "person:bea": "I sell things and run campaigns",
    "topic:robotics": "robots, servos, machines",
}
PLACES = {"person:ada": [1.0, 0.0], "person:bea": [0.0, 1.0], "topic:robotics": [0.9, 0.1]}


def embeddings(seen: list[str] | None = None):
    """A server that answers with a vector chosen by what the text is about."""
    def handle(method, path, body):
        asked = list(json.loads(body or b"{}").get("input") or [])
        if seen is not None:
            seen.extend(asked)
        out = []
        for text in asked:
            machines = sum(w in text for w in ("robot", "machine", "servo", "fix"))
            selling = sum(w in text for w in ("sell", "campaign", "market"))
            total = (machines + selling) or 1
            out.append([machines / total, selling / total])
        return json_reply({"data": [{"embedding": v} for v in out]})

    return handle


def graph():
    return {"nodes": [{"id": i, "label": i.split(":")[1], "kind": i.split(":")[0]}
                      for i in SAID],
            "edges": [], "messages": {}}


class TestRemembering:
    def test_what_a_node_was_read_from_is_stored_and_found_again(self, server, tmp_path):
        instance = server(embeddings())
        path = tmp_path / "g.ladybug"
        with GraphStore(path) as store:
            store.write(graph())
            written = remember(store, SAID, base_url=instance.base_url, model="gemma")
        assert written == 3

        # the shape retrieval uses: a fresh read-only handle over the same store
        with GraphStore(path, read_only=True) as reader:
            assert embedded(reader, model="gemma") == 3
            # "who fixes machines" reaches the technician without sharing a word with them
            asking = [1.0, 0.0]
            near = reader.similar(asking, model="gemma", limit=3)
            assert near and near[0]["id"] in ("person:ada", "topic:robotics")
            assert [r["id"] for r in near][:2] != ["person:bea", "person:bea"]

    def test_these_are_stored_as_documents_not_as_questions(self, server, tmp_path):
        """Searching is asymmetric, and saying so is what stops the longest entry winning.

        Measured on a real graph: told a question and a paragraph were the same kind of
        text, a generic "help" entry came first for two of three questions and the person
        the question was about did not place at all.
        """
        seen: list[str] = []
        instance = server(embeddings(seen))
        with GraphStore(tmp_path / "g.ladybug") as store:
            store.write(graph())
            remember(store, SAID, base_url=instance.base_url, model="gemma")
        assert seen and all(t.startswith(DOCUMENT) for t in seen), seen[:1]
        assert not any(t.startswith(QUERY) for t in seen)

    def test_a_batch_the_server_refuses_costs_that_batch_and_no_more(self, server, tmp_path):
        """Most of a graph embedded beats a rebuild that raised halfway.

        The refusal has to be permanent for this to mean anything: the client retries, so a
        server that fails once and then works proves only that the retry works.
        """
        def refuses_one(method, path, body):
            asked = json.loads(body or b"{}").get("input") or []
            if any("sell" in text for text in asked):
                return json_reply({"error": "not this one, ever"}, 500)
            return embeddings()(method, path, body)

        instance = server(refuses_one)
        said: list[str] = []
        with GraphStore(tmp_path / "g.ladybug") as store:
            store.write(graph())
            written = remember(store, SAID, base_url=instance.base_url, model="gemma",
                               batch=1, log=said.append)
        assert written == 2, "one batch failed; the other two are still worth having"
        assert any("could not be embedded" in line for line in said)

    def test_nothing_to_say_about_a_node_is_not_an_empty_vector(self, server, tmp_path):
        instance = server(embeddings())
        with GraphStore(tmp_path / "g.ladybug") as store:
            store.write(graph())
            assert remember(store, {"person:ada": "   ", "person:bea": ""},
                            base_url=instance.base_url, model="gemma") == 0
            assert embedded(store) == 0


def test_a_store_holding_vectors_can_still_be_written_to(server, tmp_path):
    """Fails when the extension owning the vector index is not loaded for a writer.

    The message arrives at the write — "trying to delete from an index on table Embedding" —
    nowhere near the index that caused it, and only ever on a store that already has one.
    """
    instance = server(embeddings())
    path = tmp_path / "g.ladybug"
    with GraphStore(path) as store:
        store.write(graph())
        remember(store, SAID, base_url=instance.base_url, model="gemma")

    # reopened, as a rebuild does: writing and clearing both have to keep working
    with GraphStore(path) as store:
        store.write(graph())
        store._conn.execute("MATCH (e:Embedding) DELETE e")
        assert embedded(store) == 0
        assert remember(store, SAID, base_url=instance.base_url, model="gemma") == 3


def test_the_nearest_really_does_come_first(server, tmp_path):
    """Fails when similar() returns the index's own order and calls it a ranking.

    Everything that fuses rankings reads position as meaning, so an unsorted list is a leg
    of the search voting at random. Measured on a real store, the rows came back
    alphabetically by id and the best match sat fourth.
    """
    instance = server(embeddings())
    path = tmp_path / "g.ladybug"
    with GraphStore(path) as store:
        store.write(graph())
        remember(store, SAID, base_url=instance.base_url, model="gemma")

    with GraphStore(path, read_only=True) as reader:
        rows = reader.similar([1.0, 0.0], model="gemma", limit=3)
        scores = [r["similarity"] for r in rows]
        assert scores == sorted(scores, reverse=True), scores
        # and the answer to "about machines" is one of the two that are
        assert rows[0]["id"] in ("person:ada", "topic:robotics"), rows[0]["id"]
        assert rows[-1]["id"] == "person:bea", "the one about selling is furthest"


def test_stands_out_separates_a_question_from_a_greeting():
    """The gate reads the shape of the results, not how high the best score is.

    These are the numbers that were measured: a greeting scores higher than a real question
    and is still the flatter field, which is why a threshold on the score cannot work.
    """
    from ml_stack.graph.vectors import stands_out

    greeting = [0.754, 0.751, 0.744, 0.739, 0.731, 0.728]     # "hi"
    question = [0.740, 0.681, 0.652, 0.640, 0.633, 0.629]     # "someone who can sell things"

    assert greeting[0] > question[0]                          # the score says the wrong thing
    assert not stands_out(greeting)
    assert stands_out(question)


def test_stands_out_on_nothing_and_with_the_gate_off():
    from ml_stack.graph.vectors import stands_out

    assert not stands_out([])
    assert stands_out([], margin=0)                # off means everything passes, even nothing
    assert stands_out([0.7, 0.7, 0.7], margin=-1)
    assert not stands_out([0.9])                   # one result is its own mean: no margin
