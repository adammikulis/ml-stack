"""The text an ingest store's nodes are embedded from, and the store filled with the
vectors of it. Every fixture is invented; nothing hits a real model server."""

from __future__ import annotations

import json

import pytest
from conftest import json_reply

pytest.importorskip("ladybug", reason="the store needs ml-stack[store]")

from ml_stack.graph.store import GraphStore
from ml_stack.graph.vectors import embedded
from ml_stack.ingest.embed import embed_store, texts_for

GRAPH = {
    "nodes": [
        {"id": "person:ada", "kind": "person", "label": "Ada Lovelace", "mentions": 1,
         "attrs": {}, "messages": ["m1"]},
        {"id": "person:bea", "kind": "person", "label": "Bea Marlow", "mentions": 1,
         "attrs": {}, "messages": ["m2"]},
    ],
    "edges": [],
    "messages": {"m1": {"text": "Ada fixes the loading robots on the night shift"},
                "m2": {"text": "Bea handles retail sales and outreach campaigns"}},
}


def embeddings_by_topic():
    """A server that answers with a vector chosen by what the text is about."""
    def handle(method, path, body):
        asked = list(json.loads(body or b"{}").get("input") or [])
        out = []
        for text in asked:
            machines = sum(w in text for w in ("robot", "fix", "machine"))
            selling = sum(w in text for w in ("sell", "campaign", "retail"))
            total = (machines + selling) or 1
            out.append([machines / total, selling / total])
        return json_reply({"data": [{"embedding": v} for v in out]})
    return handle


# -- texts_for --------------------------------------------------------------------------


def test_texts_for_carries_the_sentences_a_node_was_read_from_not_only_its_label():
    texts = texts_for(GRAPH)
    assert texts["person:ada"] != "Ada Lovelace"
    assert "robots" in texts["person:ada"] and "robots" not in texts["person:bea"]
    assert "campaigns" in texts["person:bea"]


def test_texts_for_respects_the_character_cap():
    graph = {"nodes": [{"id": "n1", "label": "n1", "messages": ["m1"]}],
            "messages": {"m1": {"text": "word " * 1000}}}
    texts = texts_for(graph, most=50)
    assert len(texts["n1"]) == 50


def test_texts_for_skips_a_node_with_nothing_said():
    graph = {"nodes": [{"id": "n1", "label": "", "messages": []}], "messages": {}}
    assert texts_for(graph) == {"n1": " — "}


# -- embed_store --------------------------------------------------------------------------


def test_embed_store_opens_a_writable_handle_and_the_index_finds_the_vectors(server, tmp_path):
    instance = server(embeddings_by_topic())
    path = tmp_path / "g.ladybug"
    with GraphStore(path) as store:
        store.write(GRAPH)

    written = embed_store(str(path), base_url=instance.base_url, model="gemma")
    assert written == 2

    # the shape retrieval uses: a fresh read-only handle over the same store
    with GraphStore(path, read_only=True) as reader:
        assert embedded(reader, model="gemma") == 2
        near = reader.similar([1.0, 0.0], model="gemma", limit=2)
        assert near and near[0]["id"] == "person:ada"


def test_embed_store_refuses_clearly_on_a_read_only_path(tmp_path):
    blocked = tmp_path / "blocked"
    blocked.mkdir(mode=0o500)
    try:
        with pytest.raises(PermissionError):
            embed_store(str(blocked / "sub" / "g.ladybug"), base_url="http://127.0.0.1:1",
                       model="gemma")
    finally:
        blocked.chmod(0o700)


def test_a_dimension_mismatch_is_refused_a_batch_at_a_time_not_corrupted(server, tmp_path):
    def three_wide(method, path, body):
        asked = list(json.loads(body or b"{}").get("input") or [])
        return json_reply({"data": [{"embedding": [1.0, 0.0, 0.0]} for _ in asked]})

    instance = server(three_wide)
    path = tmp_path / "g.ladybug"
    with GraphStore(path) as store:
        store.write(GRAPH)
        store.set_embedding("person:ada", [1.0, 0.0], model="gemma")

    lines: list[str] = []
    written = embed_store(str(path), base_url=instance.base_url, model="gemma",
                          log=lines.append)
    assert written == 0
    assert any("expected 2" in line for line in lines), lines

    with GraphStore(path, read_only=True) as reader:
        assert embedded(reader, model="gemma") == 1, "the seeded vector is untouched"


def test_embed_raises_on_a_dimension_mismatch(server):
    """`client.embed.embed` itself: a server answering the wrong width is refused, not padded."""
    from ml_stack.client.embed import EmbeddingError, embed

    instance = server(lambda method, path, body: json_reply(
        {"data": [{"embedding": [1.0, 2.0, 3.0]}]}))
    with pytest.raises(EmbeddingError):
        embed(["hello"], base_url=instance.base_url, expect_dim=2)


class TestAReadRunEmbedsWhatItRead:
    """A store nobody embedded searches by words alone, silently -- so a read run embeds."""

    def args(self, out, **over):
        import argparse

        said = {"out": str(out), "embed": True, "embed_url": "", "embed_model": "",
                "smooth": 0, **over}
        return argparse.Namespace(**said)

    def test_it_embeds_when_the_read_finishes(self, tmp_path, monkeypatch):
        from ml_stack.ingest import run

        asked = {}

        def note(out, *, base_url, model, smooth_hops, log):
            asked.update(out=str(out), base_url=base_url, model=model)
            return 7

        monkeypatch.setattr("ml_stack.ingest.run.embed_store", note)
        run._embedded(self.args(tmp_path / "s.ladybug"))
        assert asked["base_url"] == run.EMBED_URL
        assert asked["model"] == "embed"

    def test_no_embed_leaves_it_to_the_embed_command(self, tmp_path, monkeypatch):
        from ml_stack.ingest import run

        def never(*a, **k):
            raise AssertionError("--no-embed still embedded")

        monkeypatch.setattr("ml_stack.ingest.run.embed_store", never)
        run._embedded(self.args(tmp_path / "s.ladybug", embed=False))

    def test_an_embedder_that_cannot_be_reached_says_how_to_finish_later(
            self, tmp_path, monkeypatch, capsys):
        from ml_stack.ingest import run

        def refuse(*a, **k):
            raise ConnectionError("nothing is serving on 8081")

        monkeypatch.setattr("ml_stack.ingest.run.embed_store", refuse)
        run._embedded(self.args(tmp_path / "s.ladybug"))
        said = capsys.readouterr().err
        assert "read, not embedded" in said
        assert "ml-stack-ingest embed --out" in said

    def test_the_url_and_the_model_asked_for_win(self, tmp_path, monkeypatch):
        from ml_stack.ingest import run

        asked = {}
        monkeypatch.setattr("ml_stack.ingest.run.embed_store",
                            lambda out, **kw: asked.update(kw) or 1)
        run._embedded(self.args(tmp_path / "s.ladybug", embed_url="http://127.0.0.1:9",
                                embed_model="a-embedder"))
        assert asked["base_url"] == "http://127.0.0.1:9"
        assert asked["model"] == "a-embedder"


def test_embedding_is_on_unless_the_command_line_says_otherwise():
    from ml_stack.ingest.cli import parser

    assert parser().parse_args(["doc.pdf"]).embed is True
    assert parser().parse_args(["doc.pdf", "--no-embed"]).embed is False
