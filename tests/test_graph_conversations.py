"""Where a served page keeps its conversations: a store beside the corpus, never the corpus.

Real stores in a scratch directory, a real ``ThreadingHTTPServer``, a scripted asker and no
model; every store is read back on a fresh handle after the server has written it.
"""

from __future__ import annotations

import json
import urllib.request

import pytest

from ml_stack.graph.answers import Answer
from ml_stack.graph.serve import Handler, bind
from ml_stack.graph.store import GraphStore
from ml_stack.graph.thread import conversation_store, follow

from conftest import threaded_server

GRAPH = {
    "nodes": [{"id": "person:iris", "label": "Iris Bellweather", "kind": "person"},
              {"id": "topic:surveying", "label": "surveying", "kind": "topic"},
              {"id": "person:otto", "label": "Otto Fenwick", "kind": "person"}],
    "edges": [],
}

ANSWER = Answer(content="Iris surveys land.", ids=["person:iris", "topic:surveying"],
                found=["topic:surveying", "person:iris"], read=["person:iris"], path=[],
                show=["person:iris", "person:ghost"], steps=["found 2 entries"])


def _asker(self, question, *, turns, highlighted, stream, emit):
    return ANSWER


def _post(url, body):
    req = urllib.request.Request(url, method="POST", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as r:
        return r.status, r.read()


def _get(url):
    with urllib.request.urlopen(url, timeout=10) as r:
        return json.loads(r.read())


def _tables(path):
    with GraphStore(path, read_only=True) as store:
        return {row["name"] for row in store.query("CALL SHOW_TABLES() RETURN *")}


@pytest.fixture
def corpus(tmp_path):
    path = tmp_path / "corpus.ladybug"
    with GraphStore(path) as store:
        for node in GRAPH["nodes"]:
            store.upsert_node(node)
    return path


def _served(corpus, **more):
    return Handler.configured(graph=GRAPH, store=corpus, asker=_asker, **more)


def test_conversation_store_sits_beside_the_corpus(tmp_path):
    assert conversation_store(tmp_path / "g.ladybug") == tmp_path / "g.ladybug.conversations"


def test_an_ask_leaves_the_corpus_untouched_and_the_conversation_beside_it(corpus):
    with threaded_server(_served(corpus)) as url:
        status, _ = _post(url + "/ask/stream", {"question": "who surveys land?",
                                                 "thread": "c1"})
        assert status == 200
        replay = _get(url + "/thread/c1?working=1")

    assert "Turn" not in _tables(corpus)
    with GraphStore(corpus, read_only=True) as store:
        assert sorted(n["id"] for n in store.nodes()) == sorted(n["id"] for n in GRAPH["nodes"])

    kept = conversation_store(corpus)
    assert kept.exists()
    with GraphStore(kept, read_only=True) as store:
        turns = follow(store, "c1", working=True)
        pointers = {n["id"]: (n["kind"], n["label"]) for n in store.nodes()}
    assert [(t.role, t.text) for t in turns] == [("user", "who surveys land?"),
                                                  ("assistant", "Iris surveys land.")]
    assert turns[1].drew == {"found": ["person:iris", "topic:surveying"],
                             "read": ["person:iris"], "shown": ["person:iris"]}
    assert pointers == {"person:iris": ("person", "Iris Bellweather"),
                        "topic:surveying": ("topic", "surveying")}
    assert [t["text"] for t in replay["turns"]] == ["who surveys land?", "Iris surveys land."]


def test_the_flag_keeps_the_conversation_in_the_corpus(corpus):
    with threaded_server(_served(corpus, conversations_in_store=True)) as url:
        assert _post(url + "/ask", {"question": "who surveys land?", "thread": "c1"})[0] == 200

    assert not conversation_store(corpus).exists()
    with GraphStore(corpus, read_only=True) as store:
        turns = follow(store, "c1", working=True)
    assert [t.role for t in turns] == ["user", "assistant"]
    assert turns[1].drew["shown"] == ["person:iris"]


def test_a_pointer_never_overwrites_an_entry_the_conversation_store_holds(corpus):
    with GraphStore(corpus) as store:
        store.upsert_node({"id": "person:iris", "kind": "person", "label": "Iris Bellweather",
                           "attrs": {"role": "surveyor"}})

    def threads(self, *, write=False):
        return GraphStore(corpus, read_only=not write)

    with threaded_server(_served(corpus, threads=threads)) as url:
        assert _post(url + "/ask", {"question": "who surveys land?", "thread": "c1"})[0] == 200

    with GraphStore(corpus, read_only=True) as store:
        iris = next(n for n in store.nodes() if n["id"] == "person:iris")
        assert iris["attrs"] == {"role": "surveyor"}
        assert [t.role for t in follow(store, "c1")] == ["user", "assistant"]


def test_reading_a_thread_before_any_ask_creates_no_store(corpus):
    with threaded_server(_served(corpus)) as url:
        assert _get(url + "/thread/c1") == {"thread": "c1", "turns": []}
    assert not conversation_store(corpus).exists()


def test_the_command_line_flag_reaches_the_handler(tmp_path):
    page = tmp_path / "page.html"
    page.write_text("<p></p>", encoding="utf-8")
    for extra, inside in (([], False), (["--conversations-in-store"], True)):
        httpd = bind(["serve", "--site", str(page), "--port", "0",
                      "--store", str(tmp_path / "g.ladybug"), *extra])
        try:
            handler = httpd.RequestHandlerClass
            assert handler.conversations_in_store is inside
            assert handler.store == tmp_path / "g.ladybug"
        finally:
            httpd.server_close()
