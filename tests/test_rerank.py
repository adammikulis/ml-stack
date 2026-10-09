"""Serving a reranker, asking it, and putting its order into a search."""

from __future__ import annotations

import math
import re

import pytest

from poolhouse.bench.ranking import ndcg_at, recall_at, reciprocal_rank
from poolhouse.bench.retrieval import MODES, Finder, compare, node_texts, report, table
from poolhouse.client import rerank as rerank_client
from poolhouse.client.rerank import RerankError, rerank
from poolhouse.graph.search import RERANK, hybrid, reranked_by
from poolhouse.http import ServerError
from poolhouse.serve import admission
from poolhouse.serve.backend import LlamaServerBackend, ServerSpec
from poolhouse.serve.broker import Ask, Held
from poolhouse.serve.lifecycle_cli import _asked_spec
from poolhouse.testing.fakes import fake_llama_binary, fake_llama_server


def test_a_reranking_spec_starts_llama_server_with_the_reranking_flag(tmp_path):
    backend = LlamaServerBackend(binary=fake_llama_binary(tmp_path))
    argv = backend.command(ServerSpec(model="r.gguf", reranking=True))
    assert "--reranking" in argv
    assert "--jinja" not in argv and "--embeddings" not in argv
    plain = backend.command(ServerSpec(model="r.gguf"))
    assert "--reranking" not in plain and "--jinja" in plain


def test_the_serve_command_line_carries_reranking_into_the_spec():
    import argparse

    args = argparse.Namespace(port=8080, context=4096, parallel=1, reranking=True)
    assert _asked_spec(args, "r.gguf", ()).reranking is True


def test_an_ask_may_name_reranking_and_the_broker_knows_it_as_a_setting():
    ask = Ask.from_json({"purpose": "rerank", "models": ["r.gguf"],
                         "spec": {"reranking": True}})
    assert ask.server_spec(8080).reranking is True


@pytest.mark.parametrize("have,asked", [({"reranking": True}, {"reranking": False}),
                                        ({"reranking": False}, {"reranking": True}),
                                        ({"embedding": True, "reranking": False},
                                         {"embedding": False, "reranking": True})])
def test_a_reranking_server_never_shares_a_lease_with_chat_or_embedding(have, asked):
    held = Held(port=7, model="m", shape={"device": "gpu", "context": 512, **have})
    assert "reranking=" in held.short_of({"context": 512, **asked}) \
        or "embedding=" in held.short_of({"context": 512, **asked})
    assert held.short_of({"context": 512, **have}) == ""


def test_admission_does_not_reuse_a_reranker_for_chat_nor_chat_for_a_reranker():
    chat, rank = ServerSpec(model="m.gguf"), ServerSpec(model="m.gguf", reranking=True)
    assert admission.compatible(chat, {}, [])
    assert not admission.compatible(chat, {"reranking": True}, [])
    assert not admission.compatible(rank, {"reranking": False}, [])
    assert admission.compatible(rank, {"reranking": True}, [])
    assert not admission.pending_serves(chat, {"context": 4096, "reranking": True})
    assert admission.pending_serves(rank, {"context": 4096, "reranking": True})


def test_rerank_round_trips_through_a_real_http_server_and_scores_by_index():
    docs = ["pasta and olives", "robots fix machines", "machines and robots here"]
    with fake_llama_server() as fake:
        scores = rerank("robots fix machines", docs, base_url=fake.base_url)
        cut = rerank("robots fix machines", docs, base_url=fake.base_url, top_n=1)
        sent = [req for req in fake.requests if req[1] == "/v1/rerank"]
    assert scores == [0.0, 1.0, pytest.approx(2 / 3)]
    assert cut[1] == 1.0 and cut[0] == float("-inf") and cut[2] == float("-inf")
    assert b'"documents"' in sent[0][2] and b'"query"' in sent[0][2]
    assert b'"top_n": 1' in sent[1][2]


def test_rerank_of_nothing_asks_nothing_and_a_bad_answer_raises():
    with fake_llama_server() as fake:
        assert rerank("q", [], base_url=fake.base_url) == []
        assert not fake.requests
    with fake_llama_server() as fake, pytest.raises(ServerError, match="404"):
        rerank("q", ["a"], base_url=fake.base_url + "/nowhere")


@pytest.mark.parametrize("answer", [{"x": 1}, {"results": [{"index": 9, "relevance_score": 1}]},
                                    {"results": [{"index": 0}]}])
def test_an_answer_of_the_wrong_shape_is_a_rerank_error(answer):
    with hostile(says(json.dumps(answer).encode())) as url, pytest.raises(RerankError):
        rerank("q", ["a"], base_url=url)


ROWS = [{"id": c, "label": c, "kind": "k"} for c in "abcdefgh"]


def test_reranked_by_reorders_only_the_window_and_keeps_membership():
    scores = {"a": 0.1, "b": 0.9, "c": 0.5, "d": 0.5, "e": 0.0, "f": 0.7}
    out = reranked_by(ROWS, "q", lambda q, rows: [scores[r["id"]] for r in rows])
    assert [r["id"] for r in out] == ["b", "f", "c", "d", "a", "e", "g", "h"]
    assert sorted(r["id"] for r in out) == sorted(r["id"] for r in ROWS)
    assert [r["id"] for r in out[RERANK:]] == ["g", "h"]


def test_reranked_by_hands_the_reranker_the_question_and_exactly_the_window():
    seen = []
    reranked_by(ROWS, "who?", lambda q, rows: seen.append((q, [r["id"] for r in rows])) or
                [0.0] * len(rows))
    assert seen == [("who?", list("abcdef"))]


def test_a_reranker_that_scores_the_wrong_number_of_hits_raises():
    with pytest.raises(ValueError, match="scored 1 of 6"):
        reranked_by(ROWS, "q", lambda q, rows: [1.0])


GRAPH = {
    "nodes": [{"id": f"n{i}", "kind": "topic", "label": f"robots {i}", "mentions": 1,
               "attrs": {}, "messages": []} for i in range(8)],
    "messages": {}, "edges": [],
}


def test_hybrid_without_a_reranker_is_unchanged_and_with_one_is_reordered():
    base = [r["id"] for r in hybrid(GRAPH, "robots", rerank=False)]
    assert [r["id"] for r in hybrid(GRAPH, "robots", rerank=True)] == base
    flipped = hybrid(GRAPH, "robots", rerank=lambda q, rows: list(range(len(rows))))
    ids = [r["id"] for r in flipped]
    assert ids[:RERANK] == list(reversed(base[:RERANK])) and ids[RERANK:] == base[RERANK:]


def test_the_metrics_match_hand_computed_cases():
    ranked_ids, expect = ["x", "a", "y", "b"], ["a", "b", "z"]
    assert recall_at(ranked_ids, expect, 2) == pytest.approx(1 / 3)
    assert recall_at(ranked_ids, expect, 4) == pytest.approx(2 / 3)
    assert reciprocal_rank(ranked_ids, expect) == 0.5
    assert reciprocal_rank(["q"], expect) == 0.0
    dcg = 1 / math.log2(3) + 1 / math.log2(5)
    ideal = 1 + 1 / math.log2(3) + 1 / math.log2(4)
    assert ndcg_at(ranked_ids, expect, 4) == pytest.approx(dcg / ideal)
    assert ndcg_at(["a"], ["a"], 3) == 1.0
    with pytest.raises(ValueError):
        recall_at(["a"], [], 3)


def test_the_comparison_scores_three_modes_and_leaves_out_questions_expecting_nobody():
    questions = [{"q": "robots 3", "expect": ["n3"]}, {"q": "robots", "expect": ["n7", "n0"]},
                 {"q": "who is nobody", "expect": []}]
    prefer_last = lambda q, rows: [float(i) for i in range(len(rows))]  # noqa: E731
    finder = Finder(GRAPH, reranker=prefer_last)
    results = compare(questions, finder=finder)
    assert list(results) == list(MODES)
    assert results["fused"]["mrr"] == results["vectors"]["mrr"]   # no store: no vector order
    assert results["reranker"]["mrr"] != results["fused"]["mrr"]
    text = table(results, asked=2, nobody=1)
    assert "2 questions scored; 1 expect nobody" in text
    assert re.search(r"^reranker\s+\d", text, re.M)
    assert finder.ranked("robots 3", "fused")[0] == "n3"
    with pytest.raises(ValueError):
        Finder(GRAPH).ranked("robots 3", "reranker")
    with pytest.raises(ValueError):
        finder.ranked("robots 3", "bogus")


def test_node_texts_carry_the_label_and_what_was_said():
    graph = {"nodes": [{"id": "a", "label": "Ada", "messages": ["m"]}],
             "messages": {"m": {"text": "I fix machines."}}}
    assert node_texts(graph) == {"a": "Ada I fix machines."}


def test_the_command_runs_all_three_modes_against_a_served_reranker():
    import argparse

    with fake_llama_server() as fake:
        args = argparse.Namespace(store="", embed_url="", embed_model="", rerank_url=fake.base_url,
                                  rerank_model="", k=6)
        text = report(args)
    assert [line.split()[0] for line in text.splitlines()[1:4]] == list(MODES)
    assert "110 expect nobody" not in text and "expect nobody" in text
    assert not [r for r in fake.requests if r[1] == "/v1/embeddings"]
    args.rerank_url = ""
    assert "reranker" not in report(args)


def test_a_model_reranker_over_http_orders_the_hits_by_the_servers_scores():
    from poolhouse.bench.retrieval import model_reranker

    graph = {**GRAPH, "nodes": [{**n, "label": f"robots {'fix machines' if i == 5 else i}"}
                                for i, n in enumerate(GRAPH["nodes"])]}
    with fake_llama_server() as fake:
        rows = hybrid(graph, "robots", rerank=model_reranker(graph, fake.base_url))
        sent = [r for r in fake.requests if r[1] == "/v1/rerank"]
    assert len(sent) == 1 and b"robots fix machines" in sent[0][2]
    assert len(rows) == len(graph["nodes"])


# ---- a hostile reranking server -----------------------------------------------------------

import json  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
from contextlib import contextmanager  # noqa: E402
from http.server import BaseHTTPRequestHandler  # noqa: E402

from poolhouse.http import Server  # noqa: E402


@contextmanager
def hostile(reply):
    """A real socket server whose /v1/rerank answers with ``reply(handler)``."""
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", "0")))
            reply(self)

        def log_message(self, *_args):
            pass

    server = Server(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        worker.join(2)


def says(raw: bytes):
    def reply(handler):
        handler.send_response(200)
        handler.send_header("Content-Type", "application/json")
        handler.send_header("Content-Length", str(len(raw)))
        handler.end_headers()
        handler.wfile.write(raw)
    return reply


def results(*rows):
    return json.dumps({"results": list(rows)}).encode()


ROW = {"index": 0, "relevance_score": 0.5}
HOSTILE = [
    pytest.param(b"not json at all", id="non-json"),
    pytest.param(b"\xff\xfe\x00", id="not-utf8"),
    pytest.param(b"", id="empty"),
    pytest.param(b"[1, 2]", id="array"),
    pytest.param(b"null", id="null"),
    pytest.param(b'"results"', id="string"),
    pytest.param(b"{}", id="no-results"),
    pytest.param(b'{"results": {"index": 0}}', id="results-a-dict"),
    pytest.param(b'{"results": "abc"}', id="results-a-string"),
    pytest.param(results(1), id="entry-int"),
    pytest.param(results("x"), id="entry-string"),
    pytest.param(results(None), id="entry-null"),
    pytest.param(results([0, 1]), id="entry-list"),
    pytest.param(results({"index": True, "relevance_score": 1.0}), id="index-bool"),
    pytest.param(results({"index": -1, "relevance_score": 1.0}), id="index-negative"),
    pytest.param(results({"index": 1, "relevance_score": 1.0}), id="index-out-of-range"),
    pytest.param(results({"index": 0.0, "relevance_score": 1.0}), id="index-float"),
    pytest.param(results({"index": "0", "relevance_score": 1.0}), id="index-string"),
    pytest.param(results({"index": None, "relevance_score": 1.0}), id="index-null"),
    pytest.param(results({"relevance_score": 1.0}), id="index-missing"),
    pytest.param(results({"index": 0, "relevance_score": None}), id="score-null"),
    pytest.param(results({"index": 0, "relevance_score": "high"}), id="score-string"),
    pytest.param(results({"index": 0, "relevance_score": "1.5"}), id="score-numeric-string"),
    pytest.param(results({"index": 0, "relevance_score": True}), id="score-bool"),
    pytest.param(results({"index": 0, "relevance_score": [1]}), id="score-list"),
    pytest.param(b'{"results": [{"index": 0, "relevance_score": NaN}]}', id="score-nan"),
    pytest.param(b'{"results": [{"index": 0, "relevance_score": Infinity}]}', id="score-inf"),
    pytest.param(b'{"results": [{"index": 0, "relevance_score": -Infinity}]}', id="score-neg-inf"),
    pytest.param(b'{"results": [{"index": 0, "relevance_score": 1e999}]}', id="score-overflow"),
    pytest.param(b"[" * 100000, id="deeply-nested"),
]


@pytest.mark.parametrize("raw", HOSTILE)
def test_a_hostile_reranker_answer_raises_rerank_error(raw):
    with hostile(says(raw)) as url, pytest.raises(RerankError):
        rerank("q", ["a"], base_url=url)


def test_an_oversized_answer_is_refused_without_reading_it_all(monkeypatch):
    monkeypatch.setattr(rerank_client, "TIMEOUT", 10.0)
    sent = []

    def flood(handler):
        handler.send_response(200)
        handler.send_header("Content-Type", "application/json")
        handler.end_headers()
        try:
            for _ in range(4096):
                handler.wfile.write(b" " * 65536)
                sent.append(1)
        except OSError:
            pass

    started = time.monotonic()
    with hostile(flood) as url, pytest.raises(RerankError, match="more than"):
        rerank("q", ["a"], base_url=url)
    assert time.monotonic() - started < 8
    assert len(sent) < 4096


def test_a_server_that_never_answers_hits_the_timeout(monkeypatch):
    monkeypatch.setattr(rerank_client, "TIMEOUT", 0.5)
    release = threading.Event()
    started = time.monotonic()
    with hostile(lambda handler: release.wait(5)) as url:
        try:
            with pytest.raises(RerankError):
                rerank("q", ["a"], base_url=url)
            assert time.monotonic() - started < 3
        finally:
            release.set()


def test_a_server_that_drips_its_answer_cannot_outlast_the_deadline(monkeypatch):
    monkeypatch.setattr(rerank_client, "TIMEOUT", 1.0)
    def drip(handler):
        handler.send_response(200)
        handler.send_header("Content-Type", "application/json")
        handler.end_headers()
        try:
            for _ in range(50):
                handler.wfile.write(b" ")
                handler.wfile.flush()
                time.sleep(0.1)
        except OSError:
            pass

    started = time.monotonic()
    with hostile(drip) as url:
        with pytest.raises(RerankError, match="did not finish"):
            rerank("q", ["a"], base_url=url)
        elapsed = time.monotonic() - started
    assert elapsed < 3


def test_a_server_error_status_is_a_rerank_error_that_keeps_the_status():
    def broken(handler):
        handler.send_response(500)
        handler.send_header("Content-Length", "0")
        handler.end_headers()

    with hostile(broken) as url, pytest.raises(RerankError) as caught:
        rerank("q", ["a"], base_url=url)
    assert caught.value.status == 500


def test_a_clean_answer_in_any_order_still_scores_by_index():
    rows = results({"index": 1, "relevance_score": 2}, {"index": 0, "relevance_score": -0.5})
    with hostile(says(rows)) as url:
        assert rerank("q", ["a", "b"], base_url=url) == [-0.5, 2.0]
