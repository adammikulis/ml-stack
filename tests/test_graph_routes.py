"""The routes behind the page's refresh button, review queue, request form and note drafter,
driven over a real socket. The graph is invented; nothing reads a data file."""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request

import pytest
from conftest import threaded_server

from ml_stack.files import read_json, write_json
from ml_stack.graph.review import Queue
from ml_stack.graph.serve import Handler

GRAPH = {
    "nodes": [{"id": "person:iris", "label": "Iris Bellweather", "kind": "person",
               "attrs": {}, "messages": []},
              {"id": "person:tobias", "label": "Tobias Renquist", "kind": "person",
               "attrs": {}, "messages": []},
              {"id": "topic:surveying", "label": "surveying", "kind": "topic",
               "attrs": {}, "messages": []}],
    "edges": [{"source": "person:iris", "rel": "experienced_in", "target": "topic:surveying"}],
    "messages": {},
}


def call(url, method="GET", body=None, headers=None):
    req = urllib.request.Request(
        url, method=method, data=json.dumps(body).encode() if body is not None else None,
        headers={"Content-Type": "application/json", **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            raw = r.read()
            return r.status, json.loads(raw) if raw else None
    except urllib.error.HTTPError as e:
        raw = e.read()
        return e.code, (json.loads(raw) if raw.startswith(b"{") or raw.startswith(b"[") else None)


def proposals():
    return {"2026-01-02T00:00:00Z|drop my topic": {
        "at": "2026-01-02T00:00:00Z", "kind": "Fix my information",
        "claimed": "person:iris", "claimedLabel": "Iris Bellweather", "attested": True,
        "concerns": [], "text": "drop my topic", "targets": [],
        "edits": [{"op": "remove_edge", "target": "person:iris", "other": "topic:surveying",
                   "name": "experienced_in", "reason": "asked"}],
        "status": "proposed"}}


class Quiet(Handler):
    def log_message(self, *a):
        pass


# ------------------------------------------------------------------------ refresh


def test_refresh_streams_each_stage_as_a_frame_and_runs_one_at_a_time(tmp_path):
    began = threading.Event()
    go = threading.Event()

    def stages(self):
        yield "scraping", "reading"
        began.set()
        go.wait(5)
        yield "done", "pipeline exit 0"

    handler = Quiet.configured(name="Refreshing", stages=stages)
    with threaded_server(handler) as url:
        got: list = []

        def first():
            with urllib.request.urlopen(url + "/refresh", timeout=10) as r:
                got.append((r.headers.get("Content-Type"), r.read().decode()))
        t = threading.Thread(target=first)
        t.start()
        assert began.wait(5)
        assert call(url + "/refresh")[0] == 409, "a second refresh while one runs"
        go.set()
        t.join(5)
        kind, raw = got[0]
        assert kind == "text/event-stream"
        events = [json.loads(f[len("data: "):]) for f in raw.split("\n\n") if f]
        assert [(e["stage"], e["detail"]) for e in events] == [("scraping", "reading"),
                                                              ("done", "pipeline exit 0")]
        assert all(abs(e["t"] - time.time()) < 60 for e in events)
        with urllib.request.urlopen(url + "/refresh", timeout=10) as r:
            assert r.status == 200, "and again once it is over"


def test_refresh_is_a_404_without_stages_and_a_403_through_a_proxy():
    with threaded_server(Quiet.configured(name="Bare")) as url:
        assert call(url + "/refresh")[0] == 404
    with threaded_server(Quiet.configured(name="Staged", stages=lambda self: iter([("done", "")]))) as url:
        for headers in ({"Cf-Ray": "8f3-XYZ"}, {"X-Forwarded-For": "203.0.113.9"},
                        {"cf-access-jwt-assertion": "eyJ"}):
            assert call(url + "/refresh", headers=headers)[0] == 403, headers


# ------------------------------------------------------------------------- review


def test_the_review_list_is_the_queue_and_acting_changes_what_is_listed(tmp_path):
    write_json(tmp_path / "proposals.json", proposals())
    queue = Queue(tmp_path / "proposals.json")
    with threaded_server(Quiet.configured(name="Reviewing", queue=queue)) as url:
        code, rows = call(url + "/review")
        assert code == 200
        assert [r["id"] for r in rows] == ["2026-01-02T00:00:00Z|drop my topic"]
        assert rows[0]["index"] == 1 and rows[0]["status"] == "proposed"

        code, out = call(url + "/review", "POST",
                         {"id": rows[0]["id"], "action": "refuse"})
        assert code == 200 and out == {"ok": True, "problems": []}
        assert call(url + "/review")[1][0]["status"] == "refused"
        assert read_json(tmp_path / "proposals.json", {})[rows[0]["id"]]["status"] == "refused"

        assert call(url + "/review", "POST", {"id": "nope", "action": "accept"})[0] == 404
        code, out = call(url + "/review", "POST", {"id": rows[0]["id"], "action": "shred"})
        assert code == 400 and "accept, refuse, undo" in out["error"]


def test_review_is_a_404_without_a_queue_and_a_403_through_a_proxy(tmp_path):
    with threaded_server(Quiet.configured(name="NoQueue")) as url:
        assert call(url + "/review")[0] == 404
        assert call(url + "/review", "POST", {"id": "x", "action": "accept"})[0] == 404
    write_json(tmp_path / "proposals.json", proposals())
    queue = Queue(tmp_path / "proposals.json")
    with threaded_server(Quiet.configured(name="Proxied", queue=queue)) as url:
        key = next(iter(proposals()))
        for headers in ({"Cf-Ray": "8f3-XYZ"}, {"CF-CONNECTING-IP": "203.0.113.9"}):
            assert call(url + "/review", headers=headers)[0] == 403
            assert call(url + "/review", "POST", {"id": key, "action": "accept"},
                        headers)[0] == 403
        assert read_json(tmp_path / "proposals.json", {})[key]["status"] == "proposed"


# ------------------------------------------------------------------------ request


def test_a_request_is_on_disk_before_the_204_and_proposed_for_after_it(tmp_path):
    seen: list = []
    order: list = []

    def proposed(self, row):
        order.append(("proposed", self.requests.read_text().count("\n")))
        seen.append(row)

    handler = Quiet.configured(name="Requesting", requests=tmp_path / "in" / "requests.jsonl",
                               proposed=proposed)
    with threaded_server(handler) as url:
        body = {"at": "2026-01-02T00:00:00Z", "kind": "Fix my information" * 20,
                "claimed": "person:iris", "claimedLabel": "Iris Bellweather", "attested": True,
                "text": "drop my topic", "targets": [{"key": "node:topic:surveying",
                                                      "label": "surveying"}]}
        code, _ = call(url + "/request", "POST", body)
        assert code == 204
        for _ in range(50):
            if seen:
                break
            time.sleep(0.05)
        rows = [json.loads(l) for l in (tmp_path / "in" / "requests.jsonl").read_text().splitlines()]
        assert len(rows) == 1
        assert rows[0]["text"] == "drop my topic" and rows[0]["attested"] is True
        assert rows[0]["kind"] == ("Fix my information" * 20)[:60], "cut to what a kind may be"
        assert rows[0]["targets"] == body["targets"]
        assert seen == rows
        assert order == [("proposed", 1)], "the row was on disk when proposed ran"

        assert call(url + "/request", "POST", {"text": "   "})[0] == 400
        assert call(url + "/request", "POST", {"nothing": True})[0] == 400
        assert len((tmp_path / "in" / "requests.jsonl").read_text().splitlines()) == 1


def test_request_is_a_404_without_a_file():
    with threaded_server(Quiet.configured(name="NoRequests")) as url:
        assert call(url + "/request", "POST", {"text": "hello"})[0] == 404


# -------------------------------------------------------------------------- draft


def test_draft_hands_the_ids_question_and_answer_to_the_drafter():
    asked: list = []

    def drafter(self, ids, question, answer):
        asked.append((ids, question, answer))
        return {"text": "Iris, meet Tobias.", "ids": ids}

    with threaded_server(Quiet.configured(name="Drafting", drafter=drafter)) as url:
        code, out = call(url + "/draft", "POST",
                         {"ids": ["person:iris", "person:tobias"], "question": "who surveys?",
                          "answer": "Iris does."})
        assert code == 200 and out == {"text": "Iris, meet Tobias.",
                                       "ids": ["person:iris", "person:tobias"]}
        assert asked == [(["person:iris", "person:tobias"], "who surveys?", "Iris does.")]
        code, out = call(url + "/draft", "POST", {"ids": "person:iris"})
        assert code == 400 and out == {"error": "no ids"}


def test_draft_is_a_404_without_a_drafter_and_a_500_when_it_raises(capsys):
    with threaded_server(Quiet.configured(name="NoDrafter")) as url:
        assert call(url + "/draft", "POST", {"ids": ["a", "b"]})[0] == 404

    def drafter(self, ids, question, answer):
        raise RuntimeError("the model is away")

    with threaded_server(Quiet.configured(name="Raising", drafter=drafter)) as url:
        code, out = call(url + "/draft", "POST", {"ids": ["a", "b"]})
        assert code == 500 and out == {"error": "the model is away"}
    assert "draft failed: the model is away" in capsys.readouterr().err


# --------------------------------------------------------------------------- page


def test_the_served_page_is_a_whole_document_with_the_live_sign_first(tmp_path):
    from ml_stack.graph.page import render

    page = tmp_path / "index.html"
    page.write_text(render(GRAPH, title="Invented"), encoding="utf-8")
    with threaded_server(Quiet.configured(name="Paged", site=page)) as url:
        with urllib.request.urlopen(url + "/", timeout=10) as r:
            body = r.read()
            assert r.headers.get("Cache-Control") == "no-store"
            assert int(r.headers.get("Content-Length")) == len(body)
    assert body.startswith(b"<!doctype html><html><head><meta charset='utf-8'>")
    assert body.endswith(b"</html>")
    assert body.index(b"GRAPH_LIVE") < body.index(b"<title>Invented</title>")
    assert b"customElements.define('graph-view'" in body


@pytest.mark.parametrize("path", ["/review", "/refresh"])
def test_the_bare_handler_still_404s_the_optional_routes(path):
    with threaded_server(Quiet.configured(name="Bare2")) as url:
        assert call(url + path)[0] == 404


# ------------------------------------------------------------------ the command


def _bound(argv):
    from ml_stack.graph.serve import bind

    httpd = bind(argv)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, f"http://127.0.0.1:{httpd.server_address[1]}"


def _listed(url):
    rows: list = []
    for _ in range(100):
        rows = call(url + "/review")[1]
        if rows:
            break
        time.sleep(0.05)
    return rows


def test_the_command_files_a_request_into_the_review_queue_beside_the_store(tmp_path):
    page = tmp_path / "page.html"
    page.write_text("<p></p>", encoding="utf-8")
    graph = tmp_path / "graph.json"
    write_json(graph, GRAPH)
    store = tmp_path / "g.ladybug"
    httpd, url = _bound(["serve", "--site", str(page), "--port", "0", "--graph", str(graph),
                         "--store", str(store)])
    try:
        assert call(url + "/review") == (200, [])
        body = {"at": "2026-01-02T00:00:00Z", "kind": "Fix my information",
                "claimed": "person:iris", "claimedLabel": "Iris Bellweather", "attested": True,
                "text": "drop my topic", "targets": []}
        assert call(url + "/request", "POST", body)[0] == 204
        rows = _listed(url)
        assert [r["text"] for r in rows] == ["drop my topic"]
        assert rows[0]["status"] == "proposed" and rows[0]["edits"] == []
        assert any("model" in c for c in rows[0]["concerns"]), rows[0]["concerns"]
        assert call(url + "/review", "POST", {"id": rows[0]["id"], "action": "refuse"})[0] == 200
    finally:
        httpd.shutdown()
        httpd.server_close()
    assert (tmp_path / "g.ladybug.requests.jsonl").is_file()
    assert read_json(tmp_path / "g.ladybug.review.json", {})[rows[0]["id"]]["status"] == "refused"
    assert not store.exists(), "nothing is written into the store"


def test_a_request_is_in_the_queue_while_the_model_is_still_reading_it(tmp_path):
    from ml_stack.graph.serve import READING

    reading = threading.Event()

    def client_on_slot(self, *, index=0, **over):
        reading.wait(10)
        raise RuntimeError("the model went away")

    handler = Quiet.configured(name="Reading", graph=GRAPH, client_on_slot=client_on_slot,
                               requests=tmp_path / "requests.jsonl",
                               queue=Queue(tmp_path / "review.json"))
    with threaded_server(handler) as url:
        assert call(url + "/request", "POST", {"text": "drop my topic"})[0] == 204
        rows = _listed(url)
        assert [r["text"] for r in rows] == ["drop my topic"] and READING in rows[0]["concerns"]
        reading.set()
        for _ in range(100):
            rows = call(url + "/review")[1]
            if READING not in rows[0]["concerns"]:
                break
            time.sleep(0.05)
        assert "not read by a model: the model went away" in rows[0]["concerns"]
        assert len(rows) == 1


def test_the_command_puts_requests_and_the_queue_where_it_is_told(tmp_path):
    page = tmp_path / "page.html"
    page.write_text("<p></p>", encoding="utf-8")
    httpd, url = _bound(["serve", "--site", str(page), "--port", "0",
                         "--requests", str(tmp_path / "r" / "in.jsonl"),
                         "--review", str(tmp_path / "q" / "queue.json")])
    try:
        assert call(url + "/request", "POST", {"text": "add my town"})[0] == 204
        assert [r["text"] for r in _listed(url)] == ["add my town"]
    finally:
        httpd.shutdown()
        httpd.server_close()
    assert (tmp_path / "r" / "in.jsonl").is_file()
    assert [p["text"] for p in read_json(tmp_path / "q" / "queue.json", {}).values()] \
        == ["add my town"]
    httpd, _ = _bound(["serve", "--site", str(page), "--port", "0"])
    try:
        handler = httpd.RequestHandlerClass
        assert handler.requests == tmp_path / "page.html.requests.jsonl"
        assert handler.queue.path == tmp_path / "page.html.review.json"
    finally:
        httpd.shutdown()
        httpd.server_close()
