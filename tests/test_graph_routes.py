"""The routes behind the page's refresh button, review queue, request form and note drafter,
driven over a real socket. The graph is invented; nothing reads a data file."""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request

import pytest

from ml_stack.graph.review import Queue
from ml_stack.graph.serve import Handler
from ml_stack.files import read_json, write_json

from conftest import threaded_server

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
