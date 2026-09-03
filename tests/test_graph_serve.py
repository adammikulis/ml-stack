"""The page's three routes, driven over a real socket with a scripted asker and no model.

Every fixture is invented; nothing reads a real graph. The server is a real
``ThreadingHTTPServer`` on a free port, because the bugs these routes exist to prevent are
transport-shaped: a frame that was buffered rather than flushed, a ``done`` that arrived
without the answer on it, a 409 sent after the stream headers.
"""

from __future__ import annotations

import json
from pathlib import Path
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from ml_stack.graph.ask import Answer
from ml_stack.graph.serve import Ask, AskRoutes, History, answer_payload, sse, thread_request
from ml_stack.graph.store import GraphStore

from conftest import threaded_server
from ml_stack.graph.thread import SUMMARY, WINDOW, follow

GRAPH = {
    "nodes": [{"id": "person:iris", "label": "Iris Bellweather", "kind": "person",
               "attrs": {}, "messages": []},
              {"id": "topic:surveying", "label": "surveying", "kind": "topic",
               "attrs": {}, "messages": []}],
    "edges": [{"source": "person:iris", "rel": "experienced_in", "target": "topic:surveying"}],
    "messages": {},
}

EVENTS = [{"event": "tool", "name": "look_up", "detail": "'surveying'"},
          {"event": "tool_result", "name": "look_up", "count": 2},
          {"event": "answer", "text": "Iris surveys land."}]

ANSWER = Answer(content="Iris surveys land.", ids=["person:iris", "topic:surveying"],
                found=["topic:surveying", "person:iris"], read=["person:iris"], path=[],
                show=["person:iris"], steps=["found 2 entries", "read 1 entry"])


class Scripted(AskRoutes, BaseHTTPRequestHandler):
    """A handler that answers every question the same way and keeps what it was asked."""

    store_path = None
    asked: list = []
    answer = ANSWER
    reason = None
    raise_with = None

    def log_message(self, *a):
        pass

    def asker(self, question, *, turns, held, stream, emit):
        type(self).asked.append({"question": question, "turns": list(turns), "held": held,
                                 "stream": stream, "summary": getattr(turns, "summary", None),
                                 "recalled": list(getattr(turns, "recalled", ()))})
        if self.raise_with:
            raise self.raise_with
        if stream:
            for event in EVENTS:
                emit(event)
            emit({"event": "done"})       # converse_stream sends its own; the route drops it
        return self.answer

    def threads(self, *, write=False):
        if self.store_path is None:
            return None
        return GraphStore(self.store_path, read_only=not write)

    def ready(self):
        return self.reason

    model = ""

    def model_name(self):
        return self.model

    def do_GET(self):
        if self.path == "/ask/model":
            self.handle_model()
            return
        if self.path == "/metrics":
            self.handle_metrics()
            return
        if self.path == "/metrics.prom":
            self.handle_metrics_prom()
            return
        want = thread_request(self.path)
        if want:
            self.handle_thread(*want)
            return
        self.send_response(404)
        self.end_headers()

    def do_POST(self):
        body = self.read_body() or {}
        if self.path == "/ask":
            self.handle_ask(body)
        elif self.path == "/ask/stream":
            self.handle_ask_stream(body)
        else:
            self.send_response(404)
            self.end_headers()


@pytest.fixture
def served(tmp_path):
    """A live server with a conversation store beside it; yields (url, handler class)."""
    with GraphStore(tmp_path / "graph.ladybug") as held:
        held.write(GRAPH)

    class Handler(Scripted):
        store_path = tmp_path / "graph.ladybug"
        asked = []

    with threaded_server(Handler) as url:
        yield url, Handler


def call(url, method="GET", body=None):
    req = urllib.request.Request(
        url, method=method, data=json.dumps(body).encode() if body is not None else None,
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            raw = r.read()
            return r.status, r.headers.get("Content-Type"), json.loads(raw) if raw else None
    except urllib.error.HTTPError as e:
        return e.code, e.headers.get("Content-Type"), json.loads(e.read() or b"null")


def stream(url, body):
    req = urllib.request.Request(url, method="POST", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as r:
        return r.headers.get("Content-Type"), r.read().decode()


def frames(raw: str) -> list[dict]:
    """Parse SSE text strictly: every frame is one data line, then a blank line."""
    out = []
    for frame in raw.split("\n\n"):
        if not frame:
            continue
        assert frame.startswith("data: ") and "\n" not in frame, frame
        out.append(json.loads(frame[len("data: "):]))
    return out


# --------------------------------------------------------------------- the stream


def test_the_stream_relays_each_event_then_a_done_frame_carrying_the_whole_answer(served):
    url, handler = served
    kind, raw = stream(url + "/ask/stream", {"question": "who surveys land?"})
    assert kind == "text/event-stream"
    assert raw.endswith("\n\n")
    events = frames(raw)
    assert [e["event"] for e in events] == ["tool", "tool_result", "answer", "done"]
    assert events[:3] == EVENTS
    done = events[-1]
    # the closing frame is the plain route's answer with an `event` on it
    assert done == {"event": "done", **answer_payload(ANSWER)}
    assert done["content"] == "Iris surveys land."
    assert done["read"] == ["person:iris"] and done["show"] == ["person:iris"]
    assert done["steps"] == ["found 2 entries", "read 1 entry"]
    assert done["why"] == "found 2 entries; read 1 entry"
    assert handler.asked[-1]["stream"] is True


def test_the_plain_route_returns_the_same_payload_in_one_body(served):
    url, handler = served
    code, kind, out = call(url + "/ask", "POST", {"question": "who surveys land?",
                                                   "held": ["topic:surveying"]})
    assert code == 200 and kind == "application/json"
    assert out == answer_payload(ANSWER)
    assert handler.asked[-1] == {"question": "who surveys land?", "turns": [],
                                 "held": ["topic:surveying"], "stream": False,
                                 "summary": None, "recalled": []}


def test_a_generator_asker_yields_its_events_and_returns_its_answer(served):
    url, handler = served

    def asker(self, question, *, turns, held, stream, emit):
        yield EVENTS[0]
        yield {"event": "done"}
        return {"content": "Iris does.", "ids": ["person:iris"], "steps": ["one step"]}

    handler.asker = asker
    kind, raw = stream(url + "/ask/stream", {"question": "who?"})
    events = frames(raw)
    assert [e["event"] for e in events] == ["tool", "done"]
    assert events[-1]["content"] == "Iris does." and events[-1]["why"] == "one step"
    # and the plain route drains the same generator without a listener
    code, _, out = call(url + "/ask", "POST", {"question": "who?"})
    assert code == 200 and out["content"] == "Iris does."


# ---------------------------------------------------------------- refusals


def test_an_empty_question_is_a_400_on_both_routes(served):
    url, handler = served
    for route in ("/ask", "/ask/stream"):
        code, kind, out = call(url + route, "POST", {"question": "   "})
        assert code == 400 and kind == "application/json", route
        assert out == {"error": "no question"}
    assert handler.asked == []


def test_a_server_that_is_not_ready_says_so_as_a_409_before_any_stream_starts(served):
    url, handler = served
    handler.reason = "no graph yet"
    for route in ("/ask", "/ask/stream"):
        code, kind, out = call(url + route, "POST", {"question": "who?"})
        assert code == 409 and kind == "application/json", route
        assert out == {"error": "no graph yet"}
    assert handler.asked == []


def test_an_asker_that_raises_is_a_500_in_one_body_and_an_error_frame_in_a_stream(served, capsys):
    url, handler = served
    handler.raise_with = RuntimeError("the model went away")
    code, _, out = call(url + "/ask", "POST", {"question": "who?"})
    assert code == 500 and out == {"error": "the model went away"}
    kind, raw = stream(url + "/ask/stream", {"question": "who?"})
    assert kind == "text/event-stream"
    assert frames(raw) == [{"event": "error", "error": "the model went away"}]
    assert "the model went away" in capsys.readouterr().err


# ---------------------------------------------------------- conversations


def test_a_thread_is_remembered_and_read_back_with_each_answers_steps(served):
    """The page reads ``meta.steps`` off each assistant turn to get its trace tally back."""
    url, handler = served
    code, _, out = call(url + "/ask", "POST", {"question": "who surveys land?", "thread": "c1"})
    assert code == 200

    code, _, replay = call(url + "/thread/c1")
    assert code == 200 and replay["thread"] == "c1"
    user, answer = replay["turns"]
    assert user["role"] == "user" and user["text"] == "who surveys land?" and user["meta"] == {}
    assert answer["role"] == "assistant" and answer["text"] == "Iris surveys land."
    assert answer["meta"]["steps"] == out["steps"] == ["found 2 entries", "read 1 entry"]
    assert answer["meta"]["why"] == "; ".join(out["steps"])
    assert [t["seq"] for t in replay["turns"]] == [1, 2]
    # without `working` the turns come back as plain words; with it, what each drew on
    assert answer["drew"] == {}
    _, _, working = call(url + "/thread/c1?working=1")
    assert working["turns"][1]["drew"] == {"found": ["topic:surveying", "person:iris"],
                                           "read": ["person:iris"], "shown": ["person:iris"]}
    with GraphStore(handler.store_path, read_only=True) as held:
        assert [t.role for t in follow(held, "c1")] == ["user", "assistant"]


def test_the_stores_history_goes_back_to_the_model_rather_than_the_pages(served):
    """A reopened page sends nothing; a stale tab sends the wrong thing. The store wins."""
    url, handler = served
    call(url + "/ask", "POST", {"question": "who surveys land?", "thread": "c1"})
    stale = [{"role": "user", "content": "something else entirely"}]
    call(url + "/ask", "POST", {"question": "and who else?", "thread": "c1", "turns": stale})
    assert handler.asked[-1]["turns"] == [
        {"role": "user", "content": "who surveys land?"},
        {"role": "assistant", "content": "Iris surveys land."}]
    # a thread the store has never heard of falls through to what the page sent
    call(url + "/ask", "POST", {"question": "hm?", "thread": "c-new", "turns": stale})
    assert handler.asked[-1]["turns"] == stale
    # and no thread at all is the page's own list, with nothing written
    call(url + "/ask", "POST", {"question": "hm?", "turns": stale})
    assert handler.asked[-1]["turns"] == stale
    _, _, replay = call(url + "/thread/c1")
    assert [t["seq"] for t in replay["turns"]] == [1, 2, 3, 4]


def test_a_handler_with_no_conversation_store_still_answers_and_has_no_history(tmp_path):
    class Handler(Scripted):
        asked = []

    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{srv.server_address[1]}"
        code, _, out = call(url + "/ask", "POST", {"question": "who?", "thread": "c1",
                                                   "turns": [{"role": "user", "content": "hi"}]})
        assert code == 200 and out["content"] == "Iris surveys land."
        assert Handler.asked[-1]["turns"] == [{"role": "user", "content": "hi"}]
        code, _, replay = call(url + "/thread/c1")
        assert code == 200 and replay == {"thread": "c1", "turns": []}
    finally:
        srv.shutdown()


def test_a_store_that_cannot_be_opened_is_an_empty_thread_with_a_note_not_an_error(tmp_path):
    class Handler(Scripted):
        asked = []

        def threads(self, *, write=False):
            raise OSError("another process holds the writer")

    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{srv.server_address[1]}"
        code, _, replay = call(url + "/thread/c1")
        assert code == 200 and replay["turns"] == []
        assert replay["note"].startswith("another process")
        code, _, out = call(url + "/ask", "POST", {"question": "who?", "thread": "c1"})
        assert code == 200 and out["content"] == "Iris surveys land."
    finally:
        srv.shutdown()


# ---------------------------------------------------------------- the pieces


def test_answer_payload_keeps_a_mappings_keys_and_fills_why_from_steps():
    got = answer_payload({"content": "hi", "ids": [], "steps": ["a", "b"], "raised": [{"id": "x"}]})
    assert got["why"] == "a; b" and got["raised"] == [{"id": "x"}]
    assert answer_payload({"content": "hi", "why": "a; b"})["steps"] == ["a", "b"]
    assert answer_payload(Answer(content="Nobody."), seat=1) == {
        "content": "Nobody.", "ids": [], "found": [], "read": [], "path": [], "show": [],
        "why": "", "steps": [], "seat": 1}


def test_sse_writes_one_flushed_frame_per_event():
    class Sink:
        def __init__(self):
            self.chunks, self.flushed = [], 0

        def write(self, b):
            self.chunks.append(b)

        def flush(self):
            self.flushed += 1

    sink = Sink()
    sse(sink, {"event": "answer", "text": "Iris — ünïcode"})
    assert sink.chunks == ['data: {"event": "answer", "text": "Iris — ünïcode"}\n\n'.encode()]
    assert sink.flushed == 1


def test_thread_request_reads_the_name_and_the_working_flag():
    assert thread_request("/thread/c1") == ("c1", False)
    assert thread_request("/thread/c1?working=1") == ("c1", True)
    assert thread_request("/thread/" + "x" * 90)[0] == "x" * 64
    assert thread_request("/review") is None


def test_an_ask_checks_the_body_the_way_the_page_sends_it():
    ask = Ask({"question": "  who?  ", "thread": "t" * 80, "held": ["a", 3], "turns": "no"})
    assert ask.question == "who?" and len(ask.thread) == 64
    assert ask.held == [] and ask.sent == [] and ask.took_s >= 0
    assert Ask({"held": ["a"]}).held == ["a"]


# ------------------------------------------------------------ of any length


def notes(turns):
    """A scripted summary writer that keeps every id it is given."""
    ids = dict.fromkeys(i for t in turns for how in t.drew for i in t.drew[how])
    return f"So far: Iris surveys land. Rests on: {', '.join(ids)}."


@pytest.mark.slow


def test_the_window_is_the_last_ten_turns_in_order_and_nothing_else(served):
    """With recall off and no summariser, what goes back is exactly the last WINDOW turns,
    chosen by recency, in the order they were said -- as it always was, only longer."""
    url, handler = served
    handler.recalled_turns = 0
    assert handler.remembered_turns == WINDOW == 10
    for n in range(6):
        call(url + "/ask", "POST", {"question": f"question {n}", "thread": "w1"})
    call(url + "/ask", "POST", {"question": "and who else?", "thread": "w1"})
    last = handler.asked[-1]
    want = []
    for n in range(1, 6):
        want += [{"role": "user", "content": f"question {n}"},
                 {"role": "assistant", "content": "Iris surveys land."}]
    assert last["turns"] == want and len(last["turns"]) == WINDOW
    assert last["summary"] is None and last["recalled"] == []


@pytest.mark.slow


def test_history_carries_the_summary_and_the_recalled_turns_ahead_of_the_window(tmp_path):
    """The new shape: the window as before, with the summary and what was recalled on it."""
    with GraphStore(tmp_path / "graph.ladybug") as held:
        held.write(GRAPH)

    class Handler(Scripted):
        store_path = tmp_path / "graph.ladybug"
        asked = []
        remembered_turns = 2          # a short window, so there is something to recall
        summary_every = 4

        def summariser(self):
            return notes

    with threaded_server(Handler) as url:
        call(url + "/ask", "POST", {"question": "who surveys land?", "thread": "c1"})
        assert Handler.asked[-1]["summary"] is None and Handler.asked[-1]["recalled"] == []
        call(url + "/ask", "POST", {"question": "what else?", "thread": "c1"})
        # four turns said: the summary was written after that answer went out
        call(url + "/ask", "POST", {"question": "so who surveys land again?", "thread": "c1"})
        last = Handler.asked[-1]
        assert last["turns"] == [{"role": "user", "content": "what else?"},
                                 {"role": "assistant", "content": "Iris surveys land."}]
        assert last["summary"].role == SUMMARY
        assert last["summary"].text == "So far: Iris surveys land. Rests on: topic:surveying, person:iris."
        assert last["summary"].drew == {"shown": ["topic:surveying", "person:iris"]}
        # recalled from outside the window, never from inside it
        assert [t.text for t in last["recalled"]] == ["who surveys land?", "Iris surveys land."]
        assert [t.seq for t in last["recalled"]] == [1, 2]
        assert last["recalled"][1].drew["shown"] == ["person:iris"]

        # the same thing by name, without a request
        handler = Handler.__new__(Handler)
        history = handler.history("c1", [], question="who surveys land?")
        assert isinstance(history, History) and isinstance(history, list)
        assert set(history.as_dict()) == {"summary", "recalled", "turns"}
        assert history.as_dict()["turns"] == list(history) and len(history) == 2
        # six turns now, so turn 4 ("Iris surveys land." again) is outside the window too
        assert history.summary.role == SUMMARY and [t.seq for t in history.recalled] == [1, 2, 4]
        # a wider window is asked for by name, and recall keeps out of it
        wide = handler.history("c1", [], question="who surveys land?", window=WINDOW)
        assert len(wide) == 6 and wide.recalled == []
        # no question, nothing recalled; no thread, the page's own list
        assert handler.history("c1", []).recalled == []
        assert handler.history("", [{"role": "user", "content": "x"}]).as_dict() == {
            "summary": None, "recalled": [], "turns": [{"role": "user", "content": "x"}]}

        # the page never sees the summary as a turn
        _, _, replay = call(url + "/thread/c1")
        assert [t["role"] for t in replay["turns"]] == ["user", "assistant"] * 3
        with GraphStore(Handler.store_path, read_only=True) as held:
            assert [t.role for t in follow(held, "c1", summaries=True)].count(SUMMARY) == 1


def test_a_summariser_that_raises_loses_the_summary_not_the_answer(tmp_path, capsys):
    with GraphStore(tmp_path / "graph.ladybug") as held:
        held.write(GRAPH)

    class Handler(Scripted):
        store_path = tmp_path / "graph.ladybug"
        asked = []
        summary_every = 2

        def summariser(self):
            def broken(turns):
                raise RuntimeError("the writer went away")
            return broken

    with threaded_server(Handler) as url:
        code, _, out = call(url + "/ask", "POST", {"question": "who?", "thread": "c1"})
        assert code == 200 and out["content"] == "Iris surveys land."
        _, _, replay = call(url + "/thread/c1")
        assert len(replay["turns"]) == 2
    assert "the writer went away" in capsys.readouterr().err


def test_the_done_frame_and_the_plain_answer_say_which_model_answered_and_what_it_spent():
    """An `Answer` carries `spent`; the payload carries the model's name and the record --
    calls, seconds, tokens read, cached, written, drafted -- for the page's footer."""
    from ml_stack.client.chat import Reply

    out = Answer(content="Iris surveys land.")
    out.spent.note(Reply(content="ok", raw={"model": "tiny-Q4.gguf",
                                            "usage": {"prompt_tokens": 900, "completion_tokens": 40},
                                            "timings": {"prompt_n": 300, "cache_n": 600,
                                                        "prompt_ms": 100.0, "predicted_ms": 400.0,
                                                        "draft_n": 20, "draft_n_accepted": 15}}),
                   took=0.8)
    payload = answer_payload(out)
    assert payload["model"] == "tiny-Q4.gguf"
    spent = payload["spent"]
    assert spent["calls"] == 1 and spent["read_tokens"] == 300 and spent["cached_tokens"] == 600
    assert spent["completion_tokens"] == 40 and spent["drafted"] is True
    assert spent["acceptance"] == 0.75 and spent["tokens_per_second"] == 80.0
    # the model's own pace is decode alone; reading the prompt is its own number
    assert spent["decode_tokens_per_second"] == 100.0 and spent["prompt_tokens_per_second"] == 3000
    assert spent["first_token"] == 0.4
    assert json.dumps(payload)                      # JSON-ready, derived fields included


def test_the_model_route_names_what_is_serving_before_anything_is_asked(served):
    from urllib.request import urlopen

    def get(where):
        with urlopen(where, timeout=5) as r:
            return json.loads(r.read().decode("utf-8"))

    url, handler = served
    handler.model = ""
    assert get(url + "/ask/model") == {"model": "", "ready": None}
    handler.model = "hf:someone/tiny-GGUF/tiny-Q4.gguf"
    assert get(url + "/ask/model")["model"] == "hf:someone/tiny-GGUF/tiny-Q4.gguf"


def test_the_page_shows_the_served_model_and_what_each_answer_spent():
    import ml_stack.graph as graph_package

    html = (Path(graph_package.__file__).parent / "web" / "graph.html").read_text(encoding="utf-8")
    assert "fetch('/ask/model')" in html and "answered by" in html
    assert "askpane-title" in html and "ev.spent" in html


def test_the_session_totals_add_up_every_answer_in_the_thread(served):
    """Each remembered answer keeps its `spent`; the done frame and `/thread` carry the
    session's totals -- the conversation's cost so far, not the last turn's."""
    from urllib.request import urlopen

    from urllib.request import Request

    def post(where, body):
        req = Request(where, data=json.dumps(body).encode("utf-8"),
                      headers={"Content-Type": "application/json"}, method="POST")
        with urlopen(req, timeout=10) as r:
            return json.loads(r.read().decode("utf-8"))

    url, handler = served
    kept = handler.answer
    try:
        from ml_stack.client.chat import Reply

        out = Answer(content="Iris surveys land.", ids=["person:iris"], read=["person:iris"],
                     show=["person:iris"], steps=["found 2 entries"])
        out.spent.note(Reply(content="x", raw={"model": "tiny-Q4.gguf",
                                               "usage": {"prompt_tokens": 100, "completion_tokens": 10},
                                               "timings": {"prompt_n": 60, "cache_n": 40,
                                                           "predicted_ms": 200.0}}), took=0.5)
        handler.answer = out
        first = post(url + "/ask", {"question": "who surveys?", "thread": "totals"})
        assert first["spent"]["calls"] == 1
        assert first["session"]["answers"] == 1 and first["session"]["read_tokens"] == 60
        _, raw = stream(url + "/ask/stream", {"question": "and again?", "thread": "totals"})
        done = frames(raw)[-1]
        assert done["session"]["answers"] == 2
        assert done["session"]["read_tokens"] == 120 and done["session"]["cached_tokens"] == 80
        assert done["session"]["completion_tokens"] == 20 and done["session"]["calls"] == 2
        assert done["session"]["models"] == ["tiny-Q4.gguf"]
        assert done["session"]["tokens_per_second"] == 50.0
        with urlopen(url + "/thread/totals", timeout=5) as r:
            thread = json.loads(r.read())
        assert thread["session"]["answers"] == 2
        assert [t["meta"]["spent"]["calls"] for t in thread["turns"] if t["role"] == "assistant"] == [1, 1]
    finally:
        handler.answer = kept


def test_totals_over_nothing_is_an_empty_session():
    from ml_stack.client.spent import Spent

    assert Spent.totals([])["answers"] == 0
    assert Spent.totals([None, {"calls": 0}])["answers"] == 0


def test_the_session_totals_carry_the_peak_context_and_the_parts_summed():
    from ml_stack.client.spent import Spent

    a = {"calls": 2, "context_peak": 9000, "parts": {"system": 400, "window": 2000}}
    b = {"calls": 1, "context_peak": 4000, "parts": {"system": 400, "recalled": 800}}
    got = Spent.totals([a, b])
    assert got["context_peak"] == 9000 and got["context_mean"] == 6500
    assert got["parts"] == {"system": 800, "window": 2000, "recalled": 800}


# -- what the server has spent ---------------------------------------------------------
#
# `/metrics` and `/metrics.prom` are the same ring of answers, as JSON and as Prometheus
# text. The ring is on the handler class, so these drive a real server and read it back
# rather than calling `metrics()` on an instance that never served anything.


def spent_answer(*, calls=1, content="Iris surveys land.", **timings):
    """An `Answer` carrying a `Spent` with ``calls`` replies noted into it."""
    from ml_stack.client.chat import Reply

    out = Answer(content=content, ids=["person:iris"], read=["person:iris"],
                 show=["person:iris"], steps=["found 2 entries"])
    for _ in range(calls):
        out.spent.note(Reply(content=content, finish_reason="stop", raw={
            "model": "tiny-Q4.gguf",
            "usage": {"prompt_tokens": 900, "completion_tokens": 40},
            "timings": {"prompt_ms": 100.0, "predicted_ms": 400.0,
                        "prompt_n": timings.get("prompt_n", 300),
                        "cache_n": timings.get("cache_n", 600),
                        "predicted_n": 40,
                        "draft_n": timings.get("draft_n", 20),
                        "draft_n_accepted": timings.get("draft_taken", 15)}}), 0.8)
    return out


def get(url):
    with urllib.request.urlopen(url, timeout=5) as r:
        return r.status, r.headers.get("Content-Type"), r.read().decode("utf-8")


def post(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read().decode("utf-8"))


def test_metrics_answers_before_anything_was_asked(served):
    """A server that has answered nothing reports zeros and its uptime -- which is how a
    scraper tells a quiet server from one that is not there at all."""
    url, _ = served
    status, kind, raw = get(url + "/metrics")
    got = json.loads(raw)
    assert status == 200 and "application/json" in kind
    assert got["answers"] == 0 and got["recent"] == [] and got["totals"]["answers"] == 0
    assert got["uptime"] >= 0 and got["threads"] == {}


def test_metrics_counts_every_answer_and_keeps_the_last_ones(served):
    url, handler = served
    kept = handler.answer
    try:
        handler.answer = spent_answer(calls=2)
        post(url + "/ask", {"question": "who surveys?", "thread": "t1"})
        stream(url + "/ask/stream", {"question": "and again?", "thread": "t2"})
        got = json.loads(get(url + "/metrics")[2])
        assert got["answers"] == 2 and len(got["recent"]) == 2
        assert got["totals"]["calls"] == 4 and got["totals"]["read_tokens"] == 1200
        assert got["totals"]["cached_tokens"] == 2400 and got["totals"]["context_peak"] == 940
        assert got["model"] == "tiny-Q4.gguf"
        # each record is `Spent.public()` with the thread and the clock on it
        first = got["recent"][0]
        assert first["calls"] == 2 and first["thread"] == "t1" and first["at"] > 0
        assert set(got["threads"]) == {"t1", "t2"}
        assert got["threads"]["t1"]["answers"] == 1 and got["threads"]["t1"]["calls"] == 2
    finally:
        handler.answer = kept


def test_an_answer_no_model_was_asked_for_is_counted_and_not_kept(served):
    """A greeting, or an answer that came back from the cache, spent nothing. It still
    happened -- so it is counted -- but there is no record of a call to keep."""
    url, handler = served
    got = json.loads(get(url + "/metrics")[2])
    assert got["answers"] == 0
    post(url + "/ask", {"question": "hello?"})           # the fixture's answer has no spent
    got = json.loads(get(url + "/metrics")[2])
    assert got["answers"] == 1 and got["recent"] == []


def test_the_ring_is_bounded_so_a_server_that_answers_all_week_does_not_grow(tmp_path):
    class Handler(Scripted):
        asked = []
        keep_answers = 3
        answer = spent_answer()

    with threaded_server(Handler) as url:
        for i in range(5):
            post(url + "/ask", {"question": f"who is {i}?", "thread": "long"})
        got = json.loads(get(url + "/metrics")[2])
    assert got["answers"] == 5 and got["kept"] == 3 and len(got["recent"]) == 3
    assert got["totals"]["answers"] == 3, "the totals are over what is kept, and say so"


def test_two_servers_in_one_process_do_not_add_up_together(tmp_path):
    """The ring lives on the concrete handler class, not on `AskRoutes`."""
    class One(Scripted):
        asked = []
        answer = spent_answer()

    class Two(Scripted):
        asked = []
        answer = spent_answer()

    with threaded_server(One) as first, threaded_server(Two) as second:
        post(first + "/ask", {"question": "who?"})
        post(first + "/ask", {"question": "who else?"})
        post(second + "/ask", {"question": "and here?"})
        assert json.loads(get(first + "/metrics")[2])["answers"] == 2
        assert json.loads(get(second + "/metrics")[2])["answers"] == 1


def test_metrics_prom_is_the_same_numbers_a_scraper_can_read(served):
    url, handler = served
    kept = handler.answer
    try:
        handler.answer = spent_answer(calls=2)
        handler.model = "tiny-Q4.gguf"
        post(url + "/ask", {"question": "who surveys?", "thread": "t1"})
        status, kind, body = get(url + "/metrics.prom")
        assert status == 200 and kind.startswith("text/plain")
        lines = [ln for ln in body.splitlines() if ln and not ln.startswith("#")]
        said = dict(ln.split(" ", 1) for ln in lines if not ln.startswith("model_info"))
        assert said["answers_total"] == "1" and said["calls_total"] == "2"
        assert said["tokens_read_total"] == "600" and said["tokens_cached_total"] == "1200"
        assert said["tokens_written_total"] == "80" and said["context_peak"] == "940"
        assert said["draft_tokens_total"] == "40" and said["draft_accepted_total"] == "30"
        assert float(said["seconds_total"]) == 1.6
        assert 'model_info{model="tiny-Q4.gguf"} 1' in body
        # every metric a scraper reads is declared, once, before its value
        for name, _, kind_, _said in __import__("ml_stack.graph.serve", fromlist=["PROM"]).PROM:
            assert f"# TYPE {name} {kind_}" in body and f"# HELP {name} " in body
    finally:
        handler.answer = kept
        handler.model = ""


def test_a_model_name_with_a_quote_in_it_does_not_break_the_exposition(served):
    """A label value is escaped, or one odd model name makes the whole scrape unparsable."""
    from ml_stack.graph.serve import prometheus

    body = prometheus({"answers": 1}, model='tiny "Q4"\\x.gguf', uptime=3.0)
    line = [ln for ln in body.splitlines() if ln.startswith("model_info")][0]
    assert line == 'model_info{model="tiny \\"Q4\\"\\\\x.gguf"} 1'
    assert body.endswith("\n")


# -- the concrete handler: the page, the exports, and a 404 for the rest ------------------

from ml_stack.graph.serve import Handler, bind, main  # noqa: E402


def fetch(url):
    """(status, content-type, cache-control, body) for one GET, error bodies included."""
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            return r.status, r.headers.get("Content-Type"), r.headers.get("Cache-Control"), r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.headers.get("Content-Type"), e.headers.get("Cache-Control"), e.read()


@pytest.fixture
def site(tmp_path):
    """A rendered page and an export root with one JSON file in a subdirectory."""
    from ml_stack.graph.page import render

    page = tmp_path / "site" / "index.html"
    page.parent.mkdir()
    page.write_text(render(GRAPH, title="Invented"), encoding="utf-8")
    export = tmp_path / "export"
    (export / "a").mkdir(parents=True)
    (export / "a" / "b.json").write_text(json.dumps({"kept": True}))
    (export / "a" / "note.html").write_text("<p>hello</p>")
    (export / "a" / "blob.bin").write_bytes(b"\x00\x01")
    (tmp_path / "pyproject.toml").write_text("[secret]\n")
    return page, export


def test_the_root_serves_the_page_with_the_live_script_and_no_store(site):
    page, export = site
    with threaded_server(Handler.configured(site=page, export=export)) as url:
        for path in ("/", "/index.html"):
            status, ctype, cache, body = fetch(url + path)
            assert status == 200
            assert ctype.startswith("text/html")
            assert cache == "no-store"
            assert b"window.GRAPH_LIVE=1" in body
            assert b"<title>Invented</title>" in body
            assert body.index(b"GRAPH_LIVE") < body.index(b"<title>")


def test_an_export_is_served_by_its_suffix(site):
    page, export = site
    with threaded_server(Handler.configured(site=page, export=export)) as url:
        status, ctype, _, body = fetch(url + "/export/a/b.json")
        assert (status, ctype) == (200, "application/json")
        assert json.loads(body) == {"kept": True}
        status, ctype, _, body = fetch(url + "/export/a/note.html")
        assert status == 200 and ctype.startswith("text/html") and body == b"<p>hello</p>"
        status, ctype, _, body = fetch(url + "/export/a/blob.bin")
        assert (status, ctype, body) == (200, "application/octet-stream", b"\x00\x01")


def test_a_path_that_climbs_out_of_the_export_root_is_a_404(site):
    page, export = site
    with threaded_server(Handler.configured(site=page, export=export)) as url:
        for path in ("/export/../pyproject.toml", "/export/a/../../pyproject.toml",
                     "/export/%2e%2e/pyproject.toml", "/export/a", "/export/missing.json",
                     "/export/"):
            status, _, _, body = fetch(url + path)
            assert status == 404, path
            assert b"secret" not in body


def test_anything_else_is_a_404_and_the_ask_routes_still_answer(site):
    page, export = site
    with threaded_server(Handler.configured(site=page, export=export)) as url:
        assert fetch(url + "/nothing")[0] == 404
        assert fetch(url + "/site/index.html")[0] == 404
        status, _, body = call(url + "/ask/model")
        assert status == 200 and body["model"] == ""
        assert call(url + "/metrics")[0] == 200
        status, _, body = call(url + "/ask", "POST", {"question": "who?"})
        assert status == 500 and "graph" in body["error"]


def test_a_handler_without_an_export_root_has_no_export_route(site):
    page, _ = site
    with threaded_server(Handler.configured(site=page)) as url:
        assert fetch(url + "/export/a/b.json")[0] == 404
        assert fetch(url + "/")[0] == 200


def test_bind_takes_the_command_line_and_answers_on_loopback(site, monkeypatch):
    page, export = site
    httpd = bind(["serve", "--site", str(page), "--export", str(export), "--port", "0",
                  "--model", "hf:invented/model-GGUF/model.gguf"])
    try:
        assert httpd.server_address[0] == "127.0.0.1"
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        url = f"http://127.0.0.1:{httpd.server_address[1]}"
        assert b"GRAPH_LIVE" in fetch(url + "/")[3]
        status, _, body = call(url + "/ask/model")
        assert status == 200 and body["model"] == "model.gguf"
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_main_prints_where_it_is_serving_then_serves(site, monkeypatch, capsys):
    page, _ = site
    served = []
    monkeypatch.setattr("http.server.ThreadingHTTPServer.serve_forever",
                        lambda self, *a, **k: served.append(self.server_address))
    assert main(["serve", "--site", str(page), "--port", "0"]) == 0
    assert served and served[0][0] == "127.0.0.1"
    assert f"http://127.0.0.1:{served[0][1]}" in capsys.readouterr().out
