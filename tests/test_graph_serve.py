"""The page's three routes, driven over a real socket with a scripted asker and no model.

Every fixture is invented; nothing reads a real graph. The server is a real
``ThreadingHTTPServer`` on a free port, because the bugs these routes exist to prevent are
transport-shaped: a frame that was buffered rather than flushed, a ``done`` that arrived
without the answer on it, a 409 sent after the stream headers.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from ml_stack.graph.ask import Answer
from ml_stack.graph.serve import Ask, AskRoutes, History, answer_payload, sse, thread_request
from ml_stack.graph.store import GraphStore
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

    def do_GET(self):
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


@contextmanager
def running(handler):
    """A live server on a free port for ``handler``; yields its url."""
    srv = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{srv.server_address[1]}"
    finally:
        srv.shutdown()


@pytest.fixture
def served(tmp_path):
    """A live server with a conversation store beside it; yields (url, handler class)."""
    with GraphStore(tmp_path / "graph.ladybug") as held:
        held.write(GRAPH)

    class Handler(Scripted):
        store_path = tmp_path / "graph.ladybug"
        asked = []

    with running(Handler) as url:
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

    with running(Handler) as url:
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

    with running(Handler) as url:
        code, _, out = call(url + "/ask", "POST", {"question": "who?", "thread": "c1"})
        assert code == 200 and out["content"] == "Iris surveys land."
        _, _, replay = call(url + "/thread/c1")
        assert len(replay["turns"]) == 2
    assert "the writer went away" in capsys.readouterr().err
