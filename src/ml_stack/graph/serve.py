"""The routes the graph page talks to, for any ``http.server`` handler to mix in.

``graph.html`` POSTs a question to ``/ask/stream`` and reads the answer as it is found --
server-sent events, one frame per thing the model did, then a ``done`` frame carrying the
whole answer. A server without that route gets the same question at ``/ask`` and answers in
one JSON body. And ``GET /thread/<name>`` reopens a conversation the graph is holding, so
closing the tab does not end it and another machine can pick it up.

Those three routes were the same in every project that rendered the page, and lived in none
of them here. ``AskRoutes`` is that server-side half, with no opinion about where the graph
comes from or which model answers: a subclass says how a question is answered (``asker``)
and where conversations are kept (``threads``), and hangs whatever it wants -- a journal, a
review queue, a seat number -- off ``answered`` and ``failed``.

What goes back with a question is ``history``'s ``History``: the last ``WINDOW`` turns as
messages, always, and on it the latest ``summary`` and the earlier turns ``recalled`` for
this question, for ``converse(..., summary=, recalled=)``. A subclass that returns an
``embedder`` gets turns recalled by meaning as well as by their words; one that returns a
``summariser`` gets the summary rolled forward every ``summary_every`` turns.

::

    class Handler(AskRoutes, BaseHTTPRequestHandler):
        def asker(self, question, *, turns, held, stream, emit):
            if stream:
                return converse_stream(question, graph, client, turns=turns, held=held,
                                       on_event=emit)
            return converse(question, graph, client, turns=turns, held=held)

        def threads(self, *, write=False):
            return GraphStore(path, read_only=not write)

        def do_POST(self):
            body = self.read_body() or {}
            if self.path == "/ask":
                self.handle_ask(body)
            elif self.path == "/ask/stream":
                self.handle_ask_stream(body)

Nothing here decides who may ask. A page served over a tunnel and one on loopback get the
same routes; refusing one is the subclass's policy, checked before these are called.
"""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, nullcontext
from typing import Any

from ml_stack.graph.thread import EVERY, WINDOW

__all__ = ["Ask", "AskRoutes", "History", "answer_payload", "sse", "thread_request"]

# what an answer carries to the page, in the order the page reads them: `content` is the
# words, `ids` everything the tools touched, then the four ways they touched it, then the
# working as one line (`why`) and one step at a time (`steps`)
PAYLOAD = ("content", "ids", "found", "read", "path", "show", "why", "steps")


def sse(wfile: Any, event: Mapping[str, Any]) -> None:
    """Write one server-sent event frame -- ``data: {json}\\n\\n`` -- and flush it.

    One frame per event, JSON in the data line and nothing else, because that is the whole
    of what the page's reader parses. Flushing is the point: a frame held in a buffer until
    the answer is complete is an answer that did not stream.
    """
    wfile.write(f"data: {json.dumps(dict(event), ensure_ascii=False)}\n\n".encode())
    wfile.flush()


def answer_payload(out: Any, **extra: Any) -> dict[str, Any]:
    """The shape both ask routes return: an ``Answer``, or a mapping already in that shape.

    ``content`` is what to say; ``ids`` what to light; ``found``, ``read``, ``path`` and
    ``show`` how each entry was touched, kept apart because lighting what was merely found
    as though it were the answer floods the graph; ``steps`` the working one step at a
    time, and ``why`` the same steps joined, for anything that reads a single line.

    A mapping keeps every key it came with -- a project that adds ``raised`` or
    ``remembered`` to its answers gets them through untouched -- and only ``why`` and
    ``steps`` are filled in from each other when one is missing. ``extra`` rides along on
    top, for what the route knows and the answer does not.
    """
    if isinstance(out, Mapping):
        payload = dict(out)
    else:
        payload = {k: getattr(out, k) for k in PAYLOAD if hasattr(out, k)}
        spent = getattr(out, "spent", None)
        if spent is not None and hasattr(spent, "public") and getattr(spent, "calls", 0):
            # which model answered and what it cost: calls, seconds, tokens read, written,
            # cached and drafted -- for the page's footer and for anyone testing an answer.
            # An answer no model was asked for (a greeting, a cached one) carries neither.
            payload["model"] = spent.model
            payload["spent"] = spent.public()
    payload.setdefault("content", "")
    steps = payload.get("steps")
    if steps is not None:
        payload["steps"] = [str(s) for s in steps]
    if payload.get("why") is None and steps is not None:
        payload["why"] = "; ".join(payload["steps"])
    if steps is None and payload.get("why"):
        payload["steps"] = [s for s in str(payload["why"]).split("; ") if s]
    payload.setdefault("steps", [])
    payload.setdefault("why", "")
    for key in ("ids", "found", "read", "path", "show"):
        if key in payload and payload[key] is not None:
            payload[key] = [str(i) for i in payload[key]]
    payload.update(extra)
    return payload


def thread_request(path: str) -> tuple[str, bool] | None:
    """``/thread/<name>[?working=1]`` read out of a request path; None for any other path.

    The name is capped at 64 characters, which is a page's conversation id and not a
    query. ``working`` asks for what each turn drew on, which is what lets an old answer
    light the graph again.
    """
    if not path.startswith("/thread/"):
        return None
    rest = path[len("/thread/"):]
    name, _, query = rest.partition("?")
    return name[:64], "working=1" in query


class History(list):
    """What goes back with a question: the window as messages, and what goes ahead of it.

    A list of ``{"role", "content"}`` -- the last ``WINDOW`` turns, chosen by recency alone,
    exactly what ``history`` always returned -- so every asker that passes ``turns`` on
    keeps working. On it, for ``converse(..., summary=, recalled=)``: ``summary`` is the
    latest summary ``Turn`` or None, ``recalled`` the earlier turns found for this
    question, oldest first. ``as_dict`` is the same three things by name.
    """

    __slots__ = ("summary", "recalled")

    def __init__(self, turns: Sequence[Mapping[str, str]] = (), *, summary: Any = None,
                 recalled: Sequence[Any] = ()) -> None:
        super().__init__(turns)
        self.summary = summary
        self.recalled = list(recalled)

    def as_dict(self) -> dict[str, Any]:
        return {"summary": self.summary, "recalled": list(self.recalled), "turns": list(self)}


class Ask:
    """One question as the page sent it, after the body was checked and history resolved."""

    __slots__ = ("question", "sent", "turns", "thread", "held", "body", "began")

    def __init__(self, body: Mapping[str, Any]) -> None:
        self.body = dict(body)
        self.question = str(body.get("question") or "").strip()
        self.sent = body.get("turns") if isinstance(body.get("turns"), list) else []
        self.turns: list = list(self.sent)
        self.thread = str(body.get("thread") or "")[:64]
        held = body.get("held")
        self.held = list(held) if isinstance(held, list) and all(isinstance(h, str) for h in held) else []
        self.began = time.time()

    @property
    def took_s(self) -> float:
        return round(time.time() - self.began, 1)

    @property
    def summary(self) -> Any:
        """The latest summary of the thread, when ``history`` found one."""
        return getattr(self.turns, "summary", None)

    @property
    def recalled(self) -> list:
        """The earlier turns recalled for this question, oldest first."""
        return list(getattr(self.turns, "recalled", ()) or ())


class AskRoutes:
    """``/ask``, ``/ask/stream`` and ``/thread/<name>`` for a ``BaseHTTPRequestHandler``.

    Mix it in ahead of the handler class and call the three ``handle_*`` methods from
    ``do_GET`` and ``do_POST``; each writes its own response and returns what it sent.
    Two things are the subclass's to say, and the rest has a default:

    ``asker(question, *, turns, held, stream, emit)``
        Answers. Returns an ``Answer`` (or a mapping in ``answer_payload``'s shape).
        When ``stream`` is true the model's events are reported through ``emit`` as they
        happen -- ``converse_stream``'s ``on_event`` -- or the asker may instead return an
        iterator that yields them and returns the answer. ``self.asking`` is the ``Ask``
        while a question is being answered, for anything else the body carried.
    ``threads(*, write=False)``
        A context manager yielding the store conversations are kept in, opened for reading
        or for writing. None means no history: questions still get answers, and
        ``/thread`` comes back empty.
    ``ready()``
        A reason the server cannot answer yet, or None. Checked before anything is written,
        so it can still be a 409 rather than an error frame.
    ``answered(ask, out, payload)``
        Called with the answer before the response goes out; returns the payload to send.
        The default remembers the pair of turns in the thread. A subclass that journals or
        queues something does it here, after ``super()``, so the page is never told a
        thing happened before it has.
    ``failed(ask, exc)``
        Called when answering raised, before the error is sent.
    ``remembered_turns``
        How many turns of history go back to the model with the question: the window,
        chosen by recency alone and never trimmed for what else is sent. ``WINDOW``.
    ``recalled_turns``
        How many earlier turns ``recall`` may add ahead of the window; 0 turns it off.
    ``embedder()``
        ``texts -> vectors``, or None. Given, every turn is embedded as it is remembered
        and recalled by meaning as well as by its words -- one embedding call per turn
        written and one per question asked.
    ``summariser()``
        ``turns -> str``, or None. Given, the summary is rolled forward every
        ``summary_every`` turns, after the answer has gone out: one more short model
        call, over that many turns and the previous paragraph, once per ``summary_every``
        turns. ``partial(thread.write_summary, client)`` is that writer for a chat client.
    """

    remembered_turns: int = WINDOW
    recalled_turns: int = 3
    summary_every: int = EVERY
    asking: Ask | None = None

    # ------------------------------------------------------------- what a subclass says

    def asker(self, question: str, *, turns: list, held: list, stream: bool,
              emit: Any) -> Any:
        raise NotImplementedError("a subclass says how a question is answered")

    def threads(self, *, write: bool = False) -> AbstractContextManager[Any] | None:
        return None

    def ready(self) -> str | None:
        return None

    def model_name(self) -> str:
        """What is answering, for a reader who has not asked anything yet -- the served
        model's name, as the subclass knows it (a lease, a config, a probe of the server).
        Empty when it cannot say; every answer still carries the name the server reported."""
        return ""

    def handle_model(self) -> dict[str, Any]:
        """``GET /ask/model``: ``{"model": name}`` -- the page shows it in the ask pane."""
        payload = {"model": self.model_name() or "", "ready": self.ready()}
        self.send_json(200, payload)
        return payload

    def embedder(self) -> Callable[[Sequence[str]], Sequence[Sequence[float]]] | None:
        return None

    def summariser(self) -> Callable[[Sequence[Any]], str] | None:
        return None

    def answered(self, ask: Ask, out: Any, payload: dict[str, Any]) -> dict[str, Any]:
        self.remember(ask, out)
        if ask.thread:
            # the whole conversation's cost so far, this answer included -- the page shows
            # it as "this session", beside what the one answer spent
            session = self.session(ask.thread)
            if session and session.get("answers"):
                payload["session"] = session
        return payload

    def session(self, thread: str) -> dict[str, Any] | None:
        """`Spent.totals` over every answer remembered in ``thread``, or None without a
        store: what the session has cost, not just the last turn."""
        from ml_stack.client.spent import Spent
        from ml_stack.graph.thread import follow

        try:
            with self._opened() as held:
                if held is None:
                    return None
                records = [(t.meta or {}).get("spent") for t in follow(held, str(thread)[:64],
                                                                     working=False)
                           if t.role == "assistant"]
        except Exception:  # noqa: BLE001 - a session that cannot be read is no session
            return None
        return Spent.totals(records)

    def failed(self, ask: Ask, exc: BaseException) -> None:
        print(f"{time.strftime('%FT%T')} ask failed: {exc}", file=sys.stderr)

    # ---------------------------------------------------------------- the routes

    def handle_ask(self, body: Mapping[str, Any]) -> dict[str, Any]:
        """Answer in one JSON body. Returns what was sent."""
        ask = self._checked(body)
        if ask is None:
            return self.send_json(400, {"error": "no question"})
        reason = self.ready()
        if reason:
            return self.send_json(409, {"error": reason})
        try:
            out = self._answer(ask, stream=False, emit=None)
            payload = self.answered(ask, out, answer_payload(out))
        except Exception as exc:  # noqa: BLE001 - the page is told, whatever it was
            self.failed(ask, exc)
            return self.send_json(500, {"error": str(exc)[:200]})
        return self.send_json(200, payload)

    def handle_ask_stream(self, body: Mapping[str, Any]) -> dict[str, Any]:
        """Answer as server-sent events; the ``done`` frame carries the whole answer.

        Every event the asker reports is relayed as it happens except ``done``, which is
        sent last with the full payload on it -- the same one ``/ask`` would have returned
        -- so the page finishes a streamed answer exactly as it finishes a plain one. A
        failure after the headers is an ``error`` frame, since a status code is no longer
        available. Returns the ``done`` or ``error`` frame that was sent.
        """
        ask = self._checked(body)
        if ask is None:
            return self.send_json(400, {"error": "no question"})
        reason = self.ready()
        if reason:
            return self.send_json(409, {"error": reason})
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()

        def relay(event: Mapping[str, Any]) -> None:
            if event.get("event") != "done":
                sse(self.wfile, event)

        try:
            out = self._answer(ask, stream=True, emit=relay)
            payload = self.answered(ask, out, answer_payload(out))
            frame = {"event": "done", **payload}
            sse(self.wfile, frame)
            return frame
        except (BrokenPipeError, ConnectionResetError):
            return {"event": "error", "error": "the page went away"}
        except Exception as exc:  # noqa: BLE001
            self.failed(ask, exc)
            frame = {"event": "error", "error": str(exc)[:200]}
            try:
                sse(self.wfile, frame)
            except OSError:
                pass
            return frame

    def handle_thread(self, name: str, working: bool = False) -> dict[str, Any]:
        """The turns of one conversation, oldest first, each with its ``meta``.

        ``working`` brings back what each answer drew on. No history is not a broken page:
        a store that cannot be opened, or none at all, is an empty conversation with a
        note, never an error.
        """
        from ml_stack.graph.thread import follow

        name = str(name)[:64]
        try:
            with self._opened() as held:
                turns = [] if held is None else [t.as_dict() for t in follow(held, name, working=working)]
            from ml_stack.client.spent import Spent

            session = Spent.totals([(t.get("meta") or {}).get("spent") for t in turns
                                    if t.get("role") == "assistant"])
            held_back = {"thread": name, "turns": turns}
            if session.get("answers"):
                held_back["session"] = session      # nothing spent is nothing to show
            return self.send_json(200, held_back)
        except Exception as exc:  # noqa: BLE001
            return self.send_json(200, {"thread": name, "turns": [], "note": str(exc)[:120]})

    # ----------------------------------------------------------- conversations

    def history(self, thread: str, sent: list, *, question: str = "",
                window: int | None = None) -> History:
        """The turns before this one: the store's, when it has any, else the page's.

        A ``History``: the last ``window`` turns as messages (``remembered_turns`` by
        default) -- chosen by recency alone, so a follow-up resolves from them with nothing
        else -- carrying the thread's latest ``summary`` and, given the ``question``, the
        earlier turns ``recalled`` for it. Neither is ever taken out of the window.

        The page's own list stays as a fallback, so a reader whose store is unavailable
        still gets a conversation rather than an error -- and a reopened page, which sends
        nothing, gets the conversation the graph is holding.
        """
        if not thread:
            return History(sent)
        keep = int(self.remembered_turns if window is None else window)
        try:
            from ml_stack.graph.thread import latest_summary, recall, recent

            with self._opened() as held:
                if held is None:
                    return History(sent)
                kept = recent(held, thread, turns=keep)
                summary = latest_summary(held, thread)
                recalled = []
                if question and int(self.recalled_turns) > 0:
                    recalled = recall(held, thread, question, embedder=self.embedder(),
                                      limit=int(self.recalled_turns), window=keep)
            return History(kept or sent, summary=summary, recalled=recalled)
        except Exception:  # noqa: BLE001 - a conversation is not worth failing an answer for
            return History(sent)

    def remember(self, ask: Ask, out: Any) -> None:
        """Write the pair of turns, joined to what the answer drew on. Best effort.

        The steps ride along as a list, so a reopened conversation gets its trace tally
        back; ``why`` is the same steps joined, kept beside them for older readers. A store
        that will not open -- another process holds the writer -- loses the record, not
        the answer, which the reader already has.

        With an ``embedder`` both turns are embedded as they are written; with a
        ``summariser`` the summary is rolled forward when ``summary_every`` turns have
        been said since the last -- one more model call, made here, after the answer has
        gone out.
        """
        if not ask.thread:
            return
        try:
            from ml_stack.graph.thread import drew_on, remember_turn, summarise

            payload = answer_payload(out)
            embed = self.embedder()
            with self._opened(write=True) as held:
                if held is None:
                    return
                remember_turn(held, thread=ask.thread, role="user", text=ask.question,
                              embedder=embed)
                remember_turn(held, thread=ask.thread, role="assistant",
                              text=str(payload.get("content") or ""), drew=drew_on(payload),
                              meta={"why": payload.get("why", ""), "steps": payload.get("steps", []),
                                    "model": payload.get("model", ""),
                                    "spent": payload.get("spent")},
                              embedder=embed)
                writer = self.summariser()
                if writer is not None:
                    summarise(held, ask.thread, writer, every=int(self.summary_every))
        except Exception as exc:  # noqa: BLE001
            print(f"{time.strftime('%FT%T')} turn not remembered: {exc}", file=sys.stderr)

    # ---------------------------------------------------------------- plumbing

    def read_body(self) -> dict[str, Any] | None:
        """The request's JSON object, or None when there was not one."""
        try:
            raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
            body = json.loads(raw or b"{}")
        except (ValueError, TypeError):
            return None
        return body if isinstance(body, dict) else None

    def send_json(self, code: int, payload: Any) -> Any:
        """Write one JSON response with its length. Returns the payload, for the caller."""
        blob = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(blob)))
        self.end_headers()
        self.wfile.write(blob)
        return payload

    def _opened(self, *, write: bool = False) -> AbstractContextManager[Any]:
        """The conversation store as a context manager, or one that yields None."""
        held = self.threads(write=write)
        return nullcontext(None) if held is None else held

    def _checked(self, body: Mapping[str, Any]) -> Ask | None:
        ask = Ask(body)
        if not ask.question:
            return None
        ask.turns = self.history(ask.thread, ask.sent, question=ask.question)
        return ask

    def _answer(self, ask: Ask, *, stream: bool, emit: Any) -> Any:
        self.asking = ask
        try:
            got = self.asker(ask.question, turns=ask.turns, held=ask.held, stream=stream,
                             emit=emit)
            if isinstance(got, Iterator):
                got = _drained(got, emit)
            return got
        finally:
            self.asking = None


def _drained(events: Iterator[Any], emit: Any) -> Any:
    """Relay what an iterator of events yields; the answer is what it returns.

    A generator that ``yield``s events and ``return``s the answer is the other natural
    shape for a streaming asker. One that returns nothing is taken at its last ``done``
    event, minus the ``event`` key.
    """
    last: Any = None
    while True:
        try:
            event = next(events)
        except StopIteration as stop:
            if stop.value is not None:
                return stop.value
            break
        if isinstance(event, Mapping) and event.get("event") == "done":
            last = {k: v for k, v in event.items() if k != "event"}
        elif emit is not None:
            emit(event)
    return last or {}
