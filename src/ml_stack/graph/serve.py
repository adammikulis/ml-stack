"""The routes the graph page talks to, for any ``http.server`` handler to mix in.

``graph.html`` POSTs a question to ``/ask/stream`` and reads the answer as it is found --
server-sent events, one frame per thing the model did, then a ``done`` frame carrying the
whole answer. A server without that route gets the same question at ``/ask`` and answers in
one JSON body. And ``GET /thread/<name>`` reopens a conversation the graph is holding, so
closing the tab does not end it and another machine can pick it up.

``GET /metrics`` and ``GET /metrics.prom`` are the same server saying what it has spent:
every answer's `Spent` is kept in a ring on the handler class as it goes out, and the two
routes are that ring as JSON and as Prometheus text. Nothing has to be running for them to
answer -- a server that has answered nothing reports zeros and its uptime, which is how a
scraper tells a quiet server from a missing one.

Those routes were the same in every project that rendered the page, and lived in none
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

        def do_GET(self):
            if self.path == "/metrics":
                self.handle_metrics()
            elif self.path == "/metrics.prom":
                self.handle_metrics_prom()

Nothing here decides who may ask. A page served over a tunnel and one on loopback get the
same routes; refusing one is the subclass's policy, checked before these are called.
"""

from __future__ import annotations

import json
import sys
import time
from collections import deque
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, nullcontext
from typing import Any

from ml_stack.graph.thread import EVERY, WINDOW

__all__ = ["Ask", "AskRoutes", "History", "KEEP_ANSWERS", "STARTED", "answer_payload",
           "prometheus", "sse", "thread_request"]

STARTED = time.time()
"""When this process began, near enough: the moment this module was imported. ``/metrics``
reports uptime against it, which is what tells a scraper that a server was restarted
between two scrapes rather than having answered nothing."""

KEEP_ANSWERS = 200
"""How many answers the metrics ring holds. A ring and not a log: a server that answers for
a week must not grow for a week, and what a scraper wants is the recent shape, not the
history -- the history is in the conversation store, which is on disk and is not this."""

# What `/metrics.prom` exposes, in order: the name a scraper sees, the key on
# `Spent.totals`, whether it only ever goes up, and the line that says what it is.
PROM = (
    ("answers_total", "answers", "counter", "Answers given since this process started."),
    ("calls_total", "calls", "counter", "Calls made to the model server."),
    ("tokens_read_total", "read_tokens", "counter",
     "Prompt tokens the server actually read (timings.prompt_n)."),
    ("tokens_cached_total", "cached_tokens", "counter",
     "Prompt tokens it kept from the call before (timings.cache_n)."),
    ("tokens_written_total", "completion_tokens", "counter", "Tokens generated."),
    ("draft_tokens_total", "draft_tokens", "counter", "Tokens guessed ahead by a draft head."),
    ("draft_accepted_total", "draft_taken", "counter", "And accepted by the large model."),
    ("seconds_total", "seconds", "counter", "Wall clock spent answering, on this side."),
    ("context_peak", "context_peak", "gauge",
     "The most one slot held: prompt plus answer, the largest of any single call."),
)


def prometheus(totals: Mapping[str, Any], *, model: str = "", uptime: float | None = None,
               ) -> str:
    """`Spent.totals` in the Prometheus text exposition format, ready to be scraped.

    One HELP and one TYPE line per metric, then the value: counters for everything that
    only goes up, gauges for the peak and the uptime, and ``model_info`` carrying the
    served model's name as a label, which is how a name is exposed to a system that only
    stores numbers. A missing total is 0 and not a missing line -- a scraper that loses a
    series cannot tell a quiet server from a broken one.
    """
    lines: list[str] = []
    for name, key, kind, said in PROM:
        value = totals.get(key) or 0
        lines += [f"# HELP {name} {said}", f"# TYPE {name} {kind}",
                  f"{name} {_number(value)}"]
    if uptime is not None:
        lines += ["# HELP uptime_seconds How long this process has been up.",
                  "# TYPE uptime_seconds gauge", f"uptime_seconds {_number(uptime)}"]
    lines += ["# HELP model_info The served model, as a label; the value is always 1.",
              "# TYPE model_info gauge",
              f'model_info{{model="{_label(model)}"}} 1']
    return "\n".join(lines) + "\n"


def _number(value: Any) -> str:
    """A metric value: an int stays an int, everything else is a float a scraper parses."""
    if isinstance(value, bool) or value is None:
        return "0"
    if isinstance(value, int):
        return str(value)
    try:
        return repr(round(float(value), 3))
    except (TypeError, ValueError):
        return "0"


def _label(text: Any) -> str:
    """A label value, escaped the way the exposition format asks: backslash, quote, newline."""
    return (str(text or "").replace("\\", "\\\\").replace('"', '\\"')
            .replace("\n", "\\n"))

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

    ``run``
        The :class:`~ml_stack.serve.Run` this page answers with: the shape its model is
        served in, the ways it is asked, and the client. Given one, ``seated()`` leases the
        server and hands out a seat of it, and ``model_name`` and ``serving_url`` answer
        from it.
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
    ``keep_answers``
        How many answers `metrics` keeps. The ring is per concrete handler class, made on
        first use, and every answer is recorded into it by the ask routes themselves -- a
        subclass that overrides ``answered`` without calling ``super()`` still counts.
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
    run: Any = None

    # ------------------------------------------------------------- what a subclass says

    def asker(self, question: str, *, turns: list, held: list, stream: bool,
              emit: Any) -> Any:
        raise NotImplementedError("a subclass says how a question is answered")

    def seated(self, *, index: int = 0, **over: Any) -> Any:
        """A client on one seat of ``run``'s server, leased on the first question.

        ``run`` is a :class:`~ml_stack.serve.Run`: the same object a bench row is measured
        from and `seat` elsewhere is given, so a page answer and a measurement of the
        page's model are one lease and one way of asking. llama.cpp serves one shape per
        port, and a page that spelled its lease out beside the bench's stopped the server
        and loaded the weights again the first time either was edited.

        ``over`` is `Run.over`'s: a knob for this seat, routed to the section that owns it.
        """
        if self.run is None:
            raise RuntimeError("no run on this handler: set `run`, or override `seated`")
        from ml_stack.serve.shape import seat

        return seat(self.run.over(**over) if over else self.run, index=index)

    def threads(self, *, write: bool = False) -> AbstractContextManager[Any] | None:
        return None

    def ready(self) -> str | None:
        return None

    def model_name(self) -> str:
        """What is answering, for a reader who has not asked anything yet -- the served
        model's name, as the subclass knows it (a lease, a config, a probe of the server).
        ``run``'s model when there is one; every answer carries the name the server
        reported whether or not this can say."""
        if self.run is None:
            return ""
        return str(self.run.model).rsplit("/", 1)[-1]

    def serving_url(self) -> str:
        """The answering server's base URL, when the subclass knows it; ``/ask/model`` then
        also says how much context each slot holds and how many slots there are, which is
        what a peak in `spent` is measured against.

        With a ``run``, the server `seat` is holding on that run's port -- so this answers
        once a question has been asked and not before."""
        if self.run is None:
            return ""
        from ml_stack.serve.shape import held

        return held().get(self.run.port, "")

    def handle_model(self) -> dict[str, Any]:
        """``GET /ask/model``: ``{"model": name, "slot_context": n, "slots": n}`` -- the
        page shows the name in the ask pane and the slot beside each answer's peak."""
        payload: dict[str, Any] = {"model": self.model_name() or "", "ready": self.ready()}
        where = self.serving_url()
        if where:
            from ml_stack.client.health import serving_params

            got = serving_params(where, timeout=1.0)
            if got is not None:
                payload["slot_context"] = got.n_ctx
                payload["slots"] = got.total_slots
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

    # ------------------------------------------------------------ what it has spent

    keep_answers: int = KEEP_ANSWERS

    @classmethod
    def _kept(cls) -> dict[str, Any]:
        """This handler class's telemetry: when it started, how many answers, and a ring.

        On the class, because ``http.server`` builds a handler per request and an instance
        remembers nothing. On the *concrete* class and not on `AskRoutes`, so two servers
        in one process -- a test's, and the one it is testing -- do not add up together.
        """
        held = cls.__dict__.get("_telemetry")
        if held is None:
            held = {"started": time.time(), "answers": 0,
                    "ring": deque(maxlen=max(1, int(cls.keep_answers)))}
            cls._telemetry = held               # on this class, not on a base of it
        return held

    def record(self, ask: Ask | None, payload: Mapping[str, Any]) -> None:
        """Note one answered question in the ring. Never raises: telemetry is not the answer.

        What is kept is the answer's `Spent.public()` with the thread and the clock on it,
        which is exactly what `Spent.totals` reads -- so the totals over a ring, over a
        thread's slice of it, and over a conversation in the store are all the same
        function over the same records. An answer no model was asked for -- a greeting, one
        that came back from the cache -- has nothing to add and is counted, not kept.
        """
        try:
            kept = self._kept()
            kept["answers"] += 1
            spent = payload.get("spent")
            if isinstance(spent, Mapping) and spent.get("calls"):
                kept["ring"].append({**spent, "at": round(time.time(), 3),
                                     "thread": (ask.thread if ask else "")})
        except Exception:  # noqa: BLE001 - a metric is never worth an answer
            pass

    def metrics(self) -> dict[str, Any]:
        """The whole of this process's telemetry, as `/metrics` sends it.

        ``answers`` counts every answer since the process started; ``recent`` is the last
        `keep_answers` of them as `Spent.public()` records, newest last; ``totals`` is
        `Spent.totals` over that ring and ``threads`` the same totals per conversation, so
        a reader can see which thread is spending the tokens without opening the store.
        """
        from ml_stack.client.spent import Spent

        kept = self._kept()
        recent = list(kept["ring"])
        threads: dict[str, Any] = {}
        for name in dict.fromkeys(str(one.get("thread") or "") for one in recent):
            if name:
                threads[name] = Spent.totals([one for one in recent
                                              if str(one.get("thread") or "") == name])
        return {"answers": int(kept["answers"]),
                "kept": len(recent),
                "uptime": round(time.time() - STARTED, 1),
                "started": round(kept["started"], 3),
                "model": self.model_name() or (recent[-1].get("model") if recent else "") or "",
                "url": self.serving_url(),
                "ready": self.ready(),
                "totals": Spent.totals(recent),
                "threads": threads,
                "recent": recent}

    def handle_metrics(self) -> dict[str, Any]:
        """``GET /metrics``: this process's telemetry as one JSON body."""
        return self.send_json(200, self.metrics())

    def handle_metrics_prom(self) -> str:
        """``GET /metrics.prom``: the same numbers in the Prometheus text exposition format.

        The point of the second route is that nothing has to know anything about this
        server to watch it: a scraper already installed picks up answers, calls, tokens
        read against tokens cached, draft acceptance and the context peak, and the peak is
        the number that says how many more users this machine holds.
        """
        got = self.metrics()
        body = prometheus(got["totals"], model=got["model"], uptime=got["uptime"])
        return self.send_text(200, body, "text/plain; version=0.0.4; charset=utf-8")

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
            self.record(ask, payload)
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
            self.record(ask, payload)
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

    def send_text(self, code: int, text: str, content_type: str = "text/plain; charset=utf-8",
                  ) -> str:
        """Write one text response with its length. Returns the text, for the caller."""
        blob = text.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(blob)))
        self.end_headers()
        self.wfile.write(blob)
        return text

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
