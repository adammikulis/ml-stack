"""The routes the graph page talks to, for any ``http.server`` handler to mix in.

``graph.html`` POSTs a question to ``/ask/stream`` and reads the answer as it is found --
server-sent events, one frame per thing the model did, then a ``done`` frame carrying the
whole answer. A server without that route gets the same question at ``/ask`` and answers in
one JSON body. And ``GET /thread/<name>`` reopens a conversation the graph is holding, so
closing the tab does not end it and another machine can pick it up.

``GET /metrics`` and ``GET /metrics.prom`` are `ml_stack.graph.metrics`, mixed in here; the
routes beside the ask ones -- refresh, review, request, draft -- are
`ml_stack.graph.routes`.

``AskRoutes`` is the server-side half, with no opinion about where the graph
comes from or which model answers: a subclass says how a question is answered (``asker``)
and where conversations are kept (``threads``), and hangs whatever it wants -- a journal, a
review queue, a slot number -- off ``answered`` and ``failed``.

What goes back with a question is ``history``'s ``History``: the last ``WINDOW`` turns as
messages, always, and on it the latest ``summary`` and the earlier turns ``recalled`` for
this question, for ``converse(..., summary=, recalled=)``. A subclass that returns an
``embedder`` gets turns recalled by meaning as well as by their words; one that returns a
``summariser`` gets the summary rolled forward every ``summary_every`` turns.

::

    class Handler(AskRoutes, BaseHTTPRequestHandler):
        def asker(self, question, *, turns, highlighted, stream, emit):
            if stream:
                return converse_stream(question, graph, client, turns=turns,
                                       highlighted=highlighted,
                                       on_event=emit)
            return converse(question, graph, client, turns=turns, highlighted=highlighted)

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

``Handler`` is that subclass, once: ``GET /`` is a rendered page with ``window.GRAPH_LIVE``
set ahead of it, ``GET /export/<path>`` a file under an export root, and the ask routes
above. ``ml-stack-graph serve --site FILE`` binds one on loopback.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, nullcontext
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

from ml_stack.asking import ASKING
from ml_stack.graph.conversation import converse, converse_stream
from ml_stack.graph.metrics import MetricsRoutes
from ml_stack.graph.payloads import answer_payload, drained, sse, thread_request
from ml_stack.graph.questions import Ask, History
from ml_stack.graph.routes import (
    DraftRoutes,
    RefreshRoutes,
    RequestRoutes,
    ReviewRoutes,
)
from ml_stack.graph.store import GraphStore
from ml_stack.graph.thread import EVERY, WINDOW
from ml_stack.log import say, warn

__all__ = ["EXPORT_TYPES", "LIVE", "PORT", "AskRoutes", "Handler", "bind", "exported", "main"]

LIVE = b"<script>window.GRAPH_LIVE=1</script>"
"""What goes ahead of a served page and not a published one: the page asks a server it
finds this on for answers, and asks nothing otherwise."""


PORT = 8794
"""Where ``ml-stack-graph serve`` listens unless told otherwise."""

# what an export is sent as, by suffix; anything else is bytes
EXPORT_TYPES = {".json": "application/json", ".html": "text/html; charset=utf-8"}


class AskRoutes(MetricsRoutes):
    """``/ask``, ``/ask/stream`` and ``/thread/<name>`` for a ``BaseHTTPRequestHandler``.

    Mix it in ahead of the handler class and call the three ``handle_*`` methods from
    ``do_GET`` and ``do_POST``; each writes its own response and returns what it sent.
    Two things are the subclass's to say, and the rest has a default:

    ``config``
        The :class:`~ml_stack.serve.Config` this page answers with: the settings its model
        is served with, the asking, and the client. Given one, ``client_on_slot()``
        leases the server and hands out a slot of it, and ``model_name`` and
        ``serving_url`` answer from it.
    ``asker(question, *, turns, highlighted, stream, emit)``
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
    ``finder()``
        ``text -> hits``, or None. Given, it replaces what ``look_up`` calls: the model's
        searches go through it instead of `ask.look_up`, so a subclass holding a store and
        an embedder can hand the question to `search.hybrid` and have meaning vote beside
        the words. Each hit is a mapping with ``id``, ``label`` and ``kind``.
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
    config: Any = None

    # ------------------------------------------------------------- what a subclass says

    def asker(self, question: str, *, turns: list, highlighted: list, stream: bool,
              emit: Any) -> Any:
        raise NotImplementedError("a subclass says how a question is answered")

    def client_on_slot(self, *, index: int = 0, **over: Any) -> Any:
        """A client on one slot of ``config``'s server, leased on the first question.

        ``config`` is a :class:`~ml_stack.serve.Config`: the same object a bench row is measured
        from and `slot` elsewhere is given, so a page answer and a measurement of the
        page's model are one lease and one way of asking. llama.cpp serves one set of
        settings per port, and a page that spelled its lease out beside the bench's stopped the server
        and loaded the weights again the first time either was edited.

        ``over`` is `Config.over`'s: a knob for this slot, routed to the section that owns it.
        """
        if self.config is None:
            raise RuntimeError("no config on this handler: set `config`, or override "
                               "`client_on_slot`")
        from ml_stack.serve.serving import slot

        return slot(self.config.over(**over) if over else self.config, index=index)

    def threads(self, *, write: bool = False) -> AbstractContextManager[Any] | None:
        return None

    def ready(self) -> str | None:
        return None

    def model_name(self) -> str:
        """What is answering, for a reader who has not asked anything yet -- the served
        model's name, as the subclass knows it (a lease, a config, a probe of the server).
        ``config``'s model when there is one; every answer carries the name the server
        reported whether or not this can say."""
        if self.config is None:
            return ""
        return str(self.config.model).rsplit("/", 1)[-1]

    def serving_url(self) -> str:
        """The answering server's base URL, when the subclass knows it; ``/ask/model`` then
        also says how much context each slot holds and how many slots there are, which is
        what a peak in `spent` is measured against.

        With a ``config``, the server `slot` is holding on that config's port -- so this answers
        once a question has been asked and not before."""
        if self.config is None:
            return ""
        from ml_stack.serve.serving import servers

        return servers().get(self.config.port, "")

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

    def finder(self) -> Callable[[str], list[dict[str, Any]]] | None:
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
            with self._opened() as store:
                if store is None:
                    return None
                records = [(t.meta or {}).get("spent") for t in follow(store, str(thread)[:64],
                                                                     working=False)
                           if t.role == "assistant"]
        except Exception:  # noqa: BLE001 - a session that cannot be read is no session
            return None
        return Spent.totals(records)

    def failed(self, ask: Ask, exc: BaseException) -> None:
        warn(f"{time.strftime('%FT%T')} ask failed: {exc}")

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
            with self._opened() as store:
                turns = ([] if store is None
                         else [t.as_dict() for t in follow(store, name, working=working)])
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

            with self._opened() as store:
                if store is None:
                    return History(sent)
                kept = recent(store, thread, turns=keep)
                summary = latest_summary(store, thread)
                recalled = []
                if question and int(self.recalled_turns) > 0:
                    recalled = recall(store, thread, question, embedder=self.embedder(),
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
            with self._opened(write=True) as store:
                if store is None:
                    return
                remember_turn(store, thread=ask.thread, role="user", text=ask.question,
                              embedder=embed)
                remember_turn(store, thread=ask.thread, role="assistant",
                              text=str(payload.get("content") or ""), drew=drew_on(payload),
                              meta={"why": payload.get("why", ""), "steps": payload.get("steps", []),
                                    "model": payload.get("model", ""),
                                    "spent": payload.get("spent")},
                              embedder=embed)
                writer = self.summariser()
                if writer is not None:
                    summarise(store, ask.thread, writer, every=int(self.summary_every))
        except Exception as exc:  # noqa: BLE001
            warn(f"{time.strftime('%FT%T')} turn not remembered: {exc}")

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
        store = self.threads(write=write)
        return nullcontext(None) if store is None else store

    def _checked(self, body: Mapping[str, Any]) -> Ask | None:
        ask = Ask(body)
        if not ask.question:
            return None
        ask.turns = self.history(ask.thread, ask.sent, question=ask.question)
        return ask

    def _answer(self, ask: Ask, *, stream: bool, emit: Any) -> Any:
        self.asking = ask
        try:
            got = self.asker(ask.question, turns=ask.turns, highlighted=ask.highlighted,
                             stream=stream,
                             emit=emit)
            if isinstance(got, Iterator):
                got = drained(got, emit)
            return got
        finally:
            self.asking = None


class Handler(RefreshRoutes, ReviewRoutes, RequestRoutes, DraftRoutes, AskRoutes,
              BaseHTTPRequestHandler):
    """The page, its exports and every route the page's components talk to, on one
    ``http.server`` handler.

    Configured on the class, since ``http.server`` makes an instance per request:
    ``site`` is the page file ``GET /`` serves, ``export`` the root ``GET /export/<path>``
    reads under, ``graph`` what questions are answered over, ``store`` where conversations
    are kept, and ``config`` the model (`AskRoutes.config`); ``queue``, ``requests``, ``stages``
    and ``drafter`` are the review, request, refresh and draft routes' (each a 404 until
    set). :meth:`configured` makes a subclass with those set, so two servers in one process
    do not share them.
    """

    site: Path | None = None
    export: Path | None = None
    graph: Mapping[str, Any] | None = None
    store: Path | None = None

    @classmethod
    def configured(cls, *, site: Path | str | None = None, export: Path | str | None = None,
                   graph: Mapping[str, Any] | None = None, store: Path | str | None = None,
                   config: Any = None, queue: Any = None, requests: Path | str | None = None,
                   name: str = "Configured", **more: Any) -> type[Handler]:
        """A subclass of this handler with the given collaborators on it; ``more`` is any
        other attribute or method to set on it (``stages``, ``drafter``, ``proposed``)."""
        fields = {"site": Path(site) if site else None,
                  "export": Path(export).resolve() if export else None,
                  "graph": graph, "store": Path(store) if store else None, "config": config,
                  "queue": queue, "requests": Path(requests) if requests else None, **more}
        return type(name, (cls,), fields)

    # ------------------------------------------------------------- what AskRoutes asks

    def asker(self, question: str, *, turns: list, highlighted: list, stream: bool,
              emit: Any) -> Any:
        if self.graph is None:
            raise RuntimeError("no graph on this server: serve with --graph FILE")
        client = self.client_on_slot(index=0)
        asked = {"asking": self.config.asking if self.config is not None else ASKING,
                "turns": turns, "highlighted": highlighted, "finder": self.finder(),
                "summary": getattr(turns, "summary", None),
                "recalled": list(getattr(turns, "recalled", ()) or ())}
        if stream:
            return converse_stream(question, self.graph, client, on_event=emit, **asked)
        return converse(question, self.graph, client, **asked)

    def threads(self, *, write: bool = False) -> AbstractContextManager[Any] | None:
        if self.store is None:
            return None
        return GraphStore(self.store, read_only=not write)

    # ------------------------------------------------------------------- the routes

    def do_GET(self) -> None:
        path = urlsplit(self.path).path
        if path in ("/", "/index.html"):
            self.handle_page()
        elif path == "/ask/model":
            self.handle_model()
        elif path == "/metrics":
            self.handle_metrics()
        elif path == "/metrics.prom":
            self.handle_metrics_prom()
        elif path.startswith("/export/"):
            self.handle_export(path[len("/export/"):])
        elif path == "/review":
            self.handle_review_list()
        elif path == "/refresh":
            self.handle_refresh()
        elif (want := thread_request(self.path)):
            self.handle_thread(*want)
        else:
            self.send_error(404)

    def do_POST(self) -> None:
        path = urlsplit(self.path).path
        body = self.read_body()
        if path == "/ask":
            self.handle_ask(body or {})
        elif path == "/ask/stream":
            self.handle_ask_stream(body or {})
        elif path == "/review":
            self.handle_review_act(body or {})
        elif path == "/request":
            self.handle_request(body)
        elif path == "/draft":
            self.handle_draft(body or {})
        else:
            self.send_error(404)

    def handle_page(self) -> None:
        """``GET /``: the page file as a whole document, with `LIVE` ahead of it, never
        cached. `page.render` writes the page without a doctype or ``<html>`` so it can be
        published as a fragment; served, it gets both."""
        if self.site is None or not self.site.is_file():
            self.send_error(404)
            return
        page = self.site.read_bytes()
        blob = (b"<!doctype html><html><head><meta charset='utf-8'>" + LIVE + page
                + b"</html>")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(blob)))
        self.end_headers()
        self.wfile.write(blob)

    def handle_export(self, rest: str) -> None:
        """``GET /export/<path>``: one file under the export root, or a 404."""
        target = exported(self.export, rest)
        if target is None:
            self.send_error(404)
            return
        blob = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type",
                         EXPORT_TYPES.get(target.suffix.lower(), "application/octet-stream"))
        self.send_header("Content-Length", str(len(blob)))
        self.end_headers()
        self.wfile.write(blob)


def exported(root: Path | None, rest: str) -> Path | None:
    """The file ``rest`` names under ``root``, or None when it is not a file under it."""
    if root is None or not rest:
        return None
    target = (root / unquote(rest)).resolve()
    try:
        target.relative_to(root.resolve())
    except ValueError:
        return None
    return target if target.is_file() else None


# ---------------------------------------------------------------- the command


def parser() -> argparse.ArgumentParser:
    top = argparse.ArgumentParser(prog="ml-stack-graph",
                                  description="A rendered graph page, served with a model behind it.")
    subs = top.add_subparsers(dest="command", required=True)
    serve = subs.add_parser("serve", help="serve a rendered page on loopback",
                            description="GET / is the page with window.GRAPH_LIVE set; "
                                        "/export/<path> a file under --export; /ask, "
                                        "/ask/stream, /thread/<name> and /metrics answer "
                                        "with the model.")
    serve.add_argument("--site", required=True, type=Path,
                       help="the rendered page (page.render's output) GET / serves")
    serve.add_argument("--export", type=Path, help="a directory served under /export/")
    serve.add_argument("--port", type=int, default=PORT, help=f"listen here (default {PORT})")
    serve.add_argument("--model", default="",
                       help="the model that answers, as ml-stack-serve up names it "
                            "(a path or hf:owner/repo/file); leased on the first question")
    serve.add_argument("--draft", default="auto", metavar="HEAD",
                       help="the draft head that guesses tokens ahead for the model to check in one pass: 'auto' takes the smallest one on this machine, 'none' serves without one, or name a head shipped with the model (default: %(default)s)")
    serve.add_argument("--model-port", type=int, default=8080,
                       help="the port the model is served on (default 8080)")
    serve.add_argument("--graph", type=Path,
                       help="the graph questions are answered over, as JSON")
    serve.add_argument("--store", type=Path,
                       help="a GraphStore path conversations are kept in")
    placed = subs.add_parser(
        "geocode", help="give every entry that names a place a point, and optionally join "
                        "the nearest of them",
        description="Each distinct place in the graph's node attributes goes through "
                    "Nominatim, cached, and comes back as lat/lon on the node, which is "
                    "what the page's map draws.")
    placed.add_argument("--graph", required=True, type=Path, help="the graph, as JSON")
    placed.add_argument("--cache", required=True, type=Path,
                        help="the JSON cache of place -> point, asked before Nominatim is")
    placed.add_argument("--near", type=int, default=0, metavar="K",
                        help="join each placed entry to its K closest with a `near` edge, "
                             "weighted 1/(1+km)")
    placed.add_argument("--out", type=Path,
                        help="where the graph is written (--graph itself by default)")
    return top


def geocode(args: argparse.Namespace) -> int:
    """`graph.places.geocode` over a graph file, written back as JSON."""
    from ml_stack.graph.places import geocode as place
    from ml_stack.graph.places import points

    graph = json.loads(Path(args.graph).read_text(encoding="utf-8"))
    placed = place(graph, args.cache, near=int(args.near))
    out = Path(args.out or args.graph)
    out.write_text(json.dumps(placed, ensure_ascii=False, indent=2), encoding="utf-8")
    near = sum(1 for e in placed.get("edges") or () if str(e.get("rel") or "") == "near")
    say(f"{len(points(placed))} entr(ies) placed, {near} near edge(s) -> {out}")
    return 0


def bind(argv: Sequence[str] | None = None) -> ThreadingHTTPServer:
    """A server bound on loopback from the command line, not yet serving."""
    args = parser().parse_args(argv)
    config = None
    if args.model:
        from ml_stack.serve.serving import Config, Serving, drafted

        config = drafted(
            Config(serving=Serving(model=str(args.model), port=int(args.model_port))),
            str(getattr(args, "draft", "auto") or "auto"))
    graph = None
    if args.graph:
        graph = json.loads(Path(args.graph).read_text(encoding="utf-8"))
    handler = Handler.configured(site=args.site, export=args.export, graph=graph,
                                 store=args.store, config=config)
    return ThreadingHTTPServer(("127.0.0.1", int(args.port)), handler)


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "geocode":
        return geocode(args)
    httpd = bind(argv)
    host, port = httpd.server_address[:2]
    say(f"serving http://{host}:{port}", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
