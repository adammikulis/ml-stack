"""``GET /metrics`` and ``GET /metrics.prom``: what a server has spent, as JSON and in the
Prometheus text exposition format.

Every answer's `Spent` is kept in a ring on the concrete handler class as it goes out, and
the two routes are that ring. Nothing has to be running for them to answer -- a server that
has answered nothing reports zeros and its uptime, which is how a scraper tells a quiet
server from a missing one.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Mapping
from typing import Any

__all__ = ["KEEP_ANSWERS", "PROM", "STARTED", "MetricsRoutes", "metric_label",
           "metric_number", "prometheus"]

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

    One HELP and one TYPE line per metric, then the value, and ``model_info`` carrying the
    served model's name as a label. A missing total is 0 and not a missing line -- a scraper
    that loses a series cannot tell a quiet server from a broken one.
    """
    lines: list[str] = []
    for name, key, kind, said in PROM:
        value = totals.get(key) or 0
        lines += [f"# HELP {name} {said}", f"# TYPE {name} {kind}",
                  f"{name} {metric_number(value)}"]
    if uptime is not None:
        lines += ["# HELP uptime_seconds How long this process has been up.",
                  "# TYPE uptime_seconds gauge", f"uptime_seconds {metric_number(uptime)}"]
    lines += ["# HELP model_info The served model, as a label; the value is always 1.",
              "# TYPE model_info gauge",
              f'model_info{{model="{metric_label(model)}"}} 1']
    return "\n".join(lines) + "\n"


def metric_number(value: Any) -> str:
    """A metric value: an int stays an int, everything else is a float a scraper parses."""
    if isinstance(value, bool) or value is None:
        return "0"
    if isinstance(value, int):
        return str(value)
    try:
        return repr(round(float(value), 3))
    except (TypeError, ValueError):
        return "0"


def metric_label(text: Any) -> str:
    """A label value, escaped the way the exposition format asks: backslash, quote, newline."""
    return (str(text or "").replace("\\", "\\\\").replace('"', '\\"')
            .replace("\n", "\\n"))


class MetricsRoutes:
    """``/metrics`` and ``/metrics.prom`` for a handler that also answers questions.

    ``keep_answers`` is how many answers the ring holds. The ring is per concrete handler
    class, made on first use, and every answer is recorded into it by the ask routes
    themselves. ``model_name``, ``serving_url`` and ``ready`` are the answering half's.
    """

    keep_answers: int = KEEP_ANSWERS

    @classmethod
    def _kept(cls) -> dict[str, Any]:
        """This handler class's telemetry: when it started, how many answers, and a ring.

        On the class, because ``http.server`` builds a handler per request and an instance
        remembers nothing. On the *concrete* class and not on `MetricsRoutes`, so two
        servers in one process do not add up together.
        """
        counters = cls.__dict__.get("_telemetry")
        if counters is None:
            counters = {"started": time.time(), "answers": 0,
                    "ring": deque(maxlen=max(1, int(cls.keep_answers)))}
            cls._telemetry = counters           # on this class, not on a base of it
        return counters

    def record(self, ask: Any, payload: Mapping[str, Any]) -> None:
        """Note one answered question in the ring. Never raises: telemetry is not the answer.

        What is kept is the answer's `Spent.public()` with the thread and the clock on it,
        which is exactly what `Spent.totals` reads. An answer no model was asked for -- a
        greeting, one that came back from the cache -- is counted, not kept.
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
        `Spent.totals` over that ring and ``threads`` the same totals per conversation.
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

        Nothing has to know anything about this server to watch it: a scraper already
        installed picks up answers, calls, tokens read against tokens cached, draft
        acceptance and the context peak.
        """
        got = self.metrics()
        body = prometheus(got["totals"], model=got["model"], uptime=got["uptime"])
        return self.send_text(200, body, "text/plain; version=0.0.4; charset=utf-8")
