"""What is measuring, what else holds the card, and what has been kept so far.

`status` is the whole answer -- the measurement, the servers, the queue and the runs kept
since it started; `tail` follows the log it is writing and `stop` ends it. The lines each
part says are `_status_line`, `serving_lines`, `results_since` and `beside_on_the_card`.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import signal
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

# The package is the namespace the tests and `selfcheck` patch -- `bench.home_dir()` -- so
# anything patchable is looked up there at call time, never bound here at import.
from ml_stack import bench
from ml_stack.bench.history import _epoch, _span
from ml_stack.bench.keep import note_beside
from ml_stack.bench.ops import newest
from ml_stack.bench.queue import queue_status
from ml_stack.bench.show import table
from ml_stack.bench.underway import _last_line, measuring, measuring_file
from ml_stack.log import say, warn
from ml_stack.serve.ops import processes
from ml_stack.serve.process import pid_exists
from ml_stack.serve.serving import SAMPLERS
from ml_stack.units import human_bytes

__all__ = ["beside_on_the_card", "note_beside_the_run", "results_since", "serving_lines",
           "status", "stop", "tail"]


def beside_on_the_card() -> list[dict[str, Any]]:
    """Every llama-server already running: its port, pid, model, memory, and whether a
    lease records it."""
    got = processes()
    return [{"port": int(one["port"]), "pid": int(one["pid"]),
             "model": Path(str(one.get("model") or "")).name,
             "bytes": int(one.get("rss") or 0),
             "leased": int(one["port"]) in got.leased}
            for one in got.found if not one.get("defunct")]


def note_beside_the_run() -> str:
    """Record what else holds the card while this run measures, and say it. The line said,
    or "" for a card this run has to itself."""
    found = beside_on_the_card()
    note_beside(found)
    if not found:
        return ""
    each = ", ".join(f":{one['port']} {one['model'] or '?'} {human_bytes(one['bytes'])}"
                     + ("" if one["leased"] else ", not leased") for one in found)
    said = (f"{len(found)} server(s) already hold this card: {each}. Their memory and "
            f"their work are in these timings, and in the run's record.")
    warn(said)
    return said


def serving_lines() -> list[str]:
    """One line per port a server is answering on -- what `ml-stack-serve status` knows,
    for `status` here, so what is measuring and what it is measuring against are read
    together and nobody polls ports by hand."""
    try:
        from ml_stack.serve.leases import lease_file, recorded_servers
        from ml_stack.serve.ops import look
    except Exception:  # noqa: BLE001 - no serving side installed is no servers
        return []
    try:
        records = recorded_servers(lease_file())
    except Exception:  # noqa: BLE001
        records = {}
    out = []
    for port in sorted(records):
        got = look(port, records)
        if got is None:
            continue
        served = (f"{got.context // 1024}k" if got.context else "?") + \
                 (f" x{got.slots}" if got.slots else "")
        out.append(f"  :{port}  {got.model or '?'}  {served}")
    return out


def results_since(started: str, kept: str | Path | None = None) -> str:
    """The table of every run kept since ``started`` -- what a job produced, without
    reading its log. Empty when nothing was kept."""
    where = Path(kept) if kept else bench.home_dir() / "runs.ladybug"
    if not started or not where.exists():
        return ""
    try:
        rows = newest(bench.runs(where), since=str(started)[:19])
    except Exception:  # noqa: BLE001 - a store that will not open has nothing to show
        return ""
    if not rows:
        return ""
    said = io.StringIO()
    with contextlib.redirect_stdout(said):
        table(rows)
    return said.getvalue().rstrip()


def status(*, results: bool = True) -> str:
    """What is measuring, or that nothing is; what is serving; and the rows the current or
    last job has kept so far. Exit 0 either way: a question, not a check."""
    text = _status_line()
    serving = serving_lines()
    text += "\nserving:\n" + "\n".join(serving) if serving else "\nserving: nothing"
    text += ("\n" + queued) if (queued := queue_status()) else ""
    if results:
        record = measuring()
        try:
            last = record or json.loads(measuring_file().read_text(encoding="utf-8"))
        except (OSError, ValueError):
            last = {}
        rows = results_since(str(last.get("started") or ""))
        if rows:
            text += "\nkept by it so far:\n" + rows if record else "\nkept by it:\n" + rows
    return text


def _sampling_said(sampling: Mapping[str, Any]) -> str:
    """The sampler settings as one phrase, temperature first; a zero is named greedy."""
    if not sampling:
        return "unrecorded"
    greedy = float(sampling.get("temperature", 1.0)) == 0.0
    order = [*SAMPLERS, *sorted(k for k in sampling if k not in SAMPLERS)]
    return ", ".join(f"{name} {sampling[name]}"
                     + (" (greedy)" if greedy and name == "temperature" else "")
                     for name in order if name in sampling)


def _how_said(how: Mapping[str, Any]) -> list[str]:
    """The `asking:` and `serving:` lines for a record's ``how``; nothing when it has none."""
    if not how:
        return []
    said = _sampling_said(how.get("sampling") or {})
    if how.get("card"):
        said += ", over what the model's card asks for"
    head, ahead = str(how.get("head") or ""), how.get("head_ahead")
    context, slots = int(how.get("context") or 0), int(how.get("slots") or 1)
    served = [f"draft head{'s' if ', ' in head else ''} {head}" if head else "no draft head"]
    if ahead:
        served.append(f"{ahead} ahead")
    served.append(f"{how.get('cache_type') or '?'} cache")
    if how.get("reasoning_budget") is not None:
        served.append(f"thinking budget {int(how['reasoning_budget'])}")
    served.append((f"{context // 1024}k context" if context else "the model's own context")
                  + f" across {slots} slot" + ("s" if slots != 1 else ""))
    return [f"  asking: {said}", "  serving: " + "; ".join(served)]


def _log_said(log: str) -> list[str]:
    """The `log:` and `last:` lines for a run's log, or where to read it when it has none."""
    if not log:
        return ["  log: none -- it prints to the terminal it was started in"]
    return [f"  log: {log}", f"  last: {_last_line(Path(log))}"]


def _status_line() -> str:
    record = measuring()
    if record is None:
        try:
            last = json.loads(measuring_file().read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return "nothing is measuring"
        return "\n".join([f"nothing is measuring; the last one -- ml-stack-bench "
                          f"{' '.join(last.get('argv') or ())} -- started "
                          f"{last.get('started', '?')} and has ended.",
                          *_log_said(str(last.get("log") or ""))])
    if record.get("lock_only"):
        return (f"measuring (pid {record['pid']}), which wrote no record of itself: the "
                f"measuring lock is held and that process is alive. Nothing here says what "
                f"it is asking or where its log is.")
    began = _epoch(str(record.get("started") or ""))
    return "\n".join([f"measuring {f'for {_span(time.time() - began)}, ' if began else ''}"
                      f"since {record.get('started', '?')} (pid {record.get('pid')}):",
                      f"  ml-stack-bench {' '.join(record.get('argv') or ())}",
                      *_how_said(record.get("how") or {}),
                      *_log_said(str(record.get("log") or ""))])


def _latest_log() -> Path | None:
    record = measuring()
    if record and record.get("log"):
        return Path(str(record["log"]))
    try:
        last = json.loads(measuring_file().read_text(encoding="utf-8"))
        if last.get("log") and Path(str(last["log"])).exists():
            return Path(str(last["log"]))
    except (OSError, ValueError):
        pass
    logs = sorted((bench.home_dir() / "logs").glob("*.log"), key=lambda p: p.stat().st_mtime) \
        if (bench.home_dir() / "logs").exists() else []
    return logs[-1] if logs else None


def tail(*, lines: int = 20, follow: bool = False, every: float = 0.5) -> int:
    """Print the end of the current (or latest) log; ``follow`` keeps printing until the
    measurement's pid has gone and the log has been drained."""
    log = _latest_log()
    if log is None or not log.exists():
        warn("no log yet: nothing has been detached")
        return 1
    with log.open("rb") as fh:
        text = fh.read().decode("utf-8", "replace")
        shown = text.splitlines()[-lines:] if lines > 0 else []
        if shown:
            say("\n".join(shown))
        if not follow:
            return 0
        record = measuring() or {}
        pid = record.get("pid")
        try:
            while True:
                more = fh.read().decode("utf-8", "replace")
                if more:
                    say(more, end="", flush=True)
                elif not pid_exists(pid):
                    break
                else:
                    time.sleep(every)
        except KeyboardInterrupt:
            pass
    return 0


def stop(*, wait: float = 60.0) -> str:
    """SIGTERM to the detached measurement -- by pid, never by name -- and wait for it.

    The child's handler turns the signal into a `SystemExit`, so the `serve` block it is
    in runs its exit and takes its model down; `pkill llama-server` would not, and would
    take somebody else's server with it.
    """
    record = measuring()
    if record is None:
        return "nothing is measuring"
    pid = int(record["pid"])
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return f"pid {pid} had already gone"
    began = time.monotonic()
    while pid_exists(pid) and time.monotonic() - began < wait:
        time.sleep(0.25)
    if pid_exists(pid):
        return (f"asked pid {pid} to stop; it is still running after {wait:.0f}s. Its log: "
                f"{record.get('log', '?')}")
    return f"stopped pid {pid} after {time.monotonic() - began:.1f}s; its log: {record.get('log', '?')}"
