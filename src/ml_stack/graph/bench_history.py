"""What the bench spent, day by day, read out of its logs: ``ml-stack-bench history``.

`measuring.json` knows the run that is going now and nothing remembers the day: which
commands ran, when, for how long, on which commit, how each ended, and the estimate beside
the actual. "How much GPU time did you waste" was answered by opening logs one by one.

Every detached measurement writes its log under ``HOME / "logs"`` as
``<subcommand>-<label-or-model>-<YYYYmmddTHHMMSS>.log``, so the directory is the record:
one `Entry` per file. What the log's first lines say (``argv:``, ``started:``, ``commit:``,
with or without a leading ``#``) is preferred; the filename's stamp and `measuring.json`
fill in what they can when the header is absent. A log is *running* when `measuring.json`
points at it and the pid it names is alive; *killed* when it carries ``[killed]``; *crashed*
when it carries a traceback, with the exception line; *done* when it asked questions or
kept a run and none of that happened; *unknown* when it says nothing at all. The runs a log
kept are the ones in the store whose ``at`` falls between its start and its end.

    python -m ml_stack.graph.bench_history [--home PATH] [--kept PATH] [--since WHEN] [--json]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

STAMP = "%Y%m%dT%H%M%S"                    # the log's filename
ISO = "%FT%T"                              # `measuring.json`, the header, a run's `at`
HEADER_LINES = 12                          # how far into a log a header is looked for

_HEADER = re.compile(r"^\s*#?\s*(argv|started|commit)\s*:\s*(.*?)\s*$")
_ESTIMATE = re.compile(r"^\s*#?\s*estimate\s*:\s*(.*?)\s*$", re.IGNORECASE)
_DURATION = re.compile(r"(\d+(?:\.\d+)?)\s*(h|hr|hrs|hours?|m|min|mins|minutes?|s|sec|secs|seconds?)\b")
_QUESTION = re.compile(r"^\s*\d+(?:\.\d+)?s\s+\d+\s+calls\b")
_LOG_NAME = re.compile(r"^(?P<sub>[^-]+)-(?P<name>.+)-(?P<stamp>\d{8}T\d{6})$")
_TRACEBACK = "Traceback (most recent call last):"
_KILLED = "[killed]"


@dataclass
class Entry:
    """One log under the bench's home: what ran, when, how long, and what it left."""
    log: str
    started: str                    # ISO, local time
    subcommand: str
    name: str                       # the label or model the log is named after
    argv: list[str] = field(default_factory=list)
    commit: str = ""                # empty until something records it
    ended: str = ""
    seconds: float = 0.0
    exit: str = "unknown"           # done | killed | crashed: <line> | running | unknown
    estimate_s: float | None = None
    kept: list[str] = field(default_factory=list)
    questions: int = 0
    model: str = ""                 # `server.model` of the first kept run, if any


def _now() -> float:
    return time.time()


def _epoch(iso: str) -> float | None:
    for shape in (ISO, "%Y-%m-%dT%H:%M", "%Y-%m-%d"):
        try:
            return time.mktime(time.strptime(iso, shape))
        except ValueError:
            continue
    return None


def _iso(epoch: float) -> str:
    return time.strftime(ISO, time.localtime(epoch))


def parse_duration(text: str) -> float | None:
    """``2h 15m``, ``90s``, ``1.5h``, ``~600`` -> seconds; a bare number is seconds."""
    total, found = 0.0, False
    for amount, unit in _DURATION.findall(text):
        found = True
        total += float(amount) * {"h": 3600.0, "m": 60.0, "s": 1.0}[unit[0]]
    if found:
        return total
    bare = re.search(r"\d+(?:\.\d+)?", text)
    return float(bare.group()) if bare else None


def _measuring(home: Path) -> dict[str, Any]:
    try:
        held = json.loads((home / "measuring.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return held if isinstance(held, dict) else {}


def _kept_runs(kept: Path | None) -> list[dict[str, Any]]:
    if kept is None or not kept.exists():
        return []
    from ml_stack.graph.bench import runs

    try:
        return runs(kept)
    except Exception:                        # a store that will not open is not history
        return []


def _read(log: Path, held: Mapping[str, Any], kept: Sequence[Mapping[str, Any]], *,
          now: float, alive: Callable[[Any], bool]) -> Entry:
    named = _LOG_NAME.match(log.stem)
    sub = named.group("sub") if named else log.stem
    name = named.group("name") if named else ""
    stamp = time.mktime(time.strptime(named.group("stamp"), STAMP)) if named else None

    try:
        text = log.read_text(encoding="utf-8", errors="replace")
    except OSError:
        text = ""
    lines = text.splitlines()

    header: dict[str, str] = {}
    for line in lines[:HEADER_LINES]:
        got = _HEADER.match(line)
        if got and got.group(1) not in header:
            header[got.group(1)] = got.group(2)

    mine = str(held.get("log") or "")
    this_one = bool(mine) and (Path(mine).name == log.name)
    argv = header["argv"].split() if "argv" in header else \
        [str(a) for a in (held.get("argv") or [])] if this_one else []
    commit = header.get("commit", "") or (str(held.get("commit") or "") if this_one else "")

    began = _epoch(header["started"]) if "started" in header else None
    if began is None and this_one and held.get("started"):
        began = _epoch(str(held["started"]))
    if began is None:
        began = stamp
    if began is None:
        try:
            began = log.stat().st_mtime
        except OSError:
            began = now

    running = this_one and alive(held.get("pid"))
    try:
        ended = now if running else log.stat().st_mtime
    except OSError:
        ended = now
    ended = max(ended, began)

    estimate = None
    for line in lines:
        got = _ESTIMATE.match(line)
        if got:
            estimate = parse_duration(got.group(1))
    questions = sum(1 for line in lines if _QUESTION.match(line))

    if running:
        exit = "running"
    elif any(line.strip().startswith(_KILLED) for line in lines):
        exit = "killed"
    elif any(line.startswith(_TRACEBACK) for line in lines):
        last = max(i for i, line in enumerate(lines) if line.startswith(_TRACEBACK))
        why = next((line.strip() for line in reversed(lines[last + 1:]) if line.strip()), "")
        exit = f"crashed: {why}" if why else "crashed"
    else:
        exit = "unknown"                     # settled below, once `kept` is known

    window = (_iso(began), _iso(ended))
    inside = [r for r in kept if window[0] <= str(r.get("at") or "") <= window[1]]
    labels = [str(r.get("label") or r.get("key") or "") for r in inside]
    model = ""
    for r in inside:
        server = r.get("server") or {}
        if isinstance(server, dict) and server.get("model"):
            model = str(server["model"]).rsplit("/", 1)[-1]
            break
    if exit == "unknown" and (questions or inside):
        exit = "done"

    return Entry(log=str(log), started=_iso(began), subcommand=sub, name=name, argv=argv,
                 commit=commit, ended=_iso(ended), seconds=round(ended - began, 1),
                 exit=exit, estimate_s=estimate, kept=labels, questions=questions,
                 model=model)


def history(home: str | Path, kept: str | Path | None = None, *,
            now: float | None = None, alive: Callable[[Any], bool] | None = None) -> list[Entry]:
    """Every log under ``home / "logs"``, oldest first, each joined to the runs it kept.

    ``kept`` is the runs store (``home / "runs.ladybug"`` unless given); ``now`` and
    ``alive`` are for tests, which own neither the clock nor the pid table.
    """
    home = Path(home).expanduser()
    logs = home / "logs"
    if not logs.is_dir():
        return []
    if alive is None:
        from ml_stack.serve.process import pid_exists
        alive = pid_exists
    at = _now() if now is None else now
    held = _measuring(home)
    runs = _kept_runs(Path(kept).expanduser() if kept else home / "runs.ladybug")
    found = [_read(log, held, runs, now=at, alive=alive) for log in logs.glob("*.log")]
    return sorted(found, key=lambda e: (e.started, e.log))


def since(when: str, *, now: float | None = None) -> float:
    """``today`` (local midnight), ``24h`` / ``7d`` / ``90m`` (ago), or a date -> epoch."""
    at = _now() if now is None else now
    word = when.strip().lower()
    if word == "today":
        day = time.localtime(at)
        return time.mktime((day.tm_year, day.tm_mon, day.tm_mday, 0, 0, 0, 0, 0, -1))
    ago = re.fullmatch(r"(\d+(?:\.\d+)?)\s*(h|d|m)", word)
    if ago:
        return at - float(ago.group(1)) * {"h": 3600.0, "d": 86400.0, "m": 60.0}[ago.group(2)]
    epoch = _epoch(when.strip())
    if epoch is None:
        raise ValueError(f"--since takes today, 24h, 7d or a date, not {when!r}")
    return epoch


def _span(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    whole = int(round(seconds))
    if whole < 60:
        return f"{whole}s"
    if whole < 3600:
        return f"{whole // 60}m"
    return f"{whole // 3600}h{(whole % 3600) // 60:02d}m"


HEADER = f"{'started':19}  {'sub':10}  {'model/label':30}  {'est':>7}  {'actual':>7}  {'exit':10}  kept"


def table(entries: Sequence[Entry]) -> str:
    """The entries newest last, then a line of totals: runs, GPU hours, and the hours that
    produced no kept run -- the wasted number -- with what is still running left out of it."""
    lines = [HEADER]
    for e in entries:
        shown = e.name or e.model or "-"
        lines.append(f"{e.started:19}  {e.subcommand:10}  {shown[:30]:30}  "
                     f"{_span(e.estimate_s):>7}  {_span(e.seconds):>7}  {e.exit:10}  "
                     f"{', '.join(e.kept) or '-'}")
    total = sum(e.seconds for e in entries)
    wasted = sum(e.seconds for e in entries if not e.kept and e.exit != "running")
    running = sum(1 for e in entries if e.exit == "running")
    lines.append(f"{len(entries)} runs, {total / 3600:.1f} GPU hours, "
                 f"{wasted / 3600:.1f} hours produced no kept run (wasted)"
                 + (f"; {running} still running, not counted as wasted" if running else ""))
    return "\n".join(lines)


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="ml-stack-bench history", allow_abbrev=False,
                                 description="Every measurement the bench's logs remember: "
                                             "when, how long, how it ended, the estimate "
                                             "beside the actual, and the runs it kept.")
    ap.add_argument("--home", default=None,
                    help="the bench's home (default: ~/.ml-stack/bench); its logs/ and "
                         "measuring.json are read")
    ap.add_argument("--kept", default=None,
                    help="the runs store (default: HOME/runs.ladybug)")
    ap.add_argument("--since", default=None, metavar="WHEN",
                    help="only logs started since: today, 24h, 7d, or a date")
    ap.add_argument("--json", action="store_true", help="the entries as JSON, not a table")
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(list(argv) if argv is not None else None)
    if args.home is None:
        from ml_stack.graph.bench import HOME
        home = HOME
    else:
        home = Path(args.home).expanduser()
    entries = history(home, args.kept)
    if args.since:
        try:
            floor = _iso(since(args.since))
        except ValueError as why:
            print(f"error: {why}", file=sys.stderr)
            return 2
        entries = [e for e in entries if e.started >= floor]
    if args.json:
        print(json.dumps([asdict(e) for e in entries], indent=1))
    else:
        print(table(entries))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
