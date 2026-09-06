"""Which models this machine used last, and the picker every launcher shows.

`note` records a model as a command takes it and `recent` reads the record back, newest
first. `choices` is what a person may run something on -- the servers already up, then
every model file on this disk with its measured line, when it was last used, and the draft
heads that could serve with it. `pick` prints that numbered and reads one answer;
`pick_head` does the same for one model's heads.
"""

from __future__ import annotations

import contextlib
import re
import time
from collections.abc import Callable, Sequence
from pathlib import Path, PurePath

from ml_stack import home
from ml_stack.files import read_json, write_json
from ml_stack.hub import NO_HEAD, heads_for, held, pretty_name, weight_paths
from ml_stack.log import say

__all__ = ["LIMIT", "SHOWN", "choices", "head_line", "note", "pick", "pick_head",
           "recent", "recent_file", "used_said"]

LIMIT = 50
"""Models the record keeps, newest first."""

SHOWN = 12
"""Models the picker lists before saying how many more there are."""


def recent_file() -> Path:
    """The file recording which models this machine's commands took, and when."""
    return home.moved("recent.json")


def _key(model: str) -> str:
    return PurePath(str(model)).name.lower()


def recent(path: Path | None = None) -> list[dict]:
    """Every model noted, newest first, as ``{"model", "by", "at"}`` rows."""
    parsed = read_json(path or recent_file(), [])
    if not isinstance(parsed, list):
        return []
    rows = [row for row in parsed if isinstance(row, dict) and row.get("model")]
    return sorted(rows, key=lambda row: -float(row.get("at") or 0.0))


def note(model: str, *, by: str, path: Path | None = None,
         now: float | None = None) -> list[dict]:
    """Record that the command ``by`` took ``model``, and return the record.

    One row per model file, the newest kept, `LIMIT` rows at most.
    """
    where = path or recent_file()
    row = {"model": str(model), "by": str(by),
           "at": float(time.time() if now is None else now)}
    kept = [row, *(r for r in recent(where) if _key(str(r["model"])) != _key(str(model)))]
    kept = kept[:LIMIT]
    with contextlib.suppress(OSError):
        write_json(where, kept, indent=1)
    return kept


def used_said(at: float, *, now: float | None = None) -> str:
    """``used 20:41 today``, ``used yesterday``, ``used 2 days ago``."""
    now = time.time() if now is None else now
    then, today = time.localtime(at), time.localtime(now)
    if then.tm_year == today.tm_year and then.tm_yday == today.tm_yday:
        return f"used {time.strftime('%H:%M', then)} today"
    days = max(1, round((time.mktime((*today[:3], 0, 0, 0, *today[6:]))
                         - time.mktime((*then[:3], 0, 0, 0, *then[6:]))) / 86400))
    return "used yesterday" if days == 1 else f"used {days} days ago"


def choices(*, path: Path | None = None) -> list[dict]:
    """What a person could run something on: servers already up, then models on disk.

    A server already serving costs nothing to join. Models come most recently used first,
    then by measured F1, then by name; a shard after the first is not a choice. Each model
    carries the draft heads on this machine that could serve with it, smallest first.
    """
    from ml_stack.serve import process, profile

    out: list[dict] = []
    for one in _running(process.every_server):
        out.append({"kind": "server", "port": int(one["port"]),
                    "url": f"http://127.0.0.1:{one['port']}",
                    "name": pretty_name(str(one.get("model") or "")) or "a model",
                    "note": "already running"})

    files = weight_paths()
    where: dict[str, object] = {}
    for found in files:
        where.setdefault(found.name, found)
    used = {_key(str(row["model"])): float(row["at"]) for row in recent(path)}
    ranked = {one.model: one for one in profile.profiles()}
    for name in sorted(k for k in held() if k.endswith(".gguf")):
        if re.search(r"-0000[2-9]-of-", name):
            continue
        record = ranked.get(name) or profile.profile_for(name)
        when = used.get(_key(name))
        measured = (f"{record.right:.0%} F1 measured"
                    if record is not None and getattr(record, "right", None) else "")
        told = [one for one in (measured, used_said(when) if when else "") if one]
        out.append({"kind": "model", "name": name, "record": record, "used_at": when,
                    "heads": heads_for(where.get(name, name), files=files),
                    "note": " -- ".join(told)})
    out.sort(key=lambda c: (c["kind"] != "server", -(c.get("used_at") or 0.0),
                            -(getattr(c.get("record"), "right", 0) or 0), c["name"]))
    return out


def _running(every_server) -> list[dict]:
    """The servers up on this machine, and none where the machine will not say."""
    try:
        return [one for one in every_server() if not one.get("defunct")]
    except Exception:  # noqa: BLE001 - no psutil, or a machine that will not say
        return []


def head_line(one: dict) -> str:
    """What a model would guess ahead with, or that it would run without guessing."""
    heads = list(one.get("heads") or [])
    record = one.get("record")
    if record is not None:
        head = str(getattr(record, "draft", "") or "")
        if head:
            return f"drafts ahead with {PurePath(head).name}, from its measured record"
        idle = f", though {heads[0].said()} is on this machine" if heads else ""
        return f"its measured record serves it without a draft head{idle}"
    if not heads:
        return "no draft head on this machine: it runs without speculative decoding"
    more = f", or {len(heads) - 1} more" if len(heads) > 1 else ""
    return f"drafts ahead with {heads[0].said()}{more}"


def pick(options: list[dict], *, say: Callable[[str], None] = say,
         ask: Callable[[str], str] | None = None) -> dict | None:
    """One of ``options``, chosen by the person. None when they choose nothing."""
    if not options:
        say("nothing to run: no server is up and no .gguf is on this machine")
        return None
    running = [c for c in options if c["kind"] == "server"]
    if running:
        say("already running:")
        for n, c in enumerate(options, 1):
            if c["kind"] == "server":
                say(f"  {n:2}  {c['name']}  on {c['url']}  -- joined as it stands")
    say("on this disk, most recently used first:" if running
        else "models on this machine, most recently used first:")
    shown = [c for c in options if c["kind"] == "model"][:SHOWN]
    for c in shown:
        say(f"  {options.index(c) + 1:2}  {c['name']}"
            + (f"   {c['note']}" if c["note"] else ""))
        say(f"        {head_line(c)}")
    if len(options) - len(running) > len(shown):
        say(f"      ... and {len(options) - len(running) - len(shown)} more; "
            f"name one to skip this")
    reader = ask or input
    try:
        said = reader(f"which? [{1 if options else ''}] ").strip()
    except (EOFError, KeyboardInterrupt):
        say("")
        return None
    if not said:
        return options[0]
    if said.isdigit() and 1 <= int(said) <= len(options):
        return options[int(said) - 1]
    named = [c for c in options if said.lower() in c["name"].lower()]
    if len(named) == 1:
        return named[0]
    say(f"not a choice: {said!r}")
    return None


def pick_head(heads: Sequence, *, say: Callable[[str], None] = say,
              ask: Callable[[str], str] | None = None):
    """The draft head to serve with, chosen by the person. None for no head at all."""
    kept = list(heads)
    if not kept:
        say(NO_HEAD)
        return None
    say("draft heads on this machine for it, smallest first; the size is what the head "
        "adds to memory:")
    for n, one in enumerate(kept, 1):
        say(f"  {n:2}  {one.said()}")
    say(f"  {0:2}  none -- serve without speculative decoding, which is slower")
    reader = ask or input
    try:
        said = reader("which head? [1] ").strip()
    except (EOFError, KeyboardInterrupt):
        say("")
        return None
    if not said:
        return kept[0]
    if said == "0" or said.lower() == "none":
        return None
    if said.isdigit() and 1 <= int(said) <= len(kept):
        return kept[int(said) - 1]
    named = [one for one in kept if said.lower() in one.name.lower()]
    if len(named) == 1:
        return named[0]
    say(f"not a choice: {said!r}; serving with {kept[0].name}")
    return kept[0]
