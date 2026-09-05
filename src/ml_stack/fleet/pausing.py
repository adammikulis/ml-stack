"""One pause, sent to every machine in the cluster, and what each of them said.

There is no cluster-wide switch. `pause_fleet` asks each peer the same thing
``ml-stack-fleet`` asks one -- ``POST /availability`` -- so every machine still owns its
own answer. A machine that is off never receives the message; `seen` remembers the ones
this machine has met so `pause_table` can name it.
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ml_stack.home import expand

from .discovery import derive_token, memberships
from .remote import Peer, PeerError

__all__ = [
    "SEEN_FILE",
    "SEEN_FOR_S",
    "UNHEARD",
    "Answer",
    "Fanout",
    "minutes_of",
    "pause_fleet",
    "pause_table",
    "peer_clients",
    "remember_seen",
    "seen",
]

SEEN_FILE = "fleet-seen.json"
"""Under the daemon's root: every machine this one has met."""
SEEN_FOR_S = 7 * 86400
UNHEARD = "did not answer; it will take work when it comes back"
REACH_S = 15.0


@dataclass(frozen=True, slots=True)
class Answer:
    """What one machine did when the whole fleet was paused or resumed."""

    name: str
    ok: bool
    said: str
    is_self: bool = False

    def public(self) -> dict[str, Any]:
        return {"name": self.name, "ok": self.ok, "said": self.said,
                "is_self": self.is_self}


@dataclass(frozen=True, slots=True)
class Fanout:
    """One availability change and where it is sent from."""

    root: Path | str
    resume: bool = False
    minutes: float | None = None
    reason: str = ""
    cluster_key_path: Path | str | None = None


def peer_clients(rows: Sequence[dict[str, Any]], *,
                 cluster_key_path: Path | str | None = None,
                 timeout: float = 60.0) -> dict[str, Peer]:
    """A `remote.Peer` for each row a cluster key of this machine's reaches, by name."""
    tokens = {m.group: derive_token(m.key) for m in memberships(cluster_key_path)}
    out: dict[str, Peer] = {}
    for row in rows:
        token = next((tokens[g] for g in row.get("clusters") or [] if g in tokens), "")
        if token:
            out[str(row["name"])] = Peer(str(row["base_url"]), token, timeout=timeout)
    return out


def seen(root: Path | str) -> list[dict[str, Any]]:
    """Every machine met in the last week."""
    try:
        held = json.loads((expand(root) / SEEN_FILE).read_text())
    except (OSError, ValueError):
        return []
    fresh = time.time() - SEEN_FOR_S
    return [r for r in held if isinstance(r, dict) and r.get("name")
            and float(r.get("at") or 0.0) > fresh]


def remember_seen(rows: Sequence[dict[str, Any]], root: Path | str) -> Path:
    """Write down the machines that answered, keeping the ones met before."""
    now = time.time()
    kept = {str(r["name"]): {"name": str(r["name"]),
                             "base_url": str(r.get("base_url") or ""), "at": now}
            for r in rows}
    for old in seen(root):
        kept.setdefault(str(old["name"]), old)
    path = expand(root) / SEEN_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sorted(kept.values(), key=lambda r: r["name"]), indent=1))
    return path


def minutes_of(span: str) -> float | None:
    """``'2h 30m'`` as minutes; None when it is not a length of time."""
    from ml_stack.bench.history import parse_duration

    seconds = parse_duration(span) if span else None
    return seconds / 60 if seconds else None


def _until(said: dict[str, Any]) -> str:
    """What one machine's availability reply means, in a few words."""
    if not said.get("paused"):
        return "taking work again"
    when = said.get("paused_until")
    if not when:
        return "paused until it is resumed"
    stamp = time.localtime(float(when))
    shape = "%H:%M" if stamp.tm_yday == time.localtime().tm_yday else "%a %H:%M"
    return "paused until " + time.strftime(shape, stamp)


def pause_fleet(what: Fanout, rows: Sequence[dict[str, Any]]) -> list[Answer]:
    """Send one pause or resume to each machine, and what each said."""
    remember_seen(rows, what.root)
    clients = peer_clients(rows, cluster_key_path=what.cluster_key_path, timeout=REACH_S)
    action = "resume" if what.resume else "pause"
    fields: dict[str, Any] = {}
    if not what.resume:
        fields = {k: v for k, v in (("minutes", what.minutes), ("reason", what.reason))
                  if v}
    out: list[Answer] = []
    for row in rows:
        name, mine = str(row["name"]), bool(row.get("is_self"))
        peer = clients.get(name)
        if peer is None:
            out.append(Answer(name, False, "no cluster key reaches this machine", mine))
            continue
        try:
            out.append(Answer(name, True, _until(peer.availability(action, **fields)),
                              mine))
        except (PeerError, OSError, ValueError) as exc:
            told = str(exc)
            out.append(Answer(name, False,
                              UNHEARD if "unreachable" in told else told[-160:], mine))
    answered = {a.name for a in out}
    out.extend(Answer(str(old["name"]), False, UNHEARD)
               for old in seen(what.root) if str(old["name"]) not in answered)
    return out


def pause_table(answers: Sequence[Answer], *, resume: bool = False,
                span: str = "") -> str:
    """What the machines did, as text, naming every one that did not hear it."""
    reached = [a for a in answers if a.ok]
    missed = [a for a in answers if not a.ok]
    cells = {a.name: a.name + (" (this machine)" if a.is_self else "") for a in answers}
    wide = max((len(c) for c in cells.values()), default=4)
    lines = [f"resumed {len(reached)} of {len(answers)} machines" if resume
             else (f"paused {len(reached)} of {len(answers)} machines "
                   + (f"for {span}" if span else "until they are resumed"))]
    lines += [f"  {cells[a.name]:<{wide}}  {a.said}" for a in answers]
    if missed:
        which = "machine" if len(missed) == 1 else "machines"
        lines += ["",
                  f"{len(missed)} {which} did not hear this. A machine that is off or "
                  "asleep takes work",
                  "as soon as it is running again; run this command once it is back."]
    return "\n".join(lines)
