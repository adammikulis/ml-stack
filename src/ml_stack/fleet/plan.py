"""Which model each peer serves, and how many seats, for a number of users at one context."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Sequence

from ml_stack.serve.fit import Fit
from ml_stack.serve.profile import Profile, family_of, quant_of

__all__ = ["PREFERENCES", "Placement", "Room", "Row", "fit_for", "place", "ranked",
           "room_of", "table"]

RANKED_AFTER = 20
"""A profile with fewer questions than this ranks after every one with more."""

PREFERENCES = {
    "quality": "the best measured model that fits each peer",
    "seats": "the model that seats the most users on each peer",
}
"""What each ``--prefer`` choice gives a peer."""


@dataclass(frozen=True)
class Room:
    """A peer as the planner sees it: a name, its room in bytes, and where it answers."""

    name: str
    room: int
    base_url: str = ""
    serving: tuple[str, ...] = ()


def room_of(peer: Any) -> Room:
    """A `Room` out of a `Beacon`, a `join.describe` row, or a ``/health`` body."""
    if isinstance(peer, Room):
        return peer
    if isinstance(peer, dict):
        device = peer.get("device") if isinstance(peer.get("device"), dict) else peer
        name = str(peer.get("name") or "")
        base_url = str(peer.get("base_url") or "")
    else:
        device = getattr(peer, "device", None) or {}
        name = str(getattr(peer, "name", "") or "")
        base_url = str(getattr(peer, "base_url", "") or "")
    served = tuple(str(m) for one in (device.get("serving") or [])
                   for m in (one.get("models") or []))
    return Room(name=name, room=int(device.get("room_bytes") or 0), base_url=base_url,
                serving=served)


def ranked(profiles: Sequence[Profile]) -> list[Profile]:
    """Profiles best first: F1 down, then seconds per question up. One measured on fewer
    than ``RANKED_AFTER`` questions comes after every one measured on more."""
    measured = sorted((p for p in profiles if p.questions >= RANKED_AFTER),
                      key=lambda p: (-p.right, p.seconds_per_question, p.model.lower()))
    rest = sorted((p for p in profiles if p.questions < RANKED_AFTER),
                  key=lambda p: (-p.right, p.seconds_per_question, p.model.lower()))
    return measured + rest


def _plain(name: str) -> str:
    return str(name or "").replace("\\", "/").rsplit("/", 1)[-1].strip().lower()


def fit_for(model: str, fits: Sequence[Fit], *, cache_type: str = "",
            spec: str = "") -> Fit | None:
    """The memory record for ``model``: by file name, then family and quantisation, then
    family. Among several, the one measured at ``cache_type`` and ``spec`` first."""
    asked = _plain(model)
    same = [f for f in fits if _plain(f.model) == asked]
    if not same:
        family, quant = family_of(model).lower(), quant_of(model).lower()
        if family:
            same = [f for f in fits if family_of(f.model).lower() == family
                    and quant_of(f.model).lower() == quant]
            if not same:
                same = [f for f in fits if family_of(f.model).lower() == family]
    if not same:
        return None
    wanted_cache = (cache_type or "f16").lower()
    wanted_spec = (spec or "").lower()

    def order(f: Fit) -> tuple[int, int, str]:
        return (f.cache_type.lower() != wanted_cache, f.spec.lower() != wanted_spec,
                f.model.lower())

    return min(same, key=order)


@dataclass(frozen=True)
class Row:
    """One peer, the model it serves, and how many seats at what context."""

    peer: str
    model: str
    seats: int
    context: int
    used: int
    room: int
    base_url: str = ""


@dataclass
class Placement:
    """Every seat given out, the users left without one, and every peer a model was not
    placed on with why not."""

    users: int
    context: int
    prefer: str = "quality"
    rows: list[Row] = field(default_factory=list)
    unplaced: int = 0
    why: list[tuple[str, str, str]] = field(default_factory=list)
    """``(peer, model, reason)``."""

    @property
    def seated(self) -> int:
        return sum(r.seats for r in self.rows)

    def as_dict(self) -> dict[str, Any]:
        return {"users": self.users, "context": self.context, "prefer": self.prefer,
                "seated": self.seated, "unplaced": self.unplaced,
                "rows": [asdict(r) for r in self.rows],
                "why": [{"peer": p, "model": m, "reason": r} for p, m, r in self.why]}


def _note(out: Placement, peer: str, model: str, reason: str) -> None:
    if (peer, model, reason) not in out.why:
        out.why.append((peer, model, reason))


def _seats_on(peer: Room, fit: Fit, context: int, left: int) -> tuple[int, int, str]:
    """Seats for at most ``left`` users on ``peer``, the bytes that uses, and the reason
    when there are none."""
    from ml_stack.hub import _human

    here = fit.at_room(peer.room)
    loaded, each = here.line(context)
    if loaded > peer.room:
        return 0, 0, f"room {_human(peer.room)} < {_human(loaded)} loaded"
    seats = min(left, here.free() // each) if each > 0 else left
    if seats < 1:
        return 0, 0, (f"room {_human(peer.room)} < {_human(loaded + each)} "
                      f"for one seat at {context}")
    return seats, loaded + seats * each, ""


def place(users: int, context: int, peers: Sequence[Any], profiles: Sequence[Profile],
          fits: Sequence[Fit], *, log: Callable[[str], None] | None = None,
          prefer: str = "quality") -> Placement:
    """Seats for ``users`` at ``context`` tokens each across ``peers``.

    With ``prefer="quality"`` models are taken best first (`ranked`); each goes on every
    peer with room for its loaded size and at least one seat, roomiest peer first, taking
    as many seats as fit or as are still wanted. With ``prefer="seats"`` peers are taken
    roomiest first and each serves whichever model seats the most of the users still
    waiting, ties going to the better-ranked one. A peer serves one model. Stops when
    every user has a seat; ``unplaced`` is how many did not get one.
    """
    from ml_stack.hub import _human

    say = log or (lambda line: None)
    want = prefer if prefer in PREFERENCES else "quality"
    out = Placement(users=int(users), context=int(context), prefer=want)
    left = max(0, int(users))
    open_peers = sorted((room_of(p) for p in peers), key=lambda r: -r.room)
    say(f"planning {left} user(s) at {context} tokens over {len(open_peers)} peer(s), "
        f"{PREFERENCES[want]}:")
    if not open_peers:
        out.why.append(("*", "*", "no peer answered"))
        out.unplaced = left
        say(f"  {left} user(s) without a seat: no peer answered")
        return out

    candidates: list[tuple[Profile, Fit]] = []
    for profile in ranked(profiles):
        spec = profile.spec_type if profile.draft else ""
        fit = fit_for(profile.model, fits, cache_type=profile.cache_type, spec=spec)
        if fit is None:
            _note(out, "*", profile.model, "no memory measurement")
            say(f"  {profile.model:<48} no memory measurement")
            continue
        candidates.append((profile, fit))
    ranked_names = {_plain(f.model) for _, f in candidates}
    for fit in fits:
        if _plain(fit.model) not in ranked_names:
            _note(out, "*", fit.model, "not ranked")
            say(f"  {fit.model:<48} not ranked")

    def seat(peer: Room, profile: Profile, seats: int, used: int) -> None:
        nonlocal left
        out.rows.append(Row(peer=peer.name, model=profile.model, seats=seats,
                            context=int(context), used=used, room=peer.room,
                            base_url=peer.base_url))
        say(f"  {profile.model:<48} -> {peer.name}: {seats} seat(s), "
            f"{_human(used)} of {_human(peer.room)}")
        left -= seats
        open_peers.remove(peer)

    if want == "seats":
        for peer in list(open_peers):
            if left <= 0:
                break
            if peer.room <= 0:
                _note(out, peer.name, "*", "room unknown")
                continue
            best: tuple[int, int, Profile] | None = None
            for profile, fit in candidates:
                seats, used, why = _seats_on(peer, fit, context, left)
                if seats < 1:
                    _note(out, peer.name, profile.model, why)
                    continue
                if best is None or seats > best[0]:
                    best = (seats, used, profile)
            if best is not None:
                seat(peer, best[2], best[0], best[1])
    else:
        for profile, fit in candidates:
            if left <= 0:
                break
            for peer in list(open_peers):
                if left <= 0:
                    break
                if peer.room <= 0:
                    _note(out, peer.name, "*", "room unknown")
                    continue
                seats, used, why = _seats_on(peer, fit, context, left)
                if seats < 1:
                    _note(out, peer.name, profile.model, why)
                    continue
                seat(peer, profile, seats, used)
    out.unplaced = left
    if left:
        say(f"  {left} user(s) without a seat")
    return out


def table(placement: Placement) -> str:
    """The placement, as text."""
    from ml_stack.hub import _human

    lines = [f"{'PEER':<16} {'MODEL':<48} {'SEATS':>5} {'CONTEXT':>8} {'USED':>8} {'ROOM':>8}"]
    for r in placement.rows:
        lines.append(f"{r.peer:<16} {r.model:<48} {r.seats:>5} {r.context:>8} "
                     f"{_human(r.used):>8} {_human(r.room):>8}")
    if not placement.rows:
        lines.append("nobody seated")
    lines.append(f"{placement.seated} of {placement.users} user(s) seated at "
                 f"{placement.context} tokens each")
    lines.append(f"--prefer {placement.prefer}: "
                 f"{PREFERENCES.get(placement.prefer, PREFERENCES['quality'])}")
    if placement.unplaced:
        lines.append(f"{placement.unplaced} user(s) without a seat:")
        for peer, model, reason in placement.why:
            lines.append(f"  {peer}: {model}: {reason}")
    return "\n".join(lines)
