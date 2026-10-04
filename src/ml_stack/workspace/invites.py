"""Invite codes: a person (or a joined agent, within limits) creates one, each agent redeems it once;
a code made for several agents stops at its use cap and at its expiry, whichever comes first."""

from __future__ import annotations

import hashlib
import re
import secrets
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ml_stack.files import read_json, write_json
from ml_stack.workspace.chain import held
from ml_stack.workspace.identity import Denied, valid_name

__all__ = ["Invites", "normalise"]

ALPHABET = "ABCDEFGHJKMNPQRSTUVWXYZ23456789"
GROUPS, WIDTH = 4, 4
LOCKOUT_S = 600.0
VERSION = 1
KEEP_S = 3_600.0
INVALID = "that code is not valid: it is wrong, already used or expired"


def normalise(code: str) -> str:
    """``code`` as stored: upper case, without spaces or dashes."""
    return re.sub(r"[\s-]+", "", code).upper()


def _hash(code: str) -> str:
    return hashlib.sha256(normalise(code).encode()).hexdigest()


def _names(entry: dict[str, Any]) -> list[str]:
    return [str(n) for n in entry.get("joined") or ([entry["used_by"]] if entry["used_by"] else [])]


def _spent(entry: dict[str, Any]) -> bool:
    return len(_names(entry)) >= int(entry.get("uses", 1))


class Invites:
    """Invite codes held as hashes in ``invites.json``, with a lockout after wrong guesses."""

    def __init__(self, base: Path, most_failures: int = 5,
                 clock: Callable[[], float] = time.time) -> None:
        self.path = base / "invites.json"
        self.most, self.clock = most_failures, clock

    def _load(self) -> dict[str, Any]:
        data = read_json(self.path, {})
        data = data if isinstance(data, dict) else {}
        return {"invites": dict(data.get("invites", {})), "fails": list(data.get("fails", []))}

    def _save(self, data: dict[str, Any]) -> None:
        write_json(self.path, {"version": VERSION, **data})
        self.path.chmod(0o600)

    def create(self, hint: str, ttl_s: float, project: dict[str, str] | None = None,
               uses: int = 1, origin: tuple[str, tuple[str, ...]] = ("", ())) -> str:
        """A new code, good for ``ttl_s`` seconds and ``uses`` joins (one agent by default);
        ``hint`` is only a suggested id and ``project`` the ``{"key", "name"}`` the joining
        agents are connected for. An ``origin`` of ``(issuer, can)`` makes the joiners the issuer's children,
        holding at most ``can``."""
        if uses < 1:
            raise ValueError("an invite must allow at least one agent")
        if hint and not valid_name(hint):
            raise ValueError(f"{hint!r} is not a usable agent id (a-z, 0-9, . _ -; up to 48)")
        issuer, can = origin
        code = "-".join("".join(secrets.choice(ALPHABET) for _ in range(WIDTH))
                        for _ in range(GROUPS))
        with held(self.path.with_name("invites.lock")):
            data = self._load()
            now = self.clock()
            data["invites"] = {h: e for h, e in data["invites"].items() if e["expires"] > now
                               or (e.get("issuer") and e.get("created", 0.0) > now - KEEP_S)}
            data["invites"][_hash(code)] = {"hint": hint, "expires": now + ttl_s, "used_by": "",
                                            "joined": [], "uses": uses, "project": project or {},
                                            "created": now, "issuer": issuer, "can": list(can)}
            self._save(data)
        return code

    def made_by(self, issuer: str) -> list[dict[str, Any]]:
        """``issuer``'s agent-made invites: ``created``, ``open`` (still redeemable) and ``left`` uses; never a code."""
        now = self.clock()
        return [{"created": float(e.get("created", 0.0)), "open": e["expires"] > now and not _spent(e),
                 "left": int(e.get("uses", 1)) - len(_names(e))}
                for e in self._load()["invites"].values() if e.get("issuer") == issuer]

    def leaks(self, text: str) -> bool:
        """Whether ``text`` contains the code of an invite that can still be redeemed."""
        now = self.clock()
        live = {h for h, e in self._load()["invites"].items() if e["expires"] > now and not _spent(e)}
        flat = re.sub(r"[^A-Za-z0-9]", "", text).upper()
        width = GROUPS * WIDTH
        return bool(live) and any(hashlib.sha256(flat[i:i + width].encode()).hexdigest() in live
                                  for i in range(len(flat) - width + 1))

    def void(self, issuers: list[str]) -> int:
        """Expire every open invite made by one of ``issuers``; how many were open."""
        with held(self.path.with_name("invites.lock")):
            data = self._load()
            now = self.clock()
            shut = 0
            for entry in data["invites"].values():
                if entry.get("issuer") in issuers and entry["expires"] > now:
                    entry["expires"] = now
                    shut += 1
            self._save(data)
            return shut

    def state(self, code: str) -> str:
        """``waiting`` while the code can be redeemed, ``used`` once every use is taken, else ``gone``."""
        entry = self._load()["invites"].get(_hash(code))
        if not entry or entry["expires"] <= self.clock():
            return "gone"
        return "used" if _spent(entry) else "waiting"

    def joined(self, code: str) -> list[str]:
        """The names the agents that redeemed ``code`` took, in order."""
        entry = self._load()["invites"].get(_hash(code))
        return _names(entry) if entry else []

    def joined_as(self, code: str) -> str:
        """The name the first agent that redeemed ``code`` took, or an empty string."""
        names = self.joined(code)
        return names[0] if names else ""

    def close(self, code: str) -> None:
        """Expire ``code`` now, so no further agent can use it."""
        with held(self.path.with_name("invites.lock")):
            data = self._load()
            entry = data["invites"].get(_hash(code))
            if entry:
                entry["expires"] = self.clock()
                self._save(data)

    def redeem(self, code: str, take: Callable[[str, dict[str, str], dict[str, Any]], str]) -> str:
        """Use ``code`` once: ``take`` is called with the invite's hint, project and origin (``issuer``,
        ``can``) and returns the name the
        agent now holds, and the use is spent only if it returns. `Denied` for a wrong, used or
        expired code, and for any code once too many wrong ones were tried."""
        with held(self.path.with_name("invites.lock")):
            data = self._load()
            now = self.clock()
            data["fails"] = [t for t in data["fails"] if now - t < LOCKOUT_S]
            if len(data["fails"]) >= self.most:
                self._save(data)
                raise Denied("too many wrong codes; wait ten minutes or ask the person for a new one")
            entry = data["invites"].get(_hash(code))
            if entry is None or entry["expires"] <= now or _spent(entry):
                data["fails"].append(now)
                self._save(data)
                raise Denied(INVALID)
            name = take(str(entry["hint"]), dict(entry.get("project", {})),
                        {"issuer": str(entry.get("issuer", "")), "can": tuple(entry.get("can", ()))})
            entry["joined"] = [*_names(entry), name]
            entry["used_by"] = entry["used_by"] or name
            self._save(data)
            return name
