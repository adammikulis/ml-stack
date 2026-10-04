"""One-time invite codes: a person creates one for an agent name, the agent redeems it once."""

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
INVALID = "that code is not valid: it is wrong, already used or expired"


def normalise(code: str) -> str:
    """``code`` as stored: upper case, without spaces or dashes."""
    return re.sub(r"[\s-]+", "", code).upper()


def _hash(code: str) -> str:
    return hashlib.sha256(normalise(code).encode()).hexdigest()


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

    def create(self, hint: str, ttl_s: float, project: dict[str, str] | None = None) -> str:
        """A new single-use code, good for ``ttl_s`` seconds; ``hint`` is only a suggested id
        and ``project`` the ``{"key", "name"}`` the joining agent is connected for."""
        if hint and not valid_name(hint):
            raise ValueError(f"{hint!r} is not a usable agent id (a-z, 0-9, . _ -; up to 48)")
        code = "-".join("".join(secrets.choice(ALPHABET) for _ in range(WIDTH))
                        for _ in range(GROUPS))
        with held(self.path.with_name("invites.lock")):
            data = self._load()
            now = self.clock()
            data["invites"] = {h: e for h, e in data["invites"].items() if e["expires"] > now}
            data["invites"][_hash(code)] = {"hint": hint, "expires": now + ttl_s, "used_by": "",
                                         "project": project or {}}
            self._save(data)
        return code

    def state(self, code: str) -> str:
        """``waiting`` while the code can be redeemed, ``used`` after, else ``gone``."""
        entry = self._load()["invites"].get(_hash(code))
        if not entry or entry["expires"] <= self.clock():
            return "gone"
        return "used" if entry["used_by"] else "waiting"

    def joined_as(self, code: str) -> str:
        """The name the agent that redeemed ``code`` took, or an empty string."""
        entry = self._load()["invites"].get(_hash(code))
        return str(entry["used_by"]) if entry else ""

    def redeem(self, code: str, take: Callable[[str, dict[str, str]], str]) -> str:
        """Consume ``code``: ``take`` is called with the invite's hint and project and returns the name the
        agent now holds, and the code is spent only if it returns. `Denied` for a wrong, used or
        expired code, and for any code once too many wrong ones were tried."""
        with held(self.path.with_name("invites.lock")):
            data = self._load()
            now = self.clock()
            data["fails"] = [t for t in data["fails"] if now - t < LOCKOUT_S]
            if len(data["fails"]) >= self.most:
                self._save(data)
                raise Denied("too many wrong codes; wait ten minutes or ask the person for a new one")
            entry = data["invites"].get(_hash(code))
            if entry is None or entry["expires"] <= now or entry["used_by"]:
                data["fails"].append(now)
                self._save(data)
                raise Denied(INVALID)
            entry["used_by"] = take(str(entry["hint"]), dict(entry.get("project", {})))
            self._save(data)
            return str(entry["used_by"])
