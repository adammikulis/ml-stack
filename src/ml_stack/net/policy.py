"""Which hosts a fetch may reach: an allow-list, and approvals a person recorded."""

from __future__ import annotations

import fnmatch
import json
import os
import threading
import time
import urllib.parse
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

from ml_stack import home
from ml_stack.httpguard import Refused

__all__ = ["ALLOWED", "ENV", "Approval", "NeedsApproval", "Policy", "approvals_path", "default",
           "host_of"]

ALLOWED: tuple[str, ...] = (
    "huggingface.co", "*.huggingface.co", "hf.co", "*.hf.co",
    "github.com", "api.github.com", "objects.githubusercontent.com",
    "release-assets.githubusercontent.com", "raw.githubusercontent.com",
    "codeload.github.com", "nominatim.openstreetmap.org",
)
"""Hosts that need no approval: the model hub, GitHub's release hosts and the geocoder. The
host of ``$HF_ENDPOINT`` and those in ``$ML_STACK_NET_ALLOW_HOSTS`` count as listed too."""

ENV = "ML_STACK_NET_ALLOW_HOSTS"


class NeedsApproval(Refused):
    """A host that is neither allow-listed nor approved; ``host`` is what to approve."""

    def __init__(self, host: str, purpose: str) -> None:
        super().__init__(f"{host} is not on the allow-list for {purpose}; a person approves it "
                         f"with `ml-stack-security approve-host {host}`")
        self.host, self.purpose = host, purpose


@dataclass(frozen=True, slots=True)
class Approval:
    """One approved host: who approved it, when, why, and when it lapses (0 for never)."""

    host: str
    at: float
    by: str
    note: str = ""
    until: float = 0.0


def host_of(url: str) -> str:
    """The lower-case IDNA host name of ``url``; '' when it has none."""
    try:
        name = urllib.parse.urlsplit(url.strip()).hostname or ""
        return name.encode("idna").decode("ascii").lower()
    except (UnicodeError, ValueError):
        return ""


def approvals_path() -> Path:
    """The append-only record of approved hosts."""
    return home.state("net", "approvals.jsonl")


class Policy:
    """Allow-list plus recorded approvals; an allow-list entry may hold ``*`` wildcards."""

    def __init__(self, allowed: Iterable[str] = ALLOWED, *, path: Path | None = None,
                 clock: Callable[[], float] = time.time) -> None:
        self.allowed = tuple(p.lower() for p in allowed)
        self.path = path
        self.clock = clock
        self._lock = threading.Lock()

    def _file(self) -> Path:
        return self.path or approvals_path()

    def _extra(self) -> tuple[str, ...]:
        named = tuple(p.strip().lower() for p in os.environ.get(ENV, "").split(",") if p.strip())
        mirror = host_of(os.environ.get("HF_ENDPOINT", ""))
        return (*named, mirror) if mirror else named

    def approvals(self) -> list[Approval]:
        """Every approval on record, oldest first."""
        out: list[Approval] = []
        try:
            lines = self._file().read_text(encoding="utf-8").splitlines()
        except OSError:
            return out
        for line in lines:
            try:
                row = json.loads(line)
                out.append(Approval(str(row["host"]), float(row["at"]), str(row["by"]),
                                    str(row.get("note", "")), float(row.get("until", 0.0))))
            except (ValueError, KeyError, TypeError):
                continue
        return out

    def approve(self, host: str, *, by: str, note: str = "", for_s: float = 0.0) -> Approval:
        """Record that ``by`` approved ``host`` (one exact name), for ``for_s`` seconds or for good."""
        name = host.strip().lower()
        if not name or any(ch in name for ch in "*/ "):
            raise ValueError("approve one exact host name")
        now = self.clock()
        row = Approval(name, now, by, note, now + for_s if for_s else 0.0)
        where = self._file()
        with self._lock:
            where.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            with where.open("a", encoding="utf-8") as out:
                out.write(json.dumps(asdict(row), sort_keys=True) + "\n")
        return row

    def listed(self, host: str) -> bool:
        """Whether ``host`` is on the built-in or environment allow-list."""
        return any(fnmatch.fnmatchcase(host, p) for p in (*self.allowed, *self._extra()))

    def approved(self, host: str) -> bool:
        """Whether a person approved ``host`` and the approval has not lapsed."""
        now = self.clock()
        return any(a.host == host and (not a.until or a.until > now) for a in self.approvals())

    def admit(self, url: str, purpose: str = "fetch") -> str:
        """The host of ``url`` when it may be reached; `NeedsApproval` otherwise."""
        host = host_of(url)
        if not host:
            raise Refused("that URL has no host")
        if self.listed(host) or self.approved(host):
            return host
        raise NeedsApproval(host, purpose)


_DEFAULT = Policy()


def default() -> Policy:
    """The policy every fetch uses unless it is handed another."""
    return _DEFAULT
