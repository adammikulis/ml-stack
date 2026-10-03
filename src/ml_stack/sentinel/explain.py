"""Plain words for what sentinel holds: why, what it blocks, and what to do, without ever
repeating a held subject's own text raw.

Every name, reason, peer, path and argument on a record came from outside, so `show` is the
only way any of it reaches a terminal or a notification: control, bidi and invisible
characters become visible escapes and the length is bounded. `WHY` has a sentence for every
finding kind a detector can raise; ``tests/test_sentinel_explain.py`` reads the detectors and
fails when one is missing, so a new detector cannot ship without its sentence.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import PurePath

from ml_stack.sentinel.redaction import redact
from ml_stack.sentinel.store import Record, State

__all__ = ["BLOCKS", "WHY", "Item", "age", "code_of", "describe", "name_of", "show"]

WHY: dict[str, str] = {
    "peer.forged_traffic": "This machine sent requests with a forged or replayed signature, "
                           "which no honest ml-stack peer does.",
    "peer.auth_failures": "This machine failed authentication many times in a row.",
    "peer.rate": "This machine is sending requests much faster than a peer normally does.",
    "peer.flapping": "This machine keeps joining and leaving the fleet.",
    "peer.version_mismatch": "This machine runs a different ml-stack version than the one "
                             "this fleet pinned.",
    "peer.binary_mismatch": "This machine reports a different program file than the one this "
                            "fleet pinned.",
    "server.unmanaged": "A model server is running here that ml-stack did not start.",
    "server.exe_changed": "The program behind a running model server changed after it started.",
    "integrity.manifest_mismatch": "A model file does not match the manifest it was published with.",
    "integrity.content_changed": "A pinned file's contents changed since it was pinned.",
    "integrity.missing": "A pinned file is gone.",
    "integrity.unreadable": "A pinned file can no longer be read.",
    "integrity.binary_mismatch": "The managed llama.cpp server program does not match the hash recorded when it was built, so it was not started.",
    "integrity.link_retargeted": "A pinned link now points somewhere else than when it was pinned.",
    "tools.mix_shift": "This session's mix of tool calls changed sharply from its own usual.",
    "abuse.resource": "This caller used far more calls, bytes or time than a caller normally does.",
    "score.watch": "Several weak warning signs added up for this subject.",
    "score.quarantine": "Enough warning signs added up for this subject to hold it.",
    "guard.text_denied": "A guard rule refused this text.",
    "guard.call_denied": "A guard rule refused a tool call from this session.",
    "guard.denied": "A guard rule refused something this session tried.",
    "guard.repeated_denials": "A guard rule has refused this session again and again.",
    "guard.tainted": "This session read text that tried to give it instructions.",
    "guard.tainted_text": "This text looks like an instruction aimed at a model.",
    "honey.token_seen": "A decoy secret that only an intruder could know turned up in this text.",
    "honey.path_named": "A tool call named the path of a decoy file that nothing legitimate "
                        "touches.",
    "honey.tool_called": "A decoy tool that nothing legitimate calls was called.",
    "honey.endpoint_hit": "A decoy web address that nothing legitimate visits was requested.",
    "honey.file_read": "A decoy file was read.",
    "honey.file_changed": "A decoy file was changed.",
    "honey.file_gone": "A decoy file was deleted or moved.",
    "canary.drift": "A model's answers to fixed test questions moved from what they were.",
    "canary.hard_drift": "A model's answers to fixed test questions changed completely.",
}
"""Finding kind to one sentence, in words a person who has not read the code can act on."""

BY_REASON: tuple[tuple[str, str], ...] = (
    ("held by a person", "You asked for this to be held."),
    ("model output", "A model's reply carried a decoy, a secret-shaped string or held content."),
    ("derived from frozen session", "It was built from a session that is frozen."),
    ("repeats held content", "It repeats text that is already held."),
    ("carries a decoy value", "It carries a decoy secret."),
)
"""Reasons that are written by the caller rather than by a detector, matched by their start."""

UNKNOWN = "Sentinel recorded no plain-language reason for this one; read the details."

BLOCKS: dict[str, str] = {
    "peer": "Requests from this machine are refused.",
    "caller": "Requests from this caller are refused.",
    "tool": "This tool cannot be called.",
    "session": "This session is frozen, and notes built from it are held too.",
    "message": "This text is withheld and replaced by a placeholder.",
    "tool_call": "This tool call is withheld.",
    "memory": "This stored note or summary is not loaded; it is rebuilt from clean sources.",
    "model": "This model cannot be leased or started.",
    "artifact": "This file was moved aside and is not used.",
    "binary": "This program was moved aside and is not used.",
    "config": "This file was moved aside and is not used.",
    "mcp_server": "This tool server's tools are not offered.",
    "credential": "This credential is not handed to anything.",
    "server": "Nothing is blocked: a server is only reported.",
}
"""What a quarantine of each subject kind stops. A subject kind without a line stops nothing."""

_CODE = re.compile(r"^([a-z_]+\.[a-z_]+)\b")
_EDGE = "…"


def show(value: object, limit: int = 60) -> str:
    """``value`` as one safe line of at most ``limit`` characters: secrets masked, and every
    control, format (bidi, zero-width), separator, private or unassigned character written
    as a visible escape such as ``\\x1b`` or ``\\u202e``. Cut with an ellipsis when long."""
    text = redact(str(value)[:limit * 8])
    out = "".join(ch if _plain(ch) else _escape(ch) for ch in text)
    return out if len(out) <= limit else out[:limit - 1] + _EDGE


def _plain(ch: str) -> bool:
    return ch == " " or (ch.isprintable() and unicodedata.category(ch)[0] not in "CZ")


def _escape(ch: str) -> str:
    named = {"\n": "\\n", "\r": "\\r", "\t": "\\t"}
    if ch in named:
        return named[ch]
    point = ord(ch)
    if point < 0x100:
        return f"\\x{point:02x}"
    return f"\\u{point:04x}" if point < 0x10000 else f"\\U{point:08x}"


def name_of(kind: str, key: str) -> str:
    """A short name a person recognises: ``peer 10.0.0.5``, ``server :51089``, ``model
    qwen.gguf`` (the file name of a path, never the whole path)."""
    if kind == "server":
        port = key.partition(":")[2] or key
        return f"server :{show(port, 12)}"
    if kind in ("model", "artifact", "binary", "config") and ("/" in key or "\\" in key):
        leaf = PurePath(key.replace("\\", "/")).name or key
        return f"{kind} {show(leaf, 40)}"
    return f"{kind} {show(key, 40)}"


def age(seconds: float) -> str:
    """``just now``, ``5 min ago``, ``3 h ago`` or ``2 d ago``."""
    seconds = max(0.0, seconds)
    for limit, unit, size in ((90, "", 0), (5400, "min", 60), (172800, "h", 3600)):
        if seconds < limit:
            return "just now" if not unit else f"{round(seconds / size)} {unit} ago"
    return f"{round(seconds / 86400)} d ago"


def code_of(record: Record) -> str:
    """The finding kind a record was raised for (the start of its reason), else ``""``."""
    found = _CODE.match(record.reason)
    return found[1] if found else ""


def why_of(record: Record) -> str:
    """One sentence on why a record is held."""
    code = code_of(record)
    if code in WHY:
        return WHY[code]
    return next((sentence for start, sentence in BY_REASON
                 if record.reason.startswith(start)), UNKNOWN)


@dataclass(frozen=True, slots=True)
class Item:
    """One held subject as the review screen and ``--list`` show it. Every field is already
    safe to print."""

    number: int
    id: str
    state: str
    kind: str
    name: str
    age: str
    code: str
    why: str
    blocks: str
    advice: str

    def to_json(self) -> dict[str, object]:
        return {"number": self.number, "id": self.id, "state": self.state, "kind": self.kind,
                "name": self.name, "age": self.age, "code": self.code, "why": self.why,
                "blocks": self.blocks, "advice": self.advice}


def describe(record: Record, number: int, now: float) -> Item:
    """The safe, plain-language view of ``record``."""
    code = code_of(record)
    if record.state == State.QUARANTINED:
        blocks = BLOCKS.get(record.kind, "")
        advice = ("Keep it held unless you know this is a false alarm; releasing puts it back "
                  "in use." if blocks and not blocks.startswith("Nothing") else
                  "Nothing is blocked; release to clear it.")
    else:
        blocks = "Nothing is blocked: sentinel is only watching."
        advice = "Watch only. Release to stop watching; nothing changes for anything else."
    return Item(number, record.id, record.state.value, record.kind,
                name_of(record.kind, record.key), age(now - record.updated),
                show(code, 40), why_of(record), blocks, advice)
