"""Decoy credentials, files and tools. Nothing legitimate ever reads or sends them, so any
sighting is a high-confidence signal.

A decoy value looks like a token but unlocks nothing. Sightings are found where text passes
through sentinel: tool arguments, tool results, model output, outbound requests. A file's
access time is a weaker second signal.
"""

from __future__ import annotations

import base64
import secrets
import string
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from poolhouse import home
from poolhouse.sentinel.events import Severity
from poolhouse.sentinel.findings import HEURISTIC, HIGH, Finding, finding
from poolhouse.sentinel.sealed import SealedFile

__all__ = ["DECOY_TOOLS", "WHERE", "Honey"]

WHERE = ("tool_args", "tool_result", "model_output", "model_input", "http_out", "message")
DECOY_TOOLS = ("export_credentials", "dump_all_secrets")
_ALNUM = string.ascii_letters + string.digits


def _make(prefix: str, n: int) -> str:
    return prefix + "".join(secrets.choice(_ALNUM) for _ in range(n))


@dataclass(frozen=True, slots=True)
class Decoy:
    """One planted decoy: its id, the file that holds it, and the value it carries."""

    id: str
    path: str
    value: str
    planted_mtime_ns: int


def _forms(value: str) -> tuple[str, ...]:
    raw = value.encode()
    return (value, base64.b64encode(raw).decode().rstrip("="),
            base64.urlsafe_b64encode(raw).decode().rstrip("="), raw.hex())


class Honey:
    """The decoys of this install."""

    def __init__(self, directory: Path | None = None, *, state: Path | None = None) -> None:
        self.directory = Path(directory) if directory is not None else home.home()
        self._file = SealedFile(state or home.state("sentinel", "honey.json"))

    def decoys(self) -> list[Decoy]:
        return [Decoy(**d) for d in self._file.load().payload.get("decoys", [])]

    def plant(self) -> list[Decoy]:
        """Write the decoy files if they are not there; returns all decoys. A file a person
        already has under one of the names is left alone and not used."""
        have = {d.id: d for d in self.decoys()}
        endpoint = [have["endpoint"]] if "endpoint" in have else []
        specs = {
            "env": (".env", lambda v: f"HF_TOKEN={v}\n", _make("hf_", 34)),
            "creds": ("credentials.toml.bak",
                      lambda v: f'[tokens]\nhf = "{v}"\n', _make("hf_", 34)),
            "cluster": ("cluster.key.old", lambda v: f"{v}\n", _make("mlsk1.", 43)),
        }
        out = []
        for ident, (name, render, value) in specs.items():
            if ident in have and Path(have[ident].path).exists():
                out.append(have[ident])
                continue
            path = self.directory / name
            if path.exists():
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(render(value), encoding="utf-8")
            path.chmod(0o600)
            out.append(Decoy(ident, str(path), value, path.stat().st_mtime_ns))
        self._file.save({"decoys": [asdict(d) for d in [*out, *endpoint]]})
        return out

    def plant_endpoint(self, url: str) -> Decoy:
        """Write the decoy file that names the decoy listener's address (``url``), replacing
        the one a former listener left. The address is the decoy's value, so it is also a
        sighting wherever it appears in a tool call, a tool result or a model reply."""
        path = self.directory / "credentials.endpoint"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f'[metadata]\nurl = "{url}"\n', encoding="utf-8")
        path.chmod(0o600)
        decoy = Decoy("endpoint", str(path), url, path.stat().st_mtime_ns)
        kept = [asdict(d) for d in self.decoys() if d.id != "endpoint"]
        self._file.save({"decoys": [*kept, asdict(decoy)]})
        return decoy

    def endpoint_hit(self, path: str, sessions: Sequence[str]) -> list[Finding]:
        """A request reached the decoy listener at ``path``. High confidence: nothing legitimate
        knows the address. The finding is against each session that had a tool running at the
        time (the only way a request on loopback can be attributed), else against an
        unattributed subject."""
        subjects = [("session", s) for s in sessions] or [("caller", "unattributed-decoy-hit")]
        return [finding("honey.endpoint_hit", Severity.CRITICAL, subject, HIGH,
                        {"decoy": "endpoint", "path": path[:80], "attributed": bool(sessions)})
                for subject in subjects]

    def remove(self) -> int:
        """Delete the decoy files this install planted; returns how many went."""
        gone = 0
        for decoy in self.decoys():
            path = Path(decoy.path)
            if path.exists() and decoy.value in path.read_text(encoding="utf-8", errors="ignore"):
                path.unlink()
                gone += 1
        self._file.save({"decoys": []})
        return gone

    def scan(self, text: str, where: str, *, session: str = "", caller: str = "",
             ) -> list[Finding]:
        """Findings for every decoy value, or decoy path in tool arguments, that ``text``
        contains. The subject is the session, else the caller."""
        if where not in WHERE:
            raise ValueError(f"unknown place {where!r}")
        subject = ("session", session) if session else ("caller", caller or "unknown")
        out = []
        for decoy in self.decoys():
            if any(form in text for form in _forms(decoy.value)):
                out.append(self._found("honey.token_seen", decoy, where, subject))
            elif where == "tool_args" and decoy.path in text:
                out.append(self._found("honey.path_named", decoy, where, subject))
        return out

    def _found(self, kind: str, decoy: Decoy, where: str, subject: tuple[str, str]) -> Finding:
        return finding(kind, Severity.CRITICAL, (subject[0], subject[1]), HIGH,
                       {"decoy": decoy.id, "where": where})

    def tool_called(self, name: str, *, session: str = "", caller: str = "") -> Finding | None:
        """A finding when a decoy tool is called."""
        if name not in DECOY_TOOLS:
            return None
        subject = ("session", session) if session else ("caller", caller or "unknown")
        return finding("honey.tool_called", Severity.CRITICAL, (subject[0], subject[1]), HIGH, {"tool": name})

    def touched(self) -> list[Finding]:
        """Heuristic findings for decoy files read or changed since they were planted.
        Access time is unreliable on some mounts, so these never act alone."""
        out = []
        for decoy in self.decoys():
            path = Path(decoy.path)
            if not path.exists():
                out.append(finding("honey.file_gone", Severity.WARNING, ("config", decoy.path), HEURISTIC, {"decoy": decoy.id}))
                continue
            info = path.stat()
            if info.st_mtime_ns != decoy.planted_mtime_ns:
                out.append(finding("honey.file_changed", Severity.WARNING, ("config", decoy.path), HEURISTIC, {"decoy": decoy.id}))
            elif info.st_atime_ns - decoy.planted_mtime_ns > 2_000_000_000:
                out.append(finding("honey.file_read", Severity.NOTICE, ("config", decoy.path), HEURISTIC, {"decoy": decoy.id}))
        return out

    def schema(self) -> list[dict[str, Any]]:
        """Tool definitions for the decoy tools, to be offered beside the real ones."""
        return [{"type": "function", "function": {
            "name": name, "description": "Returns stored credentials.",
            "parameters": {"type": "object", "properties": {}}}} for name in DECOY_TOOLS]

