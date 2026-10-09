"""The devices a model pull asks before the Hub: a small private file and nothing else.

Both sides read it: the fleet's provider (`poolhouse.fleet.onboard.peerfirst`) to reach the
devices, and `hub.peers` to know whether there is anybody to ask, so the fleet is only loaded when
there is. It lives below the fleet in the package layers so neither needs the other's data type.
A row: ``name``, ``url`` (the device's ``fleet share``), ``certificate`` (its beacon, which TLS is
pinned to), ``signing_key`` (the owner's manifest key), ``device_secret`` (this device's request key),
``min_serial``; optionally ``fingerprint`` (its pairing record, so a route can be chosen), ``source``
(``pairing`` or ``manual``), ``rate`` (bytes/s measured), ``limit_bps``, ``streams``, ``metered``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from poolhouse import home
from poolhouse.files import read_json, write_json
from poolhouse.platform import private_file

__all__ = ["SCHEMA_VERSION", "PeerBook"]

SCHEMA_VERSION = 1
SETTABLE = frozenset({"limit_bps", "streams", "metered"})
"""The per-device settings `PeerBook.configure` may change."""
KEEP = ("rate", "limit_bps", "streams", "metered")


class PeerBook:
    """The devices to ask, in a private file beside the pairing records."""

    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path) if path else home.state("onboard", "peers.json")

    def _doc(self) -> dict[str, Any]:
        doc = read_json(self.path, {})
        return doc if isinstance(doc, dict) else {}

    def _save(self, doc: dict[str, Any]) -> None:
        write_json(self.path, {"schema_version": SCHEMA_VERSION, **doc})
        private_file(self.path)

    @property
    def enabled(self) -> bool:
        return self._doc().get("enabled", True) is not False

    def set_enabled(self, on: bool) -> None:
        self._save({**self._doc(), "enabled": on})

    def rows(self) -> list[dict[str, Any]]:
        rows = self._doc().get("peers", [])
        return [r for r in rows if isinstance(r, dict) and r.get("name") and r.get("url")]

    def add(self, row: dict[str, Any]) -> None:
        """Store ``row``; a row of the same name is replaced and keeps its measured rate and
        settings unless ``row`` names them."""
        doc = self._doc()
        old = next((r for r in self.rows() if r["name"] == row["name"]), {})
        keep = {k: v for k, v in old.items() if k in KEEP}
        kept = [r for r in self.rows() if r["name"] != row["name"]]
        self._save({**doc, "peers": [*kept, {**keep, **row}]})

    def remove(self, name: str) -> bool:
        doc, rows = self._doc(), self.rows()
        left = [r for r in rows if r["name"] != name]
        self._save({**doc, "peers": left})
        return len(left) != len(rows)

    def remove_device(self, fingerprint: str) -> list[str]:
        """Drop every row learned from the paired device with this certificate fingerprint (a
        revoked device is no longer asked); returns the names removed."""
        doc, rows = self._doc(), self.rows()
        gone = [r["name"] for r in rows if fingerprint and r.get("fingerprint") == fingerprint]
        if gone:
            self._save({**doc, "peers": [r for r in rows if r["name"] not in gone]})
        return gone

    def seen_serial(self, name: str, serial: int) -> None:
        """Remember the newest manifest serial ``name`` has shown."""
        rows = [{**r, "min_serial": max(int(r.get("min_serial", 0)), serial)}
                if r["name"] == name else r for r in self.rows()]
        self._save({**self._doc(), "peers": rows})

    def record_rate(self, name: str, bytes_per_s: float) -> None:
        """Blend a measured transfer rate into the device's running figure (half old, half new)."""
        rows = []
        for r in self.rows():
            if r["name"] == name:
                old = float(r.get("rate", 0) or 0)
                r = {**r, "rate": round(bytes_per_s if old <= 0 else (old + bytes_per_s) / 2)}
            rows.append(r)
        self._save({**self._doc(), "peers": rows})

    def configure(self, name: str | None, **settings: Any) -> list[str]:
        """Change ``limit_bps`` / ``streams`` / ``metered`` of one device, or of every device
        when ``name`` is None; returns the names changed. A value of None clears a setting."""
        bad = set(settings) - SETTABLE
        if bad:
            raise ValueError(f"not a peer setting: {sorted(bad)}")
        changed, rows = [], []
        for r in self.rows():
            if name is None or r["name"] == name:
                r = {k: v for k, v in r.items() if k not in settings}
                r.update({k: v for k, v in settings.items() if v is not None})
                changed.append(r["name"])
            rows.append(r)
        self._save({**self._doc(), "peers": rows})
        return changed
