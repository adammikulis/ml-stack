"""Staging the managed sandbox files, their manifest, and the one command a person runs to install them."""

from __future__ import annotations

import hashlib
import json
import shlex
import time
from dataclasses import dataclass
from pathlib import Path

import agent_sandbox_profile as profile

LIFETIME = 7 * 24 * 3600
MANIFEST = "manifest.json"
SUMS = "SHA256SUMS"

DESTINATIONS = {
    "darwin": {
        "claude-managed-settings.json": "/Library/Application Support/ClaudeCode/managed-settings.json",
        "srt-settings.json": "/Library/Application Support/ClaudeCode/srt-settings.json",
        "codex-config.toml": "/etc/codex/config.toml",
    },
    "linux": {
        "claude-managed-settings.json": "/etc/claude-code/managed-settings.json",
        "srt-settings.json": "/etc/claude-code/srt-settings.json",
        "codex-config.toml": "/etc/codex/config.toml",
    },
}


@dataclass(frozen=True)
class Entry:
    """One staged file and where a person installs it."""

    name: str
    sha256: str
    destination: str


def staging_dir(state: Path) -> Path:
    """The directory prepared files are written to."""
    return state / "agent-sandbox" / "staging"


def digest(data: bytes) -> str:
    """The sha256 of ``data`` as hex."""
    return hashlib.sha256(data).hexdigest()


def rendered(layout: profile.Layout) -> dict[str, bytes]:
    """Every managed file's content, by staged name."""
    return {
        "claude-managed-settings.json": (json.dumps(profile.claude_settings(layout), indent=2) + "\n").encode(),
        "srt-settings.json": (json.dumps(profile.srt_settings(layout), indent=2) + "\n").encode(),
        "codex-config.toml": profile.codex_toml(layout).encode(),
    }


def prepare(layout: profile.Layout, device: str, now: float | None = None) -> Path:
    """Write the staged files, SHA256SUMS and manifest; return the staging directory."""
    stamp = time.time() if now is None else now
    target = staging_dir(layout.state)
    target.mkdir(parents=True, exist_ok=True)
    places = DESTINATIONS[layout.platform]
    entries = []
    for name, data in rendered(layout).items():
        (target / name).write_bytes(data)
        entries.append({"name": name, "sha256": digest(data), "destination": places[name]})
    (target / SUMS).write_text("".join(f"{item['sha256']}  {item['name']}\n" for item in entries))
    manifest = {"version": 1, "platform": layout.platform, "device_id": device, "created": stamp,
                "expires": stamp + LIFETIME, "files": entries}
    (target / MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n")
    return target


def load(state: Path) -> dict:
    """The staged manifest, or an empty dict when nothing is staged."""
    path = staging_dir(state) / MANIFEST
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def install_command(state: Path) -> str:
    """The single command a person runs at their own terminal to verify and install the staged files."""
    manifest = load(state)
    target = staging_dir(state)
    files = manifest.get("files", [])
    sums = " ".join(shlex.quote(f"{item['sha256']}  {item['name']}") for item in files)
    reader = "shasum -a 256 -c -" if manifest.get("platform") == "darwin" else "sha256sum -c -"
    steps = [f"cd {shlex.quote(str(target))}", f"printf '%s\\n' {sums} | {reader}"]
    folders = dict.fromkeys(str(Path(item["destination"]).parent) for item in files)
    steps += [f"install -d -m 755 {shlex.quote(folder)}" for folder in folders]
    steps += [f"install -m 644 {shlex.quote(item['name'])} {shlex.quote(item['destination'])}" for item in files]
    return "sudo sh -c " + shlex.quote(" && ".join(steps))


def installed(state: Path) -> list[tuple[str, str]]:
    """For each staged file: its destination and ``match``, ``differs`` or ``absent``."""
    manifest = load(state)
    out = []
    for item in manifest.get("files", []):
        try:
            found = digest(Path(item["destination"]).read_bytes())
        except OSError:
            out.append((item["destination"], "absent"))
            continue
        out.append((item["destination"], "match" if found == item["sha256"] else "differs"))
    return out


def expired(state: Path, now: float | None = None) -> bool:
    """Whether the staged files are past their expiry."""
    manifest = load(state)
    return bool(manifest) and (time.time() if now is None else now) > manifest["expires"]
