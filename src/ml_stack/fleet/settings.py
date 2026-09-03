"""What this machine has been told to do, remembered between runs."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ml_stack.files import write_json

__all__ = ["Settings", "Suggestion", "suggest"]


@dataclass
class Settings:
    """Per-machine preferences. Every field is something a person chose."""

    slots: int = 1
    labels: list[str] = field(default_factory=list)
    on_paused: str = "stop"
    """``stop`` gets the machine back now, at the cost of restarting the current job"""
    autostart: str = "manual"
    setup_done: bool = False
    """Whether the first-run wizard was finished. A machine may finish it in no cluster."""
    on_close: str = ""
    auto_update: bool = True
    """Follow releases: a bundled install replaces itself when a newer one is published,
    once nothing is running on the machine. On unless turned off here; a pip install is
    told to use pip instead."""
    track_branch: str = ""
    """Follow a branch instead -- ``main`` on a machine you trust to run what was pushed
    minutes ago, unreviewed. One or the other: a machine that tracks a branch does not also
    follow releases. Empty is off, which is what a checkout starts as."""
    track_repo: str = ""
    """Where to follow that branch from (default: the public repository)."""
    update_channel: str = "stable"
    fetch_slots: int = 2
    autodownload_models: bool = True
    context: int = 8192
    """How much of a conversation a model is given to read. Costs memory per token."""

    @classmethod
    def load(cls, path: Path | str) -> "Settings":
        p = Path(path).expanduser()
        if not p.exists():
            return cls()
        try:
            raw = json.loads(p.read_text())
        except (OSError, ValueError):
            return cls()
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in raw.items() if k in known})

    def save(self, path: Path | str) -> Path:
        """Written atomically -- a half-written settings file reads as no settings."""
        p = Path(path).expanduser()
        # Every field is flat, so sorting the top level is what sort_keys did.
        write_json(p, dict(sorted(asdict(self).items())))
        return p

    def public(self) -> dict[str, Any]:
        return asdict(self)


# -- what this machine probably wants ------------------------------------
@dataclass(frozen=True, slots=True)
class Suggestion:
    """A proposed setting, and the reason -- which is shown, not hidden."""

    value: Any
    why: str


def _has_battery() -> bool:
    """Whether this looks like a laptop. Best effort, standard library only."""
    linux = Path("/sys/class/power_supply")
    if linux.is_dir():
        return any(p.name.upper().startswith("BAT") for p in linux.iterdir())
    if sys.platform == "darwin":
        try:
            out = subprocess.run(["pmset", "-g", "batt"], capture_output=True,
                                 text=True, timeout=4)
        except (OSError, subprocess.SubprocessError):
            return False
        return "InternalBattery" in out.stdout
    if sys.platform == "win32":
        try:
            out = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 "(Get-CimInstance Win32_Battery | Measure-Object).Count"],
                capture_output=True, text=True, timeout=8)
        except (OSError, subprocess.SubprocessError):
            return False
        return out.stdout.strip() not in ("", "0")
    return False


def suggest(report: dict[str, Any] | None = None) -> dict[str, Suggestion]:
    """What to pre-tick in the wizard, based on what this machine actually is."""
    report = report or {}
    accelerator = bool(report.get("accelerator") or report.get("cuda")
                       or report.get("rocm"))
    gpu = report.get("gpu") or "a GPU"
    portable = _has_battery()

    out: dict[str, Suggestion] = {}

    if accelerator:
        out["labels"] = Suggestion(
            ["train"], f"it has {gpu}, so it is the machine to train on")
    else:
        out["labels"] = Suggestion(
            ["prep"], "no GPU found, so this is a good machine for preparing data "
                      "while the others train")

    if portable:
        out["autostart"] = Suggestion(
            "login", "this looks like a laptop, so it starts when you log in rather "
                     "than running with the lid shut")
        out["work_hours"] = Suggestion(
            True, "and it will not take work during the day, since you are probably "
                  "using it")
    else:
        out["autostart"] = Suggestion(
            "login", "starts when you log in, which needs no permission from you")
        out["work_hours"] = Suggestion(False, "")

    out["on_paused"] = Suggestion(
        "stop", "pausing gives the machine back straight away; the run picks up from "
                "its last checkpoint")
    return out
