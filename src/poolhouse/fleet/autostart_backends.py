"""The service-manager commands that load, unload and probe a role's units on each platform."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from poolhouse.platform import applescript_quote

from .autostart_manifest import ROLE_INFO, Role

__all__ = ["Backend", "backend_for", "elevated", "run"]


def run(argv: list[str], timeout: float = 60) -> tuple[int, str]:
    """Run one command without a shell; its exit code and combined output (1 and the error when it cannot start)."""
    try:
        done = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        return 1, str(exc)
    return done.returncode, (done.stdout + done.stderr).strip()


def elevated(command: str, prompt: str) -> tuple[bool, str]:
    """Run a privileged shell command through the OS's own password dialog."""
    if sys.platform == "darwin":
        script = (f'do shell script "{applescript_quote(command)}" with administrator '
                  f'privileges with prompt "{applescript_quote(prompt)}"')
        argv = ["osascript", "-e", script]
    elif sys.platform.startswith("linux") and shutil.which("pkexec"):
        argv = ["pkexec", "sh", "-c", command]
    else:
        return False, ""
    code, out = run(argv, timeout=120)
    return code == 0, out


def _sh(argv: list[str], timeout: float = 60) -> tuple[int, str]:
    return run(argv, timeout)


@dataclass(frozen=True, slots=True)
class Backend:
    """Argument lists for one service manager; `loaded` runs the probe through ``runner``."""

    platform: str
    scope: str
    runner: Callable[[list[str]], tuple[int, str]] = field(default=_sh)

    def _domain(self) -> str:
        return "system" if self.scope == "system" else f"gui/{os.getuid()}"

    def _systemctl(self, *words: str) -> list[str]:
        return ["systemctl", *(["--user"] if self.scope == "user" else []), *words]

    def _active_unit(self, role: Role) -> str:
        info = ROLE_INFO[role.role]
        return f"{info['service']}.{'timer' if 'timer' in info['kinds'] else 'service'}"

    def load(self, role: Role, directory: Path) -> list[list[str]]:
        """The commands that start ``role`` and keep it started."""
        if self.platform == "darwin":
            unit = role.units[0].destination
            return [["launchctl", "bootstrap", self._domain(), unit]]
        if self.platform == "win32":
            return [["schtasks", "/Create", "/F", "/TN", role.label, "/XML", str(directory / role.units[0].name)]]
        return [self._systemctl("daemon-reload"), self._systemctl("enable", "--now", self._active_unit(role))]

    def unload(self, role: Role) -> list[list[str]]:
        """The commands that stop ``role`` and keep it from starting."""
        if self.platform == "darwin":
            return [["launchctl", "bootout", f"{self._domain()}/{role.label}"]]
        if self.platform == "win32":
            return [["schtasks", "/Delete", "/F", "/TN", role.label]]
        return [self._systemctl("disable", "--now", self._active_unit(role)), self._systemctl("daemon-reload")]

    def probe(self, role: Role) -> list[str]:
        """The command whose exit code says whether ``role`` is loaded."""
        if self.platform == "darwin":
            return ["launchctl", "print", f"{self._domain()}/{role.label}"]
        if self.platform == "win32":
            return ["schtasks", "/Query", "/TN", role.label]
        return self._systemctl("is-active", self._active_unit(role))

    def loaded(self, role: Role) -> bool:
        """Whether the service manager has ``role`` loaded."""
        return self.runner(self.probe(role))[0] == 0


def backend_for(platform: str, scope: str, runner: Callable[[list[str]], tuple[int, str]] | None = None) -> Backend:
    """The backend for ``platform`` and ``scope``, running commands through ``runner`` when given."""
    return Backend(platform, scope, runner or _sh)
