"""Telling the owner that a machine wants to join, in the way their desktop shows things.

What a notification can and cannot do, honestly: a desktop notification from a command-line
program can say something and cannot carry an Accept button (macOS needs a signed app bundle
for `UNUserNotificationCenter` actions, Windows needs a registered app id for toast
activation, `notify-send` has actions only on some servers and no way to hear the answer). So
the notification tells the owner who is asking and the exact command to answer; the answer is
given in the terminal (`ml-stack fleet accept ID`) or the web interface. That is the design,
not a stand-in for one.

What goes in a notification is text from a stranger (a hostname the asking machine chose), so
it is cleaned to one short printable line and is handed to the operating system as an
argument, never spliced into a script: an AppleScript string, a shell line or a PowerShell
here-string built from a hostname is code execution on the owner's desktop.

The pairing code is never in a notification: it exists only after the owner says yes, and is
shown to them where they said it.
"""

from __future__ import annotations

import logging
import platform
import shutil
import subprocess
from collections.abc import Callable, Sequence
from typing import Protocol

from .requests import Request, clean, short

__all__ = ["Console", "LinuxNotifier", "MacNotifier", "Notifier", "Unsupported", "WindowsNotifier",
           "compose", "pick"]

logger = logging.getLogger("ml_stack.fleet.onboard")

Runner = Callable[[Sequence[str]], int]

MAC_SCRIPT = ("on run argv\n"
              "display notification (item 1 of argv) with title (item 2 of argv)\n"
              "end run")


class Notifier(Protocol):
    name: str

    def notify(self, title: str, body: str) -> bool: ...


class Unsupported(Exception):
    pass


def compose(request: Request) -> tuple[str, str]:
    """The title and body for a request: who is asking, from where, and how to answer."""
    who = clean(request.name, 40) or "a device"
    title = f"{who} wants to join ml-stack"
    host = f" ({clean(request.hostname, 40)})" if request.hostname else ""
    model = f", {clean(request.model, 40)}" if request.model else ""
    body = (f"{who}{host}{model} at {clean(request.address, 45)}, certificate "
            f"{short(request.fingerprint)}. To answer: ml-stack fleet accept "
            f"{request.id[:8]} or ml-stack fleet decline {request.id[:8]}. "
            "Only accept a device you are holding.")
    return title, body


def _run(argv: Sequence[str]) -> int:
    try:
        return subprocess.run(list(argv), capture_output=True, timeout=10,  # noqa: S603
                              check=False).returncode
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning("notification command failed: %s", type(exc).__name__)
        return 1


class MacNotifier:
    """``osascript`` with the text as script arguments, so it is data to the script."""

    name = "macos"

    def __init__(self, run: Runner = _run) -> None:
        self.run = run

    def notify(self, title: str, body: str) -> bool:
        return self.run(["osascript", "-e", MAC_SCRIPT, "--", clean(body, 300),
                         clean(title, 80)]) == 0


class LinuxNotifier:
    """``notify-send``, with ``--`` before the text so a title that starts with a dash is
    not an option."""

    name = "linux"

    def __init__(self, run: Runner = _run) -> None:
        self.run = run

    def notify(self, title: str, body: str) -> bool:
        return self.run(["notify-send", "--app-name=ml-stack", "--urgency=normal", "--",
                         clean(title, 80), clean(body, 300)]) == 0


class WindowsNotifier:
    """Designed, not built: a toast needs an app id registered with the shell, and the
    PowerShell route builds XML from text, which is the injection described above. Until it
    is built with the text passed as data (a registered helper executable), this says so."""

    name = "windows"

    def notify(self, title: str, body: str) -> bool:
        raise Unsupported("Windows toast notifications are not built; use the terminal or "
                          "web interface to see requests")


class Console:
    """Prints; the fallback where there is no desktop (and what a test collects)."""

    name = "console"

    def __init__(self, say: Callable[[str], None] = print) -> None:
        self.say = say

    def notify(self, title: str, body: str) -> bool:
        self.say(f"{clean(title, 80)}: {clean(body, 300)}")
        return True


def pick(system: str | None = None, *, which: Callable[[str], str | None] = shutil.which,
         run: Runner = _run) -> Notifier:
    """The notifier for this machine, or the console where there is none."""
    system = system or platform.system()
    if system == "Darwin" and which("osascript"):
        return MacNotifier(run)
    if system == "Linux" and which("notify-send"):
        return LinuxNotifier(run)
    if system == "Windows":
        return WindowsNotifier()
    return Console()
