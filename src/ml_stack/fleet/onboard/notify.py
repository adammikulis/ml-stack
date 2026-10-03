"""Asking the owner, on their own screen, whether a machine may join.

macOS and Linux get a dialog with real buttons (Decline, Accept as mine, Accept as someone
else's); the pairing code is in a second dialog, only after an Accept. Windows is designed
(toast buttons need a registered app id), not built. With no desktop, the console fallback
prints the request and `ml-stack fleet accept` / `decline` answer it. A stranger's text is
cleaned and passed to the system as an argument after ``--``, never spliced into a script.
``ML_STACK_NOTIFY`` picks the notifier: ``system`` (default), ``console`` or ``off``. The test
suite sets ``console``, and shims on PATH make any real attempt fail the run.
"""

from __future__ import annotations

import logging
import os
import platform
import re
import shutil
from collections.abc import Callable
from typing import Protocol

from ml_stack.desktop import ENV, WAIT_S, Result, Runner, run as _run
from ml_stack.log import say as say_

from .requests import Request, clean, short

__all__ = ["ENV", "Console", "LinuxNotifier", "MacNotifier", "Notifier", "Silent", "Unsupported",
           "WindowsNotifier", "compose", "compose_code", "parse_mac", "pick"]

logger = logging.getLogger("ml_stack.fleet.onboard")


DECLINE, MINE, OTHER = "Decline", "Accept as mine", "Accept as someone else's"
"""The three buttons; the owner says whose device it is in the same click."""

MAC_ASK = ("on run argv\n"
           "set answer to display alert (item 1 of argv) message (item 2 of argv) as critical "
           f'buttons {{"{DECLINE}", "{OTHER}", "{MINE}"}} default button "{DECLINE}" '
           f'cancel button "{DECLINE}" giving up after {WAIT_S}\n'
           "return answer\n"
           "end run")
MAC_CODE = ("on run argv\n"
            "display alert (item 1 of argv) message (item 2 of argv) "
            f'buttons {{"OK"}} default button "OK" giving up after {WAIT_S}\n'
            "end run")


class Notifier(Protocol):
    name: str

    def ask(self, title: str, body: str) -> str:
        """Put the question to the owner; ``mine``, ``other``, ``decline``, ``timeout`` or
        ``unavailable`` (no way to ask here; the command line answers)."""

    def show_code(self, title: str, body: str) -> bool: ...


class Unsupported(Exception):
    pass


def _text(request: Request) -> str:
    host = f" ({clean(request.hostname, 40)})" if request.hostname else ""
    model = f", {clean(request.model, 40)}" if request.model else ""
    return (f"{clean(request.name, 40) or 'a device'}{host}{model} at "
            f"{clean(request.address, 45)}, certificate {short(request.fingerprint)}.")


def compose(request: Request) -> tuple[str, str]:
    """The title and the question for a request: who is asking, from where."""
    title = f"{clean(request.name, 40) or 'A device'} wants to join ml-stack"
    return title, _text(request) + " Accept only a device you are holding."


def compose_code(request: Request) -> tuple[str, str]:
    """The second dialog: the code to read to the person at the new machine."""
    code = f"{request.code[:3]} {request.code[3:]}"
    return (f"Pairing code {code}",
            f"Type {code} on {clean(request.name, 40) or 'the new device'} within "
            f"{WAIT_S} seconds. It is good for three tries.")


def parse_mac(result: Result) -> str:
    """The answer in osascript's output: ``button returned:Accept as mine, gave up:false``."""
    code, out, err = result
    if code != 0:
        return "decline" if "-128" in err else "unavailable"      # -128: Decline / Escape
    if re.search(r"gave up:\s*true", out):
        return "timeout"
    got = re.search(r"button returned:\s*(.*?)(?:,\s*gave up:|$)", out.strip())
    return {MINE: "mine", OTHER: "other", DECLINE: "decline"}.get(got[1] if got else "", "decline")


class MacNotifier:
    """``osascript display alert`` with buttons; the text is script arguments after ``--``."""

    name = "macos"

    def __init__(self, run: Runner = _run) -> None:
        self.run = run

    def ask(self, title: str, body: str) -> str:
        return parse_mac(self.run(["osascript", "-e", MAC_ASK, "--", clean(title, 80),
                                   clean(body, 300)]))

    def show_code(self, title: str, body: str) -> bool:
        return self.run(["osascript", "-e", MAC_CODE, "--", clean(title, 80),
                         clean(body, 300)])[0] == 0


class LinuxNotifier:
    """``notify-send --action ... --wait`` where libnotify supports actions, else ``zenity``.
    ``--`` goes before the text so a title that starts with a dash is not an option."""

    name = "linux"

    def __init__(self, run: Runner = _run,
                 which: Callable[[str], str | None] = shutil.which) -> None:
        self.run, self.which = run, which

    def ask(self, title: str, body: str) -> str:
        title, body = clean(title, 80), clean(body, 300)
        if self.which("notify-send"):
            code, out, _ = self.run([
                "notify-send", "--app-name=ml-stack", "--urgency=critical", "--wait",
                f"--expire-time={WAIT_S * 1000}", "--action=mine=" + MINE,
                "--action=other=" + OTHER, "--action=decline=" + DECLINE, "--", title, body])
            if code == 0 and out.strip() in ("mine", "other", "decline"):
                return out.strip()
            if code == 0 and not out.strip():
                return "timeout"
        if self.which("zenity"):
            code, out, _ = self.run(["zenity", "--list", "--title", title, "--text", body,
                                     "--column", "Answer", DECLINE, MINE, OTHER,
                                     f"--timeout={WAIT_S}"])
            if code == 5:
                return "timeout"
            return {MINE: "mine", OTHER: "other"}.get(out.strip(), "decline") if code == 0 \
                else "decline"
        return "unavailable"

    def show_code(self, title: str, body: str) -> bool:
        if self.which("zenity"):
            return self.run(["zenity", "--info", "--title", clean(title, 80), "--text",
                             clean(body, 300), f"--timeout={WAIT_S}"])[0] == 0
        return self.run(["notify-send", "--app-name=ml-stack", "--", clean(title, 80),
                         clean(body, 300)])[0] == 0


class WindowsNotifier:
    """Designed, not built: toast buttons need an app id registered with the shell, and the
    PowerShell route builds XML from text, which is an injection."""

    name = "windows"

    def ask(self, title: str, body: str) -> str:
        return "unavailable"

    def show_code(self, title: str, body: str) -> bool:
        raise Unsupported("Windows toast notifications are not built; use the terminal or the "
                          "web interface to answer requests")


class Console:
    """Prints; the fallback where there is no desktop, and what the tests use."""

    name = "console"

    def __init__(self, say: Callable[[str], None] = say_) -> None:
        self.say = say

    def ask(self, title: str, body: str) -> str:
        self.say(f"{clean(title, 80)}: {clean(body, 300)} Answer with: ml-stack fleet accept "
                 "ID --mine|--other, or ml-stack fleet decline ID")
        return "unavailable"

    def show_code(self, title: str, body: str) -> bool:
        self.say(f"{clean(title, 80)}: {clean(body, 300)}")
        return True


class Silent:
    """``ML_STACK_NOTIFY=off``: nothing is shown."""

    name = "off"

    def ask(self, title: str, body: str) -> str:
        return "unavailable"

    def show_code(self, title: str, body: str) -> bool:
        return False


def pick(system: str | None = None, *, which: Callable[[str], str | None] = shutil.which,
         run: Runner = _run, env: dict[str, str] | None = None) -> Notifier:
    """The notifier for this machine: what ``ML_STACK_NOTIFY`` says, else the desktop's, else
    the console."""
    chosen = (os.environ if env is None else env).get(ENV, "system").strip().lower()
    if chosen == "off":
        return Silent()
    if chosen == "console":
        return Console()
    system = system or platform.system()
    if system == "Darwin" and which("osascript"):
        return MacNotifier(run)
    if system == "Linux" and (which("notify-send") or which("zenity")):
        return LinuxNotifier(run, which)
    if system == "Windows":
        return WindowsNotifier()
    return Console()
