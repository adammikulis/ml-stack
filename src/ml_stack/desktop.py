"""Putting a question with real buttons on the person's own screen, and opening a terminal
window there.

``ML_STACK_NOTIFY`` picks the way: ``system`` (default), ``console`` or ``off``; only the
desktop (macOS ``osascript``, Linux ``notify-send`` or ``zenity``) shows anything. The text is
cleaned, passed as an argument after ``--`` and never spliced into a script, and the button
labels are the caller's fixed constants. The test suite sets ``console`` and puts shims for
every desktop tool first on PATH; a shim records the attempt and the run fails.
"""

from __future__ import annotations

import logging
import os
import platform
import re
import shutil
import subprocess
import unicodedata
from collections.abc import Callable, Mapping, Sequence

__all__ = ["ENV", "WAIT_S", "Result", "Runner", "choose", "clean", "open_terminal", "run", "which_way"]

ENV = "ML_STACK_NOTIFY"
WAIT_S = 120
logger = logging.getLogger("ml_stack.desktop")
logger.addHandler(logging.NullHandler())

Result = tuple[int, str, str]
Runner = Callable[[Sequence[str]], Result]

MAC_CHOOSE = ("on run argv\n"
              "set answer to display alert (item 1 of argv) message (item 2 of argv) "
              "buttons {item 3 of argv, item 4 of argv} default button (item 3 of argv) "
              f"cancel button (item 3 of argv) giving up after {WAIT_S}\n"
              "return answer\n"
              "end run")


def clean(text: object, most: int = 64) -> str:
    """Text as one short printable line: control characters and bidi overrides dropped,
    spaces collapsed, cut at ``most``."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"[\r\n\t\v\f]+", " ", text)
    kept = "".join(ch for ch in text if ch == " " or unicodedata.category(ch)[0] not in "CZ")
    return " ".join(kept.split())[:most]


def run(argv: Sequence[str]) -> Result:
    """Run a desktop tool and wait for it; a failure to start is a result, not an exception."""
    try:
        done = subprocess.run(list(argv), capture_output=True, text=True,
                              timeout=WAIT_S + 15, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning("desktop command failed: %s", type(exc).__name__)
        return 1, "", type(exc).__name__
    return done.returncode, done.stdout, done.stderr


def which_way(system: str | None = None, *, env: Mapping[str, str] | None = None,
              which: Callable[[str], str | None] = shutil.which) -> str:
    """``macos``, ``notify-send``, ``zenity`` or ``none``: how a dialog can be shown here,
    honouring ``ML_STACK_NOTIFY``."""
    chosen = (os.environ if env is None else env).get(ENV, "system").strip().lower()
    system = system or platform.system()
    if chosen != "system":
        return "none"
    if system == "Darwin" and which("osascript"):
        return "macos"
    if system == "Linux":
        return next((tool for tool in ("notify-send", "zenity") if which(tool)), "none")
    return "none"


def choose(title: str, body: str, buttons: tuple[str, str], *, way: str | None = None,
           runner: Runner = run) -> str:
    """Show ``title`` and ``body`` with two buttons (``buttons``: the safe one first) and
    return the label pressed, ``timeout`` or ``unavailable``. Escape and closing the window
    answer with the first button."""
    way = which_way() if way is None else way
    title, body = clean(title, 80), clean(body, 300)
    if way == "macos":
        return _mac(runner, title, body, buttons)
    if way == "notify-send":
        return _notify_send(runner, title, body, buttons)
    if way == "zenity":
        return _zenity(runner, title, body, buttons)
    return "unavailable"


def _mac(runner: Runner, title: str, body: str, buttons: tuple[str, str]) -> str:
    code, out, err = runner(["osascript", "-e", MAC_CHOOSE, "--", title, body, *buttons])
    if code != 0:
        return buttons[0] if "-128" in err else "unavailable"
    if re.search(r"gave up:\s*true", out):
        return "timeout"
    got = re.search(r"button returned:\s*(.*?)(?:,\s*gave up:|$)", out.strip())
    return got[1] if got and got[1] in buttons else buttons[0]


def _notify_send(runner: Runner, title: str, body: str, buttons: tuple[str, str]) -> str:
    code, out, _ = runner(["notify-send", "--app-name=ml-stack", "--urgency=critical",
                           "--wait", f"--expire-time={WAIT_S * 1000}",
                           f"--action=0={buttons[0]}", f"--action=1={buttons[1]}",
                           "--", title, body])
    if code == 0 and out.strip() in ("0", "1"):
        return buttons[int(out.strip())]
    return "timeout" if code == 0 else "unavailable"


def _zenity(runner: Runner, title: str, body: str, buttons: tuple[str, str]) -> str:
    code, out, _ = runner(["zenity", "--list", "--title", title, "--text", body,
                           "--column", "Answer", *buttons, f"--timeout={WAIT_S}"])
    if code == 5:
        return "timeout"
    return out.strip() if code == 0 and out.strip() in buttons else buttons[0]


def open_terminal(command: Sequence[str], *, way: str | None = None, runner: Runner = run,
                  which: Callable[[str], str | None] = shutil.which) -> bool:
    """Open a terminal window running ``command`` (a fixed program path, no held text).
    macOS opens the file with Terminal; Linux uses the first terminal emulator found."""
    way = which_way(which=which) if way is None else way
    if way == "macos" and which("open") and len(command) == 1:
        return runner(["open", "-a", "Terminal", command[0]])[0] == 0
    if way in ("notify-send", "zenity"):
        for term in ("x-terminal-emulator", "gnome-terminal", "konsole", "xterm"):
            if which(term):
                flag = "--" if term == "gnome-terminal" else "-e"
                return runner([term, flag, *command])[0] == 0
    return False
