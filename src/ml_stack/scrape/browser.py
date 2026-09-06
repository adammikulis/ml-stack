"""A browser that remembers who you are.

Anything worth reading is usually behind a login, and a login worth having survives a reboot.
So the browser runs against a profile directory on disk rather than a fresh sandbox: sign in
once, by hand, and every run after that is already signed in.

That also sets the etiquette. A profile is a real session — the account looks online while the
browser is open, and reading a site fast enough to notice is how a session stops working. Runs
are paced and can be held to hours when somebody is meant to be awake.
"""

from __future__ import annotations

import contextlib
import os
import random
import subprocess
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


class BrowserUnavailable(RuntimeError):
    """Playwright is not installed. `pip install ml-stack[scrape]`, then `playwright install`."""


@dataclass
class Window:
    """How the browser presents itself."""

    profile: Path
    width: int = 1440
    height: int = 900
    headless: bool = True
    channel: str = "chrome"
    timeout_ms: int = 60_000
    position: tuple[int, int] | None = None

    def args(self) -> list[str]:
        """The Chromium flags this window wants."""
        at = self.position if self.position is not None else placed()
        return [f"--window-position={at[0]},{at[1]}"] if at else []



def placed() -> tuple[int, int] | None:
    """Where a headed window opens, from ``ML_STACK_WINDOW_POSITION`` as ``X,Y``."""
    said = os.environ.get("ML_STACK_WINDOW_POSITION", "").strip()
    if not said:
        return None
    try:
        x, y = (int(part) for part in said.split(",", 1))
    except ValueError:
        return None
    return x, y


@contextlib.contextmanager
def keeping_focus() -> Iterator[None]:
    """Give the screen back to whichever application had it."""
    if sys.platform != "darwin":
        yield
        return
    front = _frontmost()
    try:
        yield
    finally:
        if front:
            with contextlib.suppress(Exception):
                subprocess.run(["osascript", "-e", f'tell application "{front}" to activate'],
                               capture_output=True, timeout=10, check=False)


def _frontmost() -> str:
    """The application holding the screen, or ""."""
    try:
        done = subprocess.run(
            ["osascript", "-e",
             "tell application \"System Events\" to name of first process whose frontmost is true"],
            capture_output=True, text=True, timeout=10, check=False)
    except (OSError, subprocess.SubprocessError):
        return ""
    return done.stdout.strip()


@contextlib.contextmanager
def _paged(play: Any, window: Window) -> Iterator[Any]:
    """A page on that Playwright, over a profile directory that persists."""
    profile = Path(window.profile).expanduser()
    profile.mkdir(parents=True, exist_ok=True)
    with keeping_focus():
        context = play.chromium.launch_persistent_context(
            str(profile), headless=window.headless, channel=window.channel,
            args=window.args(),
            viewport={"width": window.width, "height": window.height})
    context.set_default_timeout(window.timeout_ms)
    page = context.pages[0] if context.pages else context.new_page()
    try:
        yield page
    finally:
        with contextlib.suppress(Exception):
            context.close()


@contextlib.contextmanager
def browser(window: Window, play: Any = None) -> Iterator[Any]:
    """A page, on a profile that persists. Closed when the block ends.

    ``play`` is a Playwright already open in this thread; Playwright's sync API allows
    only one per thread, so a caller holding one hands it over instead of opening a
    second. Without one, a Playwright is opened for the block.
    """
    if play is not None:
        with _paged(play, window) as page:
            yield page
        return
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:  # pragma: no cover - depends on what is installed
        raise BrowserUnavailable(str(exc)) from exc
    with sync_playwright() as opened, _paged(opened, window) as page:
        yield page


def sign_in(window: Window, url: str, *, done: str = "", patience_s: float = 600.0) -> bool:
    """Open a window and wait while a person signs in.

    Headed on purpose: this is the one step a person does, and doing it invisibly is how an
    hour goes missing. Returns whether ``done`` — a selector that only exists once signed in —
    turned up before patience ran out.
    """
    headed = Window(**{**window.__dict__, "headless": False})
    with browser(headed) as page:
        page.goto(url, wait_until="domcontentloaded")
        if not done:
            input("sign in, then press enter here: ")
            return True
        deadline = time.monotonic() + patience_s
        while time.monotonic() < deadline:
            with contextlib.suppress(Exception):
                if page.query_selector(done):
                    return True
            time.sleep(1.0)
    return False


def signed_in(page: Any, *, done: str) -> bool:
    """Whether the page shows the thing only a signed-in session shows."""
    with contextlib.suppress(Exception):
        return page.query_selector(done) is not None
    return False


def within_hours(first: int, last: int, *, now: datetime | None = None) -> bool:
    """Whether the clock is inside the hours a run is allowed to be visible.

    A scraper on a real account is a person appearing online. Reading a workspace at four in
    the morning is not invisible; it is conspicuous.
    """
    hour = (now or datetime.now()).hour
    return first <= hour <= last if first <= last else (hour >= first or hour <= last)


def pace(least_s: float = 0.0, most_s: float = 0.0) -> float:
    """Wait a while, unevenly. Returns the seconds spent.

    Evenly spaced requests are a signature. The spread matters more than the length.
    """
    if most_s <= 0:
        return 0.0
    spent = random.uniform(max(0.0, least_s), most_s)
    time.sleep(spent)
    return spent
