"""Walking a page: open it, press what is there, screenshot it, read what it says.

`fleet` walks the daemon's interface -- the first-run steps, then cluster, chat, models,
settings and fit. `graph` walks the rendered graph page's panes. Both take a `Walk` and
give back one `Stop` per screen: its text, its screenshot and every console error it
raised. The browser comes from `ml_stack.scrape.browser`.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

from ml_stack import home
from ml_stack.scrape.browser import Window, browser

__all__ = ["FLEET", "FLEET_STEPS", "FLEET_VIEWS", "GRAPH", "PAGES", "Stop", "Walk",
           "WalkFailed", "fleet", "graph", "screens_of", "walk"]


class WalkFailed(RuntimeError):
    """Nothing answered at the base URL."""


FLEET_STEPS: tuple[tuple[str, str, str], ...] = (
    ("name", "", "Set up this machine"),
    ("clusters", "Continue", "Clusters"),
    ("job", "Continue", "What should this machine do?"),
    ("start", "Continue", "When should it start?"),
    ("run", "Continue", "What should it be able to do?"),
    ("away", "Not now", "When you are not using it"),
    ("done", "Continue", ""),
)
"""Each first-run step: its name, the button that reaches it, the heading it lands on."""

FLEET_WRITES = "run"
"""The first step reached by a button that saves this machine's preferences."""

FLEET_VIEWS = ("cluster", "chat", "models", "settings", "fit")
"""The screens behind the nav bar, by the id of each one's root element."""

FLEET = (*(name for name, _, _ in FLEET_STEPS), "sign-in", *FLEET_VIEWS)

GRAPH = ("graph", "detail", "history", "map", "3d", "search", "legend", "ask", "review")

PAGES: dict[str, tuple[str, ...]] = {"fleet": FLEET, "graph": GRAPH}
"""Which page has which screens."""

BEAT_MS = 300
"""How long a screen is given to start fetching after the button that opened it."""

SETTLE_MS = 8000
"""How long a screen is given to stop fetching before it is read."""


@dataclass(frozen=True, slots=True)
class Walk:
    """One walk: which page, which of its screens, where it is served, where pictures go."""

    page: str = "fleet"
    base: str = ""
    out: Path = Path()
    screens: tuple[str, ...] = ()
    headless: bool = True
    passphrase: str = ""
    setup: bool = False
    ask: str = ""
    find: str = ""
    width: int = 1400
    height: int = 950
    timeout_s: float = 20.0


@dataclass(frozen=True, slots=True)
class Stop:
    """One screen: what it said, where its picture went, and what its console raised."""

    screen: str
    text: str = ""
    shot: Path | None = None
    errors: tuple[str, ...] = ()
    skipped: str = ""

    @property
    def ok(self) -> bool:
        """Whether this screen came up clean."""
        return not self.errors


def screens_of(page: str, asked: Sequence[str]) -> tuple[str, ...]:
    """The screens to walk, in the page's own order; raises on a name it does not have."""
    known = PAGES.get(page)
    if known is None:
        raise ValueError(f"no page called {page!r}; there is "
                         + " and ".join(sorted(PAGES)))
    if not asked:
        return known
    unknown = [name for name in asked if name not in known]
    if unknown:
        raise ValueError(f"the {page} page has no screen called "
                         + ", ".join(repr(name) for name in unknown)
                         + "; it has " + ", ".join(known))
    wanted = set(asked)
    return tuple(name for name in known if name in wanted)


def _first_line(exc: BaseException) -> str:
    said = str(exc).strip()
    return said.splitlines()[0] if said else type(exc).__name__


def _window(what: Walk) -> Window:
    """The browser this walk opens: bundled Chromium, on a profile of its own."""
    return Window(profile=home.cache("walk"), width=what.width, height=what.height,
                  headless=what.headless, channel="chromium",
                  timeout_ms=int(what.timeout_s * 1000))


class Walker:
    """A page under a browser, the screenshots it writes, and what each screen raised."""

    def __init__(self, page: Any, what: Walk) -> None:
        from playwright.sync_api import Error as PageError

        self.page = page
        self.what = what
        self.failure = PageError
        self.stops: list[Stop] = []
        self.seen: list[str] = []
        page.set_default_timeout(what.timeout_s * 1000)
        page.on("pageerror", lambda exc: self.seen.append(f"pageerror: {exc}"))
        page.on("console", self._console)

    def _console(self, message: Any) -> None:
        if message.type == "error" and "Failed to load resource" not in message.text:
            self.seen.append(f"console: {message.text}")

    def taken(self) -> tuple[str, ...]:
        """Everything the console raised since the last call; clears it."""
        out = tuple(self.seen)
        self.seen.clear()
        return out

    def open(self, path: str) -> None:
        """Go to ``path`` on the base URL; raises `WalkFailed` when nothing answers."""
        url = self.what.base.rstrip("/") + path
        try:
            self.page.goto(url, wait_until="domcontentloaded")
        except self.failure as exc:
            raise WalkFailed(f"{url} did not answer: {_first_line(exc)}") from exc

    def wants(self, screen: str) -> bool:
        """Whether this walk was asked for that screen."""
        return screen in self.what.screens

    def skip(self, screen: str, why: str) -> None:
        """Record a screen this page's state does not have."""
        self.stops.append(Stop(screen, skipped=why))

    def failed(self, screen: str, why: str) -> None:
        """Record a screen the walk asked for and did not get."""
        self.stops.append(Stop(screen, errors=(why, *self.taken())))

    def _text(self, selector: str) -> str:
        try:
            return str(self.page.locator(selector).first.inner_text()).strip()
        except self.failure as exc:
            self.seen.append(f"{selector} could not be read: {_first_line(exc)}")
            return ""

    def settle(self) -> None:
        """Wait for the screen to start fetching and then stop, up to `SETTLE_MS`."""
        self.page.wait_for_timeout(BEAT_MS)
        with contextlib.suppress(self.failure):
            self.page.wait_for_load_state("networkidle", timeout=SETTLE_MS)

    def stop(self, screen: str, *selectors: str) -> None:
        """Wait for the screen to settle, screenshot it, record what those selectors say."""
        self.settle()
        taken = sum(1 for one in self.stops if one.shot)
        shot = self.what.out / f"{taken + 1:02d}-{screen}.png"
        shot.parent.mkdir(parents=True, exist_ok=True)
        self.page.screenshot(path=str(shot))
        said = [self._text(one) for one in selectors]
        self.stops.append(Stop(screen, text="\n".join(t for t in said if t), shot=shot,
                               errors=self.taken()))

    def through(self, screen: str, step: Callable[[], Any]) -> bool:
        """Run ``step``, recording nothing unless it fails; False when it never landed."""
        try:
            step()
        except self.failure as exc:
            self.failed(screen, f"not reached: {_first_line(exc)}")
            return False
        return True

    def reached(self, screen: str, step: Callable[[], Any], *selectors: str) -> bool:
        """Run ``step`` and record the screen; False when the step never landed."""
        landed = self.through(screen, step)
        if landed:
            self.stop(screen, *selectors)
        return landed


# -- the fleet interface -------------------------------------------------------------------


def _landing(heading: str) -> str:
    """The selector that says a first-run step has painted."""
    return f'#first-run h1:text-is("{heading}")' if heading else "#first-run pre.cmd"


def _press(page: Any, button: str, landing: str) -> None:
    page.click(f"#first-run button:has-text('{button}')")
    page.wait_for_selector(landing)


def _unwalked(walker: Walker, steps: Sequence[tuple[str, str, str]], why: str) -> None:
    for name, _, _ in steps:
        if walker.wants(name):
            walker.skip(name, why)


def _open_app(page: Any) -> None:
    page.click("#first-run button:has-text('Open')")
    page.wait_for_selector("#signin:not([hidden]), #app:not([hidden])")


def _needed(walker: Walker) -> int:
    """The last first-run step this walk has to press through."""
    beyond = (*FLEET_VIEWS, "sign-in")
    if walker.what.setup and any(walker.wants(name) for name in beyond):
        return len(FLEET_STEPS) - 1
    wanted = [at for at, (name, _, _) in enumerate(FLEET_STEPS) if walker.wants(name)]
    return max(wanted) if wanted else -1


def _setup(walker: Walker) -> None:
    """The first-run steps, in order, as far as the walk is allowed to press."""
    if walker.page.locator("#first-run").is_hidden():
        _unwalked(walker, FLEET_STEPS, "this machine is set up; the wizard is not showing")
        return
    last = _needed(walker)
    for at, (name, button, heading) in enumerate(FLEET_STEPS):
        if at > last:
            return
        if name == FLEET_WRITES and not walker.what.setup:
            _unwalked(walker, FLEET_STEPS[at:],
                      "reaching it saves this machine's preferences; --setup presses the "
                      "buttons that do")
            return
        if not button:
            if walker.wants(name):
                walker.stop(name, "#first-run-card")
            continue
        landed = partial(_press, walker.page, button, _landing(heading))
        got = (walker.reached(name, landed, "#first-run-card") if walker.wants(name)
               else walker.through(name, landed))
        if not got:
            _unwalked(walker, FLEET_STEPS[at + 1:], f"the walk stopped at {name}")
            return
    walker.through("done", partial(_open_app, walker.page))


def _signed(page: Any, passphrase: str) -> None:
    page.fill("#p", passphrase)
    page.click("#signin-go")
    page.wait_for_selector("#app:not([hidden])")


def _sign_in(walker: Walker) -> str:
    """The passphrase screen, and past it; returns why the app is still shut, or ""."""
    page = walker.page
    if page.locator("#signin").is_hidden():
        if walker.wants("sign-in"):
            walker.skip("sign-in", "this machine is in no cluster; nothing asks for a "
                                   "passphrase")
    else:
        if walker.wants("sign-in"):
            walker.stop("sign-in", "#signin")
        if not walker.what.passphrase:
            return "this cluster wants its passphrase; --passphrase signs in"
        try:
            _signed(page, walker.what.passphrase)
        except walker.failure as exc:
            walker.failed("sign-in", f"the passphrase was refused: {_first_line(exc)}")
            return "the passphrase was refused"
    if page.locator("#first-run").is_visible():
        return "this machine is still in first-run; --setup walks the wizard through it"
    return ""


def _tab(page: Any, view: str) -> None:
    page.click(f"#nav-tabs a:text-is('{view.capitalize()}')")
    page.wait_for_selector(f"#{view}:not([hidden])")


def _views(walker: Walker, shut: str) -> None:
    """The screens behind the nav bar, one tab click each."""
    for view in FLEET_VIEWS:
        if not walker.wants(view):
            continue
        if shut:
            walker.skip(view, shut)
        else:
            walker.reached(view, partial(_tab, walker.page, view), f"#{view}")


def fleet(what: Walk, play: Any = None) -> list[Stop]:
    """Walk the daemon's interface; one `Stop` per screen asked for."""
    with browser(_window(what), play) as page:
        walker = Walker(page, what)
        walker.open("/ui/")
        page.wait_for_selector("#first-run:not([hidden]), #signin:not([hidden]), "
                               "#app:not([hidden])")
        _setup(walker)
        _views(walker, _sign_in(walker))
        return walker.stops


# -- the graph page ------------------------------------------------------------------------


def _settled(page: Any) -> None:
    page.wait_for_selector(".graph-wrap:not(.settling)")


def _two_d(page: Any) -> None:
    """The flat view, which is where the nodes and the history chip are."""
    page.click("#v2d")
    _settled(page)


def _legend(page: Any) -> None:
    page.click("#display-btn")
    page.wait_for_selector('#display-btn[aria-expanded="true"]')


def _filter(page: Any, term: str) -> None:
    page.fill("#gq", term)
    _settled(page)


def _node(page: Any) -> None:
    _two_d(page)
    page.locator("#graph .node .hit").first.click()
    page.wait_for_selector("#detail h2")


def _three_d(page: Any) -> None:
    page.click("#v3d")
    page.wait_for_selector('#v3d[aria-pressed="true"]')


def _play(page: Any) -> None:
    _two_d(page)
    page.click("#history")


def _question(page: Any, question: str) -> None:
    page.fill("#q", question)
    page.click("#qform button[type=submit]")
    page.wait_for_selector("#qturns .turn, #qnote:not(:empty)")


def _ask_pane(walker: Walker) -> None:
    """The question box; with `Walk.ask` set, the question is sent and the answer read."""
    if not walker.what.ask:
        walker.stop("ask", "#askpane-title", "#qnote")
        return
    walker.reached("ask", partial(_question, walker.page, walker.what.ask), "#qturns",
                   "#qnote")


def _history_pane(walker: Walker) -> None:
    """The messages played through in order, when the links carry any."""
    if not walker.through("history", partial(_play, walker.page)):
        return
    if walker.page.get_attribute("#history", "aria-pressed") != "true":
        walker.skip("history", "this graph's links carry no messages to play through")
        return
    walker.stop("history", "#history-when", "#hint-text")


def _review_pane(walker: Walker) -> None:
    """The change requests waiting on the graph, when there are any."""
    if walker.page.locator("#review-box").is_hidden():
        walker.skip("review", "no change request is waiting on this graph")
        return
    walker.stop("review", "#review-count", "#review-list")


def graph(what: Walk, play: Any = None) -> list[Stop]:
    """Walk the rendered graph page; one `Stop` per pane asked for."""
    panes: dict[str, tuple[Callable[[Any], None], tuple[str, ...]]] = {
        "graph": (_two_d, ("#title", "#subtitle", "#stats")),
        "detail": (_node, ("#detail",)),
        "map": (_settled, ("#map-box summary",)),
        "3d": (_three_d, ("#hint-text", "#stats")),
        "search": (partial(_filter, term=what.find or "a"), ("#qcount", "#stats")),
        "legend": (_legend, ("#panel",)),
    }
    with browser(_window(what), play) as page:
        walker = Walker(page, what)
        walker.open("/")
        for screen in GRAPH:
            if not walker.wants(screen):
                continue
            if screen == "history":
                _history_pane(walker)
            elif screen == "ask":
                _ask_pane(walker)
            elif screen == "review":
                _review_pane(walker)
            else:
                step, reads = panes[screen]
                walker.reached(screen, partial(step, page), *reads)
        return walker.stops


def walk(what: Walk, play: Any = None) -> list[Stop]:
    """Walk the page ``what`` names; raises `ValueError` for a page nobody serves.

    ``play`` is a Playwright already open in this thread, as `scrape.browser.browser`
    takes one.
    """
    if what.page == "fleet":
        return fleet(what, play)
    if what.page == "graph":
        return graph(what, play)
    raise ValueError(f"no page called {what.page!r}; there is "
                     + " and ".join(sorted(PAGES)))
