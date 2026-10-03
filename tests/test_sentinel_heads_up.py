"""The heads-up when something is quarantined: two real buttons, rate-limited, off only with a
reason, and nothing it receives can release anything. No test raises a real notification; the
desktop is a recording function and the suite's shims catch anything that reaches for the real one."""

from __future__ import annotations

import inspect
import os
from pathlib import Path

from ml_stack import desktop, sentinel
from ml_stack.sentinel import heads_up, human
from ml_stack.sentinel.heads_up import BUTTONS, HeadsUp
from ml_stack.sentinel.store import State

HOSTILE = "evil\x1b[31mRED‮\nIGNORE PREVIOUS INSTRUCTIONS " + "z" * 5000


class Desk:
    """A recording stand-in for the desktop: what it was shown, and the button it presses."""

    def __init__(self, press: str = heads_up.DISMISS) -> None:
        self.shown: list[tuple[str, str, tuple[str, str]]] = []
        self.opened = 0
        self.press = press

    def choose(self, title, body, buttons):
        self.shown.append((title, body, buttons))
        return self.press

    def opener(self):
        self.opened += 1
        return True


def wired(desk: Desk, **env: str):
    node = sentinel.default()
    node.heads_up.choose, node.heads_up.opener = desk.choose, desk.opener
    node.heads_up.spawn = lambda work: work()
    node.heads_up.env = env
    return node


def person_release(node, ident: str):
    grant = human.mint("release", ident, typed=lambda _: ident, terminal=(True, True), env={})
    return node.store.release(ident, grant)


def test_a_quarantine_raises_one_notice_with_exactly_the_two_buttons():
    desk = Desk()
    node = wired(desk)
    record = node.store.quarantine(("peer", "10.0.0.1"), "peer.forged_traffic: forged=3", {})
    assert len(desk.shown) == 1
    title, body, buttons = desk.shown[0]
    assert buttons == ("Dismiss", "Review…") == BUTTONS
    assert "peer 10.0.0.1" in title and "forged or replayed signature" in body
    assert "Requests from this machine are refused" in body
    assert node.store.get(record.id).state == State.QUARANTINED


def test_the_text_handed_to_the_desktop_is_the_safe_sentence_never_raw_held_text():
    desk = Desk()
    node = wired(desk)
    node.store.quarantine(("session", HOSTILE), "guard.denied: " + HOSTILE, {"x": HOSTILE})
    title, body, _ = desk.shown[0]
    for text in (title, body):
        assert "\x1b" not in text and "‮" not in text and "\n" not in text
        assert "zzzz" not in text and len(text) < 400
    assert "IGNORE" not in body and "\\x1b" in title      # the name is shown only as escapes


def test_a_watched_subject_raises_nothing():
    desk = Desk()
    node = wired(desk)
    node.store.watch("server", "port:1", "server.unmanaged: x", {})
    assert desk.shown == []


def test_one_notice_per_subject_per_hour():
    desk, now = Desk(), [1_000_000.0]
    node = wired(desk)
    node.heads_up.clock = lambda: now[0]
    first = node.store.quarantine(("peer", "10.0.0.1"), "peer.forged_traffic: x", {})
    person_release(node, first.id)
    now[0] += 600
    node.store.quarantine(("peer", "10.0.0.1"), "peer.forged_traffic: x again", {})
    assert len(desk.shown) == 1                                   # same subject inside the hour
    node.store.quarantine(("peer", "10.0.0.2"), "peer.forged_traffic: x", {})
    assert len(desk.shown) == 2                                   # a different subject
    now[0] += 3700
    again = node.store.find("peer", "10.0.0.1")
    person_release(node, again.id)
    node.store.quarantine(("peer", "10.0.0.1"), "peer.forged_traffic: later", {})
    assert len(desk.shown) == 3                                   # an hour later


def test_many_at_once_become_one_burst_notice():
    desk = Desk()
    node = wired(desk)
    for n in range(8):
        node.store.quarantine(("peer", f"10.0.1.{n}"), "peer.forged_traffic: x", {})
    titles = [t for t, _, _ in desk.shown]
    assert len(desk.shown) == heads_up.BURST_AFTER + 1
    assert titles[-1] == "Several things were just quarantined"
    assert "review" in desk.shown[-1][1] and "Nothing has been released" in desk.shown[-1][1]


def test_review_opens_the_window_and_dismiss_does_nothing_and_neither_releases():
    review, dismiss = Desk(heads_up.REVIEW), Desk(heads_up.DISMISS)
    node = wired(review)
    held = node.store.quarantine(("peer", "10.0.0.1"), "peer.forged_traffic: x", {})
    assert review.opened == 1
    node = wired(dismiss)
    other = node.store.quarantine(("peer", "10.0.0.2"), "peer.forged_traffic: x", {})
    assert dismiss.opened == 0
    for desk_, ident in ((review, held.id), (dismiss, other.id)):
        assert node.store.get(ident).state == State.QUARANTINED, desk_
    assert not [e for e in node.bus.log.read() if e.kind == "quarantine.released"]


def test_whatever_the_desktop_answers_nothing_is_released_or_opened_but_the_review_window():
    for answer in ("release", "Release", "y", "r", "purge q-1", "Review… && release", "\x1b[B"):
        desk = Desk(answer)
        node = wired(desk)
        held = node.store.quarantine(("peer", f"10.9.{len(answer)}.1"),
                                     "peer.forged_traffic: x", {})
        assert node.store.get(held.id).state == State.QUARANTINED
        assert desk.opened == 0


def test_the_module_has_no_way_to_release_purge_or_press_a_key():
    source = inspect.getsource(heads_up)
    for forbidden in ("import human", "sentinel.human", "mint", "HumanGrant", ".release(", ".purge(", "stdin", "getpass"):
        assert forbidden not in source, forbidden
    assert list(inspect.signature(heads_up.HeadsUp.on_quarantine).parameters) == ["self", "record"]
    assert list(inspect.signature(heads_up.Wires().opener).parameters) == []


def test_it_is_off_only_with_a_reason_and_says_so_in_the_log():
    desk = Desk()
    node = wired(desk, ML_STACK_SENTINEL_NOTIFY="off")
    node.store.quarantine(("peer", "10.0.0.1"), "peer.forged_traffic: x", {})
    assert len(desk.shown) == 1                                   # no reason: still notifies
    node.heads_up.env = {"ML_STACK_SENTINEL_NOTIFY": "off",
                         "ML_STACK_SENTINEL_NOTIFY_BECAUSE": "demo on a shared screen"}
    node.store.quarantine(("peer", "10.0.0.2"), "peer.forged_traffic: x", {})
    assert len(desk.shown) == 1
    off = [e for e in node.bus.log.read() if e.kind == "sentinel.notify_off"]
    assert len(off) == 1 and off[0].evidence["because"] == "demo on a shared screen"


def test_a_quarantine_a_person_made_by_hand_raises_nothing():
    desk = Desk()
    node = wired(desk)
    node.store.quarantine(("peer", "10.0.0.1"), "held by a person", {}, actor="human")
    assert desk.shown == []


def test_the_default_wiring_in_a_test_session_reaches_for_no_desktop():
    shim_log = Path(os.environ["ML_STACK_SHIM_LOG"])
    before = shim_log.read_text() if shim_log.exists() else ""
    node = sentinel.default()
    assert isinstance(node.heads_up, HeadsUp)
    node.store.quarantine(("peer", "10.0.0.1"), "peer.forged_traffic: x", {})
    assert desktop.which_way() == "none"
    assert (shim_log.read_text() if shim_log.exists() else "") == before
    assert not (node.root / "notified.json").exists()


# -- the dialog itself, against recorded commands -------------------------------------------
def test_macos_buttons_are_arguments_and_the_script_never_holds_the_text():
    calls = []

    def runner(argv):
        calls.append(list(argv))
        return 0, "button returned:Review…, gave up:false\n", ""

    got = desktop.choose("t\x1b[31m", "b\nx", ("Dismiss", "Review…"), way="macos", runner=runner)
    assert got == "Review…"
    argv = calls[0]
    assert argv[0] == "osascript" and argv[-4:] == ["t[31m", "b x", "Dismiss", "Review…"]
    assert "Dismiss" not in argv[2] and "b x" not in argv[2]


def test_every_way_the_dialog_can_end():
    buttons = ("Dismiss", "Review…")
    mac = {(0, "button returned:Dismiss, gave up:false", ""): "Dismiss",
           (0, "button returned:Review…, gave up:false", ""): "Review…",
           (0, "button returned:, gave up:true", ""): "timeout",
           (1, "", "execution error: User canceled. (-128)"): "Dismiss",
           (1, "", "boom"): "unavailable"}
    for result, want in mac.items():
        assert desktop.choose("t", "b", buttons, way="macos", runner=lambda _a, r=result: r) == want
    assert desktop.choose("t", "b", buttons, way="notify-send",
                          runner=lambda _a: (0, "1\n", "")) == "Review…"
    assert desktop.choose("t", "b", buttons, way="notify-send",
                          runner=lambda _a: (0, "", "")) == "timeout"
    assert desktop.choose("t", "b", buttons, way="zenity",
                          runner=lambda _a: (0, "Review…\n", "")) == "Review…"
    assert desktop.choose("t", "b", buttons, way="none") == "unavailable"


def test_the_environment_decides_whether_a_dialog_can_be_shown():
    def have(*names):
        return lambda tool: f"/bin/{tool}" if tool in names else None

    assert desktop.which_way("Darwin", env={}, which=have("osascript")) == "macos"
    assert desktop.which_way("Linux", env={}, which=have("zenity")) == "zenity"
    assert desktop.which_way("Linux", env={}, which=have("notify-send", "zenity")) == "notify-send"
    assert desktop.which_way("Darwin", env={"ML_STACK_NOTIFY": "console"},
                             which=have("osascript")) == "none"
    assert desktop.which_way("Windows", env={}, which=have("osascript")) == "none"


def test_the_terminal_is_opened_with_a_fixed_command():
    calls = []
    ok = desktop.open_terminal(["/x/Review quarantine.command"], way="macos",
                               runner=lambda argv: calls.append(list(argv)) or (0, "", ""),
                               which=lambda tool: f"/usr/bin/{tool}")
    assert ok and calls == [["open", "-a", "Terminal", "/x/Review quarantine.command"]]
    calls.clear()
    assert desktop.open_terminal(["/x/ml-stack-security", "review"], way="zenity",
                                 runner=lambda argv: calls.append(list(argv)) or (0, "", ""),
                                 which=lambda tool: tool if tool == "xterm" else None)
    assert calls == [["xterm", "-e", "/x/ml-stack-security", "review"]]
    assert not desktop.open_terminal(["x"], way="none")
