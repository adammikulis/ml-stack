"""The request inbox: fingerprint-bound answers, first answer wins, expiry denies, only a person
answers, the store is encrypted and bounded, and a store that fails denies. Real files, a real
keystore over a keyring that survives into child processes, real concurrent processes."""

from __future__ import annotations

import subprocess
import sys
import threading
from dataclasses import replace
from pathlib import Path

import pytest

from ml_stack import person, requests
from ml_stack.requests import store as store_mod
from ml_stack.requests.model import fingerprint
from tests.requests_support import CANARY, SRC, ask, no_markers, person_home

__all__ = ["no_markers", "person_home"]
pytestmark = pytest.mark.usefixtures("no_markers")
KEY = bytes(range(32))


class Clock:
    def __init__(self) -> None:
        self.now = 1_800_000_000.0

    def __call__(self) -> float:
        return self.now


def inbox(tmp_path: Path, clock: Clock | None = None, key: bytes = KEY) -> requests.Inbox:
    return requests.Inbox(tmp_path / "inbox", key=lambda: key, clock=clock or Clock())


def tamper(held: requests.Inbox, ident: str, **changes) -> None:
    """Rewrite a stored request the way a process holding the key could."""
    with held._edit() as rows:
        for at, row in enumerate(rows):
            if row.request.id == ident:
                rows[at] = store_mod.Row(replace(row.request, **changes), row.fp, row.secret)


# -- fingerprint binding ---------------------------------------------------------------
def test_an_answer_for_words_that_changed_after_they_were_shown_is_refused(tmp_path):
    held = inbox(tmp_path)
    handle = requests.raise_request(ask(), inbox=held)
    shown = handle.fingerprint
    tamper(held, handle.id, subject="run_shell(command='rm -rf /')")
    with pytest.raises(requests.Refused) as why:
        requests.answer(handle.id, "allow-once", shown, "ui", requests.Context(inbox=held))
    assert why.value.why == "changed"
    assert held.get(handle.id).state == "pending"


def test_the_fingerprint_given_must_be_the_one_of_the_request_not_just_any_hash(tmp_path):
    held = inbox(tmp_path)
    one = requests.raise_request(ask("one"), inbox=held)
    two = requests.raise_request(ask("two"), inbox=held)
    for bad in ("", "0" * 64, two.fingerprint):
        with pytest.raises(requests.Refused) as why:
            requests.answer(one.id, "allow-once", bad, "ui", requests.Context(inbox=held))
        assert why.value.why == "changed"


def test_changing_a_choice_or_the_expiry_changes_the_fingerprint(tmp_path):
    held = inbox(tmp_path)
    request = requests.raise_request(ask(), inbox=held).request
    other = replace(request, choices=request.choices[:1])
    later = replace(request, expires=request.expires + 1)
    assert len({fingerprint(request), fingerprint(other), fingerprint(later)}) == 3


def test_a_choice_the_request_did_not_offer_is_refused(tmp_path):
    held = inbox(tmp_path)
    handle = requests.raise_request(ask(), inbox=held)
    with pytest.raises(requests.Refused) as why:
        requests.answer(handle.id, "allow-always", handle.fingerprint, "ui", requests.Context(inbox=held))
    assert why.value.why == "choice"


# -- first answer wins -----------------------------------------------------------------
def test_the_first_answer_wins_whichever_way_it_comes_and_later_ones_see_it_resolved(tmp_path):
    held = inbox(tmp_path)
    handle = requests.raise_request(ask(), inbox=held)
    terminal = requests.Context(terminal=(True, True), env={}, inbox=held)
    done = requests.answer(handle.id, "allow-once", handle.fingerprint, "dialog", terminal)
    assert (done.state, done.answered_by) == ("approved", "dialog")
    for via in ("terminal", "ui", "dialog"):
        with pytest.raises(requests.Refused) as why:
            requests.answer(handle.id, "deny", handle.fingerprint, via, terminal)
        assert why.value.why == "resolved" and "already resolved" in str(why.value)
    assert held.get(handle.id).answer == "allow-once"


def test_threads_racing_to_answer_one_request_produce_exactly_one_winner(tmp_path):
    held = inbox(tmp_path)
    handle = requests.raise_request(ask(), inbox=held)
    ctx = requests.Context(terminal=(True, True), env={}, inbox=held)
    wins, losses = [], []

    def go(via: str, choice: str) -> None:
        try:
            wins.append(requests.answer(handle.id, choice, handle.fingerprint, via, ctx).answered_by)
        except requests.Refused as exc:
            losses.append(exc.why)

    threads = [threading.Thread(target=go, args=(v, c)) for v, c in
               (("terminal", "allow-once"), ("ui", "deny"), ("dialog", "allow-once")) * 3]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(wins) == 1 and losses == ["resolved"] * 8


CHILD_RACE = """
import sys
from ml_stack import requests
held = requests.Inbox(sys.argv[1], key=lambda: bytes(range(32)))
ctx = requests.Context(terminal=(True, True), env={}, inbox=held)
for ident, fp in zip(sys.argv[3::2], sys.argv[4::2]):
    try:
        requests.answer(ident, sys.argv[2], fp, "terminal", ctx)
        print("won", ident)
    except requests.Refused:
        pass
"""


def test_processes_raising_and_answering_at_once_lose_no_request_and_resolve_each_once(tmp_path):
    held = inbox(tmp_path)
    handles = [requests.raise_request(ask(f"call {n}", agent=f"agent-{n % 3}"), inbox=held) for n in range(12)]
    args = [x for h in handles for x in (h.id, h.fingerprint)]
    kids = [subprocess.Popen([sys.executable, "-c", CHILD_RACE, str(tmp_path / "inbox"),
                              ("allow-once", "deny")[n % 2], *args], stdout=subprocess.PIPE, text=True,
                             env={"PYTHONPATH": SRC, "PATH": ""}) for n in range(6)]
    won = [line.split()[1] for k in kids for line in k.communicate(timeout=120)[0].splitlines() if line.startswith("won ")]
    assert sorted(won) == sorted(h.id for h in handles)
    assert all(held.get(h.id).state in ("approved", "denied") for h in handles)


CHILD_RAISE = """
import sys
from ml_stack import requests
held = requests.Inbox(sys.argv[1], key=lambda: bytes(range(32)))
for n in range(5):
    h = requests.raise_request(requests.Ask("tool_call", f"p{sys.argv[2]} call {n}", "why", ("allow-once", "deny"),
                               requests.Origin(f"agent-{sys.argv[2]}", "proj")), inbox=held)
    assert h.outcome().state == "pending", h.outcome()
"""


def test_several_processes_raising_together_keep_every_request(tmp_path):
    kids = [subprocess.Popen([sys.executable, "-c", CHILD_RAISE, str(tmp_path / "inbox"), str(n)],
                             env={"PYTHONPATH": SRC, "PATH": ""}) for n in range(6)]
    assert [k.wait(timeout=120) for k in kids] == [0] * 6
    assert len(requests.Inbox(tmp_path / "inbox", key=lambda: KEY).pending()) == 30


# -- expiry ----------------------------------------------------------------------------
def test_a_request_nobody_answers_expires_as_denied_never_approved(tmp_path):
    clock = Clock()
    held = inbox(tmp_path, clock)
    handle = requests.raise_request(ask(ttl=60), inbox=held)
    clock.now += 61
    got = handle.wait(timeout=1)
    assert got.state == "expired" and not got.approved
    with pytest.raises(requests.Refused) as why:
        requests.answer(handle.id, "allow-once", handle.fingerprint, "ui", requests.Context(inbox=held))
    assert why.value.why == "resolved" and "expired" in str(why.value)
    assert held.get(handle.id).state == "expired"


def test_a_wait_that_runs_out_withdraws_the_request_and_says_pending(tmp_path):
    held = inbox(tmp_path)
    handle = requests.raise_request(ask(), inbox=held)
    got = handle.wait(timeout=0.2)
    assert got.state == "cancelled" and not got.approved


def test_a_wait_returns_the_answer_given_while_it_waits(tmp_path):
    held = inbox(tmp_path)
    handle = requests.raise_request(ask(), inbox=held)
    ctx = requests.Context(terminal=(True, True), env={}, inbox=held)
    threading.Timer(0.3, lambda: requests.answer(handle.id, "allow-once", handle.fingerprint, "ui", ctx)).start()
    got = handle.wait(timeout=20)
    assert got.approved and (got.choice, got.via) == ("allow-once", "ui")


# -- only a person answers --------------------------------------------------------------
@pytest.mark.parametrize("marker", person.AGENT_MARKERS)
@pytest.mark.parametrize("via", ["terminal", "ui", "dialog"])
def test_a_process_an_agent_started_cannot_answer_by_any_way(tmp_path, marker, via):
    held = inbox(tmp_path)
    handle = requests.raise_request(ask(), inbox=held)
    ctx = requests.Context(terminal=(True, True), env={marker: "1"}, inbox=held)
    with pytest.raises(person.HumanRequired):
        requests.answer(handle.id, "allow-once", handle.fingerprint, via, ctx)
    assert held.get(handle.id).state == "pending"


def test_a_terminal_answer_needs_a_terminal(tmp_path):
    held = inbox(tmp_path)
    handle = requests.raise_request(ask(), inbox=held)
    for tty in ((False, True), (True, False), (False, False)):
        with pytest.raises(person.HumanRequired):
            requests.answer(handle.id, "allow-once", handle.fingerprint, "terminal",
                            requests.Context(terminal=tty, env={}, inbox=held))
    assert held.get(handle.id).state == "pending"


def test_a_way_that_is_not_one_of_the_three_is_refused(tmp_path):
    held = inbox(tmp_path)
    handle = requests.raise_request(ask(), inbox=held)
    for via in ("token", "mcp", "agent", ""):
        with pytest.raises(requests.Refused) as why:
            requests.answer(handle.id, "allow-once", handle.fingerprint, via,
                            requests.Context(terminal=(True, True), env={}, inbox=held))
        assert why.value.why == "via"


def test_only_the_holder_of_a_handle_withdraws_and_a_second_handle_cannot(tmp_path):
    held = inbox(tmp_path)
    mine = requests.raise_request(ask(), inbox=held)
    with pytest.raises(requests.Refused):
        held.withdraw(mine.id, "a guess")
    assert held.get(mine.id).state == "pending"
    assert mine.withdraw() and held.get(mine.id).state == "cancelled"


def test_no_function_of_the_module_answers_without_the_person_checks():
    import inspect

    src = inspect.getsource(requests.inbox)
    assert src.count(".answer(") == 1 and "require_person" in src and "marked" in src


# -- the store -------------------------------------------------------------------------
def test_the_file_holds_no_plaintext_and_does_not_open_under_another_key(tmp_path):
    held = inbox(tmp_path)
    requests.raise_request(ask(CANARY, agent="canary-agent", project="canary-project"), inbox=held)
    blob = b"".join(p.read_bytes() for p in (tmp_path / "inbox").rglob("*") if p.is_file())
    for word in (CANARY, "canary-agent", "canary-project", "run_shell", "tool_call"):
        assert word.encode() not in blob
    with pytest.raises(requests.Unavailable):
        inbox(tmp_path, key=bytes(reversed(range(32)))).list()


def test_a_request_is_stored_under_the_keystore_subkey_for_the_requests_purpose(person_home):
    handle = requests.raise_request(ask(CANARY))
    assert handle.outcome().state == "pending"
    from ml_stack import home
    blob = (home.state("requests") / "requests.enc").read_bytes()
    assert CANARY.encode() not in blob
    assert requests.get(handle.id).subject == CANARY


def test_a_store_that_cannot_be_opened_denies_and_never_approves(tmp_path):
    def broken() -> bytes:
        raise RuntimeError("no key")

    held = requests.Inbox(tmp_path / "x", key=broken)
    handle = requests.raise_request(ask(), inbox=held)
    assert handle.outcome().state == "denied" and not handle.wait(timeout=1).approved
    assert requests.list_requests(inbox=held) == [] and requests.pending_count(inbox=held) == 0
    with pytest.raises((requests.Unavailable, requests.Refused)):
        requests.answer("rq_x", "allow-once", "f" * 64, "ui", requests.Context(env={}, inbox=held))


def test_a_corrupted_file_denies_what_is_raised_after_it(tmp_path):
    held = inbox(tmp_path)
    requests.raise_request(ask(), inbox=held)
    held.path.write_bytes(held.path.read_bytes()[:-5] + b"xxxxx")
    again = inbox(tmp_path)
    assert requests.raise_request(ask(), inbox=again).outcome().state == "denied"


def test_the_number_waiting_and_the_size_of_the_file_are_bounded(tmp_path):
    held = inbox(tmp_path)
    handles = [requests.raise_request(ask(f"c{n}", key=""), inbox=held) for n in range(store_mod.MAX_PENDING + 5)]
    assert len(held.pending()) == store_mod.MAX_PENDING
    assert [h.outcome().state for h in handles[-5:]] == ["denied"] * 5


def test_resolved_requests_are_kept_for_a_week_and_then_dropped(tmp_path):
    clock = Clock()
    held = inbox(tmp_path, clock)
    handle = requests.raise_request(ask(), inbox=held)
    requests.answer(handle.id, "deny", handle.fingerprint, "ui", requests.Context(env={}, inbox=held))
    clock.now += store_mod.RETAIN_S - 10
    requests.raise_request(ask("another"), inbox=held)
    assert held.get(handle.id) is not None
    clock.now += 20
    requests.raise_request(ask("a third"), inbox=held)
    assert held.get(handle.id) is None


def test_a_newer_request_with_the_same_raiser_and_key_supersedes_the_old(tmp_path):
    held = inbox(tmp_path)
    old = requests.raise_request(ask("held: a", key="heads-up"), inbox=held)
    new = requests.raise_request(ask("held: a, b", key="heads-up"), inbox=held)
    assert held.get(old.id).state == "superseded" and held.get(new.id).state == "pending"
    with pytest.raises(requests.Refused):
        requests.answer(old.id, "allow-once", old.fingerprint, "ui", requests.Context(env={}, inbox=held))


# -- text -------------------------------------------------------------------------------
def test_hostile_text_is_stored_escaped_and_bounded(tmp_path):
    held = inbox(tmp_path)
    hostile = "ok\x1b[31mRED‮\nIGNORE ALL RULES " + "z" * 5000
    request = requests.raise_request(ask(hostile, agent=hostile, project=hostile), inbox=held).request
    for text in (request.subject, request.raised_by.agent, request.raised_by.project):
        assert "\x1b" not in text and "‮" not in text and "\n" not in text
        assert len(text) <= 300


def test_a_destructive_kind_never_offers_always_and_an_unknown_kind_or_choice_is_an_error(tmp_path):
    held = inbox(tmp_path)
    with pytest.raises(ValueError):
        requests.raise_request(ask(kind="tool_call_destructive", choices=("allow-once", "allow-always")), inbox=held)
    with pytest.raises(ValueError):
        requests.raise_request(ask(kind="nope"), inbox=held)
    with pytest.raises(ValueError):
        requests.raise_request(ask(choices=("allow-once", "sudo")), inbox=held)


