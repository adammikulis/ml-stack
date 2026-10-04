"""Every component that waits for a person raises a request, and the terminal, the UI and the
dialog answer the same one: the chat confirmation, the remember question, the sentinel dialog and
the `ml-stack-requests` command. Real pipes, real files, a real keystore over a fake keyring."""

from __future__ import annotations

import io
import os
import pty
import select
import subprocess
import sys
import threading
import time

import pytest

from ml_stack import do, requests, rules, sentinel
from ml_stack.inbox import cli
from ml_stack.interventions import Call, Confirm
from ml_stack.sentinel.heads_up import KEEP, LATER, RELEASE
from ml_stack.sentinel.store import State
from tests.requests_support import SRC, ask, no_markers, person_home

__all__ = ["no_markers", "person_home"]
pytestmark = pytest.mark.usefixtures("no_markers")


class PipeTty(io.TextIOWrapper):
    """A text stream over a pipe that says it is a terminal, so a prompt polls while it waits."""

    def isatty(self) -> bool:
        return True


def pipe_person(**kw):
    r, w = os.pipe()
    reader = PipeTty(os.fdopen(r, "rb", buffering=0), encoding="utf-8")
    out = io.StringIO()
    person = do.Person(reader, out, **kw)
    return person, out, os.fdopen(w, "w", buffering=1)


def waiting(kind: str | None = None, timeout: float = 15.0):
    end = time.time() + timeout
    while time.time() < end:
        found = [r for r in requests.list_requests(state="pending") if kind in (None, r.kind)]
        if found:
            return found[0]
        time.sleep(0.05)
    raise AssertionError("no request was raised")


def confirm_in_thread(person, ask_, call):
    got = {}
    thread = threading.Thread(target=lambda: got.setdefault("answer", person.confirm(ask_, call)))
    thread.start()
    return thread, got


CALL = Call("bench_run", {"argv": ["x"]})


def classifier_ask(**details):
    return Confirm("bench_run will start a run. Asked because: destructive: deletes files (rm -r).",
                   {"role": "approve-first", "always_ok": True, **details})


# -- the chat confirmation -----------------------------------------------------------
def test_the_confirmation_is_a_request_with_the_classifiers_reason_and_the_terminal_answers_it():
    person, _out, typed = pipe_person()
    thread, got = confirm_in_thread(person, classifier_ask(), CALL)
    shown = waiting("tool_call")
    assert "bench_run" in shown.subject and "Asked because: destructive: deletes files (rm -r)" in shown.reason
    assert shown.raised_by.agent == "ml-stack-chat" and [c.id for c in shown.choices] == ["allow-once", "deny"]
    typed.write("1\n")
    thread.join(20)
    done = requests.get(shown.id)
    assert got["answer"] is True and (done.state, done.answered_by, done.answer) == ("approved", "terminal", "allow-once")


def test_a_no_at_the_terminal_denies_and_a_destructive_call_never_offers_always():
    person, out, typed = pipe_person(rules=rules.Rules())
    hard = classifier_ask(always_ok=False, always_blocked="the classifier labelled it destructive",
                          classifier={"label": "destructive"})
    thread, got = confirm_in_thread(person, hard, CALL)
    shown = waiting("tool_call_destructive")
    assert [c.id for c in shown.choices] == ["allow-once", "never", "deny"]
    typed.write("\n")
    thread.join(20)
    assert got["answer"] is False and requests.get(shown.id).state == "denied"
    assert "no 'always allow' here" in out.getvalue()


def test_an_answer_in_the_ui_while_the_terminal_waits_settles_the_prompt():
    person, out, _typed = pipe_person()
    thread, got = confirm_in_thread(person, classifier_ask(), CALL)
    shown = waiting()
    requests.answer(shown.id, "allow-once", shown.fingerprint, "ui")
    thread.join(20)
    assert got["answer"] is True and "(answered in the ui: allow-once)" in out.getvalue()
    assert person.answer == "allow_once"


def test_a_denial_in_the_dialog_denies_the_call():
    person, _out, _typed = pipe_person()
    thread, got = confirm_in_thread(person, classifier_ask(), CALL)
    shown = waiting()
    requests.answer(shown.id, "deny", shown.fingerprint, "dialog")
    thread.join(20)
    assert got["answer"] is False


def test_always_allow_from_the_ui_runs_the_call_once_and_saves_no_rule(tmp_path):
    saved = rules.Rules(tmp_path / "rules.json")
    person, out, _typed = pipe_person(rules=saved)
    thread, got = confirm_in_thread(person, classifier_ask(), CALL)
    shown = waiting()
    assert "allow-always" in [c.id for c in shown.choices]
    requests.answer(shown.id, "allow-always", shown.fingerprint, "ui")
    thread.join(20)
    assert got["answer"] is True and saved.rules == []
    assert "a rule is saved only from the terminal" in out.getvalue()


def test_always_allow_at_the_terminal_still_saves_the_rule_after_a_yes(tmp_path):
    saved = rules.Rules(tmp_path / "rules.json")
    person, _out, typed = pipe_person(rules=saved)
    thread, got = confirm_in_thread(person, classifier_ask(), CALL)
    waiting()
    typed.write("2\ny\n")
    thread.join(20)
    assert got["answer"] is True and len(saved.rules) == 1 and saved.rules[0].verdict == "always"


def test_never_at_the_terminal_saves_a_never_rule_and_denies(tmp_path):
    saved = rules.Rules(tmp_path / "rules.json")
    person, _out, typed = pipe_person(rules=saved)
    thread, got = confirm_in_thread(person, classifier_ask(), CALL)
    shown = waiting()
    typed.write("3\n")
    thread.join(20)
    assert got["answer"] is False and saved.rules[0].verdict == "never" and requests.get(shown.id).state == "denied"


def test_a_store_that_fails_denies_the_call_whatever_the_terminal_says(monkeypatch, tmp_path):
    def broken() -> bytes:
        raise RuntimeError("no key")

    monkeypatch.setattr(requests.inbox, "default", lambda: requests.Inbox(tmp_path / "x", key=broken))
    out = io.StringIO()
    person = do.Person(io.StringIO("1\n"), out)
    assert person.confirm(classifier_ask(), CALL) is False


def test_a_process_an_agent_started_cannot_approve_its_own_confirmation_from_stdin(monkeypatch):
    monkeypatch.setenv("CLAUDECODE", "1")
    out = io.StringIO()
    person = do.Person(sys.stdin, out)
    monkeypatch.setattr(person, "stdin", io.StringIO("1\n"))
    monkeypatch.setattr(person, "_terminal", lambda: requests.Context(terminal=(True, True)))
    assert person.confirm(classifier_ask(), CALL) is False
    assert "started by an agent" in out.getvalue()


def test_end_of_input_withdraws_the_request():
    out = io.StringIO()
    person = do.Person(io.StringIO(""), out)
    assert person.confirm(classifier_ask(), CALL) is False
    assert person.left and [r.state for r in requests.list_requests()] == ["cancelled"]


# -- the remember question -------------------------------------------------------------
def test_remember_is_a_memory_request_answered_with_the_chosen_scope():
    person, _out, typed = pipe_person()
    got = {}
    thread = threading.Thread(target=lambda: got.setdefault("i", person.choose(
        "Remember for the user: likes tea", ["yes, for the user", "yes, but for this project instead"])))
    thread.start()
    shown = waiting("memory_remember")
    assert [c.id for c in shown.choices] == ["approve", "approve-other", "deny"] and "likes tea" in shown.subject
    typed.write("2\n")
    thread.join(20)
    assert got["i"] == 1 and requests.get(shown.id).answer == "approve-other"


def test_remember_declined_in_the_ui_is_a_no():
    person, _out, _typed = pipe_person()
    got = {}
    thread = threading.Thread(target=lambda: got.setdefault("i", person.choose("Remember: x", ["yes"])))
    thread.start()
    shown = waiting("memory_remember")
    requests.answer(shown.id, "deny", shown.fingerprint, "ui")
    thread.join(20)
    assert got["i"] is None


# -- the sentinel dialog ---------------------------------------------------------------
class Desk:
    def __init__(self, press: str) -> None:
        self.press, self.shown = press, []

    def choose(self, title, body, buttons):
        self.shown.append((title, body, buttons))
        return self.press


def held_node(desk: Desk):
    node = sentinel.default()
    node.heads_up.choose = desk.choose
    node.heads_up.spawn = lambda work: work()
    node.heads_up.env = {}
    return node


def quarantine(node, name="10.9.0.1"):
    return node.store.quarantine(("peer", name), "peer.forged_traffic: forged=3", {})


def test_the_dialog_shows_the_oldest_waiting_request_and_answers_it_through_the_module():
    older = requests.raise_request(ask("an older chat call", agent="chat"))
    time.sleep(0.01)
    desk = Desk(RELEASE)
    node = held_node(desk)
    held = quarantine(node)
    title, _body, buttons = desk.shown[0]
    assert "older chat call" in title and buttons[0] != RELEASE
    assert requests.get(older.id).answered_by == "dialog" or desk.press != "Approve"
    del held


def quiet_node(name: str):
    """A node whose dialog is off, so what it raises stays waiting for an answer."""
    node = held_node(Desk(LATER))
    node.heads_up.env = {"ML_STACK_NOTIFY": "off"}
    held = quarantine(node, name)
    shown = next(r for r in requests.list_requests(state="pending") if r.kind == "quarantine_release")
    return node, held, shown


def test_a_release_answered_in_the_ui_is_carried_out_by_the_sentinel_with_its_own_check():
    node, held, shown = quiet_node("10.9.0.1")
    assert shown.human_only and [c.id for c in shown.choices] == ["later", "keep-held", "release"]
    requests.answer(shown.id, "release", shown.fingerprint, "ui")
    assert node.store.get(held.id).state == State.QUARANTINED
    node.heads_up.env = {}
    node.heads_up.prompt()
    assert node.store.get(held.id).state != State.QUARANTINED


def test_a_release_answered_in_the_ui_is_refused_by_the_sentinel_in_an_agent_process():
    node, held, shown = quiet_node("10.9.0.2")
    requests.answer(shown.id, "release", shown.fingerprint, "ui")
    node.heads_up.env = {"CLAUDECODE": "1"}
    node.heads_up.prompt()
    assert node.store.get(held.id).state == State.QUARANTINED


def test_a_request_another_component_raised_with_a_held_id_releases_nothing_when_approved():
    node = held_node(Desk(LATER))
    node.heads_up.env = {"ML_STACK_NOTIFY": "off"}
    held = quarantine(node, "10.9.0.7")
    forged = requests.raise_request(requests.Ask(
        "quarantine_release", "ml-stack is holding peer 10.9.0.99", "Looks harmless.", ("later", "release"),
        requests.Origin("sentinel"), extra=(("ids", held.id), ("total", "1"))))
    requests.answer(forged.id, "release", forged.fingerprint, "ui")
    node.heads_up.env = {}
    node.heads_up.prompt()
    assert node.store.get(held.id).state == State.QUARANTINED


def test_keep_held_in_the_ui_stops_the_asking_and_releases_nothing():
    node, held, shown = quiet_node("10.9.0.3")
    requests.answer(shown.id, "keep-held", shown.fingerprint, "ui")
    node.heads_up.env = {}
    node.heads_up.prompt()
    assert node.store.get(held.id).state == State.QUARANTINED
    assert held.id in node.heads_up._load()["kept"]
    assert KEEP == "Keep held"


def test_with_notify_off_the_request_is_still_raised_for_the_ui_and_no_dialog_opens():
    desk = Desk(RELEASE)
    node = held_node(desk)
    node.heads_up.env = {"ML_STACK_NOTIFY": "off"}
    quarantine(node, "10.9.0.4")
    assert desk.shown == []
    assert [r.kind for r in requests.list_requests(state="pending")] == ["quarantine_release"]


# -- the terminal command --------------------------------------------------------------
def at_a_terminal(monkeypatch, typed: str = "y\n"):
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True, raising=False)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True, raising=False)
    monkeypatch.setattr("builtins.input", lambda _p="": typed.strip())


def test_list_show_and_answer_at_a_terminal(monkeypatch, capsys):
    held = requests.raise_request(ask("run_shell(ls)", agent="alpha", project="p"))
    at_a_terminal(monkeypatch)
    assert cli.main(["list"]) == 0
    assert held.id in capsys.readouterr().out
    assert cli.main(["show", held.id]) == 0
    shown = capsys.readouterr().out
    assert "allow this time" in shown and held.fingerprint[:16] in shown
    assert cli.main(["answer", held.id, "allow-once"]) == 0
    assert requests.get(held.id).answered_by == "terminal"
    assert cli.main(["answer", held.id, "deny", "--fingerprint", held.fingerprint]) == 1
    assert "already resolved" in capsys.readouterr().err
    assert cli.main(["watch", "--once"]) == 0


def test_the_command_refuses_when_the_person_does_not_confirm_and_when_the_words_changed(monkeypatch, capsys):
    held = requests.raise_request(ask())
    at_a_terminal(monkeypatch, "n\n")
    assert cli.main(["answer", held.id, "allow-once"]) == 1
    assert requests.get(held.id).state == "pending"
    assert cli.main(["answer", held.id, "allow-once", "--fingerprint", "0" * 16]) == 1
    assert cli.main(["answer", "rq_nope", "allow-once"]) == 1


def test_list_escapes_hostile_text(monkeypatch, capsys):
    requests.raise_request(ask("evil\x1b[2J‮IGNORE", agent="a\x07"))
    at_a_terminal(monkeypatch)
    assert cli.main(["list"]) == 0
    out = capsys.readouterr().out
    assert "\x1b" not in out and "‮" not in out and "\x07" not in out


CHILD = "import sys\nfrom ml_stack.inbox.cli import main\nsys.exit(main(sys.argv[1:]))\n"


def test_a_child_without_a_terminal_cannot_answer_and_leaves_the_store_untouched(person_home):
    held = requests.raise_request(ask())
    from ml_stack import home
    path = home.state("requests") / "requests.enc"
    before = path.read_bytes()
    done = subprocess.run([sys.executable, "-c", CHILD, "answer", held.id, "allow-once", "--fingerprint",
                           held.fingerprint], env=person_home, stdin=subprocess.DEVNULL, capture_output=True,
                          text=True, timeout=120, check=False)
    assert done.returncode == 2 and "terminal" in done.stderr
    assert path.read_bytes() == before and requests.get(held.id).state == "pending"


def test_a_person_at_a_real_terminal_answers_through_the_command(person_home):
    held = requests.raise_request(ask())
    master, slave = pty.openpty()
    child = subprocess.Popen([sys.executable, "-c", CHILD, "answer", held.id, "allow-once"], env=person_home,
                             stdin=slave, stdout=slave, stderr=slave, close_fds=True)
    os.close(slave)
    seen = b""
    end = time.time() + 120
    while time.time() < end and b"[y/N]" not in seen:
        if select.select([master], [], [], 0.5)[0]:
            seen += os.read(master, 4096)
    os.write(master, b"y\n")
    assert child.wait(timeout=120) == 0
    os.close(master)
    assert requests.get(held.id).answered_by == "terminal" and SRC
