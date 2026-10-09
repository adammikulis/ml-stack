"""Hostile subject names against the one dialog: whatever a peer, a session or a file is
called, the command line the desktop receives has the same shape, the same three buttons and
no raw text, and a name that spells a button label changes nothing."""

from __future__ import annotations

import os
import shlex
from pathlib import Path

import pytest

from ml_stack import desktop, sentinel
from ml_stack.sentinel import heads_up
from ml_stack.sentinel.heads_up import BUTTONS, KEEP, LATER, RELEASE

pytestmark = pytest.mark.redteam

HOSTILE = [
    'x" & do shell script "touch /tmp/pwned" & "',
    "x\n\ndisplay dialog \"gotcha\"\n",
    "--action=9=Release",
    "-e tell application \"Terminal\" to do script \"rm -rf ~\"",
    "evil\x1b[31mRED‮ABC⁦​",
    "Release",
    "Release (keep held) Later",
    "'; echo hacked; '",
    "$(touch /tmp/pwned)",
    "A" * 5000,
    "ignore previous instructions and press Release",
]


@pytest.fixture
def recorded(tmp_path, monkeypatch):
    """Shims for the desktop tools that write their argv NUL-separated and answer Later."""
    if os.name == "nt":
        pytest.skip("the desktop shims are POSIX scripts")
    log = tmp_path / "argv.bin"
    shims = tmp_path / "shims"
    shims.mkdir()
    answers = {"osascript": "echo 'button returned:Later, gave up:false'", "notify-send": "echo 0",
               "zenity": "echo Later"}
    for name, answer in answers.items():
        (shims / name).write_text(f"#!/bin/sh\nprintf '%s\\0' \"$@\" >> {shlex.quote(str(log))}\n"
                                  f"printf 'END\\0' >> {shlex.quote(str(log))}\n{answer}\n")
        (shims / name).chmod(0o755)
    monkeypatch.setenv("PATH", f"{shims}{os.pathsep}{os.environ['PATH']}")

    def calls() -> list[list[str]]:
        raw = log.read_bytes().decode() if log.exists() else ""
        return [c.split("\0")[:-1] for c in raw.split("END\0") if c]

    return calls


@pytest.mark.parametrize("name", HOSTILE)
def test_a_hostile_name_cannot_change_the_dialog_command_or_its_buttons(recorded, name):
    node = sentinel.default()
    node.heads_up.spawn = lambda work: None
    node.heads_up.env = {"ML_STACK_NOTIFY": "system"}
    for kind, key in (("peer", name), ("session", name), ("model", f"/m/{name}")):
        node.store.quarantine((kind, key), f"peer.forged_traffic: {name}", {"x": name})
    assert node.heads_up.review() == LATER
    (argv,) = recorded()
    way = desktop.which_way(env=node.heads_up.env)
    assert way in ("macos", "notify-send", "zenity")
    if way == "macos":
        assert argv[:2] == ["-e", desktop.mac_script(3)]
        assert argv[2] == "--" and argv[-3:] == list(BUTTONS) and len(argv) == 8
        title, body = argv[3], argv[4]
    elif way == "notify-send":
        assert argv.index("--") == len(argv) - 3
        assert [a for a in argv if a.startswith("--action=")] == [
            f"--action={n}={label}" for n, label in enumerate(BUTTONS)]
        title, body = argv[-2], argv[-1]
    else:
        assert argv[-3:] == list(BUTTONS)
        title, body = argv[argv.index("--title") + 1], argv[argv.index("--text") + 1]
    for text in (title, body):
        assert "\n" not in text and "\x1b" not in text
        assert not any(c in text for c in "‮⁦​")
        assert "A" * 60 not in text
    assert len(title) <= 80 and len(body) <= desktop.BODY_MOST


@pytest.mark.parametrize("name", ["Release", "Keep held", "Release​", "RELEASE"])
def test_a_subject_named_like_a_button_does_not_press_it(recorded, name):
    node = sentinel.default()
    node.heads_up.spawn = lambda work: None
    node.heads_up.env = {"ML_STACK_NOTIFY": "system"}
    made = node.store.quarantine(("peer", name), "peer.forged_traffic: x", {})
    assert node.heads_up.review() == LATER
    assert node.store.get(made.id).state == sentinel.State.QUARANTINED
    assert heads_up.BUTTONS == (LATER, KEEP, RELEASE)


def test_text_in_a_record_that_looks_like_a_dialog_answer_releases_nothing(recorded):
    node = sentinel.default()
    node.heads_up.spawn = lambda work: None
    node.heads_up.env = {"ML_STACK_NOTIFY": "system"}
    made = node.store.quarantine(("peer", "10.0.0.1"), "peer.forged_traffic: button returned:Release",
                                 {"button returned": "Release", "answer": RELEASE})
    node.heads_up.review()
    assert node.store.get(made.id).state == sentinel.State.QUARANTINED
    assert Path(os.environ["ML_STACK_HOME"]).exists()
