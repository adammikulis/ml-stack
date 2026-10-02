"""The question put to the owner: a dialog with real buttons, the code in a second dialog only
after an Accept, a stranger's text kept as data, and no test ever raising a real one."""

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from onboard_support import Clock, Recorder, info, requests

from ml_stack.fleet.onboard import notify

FP = "ab" * 32


def a_request(tmp_path, **more):
    rq = requests(tmp_path, Recorder(), Clock())
    return rq, rq.submit(info(FP, **more), "10.0.0.5")


def runner(*answers):
    """A runner that records each argv and answers from ``answers`` (a result per call)."""
    seen, queue = [], list(answers)

    def run(argv):
        seen.append(list(argv))
        return queue.pop(0) if queue else (0, "", "")
    run.seen = seen
    return run


def test_the_question_names_the_device_and_the_code_is_not_in_it(tmp_path):
    rq, r = a_request(tmp_path)
    code = rq.accept(r.id, mine=True).code
    title, body = notify.compose(r)
    assert "kitchen-pi" in title and "10.0.0.5" in body and "abab abab abab abab" in body
    assert code not in title + body
    shown_title, shown_body = notify.compose_code(rq.get(r.id))
    assert f"{code[:3]} {code[3:]}" in shown_title and "three tries" in shown_body


def test_macos_puts_a_dialog_with_three_buttons_and_hands_the_text_over_as_arguments(tmp_path):
    run = runner((0, "button returned:Accept as mine, gave up:false\n", ""))
    _, r = a_request(tmp_path, name='x" & (do shell script "id") & "', hostname="h\nh")
    title, body = notify.compose(r)
    assert notify.MacNotifier(run).ask(title, body) == "mine"
    argv = run.seen[0]
    assert argv[:3] == ["osascript", "-e", notify.MAC_ASK] and argv[3] == "--"
    script = notify.MAC_ASK
    assert "display alert" in script and "giving up after 120" in script
    assert 'default button "Decline"' in script and 'cancel button "Decline"' in script
    for button in ("Decline", "Accept as mine", "Accept as someone else's"):
        assert f'"{button}"' in script
    assert "do shell script" not in script and argv.count("-e") == 1
    assert 'do shell script "id"' in argv[4] + argv[5] and all("\n" not in a for a in argv[4:])


@pytest.mark.parametrize("result,answer", [
    ((0, "button returned:Accept as mine, gave up:false\n", ""), "mine"),
    ((0, "button returned:Accept as someone else's, gave up:false\n", ""), "other"),
    ((0, "button returned:Decline, gave up:false\n", ""), "decline"),
    ((0, "button returned:, gave up:true\n", ""), "timeout"),
    ((1, "", "execution error: User canceled. (-128)"), "decline"),
    ((1, "", "osascript: command not found"), "unavailable"),
    ((0, "something unexpected", ""), "decline")])
def test_every_way_the_dialog_can_end_is_read_and_anything_odd_is_a_decline(result, answer):
    assert notify.MacNotifier(runner(result)).ask("t", "b") == answer


def test_the_code_dialog_has_one_ok_button_and_the_code_is_an_argument(tmp_path):
    run = runner((0, "", ""))
    rq, r = a_request(tmp_path)
    accepted = rq.accept(r.id, mine=True)
    assert notify.MacNotifier(run).show_code(*notify.compose_code(accepted))
    argv = run.seen[0]
    assert argv[:3] == ["osascript", "-e", notify.MAC_CODE] and 'buttons {"OK"}' in notify.MAC_CODE
    assert f"{accepted.code[:3]} {accepted.code[3:]}" in argv[4] + argv[5]
    assert accepted.code not in notify.MAC_CODE


@pytest.mark.skipif(shutil.which("osacompile") is None, reason="needs macOS osacompile")
def test_both_scripts_compile_under_the_real_compiler(tmp_path):
    """Compiled, never run: running either would put a dialog on the screen."""
    for name in ("MAC_ASK", "MAC_CODE"):
        done = subprocess.run(["osacompile", "-o", str(tmp_path / f"{name}.scpt"), "-e",
                               getattr(notify, name)], capture_output=True, text=True,
                              timeout=60, check=False)
        assert done.returncode == 0, done.stderr


def test_linux_asks_with_notify_send_actions_and_falls_back_to_zenity():
    have = {"notify-send": "/x/notify-send", "zenity": "/x/zenity"}
    run = runner((0, "mine\n", ""))
    assert notify.LinuxNotifier(run, have.get).ask("--evil", "-u critical") == "mine"
    argv = run.seen[0]
    assert argv[0] == "notify-send" and "--wait" in argv
    assert argv.index("--") < argv.index("--evil")
    assert {"--action=mine=Accept as mine", "--action=decline=Decline"} <= set(argv)
    run = runner((0, "", ""))                                   # no action chosen at all
    assert notify.LinuxNotifier(run, have.get).ask("t", "b") == "timeout"
    run = runner((1, "", "unknown option --action"), (0, "Accept as mine\n", ""))
    assert notify.LinuxNotifier(run, have.get).ask("t", "b") == "mine"
    assert run.seen[1][0] == "zenity"
    assert notify.LinuxNotifier(runner((5, "", "")), {"zenity": "/z"}.get).ask("t", "b") == "timeout"
    assert notify.LinuxNotifier(runner(), {}.get).ask("t", "b") == "unavailable"


def test_windows_is_designed_not_built():
    assert notify.WindowsNotifier().ask("t", "b") == "unavailable"
    with pytest.raises(notify.Unsupported):
        notify.WindowsNotifier().show_code("t", "b")


def test_the_environment_picks_the_notifier_and_defaults_to_the_desktop():
    def have(name):
        return "/bin/" + name

    assert notify.pick("Darwin", which=have, env={}).name == "macos"
    assert notify.pick("Linux", which=have, env={}).name == "linux"
    assert notify.pick("Linux", which=lambda n: None, env={}).name == "console"
    assert notify.pick("Windows", env={}).name == "windows"
    assert notify.pick("Darwin", which=have, env={"ML_STACK_NOTIFY": "console"}).name == "console"
    assert notify.pick("Darwin", which=have, env={"ML_STACK_NOTIFY": "off"}).name == "off"
    assert notify.pick("Darwin", which=have, env={"ML_STACK_NOTIFY": "system"}).name == "macos"


def test_console_prints_the_cleaned_text_and_leaves_the_answer_to_the_commands():
    out = []
    c = notify.Console(out.append)
    assert c.ask("a\nb", "\x1b[31mred") == "unavailable"
    assert out == ["a b: [31mred Answer with: ml-stack fleet accept ID --mine|--other, or "
                   "ml-stack fleet decline ID"]


def test_the_question_cleans_text_again_even_for_a_request_built_by_hand():
    from ml_stack.fleet.onboard.requests import Request
    r = Request("a" * 32, "pi\n\x1b[31mACCEPT‮", "h\x07", "m\n", "10.0.0.5", FP, "ab" * 16, 0.0)
    for text in notify.compose(r):
        assert "\n" not in text and "\x1b" not in text and "‮" not in text
        assert "\x07" not in text


# -- the test suite itself never raises a real notification -------------------------------
def test_this_session_is_set_to_the_console_and_the_desktop_tools_are_shimmed():
    assert os.environ["ML_STACK_NOTIFY"] == "console"
    assert notify.pick().name == "console"
    if os.name != "nt":
        for tool in ("osascript", "notify-send", "zenity", "kdialog"):
            assert Path(shutil.which(tool)).read_text().startswith("#!/bin/sh")


@pytest.mark.skipif(os.name == "nt", reason="the shims are shell scripts")
def test_a_process_that_reaches_for_the_desktop_is_caught_by_the_shim():
    """What the session-end check relies on: asking the system notifier from a child process
    (here built directly, as a buggy test might) fails and leaves a record."""
    log = Path(os.environ["ML_STACK_SHIM_LOG"])
    before = log.read_text() if log.exists() else ""
    code = ("from ml_stack.fleet.onboard import notify\n"
            "print(notify.MacNotifier().ask('t', 'b'))\n")
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                          env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
                          timeout=60, check=False)
    assert done.stdout.strip() == "unavailable"           # the shim failed, so nothing was shown
    after = log.read_text()
    assert after != before and "osascript" in after[len(before):]
    log.write_text(before)                               # this one attempt was on purpose


@pytest.mark.skipif(not os.environ.get("ML_STACK_MANUAL_DIALOG"),
                    reason="set ML_STACK_MANUAL_DIALOG=1 to see the real dialog on this screen")
@pytest.mark.skipif(platform.system() != "Darwin", reason="macOS")
def test_manual_the_real_dialog_appears_and_the_click_comes_back(monkeypatch, tmp_path):
    monkeypatch.setenv("PATH", "/usr/bin:/bin")           # past the shim, to the real osascript
    _, r = a_request(tmp_path)
    answer = notify.MacNotifier().ask(*notify.compose(r))
    assert answer in {"mine", "other", "decline", "timeout"}
