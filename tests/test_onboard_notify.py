"""Notifications: who is asking, never the code, and a stranger's text stays data."""

import shutil
import subprocess
import sys

import pytest
from onboard_support import Clock, Recorder, info, requests

from ml_stack.fleet.onboard import notify

FP = "ab" * 32


def a_request(tmp_path, **more):
    rq = requests(tmp_path, Recorder(), Clock())
    return rq, rq.submit(info(FP, **more), "10.0.0.5")


def test_the_text_names_the_device_and_how_to_answer_and_holds_no_code(tmp_path):
    rq, r = a_request(tmp_path)
    code = rq.accept(r.id).code
    title, body = notify.compose(r)
    assert "kitchen-pi" in title and "10.0.0.5" in body and "abab abab abab abab" in body
    assert "ml-stack fleet accept" in body and r.id[:8] in body
    assert code not in title + body


def test_macos_hands_the_text_to_osascript_as_arguments_not_as_script(tmp_path):
    seen = []
    n = notify.MacNotifier(lambda argv: seen.append(list(argv)) or 0)
    _, r = a_request(tmp_path, name='x" & (do shell script "id") & "', hostname="h\nh")
    assert n.notify(*notify.compose(r))
    argv = seen[0]
    assert argv[:3] == ["osascript", "-e", notify.MAC_SCRIPT] and argv[3] == "--"
    assert "do shell script" not in notify.MAC_SCRIPT
    assert argv.count("-e") == 1                            # nothing else is a script
    assert 'do shell script "id"' in argv[4] or 'do shell script "id"' in argv[5]
    assert all("\n" not in a for a in argv[4:])


def test_linux_puts_double_dash_before_the_text_so_it_cannot_be_an_option():
    seen = []
    n = notify.LinuxNotifier(lambda argv: seen.append(list(argv)) or 0)
    assert n.notify("--hint=string:x-evil:1", "-u critical")
    argv = seen[0]
    assert argv[0] == "notify-send" and argv.index("--") < argv.index("--hint=string:x-evil:1")


def test_a_failing_command_is_false_not_an_exception():
    assert notify.MacNotifier(lambda argv: 1).notify("t", "b") is False
    assert notify._run(["/definitely/not/a/program"]) == 1


def test_windows_says_it_is_not_built():
    with pytest.raises(notify.Unsupported):
        notify.WindowsNotifier().notify("t", "b")


def test_pick_chooses_by_platform_and_falls_back_to_the_console():
    have = lambda name: "/bin/" + name  # noqa: E731
    assert notify.pick("Darwin", which=have).name == "macos"
    assert notify.pick("Linux", which=have).name == "linux"
    assert notify.pick("Linux", which=lambda n: None).name == "console"
    assert notify.pick("Windows").name == "windows"
    assert notify.pick("Plan9").name == "console"


def test_console_prints_the_cleaned_text():
    out = []
    notify.Console(out.append).notify("a\nb", "\x1b[31mred")
    assert out == ["a b: [31mred"]


@pytest.mark.skipif(sys.platform != "darwin" or not shutil.which("osascript"),
                    reason="needs macOS osascript")
def test_real_osascript_treats_arguments_after_double_dash_as_data():
    """Same argument layout as the notifier, with a script that returns what it was given
    instead of showing it: no notification appears."""
    script = 'on run argv\nreturn (item 1 of argv) & "|" & (item 2 of argv)\nend run'
    hostile = 'x" & (do shell script "touch /tmp/should-not-exist") & "'
    done = subprocess.run(["osascript", "-e", script, "--", hostile, "-e evil"],
                          capture_output=True, text=True, timeout=30, check=False)
    assert done.returncode == 0 and done.stdout.strip() == hostile + "|-e evil"
