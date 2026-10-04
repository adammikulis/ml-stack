"""Raising the GPU wiring limit, attacked: an agent, a token, a hostile number or model name.

The privileged call is a recording stand-in; a child process is used where the real command
line is the thing under attack.
"""

from __future__ import annotations

import io
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
from test_room_routes import room  # noqa: F401
from test_wired import SRC, Recorder, hooks

from ml_stack.sentinel.human import HumanRequired
from ml_stack.serve import wired_apply as apply

SAFE_SCRIPT = re.compile(r"[A-Za-z0-9 /:.=_;'%\\\-<>?\",!()&|\s]*")


@pytest.fixture(autouse=True)
def state_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("ML_STACK_HOME", str(tmp_path / "home"))
    for name in apply.AGENT_MARKERS:
        monkeypatch.delenv(name, raising=False)


def test_no_model_tool_chat_tool_or_slash_command_touches_the_wiring_limit():
    from ml_stack import chat, do, mcp

    words = ("wired", "sysctl", "iogpu", "wiring")
    for tool in mcp.TOOLS:
        assert not any(w in f"{tool.name} {tool.description}".lower() for w in words), tool.name
    person = do.Person(io.StringIO(""), io.StringIO(""))
    session = chat.Chat(None, person, role="operator", extension=chat.extensions(person))
    for spec, _ in session.offered:
        text = f"{spec['function']['name']} {spec['function'].get('description', '')}".lower()
        assert not any(w in text for w in words), spec["function"]["name"]


def test_only_the_command_and_the_ui_route_import_the_privileged_change():
    root = Path(SRC) / "ml_stack"
    users = sorted(p.relative_to(root).as_posix() for p in root.rglob("*.py")
                   if re.search(r"import .*wired_apply|wired_apply import",
                                p.read_text(encoding="utf-8")))
    assert users == ["fleet/room_routes.py", "serve/wired_cli.py"]


@pytest.mark.parametrize("argv", [["--reset"], ["--apply", "65536"], ["--persist", "65536"],
                                  ["--unpersist"]])
def test_a_command_run_by_an_agent_is_refused_before_anything_is_written(tmp_path, argv):
    env = {**os.environ, "PYTHONPATH": SRC, "ML_STACK_HOME": str(tmp_path / "home"),
           "ML_STACK_NO_REAL_KEYSTORE": "1", "CLAUDECODE": "1"}
    code = ("import sys; from ml_stack.serve import cli; "
            f"sys.exit(cli.COMMANDS.run(['memory', *{argv!r}]))")
    got = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True,
                         stdin=subprocess.DEVNULL, timeout=60)
    assert got.returncode != 0 and "agent" in got.stdout + got.stderr
    assert not (tmp_path / "home" / "wired-limit.json").exists()


@pytest.mark.parametrize("name", ["x; touch PWNED", "$(touch PWNED)", "`touch PWNED`",
                                  "../../etc/passwd", "a\ntouch PWNED"])
def test_a_model_name_is_a_lookup_and_never_a_command(tmp_path, name):
    env = {**os.environ, "PYTHONPATH": SRC, "ML_STACK_HOME": str(tmp_path / "home"),
           "ML_STACK_NO_REAL_KEYSTORE": "1", "CLAUDECODE": ""}
    got = subprocess.run([sys.executable, "-c",
                          "import sys; from ml_stack.serve import cli; "
                          f"sys.exit(cli.COMMANDS.run(['memory', '--for', {name!r}, '--apply']))"],
                         env=env, capture_output=True, text=True, stdin=subprocess.DEVNULL,
                         timeout=60, cwd=tmp_path)
    assert got.returncode != 0
    assert not (tmp_path / "PWNED").exists()


@pytest.mark.parametrize("mb", [4096, 65536, 98304, 120 * 1024])
def test_the_script_for_any_valid_integer_is_fixed_text_around_that_integer(mb, tmp_path):
    for keep in (True, False, None):
        script = apply.script_for(mb, keep, tmp_path / "d.plist")
        assert SAFE_SCRIPT.fullmatch(script), script
        assert script.count(str(mb)) >= 1 and "$" not in script and "`" not in script


@pytest.mark.parametrize("bad", ["1; touch PWNED", "$(id)", -1, 0, 1.0, True, 10**12, [], None])
def test_a_hostile_number_never_reaches_the_runner(bad, tmp_path):
    ran = Recorder()
    with pytest.raises(ValueError):
        apply.set_limit(bad, keep=True, via="osascript", hooks=hooks(ran, tmp_path / "d.plist"))
    assert ran.calls == []


def test_an_agent_marker_stops_every_entry_point(tmp_path):
    ran = Recorder()
    h = hooks(ran, tmp_path / "d.plist", env={"ML_STACK_AGENT": "1"}, terminal=(True, True))
    for call in (lambda: apply.set_limit(65536, via="osascript", hooks=h),
                 lambda: apply.set_limit(65536, keep=True, via="sudo", hooks=h),
                 lambda: apply.set_limit(None, keep=False, via="osascript", hooks=h),
                 lambda: apply.reset(via="osascript", hooks=h)):
        with pytest.raises(HumanRequired):
            call()
    assert ran.calls == []


@pytest.mark.parametrize("path", ["/ui/room/apply", "/ui/room/keep", "/ui/room/reset"])
def test_a_token_or_a_foreign_page_cannot_reach_any_changing_route(room, path):  # noqa: F811
    body = {"mb": 65536, "keep": True}
    for headers in ({"Authorization": "Bearer x"}, {"Origin": "https://evil.example"},
                    {"Host": "rebind.example.com"}):
        status, _, _ = room.call(path, method="POST", body=body, headers=headers)
        assert status == 403, (path, headers)
    assert room.ran.calls == []
