"""``/ui/room``: the page's route for sizing the wiring limit and the person-only change.

A real daemon on a real socket; the privileged call is a recording stand-in.
"""

from __future__ import annotations

import platform

import pytest
from conftest import write_gguf
from test_fleet_ui import WORDS, Serving, a_keystore, counting  # noqa: F401
from test_wired import Recorder, dense

from ml_stack.serve import wired, wired_apply as apply


@pytest.fixture(autouse=True)
def the_passphrase_is_kept(a_keystore):  # noqa: F811
    """Joining stores the passphrase and signing in compares against it."""


@pytest.fixture(autouse=True)
def the_cluster_already_exists(monkeypatch, tmp_path):
    from ml_stack.fleet.onboard import joining

    monkeypatch.setattr(joining, "find_joiners", lambda *a, **k: [])
    monkeypatch.setenv("ML_STACK_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    for name in apply.AGENT_MARKERS:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def room(tmp_path):
    ran = Recorder(start=98304)
    s = Serving(tmp_path)
    s.ui.room_hooks = wired.Hooks(total=128 * 1024**3, runner=ran, system="Darwin",
                                  read=ran.read, daemon=tmp_path / "LaunchDaemons" / "d.plist")
    s.ran = ran
    try:
        yield s
    finally:
        s.close()


def test_the_page_carries_the_slider_on_the_models_and_settings_screens(room):
    from ml_stack.fleet.page import render

    page = render()
    assert page.count("<wired-memory>") == 2 and "Make room for this model" in page
    assert 'customElements.define("wired-memory"' in page
    assert "Keep this after restart" in page
    assert room.call("/ui/")[0] == 200


def test_a_click_raises_the_limit_through_the_one_stand_in_call(room):
    status, body, _ = room.call("/ui/room/apply", method="POST", body={"mb": 65536})
    assert status == 200 and body["ok"] and body["after_mb"] == 65536, body
    assert body["machine"]["original_mb"] == 98304 and body["machine"]["kept"] is False
    assert room.ran.calls == [["osascript", "-e",
                               'do shell script "/usr/sbin/sysctl -w iogpu.wired_limit_mb=65536"'
                               ' with administrator privileges']]
    status, body, _ = room.call("/ui/room/reset", method="POST", body={})
    assert status == 200 and room.ran.limit == 98304


@pytest.mark.parametrize("bad", ["65536; reboot", -5, 0, 1.5, None, 10**9, True, [1]])
def test_a_value_that_is_not_an_integer_in_range_is_refused_and_runs_nothing(room, bad):
    status, body, _ = room.call("/ui/room/apply", method="POST", body={"mb": bad})
    assert status == 400 and "error" in body
    assert room.ran.calls == []


def test_the_request_must_carry_the_ui_header(room):
    status, _, _ = room.call("/ui/room/apply", method="POST", body={"mb": 65536},
                             ui_header=False)
    assert status == 403 and room.ran.calls == []


def test_another_web_page_cannot_make_the_request(room):
    status, body, _ = room.call("/ui/room/apply", method="POST", body={"mb": 65536},
                                headers={"Origin": "https://evil.example"})
    assert status == 403 and "another web page" in body["error"]
    assert room.ran.calls == []


def test_a_hostname_in_the_host_header_is_refused(room):
    status, _, _ = room.call("/ui/room/apply", method="POST", body={"mb": 65536},
                             headers={"Host": "rebind.example.com"})
    assert status == 403 and room.ran.calls == []


def test_an_access_token_header_cannot_make_the_request(room):
    for header in ({"Authorization": "Bearer abc"}, {"X-ML-Stack-Token": "abc"}):
        status, _, _ = room.call("/ui/room/apply", method="POST", body={"mb": 65536},
                                 headers=header)
        assert status == 403
    assert room.ran.calls == []


def test_a_session_opened_with_a_token_cannot_make_the_request(room):
    room.call("/ui/setup/join", method="POST", body={"passphrase": WORDS, "group": "home"})
    token = next(iter(room._cluster_tokens()))
    _, _, headers = room.call("/ui/session", method="POST", body={"token": token})
    cookie = headers["Set-Cookie"].split(";")[0]
    status, body, _ = room.call("/ui/room/apply", method="POST", body={"mb": 65536},
                                cookie=cookie)
    assert status == 403 and "token" in body["error"]
    _, _, headers = room.call("/ui/session", method="POST", body={"passphrase": WORDS})
    person = headers["Set-Cookie"].split(";")[0]
    status, _, _ = room.call("/ui/room/apply", method="POST", body={"mb": 65536}, cookie=person)
    assert status == 200 or status == 502
    assert len(room.ran.calls) == 1


def test_a_joined_machine_asks_for_a_session_first(room):
    room.call("/ui/setup/join", method="POST", body={"passphrase": WORDS, "group": "home"})
    status, _, _ = room.call("/ui/room/apply", method="POST", body={"mb": 65536})
    assert status == 401 and room.ran.calls == []


def test_a_model_name_with_shell_text_is_only_ever_a_lookup(room):
    status, body, _ = room.call("/ui/room?model=%24%28touch%20x%29%3B%20rm%20-rf%20~&ctx=8192")
    assert status == 400 and "not installed" in body["error"]
    assert room.ran.calls == []


def test_the_plan_for_an_installed_model_comes_back_as_plain_numbers(room, tmp_path):
    path = write_gguf(tmp_path / "m-Q4_K_M.gguf", dense())
    with path.open("ab") as f:
        f.truncate(2 * 1024**3)
    status, body, _ = room.call(f"/ui/room?model={path}&ctx=16384&kv=q8_0&mtp=1")
    assert status == 200
    plan = body["plan"]
    assert plan["context"] == 16384 and plan["kv"] == "q8_0" and plan["table"]
    assert {g["where"] for g in plan["grows"]} == {"host RAM"}
    status, _, _ = room.call(f"/ui/room?model={path}&kv=q5_0")
    assert status == 400


def test_an_agent_marker_in_the_daemons_environment_refuses_even_a_good_request(room, monkeypatch):
    monkeypatch.setenv("CLAUDECODE", "1")
    status, body, _ = room.call("/ui/room/apply", method="POST", body={"mb": 65536})
    assert status == 403 and "agent" in body["error"]
    assert room.ran.calls == []


def test_another_machine_on_the_network_cannot_make_the_request_even_signed_in(room):
    from ml_stack.fleet.discovery import primary_ip

    lan = primary_ip()
    if lan.startswith("127."):
        pytest.skip("this machine has no address but loopback")
    room.call("/ui/setup/join", method="POST", body={"passphrase": WORDS, "group": "home"})
    _, _, headers = room.call("/ui/session", method="POST", body={"passphrase": WORDS}, host=lan)
    cookie = headers["Set-Cookie"].split(";")[0]
    status, body, _ = room.call("/ui/room/apply", method="POST", body={"mb": 65536},
                                host=lan, cookie=cookie)
    assert status == 403 and "its own browser" in body["error"], (status, body)
    assert room.ran.calls == []
