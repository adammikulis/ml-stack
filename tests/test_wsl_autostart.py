import io
import json
from unittest.mock import Mock

import pytest

from poolhouse.fleet import autostart, wsl_startup


def request(**values):
    return {"action": "configure", "mode": "login", "slots": 1,
            "labels": [], "report": "", **values}


@pytest.mark.parametrize("values", [{"mode": "boot"}, {"slots": True},
                                     {"labels": ["gpu&calc"]}, {"report": "%PATH%"},
                                     {"labels": "gpu"}, {"action": "run"}])
def test_hostile_request_refused(values):
    with pytest.raises(ValueError):
        wsl_startup.validate(request(**values))


def test_guest_preferences_delegate_without_linux_service(monkeypatch):
    monkeypatch.setattr(wsl_startup, "guest", lambda: True)
    call = Mock(return_value={"installed": True, "mode": "login"})
    monkeypatch.setattr(wsl_startup, "call", call)
    monkeypatch.setattr(autostart, "uninstall", Mock(side_effect=AssertionError))
    assert autostart.install("login").installed
    assert autostart.status()["mode"] == "login"
    assert call.call_args_list[0].args == ("configure",)


def test_windows_owner_preserves_running_launcher(monkeypatch, tmp_path):
    run = Mock()
    install = Mock(return_value=autostart.Autostart("login", True))
    monkeypatch.setattr(autostart.subprocess, "run", run)
    monkeypatch.setattr(autostart, "_windows_startup", lambda: tmp_path / "startup.cmd")
    monkeypatch.setattr(autostart, "_windows_install", install)
    monkeypatch.setattr(autostart, "plan", lambda *a, **k: ["launcher"])
    assert autostart._windows_owner(request())["installed"]
    assert install.call_args.kwargs == {"start_now": False}
    assert all("/End" not in call.args[0] for call in run.call_args_list)
    assert all("/Run" not in call.args[0] for call in run.call_args_list)


def test_invalid_wire_never_mutates_tasks(monkeypatch):
    configure = Mock(side_effect=AssertionError)
    monkeypatch.setattr(wsl_startup.sys, "platform", "win32")
    monkeypatch.setattr(wsl_startup.sys, "stdin", io.StringIO(json.dumps(request(labels=["%bad%"]))))
    assert wsl_startup.answer(Mock(), configure) == 2
    configure.assert_not_called()


def test_windows_plan_launches_bridge_owner(monkeypatch):
    monkeypatch.setattr(autostart.sys, "platform", "win32")
    assert autostart.plan("login")[1:4] == ["-m", "poolhouse.fleet.launch", "--no-browser"]
