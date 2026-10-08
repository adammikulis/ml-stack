"""The launchd plist, systemd units and Windows task XML for a role, rendered as bytes."""

from __future__ import annotations

import plistlib
import subprocess
import xml.etree.ElementTree as ET

from .autostart_manifest import ROLE_INFO, Role

__all__ = ["ENSURE_INTERVAL_S", "render"]

ENSURE_INTERVAL_S = 3600
_NS = "http://schemas.microsoft.com/windows/2004/02/mit/task"


def _plist(role: Role, scope: str, user: str) -> bytes:
    restart, logs = role.restart, role.logs
    unit: dict[str, object] = {
        "Label": role.label, "ProgramArguments": list(role.argv), "RunAtLoad": True,
        "WorkingDirectory": role.workdir, "EnvironmentVariables": dict(role.environment),
        "StandardOutPath": logs["stdout"], "StandardErrorPath": logs["stderr"],
        "ThrottleInterval": restart["backoff_s"], "ExitTimeOut": 30, "ProcessType": "Background",
    }
    if role.role == "pool-daemon":
        unit["KeepAlive"] = {"SuccessfulExit": False}
    else:
        unit.update(StartInterval=ENSURE_INTERVAL_S, LowPriorityIO=True, Nice=10)
    if scope == "system":
        unit["UserName"] = user
    return plistlib.dumps(unit, sort_keys=True)


def _q(text: str) -> str:
    """One systemd word: quoted, with the characters systemd expands written out literally."""
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"').replace("%", "%%").replace("$", "$$") + '"'


def _service(role: Role, scope: str, user: str) -> bytes:
    restart = role.restart
    daemon = role.role == "pool-daemon"
    target = "multi-user.target" if scope == "system" else "default.target"
    lines = ["[Unit]", f"Description=ml-stack {role.role.replace('-', ' ')}"]
    if daemon:
        lines += ["After=network-online.target", "Wants=network-online.target",
                  f"StartLimitIntervalSec={restart['interval_s']}", f"StartLimitBurst={restart['burst']}"]
    lines += ["", "[Service]", f"Type={'simple' if daemon else 'oneshot'}"]
    if scope == "system":
        lines.append(f"User={user}")
    lines += [f"WorkingDirectory={_q(role.workdir)}"]
    lines += [f"Environment={_q(f'{k}={v}')}" for k, v in sorted(role.environment.items())]
    lines += [f"ExecStart={' '.join(_q(a) for a in role.argv)}", "NoNewPrivileges=yes",
              "TimeoutStopSec=30", "StandardOutput=journal", "StandardError=journal",
              "LogRateLimitIntervalSec=30s", "LogRateLimitBurst=1000"]
    if daemon:
        lines += ["Restart=on-failure", f"RestartSec={restart['backoff_s']}",
                  "", "[Install]", f"WantedBy={target}"]
    else:
        lines += ["Nice=10", "TimeoutStartSec=3600"]
    return ("\n".join(lines) + "\n").encode()


def _timer(role: Role) -> bytes:
    name = f"{ROLE_INFO[role.role]['service']}.service"
    return ("[Unit]\n"
            f"Description=ml-stack {role.role.replace('-', ' ')} schedule\n\n"
            "[Timer]\n"
            "OnBootSec=120\n"
            f"OnUnitActiveSec={ENSURE_INTERVAL_S}\n"
            "Persistent=true\n"
            "RandomizedDelaySec=120\n"
            f"Unit={name}\n\n"
            "[Install]\n"
            "WantedBy=timers.target\n").encode()


def _sub(parent: ET.Element, tag: str, text: str = "") -> ET.Element:
    child = ET.SubElement(parent, f"{{{_NS}}}{tag}")
    child.text = text or None
    return child


def _task(role: Role) -> bytes:
    ET.register_namespace("", _NS)
    task = ET.Element(f"{{{_NS}}}Task", version="1.2")
    triggers = _sub(task, "Triggers")
    _sub(_sub(triggers, "LogonTrigger"), "Enabled", "true")
    if role.role == "runtime-ensure":
        timed = _sub(triggers, "TimeTrigger")
        repeat = _sub(timed, "Repetition")
        _sub(repeat, "Interval", f"PT{ENSURE_INTERVAL_S // 3600}H")
        _sub(repeat, "StopAtDurationEnd", "false")
        _sub(timed, "StartBoundary", "2000-01-01T00:00:00")
    principal = _sub(_sub(task, "Principals"), "Principal", "")
    _sub(principal, "LogonType", "InteractiveToken")
    _sub(principal, "RunLevel", "LeastPrivilege")
    settings = _sub(task, "Settings")
    _sub(settings, "MultipleInstancesPolicy", "IgnoreNew")
    _sub(settings, "StartWhenAvailable", "true")
    _sub(settings, "ExecutionTimeLimit", "PT0S" if role.role == "pool-daemon" else "PT2H")
    if role.role == "pool-daemon":
        again = _sub(settings, "RestartOnFailure")
        _sub(again, "Interval", "PT1M")
        _sub(again, "Count", str(role.restart["burst"]))
    run = _sub(_sub(task, "Actions"), "Exec")
    _sub(run, "Command", role.argv[0])
    _sub(run, "Arguments", subprocess.list2cmdline(list(role.argv[1:])))
    _sub(run, "WorkingDirectory", role.workdir)
    return ET.tostring(task, encoding="utf-16", xml_declaration=True)


def render(role: Role, platform: str, scope: str, user: str) -> dict[str, bytes]:
    """Each unit kind ``role`` installs on ``platform`` mapped to its file bytes."""
    if platform == "darwin":
        return {"plist": _plist(role, scope, user)}
    if platform == "win32":
        return {"task": _task(role)}
    out = {"service": _service(role, scope, user)}
    if role.role == "runtime-ensure":
        out["timer"] = _timer(role)
    return out
