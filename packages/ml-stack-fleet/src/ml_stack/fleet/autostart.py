"""Install the daemon to start at boot or at login, per platform."""

from __future__ import annotations

import os
import plistlib
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

__all__ = ["Autostart", "LABEL", "plan", "install", "uninstall", "status"]

LABEL = "com.ml-stack.traind"
SERVICE = "ml-stack-traind"
MODES = ("boot", "login", "manual")


@dataclass(frozen=True, slots=True)
class Autostart:
    """What was done, or what still needs a human. ``command`` is empty when finished."""

    mode: str
    installed: bool
    path: Path | None = None
    command: str = ""
    note: str = ""


def _executable() -> list[str]:
    """How to start the daemon on this machine."""
    found = shutil.which(SERVICE)
    if found:
        return [found]
    return [sys.executable, "-m", "ml_stack.fleet.daemon"]


def _args(slots: int = 1, labels: tuple[str, ...] = (), report: str = "") -> list[str]:
    out: list[str] = []
    if slots != 1:
        out += ["--slots", str(slots)]
    for label in labels:
        out += ["--label", label]
    if report:
        out += ["--report", report]
    return out


# -- macOS ---------------------------------------------------------------
def _mac_path(mode: str) -> Path:
    if mode == "boot":
        return Path("/Library/LaunchDaemons") / f"{LABEL}.plist"
    return Path.home() / "Library" / "LaunchAgents" / f"{LABEL}.plist"


def _ask_and_run(command: str, prompt: str) -> tuple[bool, str]:
    """Run a privileged command through the OS's own password dialog."""
    if sys.platform == "darwin":
        script = (f'do shell script "{command}" with administrator privileges '
                  f'with prompt "{prompt}"')
        argv = ["osascript", "-e", script]
    elif sys.platform.startswith("linux") and shutil.which("pkexec"):
        argv = ["pkexec", "sh", "-c", command]
    else:
        return False, ""
    try:
        done = subprocess.run(argv, capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.SubprocessError) as exc:
        return False, str(exc)
    return done.returncode == 0, (done.stderr or "").strip()


def _mac_install(mode: str, argv: list[str], log_dir: Path) -> Autostart:
    path = _mac_path(mode)
    plist = {
        "Label": LABEL,
        "ProgramArguments": argv,
        "RunAtLoad": True,
        "KeepAlive": {"SuccessfulExit": False},
        "StandardOutPath": str(log_dir / "traind.log"),
        "StandardErrorPath": str(log_dir / "traind.log"),
        "WorkingDirectory": str(Path.home()),
        "EnvironmentVariables": {"PATH": os.environ.get("PATH", "/usr/bin:/bin")},
    }
    body = plistlib.dumps(plist)

    if mode == "boot":
        staged = log_dir / f"{LABEL}.plist"
        staged.write_bytes(body)
        command = f"cp '{staged}' '{path}' && launchctl load -w '{path}'"
        ok, why = _ask_and_run(command, "ml-stack needs permission to start at boot")
        if ok:
            return Autostart(mode, installed=True, path=path)
        return Autostart(
            mode, installed=False, path=path, command=f"sudo {command}",
            note=why or "Permission was not given, so it will not start at boot yet.")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)
    subprocess.run(["launchctl", "unload", str(path)],
                   capture_output=True, check=False)
    done = subprocess.run(["launchctl", "load", "-w", str(path)],
                          capture_output=True, text=True, check=False)
    if done.returncode != 0:
        return Autostart(mode, installed=False, path=path,
                         command=f"launchctl load -w {path}",
                         note=done.stderr.strip() or "launchctl refused the job")
    return Autostart(mode, installed=True, path=path)


# -- systemd -------------------------------------------------------------
def _unit(argv: list[str], mode: str) -> str:
    after = "network-online.target"
    install_target = "multi-user.target" if mode == "boot" else "default.target"
    return (
        "[Unit]\n"
        "Description=ml-stack training daemon\n"
        f"After={after}\n"
        f"Wants={after}\n\n"
        "[Service]\n"
        "Type=simple\n"
        f"ExecStart={' '.join(argv)}\n"
        "Restart=on-failure\n"
        "RestartSec=5\n\n"
        "[Install]\n"
        f"WantedBy={install_target}\n"
    )


def _systemd_install(mode: str, argv: list[str], log_dir: Path) -> Autostart:
    body = _unit(argv, mode)
    if mode == "boot":
        staged = log_dir / f"{SERVICE}.service"
        staged.write_text(body)
        target = Path("/etc/systemd/system") / f"{SERVICE}.service"
        command = (f"cp '{staged}' '{target}' && systemctl daemon-reload && "
                   f"systemctl enable --now {SERVICE}")
        ok, why = _ask_and_run(command, "ml-stack needs permission to start at boot")
        if ok:
            return Autostart(mode, installed=True, path=target)
        return Autostart(
            mode, installed=False, path=target, command=f"sudo sh -c \"{command}\"",
            note=why or "Permission was not given, so it will not start at boot yet.")

    path = Path.home() / ".config" / "systemd" / "user" / f"{SERVICE}.service"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)
    subprocess.run(["systemctl", "--user", "daemon-reload"],
                   capture_output=True, check=False)
    done = subprocess.run(["systemctl", "--user", "enable", "--now", SERVICE],
                          capture_output=True, text=True, check=False)
    if done.returncode != 0:
        return Autostart(mode, installed=False, path=path,
                         command=f"systemctl --user enable --now {SERVICE}",
                         note=done.stderr.strip() or "systemctl refused the unit")
    subprocess.run(["loginctl", "enable-linger", os.environ.get("USER", "")],
                   capture_output=True, check=False)
    return Autostart(mode, installed=True, path=path)


# -- Windows -------------------------------------------------------------
def _windows_install(mode: str, argv: list[str], log_dir: Path) -> Autostart:
    quoted = " ".join(f'"{a}"' if " " in a else a for a in argv)
    if mode == "boot":
        command = (f'schtasks /Create /F /TN "{LABEL}" /TR "{quoted}" '
                   f'/SC ONSTART /RL HIGHEST /RU SYSTEM')
        try:
            done = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 f"Start-Process -Verb RunAs -Wait -FilePath cmd -ArgumentList '/c {command}'"],
                capture_output=True, text=True, timeout=120)
            if done.returncode == 0:
                return Autostart(mode, installed=True)
        except (OSError, subprocess.SubprocessError):
            pass
        return Autostart(mode, installed=False, command=command,
                         note="Permission was not given, so it will not start at boot.")
    startup = (Path(os.environ.get("APPDATA", Path.home()))
               / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Startup")
    startup.mkdir(parents=True, exist_ok=True)
    path = startup / f"{SERVICE}.cmd"
    path.write_text(f'@echo off\r\nstart "" {quoted} >> "{log_dir / "traind.log"}" 2>&1\r\n')
    return Autostart(mode, installed=True, path=path)


# -- the interface -------------------------------------------------------
def plan(mode: str, *, slots: int = 1, labels: tuple[str, ...] = (),
         report: str = "") -> list[str]:
    """The exact command line that would be installed. Shown before anything is."""
    return _executable() + _args(slots, labels, report)


def install(mode: str, *, slots: int = 1, labels: tuple[str, ...] = (),
            report: str = "", log_dir: Path | str | None = None) -> Autostart:
    """Arrange for the daemon to start, or explain what a human must run."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    if mode == "manual":
        return Autostart("manual", installed=True,
                         note="Nothing installed. Start it yourself with "
                              "'ml-stack-traind'.")
    logs = Path(log_dir).expanduser() if log_dir else Path.home() / ".ml-stack"
    logs.mkdir(parents=True, exist_ok=True)
    argv = plan(mode, slots=slots, labels=labels, report=report)
    if sys.platform == "darwin":
        return _mac_install(mode, argv, logs)
    if sys.platform == "win32":
        return _windows_install(mode, argv, logs)
    return _systemd_install(mode, argv, logs)


def uninstall(mode: str = "") -> list[Path]:
    """Remove whatever was installed. Returns what it removed."""
    removed: list[Path] = []
    candidates: list[Path] = []
    if sys.platform == "darwin":
        candidates = [_mac_path("login"), _mac_path("boot")]
    elif sys.platform == "win32":
        candidates = [Path(os.environ.get("APPDATA", Path.home())) / "Microsoft"
                      / "Windows" / "Start Menu" / "Programs" / "Startup"
                      / f"{SERVICE}.cmd"]
    else:
        candidates = [Path.home() / ".config" / "systemd" / "user" / f"{SERVICE}.service",
                      Path("/etc/systemd/system") / f"{SERVICE}.service"]
    for path in candidates:
        if not path.exists():
            continue
        if sys.platform == "darwin":
            subprocess.run(["launchctl", "unload", "-w", str(path)],
                           capture_output=True, check=False)
        elif sys.platform.startswith("linux"):
            subprocess.run(["systemctl", "--user", "disable", "--now", SERVICE],
                           capture_output=True, check=False)
        try:
            path.unlink()
            removed.append(path)
        except OSError:
            continue
    return removed


def status() -> dict[str, object]:
    """Which mode, if any, is currently installed on this machine."""
    out: dict[str, object] = {"platform": sys.platform, "mode": "manual", "paths": []}
    checks = {
        "darwin": {"login": _mac_path("login"), "boot": _mac_path("boot")},
        "win32": {"login": Path(os.environ.get("APPDATA", Path.home())) / "Microsoft"
                  / "Windows" / "Start Menu" / "Programs" / "Startup" / f"{SERVICE}.cmd"},
    }.get(sys.platform, {
        "login": Path.home() / ".config" / "systemd" / "user" / f"{SERVICE}.service",
        "boot": Path("/etc/systemd/system") / f"{SERVICE}.service",
    })
    for mode, path in checks.items():
        if path.exists():
            out["mode"] = mode
            out["paths"].append(str(path))       # type: ignore[union-attr]
    return out
