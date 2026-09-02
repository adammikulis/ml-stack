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


def _runs(argv: list[str]) -> bool:
    """Whether a command answers ``--help``, which starts nothing."""
    try:
        done = subprocess.run([*argv, "--help"], capture_output=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return False
    return done.returncode == 0


def _executable() -> list[str]:
    """How to start the daemon on this machine."""
    found = shutil.which(SERVICE)
    if found and _runs([found]):
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
# Two Scheduled Tasks, the way `ml-stack-serve build --persist` keeps llama.cpp fresh:
# LABEL at boot (as SYSTEM, needs an administrator once) and LOGIN_TASK at this user's
# logon (needs nothing). Both run a .cmd wrapper rather than the daemon directly, because a
# task's /TR cannot redirect output and a daemon with no log is a daemon nobody can debug.
# The Startup-folder .cmd is what older installs used for logon; it is still removed, and
# still the fallback when schtasks refuses.
LOGIN_TASK = f"{LABEL}.login"


def _windows_startup() -> Path:
    return (Path(os.environ.get("APPDATA", Path.home()))
            / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Startup"
            / f"{SERVICE}.cmd")


def _windows_wrapper(log_dir: Path) -> Path:
    return log_dir / f"{SERVICE}.cmd"


def _windows_wrapper_body(argv: list[str], log_dir: Path) -> str:
    quoted = _quote(argv)
    return f'@echo off\r\n{quoted} >> "{log_dir / "traind.log"}" 2>&1\r\n'


def _quote(argv: list[str]) -> str:
    return " ".join(f'"{a}"' if " " in a else a for a in argv)


def _windows_task() -> str:
    return f'schtasks /Delete /F /TN "{LABEL}"'


def _windows_task_exists() -> bool:
    """Whether the boot task is registered."""
    return _windows_task_named(LABEL)


def _windows_login_task_exists() -> bool:
    return _windows_task_named(LOGIN_TASK)


def _windows_task_named(name: str) -> bool:
    try:
        done = subprocess.run(["schtasks", "/Query", "/TN", name],
                              capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return False
    return done.returncode == 0


def _windows_install(mode: str, argv: list[str], log_dir: Path) -> Autostart:
    quoted = _quote(argv)
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

    wrapper = _windows_wrapper(log_dir)
    wrapper.parent.mkdir(parents=True, exist_ok=True)
    wrapper.write_text(_windows_wrapper_body(argv, log_dir))
    create = ["schtasks", "/Create", "/F", "/TN", LOGIN_TASK, "/TR", f'"{wrapper}"',
              "/SC", "ONLOGON", "/RL", "LIMITED"]
    try:
        done = subprocess.run(create, capture_output=True, text=True, timeout=60)
        refused = "" if done.returncode == 0 else (
            (done.stderr or done.stdout or "").strip() or "schtasks refused the task")
    except (OSError, subprocess.SubprocessError) as exc:
        refused = str(exc)
    if not refused:
        # Start it now as well: a logon trigger fires at the next logon, and "starts when
        # you log in" that does nothing until tomorrow reads as broken.
        subprocess.run(["schtasks", "/Run", "/TN", LOGIN_TASK],
                       capture_output=True, check=False)
        return Autostart(mode, installed=True, path=wrapper,
                         note=f"scheduled task {LOGIN_TASK!r} runs it at logon; "
                              f"log at {log_dir / 'traind.log'}")
    # The Startup folder needs no schtasks and no permission; it starts the daemon in a
    # visible console window, which is the price.
    path = _windows_startup()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_windows_wrapper_body(argv, log_dir))
    return Autostart(mode, installed=True, path=path,
                     note=f"schtasks refused ({refused}); placed in the Startup folder "
                          "instead, which starts it in a console window at logon")


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
    uninstall()
    if mode == "manual":
        left = _left_behind()
        if left:
            return Autostart("manual", installed=False, command=left,
                             note="Removing what starts it at boot needs "
                                  "administrator rights.")
        return Autostart("manual", installed=True,
                         note="Nothing installed. Start it yourself with "
                              "'ml-stack-traind'.")
    logs = Path(log_dir).expanduser() if log_dir else Path.home() / ".ml-stack"
    logs.mkdir(parents=True, exist_ok=True)
    argv = plan(mode, slots=slots, labels=labels, report=report)
    if not _runs(argv):
        return Autostart(
            mode, installed=False, command=" ".join(argv),
            note="This machine has no working ml-stack to start. Installing it here "
                 "would leave the operating system retrying a command that fails.")
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
        candidates = [_windows_startup(), _windows_wrapper(Path.home() / ".ml-stack")]
        if _windows_login_task_exists():
            subprocess.run(["schtasks", "/End", "/TN", LOGIN_TASK],
                           capture_output=True, check=False)
            subprocess.run(["schtasks", "/Delete", "/F", "/TN", LOGIN_TASK],
                           capture_output=True, check=False)
        if _windows_task_exists():
            subprocess.run(["schtasks", "/Delete", "/F", "/TN", LABEL],
                           capture_output=True, check=False)
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


def _left_behind() -> str:
    """What a person must run by hand to remove what is still installed."""
    paths = [Path(p) for p in status()["paths"]]  # type: ignore[union-attr]
    if sys.platform == "win32":
        return _windows_task() if _windows_task_exists() else ""
    if not paths:
        return ""
    return "sudo rm " + " ".join(f"'{p}'" for p in paths)


def status() -> dict[str, object]:
    """Which mode, if any, is currently installed on this machine."""
    out: dict[str, object] = {"platform": sys.platform, "mode": "manual", "paths": []}
    checks = {
        "darwin": {"login": _mac_path("login"), "boot": _mac_path("boot")},
        "win32": {"login": _windows_startup()},
    }.get(sys.platform, {
        "login": Path.home() / ".config" / "systemd" / "user" / f"{SERVICE}.service",
        "boot": Path("/etc/systemd/system") / f"{SERVICE}.service",
    })
    for mode, path in checks.items():
        if path.exists():
            out["mode"] = mode
            out["paths"].append(str(path))       # type: ignore[union-attr]
    if sys.platform == "win32":
        if _windows_login_task_exists():
            out["mode"] = "login"
            out["paths"].append(LOGIN_TASK)      # type: ignore[union-attr]
        if _windows_task_exists():
            out["mode"] = "boot"
            out["paths"].append(LABEL)           # type: ignore[union-attr]
    return out
