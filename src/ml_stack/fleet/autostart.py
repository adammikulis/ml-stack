"""Putting ml-stack on a machine and keeping it there.

Four things, and the network installer (`packaging/install.sh`, `install.ps1`) is a thin
caller of each: what starts the daemon at **login** (`install`) or at **boot**
(`system_service`), how it comes back on new code (`restart`), what happens to the **model
cache** already on the disk (`plan_cache` -- moved, or better, read where it is), and which
**model** a new machine should start with (`choose_model`, from the measured profiles).

Everything here that needs root only *generates* the thing root would write, so every
platform's shape has a test on a machine with no root at all.
"""

from __future__ import annotations

import os
import plistlib
import shutil
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = ["ADOPTED", "Autostart", "CachePlan", "DEFAULT_MODEL", "IN_PLACE", "LABEL",
           "LEFT_ALONE", "SYSTEM_LABEL", "SystemService", "choose_model", "install",
           "main", "models_in", "plan", "plan_cache", "restart", "service_environment",
           "status", "system_service", "uninstall"]

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


# -- what a new machine should start with ------------------------------------------------
DEFAULT_MODEL = "gemma-4-E2B-it-qat-UD-Q4_K_XL"
"""What a first install downloads unless asked otherwise: 2.6G, measured at 1.5 s a
question. The smallest thing that still answers, so the first thing a person does on a new
machine finishes while they are still watching. The bigger ones are a pick, not a default:
a machine with the room is offered E4B (4.4G, 3 s) and Flash-Next (104G, 27 s), and
``--models auto`` takes the best that fits, which is what a headless install does."""


def needs_bytes(fit: dict[str, Any], context: int = 0) -> int:
    """What one measured fit says a model costs to serve: weights, draft, cache, compute.

    The numbers come from `ml_stack.data.fit.json`, which is measured rather than guessed;
    ``context`` overrides the context that measurement used.
    """
    seats = int(context or fit.get("context") or 0)
    return (int(fit.get("weights") or 0) + int(fit.get("draft") or 0)
            + int(fit.get("per_token") or 0) * seats
            + int(fit.get("per_seq") or 0) + int(fit.get("compute") or 0))


def choose_model(room_bytes: int, *, want: str = "auto",
                 profiles: "Sequence[Any] | None" = None,
                 fits: "Sequence[dict[str, Any]] | None" = None) -> "dict[str, Any] | None":
    """The best model this machine has room for, out of the ones that were measured.

    ``profiles`` is in the order the measurements ranked them, so this walks it and takes
    the first whose smallest measured fit leaves a fifth of the room spare -- serving a
    model with nothing left over is a machine that swaps the moment somebody asks it two
    questions. ``want`` narrows it: a word matched against the model's name (``flash-next``,
    ``small``), ``none`` picks nothing, ``auto`` takes the ranking as it stands.

    Returns the model, the draft head beside it and the build it needs, which is what
    ``ml-stack-models fetch`` and ``ml-stack-serve build`` take -- or None when nothing
    measured fits, which is an answer: the caller says so rather than fetching a model the
    machine cannot load.
    """
    asked = want.strip().lower()
    if asked in ("none", "off", ""):
        return None
    if asked == "default":
        asked = DEFAULT_MODEL.lower()
    if profiles is None:
        from ml_stack.serve.profile import profiles as read_profiles

        profiles = read_profiles()
    if fits is None:
        import json as _json

        from ml_stack.data import __file__ as data_init

        fits = _json.loads((Path(data_init).parent / "fit.json").read_text())
    by_model: dict[str, int] = {}
    for one in fits or ():
        name = str(one.get("model") or "")
        cost = needs_bytes(one)
        if name and (name not in by_model or cost < by_model[name]):
            by_model[name] = cost

    narrow = "" if asked == "auto" else asked
    for profile in profiles or ():
        model = str(getattr(profile, "model", "") or "")
        if not model or (narrow and narrow not in model.lower()):
            continue
        cost = by_model.get(model, 0)
        if cost and room_bytes and cost * 1.2 > room_bytes:
            continue
        return {"model": model,
                "draft": str(getattr(profile, "draft", "") or ""),
                "build": str(getattr(profile, "build", "") or ""),
                "bytes": cost}
    return None


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


# -- per-machine, at boot, as the person who installed it --------------------------------
SYSTEM_LABEL = f"{LABEL}.system"


@dataclass(frozen=True, slots=True)
class SystemService:
    """A definition that starts the daemon at boot, before anybody logs in.

    Generated, never installed from here: writing it needs root, and this is a pure
    function so every platform's shape has a test on a machine with no root at all.
    ``install`` is the exact elevated line, which is what an installer runs and what a
    person is shown when they refuse it.

    It runs **as the user who installed it**, not as SYSTEM or a service account. That is
    the whole trick for models: ``~/.cache/huggingface`` is already full of weights on the
    machine somebody has been using, and a service under another account would have its own
    empty one and download every model again. Same user, same home, one cache -- nothing is
    copied, symlinked or fetched twice. ``HOME``, ``HF_HOME`` and ``ML_STACK_CACHE`` are
    written into the definition as well, because a process started at boot inherits no
    login environment to work them out from.
    """

    platform: str
    label: str
    path: str
    body: str
    install: str
    user: str
    environment: dict[str, str]


DEFAULT_PATH = "/usr/local/bin:/usr/bin:/bin"


def service_environment(home: "Path | str", *, path: str = "") -> dict[str, str]:
    """What the daemon needs in its environment when nobody logged in to give it one.

    ``HF_HOME`` is the point: it names the cache the models are already in, so a service
    that starts at boot resolves to the same directory the person's own ``ml-stack-models
    fetch`` writes to, rather than a second one under a service account's home.
    """
    where = Path(home).expanduser()
    return {
        "HOME": str(where),
        "HF_HOME": str(where / ".cache" / "huggingface"),
        "ML_STACK_CACHE": str(where / ".cache" / "ml_stack"),
        "PATH": path or DEFAULT_PATH,
    }


def system_service(user: str, home: "Path | str", *, argv: "list[str] | None" = None,
                   log_dir: "Path | str" = "/var/log", platform: str = "",
                   environment: "dict[str, str] | None" = None) -> SystemService:
    """The boot-time service definition for ``platform`` (this one unless named).

    macOS gets a LaunchDaemon with ``UserName``: it starts at boot with no login, and
    ``KeepAlive`` brings it back, which is simpler than anything else that survives a
    reboot on a Mac. Linux gets a system unit with ``User=``. Windows gets a Scheduled Task
    at ``ONSTART`` with ``/RU <user>`` -- chosen over ``sc create`` because a real service
    needs a service wrapper to hold a long-lived Python process, while a task at startup is
    one line, survives reboots and, unlike ``/RU SYSTEM``, keeps the model cache the person
    already has.
    """
    plat = platform or sys.platform
    argv = argv or _executable()
    # Never the installing shell's PATH: under sudo that is root's, and it may hold a
    # tilde no boot-time process expands. The venv the daemon was installed into, then a
    # plain system PATH.
    where = Path(argv[0]).parent
    env = dict(environment or service_environment(home, path=f"{where}:{DEFAULT_PATH}"
                                                  if plat != "win32" else str(where)))
    logs = Path(log_dir)
    if plat == "darwin":
        body = plistlib.dumps({
            "Label": SYSTEM_LABEL,
            "ProgramArguments": argv,
            "RunAtLoad": True,
            "KeepAlive": {"SuccessfulExit": False},
            "UserName": user,
            "StandardOutPath": str(logs / "ml-stack-traind.log"),
            "StandardErrorPath": str(logs / "ml-stack-traind.log"),
            "WorkingDirectory": str(Path(home).expanduser()),
            "EnvironmentVariables": env,
        }).decode()
        path = f"/Library/LaunchDaemons/{SYSTEM_LABEL}.plist"
        return SystemService(plat, SYSTEM_LABEL, path, body,
                             f"launchctl load -w '{path}'", user, env)
    if plat == "win32":
        path = SYSTEM_LABEL
        quoted = _quote(argv)
        body = (f'schtasks /Create /F /TN "{SYSTEM_LABEL}" /TR "{quoted}" '
                f'/SC ONSTART /RU "{user}" /RL HIGHEST')
        return SystemService(plat, SYSTEM_LABEL, path, body, body, user, env)
    lines = "\n".join(f"Environment={k}={v}" for k, v in sorted(env.items()))
    body = ("[Unit]\n"
            "Description=ml-stack training daemon (per machine)\n"
            "After=network-online.target\n"
            "Wants=network-online.target\n\n"
            "[Service]\n"
            "Type=simple\n"
            f"User={user}\n"
            f"{lines}\n"
            f"ExecStart={' '.join(argv)}\n"
            "Restart=on-failure\n"
            "RestartSec=5\n\n"
            "[Install]\n"
            "WantedBy=multi-user.target\n")
    path = f"/etc/systemd/system/{SERVICE}.service"
    return SystemService(plat, SERVICE, path, body,
                         f"systemctl daemon-reload && systemctl enable --now {SERVICE}",
                         user, env)


# -- the models already on the disk ------------------------------------------------------
IN_PLACE, ADOPTED, LEFT_ALONE = "in place", "adopted", "left alone"


@dataclass(frozen=True, slots=True)
class CachePlan:
    """What a per-machine install did about the model cache that was already there.

    Three outcomes and no others. **in place**: the service runs as the person who
    installed it, so it opens their ``~/.cache/huggingface`` exactly where it is -- nothing
    is moved, linked or fetched. **adopted**: the service must run as another account, and
    they said yes, so the cache directory is *moved* to the shared location and a symlink
    left at the old path -- their own tools keep working and every file exists once.
    **left alone**: they said no, so their cache is untouched and the service starts with
    an empty one and will download what it needs.

    Never a copy. A cache is tens of gigabytes and a machine with two of them is a machine
    with a full disk and no idea why.
    """

    decision: str
    user_cache: Path
    service_cache: Path
    models: tuple[tuple[str, int], ...] = ()
    bytes: int = 0
    said: str = ""
    error: str = ""

    def public(self) -> dict[str, Any]:
        return {"decision": self.decision, "user_cache": str(self.user_cache),
                "service_cache": str(self.service_cache),
                "models": [{"name": n, "bytes": b} for n, b in self.models],
                "bytes": self.bytes, "said": self.said, "error": self.error}


def _size(path: Path) -> int:
    total = 0
    for item in path.rglob("*"):
        try:
            if item.is_file() and not item.is_symlink():
                total += item.stat().st_size
        except OSError:
            continue
    return total


def _gb(n: int) -> str:
    return f"{n / 1e9:.1f} GB" if n >= 1e8 else f"{n / 1e6:.0f} MB"


def models_in(cache: "Path | str") -> list[tuple[str, int]]:
    """Every model in a Hub cache, with its size on disk. Biggest first.

    The Hub's layout is one ``models--owner--repo`` directory per repository; anything else
    in there is not a model and is not counted.
    """
    hub = Path(cache).expanduser()
    hub = hub / "hub" if (hub / "hub").is_dir() else hub
    if not hub.is_dir():
        return []
    found = [(one.name.removeprefix("models--").replace("--", "/"), _size(one))
             for one in sorted(hub.iterdir())
             if one.is_dir() and one.name.startswith("models--")]
    return sorted(found, key=lambda pair: -pair[1])


def plan_cache(user_cache: "Path | str", service_cache: "Path | str", *,
               same_user: bool, adopt: bool = False) -> CachePlan:
    """Decide -- and carry out -- what happens to the models already on this machine.

    ``same_user`` is the good case and the default an installer should aim for: the service
    account *is* the person, so their cache is the service's cache and this does nothing at
    all. Otherwise ``adopt`` moves it (with a symlink left behind) and not adopting leaves
    it alone; neither ever happens silently -- the caller asks, having been given
    `models_in` to show what is at stake.

    Running it twice is a no-op: an already-adopted cache is a symlink at the old path
    pointing at the new one, and that is recognised rather than moved again.
    """
    mine = Path(user_cache).expanduser()
    theirs = Path(service_cache).expanduser()
    found = tuple(models_in(mine))
    total = sum(b for _, b in found)

    if same_user:
        return CachePlan(IN_PLACE, mine, mine, found, total,
                         said=f"the service runs as this user, so it reads {mine} where it "
                              f"is -- {len(found)} model(s), {_gb(total)}, nothing moved")
    if mine.is_symlink():
        target = mine.resolve()
        if target == theirs.resolve():
            return CachePlan(ADOPTED, mine, theirs, tuple(models_in(theirs)),
                             _size(theirs) if theirs.is_dir() else 0,
                             said=f"{mine} already points at {theirs}; nothing to do")
    if not adopt:
        listing = ", ".join(f"{n} ({_gb(b)})" for n, b in found[:5]) or "nothing"
        return CachePlan(LEFT_ALONE, mine, theirs, found, total,
                         said=f"{mine} is left alone ({listing}). The service runs as "
                              f"another account and starts with an empty cache, so it will "
                              f"download what it needs again -- {_gb(total)} of it. "
                              f"Re-run with --adopt-cache to move it instead.")
    if not mine.is_dir():
        return CachePlan(LEFT_ALONE, mine, theirs, said=f"there is no cache at {mine}")
    theirs.parent.mkdir(parents=True, exist_ok=True)
    if theirs.exists():
        return CachePlan(LEFT_ALONE, mine, theirs, found, total,
                         error=f"{theirs} already exists; move or remove it and re-run, "
                               f"rather than having two caches")
    try:
        os.rename(mine, theirs)
    except OSError as exc:
        # A cross-device rename would be a copy, and a copy is the one thing this must
        # never do: it doubles tens of gigabytes on a disk that has them once.
        return CachePlan(LEFT_ALONE, mine, theirs, found, total,
                         error=f"could not move {mine} to {theirs} without copying it "
                               f"({exc}); put the shared cache on the same filesystem, or "
                               f"run the service as this user instead")
    mine.symlink_to(theirs, target_is_directory=True)
    return CachePlan(ADOPTED, mine, theirs, found, total,
                     said=f"moved {len(found)} model(s), {_gb(total)}, to {theirs} and left "
                          f"a link at {mine} -- every file exists once, and both accounts "
                          f"read the same one")


def _run(argv: list[str]) -> int:
    """A short command whose output nobody reads; its exit code is the answer."""
    try:
        return subprocess.run(argv, capture_output=True, timeout=60).returncode
    except (OSError, subprocess.SubprocessError):
        return 1


def _reexec() -> None:
    """Replace this process with the same command line, reading the code now on disk."""
    os.execv(sys.executable, [sys.executable, *sys.argv])


def restart(*, run: "Callable[[list[str]], int] | None" = None,
            reexec: "Callable[[], None] | None" = None) -> str:
    """Bring the daemon back on the code that is now on disk. Says how it did it.

    Where a login service is installed, stopping is enough: launchd's ``KeepAlive`` and
    systemd's ``Restart=on-failure`` start it again, so ``launchctl kickstart -k`` and
    ``systemctl restart`` do the whole thing and this process does not survive the call
    ("service"). Windows is the exception -- its Scheduled Task fires at logon and would
    not fire again -- so it re-execs like a daemon somebody started by hand ("exec").

    ``run`` and ``reexec`` are the seams; a test replaces both and nothing is killed.
    """
    call = run or _run
    mode = str(status()["mode"])
    if mode in ("login", "boot") and sys.platform != "win32":
        if sys.platform == "darwin":
            domain = "system" if mode == "boot" else f"gui/{os.getuid()}"
            if call(["launchctl", "kickstart", "-k", f"{domain}/{LABEL}"]) == 0:
                return "service"
            path = _mac_path(mode)
            call(["launchctl", "unload", str(path)])
            if call(["launchctl", "load", "-w", str(path)]) == 0:
                return "service"
        else:
            scope = ["--user"] if mode == "login" else []
            if call(["systemctl", *scope, "restart", SERVICE]) == 0:
                return "service"
    (reexec or _reexec)()
    return "exec"


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


# -- the installer's side of it ----------------------------------------------------------
def main(argv: "list[str] | None" = None) -> int:
    """What ``packaging/install.sh`` calls rather than writing any of it in shell.

    Three questions an installer has and a shell script should not answer for itself:
    ``system`` (write the boot service, or ``--print`` it), ``cache`` (what to do about the
    models already on the disk) and ``choose`` (which model this machine has room for).
    """
    import argparse
    import json as _json

    ap = argparse.ArgumentParser(prog="python -m ml_stack.fleet.autostart",
                                 description="what the installer asks about this machine")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sysp = sub.add_parser("system", help="the boot-time service, as the installing user")
    sysp.add_argument("--user", required=True)
    sysp.add_argument("--home", required=True)
    sysp.add_argument("--print", dest="only_print", action="store_true",
                      help="print the definition instead of installing it (needs no root)")

    cachep = sub.add_parser("cache", help="what happens to the models already here")
    cachep.add_argument("--user-cache", required=True)
    cachep.add_argument("--service-cache", default="")
    cachep.add_argument("--same-user", action="store_true",
                        help="the service runs as this user, so its cache is read in place")
    cachep.add_argument("--adopt", action="store_true",
                        help="move the cache to the shared path and link back to it")
    cachep.add_argument("--json", action="store_true", help="print the answer as JSON")

    pickp = sub.add_parser("choose", help="which measured model fits in this much room")
    pickp.add_argument("--room", type=int, default=0, metavar="BYTES")
    pickp.add_argument("--want", default="auto")
    pickp.add_argument("--json", action="store_true", help="print the answer as JSON")

    a = ap.parse_args(argv)

    if a.cmd == "system":
        made = system_service(a.user, a.home)
        if a.only_print:
            print(made.body)
            return 0
        target = Path(made.path)
        try:
            if made.platform == "win32":
                if _run(["cmd", "/c", made.body]) != 0:
                    print(f"could not register the task; run as administrator:\n  {made.body}",
                          file=sys.stderr)
                    return 2
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(made.body)
        except OSError as exc:
            print(f"needs root: {exc}\n  {made.install}", file=sys.stderr)
            return 2
        if made.platform == "darwin":
            _run(["launchctl", "unload", str(target)])
            _run(["launchctl", "load", "-w", str(target)])
        elif made.platform.startswith("linux"):
            _run(["systemctl", "daemon-reload"])
            _run(["systemctl", "enable", "--now", SERVICE])
        print(f"starts at boot as {made.user}: {made.path}")
        return 0

    if a.cmd == "cache":
        shared = a.service_cache or str(Path("/opt/ml-stack/cache/huggingface"))
        got = plan_cache(a.user_cache, shared, same_user=a.same_user, adopt=a.adopt)
        print(_json.dumps(got.public(), indent=1) if a.json else
              f"model cache {got.decision}: {got.said or got.error}")
        return 2 if got.error else 0

    picked = choose_model(a.room, want=a.want)
    if picked is None:
        print("{}" if a.json else "no measured model fits this machine; none fetched")
        return 1
    print(_json.dumps(picked) if a.json else
          " ".join(x for x in (picked["model"], picked["draft"]) if x))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
