"""Shared fixtures for the autostart tests: a state root and a user home in a temporary directory, launchers, a service-manager recorder."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ml_stack import home
from ml_stack.files import writing
from ml_stack.fleet.autostart_apply import Seams
from ml_stack.fleet.autostart_backends import backend_for
from ml_stack.fleet.autostart_manifest import LAUNCHER_HEAD, ROLE_INFO
from ml_stack.fleet.autostart_prepare import Spec, prepare

PERSON = {"terminal": (True, True), "env": {}}


@dataclass
class Recorder:
    """A stand-in for the service manager's command line: remembers each command and answers probes by `alive`."""

    alive: bool = True
    fail_load: bool = False
    ran: list[list[str]] = field(default_factory=list)

    def __call__(self, argv: list[str]) -> tuple[int, str]:
        self.ran.append(list(argv))
        words = " ".join(argv)
        if any(w in words for w in ("bootout", "disable", "/Delete")):
            self.alive = False
            return 0, ""
        if any(w in words for w in ("bootstrap", "enable --now", "/Create")):
            if self.fail_load:
                return 1, "load refused"
            self.alive = True
            return 0, ""
        if any(w in words for w in ("print", "is-active", "/Query")):
            return (0, "") if self.alive else (3, "")
        return 0, ""


def write_launcher(directory: Path, name: str, tag: str = "a", mode: int = 0o700) -> Path:
    """A launcher file with the runtime tooling's header, pointing at a runtime root under the state root."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    root = home.state("runtimes", "test-identity")
    with writing(path) as temporary:
        temporary.write_text(f"{LAUNCHER_HEAD}{str(root)!r}\nimport sys\nprint({tag!r}, *sys.argv[1:])\n", encoding="utf-8")
        temporary.chmod(mode)
    return path


def launchers(directory: Path, platform: str = "linux") -> Path:
    """A launcher directory holding a launcher for every role."""
    for info in ROLE_INFO.values():
        write_launcher(directory, info["launcher"] + (".exe" if platform == "win32" else ""))
    return directory


def seams(recorder: Recorder, platform: str = "linux", scope: str = "user", **more) -> Seams:
    """Seams for a person at a terminal answering the confirmation with the manifest id."""
    given = {"backend": backend_for(platform, scope, recorder), "probe": lambda health: True,
             "platform": platform, "show": lambda line: None, **PERSON, **more}
    return Seams(**given)


def staged(tmp_path: Path, roles=("pool-daemon", "runtime-ensure"), platform="linux", **more):
    """Prepare the roles with launchers under the state root; returns the prepared manifest."""
    spec = Spec(tuple(roles), launchers=launchers(home.state("bin"), platform), platform=platform, **more)
    return prepare(spec)


def environment(tmp_path: Path, monkeypatch) -> Path:
    """Move the state root and the account home into ``tmp_path``; returns the user home."""
    user = tmp_path / "user"
    user.mkdir()
    monkeypatch.setenv("HOME", str(user))
    monkeypatch.setenv("ML_STACK_HOME", str(tmp_path / "state"))
    for name in ("CLAUDECODE", "ML_STACK_AGENT", "ML_STACK_NONINTERACTIVE", "HF_HOME", "ML_STACK_CACHE"):
        monkeypatch.delenv(name, raising=False)
    return user
