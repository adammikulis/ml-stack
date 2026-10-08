"""Observable probes that say whether a session is inside the agent sandbox."""

from __future__ import annotations

import http.client
import os
import socket
import subprocess
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import agent_sandbox_profile as profile

PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"
DENIED_HOST = "example.com"


@dataclass(frozen=True)
class Result:
    """One probe's outcome."""

    name: str
    status: str
    detail: str = ""


def can_write(path: Path) -> bool:
    """Whether a new file can be created at ``path``; the file is removed again."""
    try:
        with path.open("x"):
            pass
    except OSError:
        return False
    path.unlink(missing_ok=True)
    return True


def can_read(path: Path) -> bool:
    """Whether ``path`` can be listed or opened."""
    try:
        if path.is_dir():
            next(iter(os.scandir(path)), None)
        else:
            with path.open("rb") as handle:
                handle.read(1)
    except OSError:
        return False
    return True


def connect_state(host: str, port: int) -> str:
    """``open``, ``refused`` (reachable, nothing listening) or ``blocked``."""
    try:
        with socket.create_connection((host, port), timeout=3):
            return "open"
    except ConnectionRefusedError:
        return "refused"
    except OSError:
        return "blocked"


def fetch_state(host: str) -> str:
    """``reached`` when ``https://host/`` answers with a success status, otherwise ``blocked``."""
    try:
        link = http.client.HTTPSConnection(host, timeout=8)
        link.request("GET", "/")
        status = link.getresponse().status
        link.close()
    except (http.client.HTTPException, OSError):
        return "blocked"
    return "reached" if status < 400 else "blocked"


def run(argv: list[str], cwd: Path) -> tuple[int, str]:
    """Return a command's exit code and its last output line."""
    done = subprocess.run(argv, cwd=cwd, capture_output=True, text=True, timeout=600, check=False)
    lines = (done.stdout + done.stderr).strip().splitlines()
    return done.returncode, lines[-1] if lines else ""


def _denied_write(name: str, path: Path) -> Result:
    if not path.parent.is_dir():
        return Result(name, SKIP, f"{path.parent} does not exist")
    return Result(name, PASS if not can_write(path) else FAIL, str(path))


def _denied_read(name: str, path: Path) -> Result:
    if not path.exists():
        return Result(name, SKIP, f"{path} does not exist")
    return Result(name, PASS if not can_read(path) else FAIL, str(path))


def filesystem_probes(layout: profile.Layout, checkout: Path) -> list[Result]:
    """Writes and reads that the profile must refuse, and writes it must allow."""
    tag = f".agent-sandbox-probe-{os.getpid()}"
    out = [
        _denied_write("write outside the allow list (home directory)", layout.home / tag),
        _denied_write("write a hook script", checkout / "scripts" / "hooks" / tag),
        _denied_write("write a Claude Code transcript path", layout.home / ".claude" / "projects" / tag),
        _denied_write("write a git hook", checkout / ".git" / "hooks" / tag),
        _denied_write("write the runtime tree", layout.state / "runtimes" / tag),
        _denied_write("write a launch agent", layout.home / "Library" / "LaunchAgents" / tag),
        _denied_write("write a systemd user unit", layout.home / ".config" / "systemd" / "user" / tag),
        _denied_read("read another agent's token", layout.state / "workspace" / "tokens" / "codex"),
        _denied_read("read the keystore state", layout.state / "keystore"),
        _denied_read("read ssh keys", layout.home / ".ssh"),
        _denied_read("read cloud credentials", layout.home / ".aws"),
        _denied_read("read the gh config", layout.home / ".config" / "gh"),
    ]
    with tempfile.TemporaryDirectory() as scratch:
        allowed = can_write(Path(scratch) / tag)
    out.append(Result("write the temp directory", PASS if allowed else FAIL))
    out.append(Result("write inside the checkout", PASS if can_write(checkout / tag) else FAIL, str(checkout)))
    return out


def environment_probes(environ: dict[str, str]) -> list[Result]:
    """One probe for each variable the profile scrubs."""
    return [Result(f"env {name} absent", FAIL if environ.get(name) else PASS)
            for name in profile.SCRUBBED_ENV]


def network_probes() -> list[Result]:
    """Loopback to the board port, and a host that is not on the list."""
    board = connect_state("127.0.0.1", int(profile.BOARD.rsplit(":", 1)[1]))
    host = fetch_state(DENIED_HOST)
    return [
        Result("reach 127.0.0.1:8770", FAIL if board == "blocked" else PASS, board),
        Result("reach a host off the list", PASS if host == "blocked" else FAIL, host),
    ]


def git_probe() -> Result:
    """Push to a local bare repository from a temporary clone."""
    with tempfile.TemporaryDirectory() as scratch:
        base = Path(scratch)
        bare, work = base / "bare.git", base / "work"
        steps = [
            ["git", "init", "-q", "--bare", str(bare)],
            ["git", "init", "-q", str(work)],
            ["git", "-C", str(work), "-c", "user.name=probe", "-c", "user.email=probe@example.invalid",
             "commit", "-q", "--allow-empty", "-m", "chore: probe"],
            ["git", "-C", str(work), "push", "-q", str(bare), "HEAD:refs/heads/probe"],
        ]
        for argv in steps:
            code, line = run(argv, base)
            if code:
                return Result("git push to a local bare repository", FAIL, line)
    return Result("git push to a local bare repository", PASS)


def test_probe(checkout: Path) -> Result:
    """Run this tree's own sandbox tests through the broker."""
    code, line = run(["scripts/test", "all", "-n", "1", "tests/test_agent_sandbox.py"], checkout)
    return Result("scripts/test on the sandbox tests", PASS if code == 0 else FAIL, line)


def all_probes(layout: profile.Layout, checkout: Path, environ: dict[str, str] | None = None,
               extra: list[Callable[[], Result]] | None = None) -> list[Result]:
    """Every probe, in the order they print."""
    out = filesystem_probes(layout, checkout)
    out += environment_probes(dict(os.environ) if environ is None else environ)
    out += network_probes()
    out.append(git_probe())
    out += [step() for step in extra or []]
    return out


def render(results: list[Result]) -> str:
    """One PASS/FAIL/SKIP line per result."""
    return "\n".join(f"{item.status}  {item.name}" + (f"  ({item.detail})" if item.detail else "")
                     for item in results)


def inside(results: list[Result]) -> bool:
    """Whether the probes show a working sandbox: nothing failed and something was refused."""
    return not any(item.status == FAIL for item in results) and any(item.status == PASS for item in results)
