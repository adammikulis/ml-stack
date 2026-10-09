"""Set this device up to join a pool, the same one command on macOS, Linux, Windows and WSL.

    python -m ml_stack.device_setup [--yes] [--dry-run] [--wait SECONDS] [--policy open|secure]

It lists every change this device needs, asks one yes (`--yes` for an agent or script that is already allowed), makes them
and ends with READY, or NOT READY and the one fix. Per platform: Windows and WSL get inbound firewall rules (one Windows
administrator prompt) and WSL gets mirrored networking in the Windows user's .wslconfig (the old file is kept; WSL then
restarts, which ends this session, so run the same command again); every platform gets the node (a binary already
present, else a cargo build, installing Rust first when cargo is missing), started with its network on, then the join check.
Rerunning changes nothing that is already right. No secret is printed.
"""

from __future__ import annotations

import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from ml_stack import device_host, node_binary, node_join, node_join_check, runtime
from ml_stack.command import Group, flag
from ml_stack.log import say, warn
from ml_stack.node_launch import default_state

SECTION = "[wsl2]"
SETTING = "networkingMode"
WANTED = "mirrored"
BACKUP_SUFFIX = ".ml-stack-backup"
OFFSITE_DIRECTORIES = ("bundle", "dist")


def mirrored_config(text: str) -> tuple[str, bool]:
    """``text`` of a .wslconfig with `networkingMode=mirrored` under [wsl2], and whether that changed it."""
    lines = text.splitlines()
    section = next((i for i, line in enumerate(lines) if line.strip().lower() == SECTION), None)
    if section is None:
        body = text.rstrip("\n")
        return (body + "\n\n" if body else "") + f"{SECTION}\n{SETTING}={WANTED}\n", True
    end = next((i for i in range(section + 1, len(lines)) if lines[i].strip().startswith("[")), len(lines))
    for i in range(section + 1, end):
        key, _, value = lines[i].partition("=")
        if key.strip().lower() == SETTING.lower():
            if value.strip().lower() == WANTED:
                return text, False
            lines[i] = f"{SETTING}={WANTED}"
            return "\n".join(lines) + "\n", True
    lines.insert(section + 1, f"{SETTING}={WANTED}")
    return "\n".join(lines) + "\n", True


def write_config(path: Path, *, now: float | None = None) -> str:
    """Make ``path`` mirrored, keeping the old file beside it; `unchanged`, `created` or the backup's name."""
    old = path.read_text(encoding="utf-8") if path.is_file() else ""
    new, changed = mirrored_config(old)
    if not changed:
        return "unchanged"
    kept = ""
    if path.is_file():
        stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(time.time() if now is None else now))
        kept = path.with_name(f"{path.name}{BACKUP_SUFFIX}-{stamp}")
        shutil.copy2(path, kept)
    path.write_text(new, encoding="utf-8")
    return kept.name if kept else "created"


@dataclass(frozen=True)
class Source:
    """Where the node binary comes from: `runtime`, `found`, `build` (cargo is here) or `rustup` (it is not)."""
    kind: str
    path: str = ""


def node_source(repo: Path, *, selected: Path | None, cargo: str | None) -> Source:
    """The first of: the selected runtime's node, a release binary already in the repository, a cargo build, installing Rust."""
    if selected is not None and node_binary.location(selected).is_file():
        return Source("runtime", str(node_binary.location(selected)))
    for folder in (repo / "app" / "target" / "release", *(repo / name for name in OFFSITE_DIRECTORIES)):
        if (folder / node_binary.NAME).is_file():
            return Source("found", str(folder / node_binary.NAME))
    return Source("build") if cargo else Source("rustup")


@dataclass(frozen=True)
class Needs:
    """What this device lacks: the kind of machine, its WSL networking mode, a .wslconfig edit, firewall rules, a node."""
    kind: str
    mode: str
    config_change: bool
    firewall_missing: bool
    source: Source

    @property
    def restart_wsl(self) -> bool:
        return self.kind == "wsl" and (self.mode == "nat" or self.config_change)


def plan(needs: Needs, *, port: int = node_join.DEFAULT_PORT, beacon: int = node_join.DEFAULT_BEACON_PORT) -> list[str]:
    """Each change the setup will make on this device, one line each: what the single yes is for."""
    steps = []
    if needs.firewall_missing:
        steps.append(f"add Windows inbound allow rules for TCP {port} and UDP {beacon} (one Windows administrator prompt: press Yes)")
    if needs.kind == "wsl" and needs.config_change:
        steps.append("write networkingMode=mirrored to the Windows user's .wslconfig (the old file is kept beside it)")
    if needs.restart_wsl:
        steps.append("restart WSL, which ends this session; run the same command again afterwards")
        return steps
    steps.append({"runtime": "use the node of the selected runtime", "found": f"use the node binary at {needs.source.path}",
                  "build": "build the node with cargo (a few minutes)",
                  "rustup": "install Rust with the standard installer, then build the node"}[needs.source.kind])
    steps.append(f"start the node with its network on (TCP {port}, UDP beacon {beacon}), join policy open, and run the join check")
    return steps


def survey(repo: Path) -> Needs:
    """Look at this machine and say what it lacks."""
    kind = device_host.current_kind()
    windows = kind in ("windows", "wsl")
    mode = node_join_check.wsl_mode(node_join_check.route_address()) if kind == "wsl" else "mirrored"
    config = device_host.wslconfig_path(kind) if kind == "wsl" else None
    changed = bool(config) and mirrored_config(config.read_text(encoding="utf-8") if config.is_file() else "")[1]
    return Needs(kind, mode, changed, windows and device_host.firewall_missing(), node_source(repo, selected=runtime.selection_prefix(), cargo=shutil.which("cargo")))


def confirmed(steps: list[str], *, yes: bool, ask=input) -> bool:
    """Show the steps and take the one yes; `yes` stands for a person who already agreed."""
    say("This will change on this device:\n" + "\n".join(f"  - {s}" for s in steps))
    return yes or (sys.stdin.isatty() and ask("Make these changes? [y/N] ").strip().lower() in ("y", "yes"))


def prepare(needs: Needs) -> str:
    """Do the Windows-side changes; the restart message when WSL had to restart, else ''."""
    if needs.firewall_missing:
        say("A Windows prompt will appear: press Yes.")
        device_host.add_firewall()
        if device_host.firewall_missing():
            raise node_join.PoolError("the firewall rules were not added; run this again and press Yes on the Windows prompt")
    if needs.kind == "wsl" and needs.config_change:
        say(f".wslconfig: {write_config(device_host.wslconfig_path('wsl'))}")
    if needs.restart_wsl:
        say("Restarting WSL; run the same command again afterwards.")
        device_host.restart_wsl()
        return "restarting"
    return ""


def binary_of(needs: Needs, repo: Path) -> Path | None:
    """The node binary for this device, installing Rust and building when it is not already there."""
    if needs.source.kind == "rustup":
        device_host.install_rust(needs.kind)
    if needs.source.kind in ("rustup", "build"):
        return node_binary.build(repo)
    return Path(needs.source.path)


def _setup(a) -> int:
    repo = Path(__file__).resolve().parents[2]
    try:
        needs = survey(repo)
        steps = plan(needs)
        if a.dry_run:
            say("\n".join(f"  - {s}" for s in steps))
            return 0
        if not confirmed(steps, yes=a.yes):
            say("Nothing changed.")
            return 1
        if prepare(needs):
            return 0
        check = node_join_check.Check(state=default_state(), port=node_join.DEFAULT_PORT, beacon_port=node_join.DEFAULT_BEACON_PORT,
                                      wait_s=a.wait, policy=a.policy, binary=binary_of(needs, repo))
        node_join.consent(check.port, check.beacon_port, yes=True)
        ok = node_join_check.walk(check)
        say(node_join_check.summary(check))
        return 0 if ok else 1
    except OSError as exc:
        warn(f"device-setup: NOT READY. {exc}")
        return 1


GROUP = Group("ml_stack.device_setup", "Set this device up to join a pool: the same command on macOS, Linux, Windows and WSL.", run=_setup, options=[
    flag("--yes", action="store_true", help="agree to the listed changes without being asked"),
    flag("--dry-run", action="store_true", help="list the changes and make none"),
    flag("--wait", type=float, default=90.0, help="seconds to wait for another device at each step that needs it"),
    flag("--policy", default="open", choices=node_join.POLICIES, help="open: devices on this network enrol by themselves; secure: a pairing code")])

main = GROUP.run


if __name__ == "__main__":
    sys.exit(main())
