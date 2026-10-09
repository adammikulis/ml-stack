"""``poolhouse migrate``: the one explicit step that moves what the project kept under its old name.

    poolhouse-migrate plan            what would move, and what is running that stops it
    poolhouse-migrate run             move it, once, and write the log

Moves ``~/.ml-stack`` to ``~/.poolhouse``, ``~/.cache/ml_stack`` to ``~/.cache/poolhouse``, the keychain
master key to the new service and a checkout's ``.ml-stack-project.json`` to ``.poolhouse-project.json``.
No import and no other command moves live state. It refuses while a process of the old name or one inside
the old state directory is alive. A directory is renamed where it is; on another volume it is copied,
checked file by file and only then removed. Each step goes to ``migrate.log`` in the new state directory.
"""

from __future__ import annotations

import argparse
import os
import shutil
import time
from collections.abc import Callable
from pathlib import Path

from poolhouse import home, keystore, legacy
from poolhouse.command import Group, flag
from poolhouse.files import CrossDevice, promote, read_json
from poolhouse.log import say, warn
from poolhouse.serve import process

__all__ = ["GROUP", "main", "move_directory", "plan", "run", "running"]

OLD_PROCESSES = ("ml-stack", "ml_stack", "poolside-node")
"""Names of the programs and modules that mean the old build is alive."""


def _named_old(argv: tuple[str, ...]) -> bool:
    words = [Path(argv[0]).name]
    if len(argv) > 1:
        words.append(Path(argv[2] if argv[1] == "-m" and len(argv) > 2 else argv[1]).name)
    return any(word.startswith(OLD_PROCESSES) for word in words)


def running(old_state: Path) -> list[str]:
    """The live processes that a move would pull the ground from: one of the old name, one run from the
    old state directory."""
    found = [f"{pid}: {' '.join(argv)[:90]}" for pid, _started, argv in process.command_lines() if _named_old(argv)]
    found += [f"{pid}: runs inside {old_state}" for pid in process.running_within(old_state)]
    return found


def _tree(root: Path) -> dict[str, int]:
    return {str(p.relative_to(root)): (p.lstat().st_size if p.is_file() and not p.is_symlink() else -1)
            for p in sorted(root.rglob("*"))}


def move_directory(old: Path, new: Path) -> str:
    """Move ``old`` to ``new``: a rename, or on a different volume a copy that matches file for file and
    is then removed. Returns what was done; raises when it could not be done safely."""
    if new.exists() or new.is_symlink():
        return "kept: the new name already exists"
    if not old.is_dir():
        return "nothing to move"
    try:
        promote(old, new)
        return "renamed"
    except CrossDevice:
        pass
    draft = new.with_name(new.name + ".migrating")
    shutil.rmtree(draft, ignore_errors=True)
    shutil.copytree(old, draft, symlinks=True)
    if _tree(old) != _tree(draft):
        shutil.rmtree(draft, ignore_errors=True)
        raise OSError(f"the copy of {old} did not match it; nothing was removed")
    promote(draft, new)
    shutil.rmtree(old)
    return "copied, checked and removed"


def plan(account: Path) -> list[tuple[str, Path | None, Path | None]]:
    """The steps as ``(kind, old, new)``, for the account whose home is ``account``."""
    steps: list[tuple[str, Path | None, Path | None]] = []
    state, cache = account / legacy.STATE_DIR, account / ".cache" / legacy.CACHE_DIR
    if not os.environ.get(home.ROOT_ENV) and state.is_dir():
        steps.append(("state", state, account / home.DEFAULT_NAME))
    if not os.environ.get(home.CACHE_ENV) and cache.is_dir():
        steps.append(("cache", cache, account / ".cache" / home.CACHE_NAME))
    steps.append(("keychain", None, None))
    return steps


def _project_files(state: Path) -> list[Path]:
    """The checkouts the connection record names that still carry a project file of the old name."""
    record = read_json(state / "workspace-connections.json", {})
    roots = record.keys() if isinstance(record, dict) else ()
    return [Path(root) / legacy.PROJECT_FILE for root in roots if (Path(root) / legacy.PROJECT_FILE).is_file()]


def run(account: Path, *, dry: bool = False, ring: Callable[[], keystore.Keystore] = keystore.default) -> int:
    """Do the plan, or only list it. Returns 0 when it is done or there was nothing to do, 1 when a process
    of the old name is alive, 2 when a step failed."""
    steps = plan(account)
    old_state = next((old for kind, old, _new in steps if kind == "state" and old), account / legacy.STATE_DIR)
    alive = running(old_state)
    for line in alive:
        warn(f"still running: {line}")
    if alive and not dry:
        warn("stop the old build first (the node, the land runner, the daemons), then run this again")
        return 1
    log: list[str] = []
    try:
        for kind, old, new in steps:
            if kind == "keychain":
                said = "would move the master key" if dry else ("moved the master key" if ring().carry_old_key()
                                                               else "no master key under the old name")
            elif dry:
                said = f"would move {old} to {new}"
            else:
                said = f"{old} to {new}: {move_directory(old, new)}"
            log.append(f"{kind}: {said}")
            say(f"{kind}: {said}")
        target = next((new for kind, _old, new in steps if kind == "state" and new), home.home())
        for path in [] if dry else _project_files(target):
            promote(path, path.with_name(".poolhouse-project.json"))
            log.append(f"project: {path} renamed")
            say(f"project: {path} renamed")
    except (OSError, keystore.KeystoreError) as exc:
        warn(f"migrate stopped: {exc}")
        log.append(f"stopped: {exc}")
        _write(account, log, dry)
        return 2
    _write(account, log, dry)
    return 0


def _write(account: Path, lines: list[str], dry: bool) -> None:
    where = home.home() if os.environ.get(home.ROOT_ENV) else account / home.DEFAULT_NAME
    if dry or not where.is_dir():
        return
    stamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    with (where / "migrate.log").open("a", encoding="utf-8") as out:
        out.writelines(f"{stamp} {line}\n" for line in lines)


ACCOUNT = flag("--account", default="", help="the account's home directory (default: the current user's)")


def _go(args: argparse.Namespace) -> int:
    return run(Path(args.account) if args.account else home.user_home(), dry=args.cmd == "plan")


GROUP = Group("poolhouse migrate", "Move the state, cache, keychain key and project files of the old name, once.")
GROUP.add("plan", _go, help="what would move, and what is running that stops it", options=[ACCOUNT])
GROUP.add("run", _go, help="move it, once, and write migrate.log", options=[ACCOUNT])

main = GROUP.run


if __name__ == "__main__":
    raise SystemExit(main())
