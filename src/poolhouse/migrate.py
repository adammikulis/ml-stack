"""``poolhouse migrate``: the one explicit step that moves what the project kept under its old name.

    poolhouse-migrate plan            what would move, and what is running that stops it
    poolhouse-migrate run             move it, once, and write migrate.log
    poolhouse-migrate verify          list any dangling symlink or old-path venv file left under the new dirs

Moves ``~/.ml-stack``, ``~/.cache/ml_stack``, the keychain master key and a checkout's old project file to the
new names; no import and no other command does. It refuses while a process of the old name is alive and when
both names of a directory exist. On another volume a directory is copied, checked (content, links, modes) and
only then removed; a cut-off run is finished by the next. The desktop app's identifier is now
``app.poolhouse.app``: its data and Keychain access list start empty and its sign-ins are repeated once.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import stat
import time
from collections.abc import Callable
from pathlib import Path

from poolhouse import home, keystore, legacy, migrate_paths
from poolhouse.command import Group, flag
from poolhouse.files import CrossDevice, promote, read_json, sha256_file
from poolhouse.log import say, warn
from poolhouse.serve import process

__all__ = ["GROUP", "BothExist", "main", "move_directory", "plan", "run", "running"]

OLD_PROCESSES = ("ml-stack", "ml_stack", ".ml-stack", "poolside-node")
"""Names of the programs, modules and directories that mean the old build is alive."""
MARKER = ".poolhouse-migrated"
"""Written into an old directory once its copy is verified; names where the copy went."""
INTERPRETER = re.compile(r"python[\d.]*(\.exe)?$")
UV_TAKES_A_VALUE = frozenset({"--with", "--python", "-p", "--project", "--directory", "--extra", "--group",
                              "--env-file", "--with-requirements", "-w", "--index"})


class BothExist(OSError):
    """A directory exists under both names, so neither can be taken for the other."""


def _old(word: str) -> bool:
    return word.startswith(OLD_PROCESSES)


def _python_runs_old(args: list[str]) -> bool:
    """Whether the interpreter arguments run an old module (-m) or an old script, flags skipped."""
    it = iter(args)
    for arg in it:
        if arg == "-m":
            return _old(next(it, ""))
        if arg.startswith("-m") and len(arg) > 2:
            return _old(arg[2:])
        if arg == "-c":
            return False
        if arg in ("-X", "-W"):
            next(it, None)
        elif not arg.startswith("-"):
            return _old(Path(arg).name)
    return False


def _named_old(argv: tuple[str, ...]) -> bool:
    """Whether a command line is the old build: its program or the directories it sits in carry the old
    name, or an interpreter (or ``uv run``) is told to run an old module or script. A plain argument that
    only starts with the old name, such as a file an editor has open, is not."""
    if not argv:
        return False
    if any(_old(part) for part in Path(argv[0]).parts):
        return True
    program = Path(argv[0]).name
    if INTERPRETER.match(program):
        return _python_runs_old(list(argv[1:]))
    if program in ("uv", "uvx"):
        rest = list(argv[1:])
        if rest[:1] == ["run"]:
            rest = rest[1:]
        while rest and rest[0].startswith("-"):
            rest = rest[2:] if rest[0] in UV_TAKES_A_VALUE else rest[1:]
        return _named_old(tuple(rest))
    return False


def running(old_state: Path) -> list[str]:
    """The live processes that a move would pull the ground from: one of the old name, one run from the
    old state directory."""
    found = [f"{pid}: {' '.join(argv)[:90]}" for pid, _started, argv in process.command_lines() if _named_old(argv)]
    found += [f"{pid}: runs inside {old_state}" for pid in process.running_within(old_state)]
    return found


def _tree(root: Path) -> dict[str, tuple]:
    """Every entry under ``root`` with what must survive a copy: kind, mode, and the content hash or link target."""
    out: dict[str, tuple] = {}
    for path in sorted(root.rglob("*")):
        info = path.lstat()
        key = str(path.relative_to(root))
        if stat.S_ISLNK(info.st_mode):
            out[key] = ("link", str(path.readlink()))
        elif stat.S_ISDIR(info.st_mode):
            out[key] = ("dir", stat.S_IMODE(info.st_mode))
        else:
            out[key] = ("file", stat.S_IMODE(info.st_mode), sha256_file(path))
    return out


def move_directory(old: Path, new: Path) -> str:
    """Move ``old`` to ``new``: a rename, or on a different volume a verified copy and then the removal.
    Returns what was done; raises `BothExist` when both exist (unless ``old`` is a verified copy that a cut-off
    run left, which is removed) and other `OSError` when it could not be done safely."""
    if new.exists() or new.is_symlink():
        if old.is_dir() and (old / MARKER).is_file() and (old / MARKER).read_text(encoding="utf-8") == str(new):
            shutil.rmtree(old)
            return "finished a cut-off move: removed the verified old copy"
        if old.is_dir():
            raise BothExist(f"{old} and {new} both exist. Merge what you need from {old} into {new} and remove "
                            f"{old}, or remove {new} if it holds nothing you want, then run this again")
        return "nothing to move"
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
    (old / MARKER).write_text(str(new), encoding="utf-8")
    promote(draft, new)
    shutil.rmtree(old)
    return "copied, checked and removed"


def _new_state(account: Path) -> Path:
    named = os.environ.get(home.ROOT_ENV)
    return home.expand(named) if named else account / home.DEFAULT_NAME


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


KEY_OUTCOMES = {
    "none": ("no master key under the old name", 0),
    "present": ("the master key is already under the new name", 0),
    "moved": ("moved the master key", 0),
    "differs": ("the old name and the new name hold different master keys, so everything wrapped under the old "
                "one stays locked; neither was changed. Restore the one you mean to keep, delete the other, "
                "and run this again", 2),
    "mismatch": ("the new keychain item did not read back as written; it was deleted and the old key was "
                 "left in place", 2),
}


def _state_step(old: Path, new: Path, dry: bool) -> tuple[str, int]:
    try:
        if dry:
            if new.exists() and old.is_dir() and not (old / MARKER).is_file():
                raise BothExist(f"{old} and {new} both exist")
            return f"would move {old} to {new}", 0
        return f"{old} to {new}: {move_directory(old, new)}", 0
    except BothExist as exc:
        return str(exc), 3


def _steps(account: Path, steps: list, dry: bool, ring: Callable[[], keystore.Keystore],
           log: list[str]) -> int:
    """Run the steps; returns the worst exit code, and stops at an OSError or a keystore error."""
    code = 0
    for kind, old, new in steps:
        if kind == "keychain":
            outcome = "would carry the master key" if dry else ring().carry_old_key()
            said, bad = ("would move the master key", 0) if dry else KEY_OUTCOMES[outcome]
        else:
            said, bad = _state_step(old, new, dry)
        log.append(f"{kind}: {said}")
        (warn if bad else say)(f"{kind}: {said}")
        code = max(code, bad)
    target = _new_state(account)
    for path in [] if dry else _project_files(target):
        new = path.with_name(".poolhouse-project.json")
        if new.exists():
            log.append(f"project: {new} exists; {path} was left")
            warn(f"project: {new} exists, so {path} was left; remove the one you do not want")
            code = max(code, 2)
            continue
        promote(path, new)
        log.append(f"project: {path} renamed")
        say(f"project: {path} renamed")
    return code


def _new_cache(account: Path) -> Path:
    named = os.environ.get(home.CACHE_ENV)
    return home.expand(named) if named else account / ".cache" / home.CACHE_NAME


def _pairs(account: Path) -> list[tuple[Path, Path]]:
    """``(old, new)`` for the state directory and the cache directory."""
    return [(account / legacy.STATE_DIR, _new_state(account)),
            (account / ".cache" / legacy.CACHE_DIR, _new_cache(account))]


def _path_roots(account: Path, dry: bool) -> list[Path]:
    """Where paths are looked for: the new directories, or before the move the old ones that will become them."""
    return [(old if dry and old.is_dir() else new) for old, new in _pairs(account) if old.is_dir() or new.is_dir()]


def _paths_step(account: Path, dry: bool, log: list[str]) -> None:
    """Repoint what still names the old directories under the new ones; list what it cannot repoint."""
    pairs, roots = _pairs(account), _path_roots(account, dry)
    found = migrate_paths.scan(roots, pairs, planning=dry)
    plists = sorted((account / "Library" / "LaunchAgents").glob("com.ml-stack.*"))
    found.report += plists
    said = f"paths: {found.counts()}"
    log.append(said)
    say(said)
    for path in found.report[:20]:
        say(f"paths: names an old path, not changed: {path}")
    if len(found.report) > 20:
        say(f"paths: ... and {len(found.report) - 20} more files (migrate.log lists them all)")
    if not dry:
        log += migrate_paths.apply(found, pairs, _new_state(account) / migrate_paths.BACKUPS)


def verify(account: Path) -> int:
    """List what is still wrong under the new directories; 0 when nothing is, 1 when something is."""
    bad = migrate_paths.verify(_path_roots(account, False), _pairs(account))
    for line in bad:
        warn(f"verify: {line}")
    say("verify: the new directories hold no old path" if not bad else f"verify: {len(bad)} problems")
    return 1 if bad else 0


def run(account: Path, *, dry: bool = False, ring: Callable[[], keystore.Keystore] = keystore.default) -> int:
    """Do the plan, or only list it. Returns 0 when it is done or there was nothing to do, 1 when a process
    of the old name is alive, 2 when a step failed, 3 when both names of a directory exist."""
    steps = plan(account)
    old_state = next((old for kind, old, _new in steps if kind == "state" and old), account / legacy.STATE_DIR)
    alive = running(old_state)
    for line in alive:
        warn(f"still running: {line}")
    if alive and not dry:
        warn("stop the old build first (the node, the land runner, the daemons), then run this again")
        return 1
    log: list[str] = []
    code = 2
    try:
        code = _steps(account, steps, dry, ring, log)
        if code == 0:
            _paths_step(account, dry, log)
    except (OSError, keystore.KeystoreError) as exc:
        warn(f"migrate stopped: {exc}")
        log.append(f"stopped: {exc}")
    finally:
        _write(account, log, dry)
    return code


def _write(account: Path, lines: list[str], dry: bool) -> None:
    """Append to ``migrate.log`` in the new state directory, or in the old one when the move has not happened."""
    new, old = _new_state(account), account / legacy.STATE_DIR
    where = new if new.is_dir() else old if old.is_dir() else None
    if dry or where is None or not lines:
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
GROUP.add("verify", lambda args: verify(Path(args.account) if args.account else home.user_home()),
          help="list any dangling symlink or old-path venv file left under the new directories",
          options=[ACCOUNT])

main = GROUP.run


if __name__ == "__main__":
    raise SystemExit(main())
