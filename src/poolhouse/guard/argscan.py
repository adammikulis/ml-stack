"""Labels for what a tool call's arguments say: commands, SQL, paths, flags and sizes."""

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping
from typing import Any

from poolhouse.guard import shellcmds, sqlscan
from poolhouse.guard.harm import Finding, outside, protected, resolve
from poolhouse.guard.verbs import verb_finding, words_of

__all__ = ["content", "generic", "walk"]

MAX_NODES = 400
MAX_DEPTH = 5
BULK = 25
SHELL_KEYS = frozenset({"command", "cmd", "script", "shell", "bash", "commandline",
                        "command_line", "cmdline", "shell_command", "bash_command"})
SHELLISH = frozenset({"shell", "bash", "sh", "zsh", "terminal", "exec", "execute", "cmd", "command",
                      "powershell", "run"})
SQL_KEYS = frozenset({"sql", "query", "statement", "stmt", "queries"})
DBISH = frozenset({"db", "sql", "database", "postgres", "postgresql", "mysql", "sqlite", "psql",
                   "execute", "query", "warehouse", "bigquery", "snowflake", "duckdb"})
FORCE_KEYS = frozenset({"force", "hard", "purge", "overwrite", "yes", "assume_yes", "noconfirm",
                        "no_confirm", "skip_confirmation", "skip_confirm", "delete", "prune",
                        "dangerous", "unsafe", "bypass", "force_push", "recursive_delete"})
PATH_KEYS = frozenset({"path", "paths", "file", "files", "filepath", "filename", "file_path", "dir", "directory",
                       "dst", "dest", "destination", "target", "output", "out", "to", "src",
                       "source", "folder", "root", "from"})
DEST_KEYS = frozenset({"path", "file", "filepath", "filename", "file_path", "dst", "dest", "destination",
                       "target", "output", "out", "to"})
BODY_KEYS = frozenset({"content", "contents", "text", "data", "body", "new_text", "file_text"})
BULK_KEYS = frozenset({"paths", "files", "ids", "items", "targets", "names", "urls", "resources",
                       "rows", "records"})
ACTION_KEYS = frozenset({"action", "operation", "op", "verb", "subcommand", "method"})
WRITES = frozenset({"write", "save", "overwrite", "put", "dump", "export", "replace", "truncate"})
MOVES = frozenset({"move", "mv", "copy", "cp", "rename", "install", "extract", "download"})
PROD = re.compile(r"\bprod(?:uction)?\b", re.I)
PROD_KEYS = frozenset({"env", "environment", "stage", "namespace", "cluster", "workspace", "profile",
                       "target", "database", "db", "host", "url", "branch", "ref", "remote",
                       "project", "account", "region", "tier", "deployment"})
PROGRAM_KEYS = frozenset({"command", "cmd", "program", "executable", "binary", "exe", "tool"})
ARG_KEYS = frozenset({"args", "arguments", "argv", "params", "parameters", "flags", "options"})
META = re.compile(r"[;|&`$<>\n]")
QUIET_KEYS = frozenset({"cwd", "workdir", "working_dir", "working_directory", "timeout", "shell",
                        "description"})
SENDING = frozenset({"post", "put", "delete", "patch"})
D, R, S, U = "destructive", "reversible", "safe", "unsure"


def walk(value: Any, key: str = "", depth: int = 0, budget: list[int] | None = None
         ) -> Iterator[tuple[str, Any]]:
    """The ``(key, value)`` leaves of ``value`` (a list's items keep their list's key), at
    most ``MAX_NODES`` of them and ``MAX_DEPTH`` deep; the keys are lower-case."""
    left = budget if budget is not None else [MAX_NODES]
    if left[0] <= 0 or depth > MAX_DEPTH:
        return
    left[0] -= 1
    if isinstance(value, Mapping):
        for k in sorted(value, key=str):
            yield from walk(value[k], str(k).lower(), depth + 1, left)
    elif isinstance(value, (list, tuple)):
        if all(isinstance(v, str) for v in value) and value:
            yield key, list(value)
        else:
            for item in value:
                yield from walk(item, key, depth + 1, left)
    else:
        yield key, value


def _shell(value: Any, roots: tuple[str, ...]) -> list[Finding]:
    if isinstance(value, list):
        return shellcmds.analyse([str(v) for v in value], shellcmds.Ctx(roots))
    return shellcmds.scan(str(value), roots)


def _programs(args: Any, roots: tuple[str, ...], depth: int = 0) -> list[Finding]:
    """Findings for a command given as one key and its arguments as another."""
    found: list[Finding] = []
    if depth > MAX_DEPTH or not isinstance(args, Mapping):
        return found
    low = {str(k).lower(): v for k, v in args.items()}
    prog = next((low[k] for k in sorted(PROGRAM_KEYS) if isinstance(low.get(k), str)), None)
    rest = next((low[k] for k in sorted(ARG_KEYS) if isinstance(low.get(k), (list, str))), None)
    if prog is not None and rest is not None:
        if isinstance(rest, list):
            found += shellcmds.analyse([prog, *[str(x) for x in rest]], shellcmds.Ctx(roots))
        else:
            found += shellcmds.scan(f"{prog} {rest}", roots)
    for value in low.values():
        found += _programs(value, roots, depth + 1)
    return found


def content(name: str, args: Mapping[str, Any], roots: tuple[str, ...]) -> tuple[list[Finding], bool]:
    """Findings from the commands and SQL in ``args``, and whether any were read (so that the
    tool's name no longer decides what it does)."""
    words = set(words_of(name))
    found: list[Finding] = _programs(args, roots)
    read = bool(found)
    for key, value in walk(args):
        if (key in SHELL_KEYS and isinstance(value, (str, list))) or (key == "argv" and isinstance(value, list)):
            found += _shell(value, roots)
            read = True
        elif key in SQL_KEYS and isinstance(value, str) and (words & DBISH or sqlscan.looks_like_sql(value)):
            found += sqlscan.scan(value)
            read = True
        elif words & SHELLISH and isinstance(value, str) and key not in PATH_KEYS | BODY_KEYS | QUIET_KEYS \
                and (not read or META.search(value)):
            found += shellcmds.scan(value, roots)
            read = True
    return found, read


def _exists(path: str, roots: tuple[str, ...]) -> bool:
    full = resolve(path, roots)
    return full.exists() or full.is_symlink()


def _paths(args: Mapping[str, Any]) -> list[tuple[str, str]]:
    found = []
    for key, value in walk(args):
        if key in PATH_KEYS:
            found += [(key, v) for v in (value if isinstance(value, list) else [value])
                      if isinstance(v, str) and v.strip() and "\n" not in v and len(v) < 1024]
    return found


def _numbers(args: Mapping[str, Any], words: set[str]) -> list[Finding]:
    if not words & {"download", "fetch", "pull", "get", "clone", "sync"}:
        return []
    found = []
    for key, value in walk(args):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        scale = 2 ** 30 if key.endswith("gb") else 2 ** 20 if key.endswith("mb") else \
            1 if key in ("size", "bytes", "max_bytes", "size_bytes", "content_length") else 0
        if scale and value * scale > 5 * 2 ** 30:
            found.append(Finding(D, "downloads more than 5 GiB"))
    return found


def generic(name: str, args: Mapping[str, Any], roots: tuple[str, ...], label: str
            ) -> list[Finding]:
    """Findings that hold whatever the tool is: forced or unconfirmed flags, destructive
    actions, writes over files or outside the project, bulk lists, production and size. A tool
    whose name labels it ``safe`` is held only to the checks that cannot be a read."""
    words = set(words_of(name))
    found: list[Finding] = []
    acts = label != S
    for key, value in walk(args):
        if acts and key in FORCE_KEYS and value is True:
            found.append(Finding(D, f"{key} is set, which forces the action or skips confirmation"))
        if key in ACTION_KEYS and isinstance(value, str):
            if value.lower() in SENDING and key == "method" and acts:
                found.append(Finding(D, f"sends or changes data on a remote server ({value.upper()})"))
            hit = verb_finding(words_of(value), "the action")
            if hit is not None and hit.label == D:
                found.append(Finding(D, f"the action is {value}"))
        if acts and key in BULK_KEYS and isinstance(value, list) and len(value) > BULK:
            found.append(Finding(D, f"touches {len(value)} items at once"))
        if acts and key in PROD_KEYS and isinstance(value, str) and len(value) < 512 \
                and PROD.search(value):
            found.append(Finding(D, "touches production"))
    if acts:
        found += _path_findings(args, words, roots)
    return [*found, *_numbers(args, words)]


def _path_findings(args: Mapping[str, Any], words: set[str], roots: tuple[str, ...]) -> list[Finding]:
    found: list[Finding] = []
    writes = bool(words & (WRITES | MOVES | {"delete", "remove", "rm", "edit", "update", "create"}))
    paths = _paths(args)
    for key, path in paths:
        if protected(path) and outside(path, roots):
            found.append(Finding(D, f"names a protected path ({path[:60]})"))
        elif outside(path, roots) and writes and key in DEST_KEYS:
            found.append(Finding(D if words & (WRITES | MOVES) else R,
                                 f"writes outside the project ({path[:60]})"))
    dest = [p for k, p in paths if k in DEST_KEYS]
    appends = any(k == "mode" and str(v).startswith(("a", "append")) for k, v in walk(args)) \
        or any(k in ("append", "append_to") and v is True for k, v in walk(args))
    if words & (WRITES | MOVES) and not appends:
        for path in dest:
            if _exists(path, roots):
                found.append(Finding(D, f"overwrites an existing file ({path[:60]})"))
    if words & WRITES and not appends:
        for key, value in walk(args):
            if key in BODY_KEYS and isinstance(value, str) and dest and len(value.strip()) <= 12:
                found.append(Finding(D, "writes almost nothing over a file, which empties it"))
    if words & {"checkout", "restore"} and any(k in ("path", "paths", "file", "files") for k, _ in paths):
        found.append(Finding(D, "discards the changes to the named files"))
    return found
