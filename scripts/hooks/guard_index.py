"""The edit guard's function index: one small shard per source file, rewritten only for a file whose
modification time or size moved."""

from __future__ import annotations

import ast
import contextlib
import hashlib
import json
import os
from pathlib import Path


def cache_dir(root: Path) -> Path:
    """The directory holding one index shard per source file of this tree."""
    home = Path(os.environ.get("MLSTACK_GUARD_CACHE") or "~/.ml-stack/guard").expanduser()
    return home / hashlib.sha256(str(root).encode()).hexdigest()[:16]


def params(fn) -> int:
    """How many parameters a function takes."""
    args = fn.args
    return (len(args.posonlyargs) + len(args.args) + len(args.kwonlyargs)
            + bool(args.vararg) + bool(args.kwarg))


def shard_for(path: Path, where: str, dup) -> tuple[list, list]:
    """`(bodies, defs)` of one file: `[sig, path, line, name, statements]` and `[name, where, line, params]` rows."""
    text = path.read_text(encoding="utf-8")
    bodies = [[sig, entry.path, entry.line, entry.name, entry.statements]
              for entry, sig in dup.functions_in(text, where)]
    defs: list = []
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return bodies, defs
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            defs.append([node.name, where, node.lineno, params(node)])
    return bodies, defs


def prune(home: Path, keep: Path) -> None:
    """Delete the single-file indexes of older versions and the shard directories left empty."""
    for old in home.glob("*.json"):
        old.unlink(missing_ok=True)
    for other in home.iterdir():
        if other.is_dir() and other != keep and not any(other.glob("*.json")):
            other.rmdir()


def load_shard(path: Path, where: str, shard: Path, dup) -> tuple[dict, bool]:
    """The shard of one file and whether it was just rebuilt."""
    stat = path.stat()
    stamp = [stat.st_mtime_ns, stat.st_size]
    try:
        held = json.loads(shard.read_text(encoding="utf-8"))
        if held["stamp"] == stamp and held["where"] == where:
            return held, False
    except (OSError, ValueError, KeyError, TypeError):
        pass
    rows, names = shard_for(path, where, dup)
    held = {"stamp": stamp, "where": where, "bodies": rows, "defs": names}
    try:
        shard.parent.mkdir(parents=True, exist_ok=True)
        shard.write_text(json.dumps(held), encoding="utf-8")
    except OSError:
        pass
    return held, True


def index_for(root: Path, dup) -> dict:
    """`{"bodies": {signature: [entry...]}, "defs": {name: [where...]}}` for the tree."""
    folder = cache_dir(root)
    bodies: dict[str, list] = {}
    defs: dict[str, list] = {}
    rebuilt = False
    for path in dup.python_files(root):
        where = str(path.relative_to(root))
        shard = folder / (hashlib.sha256(where.encode()).hexdigest()[:20] + ".json")
        try:
            held, fresh = load_shard(path, where, shard, dup)
        except (OSError, UnicodeDecodeError):
            continue
        rebuilt = rebuilt or fresh
        for sig, *entry in held["bodies"]:
            bodies.setdefault(sig, []).append(entry)
        for name, *place in held["defs"]:
            defs.setdefault(name, []).append(place)
    if rebuilt and folder.parent.exists():
        with contextlib.suppress(OSError):
            prune(folder.parent, folder)
    return {"bodies": bodies, "defs": defs}
