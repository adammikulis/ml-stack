"""Pytest plugin: records, per test file, what a run read, listed, probed, imported, wrote and warned about."""
from __future__ import annotations

import functools
import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(os.environ.get("DEV_TEST_REUSE_ROOT", ".")).resolve()
RECORD = Path(os.environ.get("DEV_TEST_REUSE_RECORD", ""))
REAL_STATE = tuple(Path.home() / part for part in (".ml-stack", "Library/Keychains", ".ssh", ".gnupg",
                                                     ".cache/huggingface", ".config"))
IGNORED = (".git", ".pytest_cache", ".testmondata", "__pycache__", ".ruff_cache", ".mypy_cache")
WRITABLE = tuple({Path(tempfile.gettempdir()), Path(tempfile.gettempdir()).resolve(), Path("/dev")})
ALLOWED_ROOT_WRITES = (".pytest_cache", "__pycache__", ".testmondata", ".coverage")
MUTATING = {"os.mkdir": (0,), "os.rmdir": (0,), "os.remove": (0,), "os.rename": (0, 1), "os.symlink": (1,),
            "os.link": (1,), "os.truncate": (0,), "os.chmod": (0,), "shutil.rmtree": (0,),
            "shutil.copyfile": (1,), "shutil.move": (1,)}
DIR_FDS = {"os.remove": (1,), "os.rmdir": (1,), "os.mkdir": (2,), "os.rename": (2, 3), "shutil.rmtree": (1,)}
SHARED = "*"
FILES: dict[str, dict] = {}
CURRENT: list[str] = []
NODES: set[str] = set()
DIED: list[str] = []
ENABLED = bool(os.environ.get("DEV_TEST_REUSE_RECORD"))
BUSY: list[bool] = []


def _entry(rel: str) -> dict:
    return FILES.setdefault(rel, {"reads": set(), "dirs": set(), "stats": set(), "links": {}, "marks": set(),
                                  "violations": set(), "skipped": False, "collected": 0})


def _under(path: Path, base: Path) -> bool:
    return path == base or base in path.parents


def _text(path: object) -> str:
    return os.fsdecode(os.fspath(path))


def _source_of(path: str) -> str:
    """The checkout-relative path for ``path``, with a compiled file mapped to its source; empty if outside."""
    try:
        target = Path(path)
        if target.suffix == ".pyc" and target.parent.name == "__pycache__":
            target = target.parent.parent / (target.name.split(".")[0] + ".py")
        rel = target.resolve().relative_to(ROOT)
    except (ValueError, OSError):
        return ""
    return "" if any(p in IGNORED for p in rel.parts) else rel.as_posix()


def _writing(mode: object, flags: object) -> bool:
    if isinstance(mode, str) and any(c in mode for c in "wax+"):
        return True
    return isinstance(flags, int) and bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND))


def _write_ok(absolute: str) -> bool:
    """Whether a write may happen: inside the checkout only to its cache directories, elsewhere only to temp."""
    try:
        target = Path(absolute).resolve()
    except OSError:
        return False
    if _under(target, ROOT):
        return any(part.startswith(ALLOWED_ROOT_WRITES) for part in target.relative_to(ROOT).parts)
    return any(_under(target, base) or _under(Path(absolute), base) for base in WRITABLE)


def _from_conftest() -> bool:
    """Whether the open comes from the suite's own isolation guard in ``tests/conftest.py``."""
    guard = str(ROOT / "tests" / "conftest.py")
    frame = sys._getframe(2)
    while frame is not None:
        if frame.f_code.co_filename == guard:
            return True
        frame = frame.f_back
    return False


def _note_path(bucket: dict, absolute: str, kind: str) -> None:
    """Record a path the test read or probed, and the target of a symlink on the way to it."""
    lexical = Path(absolute)
    rel = _source_of(absolute)
    if not rel:
        return
    bucket[kind].add(rel)
    try:
        relative = lexical.relative_to(ROOT)
    except ValueError:
        return
    here = ROOT
    for part in relative.parts:
        here = here / part
        if here.is_symlink():
            bucket["links"][here.relative_to(ROOT).as_posix()] = str(here.readlink())


def _audit(event: str, args: tuple) -> None:
    if not CURRENT or BUSY:
        return
    BUSY.append(True)
    try:
        bucket = _entry(CURRENT[-1])
        if event == "open" and isinstance(args[0], (str, bytes, os.PathLike)):
            absolute = str(Path(_text(args[0])).absolute())
            writing = _writing(args[1], args[2])
            if writing and not _write_ok(absolute):
                bucket["violations"].add(f"wrote {absolute}")
            elif not writing:
                _note_path(bucket, absolute, "reads")
            if any(_under(Path(absolute), base) for base in REAL_STATE) and (writing or not _from_conftest()):
                bucket["violations"].add(f"touched {absolute}")
        elif event in ("os.listdir", "os.scandir") and args and isinstance(args[0], (str, bytes, os.PathLike)):
            rel = _source_of(str(Path(_text(args[0])).absolute()) + "/x")
            if rel:
                bucket["dirs"].add(Path(rel).parent.as_posix())
        elif event in MUTATING:
            fds = DIR_FDS.get(event, ())
            for place, index in enumerate(MUTATING[event]):
                relative = place < len(fds) and len(args) > fds[place] and args[fds[place]] not in (None, -1)
                if not relative and len(args) > index and isinstance(args[index], (str, bytes, os.PathLike)):
                    absolute = str(Path(_text(args[index])).absolute())
                    if not _write_ok(absolute):
                        bucket["violations"].add(f"{event.split('.')[-1]} {absolute}")
    finally:
        BUSY.pop()


def _probe(original):
    @functools.wraps(original)
    def wrapper(path, *args, **kwargs):
        if CURRENT and not BUSY and isinstance(path, (str, bytes, os.PathLike)):
            BUSY.append(True)
            try:
                _note_path(_entry(CURRENT[-1]), str(Path(_text(path)).absolute()), "stats")
            finally:
                BUSY.pop()
        return original(path, *args, **kwargs)
    return wrapper


if ENABLED:
    sys.addaudithook(_audit)
    for _name in ("stat", "lstat", "access"):
        _original = getattr(os, _name)
        _wrapper = _probe(_original)
        for _support in (os.supports_follow_symlinks, os.supports_effective_ids, os.supports_dir_fd,
                         os.supports_fd):
            if _original in _support:
                _support.add(_wrapper)
        setattr(os, _name, _wrapper)


def _loaded() -> set[str]:
    return {getattr(m, "__file__", None) or "" for m in list(sys.modules.values())}


def _enter(rel: str) -> set[str]:
    before = _loaded()
    CURRENT.append(rel)
    return before


def _leave(rel: str, before: set[str]) -> None:
    CURRENT.pop()
    bucket = _entry(rel)
    for path in _loaded() - before:
        source = _source_of(path) if path else ""
        if source:
            bucket["reads"].add(source)


@pytest.hookimpl(hookwrapper=True)
def pytest_make_collect_report(collector):
    path = getattr(collector, "path", None)
    rel = _source_of(str(path)) if ENABLED and isinstance(collector, pytest.Module) and path is not None else ""
    before = _enter(rel) if rel else set()
    try:
        yield
    finally:
        if rel:
            _leave(rel, before)


@pytest.hookimpl(hookwrapper=True)
def pytest_fixture_setup(fixturedef, request):
    shared = ENABLED and fixturedef.scope != "function"
    before = _enter(SHARED) if shared else set()
    try:
        yield
    finally:
        if shared:
            _leave(SHARED, before)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_protocol(item, nextitem):
    rel = _source_of(str(item.path)) if ENABLED else ""
    if rel:
        _entry(rel)["marks"].update(m.name for m in item.iter_markers())
    before = _enter(rel) if rel else set()
    try:
        yield
    finally:
        if rel:
            _leave(rel, before)


def pytest_collection_finish(session):
    if ENABLED:
        for item in session.items:
            rel = _source_of(str(item.path))
            if rel:
                _entry(rel)["collected"] += 1


def pytest_runtest_logreport(report):
    if ENABLED and report.skipped:
        _entry(report.nodeid.split("::")[0])["skipped"] = True


def pytest_warning_recorded(warning_message, when, nodeid, location):
    if ENABLED and "isolation" in str(warning_message.message).lower():
        _entry(nodeid.split("::")[0] or "?")["violations"].add(f"warning: {warning_message.message}")


@pytest.hookimpl(optionalhook=True)
def pytest_testnodeready(node):
    NODES.add(node.gateway.id)


@pytest.hookimpl(optionalhook=True)
def pytest_testnodedown(node, error):
    if error:
        DIED.append(node.gateway.id)


def pytest_sessionfinish(session):
    if not ENABLED:
        return
    RECORD.mkdir(parents=True, exist_ok=True)
    worker = getattr(session.config, "workerinput", {}).get("workerid", "")
    meta = {"worker": worker} if worker else {"nodes": sorted(NODES), "died": DIED}
    files = {rel: {k: sorted(v) if isinstance(v, set) else v for k, v in row.items()} for rel, row in FILES.items()}
    (RECORD / f"{os.getpid()}.json").write_text(json.dumps({"meta": meta, "files": files}), encoding="utf-8")
