"""Pytest plugin: records, per test file, what a run read, imported, wrote and warned about."""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(os.environ.get("DEV_TEST_REUSE_ROOT", ".")).resolve()
RECORD = Path(os.environ.get("DEV_TEST_REUSE_RECORD", ""))
HOME = Path.home()
REAL_STATE = tuple(str(HOME / part) for part in (".ml-stack", "Library/Keychains", ".ssh", ".gnupg",
                                                   ".cache/huggingface", ".config"))
IGNORED = (".git", ".pytest_cache", ".testmondata", "__pycache__", ".ruff_cache", ".mypy_cache")
WRITABLE = tuple({*(str(Path(p).resolve()) for p in (tempfile.gettempdir(), "/tmp", "/var/folders")),
                  tempfile.gettempdir(), "/tmp", "/var/folders", "/private/tmp", "/private/var/folders", "/dev"})
ALLOWED_ROOT_WRITES = (".pytest_cache", "__pycache__", ".testmondata", ".coverage")
FILES: dict[str, dict] = {}
CURRENT: list[str] = []
ENABLED = bool(os.environ.get("DEV_TEST_REUSE_RECORD"))


def _entry(rel: str) -> dict:
    return FILES.setdefault(rel, {"reads": set(), "dirs": set(), "marks": set(), "violations": set(),
                                  "skipped": False})


def _source_of(path: str) -> str:
    """The checkout-relative path for ``path``, with a compiled file mapped to its source; empty if outside."""
    try:
        target = Path(path)
        if target.suffix == ".pyc" and target.parent.name == "__pycache__":
            target = target.parent.parent / (target.name.split(".")[0] + ".py")
        rel = target.resolve().relative_to(ROOT)
    except (ValueError, OSError):
        return ""
    return "" if rel.parts and rel.parts[0] in IGNORED or any(p in IGNORED for p in rel.parts) else rel.as_posix()


def _writing(mode: object, flags: object) -> bool:
    if isinstance(mode, str) and any(c in mode for c in "wax+"):
        return True
    return isinstance(flags, int) and bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND))


def _write_ok(absolute: str) -> bool:
    """Whether a write may happen: inside the checkout only to its cache directories, elsewhere only to temp."""
    try:
        rel = Path(absolute).resolve().relative_to(ROOT)
    except (ValueError, OSError):
        return absolute.startswith(WRITABLE)
    return any(part.startswith(ALLOWED_ROOT_WRITES) for part in rel.parts)


def _from_conftest() -> bool:
    """Whether the open comes from the suite's own isolation guard in ``tests/conftest.py``."""
    guard = str(ROOT / "tests" / "conftest.py")
    frame = sys._getframe(2)
    while frame is not None:
        if frame.f_code.co_filename == guard:
            return True
        frame = frame.f_back
    return False


def _audit(event: str, args: tuple) -> None:
    if not CURRENT:
        return
    if event == "open" and isinstance(args[0], (str, os.PathLike)):
        path = os.fspath(args[0])
        bucket = _entry(CURRENT[-1])
        absolute = os.path.abspath(path)
        if _writing(args[1], args[2]):
            if not _write_ok(absolute):
                bucket["violations"].add(f"wrote {absolute}")
        else:
            rel = _source_of(absolute)
            if rel:
                bucket["reads"].add(rel)
        if absolute.startswith(REAL_STATE) and (_writing(args[1], args[2]) or not _from_conftest()):
            bucket["violations"].add(f"touched {absolute}")
    elif event in ("os.listdir", "os.scandir") and args and isinstance(args[0], (str, os.PathLike)):
        rel = _source_of(os.path.abspath(os.fspath(args[0])) + "/x")
        if rel:
            _entry(CURRENT[-1])["dirs"].add(str(Path(rel).parent.as_posix()))


if ENABLED:
    sys.addaudithook(_audit)


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


def pytest_runtest_logreport(report):
    if ENABLED and report.skipped:
        _entry(report.nodeid.split("::")[0])["skipped"] = True


def pytest_warning_recorded(warning_message, when, nodeid, location):
    if ENABLED and "isolation" in str(warning_message.message).lower():
        _entry(nodeid.split("::")[0] or "?")["violations"].add(f"warning: {warning_message.message}")


def pytest_sessionfinish(session, exitstatus):
    if not ENABLED:
        return
    RECORD.mkdir(parents=True, exist_ok=True)
    data = {rel: {k: sorted(v) if isinstance(v, set) else v for k, v in row.items()}
            for rel, row in FILES.items()}
    (RECORD / f"{os.getpid()}.json").write_text(json.dumps(data), encoding="utf-8")
