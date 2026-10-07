"""Credential-redacted hook failure records independent of workspace services."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
import time
import traceback
import uuid
from datetime import UTC, datetime
from pathlib import Path

from ml_stack import __version__, lock
from ml_stack.files import writing
from ml_stack.redact.secrets import env_secrets, redact

MAX_RECORD = 32768
MAX_RECORDS = 200
IDENTIFIER = re.compile(r"[0-9a-f]{32}\Z")


def clean(value: object) -> str:
    """Return a bounded, credential-redacted single-line value."""
    text, _ = redact(str(value), env_secrets(os.environ))
    return " ".join(text.replace(str(Path.home()), "~").split())[:1000]


def reason(error: BaseException) -> str:
    """Return the exception category and redacted reason."""
    return f"{type(error).__name__}: {clean(error)}"


def directory() -> Path:
    """Return the configured local diagnostic directory."""
    return Path(os.environ.get("ML_STACK_HOME", str(Path.home() / ".ml-stack"))).expanduser() / "hook-diagnostics"


def _private(path: Path, *, folder: bool = False) -> None:
    held = path.lstat()
    allowed = stat.S_ISDIR(held.st_mode) if folder else stat.S_ISREG(held.st_mode)
    if sys.platform == "win32":
        from ml_stack.windows_private import problem, validate
        validate(path)
        if not allowed or problem(path):
            raise PermissionError("hook diagnostics require an owned private Windows path")
    elif not allowed or held.st_uid != os.getuid() or held.st_mode & 0o077:
        raise PermissionError("hook diagnostics require an owned private plain path")


def _ancestors(path: Path) -> None:
    if sys.platform == "win32":
        from ml_stack.windows_private import validate
        validate(path)
        return
    for ancestor in (path, *path.parents):
        try:
            held = ancestor.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(held.st_mode) or held.st_uid not in (0, os.getuid()):
            raise PermissionError("hook diagnostic ancestry is not owned plain storage")


def _open(root: Path, name: str, flags: int) -> int:
    _ancestors(root)
    _private(root, folder=True)
    if sys.platform == "win32":
        path = root / name
        from ml_stack.windows_private import restrict, validate
        validate(path)
        descriptor = os.open(path, flags, 0o600)
        try:
            if flags & os.O_CREAT:
                restrict(path)
            _private(path)
            held, current = os.fstat(descriptor), path.lstat()
            if (held.st_dev, held.st_ino) != (current.st_dev, current.st_ino):
                raise PermissionError("hook diagnostic path changed while opening")
            _ancestors(path)
        except (OSError, ValueError):
            os.close(descriptor)
            raise
        return descriptor
    parent = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        descriptor = os.open(name, flags | os.O_NOFOLLOW, 0o600, dir_fd=parent)
        held = os.fstat(descriptor)
        if not stat.S_ISREG(held.st_mode) or held.st_uid != os.getuid() or held.st_mode & 0o077:
            os.close(descriptor)
            raise PermissionError("hook diagnostic handle is not owned private plain storage")
        return descriptor
    finally:
        os.close(parent)


def _read(path: Path) -> dict:
    descriptor = _open(path.parent, path.name, os.O_RDONLY)
    with os.fdopen(descriptor, "rb") as stream:
        data = stream.read(MAX_RECORD + 1)
    if len(data) > MAX_RECORD:
        raise ValueError("hook diagnostic record exceeds its bound")
    value = json.loads(data)
    if not isinstance(value, dict):
        raise ValueError("hook diagnostic record must be an object")
    return value


def _write(path: Path, value: dict) -> None:
    data = json.dumps(value, ensure_ascii=False, sort_keys=True).encode()
    if len(data) > MAX_RECORD:
        raise ValueError("hook diagnostic record exceeds its bound")
    with writing(path) as temporary:
        if sys.platform == "win32":
            from ml_stack.windows_private import restrict
            restrict(temporary)
        descriptor = _open(path.parent, temporary.name, os.O_WRONLY)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())


def _occurrence(root: Path, body: dict) -> dict:
    signature = hashlib.sha256(f"{body['category']}:{body['reason']}".encode()).hexdigest()[:32]
    path = root / f"{signature}.signature.json"
    descriptor = _open(root, "occurrences.lock", os.O_RDWR | os.O_CREAT)
    started = time.monotonic()
    acquired = False
    try:
        while not lock.take(descriptor):
            if time.monotonic() - started > 0.15:
                raise TimeoutError("hook diagnostic occurrence index is busy")
            time.sleep(0.005)
        acquired = True
        prior = _read(path) if path.exists() else {}
        if not prior and len(list(root.glob("*.signature.json"))) >= 2000:
            raise ValueError("hook diagnostic signature capacity reached")
        value = {"signature": signature, "category": body["category"], "count": prior.get("count", 0) + 1,
                 "first_seen": prior.get("first_seen", body["time"]), "last_seen": body["time"],
                 "first_id": prior.get("first_id", body["id"]), "last_id": body["id"],
                 "first_checkout": prior.get("first_checkout", body["trace"].get("checkout", {})),
                 "first_runtime": prior.get("first_runtime", body["runtime"])}
        _write(path, value)
        return value
    finally:
        if acquired:
            lock.release(descriptor)
        os.close(descriptor)


def _runtime() -> dict[str, str]:
    marker = Path(__file__).parent / "fleet" / "built-from"
    commit = marker.read_text()[:100].strip() if marker.is_file() else "unstamped"
    return {"version": __version__, "commit": clean(commit), "python": clean(sys.executable),
            "package": clean(Path(__file__).parent)}


def record(error: BaseException, event: str, stage: str, *, metadata: dict | None = None) -> str:
    """Write a private bounded failure record and return its user-facing reference."""
    identifier = uuid.uuid4().hex
    summary = reason(error)
    try:
        root = directory()
        _ancestors(root)
        root.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        if root.parent.is_symlink():
            raise PermissionError("hook diagnostic parent must be a plain directory")
        root.mkdir(mode=0o700, exist_ok=True)
        if sys.platform == "win32":
            from ml_stack.windows_private import restrict
            restrict(root)
        _private(root, folder=True)
        frames = [{"file": clean(frame.filename), "function": clean(frame.name), "line": frame.lineno}
                  for frame in traceback.extract_tb(error.__traceback__)[-20:]]
        body = {"id": identifier, "time": datetime.now(UTC).isoformat(), "event": clean(event),
                "stage": clean(stage), "category": clean(f"{event}.{stage}.{type(error).__name__}"),
                "reason": summary, "runtime": _runtime(), "frames": frames,
                "trace": json.loads(redact(json.dumps(metadata or {}).replace(str(Path.home()), "~"), env_secrets(os.environ))[0])}
        body["occurrence"] = _occurrence(root, body)
        _write(root / f"{identifier}.json", body)
        records = sorted((item for item in root.glob("*.json") if IDENTIFIER.fullmatch(item.stem)),
                         key=lambda item: item.lstat().st_mtime, reverse=True)
        for old in records[MAX_RECORDS:]:
            if IDENTIFIER.fullmatch(old.stem):
                _private(old)
                old.unlink()
    except (OSError, ValueError, TypeError, AttributeError, ImportError, RuntimeError) as failed:
        return f"{summary} [stage={stage}; diagnostic storage unavailable: {reason(failed)}]"
    return f"[diagnostic={identifier}; stage={stage}; runtime={body['runtime']['version']}@{body['runtime']['commit']}] {summary}; inspect: python -m ml_stack.hook_diagnostics {identifier}"


def main(argv: list[str] | None = None) -> int:
    """Print recent local failure records or one diagnostic ID."""
    parser = argparse.ArgumentParser(description="Inspect credential-redacted local hook failure records.")
    parser.add_argument("id", nargs="?", help="diagnostic ID; omitted lists the latest 20 records")
    args = parser.parse_args(argv)
    try:
        root = directory()
        if not root.exists():
            print("No local hook failure records.")
            return 0
        _private(root, folder=True)
        if args.id and not IDENTIFIER.fullmatch(args.id):
            raise ValueError("diagnostic ID must contain 32 hexadecimal characters")
        paths = [root / f"{args.id}.json"] if args.id else sorted(
            root.glob("*.signature.json"), key=lambda item: item.lstat().st_mtime, reverse=True)[:20]
        for path in paths:
            value = _read(path)
            if args.id and value.get("occurrence", {}).get("signature"):
                signature = value["occurrence"]["signature"]
                if not IDENTIFIER.fullmatch(signature):
                    raise ValueError("invalid hook diagnostic signature")
                value["occurrence"] = _read(root / f"{signature}.signature.json")
            print(redact(json.dumps(value, sort_keys=True, indent=2), env_secrets(os.environ))[0])
        return 0
    except (OSError, ValueError, TypeError, AttributeError, ImportError, RuntimeError) as error:
        print(f"Hook diagnostics unavailable: {reason(error)}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
