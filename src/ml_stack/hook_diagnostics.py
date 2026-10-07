"""Credential-redacted hook failure records independent of workspace services."""

from __future__ import annotations

import argparse
import json
import os
import re
import stat
import sys
import traceback
import uuid
from datetime import UTC, datetime
from pathlib import Path

from ml_stack import __version__
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
    if not allowed or held.st_uid != os.getuid() or held.st_mode & 0o077:
        raise PermissionError("hook diagnostics require an owned private plain path")


def _runtime() -> dict[str, str]:
    marker = Path(__file__).parent / "fleet" / "built-from"
    commit = marker.read_text()[:100].strip() if marker.is_file() else "unstamped"
    return {"version": __version__, "commit": clean(commit), "python": clean(sys.executable),
            "package": clean(Path(__file__).parent)}


def record(error: BaseException, event: str, stage: str) -> str:
    """Write a private bounded failure record and return its user-facing reference."""
    identifier = uuid.uuid4().hex
    summary = reason(error)
    try:
        root = directory()
        root.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        if root.parent.is_symlink():
            raise PermissionError("hook diagnostic parent must be a plain directory")
        root.mkdir(mode=0o700, exist_ok=True)
        _private(root, folder=True)
        frames = [{"file": clean(frame.filename), "function": clean(frame.name), "line": frame.lineno}
                  for frame in traceback.extract_tb(error.__traceback__)[-20:]]
        body = {"id": identifier, "time": datetime.now(UTC).isoformat(), "event": clean(event),
                "stage": clean(stage), "reason": summary, "runtime": _runtime(), "frames": frames}
        data = json.dumps(body, ensure_ascii=False, sort_keys=True).encode()
        if len(data) > MAX_RECORD:
            raise ValueError("hook diagnostic record exceeds its bound")
        descriptor = os.open(root / f"{identifier}.json", os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        records = sorted(root.glob("*.json"), key=lambda item: item.lstat().st_mtime, reverse=True)
        for old in records[MAX_RECORDS:]:
            if IDENTIFIER.fullmatch(old.stem):
                _private(old)
                old.unlink()
    except (OSError, ValueError, TypeError) as failed:
        return f"{summary} [stage={stage}; diagnostic storage unavailable: {reason(failed)}]"
    return f"{summary} [stage={stage}; runtime={body['runtime']['version']}@{body['runtime']['commit']}; diagnostic={identifier}; inspect: python -m ml_stack.hook_diagnostics {identifier}]"


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
            root.glob("*.json"), key=lambda item: item.lstat().st_mtime, reverse=True)[:20]
        for path in paths:
            _private(path)
            if path.stat().st_size > MAX_RECORD:
                raise ValueError("hook diagnostic record exceeds its bound")
            value = json.loads(path.read_text())
            print(redact(json.dumps(value, sort_keys=True, indent=2), env_secrets(os.environ))[0])
        return 0
    except (OSError, ValueError) as error:
        print(f"Hook diagnostics unavailable: {reason(error)}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
