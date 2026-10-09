"""Writing the credentials file: atomic, mode 0600, in a directory only its owner enters."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from poolhouse.credentials.reading import CredentialError
from poolhouse.files import writing

ESCAPES = {"\\": "\\\\", '"': '\\"'}


def render(entries: dict[str, str]) -> str:
    """``entries`` as a TOML document of top-level strings, sorted by name."""
    lines = ["# poolhouse credentials: one NAME = \"value\" per line. Keep this file mode 0600."]
    for name in sorted(entries):
        value = "".join(ESCAPES.get(ch, ch) for ch in entries[name])
        lines.append(f'{name} = "{value}"')
    return "\n".join(lines) + "\n"


def write(path: Path, entries: dict[str, str]) -> None:
    """Replace ``path`` with ``entries`` in one step, readable by the owner alone."""
    parent = path.parent
    if not parent.exists():
        parent.mkdir(parents=True, mode=0o700)
    elif sys.platform != "win32" and parent.stat().st_mode & 0o022:
        raise CredentialError(f"{parent} can be written by other users; run: chmod 700 {parent}")
    with writing(path) as draft:
        draft.write_text(render(entries), encoding="utf-8")
        if sys.platform == "win32":
            _owner_only_windows(draft)
        else:
            draft.chmod(0o600)


def _owner_only_windows(path: Path) -> None:
    """Best effort: drop inherited permissions so only the current user keeps access."""
    user = os.environ.get("USERNAME")
    if not user:
        return
    subprocess.run(["icacls", str(path), "/inheritance:r", "/grant:r", f"{user}:F"],
                   capture_output=True, check=False, timeout=15)
