"""Proof, to the desktop app that opens the window, that this process is its own daemon.

Anything can listen on a port and answer ``/health`` with the right JSON. The daemon keeps a
private secret beside its settings, readable by its user alone, and answers a challenge with a
keyed hash of it, so the app that can read the same file knows who answered, and a process that
cannot read it cannot.
"""

import hashlib
import hmac
import re
import secrets
from pathlib import Path

from poolhouse.files import write_text
from poolhouse.lock import only_one
from poolhouse.platform import private_file

PROOF_KEY_FILE = "app-identity"
HEADER = "X-Poolhouse-Challenge"
CHALLENGE = re.compile(r"[0-9a-f]{32,128}")
SECRET_BYTES = 32


def secret(root: Path | str) -> bytes:
    """The root's identity secret, made on first use; owner-only, and never a link."""
    path = Path(root) / PROOF_KEY_FILE
    if path.is_symlink():
        raise ValueError("the app identity file cannot be a symbolic link")
    with only_one(Path(root) / "app-identity.lock", announce=lambda _: None):
        if not path.is_file() or len(path.read_text().strip()) != SECRET_BYTES * 2:
            write_text(path, secrets.token_hex(SECRET_BYTES))
        private_file(path)
    return bytes.fromhex(path.read_text().strip())


def prove(root: Path | str, challenge: str) -> str:
    """HMAC-SHA256 of ``challenge`` under the root's secret, as hex; empty for a malformed challenge."""
    if not CHALLENGE.fullmatch(challenge):
        return ""
    return hmac.new(secret(root), challenge.encode(), hashlib.sha256).hexdigest()
