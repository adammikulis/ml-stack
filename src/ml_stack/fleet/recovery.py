"""The cluster passphrase kept in the OS keystore, and the recovery file that joins a machine without it."""

from __future__ import annotations

import argparse
import base64
import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ml_stack import home, keystore
from ml_stack.files import read_json, write_json
from ml_stack.log import say, warn
from ml_stack.platform import private_file
from ml_stack.sentinel import human

from .discovery import (
    DEFAULT_CLUSTER,
    DiscoveryError,
    Membership,
    _write_memberships,
    clusters_path,
    memberships,
    require_name,
)

PURPOSE = "fleet-passphrase"
HEADER = (
    "# ml-stack cluster recovery file",
    "# Anyone holding this file can run commands on every machine in the cluster.",
    "# `ml-stack-fleet recovery import FILE` joins a machine with it; it does not reveal the passphrase.",
)


def _store() -> keystore.Keystore:
    return keystore.default()


def passphrases_path(path: Path | str | None = None) -> Path:
    """The file holding the keystore-wrapped passphrases, beside the memberships."""
    return clusters_path(path).with_suffix(".passphrases")


def _held(path: Path | str | None) -> dict[str, str]:
    raw = read_json(passphrases_path(path), {})
    return {str(k): str(v) for k, v in raw.items()} if isinstance(raw, dict) else {}


def remember(passphrase: str, group: str = DEFAULT_CLUSTER, path: Path | str | None = None,
             say: Callable[[str], None] = warn) -> None:
    """Store ``passphrase`` for ``group`` under the keystore; ``say`` gets a sentence when it could not."""
    try:
        blob = _store().wrap(PURPOSE, group, passphrase.strip().encode())
        rows = _held(path)
        rows[group] = base64.b64encode(blob).decode()
        target = passphrases_path(path)
        write_json(target, rows)
        private_file(target)
    except (keystore.KeystoreError, OSError) as exc:
        say(f"The passphrase was not saved ({exc}); keep it, or export a recovery file.")


def recall(group: str = "", path: Path | str | None = None) -> str | None:
    """The stored passphrase for ``group`` (default: the first cluster), or None."""
    if not group:
        rows = memberships(path)
        group = rows[0].group if rows else DEFAULT_CLUSTER
    blob = _held(path).get(group)
    if blob is None:
        return None
    return _store().unwrap(PURPOSE, group, base64.b64decode(blob)).decode()


def forget(group: str, path: Path | str | None = None) -> None:
    """Drop the stored passphrase for ``group``."""
    rows = _held(path)
    kept = {g: v for g, v in rows.items() if g != group}
    if kept == rows:
        return
    if kept:
        write_json(passphrases_path(path), kept)
    else:
        passphrases_path(path).unlink(missing_ok=True)


def export_recovery(file: Path | str, group: str = "", path: Path | str | None = None) -> Membership:
    """Write the group, salt and key of a cluster this machine is in to ``file`` (mode 600)."""
    rows = memberships(path)
    held = next((m for m in rows if not group or m.group == group), None)
    if held is None:
        raise DiscoveryError("this machine is in no cluster" if not group else f"not in a cluster called '{group}'")
    body = json.dumps({"group": held.group, "salt": held.salt, "key": held.key.decode()}, indent=1)
    target = home.expand(file)
    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as out:
        out.write("\n".join(HEADER) + "\n" + body + "\n")
    private_file(target)
    return held


def parse_recovery(text: str) -> Membership:
    """Read the cluster name, salt and key from bounded recovery-file text."""
    try:
        if not isinstance(text, str) or len(text.encode("utf-8")) > 8192:
            raise ValueError("recovery file is too large")
        data = json.loads("\n".join(ln for ln in text.splitlines() if not ln.startswith("#")))
        if not isinstance(data, dict) or not isinstance(data.get("key"), str):
            raise ValueError("invalid recovery object")
        key = data["key"]
        salt = data.get("salt", "")
        if not isinstance(salt, str) or len(base64.b64decode(key + "=" * (-len(key) % 4), altchars=b"-_", validate=True)) != 32:
            raise ValueError("invalid recovery key")
        member = Membership(group=require_name(data.get("group")), key=key.encode("ascii"), salt=salt)
        if salt and not 16 <= len(member.salt_bytes()) <= 64:
            raise ValueError("invalid recovery salt")
        return member
    except (ValueError, TypeError, UnicodeError) as exc:
        raise DiscoveryError("not a valid cluster recovery file") from exc


def adopt_recovery(member: Membership, path: Path | str | None = None) -> Membership:
    """Save a parsed membership, replacing only the same named cluster."""
    _write_memberships([member, *[m for m in memberships(path) if m.group != member.group]], path)
    return member


def import_recovery(file: Path | str, path: Path | str | None = None) -> Membership:
    """Join from a recovery file, replacing any membership of the same cluster."""
    with home.expand(file).open() as source:
        member = parse_recovery(source.read(8193))
    return adopt_recovery(member, path)


def add_commands(sub: Any) -> None:
    """Register ``passphrase`` and ``recovery export|import`` on a subparsers object."""
    p = sub.add_parser("passphrase", help="print the stored passphrase (a person at a terminal)")
    p.add_argument("--group", default="", help="which cluster (default: the first)")
    r = sub.add_parser("recovery", help="export this machine's cluster key to FILE, or import FILE to join "
                       "without the passphrase (export needs a person at a terminal)")
    r.add_argument("op", choices=("export", "import"))
    r.add_argument("file")
    r.add_argument("--group", default="", help="export: which cluster (default: the first)")


def run(args: argparse.Namespace) -> int:
    """Run ``passphrase`` or ``recovery``; the exit status."""
    path = args.cluster_key
    try:
        if args.cmd == "passphrase":
            human.require_person("passphrase")
            words = recall(args.group, path)
            if words is None:
                warn("no passphrase is stored for this cluster")
                return 1
            say(words)
        elif args.op == "export":
            human.require_person("recovery export")
            held = export_recovery(args.file, args.group, path)
            say(f"wrote the key for cluster '{held.group}' to {args.file}")
            say("Anyone holding that file can run commands on every machine in the cluster.")
        else:
            held = import_recovery(args.file, path)
            say(f"joined cluster '{held.group}'")
    except (human.HumanRequired, DiscoveryError, keystore.KeystoreError, OSError) as exc:
        warn(f"error: {exc}")
        return 2
    return 0
