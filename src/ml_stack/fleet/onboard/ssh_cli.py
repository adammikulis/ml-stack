"""``ml-stack fleet bootstrap --ssh [user@]host``: see `ssh.py`."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from ml_stack import home
from ml_stack.log import say, warn

from ..tailnet import detect
from .human import HumanRequired
from .requests import Devices
from .routes import ssh_address
from .signing import KeyStoreError, SigningKeys
from .signing_cli import confirm_signing
from .ssh import (
    NAMESPACE,
    Signed,
    SshRefused,
    bootstrap_over_ssh,
    parse_target,
    plan,
    program_entries,
)

__all__ = ["add_ssh", "cmd_ssh"]


def add_ssh(p: argparse.ArgumentParser) -> None:
    p.add_argument("--ssh-port", type=int, default=22)
    p.add_argument("--dry-run", action="store_true",
                   help="with --ssh: print the commands, the remote script in full with its "
                        "SHA-256 and the files, and run nothing")
    p.add_argument("--host-key-fingerprint", default="",
                   help="with --ssh: the SHA256:... fingerprint of the host's key, which you "
                        "read off that machine; needed when its key is not in known_hosts")


def _emit(args: argparse.Namespace, document: dict[str, Any], text: str) -> None:
    say(json.dumps(document, indent=1, default=str) if args.json else text)


def _typed_fingerprint(args: argparse.Namespace):
    def offered(prints: list[str]) -> str:
        if args.host_key_fingerprint:
            return args.host_key_fingerprint.strip()
        if not (sys.stdin.isatty() and sys.stdout.isatty()):
            return ""
        warn("this host's key is not in your known_hosts. It presents: " + ", ".join(prints))
        return input("type the fingerprint you read on that machine, in full: ").strip()
    return offered


def cmd_ssh(args: argparse.Namespace) -> int:
    try:
        target = parse_target(args.ssh, args.ssh_port)
        state = Path(args.state) if getattr(args, "state", "") else home.state("onboard")
        try:
            routed = ssh_address(target.host, target.port, Devices(state / "devices.json"), detect)
        except OSError as exc:
            raise SshRefused(str(exc)) from None
        if routed != target.host:
            target = parse_target(f"{target.user}@{routed}" if target.user else routed, target.port)
        share = Path(args.share) if args.share else None
        if share is None or not share.is_dir():
            raise SshRefused("--share DIR with the wheel to install")
        if args.dry_run:
            doc = plan(target, share)
            _emit(args, doc, "\n".join([
                "dry run: nothing was run or contacted", f"target {doc['target']}",
                *("  " + " ".join(c) for c in doc["commands"]),
                f"remote script sha256 {doc['script_sha256']}:", doc["script"],
                *(f"  {f['name']} {f['size']} bytes sha256 {f['sha256']}" for f in doc["files"])]))
            return 0
        directory = Path(args.state) if args.state else home.state("onboard")
        keys = SigningKeys(directory)
        made = keys.attest(program_entries(share), serial=int(time.time()), namespace=NAMESPACE,
                           confirm=confirm_signing)
        done = bootstrap_over_ssh(
            target, share, Signed(made["manifest"], made["sshsig"], made["allowed_signers"],
                                  made["key_id"]), _typed_fingerprint(args))
    except (SshRefused, KeyStoreError, HumanRequired) as exc:
        _emit(args, {"error": str(exc)}, f"error: {exc}")
        return 2
    _emit(args, done, f"installed on {done['target']}; it is listening on port "
                      f"{done['listen_port']}: pair with: ml-stack fleet pair --host "
                      f"{target.host} --port {done['listen_port']}")
    return 0
