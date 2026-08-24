"""``ml-stack-peers`` -- set up a cluster and see who is in it.

    ml-stack-peers init     # mint the key, print the command to run elsewhere
    ml-stack-peers ls       # who is on this LAN right now
    ml-stack-peers token    # the bearer token this key derives

Deliberately small. The whole setup story is "run init here, paste one line
there", and anything more elaborate than that is a thing to get wrong at 1am on
a machine you are logged into over ssh.
"""

from __future__ import annotations

import argparse
import json
import sys

from .discovery import (
    DiscoveryError,
    create_cluster_key,
    derive_token,
    discover,
    key_path,
    load_cluster_key,
)


def _require_key(path: str | None) -> bytes:
    key = load_cluster_key(path)
    if key is None:
        raise DiscoveryError(
            f"no cluster key at {key_path(path)} -- run 'ml-stack-peers init'")
    return key


def cmd_init(args: argparse.Namespace) -> int:
    p = key_path(args.cluster_key)
    existed = p.exists()
    key = create_cluster_key(args.cluster_key)
    print(f"cluster key {'already at' if existed else 'written to'} {p}")
    print()
    print("Run this on every other machine that should join:")
    print()
    print(f"    mkdir -p {p.parent} && printf '%s\\n' '{key}' > {p} "
          f"&& chmod 600 {p}")
    print()
    print("Then start the daemon on the box with the card:")
    print()
    print("    ml-stack-traind")
    print()
    print("Anyone holding this key can run commands on every daemon in the")
    print("cluster. Treat it exactly like an ssh private key.")
    return 0


def cmd_key(args: argparse.Namespace) -> int:
    print(_require_key(args.cluster_key).decode())
    return 0


def cmd_token(args: argparse.Namespace) -> int:
    print(derive_token(_require_key(args.cluster_key)))
    return 0


def cmd_ls(args: argparse.Namespace) -> int:
    peers = discover(_require_key(args.cluster_key), timeout_s=args.timeout)
    if args.json:
        print(json.dumps([{**p.public(), "host": p.host,
                           "base_url": p.base_url} for p in peers], indent=2))
        return 0 if peers else 1
    if not peers:
        print("no peers answered.")
        print("  - is 'ml-stack-traind' running there?")
        print("  - same LAN, and does that box hold the same cluster key?")
        return 1
    print(f"{'NAME':<16} {'URL':<28} {'FREE':<7} {'STATE':<10} DEVICE")
    for p in peers:
        state = "busy" if p.free == 0 else "idle"
        if p.queued:
            state += f" +{p.queued}"
        slots = f"{p.free}/{p.slots}"
        gpu = (p.device.get("gpu")
               or ",".join(p.device.get("backends") or [])
               or f"{p.device.get('cpus', '?')} cpu")
        vram = p.device.get("vram_free_gb")
        if vram is not None:
            gpu += f"  {vram}/{p.device.get('vram_total_gb', '?')} GB free"
        print(f"{p.name:<16} {p.base_url:<28} {slots:<7} {state:<10} {gpu}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="ml-stack-peers")
    ap.add_argument("--cluster-key", default=None,
                    help="path to the cluster key (default: ~/.ml-stack/cluster.key)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init", help="mint the cluster key and print how to join")
    sub.add_parser("key", help="print the cluster key")
    sub.add_parser("token", help="print the traind bearer token this key derives")
    ls = sub.add_parser("ls", help="list daemons on this LAN")
    ls.add_argument("--timeout", type=float, default=2.0)
    ls.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    fn = {"init": cmd_init, "key": cmd_key, "token": cmd_token, "ls": cmd_ls}[args.cmd]
    try:
        return fn(args)
    except DiscoveryError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
