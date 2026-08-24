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
import getpass
import json
import sys

from .discovery import (
    MIN_PASSPHRASE,
    DiscoveryError,
    create_cluster_key,
    derive_token,
    discover,
    join_cluster,
    key_path,
    load_cluster_key,
)

DEFAULT_GROUP_NAME = "ml-stack"
"""The group a passphrase belongs to. Two households that both chose the same words end
up in different clusters only if they also chose different group names -- so this is
worth asking about even though almost nobody will change it."""


def _require_key(path: str | None) -> bytes:
    key = load_cluster_key(path)
    if key is None:
        raise DiscoveryError(
            f"no cluster key at {key_path(path)} -- run 'ml-stack-peers init'")
    return key


def _prompt_passphrase(confirm: bool) -> str:
    """Ask twice, because a typo here does not fail -- it silently makes a cluster of one.

    That is the failure worth spending a second prompt on. A wrong passphrase produces a
    perfectly valid key for a cluster nobody else is in, the daemon starts happily, and
    the only symptom is that ``ls`` finds nobody -- which looks exactly like a network
    problem and gets debugged as one.
    """
    while True:
        first = getpass.getpass("  Passphrase: ")
        if len(first.strip()) < MIN_PASSPHRASE:
            print(f"  Too short -- at least {MIN_PASSPHRASE} characters. "
                  "A few words you will remember beats a short complicated one.")
            continue
        if not confirm:
            return first
        again = getpass.getpass("  Again:      ")
        if first != again:
            print("  Those did not match. Try again.")
            continue
        return first


def cmd_setup(args: argparse.Namespace) -> int:
    """The whole join story, for someone who has never used this before."""
    interactive = sys.stdin.isatty()
    p = key_path(args.cluster_key)

    if p.exists() and not args.force:
        print(f"This machine is already in a cluster (key at {p}).")
        print("Run 'ml-stack-peers ls' to see who else is in it,")
        print("or 'ml-stack-peers setup --force' to join a different one.")
        return 0

    print("Connect this machine to the others you want to train with.")
    print("Run this on every machine, with the SAME passphrase. That is all it takes.")
    print()

    group = args.group
    if interactive and not args.group_given:
        typed = input(f"  Group name [{DEFAULT_GROUP_NAME}]: ").strip()
        group = typed or DEFAULT_GROUP_NAME

    if args.passphrase:
        passphrase = args.passphrase
    elif interactive:
        passphrase = _prompt_passphrase(confirm=True)
    else:
        # Piped input: read one line. Scripted setup should be possible without a TTY,
        # but never by leaving the passphrase in shell history or in `ps`.
        passphrase = sys.stdin.readline()
        if not passphrase.strip():
            print("error: no passphrase on stdin", file=sys.stderr)
            return 2

    print()
    print("  Deriving the key (this is deliberately slow, once)...", flush=True)
    join_cluster(passphrase, group=group, path=args.cluster_key)

    print(f"  Joined '{group}'.")
    print()
    print("Next:")
    print("  1. Run this same command on your other machines, same passphrase.")
    print("  2. On each of them, start the daemon:")
    print()
    print("       ml-stack-traind                 # a machine with a GPU")
    print("       ml-stack-traind --slots 8       # a machine that prepares data")
    print()
    print("  3. Check they found each other:")
    print()
    print("       ml-stack-peers ls")
    print()
    print("Anyone who knows this passphrase can run commands on every machine in the")
    print("group. Treat it like the password to your house, not like a wifi password.")
    return 0


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
    setup = sub.add_parser(
        "setup", help="join a cluster with a passphrase (start here)")
    setup.add_argument("--group", default=DEFAULT_GROUP_NAME,
                       help="which cluster these words belong to, so two groups on one "
                            f"network stay separate (default: {DEFAULT_GROUP_NAME})")
    setup.add_argument("--passphrase", default="",
                       help="skip the prompt. Avoid on a shared machine: it lands in "
                            "your shell history and in 'ps'.")
    setup.add_argument("--force", action="store_true",
                       help="leave the cluster this machine is in and join another")
    sub.add_parser("init", help="mint a random key instead of using a passphrase")
    sub.add_parser("key", help="print the cluster key")
    sub.add_parser("token", help="print the traind bearer token this key derives")
    ls = sub.add_parser("ls", help="list daemons on this LAN")
    ls.add_argument("--timeout", type=float, default=2.0)
    ls.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    args.group_given = "--group" in (argv if argv is not None else sys.argv)
    fn = {"setup": cmd_setup, "init": cmd_init, "key": cmd_key,
          "token": cmd_token, "ls": cmd_ls}[args.cmd]
    try:
        return fn(args)
    except DiscoveryError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
