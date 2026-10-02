"""``ml-stack credentials``: set, list, unset and locate the credentials file."""

from __future__ import annotations

import argparse
import getpass
import json
import sys

from ml_stack import credentials
from ml_stack.command import Group, flag, option
from ml_stack.log import say, warn

COMMANDS = Group(
    "ml-stack credentials",
    "Tokens and keys ml-stack uses. `set` reads the value from a hidden prompt or from stdin, "
    "never from an argument, and nothing prints a value back. The file is "
    "~/.ml-stack/credentials.toml, mode 0600; ML_STACK_CREDENTIALS_FILE moves it.")

KEYCHAIN = flag("--keychain", action="store_true",
                help="the OS keychain, through the `keyring` package, instead of the file")


def _guarded(run):
    def inside(args: argparse.Namespace) -> int:
        try:
            return run(args)
        except credentials.CredentialError as exc:
            warn(f"ml-stack credentials: {exc}")
            return 1
    return inside


@COMMANDS.command("set", help="store NAME, reading the value from a prompt or stdin",
                  options=(flag("name"), KEYCHAIN,
                           flag("--stdin", action="store_true", help="read the value from stdin")))
@_guarded
def put(args: argparse.Namespace) -> int:
    """Store the value read from the prompt or stdin."""
    value = (sys.stdin.readline() if args.stdin or not sys.stdin.isatty()
             else getpass.getpass(f"{args.name}: "))
    say(f"{args.name} stored in {credentials.set(args.name, value, keychain=args.keychain)}")
    return 0


@COMMANDS.command("unset", help="remove NAME", options=(flag("name"), KEYCHAIN))
@_guarded
def drop(args: argparse.Namespace) -> int:
    """Remove a stored credential."""
    gone = credentials.unset(args.name, keychain=args.keychain)
    say(f"{args.name} removed" if gone else f"{args.name} was not stored")
    return 0


@COMMANDS.command("list", help="each name with where it comes from, never a value",
                  options=(option("json"),))
@_guarded
def listing(args: argparse.Namespace) -> int:
    """Print each credential's name and source."""
    rows = credentials.describe()
    if args.json:
        say(json.dumps(rows, indent=2))
        return 0
    for row in rows:
        state = row["source"] if row["present"] else row.get("error", "not set")
        say(f"{row['name']:<24} {'set' if row['present'] else 'unset':<6} {state}")
    return 0


@COMMANDS.command("path", help="where the credentials file is")
def where(_args: argparse.Namespace) -> int:
    """Print the credentials file's path."""
    say(credentials.file_path())
    return 0


command = COMMANDS.run
