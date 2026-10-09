"""Turn test shards on or off for this device's node: ``python -m ml_stack.testfarm.consent on|off|status``.

Run at the device that would run the tests, by its person or by its agent on the person's order.
Turning shards on lets any other device of the pool run a tree's tests here as this user, so it
is audited (the node writes it to the pool board with the session that did it) and `off` stops the
next upload. ``on`` takes this device's own Python 3.13 and checkout: the interpreter running this
command and the checkout it was imported from, unless ``--python`` and ``--repo`` name others.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from ml_stack import features
from ml_stack.board import session as board_session
from ml_stack.board.client import NodeError
from ml_stack.log import say

ACTIONS = ("on", "off", "status")
FEATURE = "remote-tests"
"""The experimental feature (`ml_stack.features`) that has to be on before a device takes tests or sends them."""


def own_checkout() -> str:
    """The checkout this module was imported from (``src/ml_stack/fleet/`` is three levels below it)."""
    return str(Path(__file__).resolve().parents[3])


def switch(action: str, python: str = "", repo: str = "") -> dict:
    """Apply ``action`` through this device's node; what the node now holds."""
    seat = board_session.connect()
    params: dict = {} if action == "status" else {"enabled": action == "on"}
    if action == "on":
        params.update(python=python or sys.executable, repo=repo or own_checkout())
    return seat.client.call("shard_consent", "", seat.token, **params)


def options(argv: list[str]) -> dict[str, str] | None:
    """``ACTION [--python PATH] [--repo PATH] [--json]`` as a dict (``action``, ``python``, ``repo``, ``json``), or None when it is malformed."""
    found = {"action": "", "python": "", "repo": "", "json": ""}
    words = iter(argv)
    for word in words:
        if word in ("--python", "--repo"):
            found[word[2:]] = next(words, "")
            if not found[word[2:]]:
                return None
        elif word == "--json":
            found["json"] = "1"
        elif word in ACTIONS and not found["action"]:
            found["action"] = word
        else:
            return None
    return found if found["action"] else None


def run(argv: list[str]) -> int:
    """Set or show the switch; 1 when the node refuses, 2 for a usage error."""
    given = options(argv)
    if given is None:
        say("usage: python -m ml_stack.testfarm.consent on|off|status [--python PATH] [--repo PATH] [--json]")
        return 2
    if given["action"] == "on" and not features.enabled(FEATURE):
        say(f"test shards: remote tests are an experimental feature and off here; a person turns them on with `ml-stack features enable {FEATURE}`")
        return 1
    try:
        held = switch(given["action"], given["python"], given["repo"])
    except NodeError as exc:
        say(f"test shards: {exc}")
        return 1
    say(json.dumps(held) if given["json"] else
        f"test shards: {'on' if held['enabled'] else 'off'}" + (f" (python {held['python']}, checkout {held['repo']})" if held["enabled"] else ""))
    return 0


if __name__ == "__main__":
    sys.exit(run(sys.argv[1:]))
