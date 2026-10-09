"""Turn test shards on or off for this device's node, and say whose tests it takes:

    python -m ml_stack.testfarm.consent on --from DEVICE [--from DEVICE ...] [--python PATH] [--repo PATH]
    python -m ml_stack.testfarm.consent allow DEVICE... | deny DEVICE... | off | status [--json]

Run at the device that would run the tests, by its person or by its agent on the person's order. A DEVICE
is a name from `ml-stack-test-devices` or a fingerprint. Nobody is allowed until named, however the pool
was joined. Turning on and allowing are audited on the pool board; `deny` and `off` stop the next upload.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from ml_stack import features
from ml_stack.board import session as board_session
from ml_stack.board.client import NodeError
from ml_stack.log import say
from ml_stack.testfarm.client import ShardError, choose, pool_devices

ACTIONS = ("on", "off", "status", "allow", "deny")
FEATURE = "remote-tests"
"""The experimental feature (`ml_stack.features`) that has to be on before a device takes tests or sends them."""
FINGERPRINT = re.compile(r"^[0-9a-f]{64}$")


def own_checkout() -> str:
    """The checkout this module was imported from (``src/ml_stack/fleet/`` is three levels below it)."""
    return str(Path(__file__).resolve().parents[3])


def fingerprints(named: list[str]) -> list[str]:
    """The fingerprints of the devices ``named`` (pool names or fingerprints); ShardError when one is not in the pool."""
    if "all" in named:
        raise ShardError("name the devices one by one: `all` is not a device")
    devices = pool_devices()
    return [choose(name, devices)[0]["fingerprint"] if not FINGERPRINT.match(name) else name for name in named]


def switch(action: str, python: str = "", repo: str = "", devices: list[str] | None = None) -> dict:
    """Apply ``action`` through this device's node; what the node now holds."""
    seat = board_session.connect()
    params: dict = {"enabled": action == "on"} if action in ("on", "off") else {}
    if action == "on":
        params.update(python=python or sys.executable, repo=repo or own_checkout())
    named = fingerprints(devices or [])
    if action in ("on", "allow") and named:
        params["allow"] = named
    if action == "deny":
        params["deny"] = named
    return seat.client.call("shard_consent", "", seat.token, **params)


def options(argv: list[str]) -> dict | None:
    """The command line as a dict (``action``, ``python``, ``repo``, ``json``, ``devices``), or None when it is malformed."""
    found: dict = {"action": "", "python": "", "repo": "", "json": "", "devices": []}
    words = iter(argv)
    for word in words:
        if word in ("--python", "--repo", "--from"):
            value = next(words, "")
            if not value or value.startswith("-"):
                return None
            if word == "--from":
                found["devices"].append(value)
            else:
                found[word[2:]] = value
        elif word == "--json":
            found["json"] = "1"
        elif not found["action"] and word in ACTIONS:
            found["action"] = word
        elif found["action"] in ("allow", "deny") and not word.startswith("-"):
            found["devices"].append(word)
        else:
            return None
    if found["devices"] and found["action"] not in ("on", "allow", "deny"):
        return None
    if found["action"] in ("allow", "deny") and not found["devices"]:
        return None
    return found if found["action"] else None


def describe(held: dict) -> str:
    """The line a person reads: whether it is on, with what, and whose tests it takes."""
    if not held["enabled"]:
        return "test shards: off"
    taken = f"takes tests from {len(held['allowed'])} device(s)" if held["allowed"] else "takes tests from nobody yet: `allow DEVICE` names one"
    return f"test shards: on (python {held['python']}, checkout {held['repo']}), {taken}"


def run(argv: list[str]) -> int:
    """Set or show the switch; 1 when the node refuses, 2 for a usage error."""
    given = options(argv)
    if given is None:
        say("usage: python -m ml_stack.testfarm.consent on [--from DEVICE]... | allow DEVICE... | deny DEVICE... | off | status "
            "[--python PATH] [--repo PATH] [--json]")
        return 2
    if given["action"] == "on" and not features.enabled(FEATURE):
        say(f"test shards: remote tests are an experimental feature and off here; a person turns them on with `ml-stack features enable {FEATURE}`")
        return 1
    try:
        held = switch(given["action"], given["python"], given["repo"], given["devices"])
    except (NodeError, ShardError) as exc:
        say(f"test shards: {exc}")
        return 1
    say(json.dumps(held) if given["json"] else describe(held))
    return 0


if __name__ == "__main__":
    sys.exit(run(sys.argv[1:]))
