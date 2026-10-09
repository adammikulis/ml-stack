"""Join this device to a pool by the beacon: start the node with its network on, set the join policy, and show who is in.

    python -m ml_stack.node_join join [--policy open|secure] [--state DIR] [--binary PATH] [--wait SECONDS]
    python -m ml_stack.node_join status [--state DIR]

Under policy `open` two devices on the same network, each with its node running, hear each other's signed beacon and enrol
one another on first contact; each enrolment is recorded (by `open`, with the certificate fingerprint) and can be revoked
with `member_revoke`. Under `secure` nothing enrols without a pairing code. `pool-join-check` (scripts/) walks every step
of this on one device and says which fails and what to change.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from ml_stack import node_binary, node_pool, node_supervise
from ml_stack.command import Group, flag
from ml_stack.log import say, warn
from ml_stack.node_health import call, node_health
from ml_stack.node_launch import NodeUnavailable, default_state, ensure_node, stop_node

DEFAULT_PORT = 7447
DEFAULT_BEACON_PORT = 7448
BOARD = "pool"
SESSION = "pool-join"
POLICIES = ("open", "secure")
PAIR_CODE_ENV = "ML_STACK_PAIR_CODE"
POLL_S = 1.0


class PoolError(OSError):
    """The node could not be made to join, and the message says what to do."""


EXPLANATION = """\
This turns on the node's network so this device can find and join another one on the same network.
  What:  the node listens on TCP {port} and sends and listens for a small signed beacon on UDP {beacon_port}.
  Why:   so two of your devices on the same wifi or cable can pool without a code (join policy open),
         or be paired with one (secure). Nothing leaves this network.
  What you will see: your computer may ask to allow this program to find devices on your local network
         (macOS: Local Network; Windows: a firewall prompt). Allow it. It asks once, only now.
  Undo:  python -m ml_stack.node_join join --policy secure, or stop the node with python -m ml_stack.node_launch stop.
"""


def consent(port: int, beacon_port: int, *, yes: bool, ask=input, interactive: bool | None = None) -> None:
    """Show what turning the network on does and get a yes from the person; PoolError when there is none to give.

    Nothing that runs without a person (a test, an install, a first run) calls this or turns the network on.
    """
    say(EXPLANATION.format(port=port, beacon_port=beacon_port))
    if yes:
        return
    if not (sys.stdin.isatty() if interactive is None else interactive):
        raise PoolError("turning the network on needs a person to agree: run it in a terminal, or pass --yes after reading the above")
    if ask("Turn the network on? [y/N] ").strip().lower() not in ("y", "yes"):
        raise PoolError("the network stays off")


def network_args() -> list[str]:
    """The arguments that turn the node's network on (`--lan`: every interface, TCP 7447, the beacon on UDP 7448)."""
    return node_pool.network_args()


def pin_binary(state: Path, binary: Path) -> None:
    """Make the node of ``state`` run ``binary`` (recorded with its checksum, checked at every start)."""
    if not binary.is_file():
        raise PoolError(f"{binary} is not a file; build it with: cargo build --release -p poolside-node (in app/)")
    node_supervise.point(state, binary, node_binary.sha256(binary))


def status(state: Path) -> dict:
    """The node's `pool_status`: the pool, the policy, the members, and what the beacon has sent and heard."""
    return call(state, "pool_status")


def _token(state: Path) -> str:
    """A session on the board `pool` (the same one each time); changing the pool needs a registered session."""
    return call(state, "register", {"model": "claude-sonnet-5-5", "harness": "poolside-pool", "session": SESSION}, board=BOARD)["token"]


def set_policy(state: Path, policy: str) -> dict:
    """Set the join policy of this device's pool; `open` or `secure`."""
    if policy not in POLICIES:
        raise PoolError(f"the join policy is open or secure, not {policy!r}")
    return call(state, "set_join_policy", {"policy": policy}, token=_token(state))


def _wants_restart(state: Path, port: int) -> bool:
    """Whether a running node has its network off, or on another port than the one asked for."""
    try:
        listen = status(state).get("listen")
    except (OSError, ValueError):
        return False
    return not listen or not str(listen).endswith(f":{port}")


def start(state: Path, *, port: int = DEFAULT_PORT, binary: Path | None = None,
          extra: list[str] | None = None) -> dict:
    """The node's health once it runs with its network on, restarting it when it ran without."""
    if binary is not None:
        pin_binary(state, binary)
    if node_health(state) is not None and _wants_restart(state, port):
        stop_node(state)
    try:
        return ensure_node(state, extra=[*network_args(), *(extra or [])])
    except (NodeUnavailable, node_binary.NodeBinaryError) as exc:
        raise PoolError(f"{exc}; build the node with: cargo build --release -p poolside-node (in app/), then pass --binary") from exc


def others(shown: dict) -> list[dict]:
    """The active members other than this device."""
    return [m for m in shown.get("members", []) if m.get("status") == "active" and not m.get("self")]


def join(state: Path, *, policy: str = "open", wait_s: float = 0.0, **start_args) -> dict:
    """Start the node with its network on, set the policy and wait up to ``wait_s`` for another device; the pool status."""
    start(state, **start_args)
    set_policy(state, policy)
    end = time.monotonic() + wait_s
    while True:
        shown = status(state)
        if others(shown) or time.monotonic() >= end:
            return shown
        time.sleep(POLL_S)


def _run(handler):
    def command(args) -> int:
        state = Path(args.state) if args.state else default_state()
        try:
            result = handler(state, args)
        except OSError as exc:
            warn(f"pool: {exc}")
            return 1
        say(json.dumps(result))
        return 0
    return command


STATE = flag("--state", default="", help="the node's state directory (default: the machine's)")
GROUP = Group("ml_stack.node_join", "Join this device to a pool by the beacon.")
def _join(state: Path, a) -> dict:
    consent(DEFAULT_PORT, DEFAULT_BEACON_PORT, yes=a.yes)
    return join(state, policy=a.policy, wait_s=a.wait, binary=Path(a.binary) if a.binary else None)


GROUP.add("join", _run(_join),
    help="start the node with its network on and set the join policy",
    options=[STATE, flag("--policy", default="open", choices=POLICIES, help="open: devices on this network enrol by themselves; secure: a pairing code"),
             flag("--wait", type=float, default=0.0, help="seconds to wait for another device to appear"),
             flag("--yes", action="store_true", help="agree to turning the network on without being asked"),
             flag("--binary", default="", help="run this poolside-node binary instead of the selected runtime's")])
def _invite(state: Path, a) -> dict:
    return call(state, "pair_accept", {"ttl_s": a.ttl}, token=_token(state))


def _pair(state: Path, a) -> dict:
    code = a.code or os.environ.get(PAIR_CODE_ENV, "")
    if not code:
        raise PoolError(f"give the code the other device printed: --code, or {PAIR_CODE_ENV} in the environment")
    return call(state, "pair_start", {"host": a.host, "port": a.port, "passphrase": code}, token=_token(state))


GROUP.add("invite", _run(_invite), help="make a short-lived pairing code for another device to join this pool with (policy secure)",
    options=[STATE, flag("--ttl", type=int, default=300, help="seconds the code stays valid")])
GROUP.add("pair", _run(_pair), help="join the pool of the device that printed a pairing code",
    options=[STATE, flag("--host", required=True, help="that device's address"), flag("--port", type=int, default=DEFAULT_PORT),
             flag("--code", default="", help=f"the code (else {PAIR_CODE_ENV})")])
GROUP.add("status", _run(lambda state, a: status(state)), help="the pool, its members and the beacon's counts", options=[STATE])

main = GROUP.run


if __name__ == "__main__":
    raise SystemExit(main())
