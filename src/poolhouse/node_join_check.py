"""Walk the automatic pool join on this device, one step at a time, and say which step fails and the exact change that fixes it.

Run `scripts/pool-join-check` on each of two devices on the same network. Each step is PASS, FAIL (with a `fix:` line) or SKIP
(an earlier step failed). Nothing secret is printed: fingerprints are public and no key, code or token is shown.
"""

from __future__ import annotations

import ipaddress
import platform
import socket
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from poolhouse import node_binary, node_join, runtime
from poolhouse.node_health import call

WSLCONFIG = r"%UserProfile%\.wslconfig"
HYPERV_ID = "{40E0AC32-46A5-438A-A0B2-2B479E8F2E90}"
BOARD = "pool-check"
SESSION = "pool-check"


@dataclass
class Step:
    """One check: its name, whether it held, what was found and, when it did not hold, what to change."""
    name: str
    ok: bool
    detail: str = ""
    fix: str = ""

    def line(self) -> str:
        text = f"{'PASS' if self.ok else 'FAIL'}  {self.name:<14} {self.detail}"
        return text if self.ok else f"{text}\n      fix: {self.fix}"


@dataclass
class Check:
    """What the walk knows so far."""
    state: Path
    port: int
    beacon_port: int
    wait_s: float
    policy: str
    binary: Path | None
    steps: list[Step] = field(default_factory=list)
    shown: dict = field(default_factory=dict)


def is_wsl() -> bool:
    """Whether this is Linux under WSL."""
    try:
        return "microsoft" in Path("/proc/version").read_text(encoding="utf-8").lower()
    except OSError:
        return False


def route_address() -> str:
    """This machine's address on the network its default route uses, or ''."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.connect(("192.0.2.1", 9))
            return probe.getsockname()[0]
    except OSError:
        return ""


def wsl_mode(address: str) -> str:
    """`nat` when a WSL2 machine's address is in the private 172.16/12 range Hyper-V's NAT hands out, else `mirrored`."""
    return "nat" if address and ipaddress.ip_address(address) in ipaddress.ip_network("172.16.0.0/12") else "mirrored"


def firewall_fix(port: int, beacon_port: int) -> str:
    """How to let the node's two ports in on this kind of machine."""
    if sys.platform == "darwin":
        return ("System Settings > Privacy & Security > Local Network: turn on the terminal app you run this from, then run it again. "
                "If a firewall prompt appears for poolhouse-node, choose Allow")
    if sys.platform == "win32":
        return (f'in an administrator PowerShell: New-NetFirewallRule -DisplayName poolhouse-node -Direction Inbound -Action Allow '
                f'-Profile Private -Protocol TCP -LocalPort {port}; New-NetFirewallRule -DisplayName poolhouse-beacon -Direction Inbound '
                f'-Action Allow -Profile Private -Protocol UDP -LocalPort {beacon_port}')
    if is_wsl():
        return (f"in an administrator PowerShell on Windows: Set-NetFirewallHyperVVMSetting -Name '{HYPERV_ID}' -DefaultInboundAction Allow "
                f"(or add inbound allow rules for TCP {port} and UDP {beacon_port}); then `wsl --shutdown` and start WSL again")
    return f"sudo ufw allow {port}/tcp && sudo ufw allow {beacon_port}/udp (or the same in the firewall you use)"


def step_platform(c: Check) -> Step:
    address = route_address()
    label = "WSL" if is_wsl() else platform.system()
    if not address:
        return Step("platform", False, label, "connect this device to the network (wifi or cable) that the other device is on")
    if is_wsl() and wsl_mode(address) == "nat":
        return Step("platform", False, f"{label}, NAT networking, address {address}",
                    f"WSL2 is on a private network of its own that no other device can reach or broadcast to. In Windows put "
                    f"[wsl2] and networkingMode=mirrored (on its own line) in {WSLCONFIG}, run `wsl --shutdown` in PowerShell and start WSL "
                    f"again (Windows 11 22H2 or newer); or run this check natively on Windows instead")
    return Step("platform", True, f"{label}, address {address}")


def step_binary(c: Check) -> Step:
    if c.binary is None and runtime.selection_prefix() is None:
        root = Path(__file__).resolve().parents[2]
        try:
            c.binary = node_binary.build(root)
        except (OSError, node_binary.NodeBinaryError) as exc:
            return Step("binary", False, str(exc)[:200], "install Rust (https://rustup.rs) and run: cargo build --release -p poolhouse-node (in app/), then pass --binary app/target/release/poolhouse-node")
    if c.binary is not None:
        try:
            node_join.pin_binary(c.state, c.binary)
        except OSError as exc:
            return Step("binary", False, str(exc), "pass --binary with the poolhouse-node you built")
    return Step("binary", True, str(c.binary or "the selected runtime's node"))


def step_node(c: Check) -> Step:
    try:
        health = node_join.start(c.state)
    except OSError as exc:
        return Step("node", False, str(exc), f"read {c.state / 'node.log'} for why it did not start")
    return Step("node", True, f"pid {health.get('pid')}, state {c.state}")


def step_listening(c: Check) -> Step:
    c.shown = node_join.status(c.state)
    if not c.shown.get("listen") or not c.shown.get("beacon"):
        return Step("listening", False, f"listen={c.shown.get('listen')} beacon={c.shown.get('beacon')}", f"stop the node and run this again: python -m poolhouse.node_launch stop --state {c.state}")
    return Step("listening", True, f"{c.shown['listen']} (beacon on UDP {c.beacon_port})")


def step_beacon_send(c: Check) -> Step:
    shown = _until(c, lambda s: s["beacon_sent"] > 0, 15)
    if shown["beacon_sent"] == 0:
        return Step("beacon-send", False, shown.get("beacon_last_send_error") or "nothing sent", firewall_fix(c.port, c.beacon_port))
    return Step("beacon-send", True, f"{shown['beacon_sent']} sent, {shown['beacon_send_errors']} errors")


def step_beacon_receive(c: Check) -> Step:
    shown = _until(c, lambda s: s["beacon_own"] > 0, 15)
    if shown["beacon_own"] == 0:
        return Step("beacon-receive", False, "this device does not hear its own beacon on the network", firewall_fix(c.port, c.beacon_port))
    return Step("beacon-receive", True, f"heard its own beacon {shown['beacon_own']} times (multicast and broadcast arrive)")


def step_policy(c: Check) -> Step:
    done = node_join.set_policy(c.state, c.policy)
    return Step("policy", True, f"join policy is {done['policy']}" + (" (devices on this network enrol one another; revoke one with member_revoke)" if c.policy == "open" else ""))


def step_peer_beacon(c: Check) -> Step:
    shown = _until(c, lambda s: s["beacon_heard"] > 0, c.wait_s)
    if shown["beacon_heard"] == 0:
        return Step("peer-beacon", False, f"no other device beaconed in {c.wait_s:.0f}s",
                    "run scripts/pool-join-check on the other device now, on the same network (not a guest network, not through a VPN)")
    return Step("peer-beacon", True, f"heard {shown['beacon_heard']} beacons from other devices")


def step_enrolled(c: Check) -> Step:
    shown = _until(c, lambda s: bool(node_join.others(s)), c.wait_s)
    others = node_join.others(shown)
    if not others:
        why = (shown.get("last_join") or {}).get("error") or "no enrolment was attempted"
        return Step("enrolled", False, why, "both devices must be on policy open and on the same network; a device that already has members does not change pool")
    first = others[0]
    return Step("enrolled", True, f"{first['fingerprint'][:16]} by {first['by']}, pool {shown['pool']}")


def step_converge(c: Check) -> Step:
    token = call(c.state, "register", {"model": "claude-sonnet-5-5", "harness": "poolhouse-pool", "session": SESSION}, board=BOARD)["token"]
    mine = f"pool-check from {socket.gethostname()} at {int(time.time())}"
    call(c.state, "post", {"kind": "message", "fields": {"type": "status", "body": mine}}, board=BOARD, token=token)
    seen: list[str] = []

    def other_wrote(_: dict) -> bool:
        rows = call(c.state, "read", {"kind": "message", "limit": 100}, board=BOARD, token=token)["entries"]
        seen[:] = [r["fields"]["body"] for r in rows if r["fields"]["body"] != mine]
        return bool(seen)

    _until(c, other_wrote, c.wait_s)
    if not seen:
        return Step("converge", False, "the other device's message did not arrive", "run the check on the other device too, then sync: python -m poolhouse.node_join status")
    return Step("converge", True, f"received: {seen[0]}")


def _until(c: Check, ready: Callable[[dict], bool], seconds: float) -> dict:
    end = time.monotonic() + seconds
    while True:
        c.shown = node_join.status(c.state)
        if ready(c.shown) or time.monotonic() >= end:
            return c.shown
        time.sleep(1.0)


STEPS: list[Callable[[Check], Step]] = [step_platform, step_binary, step_node, step_listening, step_beacon_send,
                                       step_beacon_receive, step_policy, step_peer_beacon, step_enrolled, step_converge]


def steps_for(c: Check) -> list[Callable[[Check], Step]]:
    """All the steps under `open`; under `secure` the ones up to the policy, since nothing enrols by itself."""
    return STEPS if c.policy == "open" else STEPS[:STEPS.index(step_policy) + 1]


def walk(c: Check, out: Callable[[str], None] = print) -> bool:
    """Run the steps in order, printing each; a failed step stops the rest (they are shown SKIP). True when all passed."""
    failed = False
    for make in steps_for(c):
        if failed:
            out(f"SKIP  {make.__name__.removeprefix('step_').replace('_', '-')}")
            continue
        try:
            step = make(c)
        except (OSError, ValueError, KeyError) as exc:
            step = Step(make.__name__.removeprefix("step_").replace("_", "-"), False, f"{type(exc).__name__}: {exc}", "run it again; if it repeats, send this output")
        c.steps.append(step)
        out(step.line())
        failed = not step.ok
    return not failed


def summary(c: Check) -> str:
    """The closing lines: what the owner has now and how to undo it."""
    failed = next((s for s in c.steps if not s.ok), None)
    if failed is not None:
        return f"NOT READY at {failed.name}. The one fix: {failed.fix}\nThen run this again."
    if c.policy != "open":
        return "READY: the node runs with its network on and beacons; policy secure enrols nobody without a pairing code."
    return (f"READY: this device and {len(node_join.others(c.shown))} other(s) share pool {c.shown.get('pool')}. "
            f"Close the door again with: python -m poolhouse.node_join join --policy secure --state {c.state}")
