"""`poolhouse node permission`: make macOS ask for Local Network permission for Poolhouse, and say what the person answered.

It launches Poolhouse.app once with `probe`, which joins the beacon multicast group, sends one datagram of its own to it (and one to the router) and
listens for it coming back (`poolhouse-node probe`). The first time, macOS shows the Local Network prompt for "Poolhouse" and
holds the probe's traffic until the person answers, so the probe waits up to 30 seconds and stops at the first datagram it
hears. This is the only place that touches multicast on this Mac outside a node the person turned the network on for; tests
parse results and build command lines, and open no sockets.

    poolhouse node permission [--seconds N]
    poolhouse node build [--binary PATH] [--icon SVG]
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from poolhouse import node_app
from poolhouse.command import Group, flag
from poolhouse.log import say, warn

GRANTED, DENIED, NOT_ASKED = "GRANTED", "DENIED", "NOT ASKED"
SECONDS = 30
SLACK_S = 20.0


def parse_result(text: str) -> dict | None:
    """The probe's JSON line as a dict, or None when it wrote nothing usable."""
    try:
        row = json.loads(text)
    except ValueError:
        return None
    return row if isinstance(row, dict) else None


def verdict(result: dict | None) -> str:
    """GRANTED when the probe heard its own datagram after a send went through (sends macOS refuses while it waits for the
    person's answer do not count), DENIED when it ran and did not, NOT ASKED when it never ran."""
    if result is None:
        return NOT_ASKED
    return GRANTED if result.get("heard_self") is True and bool(result.get("sent")) else DENIED


def gateway(route_output: str) -> str:
    """The default gateway in `route -n get default` output, or ''."""
    found = re.search(r"^\s*gateway:\s*(\d+\.\d+\.\d+\.\d+)\s*$", route_output, re.MULTILINE)
    return found.group(1) if found else ""


def default_gateway() -> str:
    """The router's address on this network, or '' when there is none."""
    try:
        done = subprocess.run(["route", "-n", "get", "default"], capture_output=True, text=True, timeout=10, check=False)
    except (OSError, subprocess.SubprocessError):
        return ""
    return gateway(done.stdout)


def probe_command(bundle: Path, out: Path, seconds: int, peers: list[str] | None = None) -> list[str]:
    """Launch the bundle through Launch Services (`open`), the way a person's double-click would, and wait for it to exit."""
    return ["open", "-W", "-n", "-a", str(bundle), "--args", "probe", "--seconds", str(seconds), "--out", str(out),
            *(word for peer in peers or [] for word in ("--peer", peer))]


def explanation(verdict_: str, result: dict | None) -> str:
    """The sentence that goes with a verdict."""
    if verdict_ == GRANTED:
        return "Poolhouse heard its own message on this network, so Local Network access is allowed."
    if verdict_ == NOT_ASKED:
        return "Poolhouse did not run, so macOS was not asked. Build it with `poolhouse node build` and try again."
    detail = f" ({result.get('error') or result.get('last_error')})" if result and (result.get("error") or result.get("last_error")) else ""
    return ("Poolhouse could not hear itself on this network" + detail + ". If you clicked Don't Allow, turn Poolhouse on in "
            "System Settings > Privacy & Security > Local Network, then run this again.")


def permission(seconds: int = SECONDS) -> str:
    """Run the probe through Poolhouse.app once and return the verdict; the bundle must have been built."""
    bundle = node_app.bundle_path()
    if not node_app.executable_path(bundle).is_file():
        warn(f"node permission: {bundle} is not built; run `poolhouse node build` first")
        return NOT_ASKED
    with tempfile.TemporaryDirectory(prefix="poolhouse-probe") as scratch:
        out = Path(scratch) / "probe.json"
        say("macOS will show a Local Network prompt for Poolhouse: click Allow on the Poolhouse prompt.")
        try:
            subprocess.run(probe_command(bundle, out, seconds, [router] if (router := default_gateway()) else []), timeout=seconds + SLACK_S, check=False, capture_output=True)
        except (OSError, subprocess.SubprocessError) as exc:
            warn(f"node permission: could not launch Poolhouse: {exc}")
        result = parse_result(out.read_text(encoding="utf-8")) if out.is_file() else None
    said = verdict(result)
    say(f"{said}: {explanation(said, result)}")
    return said


def _permission_command(args) -> int:
    return 0 if permission(args.seconds) == GRANTED else 1


GROUP = Group("poolhouse.node_permission", "The macOS identity of the LAN node: build it, and ask for Local Network permission.")
GROUP.add("permission", _permission_command, help="ask macOS for Local Network permission for Poolhouse and say the answer",
          options=[flag("--seconds", type=int, default=SECONDS, help="how long the probe waits to hear itself")])
GROUP.add("build", node_app.build_command, help="build and sign Poolhouse.app from the node binary", options=[
    flag("--binary", default="", help="the node binary (default: the selected runtime's)"),
    flag("--icon", default="", help="an SVG for the icon (default: the Poolhouse mark)")])

main = GROUP.run


if __name__ == "__main__":
    sys.exit(main())
