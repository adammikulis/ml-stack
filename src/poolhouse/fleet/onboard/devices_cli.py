"""``poolhouse cluster devices``: each paired device and the way it is reached (lan, tailnet or
unreachable), and whether a Tailscale client was found. ``--learn`` stores the tailnet address
of a device whose certificate answers there."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from poolhouse import home
from poolhouse.log import say

from ..tailnet import detect
from .pairing import DEFAULT_PORT
from .requests import Devices, short
from .routes import learn, reach

__all__ = ["add_devices", "cmd_devices"]


def add_devices(sub: Any, common: Any) -> None:
    p = common(sub.add_parser("devices", help="paired devices and how each is reached: lan, "
                                              "tailnet or unreachable"))
    p.add_argument("--port", type=int, default=DEFAULT_PORT,
                   help="the port a paired device's certificate is checked on")
    p.add_argument("--learn", action="store_true",
                   help="remember the tailnet address of a device whose certificate answers "
                        "there")


def cmd_devices(args: argparse.Namespace) -> int:
    state = Path(args.state) if getattr(args, "state", "") else home.state("onboard")
    devices = Devices(state / "devices.json")
    tailnet = detect()
    if args.learn:
        learn(devices, tailnet, args.port)
    rows = [reach(d, tailnet, args.port).public() | {"paired": d.status}
            for d in devices.all() if d.status == "active"]
    document = {"tailscale": {"detected": tailnet.installed, "up": tailnet.up,
                              "state": tailnet.state}, "devices": rows}
    tail = ("tailscale: " + ("connected" if tailnet.up else "installed, not connected"
                             if tailnet.installed else "not detected"))
    lines = [f"{r['name']}  {short(r['fingerprint'])}  {r['route']}"
             + (f"  {r['address']}" if r["address"] else "") for r in rows] \
        or ["no paired devices"]
    say(json.dumps(document, indent=1) if args.json else "\n".join([tail, *lines]))
    return 0
