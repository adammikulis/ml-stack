"""``ml-stack security sandbox status|test``."""

from __future__ import annotations

import os
import socket
import sys
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any

from ml_stack.sandbox import policies
from ml_stack.sandbox.policy import Net, Policy
from ml_stack.sandbox.run import backend, run
from ml_stack.sandbox.seatbelt import BINARY, DEPRECATION

__all__ = ["status", "summary", "test"]


def status() -> dict[str, Any]:
    """Which backend this machine uses, whether it can run, and the deprecated tool in use."""
    chosen = backend()
    state = chosen.available()
    info: dict[str, Any] = {"platform": sys.platform, "backend": chosen.name,
                            "available": state.ok, "reason": state.reason}
    if chosen.name == "seatbelt":
        info |= {"binary": BINARY, "deprecated": DEPRECATION}
    return info


def summary() -> str:
    """The one line ``ml-stack security status`` prints about the sandbox."""
    info = status()
    if not info["available"]:
        return f"{info['backend']} unavailable ({info['reason']}); untrusted execution is refused"
    note = "; sandbox-exec is deprecated by Apple, kept behind one module" if "deprecated" in info else ""
    return f"{info['backend']} available{note}"


def test() -> list[dict[str, Any]]:
    """Run each guarantee against a real confined process: one row per check."""
    rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="ml-stack-sandbox-test-") as made:
        root = Path(os.path.realpath(made))
        allowed, outside = root / "allowed", root / "outside"
        allowed.mkdir()
        outside.mkdir()
        for folder in (allowed, outside):
            (folder / "f").write_text("data")
        pol = Policy("selftest", read=(str(allowed),), write=(str(allowed),), exec=policies.SYSTEM_EXEC,
                     net=Net.deny(), env=policies.system_env())

        def check(name: str, argv: list[str], want_ok: bool, using: Policy = pol) -> None:
            try:
                got = run(argv, using, diagnose="never")
                passed, detail = got.ok == want_ok, f"exit {got.returncode}"
            except RuntimeError as exc:
                passed, detail = False, str(exc)
            rows.append({"check": name, "passed": passed, "detail": detail})

        check("read inside the allow-list", [os.path.realpath("/bin/cat"), str(allowed / "f")], True)
        check("read outside the allow-list is refused", [os.path.realpath("/bin/cat"), str(outside / "f")], False)
        check("write inside the allow-list", [os.path.realpath("/bin/sh"), "-c", f"echo x > {allowed}/w"], True)
        check("write outside the allow-list is refused",
              [os.path.realpath("/bin/sh"), "-c", f"echo x > {outside}/w"], False)
        with socket.socket() as server:
            server.bind(("127.0.0.1", 0))
            server.listen(4)
            dial = [os.path.realpath("/usr/bin/nc"), "-z", "-w", "2", "127.0.0.1", str(server.getsockname()[1])]
            check("a loopback connection works when the policy allows it", dial, True,
                  replace(pol, net=Net.loopback()))
            check("the same connection is refused when the network is denied", dial, False)
    return rows
