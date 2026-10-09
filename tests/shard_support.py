"""A second pool daemon on loopback standing in for a paired device, with its own state under a folder.

Real TLS pinned to the stand-in's certificate, real signed and sealed requests, a real `JobRunner`
and `ShardHost`. As a script it serves until stdin closes:
``ML_STACK_HOME=SENDER python shard_support.py serve BASE PORT [enabled|disabled]`` prints its port.
"""

from __future__ import annotations

import base64
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

from ml_stack import home
from ml_stack.fleet import tls
from ml_stack.fleet.api import Daemon, make_handler
from ml_stack.fleet.framing import LimitedServer
from ml_stack.fleet.jobs import JobRunner
from ml_stack.fleet.onboard.requests import Device, Devices
from ml_stack.fleet.shard_host import ShardHost
from ml_stack.hub.peerbook import PeerBook

NAME = "standin"
SECRET = base64.urlsafe_b64encode(b"s" * 32).decode()


@dataclass
class Standin:
    """A running stand-in device and the pieces a test reaches into."""

    server: LimitedServer
    runner: JobRunner
    host: ShardHost
    flag: dict
    port: int

    def stop(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.runner.shutdown()


def pair_sender(ident: tls.Identity, port: int) -> None:
    """Write the sender's records (under the current state root) of a paired device at ``port``."""
    directory = home.state("onboard")
    directory.mkdir(parents=True, exist_ok=True)
    Devices(directory / "devices.json")._write([Device(
        ident.fingerprint, NAME, "standin-host", "127.0.0.1", 1, mine=True, secret=SECRET)])
    PeerBook(directory / "peers.json").add({
        "name": NAME, "url": f"https://127.0.0.1:{port}/", "certificate": ident.beacon,
        "source": "pairing", "fingerprint": ident.fingerprint, "device_secret": SECRET})


def start(base: Path, *, enabled: bool = True, sender: str = "mine", command=None, port: int = 0) -> Standin:
    """Start the stand-in under ``base`` and pair the sender (the current state root) to it; ``sender`` is
    how the stand-in knows it: ``mine`` (the owner's device), ``other`` (a paired device that is not) or ``none``."""
    ident = tls.identity(base / "identity", NAME)
    known = Device("b" * 64, "sender", "sender-host", "127.0.0.1", 1, mine=sender == "mine", secret=SECRET)
    flag = {"on": enabled}
    kwargs = {"command": command} if command else {}
    host = ShardHost(base / "files" / "test-shards", lambda: flag["on"], **kwargs)
    runner = JobRunner(base / "runner")
    daemon = Daemon(runner, base / "files", "cluster-token", devices=lambda: [known] if sender != "none" else [],
                    shards=host)
    server = LimitedServer(("127.0.0.1", port), make_handler(daemon), tls=tls.server_context(ident))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    pair_sender(ident, server.server_port)
    return Standin(server, runner, host, flag, server.server_port)


def main(argv: list[str]) -> None:
    """``serve BASE PORT [enabled|disabled]``: pair the sender under the state root, serve, wait for stdin to close."""
    base, port = Path(argv[1]), int(argv[2])
    standin = start(base, enabled=(argv[3] if len(argv) > 3 else "enabled") == "enabled", port=port)
    print(standin.port, flush=True)
    sys.stdin.read()
    standin.stop()


if __name__ == "__main__":
    main(sys.argv[1:])
