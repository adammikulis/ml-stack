"""The broker behind a loopback socket, and the calls that reach it.

One JSON object per line each way. `serve` runs the machine's one broker -- it holds
``broker.lock`` -- and writes its port and token to ``broker.json``, readable by this user
alone. Every client call finds the broker there and starts one when none answers, so no
caller starts it by hand.
"""

from __future__ import annotations

import json
import os
import secrets
import socket
import socketserver
import sys
import threading
import time
from pathlib import Path
from typing import Any

from ml_stack import home, jobs
from ml_stack.files import read_json, write_json
from ml_stack.lock import Busy, only_one
from ml_stack.log import say as say_out
from ml_stack.platform import on_quit, private_file
from ml_stack.serve.backend import ServerFailed
from ml_stack.serve.broker import IDLE_S, Ask, Broker, BrokerError, Grant
from ml_stack.serve.ports import DEFAULT_HOST
from ml_stack.serve.process import pid_exists

__all__ = ["call", "claim", "cores", "give_back_cores", "lease", "record_path", "release",
           "serve", "status", "stop", "unclaim"]

START_WAIT_S = 30.0
REAP_EVERY_S = 1.0
ADOPT_EVERY_S = 30.0
QUIET_S = 900.0


def record_path() -> Path:
    """Where the running broker writes its pid, port and token."""
    return home.state("broker.json")


def log_path() -> Path:
    """Where a broker a client started writes its output."""
    return home.cache("logs", "broker.log")


class _Handler(socketserver.StreamRequestHandler):
    server: _Server

    def handle(self) -> None:
        try:
            reply = self.server.answer(json.loads(self.rfile.readline()))
        except (BrokerError, ServerFailed, ValueError, TypeError, KeyError) as exc:
            reply = {"ok": False, "error": str(exc)}
        self.wfile.write((json.dumps(reply) + "\n").encode("utf-8"))


class _Server(socketserver.ThreadingTCPServer):
    daemon_threads = True

    def __init__(self, broker: Broker) -> None:
        super().__init__((DEFAULT_HOST, 0), _Handler)
        self.broker = broker
        self.token = secrets.token_hex(16)

    def answer(self, body: dict[str, Any]) -> dict[str, Any]:
        if body.get("token") != self.token:
            return {"ok": False, "error": "wrong broker token"}
        op, pid = body.get("op"), int(body.get("pid") or 0)
        if op == "ping":
            return {"ok": True, "pid": os.getpid()}
        if op == "lease":
            ask = Ask.from_json(body)
            return {"ok": True, **self.broker.lease(ask, timeout=float(body["wait_s"])).as_dict()}
        if op == "release":
            return {"ok": True, "released": self.broker.release(str(body["lease"]))}
        if op == "status":
            return {"ok": True, **self.broker.snapshot()}
        if op == "stop":
            return {"ok": True, **self.broker.stop(int(body["port"]), force=bool(body.get("force")))}
        if op == "claim":
            return {"ok": True, **self.broker.claim(str(body["name"]), pid, body.get("info") or {},
                                                    timeout=float(body.get("wait_s") or 0.0))}
        if op == "cores":
            return {"ok": True, **self.broker.take_cores(pid, int(body["want"]))}
        if op == "give_back_cores":
            return {"ok": True, "released": self.broker.give_back_cores(str(body["lease"]))}
        if op == "unclaim":
            return {"ok": True, "released": self.broker.unclaim(str(body["name"]), pid)}
        return {"ok": False, "error": f"no such broker call: {op!r}"}


def serve(*, idle_s: float = IDLE_S, quiet_s: float = QUIET_S, say=say_out) -> int:
    """Run this machine's broker until a quit signal, until it has had nothing to supervise
    for ``quiet_s``, or until its record is gone. Returns 0, also when one is running."""
    try:
        with only_one(home.state("broker.lock"), wait=False, announce=say):
            broker = Broker(idle_s=idle_s)
            broker.say = lambda line: say(line, flush=True)
            adopted = broker.adopt()
            server = _Server(broker)
            write_json(record_path(), {"pid": os.getpid(), "port": server.server_address[1],
                                       "token": server.token, "started": time.time()})
            private_file(record_path())
            say(f"broker on {DEFAULT_HOST}:{server.server_address[1]} (pid {os.getpid()}), "
                f"{len(broker.servers)} of {len(adopted)} running server(s) taken in")
            done = threading.Event()
            threading.Thread(target=_upkeep, args=(broker, done, server, quiet_s),
                             daemon=True).start()
            on_quit(lambda *_: threading.Thread(target=server.shutdown).start())
            try:
                server.serve_forever()
            finally:
                done.set()
                server.server_close()
                if _record().get("pid") == os.getpid():
                    record_path().unlink(missing_ok=True)
    except Busy:
        say(f"a broker is already running (pid {_record().get('pid')})")
    return 0


def _upkeep(broker: Broker, done: threading.Event, server: _Server, quiet_s: float) -> None:
    """Reap every second, take in servers started since the last look every
    ``ADOPT_EVERY_S``, and stop the server once the broker's record is gone or it has had
    nothing to supervise for ``quiet_s``."""
    looked = busy = time.monotonic()
    while not done.wait(REAP_EVERY_S):
        broker.reap()
        now = time.monotonic()
        if now - looked >= ADOPT_EVERY_S:
            broker.adopt()
            looked = now
        if broker.supervising():
            busy = now
        why = ("its record is gone" if _record().get("pid") != os.getpid()
               else f"nothing to supervise for {quiet_s:.0f}s" if now - busy >= quiet_s else "")
        if why:
            broker.say(f"broker stopping: {why}")
            server.shutdown()
            return


def _record() -> dict[str, Any]:
    held = read_json(record_path(), {})
    return held if isinstance(held, dict) else {}


def _send(record: dict[str, Any], body: dict[str, Any], *, timeout: float | None) -> dict[str, Any]:
    with socket.create_connection((DEFAULT_HOST, int(record["port"])), timeout=5.0) as sock:
        sock.settimeout(timeout)
        sock.sendall((json.dumps({**body, "token": record.get("token")}) + "\n").encode("utf-8"))
        with sock.makefile("rb") as reply:
            line = reply.readline()
    if not line:
        raise BrokerError("the broker closed the connection without answering")
    return json.loads(line)


def _answering() -> dict[str, Any] | None:
    record = _record()
    if not record or not pid_exists(record.get("pid")):
        return None
    try:
        return record if _send(record, {"op": "ping"}, timeout=5.0).get("ok") else None
    except (OSError, ValueError, BrokerError):
        return None


def _reach(*, start: bool) -> dict[str, Any]:
    record = _answering()
    if record is not None:
        return record
    if not start:
        raise BrokerError("no broker is running on this machine")
    jobs.detach("ml_stack.serve.cli", ["broker"], log=log_path())
    deadline = time.monotonic() + START_WAIT_S
    while time.monotonic() < deadline:
        record = _answering()
        if record is not None:
            return record
        time.sleep(0.2)
    raise BrokerError(f"the broker did not come up within {START_WAIT_S:.0f}s; see {log_path()}")


def call(op: str, *, timeout: float | None = 30.0, start: bool = True,
         **fields: Any) -> dict[str, Any]:
    """Send ``op`` to the broker, starting it first when ``start``. Raises `BrokerError`
    with the broker's own words when it refuses."""
    body = {"op": op, "pid": os.getpid(), **fields}
    try:
        reply = _send(_reach(start=start), body, timeout=timeout)
    except ConnectionRefusedError:
        if not start:
            raise
        reply = _send(_reach(start=True), body, timeout=timeout)  # it quit after the ping
    if not reply.get("ok"):
        raise BrokerError(str(reply.get("error") or f"the broker refused {op}"))
    return reply


def lease(purpose: str, models: list[str] | tuple[str, ...], *, spec: dict[str, Any] | None = None,
          weight: int = 0, timeout: float = 600.0) -> Grant:
    """A server for ``purpose`` serving one of ``models``, held by this process until it
    calls `release` or ends. Waits up to ``timeout`` seconds in the queue; ``weight`` is
    the bytes it will take when the weights are not on disk to measure."""
    reply = call("lease", timeout=None, purpose=purpose, models=list(models), spec=spec or {},
                 weight=weight, label=_label(), wait_s=timeout)
    return Grant(**{k: reply[k] for k in ("lease", "purpose", "model", "port", "base_url",
                                          "shared")})


def _label() -> str:
    """This process as a person would recognise it in the queue."""
    return " ".join([Path(sys.argv[0]).name, *sys.argv[1:]])[:120] if sys.argv else ""


def release(lease_id: str) -> bool:
    """Let go of a lease. False when the broker held no such lease."""
    return bool(call("release", start=False, lease=lease_id)["released"])


def status(*, start: bool = False) -> dict[str, Any]:
    """The broker's servers, their holders, the queue and the claims."""
    return call("status", start=start)


def stop(port: int, *, force: bool = False) -> dict[str, Any]:
    """Stop the server on ``port``; refused while it is held, unless ``force``."""
    return call("stop", start=False, port=port, force=force)


def claim(name: str, info: dict[str, Any], *, timeout: float = 0.0) -> dict[str, Any]:
    """Hold ``name`` for this process, or learn who holds it (``granted`` says which)."""
    return call("claim", timeout=timeout + 30.0, name=name, info=info, wait_s=timeout)


def cores(want: int) -> dict[str, Any]:
    """Cores for this process to run tests on, out of what the running broker says is free.
    Raises `BrokerError` when no broker is running."""
    return call("cores", start=False, want=want)


def give_back_cores(lease: str) -> bool:
    """Let go of a core grant."""
    return bool(call("give_back_cores", start=False, lease=lease)["released"])


def unclaim(name: str) -> bool:
    """Let go of ``name`` if this process holds it."""
    return bool(call("unclaim", start=False, name=name)["released"])
