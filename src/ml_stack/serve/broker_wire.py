"""The broker behind a loopback socket, and the calls that reach it.

One JSON object per line each way. `serve` runs the machine's one broker -- it holds
``broker.lock`` -- and writes its port and token to ``broker.json``, readable by this user
alone. Every client call finds the broker there and starts one when none answers, so no
caller starts it by hand.
"""

from __future__ import annotations

import dataclasses
import json
import os
import secrets
import socket
import socketserver
import threading
import time
from pathlib import Path
from typing import Any

from ml_stack import home, jobs
from ml_stack.files import read_json, write_json
from ml_stack.lock import Busy, only_one
from ml_stack.log import say as say_out
from ml_stack.platform import on_quit, private_file
from ml_stack.serve import guarded
from ml_stack.serve.backend import LlamaServerBackend, ServerFailed, ServerInfo, ServerSpec
from ml_stack.serve.broker import IDLE_S, Ask, Broker, BrokerError, Grant, who
from ml_stack.serve.events import Caller, Growth
from ml_stack.serve.leases import lease_file
from ml_stack.serve.ports import DEFAULT_HOST
from ml_stack.serve.process import pid_exists

__all__ = ["call", "claim", "cores", "give_back_cores", "lease", "record_path", "release",
           "serve", "status", "stop", "unclaim"]

LOCAL_ENV = "ML_STACK_BROKER_LOCAL"
"""Set to ``1`` to run the broker inside the process instead of reaching the machine's."""

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
        except (BrokerError, ServerFailed, ValueError, TypeError, KeyError, OSError,
                RuntimeError) as exc:
            reply = {"ok": False, "error": str(exc), "kind": type(exc).__name__}
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
        if op == "start":
            info = self.broker.start(
                spec_from(body["spec"]), Caller(pid=pid, label=str(body.get("label") or "")),
                timeout=body.get("timeout_s"), options=body.get("options") or {})
            return {"ok": True, **info_dict(info)}
        if op == "drop":
            self.broker.drop(ServerInfo(**{**info_fields(body["info"]), "process": None}),
                             grace_s=float(body.get("grace_s") or 5.0))
            return {"ok": True}
        if op == "escalate":
            info = self.broker.escalate(
                spec_from(body["spec"]),
                Growth(int(body.get("add_slots") or 1), body.get("room"), body.get("timeout_s"),
                       bool(body.get("anyway"))),
                Caller(pid=pid), options=body.get("options") or {})
            return {"ok": True, **info_dict(info)}
        if op == "detach":
            self.broker.detach(ServerInfo(**info_fields(body["info"])))
            return {"ok": True}
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


_INFO_FIELDS = ("base_url", "port", "pid", "backend", "adopted", "load_s", "warmup_s", "lease")


def info_dict(info: ServerInfo) -> dict[str, Any]:
    """A server as it crosses the wire."""
    return {**{k: getattr(info, k) for k in _INFO_FIELDS},
            "log_path": str(info.log_path) if info.log_path else None}


def info_fields(body: dict[str, Any]) -> dict[str, Any]:
    """The `ServerInfo` arguments in ``body``."""
    out = {k: body[k] for k in _INFO_FIELDS if k in body}
    if body.get("log_path"):
        out["log_path"] = Path(body["log_path"])
    return out


def spec_to_json(spec: ServerSpec) -> dict[str, Any]:
    """A spec as it crosses the wire."""
    return {f.name: str(v) if isinstance(v, Path) else v
            for f in dataclasses.fields(spec) if f.name != "process"
            for v in [getattr(spec, f.name)]}


def spec_from(body: dict[str, Any]) -> ServerSpec:
    """The spec ``spec_to_json`` wrote; a field a spec has no name for is refused."""
    names = {f.name for f in dataclasses.fields(ServerSpec)} - {"process"}
    unknown = sorted(set(body) - names)
    if unknown:
        raise ValueError(f"not server settings: {', '.join(unknown)}")
    return ServerSpec(**{k: tuple(v) if isinstance(v, list) else v for k, v in body.items()})


class RemoteBroker:
    """The machine's broker, reached over its socket, for a `ServerManager` to start
    servers through."""

    def __init__(self, backend: LlamaServerBackend) -> None:
        self.options = {"backend": backend.options()}

    def start(self, spec: ServerSpec, caller: Caller | None = None, *,
              timeout: float | None = None, options: dict[str, Any] | None = None) -> ServerInfo:
        """A server for ``spec`` from the broker, which waits for memory to start one."""
        reply = call("start", timeout=None, label=who(), spec=spec_to_json(spec),
                     timeout_s=timeout, options={**self.options, **(options or {})})
        info = ServerInfo(**info_fields(reply))
        if caller is not None and caller.on_event is not None:
            caller.on_event({"event": "ready", "port": info.port, "adopted": info.adopted})
        return info

    def drop(self, info: ServerInfo, *, grace_s: float = 5.0) -> None:
        """Let go of the lease ``info`` was granted under."""
        call("drop", info=info_dict(info), grace_s=grace_s, start=False)

    def escalate(self, spec: ServerSpec, growth: Growth, caller: Caller | None = None) -> ServerInfo:
        """Grow the server on ``spec.port`` as ``growth`` says."""
        reply = call("escalate", timeout=None, spec=spec_to_json(spec),
                     add_slots=growth.add_slots, room=growth.room, timeout_s=growth.timeout,
                     anyway=growth.anyway, options=dict(self.options))
        return ServerInfo(**info_fields(reply))

    def detach(self, info: ServerInfo) -> None:
        """Record the server under its own pid."""
        call("detach", info=info_dict(info), start=False)


def broker_for(manager: Any) -> Any:
    """The broker ``manager`` starts servers through: the machine's, over its socket, when
    the manager keeps the machine's records and starts llama.cpp the ordinary way; else one
    that runs in this process over the manager's own records."""
    backend = manager.backend
    if (os.environ.get(LOCAL_ENV) != "1" and type(backend) is LlamaServerBackend
            and Path(manager.state_file) == lease_file()):
        return RemoteBroker(backend)
    return Broker(manager)


def serve(*, idle_s: float = IDLE_S, quiet_s: float = QUIET_S, say=say_out) -> int:
    """Run this machine's broker until a quit signal, until it has had nothing to supervise
    for ``quiet_s``, or until its record is gone. Returns 0, also when one is running."""
    try:
        with only_one(home.state("broker.lock"), wait=False, announce=say):
            broker = Broker(idle_s=idle_s)
            broker.say = lambda line: say(line, flush=True)
            scanner = guarded.arm(broker.manager, broker)
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
                scanner.stop()
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


def running() -> int | None:
    """The pid of the machine's broker when it answers, else ``None``."""
    record = _answering()
    return int(record["pid"]) if record else None


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
        raise _error(str(reply.get("kind") or ""))(
            str(reply.get("error") or f"the broker refused {op}"))
    return reply


def _error(kind: str) -> type[Exception]:
    """The exception class named ``kind`` among the server failures; else `BrokerError`."""
    todo: list[type[Exception]] = [ServerFailed]
    seen: dict[str, type[Exception]] = {}
    while todo:
        one = todo.pop()
        seen[one.__name__] = one
        todo.extend(one.__subclasses__())
    return seen.get(kind, BrokerError)


def lease(purpose: str, models: list[str] | tuple[str, ...], *, spec: dict[str, Any] | None = None,
          weight: int = 0, timeout: float = 600.0) -> Grant:
    """A server for ``purpose`` serving one of ``models``, held by this process until it
    calls `release` or ends. Waits up to ``timeout`` seconds in the queue; ``weight`` is
    the bytes it will take when the weights are not on disk to measure."""
    reply = call("lease", timeout=None, purpose=purpose, models=list(models), spec=spec or {},
                 weight=weight, label=who(), wait_s=timeout)
    return Grant(**{k: reply[k] for k in ("lease", "purpose", "model", "port", "base_url",
                                          "shared")})


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
