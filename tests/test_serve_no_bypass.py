"""A model server is started, reused and talked to through the broker and nothing else.

The first half reads the source: it finds every place that spawns a process, constructs the
proof a backend needs to launch, reaches the private start, or opens a connection to a
server, and fails on one it was not told about. The second half drives `ServerManager` and
shows each public entry reaching the broker.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from ml_stack.serve import broker_wire
from ml_stack.serve.backend import LlamaServerBackend, ServerInfo, ServerSpec
from ml_stack.serve.broker import Broker
from ml_stack.serve.manager import ServerManager
from ml_stack.testing.fakes import FakeBackend

SRC = Path(__file__).resolve().parent.parent / "src" / "ml_stack"

LAUNCHERS = {"serve/backend.py", "serve/mlx_tree.py", "serve/python_engines.py"}
"""The modules that call `backend.launch`, the one function that starts a model server
process. Each calls it from a `start` that takes the manager's `Lease`."""

NAMES_A_SERVER_BINARY = {
    "serve/backend.py": "llama-server, started with a Lease",
    "serve/build_platform.py": "compiles llama.cpp",
    "serve/build_source.py": "compiles llama.cpp",
    "setup.py": "looks for the binary and installs",
}
"""Every module that starts a process and also names the llama-server binary, and why."""

PRIVATE = ("_start_server", "_escalate", "_detach", "_stop_server")
"""`ServerManager` methods only the manager and the broker may call."""


def parsed() -> dict[str, ast.Module]:
    return {str(path.relative_to(SRC)): ast.parse(path.read_text(encoding="utf-8"))
            for path in sorted(SRC.rglob("*.py"))}


def calls(tree: ast.Module) -> list[ast.Call]:
    return [node for node in ast.walk(tree) if isinstance(node, ast.Call)]


def called_name(call: ast.Call) -> str:
    func = call.func
    return func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")


def dotted(call: ast.Call) -> str:
    func, parts = call.func, []
    while isinstance(func, ast.Attribute):
        parts.append(func.attr)
        func = func.value
    if isinstance(func, ast.Name):
        parts.append(func.id)
    return ".".join(reversed(parts))


SPAWN_CALLS = {"subprocess.Popen", "subprocess.run", "subprocess.call", "subprocess.check_call",
               "subprocess.check_output", "os.execv", "os.execve", "os.execvp", "os.spawnv",
               "os.posix_spawn", "asyncio.create_subprocess_exec",
               "asyncio.create_subprocess_shell", "Popen"}


def where(found: set[str]) -> str:
    return ", ".join(sorted(found)) or "nowhere"


def functions(tree: ast.Module) -> list[ast.FunctionDef]:
    return [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]


def test_a_server_process_is_only_launched_by_a_backend_holding_a_lease():
    found: set[str] = set()
    for name, tree in parsed().items():
        for fn in functions(tree):
            launches = [c for c in calls(ast.Module(body=fn.body, type_ignores=[]))
                        if called_name(c) == "launch" and any(k.arg == "log_path" for k in c.keywords)]
            if launches:
                found.add(name)
                if name != "serve/backend.py" or fn.name != "launch":
                    names = {called_name(c) for c in calls(ast.Module(body=fn.body, type_ignores=[]))}
                    assert "claim_port" in names, f"{name}:{fn.name} launches with no claim on the port"
    assert found <= LAUNCHERS, where(found - LAUNCHERS)


def test_only_the_known_modules_start_a_process_and_name_the_server_binary():
    found = set()
    for name, tree in parsed().items():
        spawns = any(dotted(c) in SPAWN_CALLS for c in calls(tree))
        names = any(isinstance(n, ast.Constant) and n.value in ("llama-server", "llama_server")
                    for n in ast.walk(tree)) or any(
            called_name(c) in ("require_binary", "find_binary") for c in calls(tree))
        if spawns and names:
            found.add(name)
    assert found == set(NAMES_A_SERVER_BINARY), (
        f"{where(found ^ set(NAMES_A_SERVER_BINARY))}: a module that starts a process and names "
        "llama-server starts it through ServerManager and the broker, or is listed with its reason")


def test_a_backend_is_only_started_by_the_manager():
    found = {name for name, tree in parsed().items()
             for c in calls(tree)
             if called_name(c) == "start" and any(k.arg == "lease" for k in c.keywords)}
    assert found == {"serve/manager.py"}, where(found)


def test_the_proof_a_backend_needs_is_only_made_by_the_manager():
    found = {name for name, tree in parsed().items()
             if name != "serve/backend.py"
             for c in calls(tree) if called_name(c) == "Lease"}
    assert found == {"serve/manager.py"}, where(found)


def test_the_private_starts_are_only_reached_by_the_manager_and_the_broker():
    found = set()
    for name, tree in parsed().items():
        if name in ("serve/manager.py", "serve/broker.py"):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr in PRIVATE:
                found.add(f"{name}:{node.attr}")
    assert not found, where(found)


def test_a_broker_is_only_built_where_the_machines_is_chosen():
    found = {name for name, tree in parsed().items()
             for c in calls(tree) if called_name(c) == "Broker"}
    assert found == {"serve/broker_wire.py"}, where(found)


def test_a_connection_to_a_server_goes_through_the_request_queue():
    """`ml_stack.http` queues every generation request; the one other module that opens
    a connection to a model server's port takes its turn itself."""
    opens = {name for name, tree in parsed().items()
             if any(dotted(c) in ("urllib.request.urlopen", "urlopen") for c in calls(tree))}
    assert opens == {"http.py", "fleet/api.py"}, where(opens)
    api = parsed()["fleet/api.py"]
    assert any(dotted(c) == "gate.turn" for c in calls(api))
    http = parsed()["http.py"]
    assert sum(1 for c in calls(http) if dotted(c) == "gate.turn") == 1


class Recording:
    """A broker that records what it was asked and starts nothing."""

    def __init__(self) -> None:
        self.asked: list[tuple[str, object]] = []

    def start(self, spec, **kwargs):
        self.asked.append(("start", spec))
        return ServerInfo(base_url="http://127.0.0.1:1", port=spec.port, pid=None,
                          backend="fake", lease="lease-1")

    def drop(self, info, **kwargs):
        self.asked.append(("drop", info))

    def escalate(self, spec, **kwargs):
        self.asked.append(("escalate", spec))
        return ServerInfo(base_url="http://127.0.0.1:1", port=spec.port, pid=None, backend="fake")

    def detach(self, info):
        self.asked.append(("detach", info))


def test_every_public_entry_of_the_manager_reaches_the_broker(tmp_path):
    broker = Recording()
    held = ServerManager(FakeBackend(), state_file=tmp_path / "servers.json", broker=broker)
    spec = ServerSpec(model="m.gguf", port=1)

    info = held.lease(spec)
    held.escalate(spec)
    held.detach(info)
    held.release(info)
    assert [name for name, _ in broker.asked] == ["start", "escalate", "detach", "drop"]

    held.lease(spec)
    held.stop_all()
    assert [name for name, _ in broker.asked][-2:] == ["start", "drop"]


def test_a_manager_that_keeps_the_machines_records_uses_the_machines_broker(tmp_path, monkeypatch):
    monkeypatch.delenv(broker_wire.LOCAL_ENV)
    assert isinstance(ServerManager().broker, broker_wire.RemoteBroker)
    assert isinstance(ServerManager(LlamaServerBackend(binary="llama-server")).broker,
                      broker_wire.RemoteBroker)
    own = ServerManager(FakeBackend(), state_file=tmp_path / "servers.json")
    assert isinstance(own.broker, Broker)


def test_no_flag_to_lease_skips_the_broker(tmp_path):
    import inspect

    flags = set(inspect.signature(ServerManager.lease).parameters)
    assert not {"broker", "direct", "bypass", "unmanaged"} & flags
    broker = Recording()
    held = ServerManager(FakeBackend(), state_file=tmp_path / "servers.json", broker=broker)
    with pytest.raises(TypeError):
        held.lease(ServerSpec(model="m.gguf", port=1), direct=True)  # type: ignore[call-arg]
