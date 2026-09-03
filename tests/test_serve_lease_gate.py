"""A server cannot be started outside the manager, and it is on the record before it exists.

Adam: "how do we stop a server from being launched in the first place that isn't through
the manager" -- "this wouldn't happen in the first place if we ensured that it wasn't
possible to have an untracked server."
"""

import json

import pytest

from ml_stack.serve.backend import Lease, LlamaServerBackend, ServerFailed, ServerInfo, ServerSpec
from ml_stack.serve.manager import ServerManager
from ml_stack.serve.ports import free_port
from tests.conftest import fake_binary


def test_the_backend_launches_nothing_without_a_lease(tmp_path):
    binary = fake_binary(tmp_path)
    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"GGUF" + b"\x00" * 64)
    with pytest.raises(TypeError, match="lease"):
        LlamaServerBackend(binary=binary).start(ServerSpec(model=gguf, port=free_port()))


def test_a_lease_for_another_port_is_refused(tmp_path):
    binary = fake_binary(tmp_path)
    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"GGUF" + b"\x00" * 64)
    port = free_port()
    with pytest.raises(ServerFailed, match="starts only on the port its record names"):
        LlamaServerBackend(binary=binary).start(
            ServerSpec(model=gguf, port=port), timeout=1.0, preflight=False,
            check_flags=False, lease=Lease(port=port + 1, owner_pid=1, state_file="x"))


def test_the_record_exists_before_the_process_and_is_filled_or_forgotten_after(tmp_path):
    state = tmp_path / "servers.json"
    seen = {}

    class Backend:
        name = "fake"

        def start(self, spec, *, lease, timeout=300.0, **starting):
            held = json.loads(state.read_text())[str(spec.port)]
            seen["pending"] = dict(held)
            assert isinstance(lease, Lease) and lease.port == spec.port
            assert lease.state_file == str(state)
            if "fail" in str(spec.model):
                raise ServerFailed("did not come up")
            return ServerInfo(base_url=f"http://127.0.0.1:{spec.port}", port=spec.port,
                              pid=4242, backend="fake")

        def stop(self, info, *, grace_s=5.0):
            return None

    manager = ServerManager(backend=Backend(), state_file=state)
    port = free_port()
    info = manager.lease(ServerSpec(model="fine.gguf", port=port), roam=False, timeout=1.0)
    assert seen["pending"]["pending"] is True and seen["pending"]["pid"] is None, \
        "written down before the process existed"
    after = json.loads(state.read_text())[str(port)]
    assert after["pid"] == 4242 and not after.get("pending"), "the pid filled in after"
    assert info.port == port

    other = free_port()
    with pytest.raises(ServerFailed):
        manager.lease(ServerSpec(model="fail.gguf", port=other), roam=False, timeout=1.0)
    assert str(other) not in json.loads(state.read_text()), "a start that failed is forgotten"


class _Backend:
    name = "fake"

    def start(self, spec, *, lease, timeout=300.0, **starting):
        return ServerInfo(base_url=f"http://127.0.0.1:{spec.port}", port=spec.port,
                          pid=4242, backend="fake")

    def stop(self, info, *, grace_s=5.0):
        return None


def _sleeper():
    import subprocess
    import sys

    return subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])


def _ended(proc, within: float = 10.0) -> bool:
    import time

    deadline = time.monotonic() + within
    while proc.poll() is None and time.monotonic() < deadline:
        time.sleep(0.05)
    return proc.poll() is not None


def _serving(model: str, n_ctx: int = 4096, slots: int = 1):
    """A handler answering /health, /v1/models and /props the way llama-server does."""
    from tests.conftest import json_reply

    def handle(method, path, body):
        if path.startswith("/props"):
            return json_reply({"model_path": f"/models/{model}", "total_slots": slots,
                               "default_generation_settings": {"n_ctx": n_ctx}})
        return json_reply({"data": [{"id": model}]})

    return handle


def _orphan_record(state, port: int, pid: int, model: str = "fine.gguf") -> None:
    state.write_text(json.dumps({str(port): {
        "port": port, "pid": pid, "owner_pid": 999_999_998, "backend": "fake",
        "model": model}}))


def test_a_lease_stops_an_orphan_of_another_shape_before_starting(tmp_path, server):
    """A record whose leasing process has gone while its server runs on, serving a shape
    the lease cannot use: the lease stops that server, says so, and starts its own."""
    import os

    state = tmp_path / "servers.json"
    instance = server(_serving("other-8B-Q4_K_M.gguf"))
    orphan = _sleeper()
    try:
        _orphan_record(state, instance.port, orphan.pid, model="other-8B-Q4_K_M.gguf")
        said = []
        manager = ServerManager(backend=_Backend(), state_file=state)
        info = manager.lease(ServerSpec(model="fine.gguf", port=instance.port), roam=False,
                             timeout=1.0, say=said.append)
        assert _ended(orphan), "the orphaned server is still running"
        assert said and "orphaned" in said[0] and str(orphan.pid) in said[0]
        assert "stopping" in said[0] and "model:" in said[0]
        assert info.pid == 4242 and not info.adopted
        after = json.loads(state.read_text())[str(instance.port)]
        assert after["pid"] == 4242 and after["owner_pid"] == os.getpid()
    finally:
        if orphan.poll() is None:
            orphan.kill()
        orphan.wait()


def test_a_lease_adopts_an_orphan_of_the_same_shape_and_takes_it_over(tmp_path, server):
    """The page server restarts, the process that leased the 90 GB model goes with it,
    and the new one asks for the same shape on the same port: it is adopted, not
    reloaded, and the record's owner becomes the new leaser."""
    import os

    state = tmp_path / "servers.json"
    instance = server(_serving("fine.gguf"))
    orphan = _sleeper()
    try:
        _orphan_record(state, instance.port, orphan.pid)
        said = []
        manager = ServerManager(backend=_Backend(), state_file=state)
        info = manager.lease(ServerSpec(model="fine.gguf", port=instance.port), roam=False,
                             timeout=1.0, say=said.append)
        assert orphan.poll() is None, "the matching server is kept, not reloaded"
        assert info.adopted and info.pid == orphan.pid
        assert said and "adopted" in said[0] and "orphaned" in said[0]
        after = json.loads(state.read_text())[str(instance.port)]
        assert after["pid"] == orphan.pid and after["owner_pid"] == os.getpid(), \
            "ownership transferred to this process"
    finally:
        if orphan.poll() is None:
            orphan.kill()
        orphan.wait()


def test_a_lease_leaves_a_server_another_live_process_holds(tmp_path):
    import os

    state = tmp_path / "servers.json"
    port = free_port()
    held = _sleeper()
    try:
        state.write_text(json.dumps({str(port): {
            "port": port, "pid": held.pid, "owner_pid": os.getpid(), "backend": "fake",
            "model": "fine.gguf"}}))
        said = []
        ServerManager(backend=_Backend(), state_file=state).lease(
            ServerSpec(model="fine.gguf", port=port), roam=False, timeout=1.0,
            say=said.append)
        assert held.poll() is None, "a held server is not an orphan"
        assert not any("orphaned" in line for line in said)
    finally:
        if held.poll() is None:
            held.kill()
        held.wait()
