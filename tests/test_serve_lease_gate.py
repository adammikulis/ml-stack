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


def test_a_lease_stops_the_orphan_on_its_port_before_starting(tmp_path):
    """A record whose leasing process has gone while its server runs on: the lease stops
    that server, says so, and starts its own rather than adopting it."""
    import os

    state = tmp_path / "servers.json"
    port = free_port()
    orphan = _sleeper()
    try:
        state.write_text(json.dumps({str(port): {
            "port": port, "pid": orphan.pid, "owner_pid": 999_999_998, "backend": "fake",
            "model": "fine.gguf"}}))
        said = []
        manager = ServerManager(backend=_Backend(), state_file=state)
        info = manager.lease(ServerSpec(model="fine.gguf", port=port), roam=False,
                             timeout=1.0, say=said.append)
        assert _ended(orphan), "the orphaned server is still running"
        assert said and "orphaned" in said[0] and str(orphan.pid) in said[0]
        assert info.pid == 4242 and not info.adopted
        after = json.loads(state.read_text())[str(port)]
        assert after["pid"] == 4242 and after["owner_pid"] == os.getpid()
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
