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
