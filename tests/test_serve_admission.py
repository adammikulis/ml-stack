"""A server starts when the servers up and this one fit the memory this machine allows.

The backends here start a real child process per server and put a real HTTP server on the
port, so the registry records live pids, `release` stops a real process, and a leaked server
is a process that is still there.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from ml_stack import limits
from ml_stack.serve import admission
from ml_stack.serve.backend import LlamaServerBackend, ServerBackend, ServerInfo, ServerSpec
from ml_stack.serve.leases import recorded_servers
from ml_stack.serve.manager import ServerManager
from ml_stack.serve.ports import free_port
from ml_stack.serve.process import pid_exists
from ml_stack.testing.fakes import FakeLlamaServer, Served, fake_llama_binary
from ml_stack.testing.registry import record_server

GIB = 1024 ** 3
SRC = str(Path(__file__).resolve().parent.parent / "src")


class ServedBackend(ServerBackend):
    """Starts a child process and a llama-server look-alike on the port asked for."""

    name = "fake"

    def __init__(self) -> None:
        self.started: list[ServerSpec] = []
        self.fakes: list[FakeLlamaServer] = []
        self.processes: list[subprocess.Popen] = []

    def command(self, spec: ServerSpec) -> list[str]:
        return ["sleep", "120"]

    def start(self, spec: ServerSpec, *, lease, timeout: float = 300.0, **starting) -> ServerInfo:
        self.started.append(spec)
        process = subprocess.Popen(["sleep", "120"])
        self.processes.append(process)
        slots = max(1, int(spec.parallel or 1))
        self.fakes.append(FakeLlamaServer(Served(
            model=Path(str(spec.model)).name, context=int(spec.context), slots=slots),
            port=spec.port))
        return ServerInfo(base_url=f"http://127.0.0.1:{spec.port}", port=spec.port,
                          pid=process.pid, backend=self.name, process=process)

    def close(self) -> None:
        for fake in self.fakes:
            fake.close()
        for process in self.processes:
            process.kill()
            process.wait()


@pytest.fixture
def backend():
    made = ServedBackend()
    yield made
    made.close()


@pytest.fixture
def machine(monkeypatch):
    """A machine that allows ten GiB, and starts that wait for memory give up after a second."""
    limits.changed(memory_bytes=10 * GIB)
    monkeypatch.setenv(admission.ENV_WAIT, "1")


def weights(tmp_path: Path, name: str, gib: float) -> Path:
    path = tmp_path / name
    with path.open("wb") as handle:
        handle.truncate(int(gib * GIB))
    return path


def manager(tmp_path: Path, backend: ServerBackend) -> ServerManager:
    return ServerManager(backend, state_file=tmp_path / "servers.json")


def spec(path: Path, **fields) -> ServerSpec:
    return ServerSpec(model=path, port=free_port(), context=4096, **fields)


def test_servers_that_fit_run_together_and_a_third_that_does_not_is_refused(
        tmp_path, backend, machine):
    told: list[str] = []
    first, second, third = (weights(tmp_path, f"m{n}.gguf", 3.6) for n in range(3))
    held = manager(tmp_path, backend)
    held.say = told.append

    a = held.lease(spec(first))
    assert not any("yellow" in line for line in told), "one server on ten GiB is green"
    b = held.lease(spec(second))
    assert any("yellow" in line for line in told), "8.2 of 10 GiB is said to be yellow"
    assert a.port != b.port and pid_exists(a.pid) and pid_exists(b.pid)

    began = time.monotonic()
    with pytest.raises(admission.AdmissionRefused) as why:
        held.lease(spec(third))
    said = str(why.value)
    assert 0.9 <= time.monotonic() - began < 5.0, "it waited for memory before refusing"
    assert "red" in said and "m0.gguf" in said and "m1.gguf" in said
    assert len(backend.started) == 2
    assert sorted(recorded_servers(tmp_path / "servers.json")) == sorted([a.port, b.port])


def test_a_start_that_would_be_red_goes_ahead_when_memory_comes_free(tmp_path, backend,
                                                                      machine, monkeypatch):
    monkeypatch.setenv(admission.ENV_WAIT, "20")
    first, second, third = (weights(tmp_path, f"m{n}.gguf", 3.6) for n in range(3))
    held = manager(tmp_path, backend)
    a = held.lease(spec(first))
    held.lease(spec(second))
    threading.Timer(0.8, lambda: held.release(a)).start()

    began = time.monotonic()
    c = held.lease(spec(third))
    assert time.monotonic() - began >= 0.7
    assert pid_exists(c.pid) and len(backend.started) == 3


def test_dead_records_do_not_count_against_the_memory(tmp_path, backend, machine):
    dead = subprocess.Popen(["true"])
    dead.wait()
    state = tmp_path / "servers.json"
    record_server(state, free_port(), model="big.gguf", pid=dead.pid, owner_pid=dead.pid,
                  est_bytes=9 * GIB)
    record_server(state, free_port(), model="starting.gguf", pid=None, owner_pid=dead.pid,
                  pending=True, est_bytes=9 * GIB)

    info = manager(tmp_path, backend).lease(spec(weights(tmp_path, "m.gguf", 3.6)))
    assert pid_exists(info.pid)


def test_a_server_left_by_a_process_that_has_gone_is_stopped_to_make_room(
        tmp_path, backend, machine):
    left = subprocess.Popen(["sleep", "120"])
    gone = subprocess.Popen(["true"])
    gone.wait()
    record_server(tmp_path / "servers.json", free_port(), model="leaked.gguf", pid=left.pid,
                  owner_pid=gone.pid, est_bytes=9 * GIB)
    told: list[str] = []
    held = manager(tmp_path, backend)
    held.say = told.append
    try:
        info = held.lease(spec(weights(tmp_path, "m.gguf", 3.6)))
        left.wait(timeout=10)
        assert left.returncode is not None, "the leaked server was stopped"
        assert pid_exists(info.pid)
        assert any("has exited" in line and str(left.pid) in line for line in told)
    finally:
        left.kill()
        left.wait()


def test_a_server_whose_owner_is_alive_is_never_stopped_for_room(tmp_path, backend, machine):
    held_by_someone = subprocess.Popen(["sleep", "120"])
    record_server(tmp_path / "servers.json", free_port(), model="held.gguf",
                  pid=held_by_someone.pid, owner_pid=os.getpid(), est_bytes=9 * GIB)
    try:
        with pytest.raises(admission.AdmissionRefused):
            manager(tmp_path, backend).lease(spec(weights(tmp_path, "m.gguf", 3.6)))
        assert held_by_someone.poll() is None
    finally:
        held_by_someone.kill()
        held_by_someone.wait()


def test_a_second_lease_for_the_same_model_uses_the_server_already_up(tmp_path, backend,
                                                                       machine):
    path = weights(tmp_path, "m.gguf", 1)
    first = manager(tmp_path, backend).lease(spec(path))
    second = manager(tmp_path, backend).lease(spec(path))
    assert second.port == first.port and second.adopted
    assert len(backend.started) == 1


@pytest.mark.parametrize("asked", [{"context": 16384}, {"parallel": 4}, {"embedding": True}])
def test_a_server_that_does_not_fit_the_ask_is_not_reused(tmp_path, backend, machine, asked):
    path = weights(tmp_path, "m.gguf", 1)
    first = manager(tmp_path, backend).lease(spec(path))
    wanted = ServerSpec(model=path, port=free_port(), **{"context": 4096, **asked})
    second = manager(tmp_path, backend).lease(wanted)
    assert second.port != first.port
    assert len(backend.started) == 2


def test_another_model_is_never_reused(tmp_path, backend, machine):
    first = manager(tmp_path, backend).lease(spec(weights(tmp_path, "a.gguf", 1)))
    second = manager(tmp_path, backend).lease(spec(weights(tmp_path, "b.gguf", 1)))
    assert second.port != first.port and len(backend.started) == 2


def test_two_threads_asking_for_one_model_start_it_once(tmp_path, backend, machine):
    """The second asks while the first is still loading: it waits for that server."""
    started = backend.start

    def slowly(*args, **kwargs):
        time.sleep(1.0)
        return started(*args, **kwargs)

    backend.start = slowly
    path = weights(tmp_path, "m.gguf", 1)
    got: list[ServerInfo] = []
    threads = [threading.Thread(target=lambda: got.append(manager(tmp_path, backend).lease(
        spec(path)))) for _ in range(2)]
    for thread in threads:
        thread.start()
        time.sleep(0.2)
    for thread in threads:
        thread.join(timeout=30)
    assert len(backend.started) == 1
    assert len({info.port for info in got}) == 1 and sorted(i.adopted for i in got) == [False, True]


def test_a_lease_does_not_wait_for_a_different_model_that_is_loading(tmp_path, backend,
                                                                       machine):
    started = backend.start
    together = threading.Barrier(2)

    def slowly(*args, **kwargs):
        together.wait(timeout=10)
        return started(*args, **kwargs)

    backend.start = slowly
    paths = [weights(tmp_path, f"m{n}.gguf", 1) for n in range(2)]
    threads = [threading.Thread(target=lambda path=path: manager(tmp_path, backend).lease(
        spec(path))) for path in paths]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
    assert len(backend.started) == 2
    assert not together.broken, "both model starts entered before either completed"


def test_two_processes_that_do_not_fit_together_do_not_both_start(tmp_path, machine,
                                                                    monkeypatch):
    """The second process is refused while the first, which has been killed, leaves its
    server running; then that leaked server is stopped and the second one starts."""
    binary = fake_llama_binary(tmp_path)
    first = weights(tmp_path, "first.gguf", 6)
    second = weights(tmp_path, "second.gguf", 6)
    code = ("import sys, time\n"
            "import ml_stack.serve.unmanaged as u\n"
            "u.every_server = lambda: []\n"
            "from ml_stack.serve import LlamaServerBackend, ServerManager, ServerSpec, free_port\n"
            f"m = ServerManager(LlamaServerBackend(binary={str(binary)!r}))\n"
            "i = m.lease(ServerSpec(model=sys.argv[1], port=free_port(), context=512), "
            "preflight=False, timeout=60)\n"
            "print(i.pid, flush=True)\n"
            "time.sleep(120)\n")
    env = {**os.environ, "PYTHONPATH": SRC}
    holder = subprocess.Popen([sys.executable, "-c", code, str(first)], env=env,
                              stdout=subprocess.PIPE, text=True)
    assert holder.stdout is not None
    server_pid = int(holder.stdout.readline())
    mine = ServerManager(LlamaServerBackend(binary=binary))
    try:
        with pytest.raises(admission.AdmissionRefused, match="red"):
            mine.lease(ServerSpec(model=second, port=free_port(), context=512),
                       preflight=False, timeout=60)
        assert pid_exists(server_pid)

        holder.kill()
        holder.wait(timeout=10)
        info = mine.lease(ServerSpec(model=second, port=free_port(), context=512),
                          preflight=False, timeout=60)
        deadline = time.monotonic() + 10
        while pid_exists(server_pid) and time.monotonic() < deadline:
            time.sleep(0.1)
        assert not pid_exists(server_pid), "the server the dead process left was stopped"
        assert pid_exists(info.pid)
    finally:
        holder.kill()
        mine.stop_all()


def test_the_rating_follows_the_thresholds():
    assert [admission.rate(share, 100) for share in (0, 79, 80, 94, 95, 200)] == [
        "green", "green", "yellow", "yellow", "red", "red"]
    assert admission.rate(10**12, 0) == "green", "a machine that will not say its memory is not refused"


def test_admission_the_estimator_and_the_bars_share_one_pair_of_thresholds():
    import json

    from ml_stack import ui
    from ml_stack.serve import estimate
    from ml_stack.ui import verdict

    shipped = json.loads((ui.assets_dir() / "verdict.json").read_text(encoding="utf-8"))
    wanted = (shipped["yellow_at"], shipped["red_at"])
    assert wanted == (admission.YELLOW_AT, admission.RED_AT)
    assert wanted == (estimate.YELLOW_AT, estimate.RED_AT)
    for share in (0.0, 0.5, 0.79, 0.8, 0.9, 0.94, 0.95, 1.2):
        assert (admission.rate(share * 100, 100) == estimate.rating(share)
                == verdict.verdict_of(share, 1.0)), share


@pytest.mark.parametrize("field, asked", [
    ("cache_type_k", "q8_0"), ("cache_type_v", "q8_0"),
    ("spec_type", "draft-mtp"), ("draft", "/models/mtp-qwen.gguf"), ("mtp", True),
    ("chat_template_file", "/templates/shared.jinja"),
])
def test_reuse_requires_recorded_semantic_settings(field, asked):
    from dataclasses import replace

    spec = replace(ServerSpec(model="qwen.gguf"), **{field: asked})
    assert not admission.compatible(spec, {}, [])
    assert not admission.compatible(spec, {field: "wrong"}, [])
    assert admission.compatible(spec, {field: asked}, [])


def test_compatible_draft_paths_can_name_the_same_installed_file(tmp_path):
    draft = tmp_path / "mtp-qwen.gguf"
    draft.write_bytes(b"GGUF")
    alias = tmp_path / "matching-head.gguf"
    alias.symlink_to(draft)
    spec = ServerSpec(model="qwen.gguf", draft=alias)
    assert admission.compatible(spec, {"draft": str(draft)}, [])


def test_chat_and_coding_share_matching_cache_but_not_different_cache(tmp_path, backend, machine):
    path = weights(tmp_path, "qwen.gguf", 1)
    managed = manager(tmp_path, backend)
    first = managed.lease(spec(path, cache_type_k="q8_0", cache_type_v="q8_0"))
    second = managed.lease(spec(path, cache_type_k="q8_0", cache_type_v="q8_0"))
    assert second.port == first.port and second.adopted
    assert len(backend.started) == 1
    third = managed.lease(spec(path, cache_type_k="q4_0", cache_type_v="q4_0"))
    assert third.port != first.port
    assert len(backend.started) == 2
    record = recorded_servers(tmp_path / "servers.json")[first.port]
    assert record["cache_type_k"] == record["cache_type_v"] == "q8_0"
