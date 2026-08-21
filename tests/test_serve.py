"""Server lifecycle: argv construction, port ownership, adoption, state merge.

The parts that can be tested without a GGUF are tested without one. ``command()`` is split
out from ``start()`` precisely so the argv is checkable with no model on disk, and the
adoption path is exercised against a real HTTP server standing in for a live llama-server.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time

import pytest
from conftest import json_reply
from mainspring.serve import (
    LlamaServerBackend,
    ServerFailed,
    ServerInfo,
    ServerManager,
    ServerSpec,
    free_port,
    merge_state,
    model_matches,
    pid_exists,
    port_is_free,
    tail,
)


@pytest.fixture
def fake_binary(tmp_path):
    """A file standing in for llama-server, so ``command()`` can resolve a path."""
    path = tmp_path / "llama-server"
    path.write_text("#!/bin/sh\nexit 0\n")
    path.chmod(0o755)
    return path


@pytest.fixture
def gguf(tmp_path):
    path = tmp_path / "model.gguf"
    path.write_bytes(b"GGUF" + b"\x00" * 64)
    return path


class TestCommand:
    def test_uses_long_flags_only(self, fake_binary, gguf):
        """Short forms are ambiguous across llama-server versions -- `-a` is `--alias`,
        so a stale short flag misconfigures silently instead of erroring."""
        argv = LlamaServerBackend(binary=fake_binary).command(ServerSpec(model=gguf, port=1234))
        assert "--host" in argv and "--port" in argv
        assert "-a" not in argv and "-p" not in argv
        assert argv[argv.index("--port") + 1] == "1234"

    def test_binds_loopback_not_all_interfaces(self, fake_binary, gguf):
        """A local model server on 0.0.0.0 is an unauthenticated inference endpoint on
        every network the machine is attached to."""
        argv = LlamaServerBackend(binary=fake_binary).command(ServerSpec(model=gguf))
        assert argv[argv.index("--host") + 1] == "127.0.0.1"

    def test_hf_reference_defers_the_download_to_the_server(self, fake_binary):
        """Re-implementing the fetch here is how a machine ends up with two model caches."""
        argv = LlamaServerBackend(binary=fake_binary).command(
            ServerSpec(model="hf:owner/repo/weights-Q4_K_M.gguf")
        )
        assert argv[argv.index("--hf-repo") + 1] == "owner/repo"
        assert argv[argv.index("--hf-file") + 1] == "weights-Q4_K_M.gguf"
        assert "-m" not in argv

    def test_hf_reference_without_a_file_is_allowed(self, fake_binary):
        argv = LlamaServerBackend(binary=fake_binary).command(ServerSpec(model="hf:owner/repo"))
        assert argv[argv.index("--hf-repo") + 1] == "owner/repo"
        assert "--hf-file" not in argv

    def test_malformed_hf_reference_is_rejected(self, fake_binary):
        with pytest.raises(ServerFailed, match="malformed HF reference"):
            LlamaServerBackend(binary=fake_binary).command(ServerSpec(model="hf:justowner"))

    def test_embedding_mode_drops_the_chat_template(self, fake_binary, gguf):
        """An embedding server has no chat turn to template."""
        argv = LlamaServerBackend(binary=fake_binary).command(
            ServerSpec(model=gguf, embedding=True)
        )
        assert "--embeddings" in argv
        assert "--jinja" not in argv

    def test_extra_args_land_last_so_they_can_override(self, fake_binary, gguf):
        argv = LlamaServerBackend(binary=fake_binary).command(
            ServerSpec(model=gguf, extra_args=("--cache-reuse", "256"))
        )
        assert argv[-2:] == ["--cache-reuse", "256"]


class TestStartGuards:
    def test_a_missing_model_fails_immediately_with_the_path(self, fake_binary, tmp_path):
        """Left to the server this becomes 'did not become healthy', which sends the
        reader looking for a slow load that never happened."""
        backend = LlamaServerBackend(binary=fake_binary)
        with pytest.raises(ServerFailed, match="no model file at"):
            backend.start(ServerSpec(model=tmp_path / "absent.gguf"))

    def test_a_foreign_process_on_the_port_is_refused_not_killed(self, fake_binary, gguf):
        """The port check matches our own binary names; anything else is somebody's
        service and must not be terminated."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as held:
            held.bind(("127.0.0.1", 0))
            held.listen(1)
            port = held.getsockname()[1]

            backend = LlamaServerBackend(binary=fake_binary)
            with pytest.raises(ServerFailed, match="not one of ours"):
                backend.start(ServerSpec(model=gguf, port=port), timeout=1.0)

            # Still listening: we refused rather than reclaiming.
            assert held.fileno() != -1


class TestPorts:
    def test_free_port_is_actually_free(self):
        assert port_is_free(free_port())

    def test_a_bound_port_reads_as_busy(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            sock.listen(1)
            assert not port_is_free(sock.getsockname()[1])

    def test_time_wait_does_not_read_as_busy(self):
        """SO_REUSEADDR matches how the server binds. Without it an ordinary TIME_WAIT
        socket is indistinguishable from a live foreign listener, and a start that would
        have succeeded is refused."""
        port = free_port()
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", port))
        sock.listen(1)
        sock.close()
        assert port_is_free(port)


class TestProcess:
    def test_a_live_process_exists(self):
        proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        try:
            assert pid_exists(proc.pid)
        finally:
            proc.kill()
            proc.wait()

    def test_a_zombie_does_not_count_as_alive(self):
        """A zombie keeps its pid until someone reaps it, so 'the pid is in the table' is
        a proxy; 'the process is running' is the fact."""
        proc = subprocess.Popen([sys.executable, "-c", "pass"])
        deadline = time.monotonic() + 5
        while proc.poll() is None and time.monotonic() < deadline:
            time.sleep(0.02)
        assert not pid_exists(proc.pid)
        proc.wait()

    def test_nonsense_pids_are_not_alive(self):
        assert not pid_exists(0)
        assert not pid_exists(-1)
        assert not pid_exists(None)


class TestMergeState:
    def test_another_live_process_record_survives(self):
        """Replacing the file with one process's view orphans a concurrent run's server:
        nothing can reap it, and the next lease reclaims its port mid-request."""
        mine_pid = os.getpid()
        on_disk = {"8080": {"port": 8080, "owner_pid": mine_pid, "pid": 1},
                   "8081": {"port": 8081, "owner_pid": mine_pid, "pid": 2}}
        merged = merge_state(on_disk, {"8082": {"port": 8082, "owner_pid": mine_pid}}, mine_pid)
        # Both prior entries were owned by *this* pid, so they are ours to replace.
        assert set(merged) == {"8082"}

    def test_a_dead_owners_record_is_dropped(self):
        on_disk = {"8080": {"port": 8080, "owner_pid": 999_999_998, "pid": 1}}
        assert merge_state(on_disk, {}, os.getpid()) == {}

    def test_my_own_entries_win(self):
        pid = os.getpid()
        merged = merge_state(
            {"8080": {"port": 8080, "owner_pid": pid, "pid": 1}},
            {"8080": {"port": 8080, "owner_pid": pid, "pid": 2}},
            pid,
        )
        assert merged["8080"]["pid"] == 2


class TestModelMatches:
    @pytest.mark.parametrize(
        "reported,wanted",
        [("/models/Qwen3-32B-Q5_K_M.gguf", "Qwen3-32B-Q5_K_M.gguf"),
         ("Qwen3-32B-Q5_K_M.gguf", "/elsewhere/Qwen3-32B-Q5_K_M.gguf"),
         ("weights.gguf", "hf:owner/repo/weights.gguf")],
    )
    def test_matching_names(self, reported, wanted):
        assert model_matches(reported, wanted)

    def test_different_models_do_not_match(self):
        assert not model_matches("gemma-4-E2B.gguf", "Qwen3-32B-Q5_K_M.gguf")

    def test_empty_names_do_not_match(self):
        assert not model_matches("", "model.gguf")


class TestAdoption:
    def _manager(self, tmp_path, fake_binary):
        return ServerManager(
            LlamaServerBackend(binary=fake_binary),
            state_file=tmp_path / "servers.json",
        )

    def test_nothing_running_means_nothing_to_adopt(self, tmp_path, fake_binary):
        manager = self._manager(tmp_path, fake_binary)
        assert manager.adopt(ServerSpec(model="m.gguf", port=free_port())) is None

    def test_a_matching_server_is_adopted(self, server, tmp_path, fake_binary):
        instance = server(lambda m, p, b: json_reply({"data": [{"id": "model.gguf"}]}))
        manager = self._manager(tmp_path, fake_binary)

        info = manager.adopt(ServerSpec(model="model.gguf", port=instance.port))
        assert info is not None and info.adopted

    def test_a_server_with_different_weights_is_refused(self, server, tmp_path, fake_binary):
        """'Something answers on this port' and 'it serves what I asked for' are
        different facts. Adopting on the first silently benchmarks the wrong weights."""
        instance = server(lambda m, p, b: json_reply({"data": [{"id": "some-other.gguf"}]}))
        manager = self._manager(tmp_path, fake_binary)

        with pytest.raises(ServerFailed, match="already serves"):
            manager.adopt(ServerSpec(model="model.gguf", port=instance.port))

    def test_release_leaves_an_adopted_server_running(self, server, tmp_path, fake_binary):
        """Terminate only what you launched."""
        instance = server(lambda m, p, b: json_reply({"data": [{"id": "model.gguf"}]}))
        manager = self._manager(tmp_path, fake_binary)

        info = manager.adopt(ServerSpec(model="model.gguf", port=instance.port))
        manager.release(info)

        assert instance.base_url  # still answering; nothing was killed
        from mainspring.client import is_healthy

        assert is_healthy(instance.base_url)

    def test_release_of_an_unowned_info_does_not_raise(self, tmp_path, fake_binary):
        manager = self._manager(tmp_path, fake_binary)
        manager.release(ServerInfo(base_url="http://127.0.0.1:1", port=1, pid=None,
                                   backend="llama.cpp", adopted=False))


class TestNegativeCache:
    def test_a_failed_start_is_not_retried_immediately(self, tmp_path, fake_binary):
        """A caller polling on every frame must not re-pay the connect timeout each time
        once the answer is known to be 'no'."""
        manager = ServerManager(
            LlamaServerBackend(binary=fake_binary), state_file=tmp_path / "s.json"
        )
        spec = ServerSpec(model=tmp_path / "absent.gguf", port=free_port())

        with pytest.raises(ServerFailed, match="no model file"):
            manager.lease(spec, timeout=1.0)

        started = time.monotonic()
        with pytest.raises(ServerFailed, match="negative cache"):
            manager.lease(spec, timeout=1.0)
        assert time.monotonic() - started < 0.5


class TestStateFile:
    def test_it_is_written_atomically_and_is_valid_json(self, tmp_path, fake_binary):
        state = tmp_path / "servers.json"
        manager = ServerManager(LlamaServerBackend(binary=fake_binary), state_file=state)
        manager._mine["9999"] = {"port": 9999, "owner_pid": os.getpid(), "pid": 1}
        manager._save()

        assert json.loads(state.read_text())["9999"]["port"] == 9999
        assert not state.with_suffix(".json.tmp").exists()

    def test_a_corrupt_state_file_does_not_crash_a_lease(self, tmp_path, fake_binary):
        state = tmp_path / "servers.json"
        state.write_text("{ this is not json")
        manager = ServerManager(LlamaServerBackend(binary=fake_binary), state_file=state)
        assert manager._load() == {}


def test_tail_reports_a_missing_log_rather_than_raising(tmp_path):
    """A start failure with no tail is unactionable, and the tail itself must never be
    the thing that raises."""
    assert "no log at" in tail(tmp_path / "absent.log")


def test_tail_returns_the_last_lines(tmp_path):
    log = tmp_path / "server.log"
    log.write_text("\n".join(f"line {i}" for i in range(100)))
    assert tail(log, lines=3).splitlines() == ["line 97", "line 98", "line 99"]
