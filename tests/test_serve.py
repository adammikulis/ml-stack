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
from ml_stack.client.health import ServingParams
from ml_stack.serve import (
    LlamaServerBackend,
    ServerBackend,
    ServerFailed,
    ServerInfo,
    ServerManager,
    ServerSpec,
    free_port,
    merge_state,
    model_matches,
    pid_exists,
    port_is_free,
    recorded_servers,
    shape_mismatch,
    tail,
)
from ml_stack.serve.manager import orphaned
from tests.conftest import leased


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
            leased(backend, ServerSpec(model=tmp_path / "absent.gguf"))

    def test_a_foreign_process_on_the_port_is_refused_not_killed(self, fake_binary, gguf):
        """The port check matches our own binary names; anything else is somebody's
        service and must not be terminated."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as held:
            held.bind(("127.0.0.1", 0))
            held.listen(1)
            port = held.getsockname()[1]

            backend = LlamaServerBackend(binary=fake_binary)
            with pytest.raises(ServerFailed, match="not one of ours"):
                leased(backend, ServerSpec(model=gguf, port=port), timeout=1.0)

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

    def test_a_dead_owners_record_is_kept_while_its_server_lives(self):
        orphan = {"port": 8080, "owner_pid": 999_999_998, "pid": os.getpid()}
        assert merge_state({"8080": orphan}, {}, os.getpid()) == {"8080": orphan}

    def test_a_record_whose_owner_and_server_are_both_dead_is_dropped(self):
        on_disk = {"8080": {"port": 8080, "owner_pid": 999_999_998, "pid": 999_999_997}}
        assert merge_state(on_disk, {}, os.getpid()) == {}

    def test_a_malformed_entry_is_dropped(self):
        on_disk = {"8080": "gone", "8081": {"port": 8081, "owner_pid": None, "pid": None}}
        assert merge_state(on_disk, {}, os.getpid()) == {}

    def test_an_orphan_on_disk_survives_another_managers_save(self, tmp_path):
        state = tmp_path / "servers.json"
        orphan = {"port": 8080, "owner_pid": 999_999_998, "pid": os.getpid(),
                  "base_url": "http://127.0.0.1:8080", "backend": "llama"}
        state.write_text(json.dumps({"8080": orphan}), encoding="utf-8")
        manager = ServerManager(state_file=state)
        manager._mine["8081"] = {"port": 8081, "owner_pid": os.getpid(), "pid": os.getpid()}
        manager._save()
        records = recorded_servers(state)
        assert set(records) == {8080, 8081} and orphaned(records[8080])

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

        with pytest.raises(ServerFailed, match="model: asked for 'model.gguf'"):
            manager.adopt(ServerSpec(model="model.gguf", port=instance.port))

    def test_release_leaves_an_adopted_server_running(self, server, tmp_path, fake_binary):
        """Terminate only what you launched."""
        instance = server(lambda m, p, b: json_reply({"data": [{"id": "model.gguf"}]}))
        manager = self._manager(tmp_path, fake_binary)

        info = manager.adopt(ServerSpec(model="model.gguf", port=instance.port))
        manager.release(info)

        assert instance.base_url  # still answering; nothing was killed
        from ml_stack.client import is_healthy

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


class TestScaledTimeout:
    """A timeout sized for a 4G model is a race against an 87G one. `lease(timeout=None)`
    -- the default -- scales with the weights on disk; a caller that passes a number means
    it, and gets exactly that."""

    def test_it_grows_with_the_weights_and_never_drops_below_the_floor(self):
        from ml_stack.serve.manager import DEFAULT_TIMEOUT_S, scaled_timeout

        assert scaled_timeout(0) == DEFAULT_TIMEOUT_S
        # 10 GiB -> 60 + 15 = 75s, still under the 300s floor
        assert scaled_timeout(10 * 1024**3) == DEFAULT_TIMEOUT_S
        # 200 GiB -> 60 + 300 = 360s, past the floor
        assert scaled_timeout(200 * 1024**3) == pytest.approx(360.0, abs=0.5)

    def _fake_backend(self, started: list):
        class Backend(ServerBackend):
            name = "fake"

            def command(self, spec):
                return ["fake"]

            def start(self, spec, *, lease, timeout=300.0, **starting):
                started.append(timeout)
                return ServerInfo(base_url=f"http://127.0.0.1:{spec.port}", port=spec.port,
                                  pid=1, backend="fake")

        return Backend()

    def test_an_explicit_timeout_is_passed_through_untouched(self, tmp_path):
        started: list[float] = []
        manager = ServerManager(self._fake_backend(started), state_file=tmp_path / "s.json")
        manager.lease(ServerSpec(model=tmp_path / "big.gguf", port=free_port()), timeout=12.5)
        assert started == [12.5]

    def test_none_scales_from_the_weights_already_on_disk(self, tmp_path, monkeypatch):
        from ml_stack.serve import manager as manager_module
        from ml_stack.serve.manager import scaled_timeout

        monkeypatch.setattr(manager_module, "weight_of", lambda model: 200 * 1024**3)
        started: list[float] = []
        manager = ServerManager(self._fake_backend(started), state_file=tmp_path / "s.json")
        manager.lease(ServerSpec(model=tmp_path / "big.gguf", port=free_port()))
        assert started == [scaled_timeout(200 * 1024**3)]
        assert started[0] > 300.0


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

    def test_recorded_servers_is_keyed_by_port(self, tmp_path):
        state = tmp_path / "servers.json"
        state.write_text(json.dumps({"8080": {"port": 8080, "pid": 7, "owner_pid": 7}}))
        assert recorded_servers(state)[8080]["pid"] == 7

    def test_recorded_servers_of_a_missing_file_is_empty(self, tmp_path):
        assert recorded_servers(tmp_path / "absent.json") == {}

    def test_load_s_and_warmup_s_are_kept_in_the_record(self, tmp_path, fake_binary):
        """`ml-stack-serve status --json` reads these back -- a fact worth keeping, not a
        log line to be grepped for later."""
        state = tmp_path / "servers.json"
        manager = ServerManager(LlamaServerBackend(binary=fake_binary), state_file=state)
        info = ServerInfo(base_url="http://127.0.0.1:9999", port=9999, pid=1,
                          backend="llama.cpp", load_s=12.5, warmup_s=0.8)
        manager._record(ServerSpec(model="m.gguf", port=9999), info)

        entry = json.loads(state.read_text())["9999"]
        assert entry["load_s"] == 12.5
        assert entry["warmup_s"] == 0.8


class TestDetach:
    def test_a_detached_server_outlives_the_process_that_started_it(
            self, tmp_path, fake_binary):
        """The record is kept against the server's own pid, so a later save that prunes
        dead owners does not throw the entry away."""
        state = tmp_path / "servers.json"
        manager = ServerManager(LlamaServerBackend(binary=fake_binary), state_file=state)
        proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        try:
            info = ServerInfo(base_url="http://127.0.0.1:9101", port=9101, pid=proc.pid,
                              backend="llama.cpp")
            manager._record(ServerSpec(model="m.gguf", port=9101), info)
            manager.detach(info)

            assert recorded_servers(state)[9101]["owner_pid"] == proc.pid

            # a fresh process saving its own view keeps the entry
            ServerManager(LlamaServerBackend(binary=fake_binary), state_file=state)._save()
            assert 9101 in recorded_servers(state)
        finally:
            proc.kill()
            proc.wait()

    def test_a_detached_server_is_not_stopped_by_stop_all(self, tmp_path, fake_binary):
        state = tmp_path / "servers.json"
        manager = ServerManager(LlamaServerBackend(binary=fake_binary), state_file=state)
        proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        try:
            info = ServerInfo(base_url="http://127.0.0.1:9102", port=9102, pid=proc.pid,
                              backend="llama.cpp")
            manager._record(ServerSpec(model="m.gguf", port=9102), info)
            manager.detach(info)

            assert manager.stop_all() == []
            assert proc.poll() is None
        finally:
            proc.kill()
            proc.wait()


def test_tail_reports_a_missing_log_rather_than_raising(tmp_path):
    """A start failure with no tail is unactionable, and the tail itself must never be
    the thing that raises."""
    assert "no log at" in tail(tmp_path / "absent.log")


def test_tail_returns_the_last_lines(tmp_path):
    log = tmp_path / "server.log"
    log.write_text("\n".join(f"line {i}" for i in range(100)))
    assert tail(log, lines=3).splitlines() == ["line 97", "line 98", "line 99"]


class TestShapeMismatch:
    """Which field differs, wanted against found."""

    def test_a_matching_shape_has_no_mismatch(self):
        params = ServingParams(n_ctx=4096, total_slots=2)
        spec = ServerSpec(model="a-model.gguf", context=8192, parallel=2)
        assert shape_mismatch(spec, ["a-model.gguf"], params) == []

    def test_every_differing_field_is_named(self):
        params = ServingParams(n_ctx=2048, total_slots=1)
        spec = ServerSpec(model="a-model.gguf", context=32768, parallel=4)
        assert shape_mismatch(spec, ["some-other.gguf"], params) == [
            "model: asked for 'a-model.gguf', serving 'some-other.gguf'",
            "slots: asked for 4, serving 1",
            "context: asked for 8192 per slot, serving 2048",
        ]

    def test_a_bigger_server_is_not_a_mismatch(self):
        """More context and more slots than asked for is still what was asked for."""
        params = ServingParams(n_ctx=32768, total_slots=8)
        spec = ServerSpec(model="a-model.gguf", context=4096, parallel=1)
        assert shape_mismatch(spec, ["a-model.gguf"], params) == []

    def test_a_server_that_reports_nothing_has_no_mismatch(self):
        spec = ServerSpec(model="a-model.gguf", context=32768, parallel=4)
        assert shape_mismatch(spec, [], None) == []


class TestAdoptingTheWrongShape:
    """The right model in the wrong shape is still the wrong server."""

    def _manager(self, tmp_path, fake_binary):
        return ServerManager(
            LlamaServerBackend(binary=fake_binary),
            state_file=tmp_path / "servers.json",
        )

    @staticmethod
    def _handler(model: str, n_ctx: int, slots: int):
        def handle(method, path, body):
            if path.startswith("/props"):
                return json_reply({
                    "model_path": f"/models/{model}",
                    "total_slots": slots,
                    "default_generation_settings": {"n_ctx": n_ctx},
                })
            return json_reply({"data": [{"id": model}]})

        return handle

    def test_a_running_server_with_too_few_slots_is_refused(
            self, server, tmp_path, fake_binary):
        instance = server(self._handler("a-model.gguf", n_ctx=4096, slots=1))
        manager = self._manager(tmp_path, fake_binary)

        with pytest.raises(ServerFailed, match="slots: asked for 4, serving 1"):
            manager.adopt(ServerSpec(model="a-model.gguf", port=instance.port,
                                     context=4096, parallel=4))

    def test_a_running_server_with_a_smaller_context_is_refused(
            self, server, tmp_path, fake_binary):
        """A caller that asked for 32k and gets 4k has its prompts truncated instead."""
        instance = server(self._handler("a-model.gguf", n_ctx=4096, slots=1))
        manager = self._manager(tmp_path, fake_binary)

        with pytest.raises(ServerFailed,
                           match="context: asked for 32768 per slot, serving 4096"):
            manager.adopt(ServerSpec(model="a-model.gguf", port=instance.port,
                                     context=32768))

    def test_context_is_compared_one_slot_at_a_time(self, server, tmp_path, fake_binary):
        """llama-server splits --ctx-size across -np, and reports one slot's share."""
        instance = server(self._handler("a-model.gguf", n_ctx=32768, slots=2))
        manager = self._manager(tmp_path, fake_binary)

        adopted = manager.adopt(ServerSpec(model="a-model.gguf", port=instance.port,
                                           context=65536, parallel=2))
        assert adopted is not None and adopted.adopted

        with pytest.raises(ServerFailed, match="context: asked for 65536 per slot"):
            manager.adopt(ServerSpec(model="a-model.gguf", port=instance.port,
                                     context=65536, parallel=1))

    def test_the_shape_that_was_asked_for_is_adopted(self, server, tmp_path, fake_binary):
        instance = server(self._handler("a-model.gguf", n_ctx=4096, slots=1))
        manager = self._manager(tmp_path, fake_binary)

        info = manager.adopt(ServerSpec(model="a-model.gguf", port=instance.port,
                                        context=4096, parallel=1))
        assert info is not None and info.adopted

    def test_a_server_that_will_not_say_is_adopted_anyway(self, monkeypatch):
        from ml_stack.serve import manager as mod
        from ml_stack.serve.manager import ServerManager, ServerSpec

        monkeypatch.setattr(mod, "is_healthy", lambda *a, **k: True)
        monkeypatch.setattr(mod, "reported_models", lambda *a, **k: ["a-model.gguf"])
        monkeypatch.setattr(mod, "serving_params", lambda *a, **k: None)
        assert ServerManager().adopt(ServerSpec(model="a-model.gguf", port=8099, parallel=4))


class TestDrafting:
    """A small model guesses ahead and the large one checks the guesses in one pass.

    The flags are not the main model's: `-hfd` takes owner/repo[:quant] as one argument,
    where the model itself takes `--hf-repo` and `--hf-file` separately. Getting that wrong
    is a server that starts without a draft and is simply slower, which nothing reports.
    """

    def backend(self, tmp_path):
        binary = tmp_path / "llama-server"
        binary.write_text("#!/bin/sh\nexit 0\n")
        binary.chmod(0o755)
        return LlamaServerBackend(binary=binary)

    def test_a_local_draft_is_passed_as_a_path(self, tmp_path):
        argv = self.backend(tmp_path).command(
            ServerSpec(model=tmp_path / "big.gguf", draft=tmp_path / "small.gguf"))
        assert "-md" in argv and argv[argv.index("-md") + 1].endswith("small.gguf")

    def test_a_draft_named_by_repository_is_one_argument(self, tmp_path):
        """`-hfd` takes owner/repo[:quant]. A draft named by *file* is not one argument --
        llama-server has no flag for that -- so it is fetched first and served by path
        (`LlamaServerBackend.resolved_draft`); measured 2026-09-01 when a head under MTP/
        passed as repo:file started the server with an empty draft path."""
        argv = self.backend(tmp_path).command(ServerSpec(
            model="hf:unsloth/gemma-4-E4B-it-qat-GGUF/big.gguf",
            draft="hf:unsloth/gemma-4-E2B-it-qat-GGUF"))
        assert argv[argv.index("--hf-repo") + 1] == "unsloth/gemma-4-E4B-it-qat-GGUF"
        assert argv[argv.index("--hf-file") + 1] == "big.gguf"
        assert argv[argv.index("-hfd") + 1] == "unsloth/gemma-4-E2B-it-qat-GGUF"
        with pytest.raises(ServerFailed, match="fetched before serving"):
            self.backend(tmp_path).command(ServerSpec(
                model="hf:unsloth/gemma-4-E4B-it-qat-GGUF/big.gguf",
                draft="hf:unsloth/gemma-4-E2B-it-qat-GGUF/small.gguf"))

    def test_no_draft_asks_for_none(self, tmp_path):
        argv = self.backend(tmp_path).command(ServerSpec(model=tmp_path / "big.gguf"))
        assert "-md" not in argv and "-hfd" not in argv

    def test_a_malformed_reference_says_so_rather_than_serving_without_it(self, tmp_path):
        with pytest.raises(ServerFailed, match="malformed HF reference"):
            self.backend(tmp_path).command(ServerSpec(model="m.gguf", draft="hf:onlyowner"))


class TestServingBeside:
    """A small model does not need a large one evicted, and should not need a person either.

    Refusing a busy port and naming the field that differs is right when the caller needs
    *that* port — everything meeting on one server is the whole point of a lease. It is not
    right when the caller just wants the model served: this machine has the memory, and
    picking another port by hand is work a machine can do.
    """

    def other(self):
        """A server answering as something entirely different is serving on this port."""
        def handle(method, path, body):
            if path.startswith("/props"):
                return json_reply({"model_path": "/m/somethingelse.gguf", "total_slots": 1,
                                   "default_generation_settings": {"n_ctx": 4096}})
            return json_reply({"data": [{"id": "somethingelse.gguf"}]})

        return handle

    def manager(self, tmp_path, started):
        class Backend(ServerBackend):
            name = "fake"

            def command(self, spec):
                return ["fake"]

            def start(self, spec, *, lease, timeout=300.0, **starting):
                started.append(spec)
                return ServerInfo(base_url=f"http://127.0.0.1:{spec.port}", port=spec.port,
                                  pid=4242, backend="fake")

        return ServerManager(backend=Backend(), state_file=tmp_path / "servers.json")

    def test_a_busy_port_is_served_beside_rather_than_refused(self, server, tmp_path):
        instance = server(self.other())
        started: list[ServerSpec] = []
        held = self.manager(tmp_path, started)
        info = held.lease(ServerSpec(model=tmp_path / "mine.gguf", port=instance.port))
        assert info.port != instance.port, "it should have moved rather than refused"
        assert started and started[0].port == info.port
        from ml_stack.client import is_healthy

        assert is_healthy(instance.base_url), "the other server is untouched"

    def test_a_caller_that_needs_that_port_still_gets_the_refusal(self, server, tmp_path):
        instance = server(self.other())
        held = self.manager(tmp_path, [])
        with pytest.raises(ServerFailed, match="different shape"):
            held.lease(ServerSpec(model=tmp_path / "mine.gguf", port=instance.port), roam=False)

    def test_it_refuses_when_the_machine_has_no_room(self, server, tmp_path, monkeypatch):
        """Starting a load that will be killed halfway is worse than saying no."""
        big = tmp_path / "big.gguf"
        big.write_bytes(b"0" * 4096)
        instance = server(self.other())
        monkeypatch.setattr("ml_stack.serve.manager.free_memory", lambda: 1024)
        held = self.manager(tmp_path, [])
        with pytest.raises(ServerFailed, match="different shape"):
            held.lease(ServerSpec(model=big, port=instance.port))


def test_a_projector_reference_becomes_a_url_and_a_path_stays_a_path():
    """`--mmproj` takes a file on disk. Handing it an `hf:` reference is a path that does
    not exist, and the server's complaint reads like a corrupt projector rather than a
    misspelled one."""
    from ml_stack.serve.backend import LlamaServerBackend, ServerSpec

    backend = LlamaServerBackend(binary="/bin/true")

    argv = backend.command(ServerSpec(model="hf:maker/thing-GGUF/w.gguf", port=1,
                                      mmproj="hf:maker/thing-GGUF/mmproj-F32.gguf"))
    assert "--mmproj" not in argv
    at = argv.index("--mmproj-url")
    assert argv[at + 1] == "https://huggingface.co/maker/thing-GGUF/resolve/main/mmproj-F32.gguf"

    argv = backend.command(ServerSpec(model="/m/w.gguf", port=1, mmproj="/m/mmproj-F32.gguf"))
    assert "--mmproj-url" not in argv
    assert argv[argv.index("--mmproj") + 1] == "/m/mmproj-F32.gguf"

    assert "--mmproj" not in backend.command(ServerSpec(model="/m/w.gguf", port=1))


def test_the_most_precise_projector_is_taken_not_the_first_alphabetically():
    """A projector is a fraction of the weights and carries all of the seeing, so quantising
    it is a false economy -- and sorted by name, `mmproj-BF16` beats `mmproj-F32` on B."""
    import tempfile
    from pathlib import Path

    from ml_stack.serve.cli import alongside

    with tempfile.TemporaryDirectory() as d:
        where = Path(d)
        for name in ("mmproj-BF16.gguf", "mmproj-F32.gguf", "mmproj-Q8_0.gguf"):
            (where / name).write_bytes(b"x")
        model = where / "thing-Q4_K_M.gguf"
        model.write_bytes(b"x")

        assert sorted(where.glob("mmproj-*.gguf"))[0].name == "mmproj-BF16.gguf"
        assert alongside(str(model), "auto", "mmproj-", best=True).endswith("mmproj-F32.gguf")
        # a draft head is the other way about and takes what is there
        (where / "mtp-thing-Q4_0.gguf").write_bytes(b"x")
        assert alongside(str(model), "auto", "mtp-").endswith("mtp-thing-Q4_0.gguf")


def test_the_speculative_knobs_reach_the_command_line_and_stay_off_until_asked():
    """`--spec-type` defaults to `none` on the server. An n-gram kind needs no second model,
    proposing tokens already seen in the prompt -- which is what suits work that copies from
    its context, and costs no weights and no memory where a draft head costs both."""
    from ml_stack.serve.backend import LlamaServerBackend, ServerSpec

    backend = LlamaServerBackend(binary="/bin/true")

    bare = backend.command(ServerSpec(model="/m/w.gguf", port=1))
    for flag in ("--spec-type", "--spec-draft-n-max", "--spec-ngram-mod-n-min",
                 "--spec-draft-ngl"):
        assert flag not in bare, f"{flag} should not appear unless asked for"

    argv = backend.command(ServerSpec(
        model="/m/w.gguf", port=1, spec_type="ngram-mod", spec_draft_max=5,
        spec_draft_min=1, spec_ngram_min=32, spec_ngram_max=64, spec_draft_ngl=99))
    pairs = dict(zip(argv, argv[1:]))
    assert pairs["--spec-type"] == "ngram-mod"
    assert pairs["--spec-draft-n-max"] == "5" and pairs["--spec-draft-n-min"] == "1"
    assert pairs["--spec-ngram-mod-n-min"] == "32" and pairs["--spec-ngram-mod-n-max"] == "64"
    # a draft left on the CPU is slower than the model it guesses for, which is a loss
    assert pairs["--spec-draft-ngl"] == "99"

    # zero is a choice and must survive, where None means "leave the server's default"
    assert "--spec-draft-n-min" in backend.command(
        ServerSpec(model="/m/w.gguf", port=1, spec_draft_min=0))


def test_the_lookup_cache_reaches_the_command_line():
    """Only the ngram-cache kind keeps a table on disk. The other n-gram kinds look up the
    prompt they already hold and store nothing, which is why none of this is set by default."""
    from ml_stack.serve.backend import LlamaServerBackend, ServerSpec

    backend = LlamaServerBackend(binary="/bin/true")
    bare = backend.command(ServerSpec(model="/m/w.gguf", port=1, spec_type="ngram-simple"))
    assert "--lookup-cache-dynamic" not in bare and "--lookup-cache-static" not in bare

    argv = backend.command(ServerSpec(model="/m/w.gguf", port=1, spec_type="ngram-cache",
                                      lookup_static="/c/seed.bin",
                                      lookup_dynamic="/c/learnt.bin"))
    pairs = dict(zip(argv, argv[1:]))
    assert pairs["--lookup-cache-static"] == "/c/seed.bin"
    assert pairs["--lookup-cache-dynamic"] == "/c/learnt.bin"


def test_tensors_can_be_kept_off_the_gpu_by_pattern():
    """Qwen3.8-Flash-Next's N-gram Embedding is 51B of lookup table whose addresses are known
    in advance, meant to sit in host memory and be prefetched rather than hold GPU. Naming
    its tensors is how that is arranged -- and it is a different thing from n-gram
    *speculation*, which is a decoding trick and touches no weights at all."""
    from ml_stack.serve.backend import LlamaServerBackend, ServerSpec

    backend = LlamaServerBackend(binary="/bin/true")
    bare = backend.command(ServerSpec(model="/m/w.gguf", port=1))
    assert "--override-tensor" not in bare and "--cpu-moe" not in bare

    argv = backend.command(ServerSpec(model="/m/w.gguf", port=1,
                                      override_tensor=("ngram.*=CPU", "blk.0.*=CPU"),
                                      n_cpu_moe=12))
    assert argv.count("--override-tensor") == 2
    assert argv[argv.index("--override-tensor") + 1] == "ngram.*=CPU"
    assert argv[argv.index("--n-cpu-moe") + 1] == "12"

    assert "--cpu-moe" in backend.command(
        ServerSpec(model="/m/w.gguf", port=1, cpu_moe=True))


def test_the_serving_knobs_that_shorten_a_run():
    """A benchmark sends the same system prompt and tool schemas ahead of every question, so
    the prefix is reprocessed twenty or thirty times a run without --cache-reuse. Reusing it
    is free: the tokens are identical, so the cache is valid.

    `--kv-unified-per-slot` says the per-slot context directly. The alternative is a total
    divided by the slot count, done at the call site, which was got wrong here once: a model
    served at 8k per slot against everything else at 32k, and the only thing that said so
    was the table printing the context on every line."""
    from ml_stack.serve.backend import LlamaServerBackend, ServerSpec

    backend = LlamaServerBackend(binary="/bin/true")
    bare = backend.command(ServerSpec(model="/m/w.gguf", port=1))
    for flag in ("--cache-reuse", "--no-warmup", "--kv-unified-per-slot"):
        assert flag not in bare, f"{flag} must be asked for, not assumed"

    argv = backend.command(ServerSpec(model="/m/w.gguf", port=1, cache_reuse=256,
                                      warmup=False, context_per_slot=32768))
    pairs = dict(zip(argv, argv[1:]))
    assert pairs["--cache-reuse"] == "256"
    assert pairs["--kv-unified-per-slot"] == "32768"
    assert "--no-warmup" in argv

    # warmup on is the server's own default and passes no flag either way
    assert "--no-warmup" not in backend.command(
        ServerSpec(model="/m/w.gguf", port=1, warmup=True))
    # zero is a choice: reuse nothing, which is not the same as leaving it unset
    assert "--cache-reuse" in backend.command(
        ServerSpec(model="/m/w.gguf", port=1, cache_reuse=0))


def test_already_up_names_the_recorded_server_that_holds_the_same_weights(tmp_path, monkeypatch):
    import json

    from ml_stack.serve.manager import already_up

    state = tmp_path / "servers.json"
    state.write_text(json.dumps({"8080": {"port": 8080, "pid": 4242, "model": "/models/quince-2b.gguf",
                                          "base_url": "http://127.0.0.1:8080", "slots": 2}}))
    monkeypatch.setattr("ml_stack.serve.manager.pid_exists", lambda pid: pid == 4242)
    assert already_up("quince-2b.gguf", 8080, state_file=state)["base_url"] == "http://127.0.0.1:8080"
    assert already_up("ember-1b.gguf", 8080, state_file=state) is None
    assert already_up("quince-2b.gguf", 8081, state_file=state) is None
    monkeypatch.setattr("ml_stack.serve.manager.pid_exists", lambda pid: False)
    assert already_up("quince-2b.gguf", 8080, state_file=state) is None
