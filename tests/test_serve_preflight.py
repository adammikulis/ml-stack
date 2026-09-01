"""Everything worth knowing about a model before a process is started for it.

A benchmark that loads five models, some 87G, pays for a missing shard, an architecture
the build cannot read, or a flag a release renamed at the far end of the load rather than
the start of one. These tests write real, tiny GGUF files by hand -- the format is simple
enough that no fixture needs the real weights, the real binary, or the network to prove the
checks read it correctly. Nothing here downloads anything or touches a GPU; the load-timing
tests spawn a tiny real HTTP server standing in for llama-server, over a real loopback
socket, the same way the rest of this suite prefers a real server to a mocked transport.
"""

from __future__ import annotations

import struct
import subprocess
from pathlib import Path

import pytest
from ml_stack.serve import backend as backend_module
from ml_stack.serve import preflight
from ml_stack.serve.backend import LlamaServerBackend, ServerSpec
from ml_stack.serve.preflight import Preflight, PreflightFailed, read_gguf_header, shard_names


def write_gguf(path: Path, metadata: dict, *, tensor_count: int = 0) -> Path:
    """A real, minimal GGUF v3 file: magic, version, counts, one key/value pair per
    metadata item -- ints as uint32, floats as float32, strings as strings -- and no
    tensors, because nothing under test reads them."""

    def kv(name: str, value: object) -> bytes:
        head = struct.pack("<Q", len(name.encode())) + name.encode()
        if isinstance(value, bool):
            return head + struct.pack("<I", 7) + struct.pack("<?", value)
        if isinstance(value, int):
            return head + struct.pack("<I", 4) + struct.pack("<I", value)
        if isinstance(value, float):
            return head + struct.pack("<I", 6) + struct.pack("<f", value)
        if isinstance(value, str):
            encoded = value.encode()
            return head + struct.pack("<I", 8) + struct.pack("<Q", len(encoded)) + encoded
        raise TypeError(f"unsupported metadata type: {type(value)}")

    body = (b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", tensor_count)
           + struct.pack("<Q", len(metadata)))
    for name, value in metadata.items():
        body += kv(name, value)
    path.write_bytes(body)
    return path


LLAMA_META = {
    "general.architecture": "llama",
    "llama.block_count": 32,
    "llama.attention.head_count_kv": 8,
    "llama.attention.key_length": 128,
}


# Enough of a real --help to answer every flag `command()` emits for a bare ServerSpec, so
# the pre-existing flag check (which nothing here is testing) does not itself refuse.
FULL_HELP = (
    "-m,    --model FNAME                    model path\n"
    "-c,    --ctx-size N                     size of the prompt context\n"
    "-ngl,  --gpu-layers, --n-gpu-layers N   number of layers to store in VRAM\n"
    "-fa,   --flash-attn [on|off|auto]       set Flash Attention use\n"
    "       --host HOST                      ip address to listen on\n"
    "       --port PORT                      port to listen on\n"
    "       --jinja                          use jinja template for chat\n"
)


def fake_binary(tmp_path: Path, *, help_text: str = "-m, --model FNAME  model path\n") -> Path:
    path = tmp_path / "llama-server"
    path.write_text("#!/bin/sh\nif [ \"$1\" = --help ]; then cat <<'HELP'\n"
                    + help_text + "HELP\nexit 0\nfi\nexit 0\n")
    path.chmod(0o755)
    return path


@pytest.fixture(autouse=True)
def _fresh_flag_cache(monkeypatch):
    monkeypatch.setattr(backend_module, "_FLAGS", {})


class TestReadGGUFHeader:
    def test_it_reads_strings_ints_and_floats(self, tmp_path):
        gguf = write_gguf(tmp_path / "m.gguf", {
            "general.architecture": "llama", "llama.block_count": 32,
            "general.sampling.temperature": 0.7, "general.file_type": True,
        })
        meta = read_gguf_header(gguf)
        assert meta["general.architecture"] == "llama"
        assert meta["llama.block_count"] == 32
        assert meta["general.sampling.temperature"] == pytest.approx(0.7, abs=1e-6)
        assert meta["general.file_type"] is True

    def test_it_stops_before_the_tensor_list(self, tmp_path):
        """A tensor list would need shapes and offsets this writer never emits; if the
        reader tried to walk it, this would raise instead of returning cleanly."""
        gguf = write_gguf(tmp_path / "m.gguf", {"general.architecture": "llama"},
                          tensor_count=9)
        assert read_gguf_header(gguf)["general.architecture"] == "llama"

    def test_a_file_with_no_gguf_magic_is_rejected(self, tmp_path):
        not_gguf = tmp_path / "m.gguf"
        not_gguf.write_bytes(b"not a gguf file at all")
        with pytest.raises(ValueError, match="not a GGUF file"):
            read_gguf_header(not_gguf)


class TestShardNames:
    def test_an_unsharded_name_is_its_own_answer(self):
        assert shard_names("model-Q4_K_M.gguf") == ["model-Q4_K_M.gguf"]

    def test_every_shard_is_named_from_the_first(self):
        assert shard_names("model-00001-of-00003.gguf") == [
            "model-00001-of-00003.gguf",
            "model-00002-of-00003.gguf",
            "model-00003-of-00003.gguf",
        ]

    def test_a_directory_prefix_is_kept(self):
        assert shard_names("UD-Q4_K_XL/thing-00001-of-00002.gguf") == [
            "UD-Q4_K_XL/thing-00001-of-00002.gguf",
            "UD-Q4_K_XL/thing-00002-of-00002.gguf",
        ]


class TestLocalShards:
    def test_a_single_file_that_exists_and_is_not_empty_is_complete(self, tmp_path, monkeypatch):
        gguf = write_gguf(tmp_path / "model.gguf", LLAMA_META)
        import ml_stack.setup as setup_module
        monkeypatch.setattr(setup_module, "_arches", lambda binary: {"llama"})
        report = Preflight(ServerSpec(model=gguf), binary=fake_binary(tmp_path))
        shards = next(c for c in report.checks if c.name == "shards")
        assert shards.ok
        assert report.weights_bytes == gguf.stat().st_size

    def test_a_missing_shard_is_named(self, tmp_path):
        first = tmp_path / "model-00001-of-00002.gguf"
        write_gguf(first, LLAMA_META)
        # the second shard is never written
        report = Preflight(ServerSpec(model=first), binary=fake_binary(tmp_path))
        shards = next(c for c in report.checks if c.name == "shards")
        assert not shards.ok
        assert "model-00002-of-00002.gguf" in shards.detail
        assert not report.ok

    def test_an_empty_shard_counts_as_missing(self, tmp_path):
        first = tmp_path / "model-00001-of-00002.gguf"
        second = tmp_path / "model-00002-of-00002.gguf"
        write_gguf(first, LLAMA_META)
        second.write_bytes(b"")
        report = Preflight(ServerSpec(model=first), binary=fake_binary(tmp_path))
        shards = next(c for c in report.checks if c.name == "shards")
        assert not shards.ok
        assert second.name in shards.detail


class TestArchitecture:
    def test_an_architecture_the_build_reads_passes(self, tmp_path, monkeypatch):
        import ml_stack.setup as setup_module

        monkeypatch.setattr(setup_module, "_arches", lambda binary: {"llama", "gemma4"})
        gguf = write_gguf(tmp_path / "model.gguf", LLAMA_META)
        report = Preflight(ServerSpec(model=gguf), binary=fake_binary(tmp_path))
        arch = next(c for c in report.checks if c.name == "architecture")
        assert arch.ok and arch.detail == "llama"

    def test_an_architecture_the_build_does_not_read_fails(self, tmp_path, monkeypatch):
        import ml_stack.setup as setup_module

        monkeypatch.setattr(setup_module, "_arches", lambda binary: {"gemma4", "qwen3moe"})
        gguf = write_gguf(tmp_path / "model.gguf", LLAMA_META)
        report = Preflight(ServerSpec(model=gguf), binary=fake_binary(tmp_path))
        arch = next(c for c in report.checks if c.name == "architecture")
        assert not arch.ok
        assert "llama" in arch.detail and not report.ok

    def test_a_build_whose_architectures_could_not_be_read_gets_no_opinion(
            self, tmp_path, monkeypatch):
        """Same philosophy as `flags_of`: an unknown build is given no opinion, not told
        it supports nothing."""
        import ml_stack.setup as setup_module

        monkeypatch.setattr(setup_module, "_arches", lambda binary: set())
        gguf = write_gguf(tmp_path / "model.gguf", LLAMA_META)
        report = Preflight(ServerSpec(model=gguf), binary=fake_binary(tmp_path))
        arch = next(c for c in report.checks if c.name == "architecture")
        assert arch.ok

    def test_a_file_with_no_architecture_key_is_unknown_not_failed(self, tmp_path, monkeypatch):
        """The fact could not be read, which is not the same as reading it and finding it
        wrong -- same philosophy as an unknown build getting no opinion on its flags. This
        is also the shape of the stand-in GGUF (`b\"GGUF\" + 64 zero bytes`) used across
        this whole test suite wherever only *a file exists* matters, not its contents."""
        import ml_stack.setup as setup_module

        monkeypatch.setattr(setup_module, "_arches", lambda binary: {"llama"})
        gguf = write_gguf(tmp_path / "model.gguf", {"llama.block_count": 32})
        report = Preflight(ServerSpec(model=gguf), binary=fake_binary(tmp_path))
        arch = next(c for c in report.checks if c.name == "architecture")
        assert arch.ok

    def test_a_file_that_is_not_a_gguf_at_all_is_unknown_not_failed(self, tmp_path, monkeypatch):
        import ml_stack.setup as setup_module

        monkeypatch.setattr(setup_module, "_arches", lambda binary: {"llama"})
        not_gguf = tmp_path / "model.gguf"
        not_gguf.write_bytes(b"not a real gguf file")
        report = Preflight(ServerSpec(model=not_gguf), binary=fake_binary(tmp_path))
        arch = next(c for c in report.checks if c.name == "architecture")
        assert arch.ok


class TestFit:
    def test_the_kv_estimate_matches_the_formula(self, tmp_path, monkeypatch):
        import ml_stack.setup as setup_module

        monkeypatch.setattr(setup_module, "_arches", lambda binary: {"llama"})
        gguf = write_gguf(tmp_path / "model.gguf", LLAMA_META)
        context = 4096
        report = Preflight(ServerSpec(model=gguf, context=context),
                           binary=fake_binary(tmp_path))
        # 2 x n_layer x n_kv_heads x head_dim x context x bytes_per_element(f16=2)
        expected = 32 * 8 * 128 * context * (2.0 + 2.0)
        assert report.kv_estimate_bytes == int(expected)

    def test_a_narrower_cache_type_estimates_smaller(self, tmp_path, monkeypatch):
        import ml_stack.setup as setup_module

        monkeypatch.setattr(setup_module, "_arches", lambda binary: {"llama"})
        gguf = write_gguf(tmp_path / "model.gguf", LLAMA_META)
        f16 = Preflight(ServerSpec(model=gguf, context=4096), binary=fake_binary(tmp_path))
        q8 = Preflight(ServerSpec(model=gguf, context=4096, cache_type_k="q8_0",
                                  cache_type_v="q8_0"), binary=fake_binary(tmp_path))
        assert q8.kv_estimate_bytes < f16.kv_estimate_bytes

    def test_head_dim_falls_back_to_embedding_over_head_count(self, tmp_path, monkeypatch):
        import ml_stack.setup as setup_module

        monkeypatch.setattr(setup_module, "_arches", lambda binary: {"llama"})
        meta = {
            "general.architecture": "llama", "llama.block_count": 32,
            "llama.attention.head_count_kv": 8, "llama.embedding_length": 4096,
            "llama.attention.head_count": 32,   # head_dim = 4096 / 32 = 128
        }
        gguf = write_gguf(tmp_path / "model.gguf", meta)
        report = Preflight(ServerSpec(model=gguf, context=4096), binary=fake_binary(tmp_path))
        assert report.kv_estimate_bytes == int(32 * 8 * 128 * 4096 * 4.0)

    def test_it_reports_the_number_against_a_limit_either_way(self, tmp_path, monkeypatch):
        import ml_stack.setup as setup_module

        monkeypatch.setattr(setup_module, "_arches", lambda binary: {"llama"})
        gguf = write_gguf(tmp_path / "model.gguf", LLAMA_META)
        under = Preflight(ServerSpec(model=gguf, context=512), binary=fake_binary(tmp_path),
                          limit_bytes=10 * 1024**3)
        fit = next(c for c in under.checks if c.name == "fit")
        assert fit.ok

        over = Preflight(ServerSpec(model=gguf, context=512), binary=fake_binary(tmp_path),
                         limit_bytes=1)
        fit = next(c for c in over.checks if c.name == "fit")
        assert not fit.ok
        assert not over.ok

    def test_no_known_limit_still_reports_the_estimate_and_passes(self, tmp_path, monkeypatch):
        import ml_stack.setup as setup_module

        monkeypatch.setattr(setup_module, "_arches", lambda binary: {"llama"})
        gguf = write_gguf(tmp_path / "model.gguf", LLAMA_META)
        report = Preflight(ServerSpec(model=gguf), binary=fake_binary(tmp_path), limit_bytes=0)
        fit = next(c for c in report.checks if c.name == "fit")
        assert fit.ok and "estimated" in fit.detail


class TestFlags:
    def test_a_flag_the_build_lacks_is_named(self, tmp_path, monkeypatch):
        import ml_stack.setup as setup_module

        monkeypatch.setattr(setup_module, "_arches", lambda binary: {"llama"})
        gguf = write_gguf(tmp_path / "model.gguf", LLAMA_META)
        binary = fake_binary(tmp_path, help_text="-m, --model FNAME  model path\n")
        report = Preflight(ServerSpec(model=gguf, cache_type_k="q8_0"), binary=binary)
        flags = next(c for c in report.checks if c.name == "flags")
        assert not flags.ok
        assert "--cache-type-k" in flags.detail


class TestHfReference:
    def test_the_hub_being_unreachable_is_unknown_not_failed(self, tmp_path, monkeypatch):
        """What the build should hold could not even be asked -- not the same as asking
        and finding a shard absent. A network fault must not read as a missing file."""
        import ml_stack.hub as hub_module

        def broken(repo, **kw):
            raise OSError("no route to host")

        monkeypatch.setattr(hub_module, "files", broken)
        report = Preflight(ServerSpec(model="hf:maker/thing-GGUF/thing-Q4_K_M.gguf"),
                           binary=fake_binary(tmp_path))
        shards = next(c for c in report.checks if c.name == "shards")
        assert shards.ok

    def test_a_shard_missing_from_this_machine_is_named(self, tmp_path, monkeypatch):
        """`ServerSpec.hf_parts` keeps only the final path segment of an `hf:` reference
        (the same thing `-hf-file` is given), so the shelf and the reference are both
        flat filenames here -- a directory-per-quantisation repository is a separate,
        pre-existing limitation of that parsing, not of this check."""
        import ml_stack.hub as hub_module

        shelves = [
            ("thing-00001-of-00002.gguf", 4_000_000_000),
            ("thing-00002-of-00002.gguf", 3_000_000_000),
        ]
        monkeypatch.setattr(hub_module, "files", lambda repo, **kw: shelves)
        first = tmp_path / "thing-00001-of-00002.gguf"
        write_gguf(first, LLAMA_META)
        monkeypatch.setattr(preflight, "_local_index",
                            lambda: {"thing-00001-of-00002.gguf": first})
        report = Preflight(
            ServerSpec(model="hf:maker/thing-GGUF/thing-00001-of-00002.gguf"),
            binary=fake_binary(tmp_path))
        shards = next(c for c in report.checks if c.name == "shards")
        assert not shards.ok
        assert "thing-00002-of-00002.gguf" in shards.detail
        assert report.weights_bytes == 7_000_000_000

    def test_every_shard_present_locally_is_complete(self, tmp_path, monkeypatch):
        import ml_stack.hub as hub_module
        import ml_stack.setup as setup_module

        monkeypatch.setattr(setup_module, "_arches", lambda binary: {"llama"})
        shelves = [("thing-Q4_K_M.gguf", 4_000_000_000)]
        monkeypatch.setattr(hub_module, "files", lambda repo, **kw: shelves)
        local = tmp_path / "thing-Q4_K_M.gguf"
        write_gguf(local, LLAMA_META)
        monkeypatch.setattr(preflight, "_local_index",
                            lambda: {"thing-Q4_K_M.gguf": local})
        report = Preflight(ServerSpec(model="hf:maker/thing-GGUF/thing-Q4_K_M.gguf"),
                           binary=fake_binary(tmp_path))
        shards = next(c for c in report.checks if c.name == "shards")
        assert shards.ok
        arch = next(c for c in report.checks if c.name == "architecture")
        assert arch.ok and arch.detail == "llama"

    def test_a_bare_repository_reference_has_nothing_to_check_yet(self, tmp_path):
        """`hf:owner/repo` with no file is legal -- the server resolves a default quant
        itself -- and a preflight cannot fault a file that has not been named. (The flags
        check is left out of this assertion: it is about the fake binary's own --help,
        which is unrelated to what this test is about.)"""
        report = Preflight(ServerSpec(model="hf:maker/thing-GGUF"), binary=fake_binary(tmp_path))
        by_name = {c.name: c for c in report.checks}
        assert by_name["shards"].ok
        assert by_name["architecture"].ok
        assert by_name["fit"].ok


class TestReportFormatting:
    def test_said_is_one_line_per_check(self, tmp_path, monkeypatch):
        import ml_stack.setup as setup_module

        monkeypatch.setattr(setup_module, "_arches", lambda binary: {"gemma4"})  # wrong on purpose
        gguf = write_gguf(tmp_path / "model.gguf", LLAMA_META)
        report = Preflight(ServerSpec(model=gguf), binary=fake_binary(tmp_path))
        lines = report.said().splitlines()
        assert len(lines) == len(report.checks)
        assert any(line.startswith("FAIL") for line in lines)
        assert any(line.startswith("ok  ") for line in lines)


class TestStartRunsPreflight:
    """The integration: `LlamaServerBackend.start` runs `Preflight` itself, and refuses
    before `Popen` when it comes back wrong."""

    def test_a_failing_preflight_stops_the_launch(self, tmp_path, monkeypatch):
        import ml_stack.setup as setup_module

        monkeypatch.setattr(setup_module, "_arches", lambda binary: {"gemma4"})  # not "llama"
        gguf = write_gguf(tmp_path / "model.gguf", LLAMA_META)
        binary = fake_binary(tmp_path, help_text=FULL_HELP)
        backend_module.flags_of(binary)   # the help is read here, once; after this nothing may run

        def popen(*a, **k):
            raise AssertionError("a process was started despite a failing preflight")

        from ml_stack.serve.ports import free_port

        monkeypatch.setattr(subprocess, "Popen", popen)
        spec = ServerSpec(model=gguf, port=free_port())
        with pytest.raises(PreflightFailed, match="architecture"):
            LlamaServerBackend(binary=binary).start(spec, timeout=1.0)

    def test_preflight_can_be_turned_off(self, tmp_path, monkeypatch):
        import ml_stack.setup as setup_module

        monkeypatch.setattr(setup_module, "_arches", lambda binary: {"gemma4"})
        gguf = write_gguf(tmp_path / "model.gguf", LLAMA_META)
        binary = fake_binary(tmp_path, help_text=FULL_HELP)
        backend_module.flags_of(binary)   # the help is read here, once; after this nothing may run
        reached = []

        def popen(*a, **k):
            reached.append(True)
            raise OSError("stop here -- past the preflight, which is what is under test")

        monkeypatch.setattr(subprocess, "Popen", popen)
        with pytest.raises(OSError, match="stop here"):
            LlamaServerBackend(binary=binary).start(
                ServerSpec(model=gguf), timeout=1.0, preflight=False)
        assert reached == [True]


def fake_server_process(tmp_path: Path) -> Path:
    """A real, tiny executable standing in for llama-server: answers ``--help`` with
    ``FULL_HELP``, otherwise serves ``/health``, ``/props``, ``/v1/models`` and
    ``/completion`` over a real loopback socket. Spawned for real by ``Popen``, so this
    exercises the whole path -- health polling, ``load_s``, the warm-up completion -- with
    nothing mocked below the socket."""
    script = tmp_path / "llama-server"
    script.write_text(
        "#!/usr/bin/env python3\n"
        "import sys, json, http.server, socketserver\n"
        "argv = sys.argv[1:]\n"
        f"HELP = {FULL_HELP!r}\n"
        "if '--help' in argv:\n"
        "    sys.stdout.write(HELP)\n"
        "    sys.exit(0)\n"
        "def opt(flag, default=None):\n"
        "    return argv[argv.index(flag) + 1] if flag in argv else default\n"
        "host = opt('--host', '127.0.0.1')\n"
        "port = int(opt('--port', '8080'))\n"
        "class H(http.server.BaseHTTPRequestHandler):\n"
        "    def _send(self, obj):\n"
        "        body = json.dumps(obj).encode()\n"
        "        self.send_response(200)\n"
        "        self.send_header('Content-Length', str(len(body)))\n"
        "        self.end_headers()\n"
        "        self.wfile.write(body)\n"
        "    def do_GET(self):\n"
        "        if self.path.startswith('/health'):\n"
        "            self._send({'status': 'ok'})\n"
        "        elif self.path.startswith('/v1/models'):\n"
        "            self._send({'data': [{'id': 'model.gguf'}]})\n"
        "        elif self.path.startswith('/props'):\n"
        "            self._send({'model_path': 'model.gguf', 'total_slots': 1})\n"
        "        else:\n"
        "            self._send({})\n"
        "    def do_POST(self):\n"
        "        length = int(self.headers.get('content-length') or 0)\n"
        "        self.rfile.read(length)\n"
        "        self._send({'content': 'warm', 'stopped_limit': False})\n"
        "    def log_message(self, *a):\n"
        "        pass\n"
        "socketserver.TCPServer.allow_reuse_address = True\n"
        "with socketserver.TCPServer((host, port), H) as httpd:\n"
        "    httpd.serve_forever()\n"
    )
    script.chmod(0o755)
    return script


class TestLoadTimingAndWarmUp:
    """``load_s`` is wall time to the health check; ``warmup_s`` is one short completion
    sent through the real client afterward, so the first measured question is not the one
    that pays for shader compilation and the first KV allocation."""

    def _spec(self, tmp_path, monkeypatch):
        import ml_stack.setup as setup_module
        from ml_stack.serve.ports import free_port

        monkeypatch.setattr(setup_module, "_arches", lambda binary: {"llama"})
        gguf = write_gguf(tmp_path / "model.gguf", LLAMA_META)
        return ServerSpec(model=gguf, port=free_port())

    def test_load_s_and_warmup_s_are_recorded(self, tmp_path, monkeypatch):
        from ml_stack.serve.process import kill_process_tree

        spec = self._spec(tmp_path, monkeypatch)
        binary = fake_server_process(tmp_path)
        info = LlamaServerBackend(binary=binary).start(spec, timeout=10.0)
        try:
            assert info.load_s is not None and info.load_s >= 0.0
            assert info.warmup_s is not None and info.warmup_s >= 0.0
        finally:
            kill_process_tree(info.pid)

    def test_warmup_is_skipped_when_asked(self, tmp_path, monkeypatch):
        from ml_stack.serve.process import kill_process_tree

        spec = self._spec(tmp_path, monkeypatch)
        binary = fake_server_process(tmp_path)
        info = LlamaServerBackend(binary=binary).start(
            spec, timeout=10.0, warmup_request=False)
        try:
            assert info.load_s is not None
            assert info.warmup_s is None
        finally:
            kill_process_tree(info.pid)


def test_a_per_layer_head_count_is_summed_not_multiplied():
    """gemma-4-26B-A4B stores attention.head_count_kv as one entry per block; a dense model
    stores one integer. Measured 2026-09-01: the array crashed the estimate and with it the
    load. Mutation: multiply the raw value."""
    from ml_stack.serve.preflight import _kv_estimate_bytes

    dense = {"general.architecture": "x", "x.block_count": 4, "x.attention.head_count_kv": 2,
             "x.attention.key_length": 8}
    per_layer = {**dense, "x.attention.head_count_kv": [2, 2, 4, 4]}
    assert _kv_estimate_bytes(dense, 100, "", "") == 4 * 2 * 8 * 100 * 4
    assert _kv_estimate_bytes(per_layer, 100, "", "") == (2 + 2 + 4 + 4) * 8 * 100 * 4
    assert _kv_estimate_bytes({**dense, "x.attention.head_count_kv": "nonsense"}, 100, "", "") == 0
