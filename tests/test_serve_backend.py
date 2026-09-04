"""What a build of llama-server accepts, read before a model is loaded.

llama.cpp renames flags between releases -- `--draft-max` became `--spec-draft-n-max` --
and a flag the build lacks is an error at the far end of a long load. These tests stand a
shell script in for the binary, printing a help text of llama.cpp's shape; no server is
started and nothing on this machine is read.
"""

from __future__ import annotations

import os
import subprocess

import pytest
from ml_stack.serve import backend
from ml_stack.serve.backend import (
    LlamaServerBackend,
    ServerSpec,
    UnknownFlag,
    emitted_flags,
    flags_of,
    parse_context,
    trained_context,
    unknown_flags,
)
from tests.conftest import leased, write_gguf

HELP = """\
usage: llama-server [options]

common params:

-h,    --help, --usage                  print usage and exit
-c,    --ctx-size N                     size of the prompt context (default: 4096, -1 = auto)
-m,    --model FNAME                    model path (default: models/7B/ggml-model-f16.gguf)
-ngl,  --gpu-layers, --n-gpu-layers N   number of layers to store in VRAM
-np,   --parallel N                     number of server slots (default: 1)
-fa,   --flash-attn [on|off|auto]       set Flash Attention use ('on', 'off', or 'auto', default: 'auto')
       --host HOST                      ip address to listen on (default: 127.0.0.1)
       --port PORT                      port to listen on (default: 8080)
       --jinja                          use jinja template for chat (default: enabled)
       --mmproj FILE                    path to a multimodal projector file
-md,   --model-draft FNAME              draft model for speculative decoding (default: unused)
--spec-draft-n-max N                    number of tokens to draft for speculative decoding (default: 3)
--spec-draft-n-min N                    minimum number of draft tokens to use for speculative decoding
--draft, --draft-n, --draft-max N       the argument has been removed. use --spec-draft-n-max or
                                        LLAMA_ARG_SPEC_DRAFT_N_MAX instead
       --no-warmup                      skip warming up the model with an empty run
-kvu,  --kv-unified, --no-kv-unified    use single unified KV buffer shared across all sequences (default: disabled)
       --kv-unified-per-slot N          context per slot with a unified KV buffer
-cram, --cache-ram N                    set the maximum cache size in MiB (default: 8192; -1 = no limit, 0 = disable)
       --cache-idle-slots, --no-cache-idle-slots
                                        cache prompts of idle slots (default: enabled)
-sps,  --slot-prompt-similarity SIMILARITY
                                        how much the prompt of a request must match the prompt of a slot in order to use that slot (default: 0.10, 0.0 = disabled)
       --slot-save-path PATH            path to save slot kv cache (default: disabled)
"""


def fake_server(tmp_path, help_text: str = HELP, *, name: str = "llama-server"):
    """A script that answers ``--help`` the way llama-server does, and nothing else."""
    path = tmp_path / name
    path.write_text("#!/bin/sh\nif [ \"$1\" = --help ]; then cat <<'HELP'\n"
                    + help_text + "HELP\nexit 0\nfi\nexit 0\n")
    path.chmod(0o755)
    return path


@pytest.fixture(autouse=True)
def _fresh_cache(monkeypatch):
    monkeypatch.setattr(backend, "_FLAGS", {})


class TestFlagsOf:
    def test_it_reads_the_flags_out_of_the_help_text(self, tmp_path):
        known = flags_of(fake_server(tmp_path))
        assert {"-c", "--ctx-size", "-m", "--model", "-ngl", "--gpu-layers",
                "--n-gpu-layers", "--host", "--port", "--spec-draft-n-max",
                "--no-warmup", "-fa", "--flash-attn"} <= known

    def test_a_flag_the_help_says_was_removed_is_not_known(self, tmp_path):
        """The build lists `--draft-max` only to say it is gone, and passing it is an
        error just the same. Measured on llama.cpp 0.3.0."""
        known = flags_of(fake_server(tmp_path))
        assert "--draft-max" not in known
        assert "--draft" not in known
        assert "--spec-draft-n-max" in known

    def test_words_in_the_descriptions_are_not_flags(self, tmp_path):
        known = flags_of(fake_server(tmp_path))
        assert not any(flag.strip("-").isdigit() for flag in known), known
        assert "-1" not in known
        assert "LLAMA_ARG_SPEC_DRAFT_N_MAX" not in known

    def test_a_binary_is_read_once_until_it_changes(self, tmp_path):
        binary = fake_server(tmp_path)
        first = flags_of(binary)
        assert "--spec-draft-n-max" in first

        # A rebuild at the same path, with a newer mtime, is read again.
        fake_server(tmp_path, HELP.replace("--spec-draft-n-max", "--spec-draft-max"))
        newer = os.stat(binary).st_mtime + 10
        os.utime(binary, (newer, newer))
        second = flags_of(binary)
        assert "--spec-draft-max" in second and "--spec-draft-n-max" not in second

        # The same mtime is the cache, whatever the file now says.
        fake_server(tmp_path, HELP)
        os.utime(binary, (newer, newer))
        assert flags_of(binary) == second

    def test_a_binary_that_prints_no_help_is_unknown_not_empty_handed(self, tmp_path):
        silent = tmp_path / "llama-server"
        silent.write_text("#!/bin/sh\nexit 0\n")
        silent.chmod(0o755)
        assert flags_of(silent) == frozenset()

    def test_a_binary_that_is_not_there_is_unknown(self, tmp_path):
        assert flags_of(tmp_path / "absent") == frozenset()

    def test_a_binary_that_hangs_is_unknown(self, tmp_path):
        slow = tmp_path / "llama-server"
        slow.write_text("#!/bin/sh\nsleep 5\n")
        slow.chmod(0o755)
        assert flags_of(slow, timeout=0.2) == frozenset()

    def test_a_binary_that_fails_is_unknown(self, tmp_path):
        broken = tmp_path / "llama-server"
        broken.write_text("#!/bin/sh\necho 'dyld: library not loaded' >&2\nexit 1\n")
        broken.chmod(0o755)
        assert flags_of(broken) == frozenset()


class TestUnknownFlags:
    def test_a_missing_flag_is_paired_with_the_nearest_the_build_has(self, tmp_path):
        known = flags_of(fake_server(tmp_path))
        argv = ["/x/llama-server", "-m", "a.gguf", "--draft-max", "3", "-c", "4096"]
        assert unknown_flags(argv, known) == [("--draft-max", "--spec-draft-n-max")]

    def test_nothing_close_is_the_empty_string(self, tmp_path):
        known = flags_of(fake_server(tmp_path))
        assert unknown_flags(["--zzqx-nothing-like-it"], known) == [("--zzqx-nothing-like-it", "")]

    def test_values_that_start_with_a_dash_are_not_flags(self, tmp_path):
        known = flags_of(fake_server(tmp_path))
        assert unknown_flags(["-c", "-1", "--port", "-8"], known) == []

    def test_an_unknown_build_is_given_no_opinion(self):
        assert unknown_flags(["--draft-max", "3"], frozenset()) == []

    def test_each_flag_is_named_once(self, tmp_path):
        known = flags_of(fake_server(tmp_path))
        assert unknown_flags(["--draft-max", "3", "--draft-max", "4"], known) == [
            ("--draft-max", "--spec-draft-n-max")]


class TestConversationCacheFlags:
    """How many conversations a server holds at once is llama.cpp's business, through
    flags a spec had no field for. Typed here so a bench can vary them and the build can
    be asked whether it has them before a load."""

    def test_none_on_every_field_emits_nothing(self, tmp_path):
        argv = LlamaServerBackend(binary=fake_server(tmp_path)).command(
            ServerSpec(model="m.gguf"))
        for flag in ("--kv-unified", "--no-kv-unified", "--cache-ram", "--cache-idle-slots",
                     "--no-cache-idle-slots", "--slot-prompt-similarity", "--slot-save-path"):
            assert flag not in argv, flag

    def test_true_and_false_emit_the_flag_and_its_no_form(self, tmp_path):
        backend_ = LlamaServerBackend(binary=fake_server(tmp_path))
        on = backend_.command(ServerSpec(model="m.gguf", kv_unified=True, cache_idle_slots=True))
        assert "--kv-unified" in on and "--cache-idle-slots" in on
        assert "--no-kv-unified" not in on and "--no-cache-idle-slots" not in on
        off = backend_.command(ServerSpec(model="m.gguf", kv_unified=False,
                                          cache_idle_slots=False))
        assert "--no-kv-unified" in off and "--no-cache-idle-slots" in off
        assert "--kv-unified" not in off and "--cache-idle-slots" not in off

    def test_the_valued_flags_carry_their_values(self, tmp_path):
        argv = LlamaServerBackend(binary=fake_server(tmp_path)).command(
            ServerSpec(model="m.gguf", cache_ram_mb=4096, slot_prompt_similarity=0.25,
                       slot_save_path=tmp_path / "slots"))
        assert argv[argv.index("--cache-ram") + 1] == "4096"
        assert argv[argv.index("--slot-prompt-similarity") + 1] == "0.25"
        assert argv[argv.index("--slot-save-path") + 1] == str(tmp_path / "slots")
        # nought is a choice, not an absence: --cache-ram 0 disables the cache
        argv = LlamaServerBackend(binary=fake_server(tmp_path)).command(
            ServerSpec(model="m.gguf", cache_ram_mb=0, slot_prompt_similarity=0.0))
        assert argv[argv.index("--cache-ram") + 1] == "0"
        assert argv[argv.index("--slot-prompt-similarity") + 1] == "0.0"

    def test_a_build_that_lists_them_accepts_them(self, tmp_path):
        binary = fake_server(tmp_path)
        known = flags_of(binary)
        assert {"--kv-unified", "--no-kv-unified", "--cache-ram", "--cache-idle-slots",
                "--no-cache-idle-slots", "--slot-prompt-similarity", "--slot-save-path"} <= known
        argv = LlamaServerBackend(binary=binary).command(
            ServerSpec(model="m.gguf", kv_unified=False, cache_ram_mb=8192,
                       cache_idle_slots=True, slot_prompt_similarity=0.5,
                       slot_save_path="slots"))
        assert unknown_flags(argv, known) == []

    def test_a_build_without_them_names_them_before_the_load(self, tmp_path):
        older = fake_server(tmp_path, "\n".join(
            line for line in HELP.splitlines()
            if "kv-unified," not in line and "cache-ram" not in line) + "\n")
        argv = LlamaServerBackend(binary=older).command(
            ServerSpec(model="m.gguf", kv_unified=True, cache_ram_mb=8192))
        lacking = dict(unknown_flags(argv, flags_of(older)))
        assert set(lacking) == {"--kv-unified", "--cache-ram"}


class TestEmittedFlags:
    def test_every_flag_the_argv_builder_knows_appears(self, tmp_path):
        flags = emitted_flags(LlamaServerBackend(binary=fake_server(tmp_path)))
        for flag in ("-m", "--hf-repo", "--hf-file", "--mmproj", "--mmproj-url", "-md",
                     "-hfd", "--spec-type", "--spec-draft-n-max", "--spec-draft-ngl",
                     "--override-tensor", "--cpu-moe", "--n-cpu-moe", "--cache-reuse",
                     "--no-warmup", "--kv-unified-per-slot", "--lookup-cache-static",
                     "--lookup-cache-dynamic", "-fa", "--jinja", "--embeddings",
                     "--pooling", "--kv-unified", "--no-kv-unified", "--cache-ram",
                     "--cache-idle-slots", "--no-cache-idle-slots",
                     "--slot-prompt-similarity", "--slot-save-path", "--cache-type-k",
                     "--cache-type-v", "--no-mmap", "--mlock", "--reasoning-budget",
                     "--rope-scaling", "--rope-scale", "--yarn-orig-ctx",
                     "--yarn-ext-factor", "--yarn-attn-factor", "--yarn-beta-fast",
                     "--yarn-beta-slow"):
            assert flag in flags, flag
        assert len(flags) == len(set(flags))
        assert not any(token.endswith(".gguf") for token in flags)


class TestMemoryFlags:
    """How the KV cache is stored, and whether the weights are mmapped or locked -- typed
    so a preflight's fit estimate can read the cache type back, and so a bench can vary
    them the way it varies everything else through ``ServerSpec``."""

    def test_none_on_every_field_emits_nothing(self, tmp_path):
        argv = LlamaServerBackend(binary=fake_server(tmp_path)).command(
            ServerSpec(model="m.gguf"))
        for flag in ("--cache-type-k", "--cache-type-v", "--no-mmap", "--mlock"):
            assert flag not in argv, flag

    def test_cache_types_carry_their_values(self, tmp_path):
        argv = LlamaServerBackend(binary=fake_server(tmp_path)).command(
            ServerSpec(model="m.gguf", cache_type_k="q8_0", cache_type_v="q4_0"))
        assert argv[argv.index("--cache-type-k") + 1] == "q8_0"
        assert argv[argv.index("--cache-type-v") + 1] == "q4_0"

    def test_mmap_false_emits_no_mmap_and_true_emits_nothing(self, tmp_path):
        """mmap on is the server's own default and passes no flag either way."""
        backend_ = LlamaServerBackend(binary=fake_server(tmp_path))
        assert "--no-mmap" in backend_.command(ServerSpec(model="m.gguf", mmap=False))
        assert "--no-mmap" not in backend_.command(ServerSpec(model="m.gguf", mmap=True))
        assert "--no-mmap" not in backend_.command(ServerSpec(model="m.gguf"))

    def test_mlock_true_emits_mlock(self, tmp_path):
        backend_ = LlamaServerBackend(binary=fake_server(tmp_path))
        assert "--mlock" in backend_.command(ServerSpec(model="m.gguf", mlock=True))
        assert "--mlock" not in backend_.command(ServerSpec(model="m.gguf", mlock=False))


class TestReasoningBudget:
    """A ceiling (`n_predict`) cuts the answer; a budget stops the thinking. Typed so a
    sweep can bind it per served model and record it, like the cache type."""

    def test_a_budget_is_emitted_with_its_value(self, tmp_path):
        argv = LlamaServerBackend(binary=fake_server(tmp_path)).command(
            ServerSpec(model="m.gguf", reasoning_budget=2048))
        assert argv[argv.index("--reasoning-budget") + 1] == "2048"

    def test_none_emits_nothing_and_minus_one_is_unlimited_not_nothing(self, tmp_path):
        backend_ = LlamaServerBackend(binary=fake_server(tmp_path))
        assert "--reasoning-budget" not in backend_.command(ServerSpec(model="m.gguf"))
        argv = backend_.command(ServerSpec(model="m.gguf", reasoning_budget=-1))
        assert argv[argv.index("--reasoning-budget") + 1] == "-1", "llama.cpp's own unlimited"

    def test_a_build_without_the_flag_is_told_before_anything_starts(self, tmp_path):
        """The fake help has no --reasoning-budget, like a release from before it existed."""
        binary = fake_server(tmp_path)
        argv = LlamaServerBackend(binary=binary).command(
            ServerSpec(model="m.gguf", reasoning_budget=512))
        assert "--reasoning-budget" in dict(unknown_flags(argv, flags_of(binary)))


class TestRopeYarnFlags:
    """RoPE/YaRN: how a context past the model's own training length is read. None on
    every field emits nothing, so the model's own default (unscaled) stands."""

    def test_none_on_every_field_emits_nothing(self, tmp_path):
        argv = LlamaServerBackend(binary=fake_server(tmp_path)).command(
            ServerSpec(model="m.gguf"))
        for flag in ("--rope-scaling", "--rope-scale", "--yarn-orig-ctx",
                     "--yarn-ext-factor", "--yarn-attn-factor", "--yarn-beta-fast",
                     "--yarn-beta-slow"):
            assert flag not in argv, flag

    def test_every_field_carries_its_value(self, tmp_path):
        argv = LlamaServerBackend(binary=fake_server(tmp_path)).command(
            ServerSpec(model="m.gguf", rope_scaling="yarn", rope_scale=8.0,
                       yarn_orig_ctx=131072, yarn_ext_factor=1.0, yarn_attn_factor=1.0,
                       yarn_beta_fast=32.0, yarn_beta_slow=1.0))
        assert argv[argv.index("--rope-scaling") + 1] == "yarn"
        assert argv[argv.index("--rope-scale") + 1] == "8.0"
        assert argv[argv.index("--yarn-orig-ctx") + 1] == "131072"
        assert argv[argv.index("--yarn-ext-factor") + 1] == "1.0"
        assert argv[argv.index("--yarn-attn-factor") + 1] == "1.0"
        assert argv[argv.index("--yarn-beta-fast") + 1] == "32.0"
        assert argv[argv.index("--yarn-beta-slow") + 1] == "1.0"

    def test_a_build_without_them_names_them_before_the_load(self, tmp_path):
        binary = fake_server(tmp_path)
        argv = LlamaServerBackend(binary=binary).command(
            ServerSpec(model="m.gguf", rope_scaling="yarn", rope_scale=4.0,
                       yarn_orig_ctx=32768))
        lacking = dict(unknown_flags(argv, flags_of(binary)))
        assert {"--rope-scaling", "--rope-scale", "--yarn-orig-ctx"} <= set(lacking)


class TestParseContext:
    """``--context`` the way a person names it, not only the raw integer llama-server
    takes: ``32768``, ``256k``, ``1m``."""

    @pytest.mark.parametrize("text, want", [
        ("32768", 32768),
        ("0", 0),
        ("4096", 4096),
        ("256k", 256_000),
        ("256K", 256_000),
        ("1m", 1_000_000),
        ("1M", 1_000_000),
        ("1.5m", 1_500_000),
        (" 1m ", 1_000_000),
    ])
    def test_the_edges(self, text, want):
        assert parse_context(text) == want

    def test_an_int_passes_through(self):
        assert parse_context(32768) == 32768

    @pytest.mark.parametrize("text", ["", "abc", "1g", "-1m", "1 m 2", "k"])
    def test_nonsense_is_refused(self, text):
        with pytest.raises(ValueError):
            parse_context(text)


class TestTrainedContext:
    def test_reads_the_context_length_off_the_header(self, tmp_path):
        gguf = write_gguf(tmp_path / "m.gguf",
                          {"general.architecture": "llama", "llama.block_count": 1,
                           "llama.context_length": 131072})
        assert trained_context(gguf) == 131072

    def test_a_missing_file_is_unknown_not_zero_trained(self, tmp_path):
        assert trained_context(tmp_path / "absent.gguf") == 0

    def test_an_hf_reference_not_on_disk_is_unknown(self):
        assert trained_context("hf:owner/repo/model.gguf") == 0

    def test_a_file_with_no_context_length_key_is_zero(self, tmp_path):
        gguf = write_gguf(tmp_path / "m.gguf",
                          {"general.architecture": "llama", "llama.block_count": 1})
        assert trained_context(gguf) == 0


class TestResolvedContext:
    """``LlamaServerBackend.resolved_context`` turns YaRN on by itself only when the ask
    is past what the model trained at, and leaves everything else exactly as it was."""

    def test_a_context_at_or_under_the_trained_length_is_untouched(self, tmp_path):
        gguf = write_gguf(tmp_path / "m.gguf",
                          {"general.architecture": "llama", "llama.block_count": 1,
                           "llama.context_length": 131072})
        for context in (4096, 131072):
            spec = ServerSpec(model=gguf, context=context)
            resolved, said = LlamaServerBackend.resolved_context(spec)
            assert resolved == spec
            assert said == ""

    def test_a_context_past_the_trained_length_turns_yarn_on(self, tmp_path):
        gguf = write_gguf(tmp_path / "m.gguf",
                          {"general.architecture": "llama", "llama.block_count": 1,
                           "llama.context_length": 131072})
        spec = ServerSpec(model=gguf, context=1_000_000)
        resolved, said = LlamaServerBackend.resolved_context(spec)
        assert resolved.rope_scaling == "yarn"
        assert resolved.yarn_orig_ctx == 131072
        # ceil(1_000_000 / 131_072 * 100) / 100, rounded up so the scaled context always
        # covers what was asked
        assert resolved.rope_scale == pytest.approx(7.63)
        assert 131072 * resolved.rope_scale >= 1_000_000
        assert "131,072" in said and "1,000,000" in said and "YaRN" in said

    def test_an_unknown_trained_length_is_untouched(self, tmp_path):
        gguf = tmp_path / "m.gguf"
        gguf.write_bytes(b"not a gguf at all")
        spec = ServerSpec(model=gguf, context=1_000_000)
        resolved, said = LlamaServerBackend.resolved_context(spec)
        assert resolved == spec
        assert said == ""

    def test_a_spec_that_already_names_a_rope_field_is_left_alone(self, tmp_path):
        gguf = write_gguf(tmp_path / "m.gguf",
                          {"general.architecture": "llama", "llama.block_count": 1,
                           "llama.context_length": 131072})
        spec = ServerSpec(model=gguf, context=1_000_000, rope_scaling="linear")
        resolved, said = LlamaServerBackend.resolved_context(spec)
        assert resolved == spec
        assert said == ""


class TestLaunchRefusal:
    def test_it_refuses_before_anything_is_started(self, tmp_path, monkeypatch):
        """A refusal that comes after the load costs the load. Nothing may be started."""
        gguf = tmp_path / "model.gguf"
        gguf.write_bytes(b"GGUF" + b"\x00" * 64)
        binary = fake_server(tmp_path)
        flags_of(binary)    # the help is read once, here; after this nothing may run
        started: list[list[str]] = []

        def popen(argv, *a, **k):
            started.append(argv)
            raise AssertionError("a process was started")

        monkeypatch.setattr(subprocess, "Popen", popen)
        spec = ServerSpec(model=gguf, extra_args=("--draft-max", "3"))

        with pytest.raises(UnknownFlag) as caught:
            leased(LlamaServerBackend(binary=binary), spec, timeout=1.0)
        assert str(caught.value) == "this llama-server has no --draft-max; it has --spec-draft-n-max"
        assert started == []

    def test_one_line_per_flag(self, tmp_path, monkeypatch):
        binary = fake_server(tmp_path)
        flags_of(binary)
        monkeypatch.setattr(subprocess, "Popen",
                            lambda *a, **k: (_ for _ in ()).throw(AssertionError("started")))
        gguf = tmp_path / "model.gguf"
        gguf.write_bytes(b"GGUF" + b"\x00" * 64)
        spec = ServerSpec(model=gguf, extra_args=("--draft-max", "3", "--zzqx", "1"))
        with pytest.raises(UnknownFlag) as caught:
            leased(LlamaServerBackend(binary=binary), spec, timeout=1.0)
        assert str(caught.value).splitlines() == [
            "this llama-server has no --draft-max; it has --spec-draft-n-max",
            "this llama-server has no --zzqx",
        ]

    def test_it_is_a_value_error_and_not_a_failed_server(self, tmp_path):
        """A ServerFailed puts the port in the negative cache; a wrong flag is a caller's
        mistake and must read as one."""
        from ml_stack.serve import ServerFailed

        assert issubclass(UnknownFlag, ValueError)
        assert not issubclass(UnknownFlag, ServerFailed)

    def test_the_check_can_be_skipped_for_a_stand_in_binary(self, tmp_path, monkeypatch):
        reached: list[str] = []

        def popen(argv, *a, **k):
            reached.append("popen")
            raise OSError("stop here")

        binary = fake_server(tmp_path)
        monkeypatch.setattr(subprocess, "Popen", popen)
        gguf = tmp_path / "model.gguf"
        gguf.write_bytes(b"GGUF" + b"\x00" * 64)
        spec = ServerSpec(model=gguf, extra_args=("--draft-max", "3"))
        with pytest.raises(OSError, match="stop here"):
            leased(LlamaServerBackend(binary=binary), 
                spec, timeout=1.0, check_flags=False, preflight=False)
        assert reached == ["popen"]

    def test_a_build_that_prints_no_help_is_not_refused(self, tmp_path, monkeypatch):
        """The stand-ins in the other test files print nothing; they must keep working."""
        monkeypatch.setattr(subprocess, "Popen",
                            lambda *a, **k: (_ for _ in ()).throw(OSError("stop here")))
        silent = tmp_path / "llama-server"
        silent.write_text("#!/bin/sh\nexit 0\n")
        silent.chmod(0o755)
        gguf = tmp_path / "model.gguf"
        gguf.write_bytes(b"GGUF" + b"\x00" * 64)
        with pytest.raises(OSError, match="stop here"):
            # preflight=False: this stand-in gguf carries no real metadata, and what this
            # test is about is the flag check, not the preflight one.
            leased(LlamaServerBackend(binary=silent), 
                ServerSpec(model=gguf, extra_args=("--draft-max", "3")), timeout=1.0,
                preflight=False)


def test_an_hf_reference_keeps_the_file_under_its_directory():
    """A draft head lives under MTP/; a reference that kept only the last segment fetched
    nothing and llama-server was started with an empty draft path (measured 2026-09-01).
    Mutation: `parts[-1]` instead of the join."""
    from ml_stack.serve.backend import ServerSpec

    repo, name = ServerSpec.hf_parts("hf:owner/repo-GGUF/MTP/mtp-head-BF16.gguf")
    assert (repo, name) == ("owner/repo-GGUF", "MTP/mtp-head-BF16.gguf")
    assert ServerSpec.hf_parts("hf:owner/repo-GGUF") == ("owner/repo-GGUF", "")


def test_a_draft_named_by_file_is_fetched_and_served_by_path(monkeypatch, tmp_path):
    """`-hfd` takes owner/repo[:quant], never a file, so a head named by file is fetched
    into the cache first and passed with -md. Mutation: drop resolved_draft() from start."""
    binary = tmp_path / "llama-server"
    binary.write_text("#!/bin/sh\necho usage: llama-server\n")
    binary.chmod(0o755)
    from ml_stack.serve import backend as be

    head = tmp_path / "mtp-head-BF16.gguf"
    head.write_bytes(b"GGUF")
    asked = []
    monkeypatch.setattr("ml_stack.hub.fetch", lambda ref: asked.append(ref) or head)
    spec = be.ServerSpec(model=tmp_path / "m.gguf", draft="hf:owner/repo-GGUF/MTP/mtp-head-BF16.gguf")
    resolved = be.LlamaServerBackend.resolved_draft(spec)
    assert asked == ["hf:owner/repo-GGUF/MTP/mtp-head-BF16.gguf"]
    argv = be.LlamaServerBackend(binary=binary).command(resolved)
    assert argv[argv.index("-md") + 1] == str(head)
    with pytest.raises(be.ServerFailed, match="fetched before serving"):
        be.LlamaServerBackend(binary=binary).command(spec)
    quant = be.ServerSpec(model=tmp_path / "m.gguf", draft="hf:owner/repo-GGUF")
    assert "-hfd" in be.LlamaServerBackend(binary=binary).command(quant)
