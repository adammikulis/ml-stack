"""Tool discovery, and a real GGUF round trip through the metadata rewriter.

The metadata tests build an actual GGUF with the ``gguf`` writer, patch it, and read it
back. Asserting on a mocked reader would prove only that the mock was configured, and the
bug this module exists to prevent is precisely a key that is silently absent.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ml_stack.gguf import tools

#: Captured before any test patches it, so the fixture below cannot hide
#: what the shipped default actually is.
REAL_SOURCE_DIRS = tools.SOURCE_DIRS

from ml_stack.gguf import (  # noqa: E402  -- after REAL_SOURCE_DIRS, on purpose
    ADD_SPACE_PREFIX,
    ConversionError,
    ToolNotFound,
    VocabPatchError,
    convert,
    find_converter,
    find_quantize,
    fix_space_prefix,
    quantize,
    read_metadata,
    require_converter,
    require_quantize,
    set_metadata,
)

gguf = pytest.importorskip("gguf")
np = pytest.importorskip("numpy")


@pytest.fixture
def sample_gguf(tmp_path):
    """A minimal but genuine GGUF: architecture, a few typed keys, one tensor."""
    path = tmp_path / "sample.gguf"
    writer = gguf.GGUFWriter(str(path), "llama")
    writer.add_string("general.name", "sample")
    writer.add_uint32("llama.context_length", 2048)
    writer.add_array("tokenizer.ggml.tokens", ["<pad>", "a", "b"])
    writer.add_tensor("token_embd.weight", np.zeros((3, 4), dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return path


class TestToolDiscovery:
    @pytest.fixture(autouse=True)
    def _isolate_search_path(self, monkeypatch):
        """Empty SOURCE_DIRS for every discovery test.

        Without this, "not found" tests pass or fail depending on whether the
        machine running them happens to have a llama.cpp checkout in one of the
        default locations -- so they assert something about the developer's home
        directory rather than about the code. Adding a single common path to
        SOURCE_DIRS turned three of them red, which is how this was noticed.
        Tests that need a directory searched put one there explicitly.
        """
        monkeypatch.setattr(tools, "SOURCE_DIRS", ())

    def test_the_unsloth_checkout_is_searched(self):
        """unsloth vendors llama.cpp, so many machines have the converter without
        having installed it deliberately."""
        assert Path.home() / ".unsloth" / "llama.cpp" in REAL_SOURCE_DIRS

    def test_explicit_path_wins(self, tmp_path):
        script = tmp_path / "convert_hf_to_gguf.py"
        script.write_text("# converter\n")
        assert find_converter(script) == script.resolve()

    def test_a_nonexistent_explicit_path_falls_through(self, tmp_path, monkeypatch):
        """A stale config should not hard-fail when a usable copy exists elsewhere."""
        monkeypatch.delenv("LLAMA_CPP_ROOT", raising=False)
        monkeypatch.delenv("LLAMA_CPP_DIR", raising=False)
        assert find_converter(tmp_path / "absent.py") is None

    def test_env_root_is_searched(self, tmp_path, monkeypatch):
        script = tmp_path / "convert_hf_to_gguf.py"
        script.write_text("# converter\n")
        monkeypatch.setenv("LLAMA_CPP_ROOT", str(tmp_path))
        assert find_converter() == script.resolve()

    def test_require_names_what_it_tried_and_how_to_fix_it(self, tmp_path, monkeypatch):
        """An error that only says 'not found' makes the reader go hunting."""
        monkeypatch.delenv("LLAMA_CPP_ROOT", raising=False)
        monkeypatch.delenv("LLAMA_CPP_DIR", raising=False)
        monkeypatch.setenv("ML_STACK_CACHE", str(tmp_path / "cache"))

        with pytest.raises(ToolNotFound) as excinfo:
            require_converter(tmp_path / "absent.py")
        message = str(excinfo.value)
        assert "LLAMA_CPP_ROOT" in message
        assert "git clone" in message

    def test_quantize_is_found_in_a_build_dir(self, tmp_path, monkeypatch):
        """The quantiser ships as a binary, usually under build/bin."""
        binary = tmp_path / "build" / "bin" / "llama-quantize"
        binary.parent.mkdir(parents=True)
        binary.write_text("#!/bin/sh\n")
        binary.chmod(0o755)
        monkeypatch.setenv("LLAMA_CPP_ROOT", str(tmp_path))
        assert find_quantize() == binary.resolve()

    def test_require_quantize_raises_when_absent(self, tmp_path, monkeypatch):
        monkeypatch.delenv("LLAMA_CPP_ROOT", raising=False)
        monkeypatch.delenv("LLAMA_CPP_DIR", raising=False)
        monkeypatch.setenv("PATH", str(tmp_path))
        with pytest.raises(ToolNotFound, match="llama-quantize"):
            require_quantize()


class TestMetadataRewrite:
    def test_round_trip_preserves_every_key_and_tensor(self, sample_gguf, tmp_path):
        before = read_metadata(sample_gguf)
        out = set_metadata(sample_gguf, tmp_path / "out.gguf", {"general.name": "patched"})
        after = read_metadata(out)

        assert after["general.name"] == "patched"
        assert after["llama.context_length"] == before["llama.context_length"]
        assert set(before) <= set(after), "metadata keys went missing in the rewrite"

        reader = gguf.GGUFReader(str(out))
        assert [t.name for t in reader.tensors] == ["token_embd.weight"]

    def test_the_missing_key_gets_written(self, sample_gguf):
        """Absent, llama.cpp defaults this to true and inserts a space after every
        special token, so the runtime tokenizer and the training tokenizer disagree on
        every sequence -- and nothing reports it."""
        assert ADD_SPACE_PREFIX not in read_metadata(sample_gguf)

        fix_space_prefix(sample_gguf)

        after = read_metadata(sample_gguf)
        assert ADD_SPACE_PREFIX in after
        assert bool(after[ADD_SPACE_PREFIX]) is False

    def test_the_value_is_settable(self, sample_gguf):
        fix_space_prefix(sample_gguf, add_space_prefix=True)
        assert bool(read_metadata(sample_gguf)[ADD_SPACE_PREFIX]) is True

    def test_writing_to_a_separate_destination_leaves_the_source_alone(
        self, sample_gguf, tmp_path
    ):
        out = tmp_path / "fixed.gguf"
        fix_space_prefix(sample_gguf, out)
        assert ADD_SPACE_PREFIX in read_metadata(out)
        assert ADD_SPACE_PREFIX not in read_metadata(sample_gguf)

    def test_no_part_file_is_left_behind(self, sample_gguf, tmp_path):
        out = tmp_path / "out.gguf"
        set_metadata(sample_gguf, out, {"general.name": "x"})
        assert not out.with_suffix(".gguf.part").exists()
        assert not out.with_suffix(".gguf.fixed").exists()

    def test_a_non_gguf_is_rejected(self, tmp_path):
        junk = tmp_path / "not.gguf"
        junk.write_bytes(b"this is not a GGUF file at all")
        with pytest.raises(Exception):
            set_metadata(junk, tmp_path / "out.gguf", {"k": "v"})

    def test_a_missing_source_is_reported_clearly(self, tmp_path):
        with pytest.raises(VocabPatchError, match="no GGUF at"):
            set_metadata(tmp_path / "absent.gguf", tmp_path / "out.gguf", {"k": "v"})

    def test_an_unsupported_value_type_raises(self, sample_gguf, tmp_path):
        with pytest.raises(VocabPatchError, match="cannot write metadata"):
            set_metadata(sample_gguf, tmp_path / "out.gguf", {"k": {"nested": "dict"}})


class TestConvertGuards:
    def test_a_missing_model_directory_is_caught_before_the_converter_runs(self, tmp_path):
        with pytest.raises(ConversionError, match="no model directory"):
            convert(tmp_path / "absent", tmp_path / "out.gguf")

    def test_quantize_rejects_a_missing_source(self, tmp_path):
        with pytest.raises(ConversionError, match="no GGUF at"):
            quantize(tmp_path / "absent.gguf", tmp_path / "out.gguf")


class TestExportVerification:
    """A gate nobody has watched fail is a gate of unknown meaning, so each of
    these drives the check with a genuinely broken artifact."""

    def test_a_missing_space_prefix_key_is_a_failure_not_a_default(self, sample_gguf):
        """ABSENT is the dangerous state: llama.cpp reads it as TRUE."""
        from ml_stack.gguf import verify_metadata
        r = verify_metadata(sample_gguf)
        assert not r.ok
        assert any("ABSENT" in c.detail for c in r.failures)

    def test_it_passes_once_the_key_is_written(self, sample_gguf):
        from ml_stack.gguf import fix_space_prefix, verify_metadata
        fix_space_prefix(sample_gguf)
        r = verify_metadata(sample_gguf)
        assert r.ok, str(r)

    def test_a_file_that_does_not_open_fails_the_first_check_and_stops(self, tmp_path):
        bad = tmp_path / "truncated.gguf"
        bad.write_bytes(b"GGUF\x03\x00\x00\x00garbage")
        from ml_stack.gguf import verify_metadata
        r = verify_metadata(bad)
        assert not r.ok
        assert r.checks[0].name == "opens" and not r.checks[0].ok
        assert len(r.checks) == 1, "it kept checking a file it could not read"

    def test_the_wrong_value_is_caught_not_just_the_missing_one(self, sample_gguf):
        from ml_stack.gguf import fix_space_prefix, verify_metadata
        fix_space_prefix(sample_gguf, add_space_prefix=True)
        r = verify_metadata(sample_gguf, expect_space_prefix=False)
        assert not r.ok

    # ---------------------------------------------------- served comparison

    @staticmethod
    def _fake_serving(server_ids):
        """A stand-in serve()/Client pair returning fixed token ids."""
        import contextlib

        class _Client:
            def __init__(self, url): pass
            def tokenize(self, text, with_pieces=False):
                ids = server_ids(text)
                if with_pieces:
                    return [{"id": i, "piece": f"<{i}>"} for i in ids]
                return ids

        @contextlib.contextmanager
        def _serve(model, context=1024, **kw):
            class _S:
                base_url = "http://127.0.0.1:0"

            yield _S()

        return _serve, _Client

    def test_identical_tokenisation_passes(self, sample_gguf):
        from ml_stack.gguf import fix_space_prefix, verify_tokenizer_fidelity
        fix_space_prefix(sample_gguf)
        def enc(text):
            return [len(text), 7]

        serve, client = self._fake_serving(lambda t: [len(t), 7])
        r = verify_tokenizer_fidelity(sample_gguf, enc, ["<user> go"],
                                      serve_fn=serve, client_cls=client)
        assert r.ok, str(r)

    def test_a_divergence_is_reported_with_both_sides_and_the_pieces(self, sample_gguf):
        """The whole point: show WHAT differs, or the next step is guesswork."""
        from ml_stack.gguf import fix_space_prefix, verify_tokenizer_fidelity
        fix_space_prefix(sample_gguf)
        serve, client = self._fake_serving(lambda t: [999, 998])
        r = verify_tokenizer_fidelity(sample_gguf, lambda t: [1, 2], ["<user> go"],
                                      serve_fn=serve, client_cls=client)
        assert not r.ok
        detail = r.failures[0].detail
        assert "reference=[1, 2]" in detail and "server=[999, 998]" in detail
        assert "pieces=" in detail

    def test_a_leading_bos_from_the_server_is_not_a_divergence(self, sample_gguf):
        """llama.cpp prepends BOS and most reference tokenizers do not. Without
        this every single probe would report a false mismatch."""
        from ml_stack.gguf import fix_space_prefix, verify_tokenizer_fidelity
        fix_space_prefix(sample_gguf)
        serve, client = self._fake_serving(lambda t: [1, 40, 41])
        r = verify_tokenizer_fidelity(sample_gguf, lambda t: [40, 41], ["x"],
                                      bos_id=1, serve_fn=serve, client_cls=client)
        assert r.ok, str(r)

    def test_no_server_is_started_when_the_file_does_not_open(self, tmp_path):
        bad = tmp_path / "bad.gguf"
        bad.write_bytes(b"nope")
        started = []

        def _serve(*a, **k):
            started.append(1)
            raise AssertionError("served a file that cannot be read")

        from ml_stack.gguf import verify_tokenizer_fidelity
        r = verify_tokenizer_fidelity(bad, lambda t: [], ["x"], serve_fn=_serve,
                                      client_cls=object)
        assert not r.ok and not started
