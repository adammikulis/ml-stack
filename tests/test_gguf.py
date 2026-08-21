"""Tool discovery, and a real GGUF round trip through the metadata rewriter.

The metadata tests build an actual GGUF with the ``gguf`` writer, patch it, and read it
back. Asserting on a mocked reader would prove only that the mock was configured, and the
bug this module exists to prevent is precisely a key that is silently absent.
"""

from __future__ import annotations

import os

import pytest

from mainspring.gguf import (
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
        monkeypatch.setenv("MAINSPRING_CACHE", str(tmp_path / "cache"))

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
