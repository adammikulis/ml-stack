"""The one bounded reader of GGUF and safetensors headers, and every caller that goes through it,
against files whose counts and lengths lie."""

from __future__ import annotations

import json
import struct
import time

import pytest

from ml_stack.hub import header, in_gguf
from ml_stack.hub.modelfile import (
    Limits,
    NotAModelFile,
    safetensors_header,
    scan_gguf,
)
from ml_stack.serve.mlx_tree import resident_bytes
from ml_stack.serve.preflight import read_gguf_header
from ml_stack.serve.tensors import tensors_of

STRING, ARRAY = 8, 9
U8, U32, F32 = 0, 4, 6


def q(n: int) -> bytes:
    return struct.pack("<Q", n)


def i(n: int) -> bytes:
    return struct.pack("<I", n)


def s(text: str) -> bytes:
    raw = text.encode()
    return q(len(raw)) + raw


def pair(key: str, kind: int, body: bytes) -> bytes:
    return s(key) + i(kind) + body


_made = iter(range(10**6))


def gguf(tmp_path, pairs: bytes = b"", *, kv: int = 0, tensors: int = 0, table: bytes = b""):
    """A GGUF whose header says ``kv`` pairs and ``tensors`` tensors, whatever it holds."""
    path = tmp_path / f"m{next(_made)}.gguf"
    path.write_bytes(b"GGUF" + i(3) + q(tensors) + q(kv) + pairs + table)
    return path


def honest(tmp_path):
    pairs = (pair("general.architecture", STRING, s("llama"))
             + pair("general.sampling.temp", F32, struct.pack("<f", 0.5))
             + pair("llama.block_count", U32, i(2))
             + pair("small", ARRAY, i(U32) + q(3) + i(1) + i(2) + i(3))
             + pair("tokenizer.ggml.tokens", ARRAY, i(STRING) + q(2) + s("a") + s("b")))
    table = s("blk.0.w") + i(2) + q(4) + q(8) + i(0) + q(0)
    return gguf(tmp_path, pairs, kv=5, tensors=1, table=table)


def hostile_tables(tmp_path):
    """Tensor tables that claim far more than they hold."""
    one = pair("k", STRING, s("v"))
    return {
        "2**60 tensors": gguf(tmp_path, one, kv=1, tensors=2**60),
        "a tensor of 2**31 dimensions": gguf(
            tmp_path, one, kv=1, tensors=1, table=s("t") + i(2**31) + b"\x00" * 64),
        "a tensor table cut short": gguf(
            tmp_path, one, kv=1, tensors=3, table=s("t") + i(1) + q(4)),
    }


def hostile_files(tmp_path):
    """Header bodies that claim far more than they hold, by what they exaggerate."""
    one = pair("k", STRING, s("v"))
    nested = i(ARRAY) + q(1)
    return {
        "a string of 2**60 bytes": gguf(tmp_path, q(2**60) + b"k", kv=1),
        "a value string of 2**40 bytes": gguf(tmp_path, s("k") + i(STRING) + q(2**40), kv=1),
        "an array of 2**50 numbers": gguf(
            tmp_path, s("k") + i(ARRAY) + i(U8) + q(2**50), kv=1),
        "an array of 2**50 strings": gguf(
            tmp_path, s("k") + i(ARRAY) + i(STRING) + q(2**50), kv=1),
        "arrays nested a thousand deep": gguf(
            tmp_path, s("k") + i(ARRAY) + nested * 1000 + i(U8) + q(0), kv=1),
        "nested arrays that each claim a tenth of the file": gguf(
            tmp_path, s("k") + i(ARRAY) + i(ARRAY) + q(1000)
            + (i(ARRAY) + q(100) + b"\x00" * 100) * 1000, kv=1),
        "2**60 pairs": gguf(tmp_path, one, kv=2**60),
        "more pairs than the file has bytes": gguf(tmp_path, one, kv=10_000),
        "an array of an unknown kind": gguf(
            tmp_path, s("k") + i(ARRAY) + i(99) + q(1) + b"\x00", kv=1),
        "a value of an unknown kind": gguf(tmp_path, s("k") + i(77) + b"\x00" * 8, kv=1),
        "a header cut inside a string": gguf(tmp_path, s("key")[:5], kv=1),
        "a header cut after the counts": gguf(tmp_path, b"", kv=1),
    }


def test_an_honest_header_is_read_with_its_tensors(tmp_path):
    got = scan_gguf(honest(tmp_path), tensors=True)
    assert got.values["general.architecture"] == "llama"
    assert got.values["small"] == [1, 2, 3]
    assert got.values["tokenizer.ggml.tokens"] == ["a", "b"]
    assert [(t.name, t.shape, t.kind) for t in got.tensors] == [("blk.0.w", (4, 8), 0)]


def test_every_hostile_header_is_refused_quickly(tmp_path):
    for what, path in {**hostile_files(tmp_path), **hostile_tables(tmp_path)}.items():
        began = time.monotonic()
        with pytest.raises(NotAModelFile):
            scan_gguf(path, tensors=True)
        assert time.monotonic() - began < 2, what


@pytest.mark.parametrize("reader", [
    lambda p: read_gguf_header(p),
    lambda p: header.meta(p) or (_ for _ in ()).throw(ValueError("not a model file")),
    lambda p: header.read(p) or (_ for _ in ()).throw(ValueError("not a model file")),
    lambda p: in_gguf(p) or (_ for _ in ()).throw(ValueError("not a model file")),
], ids=["preflight", "header.meta", "header.read", "in_gguf"])
def test_every_caller_refuses_each_hostile_header_without_hanging(tmp_path, reader):
    """Whichever reader a listing, a fit or a preflight uses, none builds the claimed string,
    loops over the claimed items or recurses through the claimed nesting."""
    for what, path in hostile_files(tmp_path).items():
        began = time.monotonic()
        with pytest.raises(ValueError):
            reader(path)
        assert time.monotonic() - began < 2, what


def test_the_tensor_listing_refuses_each_hostile_header_and_table(tmp_path):
    for what, path in {**hostile_files(tmp_path), **hostile_tables(tmp_path)}.items():
        began = time.monotonic()
        with pytest.raises(ValueError):
            tensors_of(path)
        assert time.monotonic() - began < 2, what


def test_the_pair_count_is_refused_before_any_pair_is_read(tmp_path):
    with pytest.raises(NotAModelFile, match="metadata pairs"):
        scan_gguf(gguf(tmp_path, b"", kv=2**60))


def test_a_string_longer_than_the_file_says_so(tmp_path):
    with pytest.raises(NotAModelFile, match="more than the file holds"):
        scan_gguf(gguf(tmp_path, q(2**60) + b"k", kv=1))


def test_total_array_items_are_capped_across_the_header(tmp_path):
    body = b"".join(pair(f"k{n}", ARRAY, i(U8) + q(40) + b"\x00" * 40) for n in range(10))
    path = gguf(tmp_path, body, kv=10)
    assert len(scan_gguf(path).values) == 10
    with pytest.raises(NotAModelFile, match="arrays hold more than"):
        scan_gguf(path, limits=Limits(items=100))


def test_nesting_is_capped(tmp_path):
    inner = i(ARRAY) + q(1)
    body = s("k") + i(ARRAY) + inner * 3 + i(U8) + q(1) + b"\x00"
    path = gguf(tmp_path, body, kv=1)
    assert scan_gguf(path, limits=Limits(depth=8)).values["k"] == [[[[0]]]]
    with pytest.raises(NotAModelFile, match="nest deeper"):
        scan_gguf(path, limits=Limits(depth=2))


def test_a_header_that_takes_too_long_is_refused(tmp_path):
    body = b"".join(pair(f"k{n}", U32, i(n)) for n in range(50))
    with pytest.raises(NotAModelFile, match="took more than"):
        scan_gguf(gguf(tmp_path, body, kv=50), limits=Limits(seconds=-1))


def test_a_file_that_is_not_a_gguf_is_refused_by_name(tmp_path):
    path = tmp_path / "x.gguf"
    path.write_bytes(b"NOPE" + b"\x00" * 64)
    with pytest.raises(NotAModelFile, match="not a GGUF file"):
        scan_gguf(path)
    assert header.meta(path) is None and in_gguf(path) == {}


def test_the_listing_header_keeps_short_arrays_and_counts_the_vocabulary(tmp_path):
    got = header.meta(honest(tmp_path))
    assert got is not None
    assert got["small"] == [1, 2, 3] and got[header.VOCAB] == 2
    assert "tokenizer.ggml.tokens" not in got


def test_the_sampling_settings_are_read_from_the_pairs_asked_for(tmp_path):
    assert in_gguf(honest(tmp_path)) == {"temperature": 0.5}


def test_a_hostile_file_is_a_miss_for_a_listing_and_never_an_exception(tmp_path):
    for path in hostile_files(tmp_path).values():
        assert header.meta(path) is None
        assert in_gguf(path) == {}


# ------------------------------------------------------------------ safetensors


def safetensors(tmp_path, body: bytes, declared: int | None = None, name: str = "w.safetensors"):
    path = tmp_path / name
    path.write_bytes((len(body) if declared is None else declared).to_bytes(8, "little") + body)
    return path


def test_an_honest_safetensors_header_is_read(tmp_path):
    body = json.dumps({"__metadata__": {"format": "pt"},
                       "w": {"dtype": "F16", "shape": [2], "data_offsets": [0, 4]}}).encode()
    got = safetensors_header(safetensors(tmp_path, body))
    assert got["w"]["data_offsets"] == [0, 4]


@pytest.mark.parametrize("body,declared", [
    (b"{}", 2**50),
    (b"{}", 2**64 - 1),
    (b"{}", 1000),
    (b"[]", None),
    (b"not json", None),
    (b"[" * 100_000, None),
    (json.dumps({"w": {"data_offsets": [5, 1]}}).encode(), None),
    (json.dumps({"w": {"data_offsets": [-1, 4]}}).encode(), None),
    (json.dumps({"w": {"data_offsets": ["a", "b"]}}).encode(), None),
    (json.dumps({"w": {"data_offsets": [0]}}).encode(), None),
    (json.dumps({"w": {"data_offsets": [True, 4]}}).encode(), None),
    (json.dumps({"w": 3}).encode(), None),
], ids=["2**50", "2**64", "past the end", "array", "text", "nested brackets", "reversed",
        "negative", "strings", "one offset", "boolean", "no entry"])
def test_a_lying_safetensors_header_is_refused(tmp_path, body, declared):
    began = time.monotonic()
    with pytest.raises(NotAModelFile, match="safetensors"):
        safetensors_header(safetensors(tmp_path, body, declared))
    assert time.monotonic() - began < 2


def test_a_safetensors_file_shorter_than_its_length_field_is_refused(tmp_path):
    path = tmp_path / "w.safetensors"
    path.write_bytes(b"\x01\x00")
    with pytest.raises(NotAModelFile, match="not a safetensors file"):
        safetensors_header(path)


def test_the_resident_size_of_a_model_directory_uses_the_bounded_reader(tmp_path):
    body = json.dumps({"w": {"dtype": "F16", "shape": [2], "data_offsets": [0, 40]}}).encode()
    safetensors(tmp_path, body)
    assert resident_bytes(tmp_path) == 40
    safetensors(tmp_path, b"{}", declared=2**50, name="bad.safetensors")
    with pytest.raises(ValueError, match="not a safetensors file"):
        resident_bytes(tmp_path)
