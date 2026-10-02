"""Finding every model on a machine, from trees built to look like each tool's folders."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from conftest import write_gguf

from ml_stack import hub
from ml_stack.hub import header, places, scan

META = {"general.architecture": "llama", "general.name": "Tiny", "general.size_label": "7B",
        "general.file_type": 15, "llama.context_length": 4096, "llama.block_count": 32}


def model(path: Path, extra: dict | None = None, pad: int = 0) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_gguf(path, {**META, **(extra or {})})
    if pad:
        with path.open("ab") as f:
            f.write(b"\0" * pad)
    return path


def hf_repo(cache: Path, repo: str, files: dict[str, int], rev: str = "abc123") -> Path:
    """A Hugging Face cache entry: blobs holding the bytes, snapshot symlinks to them."""
    top = cache / ("models--" + repo.replace("/", "--"))
    snap = top / "snapshots" / rev
    (top / "refs").mkdir(parents=True, exist_ok=True)
    (top / "refs" / "main").write_text(rev)
    for name, pad in files.items():
        blob = top / "blobs" / f"{abs(hash((repo, name))):x}"
        model(blob, {"general.name": name}, pad)
        link = snap / name
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(os.path.relpath(blob, link.parent))
    return snap


def ollama(root: Path, name: str, tag: str, *, projector: bool = False,
           host: str = "registry.ollama.ai", ns: str = "library") -> Path:
    blob = root / "blobs" / f"sha256-{name}{tag}"
    model(blob, {"general.name": name}, 100)
    layers = [{"mediaType": "application/vnd.ollama.image.model",
               "digest": f"sha256:{name}{tag}", "size": blob.stat().st_size}]
    if projector:
        side = root / "blobs" / f"sha256-proj{name}"
        model(side, {"general.name": "proj"})
        layers.append({"mediaType": "application/vnd.ollama.image.projector",
                       "digest": f"sha256:proj{name}", "size": side.stat().st_size})
    manifest = root / "manifests" / host / ns / name / tag
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps({"schemaVersion": 2, "layers": layers}))
    return blob


def test_a_header_gives_architecture_quantisation_context_and_parameters(tmp_path):
    got = header.read(model(tmp_path / "a.gguf"))
    assert (got.architecture, got.quantization, got.context_length) == ("llama", "Q4_K_M", 4096)
    assert (got.parameters, got.layers) == (7_000_000_000, 32)


def test_a_header_stops_before_the_tokenizer(tmp_path):
    path = write_gguf(tmp_path / "a.gguf", {**META, "tokenizer.ggml.tokens": ["a"] * 50000})
    assert header.read(path).context_length == 4096


def test_a_file_without_the_magic_has_no_header(tmp_path):
    path = tmp_path / "a.gguf"
    path.write_bytes(b"not a model at all")
    assert header.read(path) is None


def test_parameter_counts_read_from_size_labels():
    assert [header.parameters_of(x) for x in ("8B", "270M", "30B-A3B", "", "x")] == [
        8_000_000_000, 270_000_000, 30_000_000_000, 0, 0]


def test_the_hugging_face_cache_is_read_through_its_snapshot_links(tmp_path):
    hf_repo(tmp_path, "maker/thing-GGUF", {"thing-Q4_K_M.gguf": 50})
    (one,) = hub.discover([tmp_path])
    assert one.id == "hf:maker/thing-GGUF/thing-Q4_K_M.gguf"
    assert (one.repo, one.quantization, one.source) == ("maker/thing-GGUF", "Q4_K_M", "extra")
    assert one.path.is_symlink() and one.size_bytes == one.path.stat().st_size


def test_a_build_folder_stays_in_the_id(tmp_path):
    hf_repo(tmp_path, "maker/big", {"UD-Q4/big-UD-Q4-00001-of-00002.gguf": 1,
                                   "UD-Q4/big-UD-Q4-00002-of-00002.gguf": 1})
    (one,) = hub.discover([tmp_path])
    assert one.id == "hf:maker/big/UD-Q4/big-UD-Q4-00001-of-00002.gguf"
    assert (one.shards, one.is_complete) == (2, True)
    assert one.size_bytes > 2 * 100


def test_a_missing_shard_makes_the_model_incomplete(tmp_path):
    hf_repo(tmp_path, "maker/big", {"big-00001-of-00003.gguf": 1, "big-00002-of-00003.gguf": 1})
    (one,) = hub.discover([tmp_path])
    assert one.is_complete is False


def test_a_snapshot_link_to_nothing_is_not_a_model(tmp_path):
    snap = hf_repo(tmp_path, "maker/thing", {"a.gguf": 0})
    (snap / "b.gguf").symlink_to("../../blobs/gone")
    assert [m.filename for m in hub.discover([tmp_path])] == ["a.gguf"]


def test_ollama_manifests_resolve_to_their_blobs_and_readable_names(tmp_path):
    ollama(tmp_path, "llama3", "latest", projector=True)
    ollama(tmp_path, "mine", "7b", ns="someone")
    ollama(tmp_path, "far", "1", host="example.com", ns="library")
    got = {m.name: m for m in hub.discover([tmp_path])}
    assert set(got) == {"llama3:latest", "someone/mine:7b", "example.com/far:1"}
    first = got["llama3:latest"]
    assert first.id == "ollama:llama3:latest" and first.path.name == "sha256-llama3latest"
    assert first.mmproj.name == "sha256-projllama3"
    assert first.quantization == "Q4_K_M" and first.architecture == "llama"


def test_an_ollama_blob_smaller_than_its_manifest_says_is_incomplete(tmp_path):
    blob = ollama(tmp_path, "cut", "latest")
    blob.write_bytes(blob.read_bytes()[:-50])
    (one,) = hub.discover([tmp_path])
    assert one.is_complete is False


def test_ollama_tensor_manifests_are_not_gguf_models(tmp_path):
    manifest = tmp_path / "manifests" / "registry.ollama.ai" / "library" / "m" / "mlx"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"layers": [
        {"mediaType": "application/vnd.ollama.image.tensor", "digest": "sha256:x", "size": 1}]}))
    (tmp_path / "blobs").mkdir()
    assert hub.discover([tmp_path]) == []


def test_a_model_present_in_two_tools_is_one_row_with_the_better_copy_first(tmp_path):
    cache = tmp_path / "hf"
    snap = hf_repo(cache, "maker/thing-GGUF", {"thing-Q4_K_M.gguf": 10})
    twin = tmp_path / "lms" / "maker" / "thing-GGUF" / "thing-Q4_K_M.gguf"
    model(twin, {"general.name": "thing-Q4_K_M.gguf"}, 10)
    found = hub.discover([places.Place("lmstudio", "flat", twin.parents[2], True),
                          places.Place("huggingface", "hf", cache, True)])
    (one,) = found
    assert one.source == "huggingface" and one.copies == (twin,)
    assert one.path == snap / "thing-Q4_K_M.gguf"


def test_a_vision_projector_beside_the_only_model_is_its_mmproj(tmp_path):
    model(tmp_path / "pic-Q4_K_M.gguf", pad=300)
    model(tmp_path / "mmproj-F16.gguf")
    model(tmp_path / "mmproj-BF16.gguf", pad=900)
    (one,) = hub.discover([tmp_path])
    assert one.mmproj.name == "mmproj-BF16.gguf"
    assert len(hub.discover([tmp_path], companions=True)) == 3


def test_a_projector_beside_two_models_goes_to_the_one_it_names(tmp_path):
    model(tmp_path / "pic-Q4_K_M.gguf", pad=1)
    model(tmp_path / "other-Q4_K_M.gguf", pad=2)
    model(tmp_path / "mmproj-pic-F16.gguf", pad=3)
    got = {m.filename: m.mmproj for m in hub.discover([tmp_path])}
    assert got["pic-Q4_K_M.gguf"].name == "mmproj-pic-F16.gguf"
    assert got["other-Q4_K_M.gguf"] is None


def test_a_download_still_arriving_is_marked_incomplete(tmp_path):
    model(tmp_path / "a.gguf")
    (tmp_path / "a.gguf.downloading").write_bytes(b"x")
    (one,) = hub.discover([tmp_path])
    assert one.is_complete is False


def test_safetensors_folders_are_split_into_mlx_and_plain(tmp_path):
    for name, config in (("mlx-community--x", {"quantization": {"bits": 4}}),
                         ("plain", {"model_type": "bert"})):
        folder = tmp_path / name
        folder.mkdir()
        (folder / "config.json").write_text(json.dumps(config))
        (folder / "model.safetensors").write_bytes(b"0" * 64)
    got = {m.name: m.format for m in hub.discover([tmp_path])}
    assert got == {"mlx-community--x": "mlx", "plain": "safetensors"}
    assert [m.name for m in hub.discover([tmp_path], formats=("mlx",))] == ["mlx-community--x"]
    assert hub.discover([tmp_path], formats=("gguf",)) == []


def test_a_symlink_out_of_every_root_is_not_followed(tmp_path):
    outside = model(tmp_path / "elsewhere" / "secret.gguf")
    root = tmp_path / "models"
    root.mkdir()
    (root / "link.gguf").symlink_to(outside)
    assert hub.discover([root]) == []
    (one,) = hub.discover([root, tmp_path / "elsewhere"])
    assert {one.path.name, *(c.name for c in one.copies)} == {"link.gguf", "secret.gguf"}


def test_a_symlinked_folder_is_not_descended_into(tmp_path):
    target = tmp_path / "elsewhere"
    model(target / "x.gguf")
    root = tmp_path / "models"
    root.mkdir()
    (root / "loop").symlink_to(target, target_is_directory=True)
    assert hub.discover([root]) == []


def test_a_second_call_reads_no_headers(tmp_path, monkeypatch):
    model(tmp_path / "a.gguf")
    hub.discover([tmp_path])
    monkeypatch.setattr(header, "read", lambda p: pytest.fail("header read twice"))
    assert len(hub.discover([tmp_path])) == 1


def test_a_changed_file_is_read_again(tmp_path):
    path = model(tmp_path / "a.gguf")
    hub.discover([tmp_path])
    write_gguf(path, {**META, "general.size_label": "1B"})
    assert hub.discover([tmp_path])[0].parameters == 1_000_000_000


def test_formats_and_sources_filter(tmp_path):
    model(tmp_path / "a.gguf")
    assert hub.discover([tmp_path], include=["ollama"]) == []
    assert len(hub.discover([tmp_path], include=["extra"])) == 1


def test_find_matches_ids_names_and_words(tmp_path):
    ollama(tmp_path / "o", "llama3", "latest")
    model(tmp_path / "Qwen3-4B-Q4_K_M.gguf")
    pool = hub.discover([tmp_path / "o", tmp_path])
    assert [m.name for m in hub.installed_find("llama3", pool)] == ["llama3:latest"]
    assert [m.filename for m in hub.installed_find("qwen3 4b", pool)] == ["Qwen3-4B-Q4_K_M.gguf"]
    assert hub.installed_find("nothing", pool) == []


def test_an_hf_reference_finds_its_installed_copy(tmp_path):
    hf_repo(tmp_path, "maker/thing-GGUF", {"thing-Q4_K_M.gguf": 5, "thing-Q8_0.gguf": 9})
    pool = hub.discover([tmp_path])
    got = hub.installed_for("hf:maker/thing-GGUF/thing-Q8_0.gguf", pool)
    assert got.filename == "thing-Q8_0.gguf"
    assert hub.installed_for("hf:maker/thing-GGUF/thing-Q2_K.gguf", pool) is None
    assert hub.installed_for("hf:other/thing-GGUF/thing-Q8_0.gguf", pool) is None
    assert hub.installed_for("/some/path.gguf", pool) is None


def test_the_standard_folders_are_searched_when_no_roots_are_given(tmp_path, monkeypatch):
    monkeypatch.setattr(places, "system", lambda: "Linux")
    monkeypatch.setenv("HOME", str(tmp_path))
    for var in ("HF_HOME", "HF_HUB_CACHE", "XDG_CACHE_HOME", "LLAMA_CACHE", "OLLAMA_MODELS"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr("ml_stack.home.user_home", lambda: tmp_path)
    hf_repo(tmp_path / ".cache" / "huggingface" / "hub", "maker/a", {"a.gguf": 0})
    model(tmp_path / ".cache" / "llama.cpp" / "b.gguf")
    ollama(tmp_path / ".ollama" / "models", "c", "latest")
    assert {m.source for m in hub.discover()} == {"huggingface", "llama.cpp", "ollama"}


def test_scan_depth_stops_at_the_places_depth(tmp_path):
    model(tmp_path / "a" / "b" / "c.gguf")
    shallow = places.Place("manual", "flat", tmp_path, True, 1)
    assert list(scan.scan(shallow)) == []
    assert len(list(scan.scan(places.Place("manual", "flat", tmp_path, True, 2)))) == 1
