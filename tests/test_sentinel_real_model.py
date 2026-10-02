"""Canaries and integrity against real GGUF files and a real llama-server, one server at a time.

Runs only where a llama-server binary and two small GGUFs exist and no other llama-server is
running; the files in the Hugging Face cache are only read, never changed.
"""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path

import pytest

from ml_stack.client import Client
from ml_stack.sentinel import Mode, Sentinel, State, canary
from ml_stack.serve import LlamaServerBackend, ServerManager, ServerSpec, free_port
from ml_stack.serve.process import every_server

pytestmark = pytest.mark.slow


BASE = "Qwen3VL-2B-Instruct-Q4_K_M.gguf"
OTHER = "Qwen3Guard-Gen-0.6B.Q4_K_M.gguf"


def _gguf(account: Path, name: str) -> Path | None:
    root = account / ".cache" / "huggingface" / "hub"
    found = sorted(root.rglob(name)) if root.is_dir() else []
    return found[0] if found else None


def _binary(account: Path) -> Path | None:
    named = os.environ.get("ML_STACK_TEST_LLAMA_SERVER") or shutil.which("llama-server")
    if named:
        return Path(named)
    builds = sorted((account / ".ml-stack" / "llama.cpp" / "builds").glob("*/llama-server"))
    return builds[-1] if builds else None


@pytest.fixture
def files(_real_home):
    account = _real_home.state.parent
    base, other = _gguf(account, BASE), _gguf(account, OTHER)
    if base is None or other is None:
        pytest.skip(f"needs {BASE} and {OTHER} in the Hugging Face cache")
    return base, other


@pytest.fixture
def machine(files, _real_home):
    binary = _binary(_real_home.state.parent)
    if binary is None:
        pytest.skip("no llama-server binary on this machine")
    if every_server():
        pytest.skip("another llama-server is running; a test does not load a model beside it")
    return binary, list(files)[::-1]


def _asker(binary: Path, model: Path):
    """A context manager yielding ``ask(prompt)`` against one leased server."""
    held = ServerManager(LlamaServerBackend(binary=binary))
    info = held.lease(ServerSpec(model=model, port=free_port(), context=512), timeout=300)
    client = Client(info.base_url)

    def ask(prompt: str) -> str:
        reply = client.chat([{"role": "user", "content": prompt}], max_tokens=64, temperature=0.0)
        return reply.content or ""

    return held, info, ask


def test_a_real_model_passes_its_own_baseline_and_a_different_model_drifts(
        machine, tmp_path, monkeypatch):
    binary, models = machine
    monkeypatch.setenv("LLAMA_CPP_SERVER", str(binary))
    other_model, base_model = models
    node = Sentinel(tmp_path / "s", mode=Mode.GUARDED, roots=[tmp_path])
    held, info, ask = _asker(binary, base_model)
    try:
        baseline = node.baseline("real", ask, runs=3)
        again = canary.run(ask, runs=3)
        assert not canary.compare(baseline, again).drifted
        assert node.canary("real", ask, runs=3) is None
        shared = {i: baseline.passes[i] for i in baseline.runs}
        assert sum(shared.values()) >= 0.9 * sum(baseline.runs.values())
    finally:
        held.release(info)
    time.sleep(3)
    held2, info2, ask2 = _asker(binary, other_model)
    try:
        other = canary.run(ask2, runs=3)
    finally:
        held2.release(info2)
    drift = canary.compare(baseline, other)
    print(f"baseline passes {shared}; other passes {other.passes}; "
          f"drifted={drift.drifted} probes={drift.probes}")
    assert drift.drifted


def test_a_real_gguf_is_hashed_pinned_and_a_single_flipped_byte_is_caught(files, tmp_path):
    source = files[1]
    copy = tmp_path / "models" / source.name
    copy.parent.mkdir()
    shutil.copyfile(source.resolve(), copy)
    node = Sentinel(tmp_path / "s", roots=[tmp_path / "models"])
    started = time.perf_counter()
    node.manifest.pin(copy, "model", f"copy of {source.name}")
    seconds = time.perf_counter() - started
    size = copy.stat().st_size
    print(f"pinned {size / 1e6:.0f} MB in {seconds:.2f}s ({size / 1e6 / seconds:.0f} MB/s)")
    assert node.verify_before_load(copy)
    with copy.open("r+b") as handle:
        handle.seek(size // 2)
        byte = handle.read(1)
        handle.seek(size // 2)
        handle.write(bytes([byte[0] ^ 1]))
    assert node.verify_before_load(copy) is False
    assert node.store.state_of("model", str(copy)) == State.QUARANTINED
    assert not copy.exists()
    assert source.resolve().exists()
