"""Optional checks against the real Hugging Face endpoint and this machine's llama-server.

Skipped unless ``POOLHOUSE_LIVE=1``; ``POOLHOUSE_LIVE_GGUF`` names an installed GGUF and
``POOLHOUSE_LIVE_LLAMA`` the llama-server binary for the load check.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from poolhouse.hub import remote
from poolhouse.serve.backend import LlamaServerBackend, ServerSpec
from poolhouse.serve.estimate import estimate
from poolhouse.serve.measuring import measure
from poolhouse.serve.ports import free_port


def live() -> None:
    """Stop the calling test unless the person asked for the network and a model server."""
    if os.environ.get("POOLHOUSE_LIVE") != "1":
        pytest.skip("set POOLHOUSE_LIVE=1 to reach the network and a model server")


@pytest.mark.slow
def test_a_public_repository_lists_with_sizes_and_checksums():
    live()
    files = remote.listing("ggml-org/embeddinggemma-2-GGUF")
    ggufs = [f for f in files if f.path.endswith(".gguf")]
    assert ggufs and all(f.size > 0 and len(f.sha256) == 64 for f in ggufs)


@pytest.mark.slow
def test_a_search_finds_gguf_repositories():
    live()
    assert remote.search("embeddinggemma", remote.Filters(limit=3, files=False))


@pytest.mark.slow
def test_a_load_log_reports_the_quantised_cache_the_estimate_assumed():
    live()
    model = os.environ.get("POOLHOUSE_LIVE_GGUF")
    binary = os.environ.get("POOLHOUSE_LIVE_LLAMA") or shutil.which("llama-server")
    if not (model and binary and Path(model).is_file()):
        pytest.skip("POOLHOUSE_LIVE_GGUF and a llama-server are needed")
    spec = ServerSpec(model=model, port=free_port(), context=4096, cache_type_k="q8_0",
                      cache_type_v="q8_0", flash_attn=True)
    seen = measure(spec, backend=LlamaServerBackend(binary=binary), timeout=300)
    assert seen.cache_type == "q8_0"
    want = estimate(model, context=4096)
    assert want.kv_cache_bytes == pytest.approx(seen.kv_bytes + seen.swa_bytes, rel=0.1)
