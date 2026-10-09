"""The track flow against the real llama.cpp: the real smoke test on this machine's llama-server, and a
real sandboxed build of a pinned upstream tag. Needs cmake, git, a compiler, the network and a small
local GGUF; each test says what is missing."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from poolhouse import net
from poolhouse.serve import (
    binary,
    llamacpp_compile,
    llamacpp_smoke,
    llamacpp_state,
    llamacpp_update,
)
from poolhouse.serve.build_paths import BuildFailed, builds_dir
from poolhouse.serve.process import every_server
from tests.test_serve_real_llama import small_gguf

pytestmark = pytest.mark.slow

PINNED_TAG = "b11379"
"""A llama.cpp build tag (its release is a prerelease cut from master); the real build test builds this one."""


def _model(account: Path) -> Path:
    found = small_gguf(account)
    if found is None:
        pytest.skip("no small local GGUF; set POOLHOUSE_SMOKE_GGUF")
    if every_server():
        pytest.skip("another llama-server is running; a test does not load a model beside it")
    return found


def test_the_smoke_test_passes_on_this_machines_llama_server(_real_home):
    server = shutil.which("llama-server")
    if server is None:
        pytest.skip("no llama-server on PATH")
    got = llamacpp_smoke.run(Path(server), _model(_real_home.state.parent), timeout=300)
    assert got.passed, got.checks
    assert [name for name, ok, _ in got.checks] == ["lease", "health", "chat completion", "top_logprobs",
                                                    "slot save and restore"]


def test_a_pinned_upstream_tag_builds_in_the_sandbox_and_passes_the_smoke_test(_real_home):
    model = _model(_real_home.state.parent)
    try:
        tools = llamacpp_compile.toolchain()
    except BuildFailed as exc:
        pytest.skip(str(exc))
    try:
        got = llamacpp_update.update(PINNED_TAG, llamacpp_update.Env(tools=tools, say=print), model=model)
    except net.NeedsApproval as exc:
        pytest.skip(str(exc))
    assert got.status == "built", got
    assert got.build.startswith("b11379-")
    active = llamacpp_state.active()
    assert active is not None and active.name == got.build and active.good
    assert (builds_dir() / got.build / "llama-server").is_file()
    assert binary.find_binary("llama-server") == (builds_dir() / got.build / "llama-server").resolve()
    assert "build 11379" in active.info["version"]
