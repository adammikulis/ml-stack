"""A real llama-server on the GPU inside the sandbox: it answers, it holds the GPU, and the
same confinement refuses a read outside its allow-list and a connection out.

Runs only with a llama-server binary, a small GGUF and no other llama-server running; one
server at a time, stopped by the test.
"""

from __future__ import annotations

import contextlib
import os
import re
import signal
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from ml_stack import sandbox
from ml_stack.client import Client, wait_for_health
from ml_stack.sandbox import policies
from ml_stack.serve import LlamaServerBackend, ServerManager, ServerSpec, free_port
from ml_stack.serve.process import every_server, pid_exists
from tests.test_serve_real_llama import LIMIT, llama_server

pytest_plugins = ["tests.sandbox_kit"]

pytestmark = pytest.mark.slow


def chat_gguf(account: Path) -> Path | None:
    """The smallest GGUF under the Hugging Face cache that is a model on its own."""
    named = os.environ.get("ML_STACK_TEST_GGUF")
    if named:
        return Path(named)
    root = account / ".cache" / "huggingface" / "hub"
    found = [p for p in root.rglob("*.gguf") if 0 < p.stat().st_size < LIMIT
             and any(w in p.name.lower() for w in ("instruct", "chat", "-it", "granite", "qwen"))
             and not any(w in p.name.lower() for w in ("vocab", "mmproj", "mtp", "embed", "-of-"))]
    return min(found, key=lambda p: p.stat().st_size) if found else None


@pytest.fixture
def real(_real_home, seatbelt):
    if sys.platform != "darwin":
        pytest.skip("the GPU rules are Metal's")
    account = _real_home.state.parent
    binary, model = llama_server(account), chat_gguf(account)
    if binary is None or model is None:
        pytest.skip("no llama-server binary or small GGUF on this machine")
    if every_server():
        pytest.skip("another llama-server is running; a test does not load a model beside it")
    return binary, model


def test_a_confined_server_answers_with_the_model_on_the_gpu(real, monkeypatch):
    binary, model = real
    monkeypatch.setenv("LLAMA_CPP_SERVER", str(binary))
    held = ServerManager(LlamaServerBackend(binary=binary, sandboxed=True))
    info = held.lease(ServerSpec(model=model, port=free_port(), context=512,
                                 n_gpu_layers=99), timeout=300)
    try:
        assert info.pid and pid_exists(info.pid)
        answer = Client(info.base_url).chat([{"role": "user", "content": "Say hi."}],
                                            max_tokens=16)
        assert answer.content.strip()
    finally:
        held.release(info)


def test_every_layer_is_offloaded_to_the_gpu_inside_the_profile(real, tmp_path):
    binary, model = real
    port = free_port()
    policy = replace(policies.model_server(binary, model.parent, gpu=True),
                     read=(*policies.model_server(binary, model.parent).read,
                           str(Path(os.path.realpath(model)).parent)))
    argv, _tag, _chosen = sandbox.wrapped(
        [str(binary), "-m", str(model), "--host", "127.0.0.1", "--port", str(port), "-ngl", "99",
         "-c", "512", "--no-warmup", "-lv", "4"], policy)
    log = tmp_path / "server.log"
    with log.open("wb") as sink:
        process = subprocess.Popen(argv, stdout=sink, stderr=subprocess.STDOUT, env=dict(policy.env),
                                   cwd=str(Path(os.path.realpath(binary)).parent),
                                   start_new_session=True)
    try:
        assert wait_for_health(f"http://127.0.0.1:{port}", timeout=120,
                               is_alive=lambda: process.poll() is None)
        text = log.read_text(errors="replace")
        loaded = re.search(r"offloaded (\d+)/(\d+) layers to GPU", text)
        assert loaded and loaded[1] == loaded[2] and "MTL0" in text
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def test_the_same_confinement_refuses_a_read_outside_and_a_connection_out(real, tmp_path):
    binary, model = real
    decoy = tmp_path / "home" / ".ssh" / "id_rsa"
    decoy.parent.mkdir(parents=True)
    decoy.write_text("the-secret-contents")
    policy = replace(policies.model_server(binary, model.parent, gpu=True),
                     exec=(os.path.realpath(binary), "/usr/bin/curl"))
    read = sandbox.run(["/usr/bin/curl", "-sS", f"file://{decoy}"], policy)
    out = sandbox.run(["/usr/bin/curl", "-sS", "-m", "5", "http://192.0.2.1/"], policy)
    assert read.returncode != 0 and "the-secret-contents" not in read.stdout
    assert any(d["target"] == str(decoy) for d in read.denials)
    assert out.returncode != 0
    assert any(d["operation"] == "network-outbound" for d in out.denials)
    assert Path(binary).exists()
