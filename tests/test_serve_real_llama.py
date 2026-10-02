"""A real llama-server behind the broker and the request queue.

Runs only on a machine with a llama-server binary, a small GGUF (``ML_STACK_TEST_GGUF``, else the
smallest under the Hugging Face cache or ``~/.cache``), and no other llama-server running:
starting a model beside someone else's work is what this library exists to prevent.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from ml_stack import gate
from ml_stack.client import Client
from ml_stack.serve import LlamaServerBackend, ServerManager, ServerSpec, free_port
from ml_stack.serve.process import every_server, pid_exists

pytestmark = pytest.mark.slow
SRC = str(Path(__file__).resolve().parent.parent / "src")
LIMIT = 4 * 1024 ** 3


def small_gguf(account: Path) -> Path | None:
    named = os.environ.get("ML_STACK_TEST_GGUF")
    if named:
        return Path(named)
    found = [p for root in (account / ".cache" / "huggingface" / "hub", account / ".cache")
             if root.is_dir() for p in root.rglob("*.gguf")
             if "vocab" not in p.name and "mmproj" not in p.name and "-of-" not in p.name
             and 0 < p.stat().st_size < LIMIT]
    return min(found, key=lambda p: p.stat().st_size) if found else None


def llama_server(account: Path) -> Path | None:
    """The machine's own build, looked for under the account's home, which the suite moves
    away from the code under test."""
    named = os.environ.get("ML_STACK_TEST_LLAMA_SERVER") or shutil.which("llama-server")
    if named:
        return Path(named)
    builds = sorted((account / ".ml-stack" / "llama.cpp" / "builds").glob("*/llama-server"))
    return builds[-1] if builds else None


@pytest.fixture
def real(_real_home):
    account = _real_home.state.parent
    binary = llama_server(account)
    if binary is None:
        pytest.skip("no llama-server binary on this machine")
    model = small_gguf(account)
    if model is None:
        pytest.skip("no small GGUF to load; set ML_STACK_TEST_GGUF")
    if every_server():
        pytest.skip("another llama-server is running; a test does not load a model beside it")
    return binary, model


def test_a_real_server_is_leased_queued_and_stopped(real, monkeypatch):
    binary, model = real
    monkeypatch.setenv("LLAMA_CPP_SERVER", str(binary))
    held = ServerManager(LlamaServerBackend(binary=binary))
    info = held.lease(ServerSpec(model=model, port=free_port(), context=512), timeout=300)
    try:
        assert info.pid and pid_exists(info.pid) and info.lease
        url = f"{info.base_url}/v1/chat/completions"
        assert gate.pool_of(url) == "gpu"

        seen: list[list[int]] = []
        stop = threading.Event()

        def watch() -> None:
            while not stop.is_set():
                rows = gate.snapshot().get("gpu", [])
                if rows:
                    seen.append([r.get("pid") for r in rows])
                time.sleep(0.01)

        code = ("import sys\nfrom ml_stack.client import Client\n"
                "print(Client(sys.argv[1]).chat([{'role': 'user', 'content': 'Say hi.'}], "
                "max_tokens=24).content)\n")
        watcher = threading.Thread(target=watch)
        watcher.start()
        procs = [subprocess.Popen([sys.executable, "-c", code, info.base_url],
                                  env={**os.environ, "PYTHONPATH": SRC}, stdout=subprocess.PIPE,
                                  text=True) for _ in range(2)]
        outs = [p.communicate(timeout=300)[0] for p in procs]
        stop.set()
        watcher.join()
        assert all(p.returncode == 0 for p in procs), outs
        assert Client(info.base_url).chat([{"role": "user", "content": "hi"}], max_tokens=8)
        assert max(len(row) for row in seen) <= 2
    finally:
        held.release(info)
    deadline = time.monotonic() + 15
    while pid_exists(info.pid) and time.monotonic() < deadline:
        time.sleep(0.2)
    assert not pid_exists(info.pid), "releasing the last lease stopped the server"
