"""A real llama-server behind the broker and the request queue.

Runs only on a machine with a llama-server binary, a small GGUF (``POOLHOUSE_TEST_GGUF``, else the
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

from poolhouse import gate
from poolhouse.client import Client
from poolhouse.hub import header
from poolhouse.serve import LlamaServerBackend, ServerManager, ServerSpec, free_port
from poolhouse.serve.process import every_server, pid_exists

pytestmark = pytest.mark.slow
SRC = str(Path(__file__).resolve().parent.parent / "src")
LIMIT = 4 * 1024 ** 3


def generates(path: Path) -> bool:
    """Whether the file's header names a model that writes text, not an embedder or a draft head."""
    architecture = str((header.meta(path) or {}).get("general.architecture", ""))
    return bool(architecture) and "bert" not in architecture and "assistant" not in architecture


def small_gguf(account: Path) -> Path | None:
    named = os.environ.get("POOLHOUSE_TEST_GGUF")
    if named:
        return Path(named)
    found = [p for root in (account / ".cache" / "huggingface" / "hub", account / ".cache")
             if root.is_dir() for p in root.rglob("*.gguf")
             if "vocab" not in p.name and "mmproj" not in p.name and "-of-" not in p.name
             and "mtp" not in p.name.lower() and 0 < p.stat().st_size < LIMIT and generates(p)]
    return min(found, key=lambda p: p.stat().st_size) if found else None


def llama_server(account: Path) -> Path | None:
    """The machine's own build, looked for under the account's home, which the suite moves
    away from the code under test."""
    named = os.environ.get("POOLHOUSE_TEST_LLAMA_SERVER") or shutil.which("llama-server")
    if named:
        return Path(named)
    builds = sorted((account / ".poolhouse" / "llama.cpp" / "builds").glob("*/llama-server"))
    return builds[-1] if builds else None


@pytest.fixture
def real(_real_home):
    account = _real_home.state.parent
    binary = llama_server(account)
    if binary is None:
        pytest.skip("no llama-server binary on this machine")
    model = small_gguf(account)
    if model is None:
        pytest.skip("no small GGUF to load; set POOLHOUSE_TEST_GGUF")
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
        assert gate.device_of(url) == "gpu"

        seen: list[list[int]] = []
        stop = threading.Event()

        def watch() -> None:
            while not stop.is_set():
                rows = gate.snapshot().get("gpu", [])
                if rows:
                    seen.append([r.get("pid") for r in rows])
                time.sleep(0.01)

        code = ("import sys\nfrom poolhouse.client import Client\n"
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


def test_a_decision_model_is_served_only_through_the_broker(real, monkeypatch, tmp_path):
    """Labelled `decision`, the smallest local model still takes a lease, and the decide
    package then asks the server that lease returned."""
    from poolhouse.decide import router
    from poolhouse.hub import kinds

    binary, model = real
    assert kinds.classify(path=model, held=[model.parent]) == "decision"
    assert kinds.classify(path=model, held=[tmp_path]) != "decision", "a label needs a registry"
    monkeypatch.setenv("LLAMA_CPP_SERVER", str(binary))
    held = ServerManager(LlamaServerBackend(binary=binary))
    info = held.lease(ServerSpec(model=model, port=free_port(), context=512), timeout=300)
    try:
        assert info.lease and pid_exists(info.pid)
        config = router.Config(backend="logprob", url=info.base_url)
        assert router.unavailable("logprob", config) == ""
        # The smallest local model (270M) puts about 0.48 of its probability on a valid letter (measured: 0.472-0.486
        # on llama.cpp 0.5.0), under the decider's 0.5 floor. Failing closed on such a model is the decider working as
        # designed, so the call may end either way; what this test pins is that the question reached the server this
        # lease returned, and that the only other outcome is the documented refusal.
        from poolhouse.decide.types import DecideError

        try:
            got = router.decide("Which letter comes first?", {}, ["a", "b"], config=config)
        except DecideError as exc:
            assert "first token was not one of the option letters" in str(exc), exc
        else:
            assert got.choice in ("a", "b") and got.backend == "logprob"
    finally:
        held.release(info)
