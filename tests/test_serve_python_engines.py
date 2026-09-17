"""vLLM and SGLang specs are served by their engine, through the same lease as llama-server."""

from __future__ import annotations

import sys

import pytest

from ml_stack.serve import ServerManager, ServerSpec
from ml_stack.serve.backend import ServerFailed
from ml_stack.serve.ports import free_port
from ml_stack.serve.process import pid_exists
from ml_stack.serve.python_engines import ENGINES, SGLangBackend, VllmBackend


def test_each_engine_builds_its_own_server_command():
    spec = ServerSpec(model="Qwen/Qwen3-0.6B", port=9001, context=8192, parallel=4, engine="vllm")
    vllm = VllmBackend().command(spec)
    assert vllm[:3] == [sys.executable, "-m", "vllm.entrypoints.openai.api_server"]
    assert vllm[vllm.index("--max-model-len") + 1] == "8192"
    assert vllm[vllm.index("--max-num-seqs") + 1] == "4"
    sglang = SGLangBackend().command(spec)
    assert sglang[:3] == [sys.executable, "-m", "sglang.launch_server"]
    assert sglang[sglang.index("--context-length") + 1] == "8192"


def test_the_manager_serves_a_spec_with_the_engine_it_names(tmp_path):
    manager = ServerManager(state_file=tmp_path / "servers.json")
    assert manager.backend_for(ServerSpec(model="m", engine="sglang")) is ENGINES["sglang"]
    with pytest.raises(ServerFailed, match="no serving engine"):
        manager.backend_for(ServerSpec(model="m", engine="nope"))


def test_an_engine_server_is_leased_recorded_and_released(tmp_path, monkeypatch):
    """A real process on a real port: a stand-in engine module that answers /health."""
    from ml_stack.serve import python_engines

    engine = tmp_path / "engine.py"
    engine.write_text(
        "import sys\n"
        "from http.server import BaseHTTPRequestHandler, HTTPServer\n"
        "class H(BaseHTTPRequestHandler):\n"
        "    def do_GET(self):\n"
        "        self.send_response(200); self.end_headers(); self.wfile.write(b'{}')\n"
        "    def log_message(self, *a):\n"
        "        pass\n"
        "HTTPServer(('127.0.0.1', int(sys.argv[sys.argv.index('--port') + 1])), H).serve_forever()\n")
    monkeypatch.setattr(python_engines, "log_dir", lambda: tmp_path / "logs")
    monkeypatch.setattr(VllmBackend, "command",
                        lambda self, spec: [sys.executable, str(engine), "--port", str(spec.port)])
    manager = ServerManager(state_file=tmp_path / "servers.json")
    info = manager.lease(ServerSpec(model="Qwen/Qwen3-0.6B", port=free_port(), engine="vllm"),
                         roam=False, timeout=30.0)
    try:
        assert info.backend == "vllm" and pid_exists(info.pid)
    finally:
        manager.release(info)
    assert not pid_exists(info.pid)
