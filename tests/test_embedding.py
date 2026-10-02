"""`ml_stack.serve` and `ml_stack.client` as another application imports and calls them."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

import ml_stack.client
import ml_stack.serve
from ml_stack.testing.fakes import fake_llama_binary

SRC = Path(__file__).resolve().parents[1] / "src"
DOC = Path(__file__).resolve().parents[1] / "docs" / "embedding.md"

PROBE = """
import socket, sys
def refuse(*args, **kwargs):
    raise RuntimeError("socket opened at import")
socket.socket.connect = refuse
import ml_stack.serve, ml_stack.client
outside = sorted({name.split('.')[0] for name in sys.modules
                  if not name.startswith('_')
                  and name.split('.')[0] not in sys.stdlib_module_names}
                 - {'ml_stack', 'psutil', 'packaging'})
print(outside)
"""


@pytest.mark.parametrize("module", [ml_stack.serve, ml_stack.client],
                         ids=["serve", "client"])
def test_every_exported_name_resolves(module):
    assert [name for name in module.__all__ if not hasattr(module, name)] == []


def test_importing_them_loads_only_the_standard_library_and_the_core_dependencies():
    done = subprocess.run([sys.executable, "-c", PROBE], capture_output=True, text=True,
                          env={"PYTHONPATH": str(SRC)}, check=False)
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "[]"


def test_the_documented_example_runs_against_a_served_model(tmp_path, monkeypatch):
    block = re.search(r"```python\n(.*?)```", DOC.read_text(encoding="utf-8"), re.S)
    assert block, "docs/embedding.md has no python example"
    model = tmp_path / "model.gguf"
    model.write_bytes(b"GGUF" + b"\x00" * 64)
    monkeypatch.setenv("LLAMA_CPP_SERVER", str(fake_llama_binary(tmp_path)))
    code = block.group(1).replace("/models/model.gguf", str(model))
    exec(compile(code, str(DOC), "exec"), {})  # noqa: S102
