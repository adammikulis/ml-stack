"""Workspace and memory tools import independently of registry initialization order."""

import os
import subprocess
import sys

import pytest


@pytest.mark.parametrize('first', ['poolhouse.workspace.tools', 'poolhouse.memory.tools', 'poolhouse.mcp'])
def test_fresh_tool_import_order_keeps_complete_registries(first):
    code = f"import importlib; importlib.import_module({first!r}); from poolhouse import mcp; from poolhouse.workspace import tools; assert tools.HINTS; assert mcp.schema_of(tools.workspace_reputation)['type'] == 'object'"
    result = subprocess.run([sys.executable, '-c', code], env=dict(os.environ),
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
