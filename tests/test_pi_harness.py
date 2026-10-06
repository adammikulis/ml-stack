"""Pi runs with the explicit local endpoint and ml-stack's tool-policy extension."""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

from ml_stack import pi


def test_pi_launch_is_offline_and_uses_the_policy_extension(monkeypatch, tmp_path):
    seen = {}
    binary = tmp_path / "pi"
    binary.touch()
    config = tmp_path / "pi-config"
    config.mkdir()

    class Files:
        path = config

        def write(self, name, content):
            target = config / name
            target.write_text(content, encoding="utf-8")
            return target

    run = SimpleNamespace(files=Files(), cwd=tmp_path, brief="trusted role brief",
                          seat=SimpleNamespace(name="coding-test"))

    @contextmanager
    def opened(*_args, **_kwargs):
        yield run

    monkeypatch.setattr(pi.harnessing, "binary_for", lambda _: str(binary))
    monkeypatch.setattr(pi, "alias_of", lambda *_: "local-model")
    monkeypatch.setattr(pi.harnessing, "window_of", lambda _: 49152)
    monkeypatch.setattr(pi.harnessing, "opened", opened)
    monkeypatch.setattr(pi.harnessing, "protected_paths", lambda _: [str(config)])

    def run_pi(command, env):
        seen["command"], seen["env"] = command, env
        return 0

    assert pi.launch(["--on", "http://127.0.0.1:8080", "--project", str(tmp_path)],
                     say=lambda _: None, run_pi=run_pi) == 0
    assert seen["command"][:5] == [str(binary), "--provider", "mlstack", "--model", "local-model"]
    assert "--no-session" in seen["command"] and "--no-mcp" in seen["command"]
    extension = Path(seen["command"][seen["command"].index("--extension") + 1]).read_text()
    assert 'pi.on("tool_call"' in extension and "ml_stack.harnesshook" in extension
    assert seen["env"]["PI_OFFLINE"] == "1"
    assert seen["env"]["PI_CODING_AGENT_DIR"] == str(config)


def test_pi_events_stream_usage_and_fail_closed():
    from ml_stack.workspace.coding_events import event

    assert event({"type": "message_update", "assistantMessageEvent":
        {"type": "text_delta", "delta": "Hello"}}, "pi") == {"delta": "Hello"}
    assert event({"type": "turn_end", "message": {"usage": {"output": 17}}}, "pi") == {"usage": {"output": 17}}
    assert event({"type": "message_end", "message": {"role": "assistant",
        "stopReason": "error", "errorMessage": "local runtime failed"}}, "pi") == {"error": "local runtime failed"}
    assert event({"type": "tool_execution_end", "isError": True, "result":
        {"content": [{"type": "text", "text": "ml-stack: person refused"}]}}, "pi") == {"blocked": "ml-stack: person refused"}


def test_pi_extension_denies_writes_and_reports_exhausted_turns(tmp_path):
    import json
    import shutil
    import subprocess
    import sys

    import pytest

    node = shutil.which("node")
    if not node:
        pytest.skip("Node is required by Pi")
    hook = tmp_path / "policy.mjs"
    hook.write_text(pi.extension({"python": sys.executable, "role": "read-only",
        "label": "isolated-pi", "root": str(tmp_path), "protected": [], "maxTurns": 1}))
    driver = tmp_path / "driver.mjs"
    driver.write_text(
        'import extension from "./policy.mjs";\n'
        'const handlers={}; extension({on:(name,fn)=>handlers[name]=fn});\n'
        'const state={}; const ctx={abort:()=>state.aborted=true,shutdown:()=>state.stopped=true};\n'
        'handlers.turn_start({},ctx); handlers.turn_start({},ctx);\n'
        'const result=handlers.tool_call({toolName:"write",input:{path:"output.py",content:"x"}});\n'
        'console.log(JSON.stringify({state,result}));\n')
    done = subprocess.run([node, str(driver)], capture_output=True, text=True, timeout=30, check=True)
    rows = [json.loads(line) for line in done.stdout.splitlines()]
    assert rows[0] == {"type": "error", "message": "Pi turn budget exhausted"}
    assert rows[1]["state"] == {"aborted": True, "stopped": True}
    assert rows[1]["result"]["block"] and "ml-stack:" in rows[1]["result"]["reason"]
    assert not (tmp_path / "output.py").exists()
