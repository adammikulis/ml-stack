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
    assert seen["command"][seen["command"].index("--thinking") + 1] == "off"
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


def test_pi_post_checkpoint_failure_stops_the_session(tmp_path):
    import json
    import shutil
    import subprocess
    import sys

    import pytest

    node = shutil.which("node")
    if not node:
        pytest.skip("Node is required by Pi")
    hook = tmp_path / "policy.mjs"
    hook.write_text(pi.extension({"python": sys.executable, "role": "plan-and-go",
        "label": "missing-checkpoint-identity", "root": str(tmp_path), "protected": []}))
    driver = tmp_path / "driver.mjs"
    driver.write_text(
        'import extension from "./policy.mjs";\n'
        'const handlers={}; extension({on:(name,fn)=>handlers[name]=fn});\n'
        'const state={}; const ctx={abort:()=>state.aborted=true,shutdown:()=>state.stopped=true};\n'
        'handlers.tool_result({},ctx); console.log(JSON.stringify(state));\n')
    done = subprocess.run([node, str(driver)], capture_output=True, text=True, timeout=30, check=True)
    rows = [json.loads(line) for line in done.stdout.splitlines()]
    assert rows[0] == {"type": "error", "message": "Pi mutation checkpoint failed"}
    assert rows[1] == {"aborted": True, "stopped": True}


def test_pi_provider_payload_preserves_explicit_output_budget(tmp_path):
    import json
    import shutil
    import subprocess

    import pytest

    node = shutil.which("node")
    if not node:
        pytest.skip("Node is required by Pi")
    hook = tmp_path / "budget.mjs"
    hook.write_text(pi.extension({"maxOutputTokens": 32000,
                                 "thinking": {"enable_thinking": False}}))
    driver = tmp_path / "driver.mjs"
    driver.write_text(
        'import extension from "./budget.mjs";\n'
        'const handlers={}; extension({on:(name,fn)=>handlers[name]=fn});\n'
        'const original={max_tokens:4096,max_completion_tokens:4096,messages:[{role:"user",content:"hello"}],chat_template_kwargs:{other:1}};\n'
        'console.log(JSON.stringify({original,payload:handlers.before_provider_request({payload:original})}));\n')
    done = subprocess.run([node, str(driver)], capture_output=True, text=True, timeout=30, check=True)
    result = json.loads(done.stdout)
    payload = result["payload"]
    assert payload["max_tokens"] == 32000
    assert "max_completion_tokens" not in payload
    assert payload["messages"] == result["original"]["messages"]
    assert payload["chat_template_kwargs"] == {"other": 1, "enable_thinking": False}
    assert result["original"]["max_tokens"] == 4096
    model = json.loads(pi.models("http://localhost:8080", "local-model", 49152, 32000))["providers"]["mlstack"]["models"][0]
    assert model["contextWindow"] == 49152 and model["maxTokens"] == 32000


def test_native_pi_options_keep_output_effort_and_turns_separate(monkeypatch):
    from ml_stack import coding

    seen = {}
    def launch(argv, **kwargs):
        seen["args"] = argv
        return 0
    monkeypatch.setitem(coding.HARNESSES, "pi", launch)
    assert coding.launch_coding_agent("model", "read-only", "/tmp", max_output_tokens=32000,
                                      max_turns=7, effort="high", context=65536) == 0
    for flag, expected in (("--max-output-tokens", "32000"), ("--max-turns", "7"),
                           ("--effort", "high"), ("--ctx", "65536")):
        assert seen["args"][seen["args"].index(flag) + 1] == expected


def test_chat_agent_output_default_keeps_caller_caps():
    from ml_stack.workspace.localtools import Guarded, Limits

    class Client:
        def chat(self, messages, **kwargs):
            return kwargs
    for effort in ("off", "low", "medium", "high"):
        guarded = Guarded(Client(), effort=effort, limits=Limits(60, 5, max_output_tokens=8192), stop=lambda: False)
        assert guarded.chat([])["n_predict"] == 8192
        assert guarded.chat([], n_predict=32000)["n_predict"] == 32000
        assert guarded.chat([], n_predict=0)["n_predict"] == 0
        explicit = guarded.chat([], max_tokens=17000)
        assert explicit["n_predict"] == 17000 and "max_tokens" not in explicit


def test_guarded_output_cap_reaches_real_client_body():
    from ml_stack.client import Client, Request
    from ml_stack.workspace.localtools import Guarded, Limits

    class BodyClient(Client):
        def chat(self, messages, **kwargs):
            return self.build_body(messages, **kwargs)
    for api in ("llama", "openai"):
        from ml_stack.client import Transport
        client = BodyClient("http://127.0.0.1:1", model="qwen", request=Request(n_predict=2048),
                            transport=Transport(api=api))
        guarded = Guarded(client, effort="off", limits=Limits(60, 5, max_output_tokens=8192), stop=lambda: False)
        key = "max_tokens" if api == "openai" else "n_predict"
        assert guarded.chat([], max_tokens=17000)[key] == 17000
        assert guarded.chat([], max_completion_tokens=19000)[key] == 19000
        assert guarded.chat([])[key] == 8192
