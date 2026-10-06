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
