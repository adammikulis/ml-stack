"""Real child-process readiness, bounded requests and stalled inference cleanup."""

import json
import sys
import time

import pytest

from ml_stack.gym import decision_process
from ml_stack.platform import start_process

pytestmark = pytest.mark.redteam


def child(monkeypatch, script):
    monkeypatch.setattr(decision_process, "start_process", lambda argv, **kwargs:
                        start_process([sys.executable, "-u", "-c", script], **kwargs))
    return decision_process.DecisionProcess("local-model")


def wait_event(process):
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        event = process.poll()
        if event:
            return event
        time.sleep(.01)
    pytest.fail("model child did not report readiness")


def test_stalled_inference_has_one_request_and_closes_promptly(monkeypatch):
    process = child(monkeypatch, "import json,sys,time,signal\n"
                    "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
                    "print(json.dumps({'status':'ready'}),flush=True)\n"
                    "sys.stdin.readline()\ntime.sleep(60)\n")
    try:
        assert wait_event(process)["status"] == "ready"
        assert process.submit({"sequence": 1})
        assert not process.submit({"sequence": 2})
        began = time.monotonic()
        for _ in range(100):
            assert process.poll() is None
        assert time.monotonic() - began < .2
    finally:
        began = time.monotonic()
        process.close()
    assert time.monotonic() - began < 2
    assert process.handle.poll() is not None


def test_loading_and_missing_model_are_explicit(monkeypatch):
    process = child(monkeypatch, "import json,time\ntime.sleep(.2)\n"
                    "print(json.dumps({'status':'error','error':'Missing verified model files'}),flush=True)\n")
    try:
        assert process.status == "loading"
        assert not process.submit({"sequence": 1})
        assert wait_event(process)["error"] == "Missing verified model files"
        assert process.status == "error" and process.pending is None
        assert not process.submit({"sequence": 2})
    finally:
        process.close()


def test_decision_spawn_keeps_hostile_checkpoint_as_one_json_argument(monkeypatch, tmp_path):
    record = tmp_path / "argv.json"
    marker = tmp_path / "injected"
    executable = tmp_path / "python; touch injected"
    executable.write_text(f"#!{sys.executable}\nimport json,sys,time\n"
                          f"open({str(record)!r}, 'w').write(json.dumps(sys.argv[1:]))\n"
                          "print(json.dumps({'status':'ready'}),flush=True)\ntime.sleep(60)\n")
    executable.chmod(0o755)
    monkeypatch.setattr(decision_process, "interpreter", lambda: str(executable))
    checkpoint = f"$(touch {marker}); --device=cuda"
    process = decision_process.DecisionProcess(checkpoint)
    try:
        assert wait_event(process)["status"] == "ready"
        argv = json.loads(record.read_text())
        assert argv == ["-m", "ml_stack.gym.decision_process", json.dumps(checkpoint)]
        assert not marker.exists()
    finally:
        process.close()
    assert not marker.exists()
