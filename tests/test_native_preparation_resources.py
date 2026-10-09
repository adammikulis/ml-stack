"""Native preparation retains each owned resource across failures."""
import sys
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("failure", ["construct", "recheck", "terminal_close"])
def test_failed_native_preparation_closes_owned_resources(monkeypatch, tmp_path, failure):
    import contextlib

    import test_kernel_isolation
    import test_kernel_lifecycle
    import test_terminal_bank
    import testslots_runner

    calls = []
    primary = RuntimeError("native image preparation failed")

    class Terminals:
        token = "test-token"
        endpoint = "test-endpoint"
        identity = (1, 2, 3)

        def __init__(self, *arguments):
            calls.append("terminal_create")

        def close(self):
            calls.append("terminal_close")
            if failure == "terminal_close":
                raise RuntimeError("terminal close failed")

    class Confined:
        wrapped = SimpleNamespace(argv=["test-command"])

        def __init__(self, *arguments, **options):
            self.environment = {}
            calls.append("confined_create")
            if failure == "construct":
                raise primary

        def recheck_images(self):
            calls.append("recheck")
            raise primary

        def finish(self):
            calls.append("confined_close")

    admission = SimpleNamespace(endpoint="test-endpoint", token="test-token", identity=(1, 2, 3),
                                terminal_admission=lambda: None, finish=lambda: calls.append("admission_close"))
    monkeypatch.setattr(testslots_runner.sys, "platform", "darwin")
    monkeypatch.setattr(testslots_runner, "environment_for", lambda value: {"DEV_TEST_BUDGET": "1", "DEV_TEST_CONFINE": "1"})
    monkeypatch.setattr(testslots_runner.testslots, "slots_dir", lambda: tmp_path)
    monkeypatch.setattr(testslots_runner.testslots, "_reject_nested", lambda: None)
    monkeypatch.setattr(testslots_runner.testslots, "lease", lambda *args, **kwargs: contextlib.nullcontext())
    monkeypatch.setattr(testslots_runner.testslots_rpc, "UnixAdmission", lambda *args: admission)
    monkeypatch.setattr(test_terminal_bank, "TerminalBank", Terminals)
    monkeypatch.setattr(test_kernel_isolation, "ConfinedRun", Confined)
    monkeypatch.setattr(test_kernel_lifecycle, "held_shutdown", lambda: calls.append("held_shutdown"))
    monkeypatch.setattr(test_kernel_lifecycle, "retired_probe", lambda value: calls.append("retired_probe"))

    def spawn(*arguments, **options):
        raise AssertionError("failed preparation launched pytest")

    monkeypatch.setattr(testslots_runner.subprocess, "Popen", spawn)
    with pytest.raises(RuntimeError) as caught:
        testslots_runner.run_pytest([sys.executable, "-m", "pytest", "tests/test_layers.py"], 1)
    assert calls.count("admission_close") == 1
    assert calls.count("terminal_close") == 1
    assert calls.count("confined_close") == (0 if failure == "construct" else 1)
    if failure == "terminal_close":
        assert caught.value.__context__ is primary
    else:
        assert caught.value is primary
