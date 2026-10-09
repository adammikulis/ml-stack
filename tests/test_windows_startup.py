"""Startup imports and resource-limit refusals without POSIX resources."""

import importlib
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class WindowsStartupTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        state = Path(directory.name)
        isolated = patch.dict(os.environ, {
            "POOLHOUSE_HOME": str(state / "state"), "POOLHOUSE_CACHE": str(state / "cache"),
            "POOLHOUSE_NO_REAL_KEYSTORE": "1", "PYTHON_KEYRING_BACKEND": "keyring.backends.fail.Keyring",
            "POOLHOUSE_NOTIFY": "off",
        })
        isolated.start()
        self.addCleanup(isolated.stop)
        self.sandbox_run = importlib.import_module("poolhouse.sandbox.run")
        self.tailnet = importlib.import_module("poolhouse.fleet.tailnet")

    @unittest.skipUnless(os.name == "nt", "Windows startup")
    def test_startup_without_resource_module(self):
        code = """
import builtins
original = builtins.__import__
def without_resource(name, *args, **kwargs):
    if name == 'resource':
        raise ModuleNotFoundError('resource is unavailable')
    return original(name, *args, **kwargs)
builtins.__import__ = without_resource
import poolhouse.sandbox.run
import poolhouse.fleet.launch
import poolhouse.setup
"""
        env = dict(os.environ, POOLHOUSE_NO_REAL_KEYSTORE="1", POOLHOUSE_NOTIFY="off")
        done = subprocess.run([sys.executable, "-c", code], env=env,
                              capture_output=True, text=True, timeout=30)
        self.assertEqual(done.returncode, 0, done.stderr)

    def test_resource_limits_refuse_without_starting_child(self):
        from poolhouse.sandbox.backend import SandboxUnavailable
        from poolhouse.sandbox.policy import AllowUnsandboxed, Limits, Policy

        sandbox_run = self.sandbox_run
        with patch.object(sandbox_run, "resource", None), patch.object(
                sandbox_run, "wrapped", return_value=(["command"], "", None)), patch.object(
                sandbox_run.subprocess, "Popen") as spawn:
            with self.assertRaisesRegex(SandboxUnavailable, "resource limits"):
                sandbox_run.run(["command"], Policy(name="test", limits=Limits()),
                                unsandboxed=AllowUnsandboxed("test"))
            spawn.assert_not_called()

    def test_tailnet_refuses_without_output_limits(self):
        tailnet = self.tailnet
        with patch.object(tailnet, "resource", None), patch.object(
                tailnet.subprocess, "run") as spawn:
            result = tailnet.detect(cli="tailscale")
            self.assertTrue(result.installed)
            self.assertEqual(result.state, "unavailable")
            spawn.assert_not_called()


if __name__ == "__main__":
    unittest.main()
