"""Startup imports and resource-limit refusals without POSIX resources."""

import importlib
import os
import subprocess
import sys
import unittest
from unittest.mock import patch

from ml_stack.fleet import tailnet
from ml_stack.sandbox.backend import SandboxUnavailable
from ml_stack.sandbox.policy import AllowUnsandboxed, Limits, Policy

sandbox_run = importlib.import_module("ml_stack.sandbox.run")


class WindowsStartupTests(unittest.TestCase):
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
import ml_stack.sandbox.run
import ml_stack.fleet.launch
import ml_stack.setup
"""
        env = dict(os.environ, ML_STACK_NO_REAL_KEYSTORE="1", ML_STACK_NOTIFY="off")
        done = subprocess.run([sys.executable, "-c", code], env=env,
                              capture_output=True, text=True, timeout=30)
        self.assertEqual(done.returncode, 0, done.stderr)

    def test_resource_limits_refuse_without_starting_child(self):
        with patch.object(sandbox_run, "resource", None), patch.object(
                sandbox_run, "wrapped", return_value=(["command"], "", None)), patch.object(
                sandbox_run.subprocess, "Popen") as spawn:
            with self.assertRaisesRegex(SandboxUnavailable, "resource limits"):
                sandbox_run.run(["command"], Policy(name="test", limits=Limits()),
                                unsandboxed=AllowUnsandboxed("test"))
            spawn.assert_not_called()

    def test_tailnet_refuses_without_output_limits(self):
        with patch.object(tailnet, "resource", None), patch.object(
                tailnet.subprocess, "run") as spawn:
            result = tailnet.detect(cli="tailscale")
            self.assertTrue(result.installed)
            self.assertEqual(result.state, "unavailable")
            spawn.assert_not_called()


if __name__ == "__main__":
    unittest.main()
