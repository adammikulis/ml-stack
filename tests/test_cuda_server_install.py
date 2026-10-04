"""CUDA startup server selection and matching runtime installation."""

import importlib
import os
import subprocess
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import Mock, patch


class CudaServerInstallTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        isolated = patch.dict(os.environ, {"ML_STACK_HOME": str(self.root / "state"),
                                          "ML_STACK_CACHE": str(self.root / "cache"),
                                          "ML_STACK_NO_REAL_KEYSTORE": "1"})
        isolated.start()
        self.addCleanup(isolated.stop)
        self.llama = importlib.import_module("ml_stack.fleet.llama")
        self.token = "ubuntu-cuda-{version}-x64"
        self.name = "llama-nightly-bin-ubuntu-cuda-12.8-x64.zip"
        self.asset = {"name": self.name}
        self.companion = {"name": "cudart-" + self.name}
        self.release = {"assets": [self.companion, self.asset,
                                  {"name": "llama-nightly-bin-ubuntu-x64.zip"}]}

    def test_detected_nvidia_card_selects_cuda_without_a_compiler(self):
        card = Mock(vendor="nvidia")
        with patch.object(self.llama.sys, "platform", "linux"), patch.object(
                self.llama.platform, "machine", return_value="x86_64"), patch.object(
                self.llama, "machine_memory", return_value=Mock(gpus=[card])):
            self.assertEqual(self.llama._tokens(), (self.token,))

    def test_cuda_discovery_does_not_depend_on_help_initializing_devices(self):
        backend = importlib.import_module("ml_stack.serve.backend")
        binary = self.root / "llama-server"
        binary.write_text("test binary")
        def inspect(argv, **options):
            output = ("Available devices:\n  CUDA0: NVIDIA GeForce RTX 3090 Ti "
                      "(24563 MiB, 23281 MiB free)\n") if argv[1] == "--list-devices" else "--help usage"
            return subprocess.CompletedProcess(argv, 0, output, "")
        with patch.object(backend.subprocess, "run", side_effect=inspect) as run:
            self.assertEqual(backend.help_of(binary).strip(), "--help usage")
            self.assertTrue(self.llama.cuda_ready(binary))
            self.assertEqual([call.args[0] for call in run.call_args_list],
                             [[str(binary), "--help"], [str(binary), "--list-devices"]])

    def test_failed_or_non_cuda_device_discovery_is_refused(self):
        backend = importlib.import_module("ml_stack.serve.backend")
        for code, output in ((1, "CUDA0: failed"), (0, "CPU: host"), (0, "CUDA0:")):
            with self.subTest(code=code, output=output), patch.object(
                    backend.subprocess, "run", return_value=subprocess.CompletedProcess(
                        [], code, output, "")):
                self.assertFalse(self.llama.cuda_ready(self.root / "llama-server"))

    def test_runtime_cannot_be_selected_as_server_or_cpu_as_cuda(self):
        with patch.object(self.llama, "_tokens", return_value=(self.token,)):
            self.assertIs(self.llama.asset_for_this_machine(self.release), self.asset)
            self.assertIsNone(self.llama.asset_for_this_machine({"assets": [self.companion,
                {"name": "llama-nightly-bin-ubuntu-x64.zip"}]}))
        self.assertIs(self.llama.cuda_companion(self.asset, self.release), self.companion)
        with self.assertRaisesRegex(self.llama.LlamaError, "matching CUDA runtime"):
            self.llama.cuda_companion(self.asset, {"assets": [self.asset]})

    def archives(self):
        server = self.root / "server.zip"
        runtime = self.root / "runtime.zip"
        with zipfile.ZipFile(server, "w") as archive:
            archive.writestr("bin/" + self.llama.SERVER, "server")
            archive.writestr("bin/libggml-cuda.so", "backend")
        with zipfile.ZipFile(runtime, "w") as archive:
            archive.writestr("runtime/libcudart.so", "runtime")
        return server, runtime

    def test_both_archives_are_downloaded_and_runtime_is_beside_server(self):
        server, runtime = self.archives()
        messages = []
        with patch.object(self.llama, "_tokens", return_value=(self.token,)), patch.object(
                self.llama, "latest", return_value=self.release), patch.object(
                self.llama.shutil, "which", return_value=None), patch.object(
                self.llama, "download", side_effect=[server, runtime]) as download, patch.object(
                self.llama, "cuda_ready", return_value=True):
            binary = self.llama.ensure_server(self.root, on_progress=messages.append)
            self.assertEqual(download.call_args_list[0].args[0], self.asset)
            self.assertEqual(download.call_args_list[1].args[0], self.companion)
            self.assertEqual((binary.parent / "libcudart.so").read_text(), "runtime")
            self.assertTrue(any(self.companion['name'] in message for message in messages))
            self.assertTrue(any(message.startswith('Extracting') for message in messages))
            self.assertEqual(messages[-1], 'Checking the installed server and CUDA devices')

    def test_runtime_without_github_digest_is_refused(self):
        server, _runtime = self.archives()
        original = self.llama.download
        def fetch(asset, directory, **options):
            return server if asset is self.asset else original(asset, directory, **options)
        with patch.object(self.llama, "_tokens", return_value=(self.token,)), patch.object(
                self.llama, "latest", return_value=self.release), patch.object(
                self.llama.shutil, "which", return_value=None), patch.object(
                self.llama, "download", side_effect=fetch), self.assertRaisesRegex(
                self.llama.LlamaError, "sha256 digest"):
            self.llama.ensure_server(self.root)

    def test_server_that_cannot_initialize_cuda_is_refused(self):
        server, runtime = self.archives()
        with patch.object(self.llama, "_tokens", return_value=(self.token,)), patch.object(
                self.llama, "latest", return_value=self.release), patch.object(
                self.llama.shutil, "which", return_value=None), patch.object(
                self.llama, "download", side_effect=[server, runtime]), patch.object(
                self.llama, "cuda_ready", return_value=False), self.assertRaisesRegex(
                self.llama.LlamaError, "cannot initialize a CUDA device"):
            self.llama.ensure_server(self.root)


if __name__ == "__main__":
    unittest.main()
