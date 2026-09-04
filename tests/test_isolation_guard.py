"""Both isolation guards in conftest.py actually trip.

Each check runs pytest in a subprocess against a tiny generated test module, so a
guard's own failure lands in that subprocess's exit code and output rather than in this
suite. The generated module lives under ``tests/`` for the run so it inherits the real
``conftest.py`` -- these are not reimplementations of the guards, they are the guards.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
TESTS_DIR = Path(__file__).resolve().parent


def _run_generated(tmp_path: Path, body: str, *, env: dict[str, str] | None = None):
    """Write ``body`` as a test module under ``tests/`` and run pytest on it alone."""
    generated = TESTS_DIR / f"_generated_{tmp_path.name}"
    generated.mkdir()
    try:
        module = generated / "test_generated.py"
        module.write_text(textwrap.dedent(body), encoding="utf-8")
        result = subprocess.run(
            [sys.executable, "-m", "pytest", str(module), "-q", "-p", "no:cacheprovider"],
            cwd=REPO, capture_output=True, text=True, timeout=120, env=env,
        )
        return result.returncode, result.stdout + result.stderr
    finally:
        shutil.rmtree(generated, ignore_errors=True)


pytestmark = pytest.mark.slow


class TestPortGuard:
    def test_a_real_bind_fails_the_test_and_names_the_port(self, tmp_path):
        body = """
            import socket

            def test_binds_a_real_port():
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    s.bind(("127.0.0.1", 8095))
        """
        code, out = _run_generated(tmp_path, body)
        assert code != 0, out
        assert "8095" in out, out
        assert "test_generated.py" in out, out

    def test_the_real_port_marker_opts_out(self, tmp_path):
        body = """
            import socket
            import pytest

            @pytest.mark.real_port
            def test_binds_a_real_port_on_purpose():
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    s.bind(("127.0.0.1", 8096))
        """
        code, out = _run_generated(tmp_path, body)
        assert code == 0, out

    def test_binding_the_ephemeral_port_is_unaffected(self, tmp_path):
        body = """
            import socket

            def test_binds_port_zero():
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(("127.0.0.1", 0))
        """
        code, out = _run_generated(tmp_path, body)
        assert code == 0, out


class TestCacheGuard:
    def test_a_write_to_the_real_state_file_fails_the_run(self, tmp_path):
        fake_home_cache = tmp_path / "impersonated-real-cache"
        body = """
            def test_writes_directly_to_state_file():
                from ml_stack.serve.manager import STATE_FILE

                STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
                STATE_FILE.write_text("{}")
        """
        env = {**os.environ, "ML_STACK_CACHE": str(fake_home_cache)}
        try:
            code, out = _run_generated(tmp_path, body, env=env)
            assert code != 0, out
            assert "real ml_stack state changed" in out, out
            assert "servers.json" in out, out
        finally:
            shutil.rmtree(fake_home_cache, ignore_errors=True)

    def test_another_process_first_server_starting_does_not_fail_the_run(self, tmp_path):
        """A person's serve on this machine writes the state file while the suite runs. When
        it is the first one, the file appears; that is theirs, not a test's."""
        fake_home_cache = tmp_path / "impersonated-real-cache-theirs"
        body = """
            def test_somebody_else_starts_a_server():
                import json

                from ml_stack.serve.manager import STATE_FILE

                STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
                # a pid past this machine's ceiling: no process of ours owns it
                STATE_FILE.write_text(json.dumps({"8080": {"owner_pid": 999999, "port": 8080}}))
        """
        env = {**os.environ, "ML_STACK_CACHE": str(fake_home_cache)}
        try:
            code, out = _run_generated(tmp_path, body, env=env)
            assert code == 0, out
        finally:
            shutil.rmtree(fake_home_cache, ignore_errors=True)

    def test_leaving_it_untouched_passes(self, tmp_path):
        fake_home_cache = tmp_path / "impersonated-real-cache-untouched"
        body = """
            def test_does_nothing():
                assert True
        """
        env = {**os.environ, "ML_STACK_CACHE": str(fake_home_cache)}
        try:
            code, out = _run_generated(tmp_path, body, env=env)
            assert code == 0, out
        finally:
            shutil.rmtree(fake_home_cache, ignore_errors=True)
