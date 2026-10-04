"""Supervise bounded pytest collection and release idle worker CPU permits."""
from __future__ import annotations

import contextlib
import json
import os
import secrets
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import testslots


def run_pytest(command: list[str], want: int = 0, label: str = "pytest", env: dict[str, str] | None = None) -> int:
    environment = dict(os.environ if env is None else env)
    directory = testslots.slots_dir().resolve()
    control = Path(tempfile.mkdtemp(prefix="pytest-", dir=directory))
    token = secrets.token_hex(24)
    process = None
    try:
        with testslots.lease(want, 1, label=f"{label}: collection") as got:
            environment.update(DEV_TEST_SLOTS_DIR=str(directory), DEV_TEST_WORKERS=str(got.workers),
                               DEV_TEST_PYTEST_CONTROL=str(control), DEV_TEST_PYTEST_TOKEN=token)
            for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
                environment.setdefault(name, "1")
            command = [part.replace("{workers}", str(got.workers)) for part in command]
            process = subprocess.Popen(command, env=environment)
            deadline = time.monotonic() + float(environment.get("DEV_TEST_WAIT_S", "3600"))
            while not (control / "ready").exists():
                if process.poll() is not None:
                    return process.returncode
                if time.monotonic() > deadline:
                    raise TimeoutError("testslots: pytest collection did not finish before the wait limit")
                time.sleep(0.02)
            if (control / "ready").read_text() != token:
                raise RuntimeError("testslots: invalid pytest collection token")
        temporary = control / "admitted.tmp"
        temporary.write_text(json.dumps({"token": token}))
        temporary.replace(control / "admitted")
        return process.wait()
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            with contextlib.suppress(subprocess.TimeoutExpired):
                process.wait(timeout=10)
            if process.poll() is None:
                process.kill()
                process.wait()
        shutil.rmtree(control)
