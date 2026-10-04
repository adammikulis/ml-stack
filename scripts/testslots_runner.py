"""Supervise bounded pytest collection and release idle worker CPU permits."""
from __future__ import annotations

import contextlib
import os
import subprocess
import time

import testslots
import testslots_rpc


def run_pytest(command: list[str], want: int = 0, label: str = "pytest", env: dict[str, str] | None = None,
               *, container: bool = False) -> int:
    environment = dict(os.environ if env is None else env)
    for name in ("PYTEST_XDIST_WORKER", "PYTEST_XDIST_WORKER_COUNT", "PYTEST_XDIST_TESTRUNUID"):
        environment.pop(name, None)
    testslots._reject_nested()
    admission = testslots_rpc.Admission(container)
    process = None
    try:
        with testslots.lease(1, 1, label=f"{label}: coordinator"):
            maximum = int(environment.get("DEV_TEST_BUDGET", "0")) or testslots.base_budget()
            workers = min(want or maximum, maximum)
            environment.update(DEV_TEST_SLOTS_DIR=str(testslots.slots_dir().resolve()), DEV_TEST_WORKERS=str(workers),
                               DEV_TEST_PYTEST_ENDPOINT=admission.endpoint, DEV_TEST_PYTEST_TOKEN=admission.token,
                               DEV_TEST_REMOTE_BROKER=str(testslots.slots_dir().resolve()))
            for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
                environment.setdefault(name, "1")
            command = [part.replace("{workers}", str(workers)) for part in command]
            process = subprocess.Popen(command, env=environment)
            deadline = time.monotonic() + float(environment.get("DEV_TEST_WAIT_S", "3600"))
            while not admission.ready.wait(.01):
                if process.poll() is not None:
                    return process.returncode
                if time.monotonic() > deadline:
                    raise TimeoutError("testslots: pytest collection did not finish before the wait limit")
        admission.admitted.set()
        return process.wait()
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            with contextlib.suppress(subprocess.TimeoutExpired):
                process.wait(timeout=10)
            if process.poll() is None:
                process.kill()
                process.wait()
        admission.finish()
