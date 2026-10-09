"""Under `scripts/test` on a host that can apply it, a test process cannot write under the real state
root (scripts/testwritedeny.py). Where the run is not under the denial (Windows, a host with
no way to apply it, `DEV_TEST_WRITE_DENY=0`, a direct pytest) this skips; it never writes the real
root unless the environment says the denial is in force, and removes what it made if that was wrong."""

from __future__ import annotations

import os

import pytest
import testwritedeny as deny

from ml_stack import home


@pytest.mark.skipif(os.environ.get(deny.ENV) != "1", reason="this run is not under the real-state write denial")
def test_the_real_state_root_cannot_be_written_from_a_test():
    root = home.account_roots()[0]
    probe = root / f".write-deny-test-{os.getpid()}"
    try:
        with pytest.raises(PermissionError):
            probe.write_text("x")
    finally:
        if probe.exists():  # only when the denial failed; a call here is a write the reuse store records
            probe.unlink()


@pytest.mark.skipif(os.environ.get(deny.ENV) != "1", reason="this run is not under the real-state write denial")
def test_a_child_process_of_the_test_is_denied_too(tmp_path):
    import subprocess
    import sys

    root = home.account_roots()[0]
    code = "import sys, pathlib\ntry:\n pathlib.Path(sys.argv[1]).write_text('x')\nexcept PermissionError:\n sys.exit(0)\nsys.exit(9)"
    target = root / f".write-deny-child-{os.getpid()}"
    done = subprocess.run([sys.executable, "-c", code, str(target)], check=False)
    if target.exists():
        target.unlink()
    assert done.returncode == 0
