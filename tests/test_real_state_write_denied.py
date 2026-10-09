"""Under `scripts/test` on a host that can apply it, a test process cannot write under the real state
root (scripts/testwritedeny.py). The environment variable only says the supervisor wrapped this run;
the proof is by effect: opening an existing file under the root for writing (nothing is created,
truncated or written) must be refused. When it is not, these fail without writing anything, whatever
the variable says. Where the run is not under the denial (Windows, a host with no way to apply it,
`DEV_TEST_WRITE_DENY=0`, a direct pytest) they skip."""

from __future__ import annotations

import os
import subprocess
import sys

import pytest
import testwritedeny as deny

from poolhouse import home

CLAIMED = os.environ.get(deny.ENV) == "1"
WHY = "this run is not under the real-state write denial"


@pytest.mark.skipif(not CLAIMED, reason=WHY)
def test_the_real_state_root_is_refused_for_writing_in_this_process():
    root = home.account_roots()[0]
    if deny.probe_file(root) is None:
        pytest.skip("the real state root holds no file to try")
    assert deny.denied_here(root) is True, "the run says it is denied but a write here would succeed"
    for other in deny.candidates():
        if other.exists() and deny.probe_file(other) is not None:
            assert deny.denied_here(other) is True, other


@pytest.mark.skipif(not CLAIMED, reason=WHY)
def test_a_child_process_of_the_test_is_denied_too():
    root = home.account_roots()[0]
    target = deny.probe_file(root)
    if target is None:
        pytest.skip("the real state root holds no file to try")
    code = ("import os, sys\ntry:\n os.close(os.open(sys.argv[1], os.O_WRONLY))\n"
            "except OSError:\n sys.exit(0)\nsys.exit(9)")
    assert subprocess.run([sys.executable, "-c", code, str(target)], check=False).returncode == 0
