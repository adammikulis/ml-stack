"""Two commands changing `credentials.json` or the passphrase file at once must both land. Separate
processes race over the real files; each child widens the read-to-write window so a missing lock loses a
value every run."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from tests.test_keystore import CHILD_NOWAIT, child_env

RACERS = 8

CREDENTIALS = """
import os, sys, time
from pathlib import Path
from ml_stack import credentials
real = credentials._wrapped
def slow():
    table = real()
    time.sleep(0.15)
    return table
credentials._wrapped = slow
while not Path(os.environ["GO"]).exists():
    time.sleep(0.005)
credentials._keep(sys.argv[1], "value-" + sys.argv[1])
"""

PASSPHRASES = """
import os, sys, time
from pathlib import Path
from ml_stack.fleet import recovery
real = recovery._held
def slow(path):
    rows = real(path)
    time.sleep(0.15)
    return rows
recovery._held = slow
while not Path(os.environ["GO"]).exists():
    time.sleep(0.005)
recovery.remember("pass-" + sys.argv[1], sys.argv[1], sys.argv[2], say=lambda m: print(m, file=sys.stderr))
"""


def race(tmp_path: Path, script: str, *extra: str) -> None:
    env = {**child_env(tmp_path), "ML_STACK_TEST_KEYRING_DELAY": "0"}
    seed = subprocess.run([sys.executable, "-c", CHILD_NOWAIT], env=env, capture_output=True, text=True, timeout=120)
    assert seed.returncode == 0, seed.stderr
    kids = [subprocess.Popen([sys.executable, "-c", script, f"n{i}", *extra], env=env, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, text=True) for i in range(RACERS)]
    (tmp_path / "go").write_text("go")
    results = [(k.communicate(timeout=240), k.returncode) for k in kids]
    assert [code for _, code in results] == [0] * RACERS, [err for (_, err), _ in results if err][:2]
    assert not [err for (_, err), _ in results if "not saved" in err]


def test_concurrent_keychain_sets_all_land_in_credentials_json(tmp_path):
    race(tmp_path, CREDENTIALS)
    values = json.loads((tmp_path / "home" / "keystore" / "credentials.json").read_text())["values"]
    assert sorted(values) == sorted(f"n{i}" for i in range(RACERS))


def test_concurrent_passphrase_saves_all_land_in_the_passphrase_file(tmp_path):
    memberships = tmp_path / "home" / "clusters.json"
    race(tmp_path, PASSPHRASES, str(memberships))
    held = json.loads(memberships.with_suffix(".passphrases").read_text())
    assert sorted(held) == sorted(f"n{i}" for i in range(RACERS))
