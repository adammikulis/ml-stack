"""A small git tree to ship in a shard test: a stub `scripts/test` that plays the test runner, and two test files.

The stub is the shipped tree's own `scripts/test`, which is what a real shard runs, so the whole path
below the node is real: unpack, a scratch checkout, a child process, junit, the result. What it does
depends on the files it is given: `tests/test_fail.py` fails, `tests/test_hang.py` starts a child and
sleeps (and writes both pids to a file the test names), anything else passes. It also fails when
a variable that marks an agent's shell reached it.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

MARKERS = ("CLAUDECODE", "AI_AGENT", "CLAUDE_CODE_ENTRYPOINT", "POOLHOUSE_WORKSPACE_AGENT", "POOLHOUSE_SESSION_ID",
           "POOLHOUSE_SESSION_HARNESS", "DEV_TEST_AGENT")

STUB = '''\
import os
import subprocess
import sys
import time

MARKERS = %(markers)r
PIDS = %(pids)r
tier = sys.argv[1]
junit = next(a.split("=", 1)[1] for a in sys.argv if a.startswith("--junitxml="))
files = [a for a in sys.argv[2:] if a.startswith("tests/")]
if "tests/test_hang.py" in files:
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"])
    with open(PIDS, "w") as out:
        out.write(f"{os.getpid()} {child.pid}")
    time.sleep(600)
if tier == "gate":
    print("gate: every structural check passed")
    raise SystemExit(0)
leaked = [m for m in MARKERS if m in os.environ]
cases = []
for name in files or ["tests/test_a.py", "tests/test_b.py"]:
    cls = name[:-3].replace("/", ".")
    body = '<failure message="boom: expected 1"/>' if name == "tests/test_fail.py" else ""
    cases.append(f'<testcase classname="{cls}" name="test_one" time="0.25">{body}</testcase>')
if leaked:
    cases.append(f'<testcase classname="tests.test_a" name="test_env" time="0"><failure message="leaked {leaked}"/></testcase>')
with open(junit, "w") as out:
    out.write("<testsuites><testsuite>" + "".join(cases) + "</testsuite></testsuites>")
print("stub ran", tier, files)
raise SystemExit(1 if "tests/test_fail.py" in files or leaked else 0)
'''


def make_tree(root: Path, pids: Path) -> Path:
    """A git working tree at ``root`` with the stub runner and the test files."""
    (root / "scripts").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "scripts" / "test").write_text(STUB % {"markers": MARKERS, "pids": str(pids)}, encoding="utf-8")
    for name in ("test_a", "test_b", "test_fail", "test_hang"):
        (root / "tests" / f"{name}.py").write_text("def test_one():\n    assert True\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=root, check=True, capture_output=True)
    return root
