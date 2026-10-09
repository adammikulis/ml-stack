"""A tiny real git project with a configurable fast gate, for exercising ``scripts/land``."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LAND = ROOT / "scripts" / "land"
DEV = "devline"

FAKE_TEST = '''\
import os, pathlib, sys
log = os.environ.get("LAND_FAKE_LOG")
if log:
    with open(log, "a") as f:
        f.write(" ".join(sys.argv[1:]) + "\\n")
here = pathlib.Path(".")
if list(here.glob("HANG_GATE*")):
    import time
    time.sleep(120)
if sys.argv[1] == "gate":
    both = (here / "COMBO_A").exists() and (here / "COMBO_B").exists()
    sys.exit(1 if list(here.glob("BAD_GATE*")) or both else 0)
bad = 0
for a in sys.argv[2:]:
    p = pathlib.Path(a)
    if a.startswith("tests/") and p.exists() and "FAILME" in p.read_text():
        print(f"FAILED {a}::test_x - boom")
        bad = 1
sys.exit(bad)
'''


def git(cwd: Path, *args: str) -> str:
    """Run git in ``cwd`` and return its output."""
    done = subprocess.run(["git", "-C", str(cwd), *args], capture_output=True, text=True, check=True)
    return done.stdout.strip()


class Project:
    """A repository with a target branch, sibling worktrees for branches and a land runner."""

    def __init__(self, base: Path) -> None:
        self.base = base
        self.root = base / "proj"
        self.root.mkdir()
        git(self.root, "init", "-q", "-b", DEV)
        git(self.root, "config", "user.name", "Test Agent")
        git(self.root, "config", "user.email", "agent@example.invalid")
        git(self.root, "config", "commit.gpgsign", "false")
        self.write(self.root, "scripts/test", FAKE_TEST)
        self.write(self.root, "scripts/budgets", "import sys\nsys.exit(0)\n")
        self.write(self.root, "src/ml_stack/mod.py", "VALUE = 1\n")
        self.write(self.root, "tests/test_mod.py", "import ml_stack.mod\n")
        self.write(self.root, "docs/a.md", "a\n")
        self.commit(self.root, "chore: seed")
        self.log = base / "calls.log"
        self.env = {**os.environ, "ML_STACK_HOME": str(base / "home"), "DEV_TEST_SLOTS_DIR": str(base / "slots"),
                    "LAND_FAKE_LOG": str(self.log), "ML_STACK_DEV_BRANCH": DEV, "ML_STACK_NO_REAL_KEYSTORE": "1",
                    "PYTHON_KEYRING_BACKEND": "onboard_support.FileKeyring",
                    "ML_STACK_TEST_KEYRING": str(base / "keyring.json"),
                    "PYTHONPATH": os.pathsep.join((str(ROOT / "tests"), os.environ.get("PYTHONPATH", "")))}

        # The runner records its passes in the activity log, which a background process may write
        # only once a person has provisioned the keystore; a headless runner has no person to ask.
        subprocess.run([sys.executable, "-c",
                        "from ml_stack import keystore\nkeystore.interactive = lambda: True\n"
                        "keystore.default().provision()\n"],
                       env=self.env, check=True, capture_output=True)

    @staticmethod
    def write(where: Path, rel: str, text: str) -> None:
        """Write ``text`` to ``rel`` under ``where``."""
        path = where / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    @staticmethod
    def commit(where: Path, message: str) -> str:
        """Commit everything in ``where`` and return the new sha."""
        git(where, "add", "-A")
        git(where, "commit", "-q", "-m", message)
        return git(where, "rev-parse", "HEAD")

    def branch(self, name: str, files: dict[str, str], start: str = DEV) -> Path:
        """A worktree on a new branch cut from ``start`` holding one commit of ``files``."""
        where = self.base / name.replace("/", "-")
        git(self.root, "worktree", "add", "-q", "-b", name, str(where), start)
        for rel, text in files.items():
            self.write(where, rel, text)
        if files:
            self.commit(where, f"feat: {name}")
        return where

    def land(self, *args: str) -> tuple[int, str, dict]:
        """Run ``scripts/land`` and return its status, output and JSON summary."""
        done = subprocess.run([sys.executable, str(LAND), *args], cwd=self.root, env=self.env,
                              capture_output=True, text=True, check=False)
        last = done.stdout.strip().splitlines()[-1] if done.stdout.strip() else "{}"
        return done.returncode, done.stdout + done.stderr, json.loads(last)

    def calls(self) -> list[str]:
        """The arguments of every ``scripts/test`` call made so far."""
        return self.log.read_text().splitlines() if self.log.exists() else []
