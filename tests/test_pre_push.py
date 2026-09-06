"""The git hook that refuses a push nobody asked for."""

import os
import subprocess
from pathlib import Path

HOOK = Path(__file__).resolve().parent.parent / "scripts" / "hooks" / "pre-push"
LINE = "refs/heads/main deadbee refs/heads/main f00ba12\n"


def push(**env: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [str(HOOK), "origin", "https://example.invalid/x.git"],
        text=True, capture_output=True, input=LINE, env={**os.environ, **env})


def test_a_push_is_refused():
    done = push()
    assert done.returncode != 0
    assert "refused" in done.stderr


def test_the_refusal_says_what_to_do_instead():
    assert "ML_STACK_PUSH=yes" in push().stderr


def test_a_person_can_push_on_purpose():
    assert push(ML_STACK_PUSH="yes").returncode == 0


def test_the_installer_wires_it_up():
    installer = HOOK.parent.parent / "install-hooks.sh"
    assert "pre-push pre-push" in installer.read_text()
