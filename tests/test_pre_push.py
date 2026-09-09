"""The git hook that refuses a push from an agent."""

import os
import subprocess
from pathlib import Path

HOOK = Path(__file__).resolve().parent.parent / "scripts" / "hooks" / "pre-push"
LINE = "refs/heads/main deadbee refs/heads/main f00ba12\n"


def push(**env: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [str(HOOK), "origin", "https://example.invalid/x.git"],
        text=True, capture_output=True, input=LINE, env={**os.environ, **env})


def test_an_agents_push_is_refused():
    done = push(CLAUDECODE="1")
    assert done.returncode != 0
    assert "refused" in done.stderr


def test_the_refusal_says_what_to_do_instead():
    assert "ML_STACK_PUSH=yes" in push(CLAUDECODE="1").stderr


def test_an_agent_told_to_push_can():
    assert push(CLAUDECODE="1", ML_STACK_PUSH="yes").returncode == 0


def test_a_person_is_not_stopped():
    """No terminal sets CLAUDECODE, and neither does a GUI client, so the owner's own
    push meets nothing -- the one push this hook must never be in the way of."""
    done = push(CLAUDECODE="")
    assert done.returncode == 0, done.stderr
    assert done.stderr == ""


def test_the_installer_wires_it_up():
    installer = HOOK.parent.parent / "install-hooks.sh"
    assert "pre-push pre-push" in installer.read_text()
