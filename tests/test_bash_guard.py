"""The Claude Code hook that refuses shells which should have been ml-stack commands.

Every rule it enforces was written after the thing it refuses cost real time, and the guard
itself has one property worth testing hard: it must not fire on text being *written*. It
blocked its own installation twice that way, and it cannot be repaired through the tool it
guards -- so a false positive is far more expensive here than elsewhere.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

GUARD = Path(__file__).resolve().parent.parent / "scripts" / "hooks" / "claude-bash-guard"

BLOCKED, ALLOWED = 2, 0


def guard(command: str, tool: str = "Bash", **env: str) -> int:
    done = subprocess.run(
        [str(GUARD)], text=True, capture_output=True, env={**os.environ, **env},
        input=json.dumps({"tool_name": tool, "tool_input": {"command": command}}))
    assert done.returncode in (BLOCKED, ALLOWED), done.stderr
    if done.returncode == BLOCKED:
        assert "blocked:" in done.stderr, "a refusal has to say what to do instead"
    return done.returncode


@pytest.mark.parametrize("command, why", [
    ('until ! pgrep -f bench; do sleep 20; done', "a waiter that can never fire"),
    ('nohup python3 run.py &', "a background job nobody watches"),
    ('echo hi; nohup ./x &', "still a nohup after a semicolon"),
    ('/opt/homebrew/bin/llama-server --port 8080 -m x.gguf', "a server started by hand"),
    ("find ~/.cache -name '*.gguf'", "hunting for GGUFs"),
    ('curl -s http://127.0.0.1:8080/v1/chat/completions -d "{}"', "an ad-hoc model probe"),
    ('hf download unsloth/Qwen3.5-4B-GGUF', "a download by hand"),
    ('pkill -f llama-server', "a kill that leaves the lease behind"),
    ('SKIP_NAME_CHECK=1 git commit -m x', "the name hook is not an agent's to skip"),
    ('git add -A && git commit -m x', "staging everything sweeps in another agent's half-written work"),
    ('git add --all', "the long spelling of the same"),
    ('git add . ', "the dot is the same"),
    ('git add -u', "every tracked change is the same"),
    ('git commit -am "x"', "commit -a stages everything too"),
])
def test_the_shells_that_should_have_been_commands_are_refused(command, why):
    assert guard(command) == BLOCKED, why


@pytest.mark.parametrize("command", [
    'git status',
    'python3 -m pytest tests -q',
    'ml-stack-serve up model.gguf --port 8080',
    'ml-stack-bench sweep --serve foo.gguf --smoke',
    'pgrep -fl llama-server',
    'grep -rn llama-server src/',
    'ls ~/.cache/huggingface',
    'git add HANDOFF.md',
    'git add src/one.py tests/test_one.py && git commit -m "x"',
    'git add -p src/one.py',
    'git commit -m "x"',
])
def test_ordinary_work_is_not_refused(command):
    """A guard that fires on ordinary commands is a guard that gets switched off."""
    assert guard(command) == ALLOWED


@pytest.mark.parametrize("body", [
    'never nohup a job &',
    'SKIP_NAME_CHECK=1 is a person\'s to set, not an agent\'s',
    'do not run llama-server by hand',
    'no pkill -f llama-server either',
])
def test_writing_about_a_forbidden_shell_is_not_running_one(body):
    """The guard blocked its own installation twice: the command wrote documentation about a
    forbidden shell through a heredoc and the pattern matched the prose."""
    assert guard(f"cat > notes.md <<'MD'\n{body}\nMD") == ALLOWED


def test_a_heredoc_body_does_not_hide_a_real_command_after_it():
    """Skipping to the terminator must not swallow what follows it."""
    assert guard("cat > f <<'MD'\nharmless\nMD\npkill -f llama-server") == BLOCKED


def test_only_bash_is_guarded():
    assert guard("nohup python3 run.py &", tool="Read") == ALLOWED


def test_the_guard_can_be_switched_off_for_a_session():
    """MLSTACK_GUARD=off is the escape hatch; needing it means a rule is wrong, but it must work."""
    assert guard("pkill -f llama-server", MLSTACK_GUARD="off") == ALLOWED
