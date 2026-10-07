"""The entry the workspace runner calls to start a local-model coding agent.

``launch_coding_agent(model, role, project)`` runs Pi (or another supported coding harness) on a model served through
the broker's lease: a context selected for this device, a role that sets the harness's sandbox and approval
mode, a PreToolUse hook through the destructive-action classifier, and a workspace identity minted
for the session, placed on the project's board and revoked when the session ends.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ml_stack import claude, codex, harnessing, pi
from ml_stack.log import say as default_say

__all__ = ["HARNESSES", "launch_coding_agent"]

HARNESSES = {"pi": pi.launch, "codex": codex.launch, "claude": claude.launch}


def launch_coding_agent(model: str, role: str, project: str | Path, harness: str = "pi",
                        **options: Any) -> int:
    """Run ``harness`` on ``model`` in ``project`` under ``role``; returns the harness's exit code.

    ``model`` "" takes the coding default. Options: ``name`` overrides the workspace identity
    (``local-<model>-<harness>``); ``orders_from`` lists identities the agent obeys besides the
    person and the lead; ``harness_args`` go to the harness after ``--``; ``say`` receives the
    launcher's lines; ``run_codex`` or ``run_claude`` replaces the process start.
    """
    say = options.pop("say", default_say)
    if harness not in HARNESSES:
        say(f"error: no harness {harness!r}: the harnesses are {', '.join(HARNESSES)}")
        return 2
    argv = [model or harnessing.DEFAULT_MODEL, "--role", role, "--project", str(project)]
    if context := options.pop("context", 0):
        argv += ["--ctx", str(context)]
    if harness == "pi" and (max_turns := options.pop("max_turns", 0)):
        argv += ["--max-turns", str(max_turns)]
    if harness == "pi":
        for key, flag in (("max_output_tokens", "--max-output-tokens"), ("effort", "--effort")):
            if key in options:
                value = options.pop(key)
                if value is not None:
                    argv += [flag, str(value)]
    if draft := options.pop("draft", ""):
        argv += ["--draft", draft]
    if options.get("name"):
        argv += ["--name", options.pop("name")]
    for each in options.pop("orders_from", ()):
        argv += ["--orders-from", each]
    if harness_args := options.pop("harness_args", ()):
        argv += ["--", *harness_args]
    return HARNESSES[harness](argv, say=say, **options)
