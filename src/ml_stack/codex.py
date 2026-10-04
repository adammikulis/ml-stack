"""Codex on a model this machine serves: ``ml-stack-codex MODEL [-- codex args]``.

Codex speaks only the Responses API (``wire_api = "responses"``), which llama-server serves at
``/v1/responses``. The launcher leases the model (262,144 tokens, one slot, q8_0 KV cache), writes
a ``config.toml`` for this run into a private ``CODEX_HOME`` under the state root, and runs
``codex`` with that home so the person's ``~/.codex`` is never read or changed. The config holds
the provider, the context window and auto-compact limit, the sandbox and approval policy for the
role, and a PreToolUse hook (``ml_stack.harnesshook``) that sends Bash, ``apply_patch`` and MCP calls
through the destructive-action classifier. The hook is pre-trusted for this run with
``--dangerously-bypass-hook-trust``, since the launcher wrote it. The server goes when Codex exits.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from ml_stack import harnessing, serverkeys
from ml_stack.chatpolicy import READ_ONLY
from ml_stack.claude import DEFAULT_PORT, DEFAULT_SLOTS, alias_of
from ml_stack.log import say
from ml_stack.serve import provenance
from ml_stack.workspace import tokens
from ml_stack.workspace.identity import TOKEN_ENV

__all__ = ["config_toml", "environment", "launch", "policy_flags"]

PROVIDER = "mlstack"
KEY_ENV = "MLSTACK_API_KEY"
COMPACT_SHARE = 0.9


def _q(text: str) -> str:
    return json.dumps(text, ensure_ascii=True)


def config_toml(base_url: str, alias: str, window: int, hooks: tuple[str, str, float]) -> str:
    """The ``config.toml`` for one run: provider, window, and the (pre, post, wait) hooks. Same
    input, same bytes."""
    pre, post, wait = hooks
    lines = [
        f"model = {_q(alias)}",
        f"model_provider = {_q(PROVIDER)}",
        "check_for_update_on_startup = false",
    ]
    if window:
        lines += [f"model_context_window = {int(window)}",
                  f"model_auto_compact_token_limit = {int(window * COMPACT_SHARE)}"]
    lines += [
        "", "[shell_environment_policy]", 'inherit = "core"',
        "", "[features]", "hooks = true",
        "", "[features.code_mode]", "enabled = false",
        'direct_only_tool_namespaces = ["mcp__workspace"]',
        "", f"[model_providers.{PROVIDER}]", 'name = "ml-stack"',
        f"base_url = {_q(base_url.rstrip('/') + '/v1')}", 'wire_api = "responses"', f"env_key = {_q(KEY_ENV)}",
        "", "[[hooks.PreToolUse]]", 'matcher = ".*"',
        "", "[[hooks.PreToolUse.hooks]]", 'type = "command"', f"command = {_q(pre)}",
        f"timeout = {int(wait) + 30}", 'statusMessage = "ml-stack: checking the call"',
        "", "[[hooks.PostToolUse]]", 'matcher = ".*"',
        "", "[[hooks.PostToolUse.hooks]]", 'type = "command"', f"command = {_q(post)}", "timeout = 10",
    ]
    return "\n".join(lines) + "\n"


def policy_flags(role: str) -> list[str]:
    """Codex's own sandbox and approval flags for ``role``: read-only never writes; the others write
    inside the working directory and ask before leaving it."""
    if role == READ_ONLY:
        return ["--sandbox", "read-only", "--ask-for-approval", "never"]
    return ["--sandbox", "workspace-write", "--ask-for-approval", "on-request"]


def environment(home: Path, base: Mapping[str, str] | None = None, key: str = "") -> dict[str, str]:
    """The process environment Codex runs with: its own home and the local server's key, no real API key."""
    env = dict(os.environ if base is None else base)
    for name in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "CODEX_API_KEY"):
        env.pop(name, None)
    env["CODEX_HOME"] = str(home)
    env[KEY_ENV] = key or "local"
    return env


def launch(argv: Sequence[str] | None = None, *, say: Callable[[str], None] = say,
           run_codex: Callable[..., int] | None = None,
           seat_factory: Callable[..., Any] | None = None) -> int:
    """Lease the model, run ``codex`` inside the lease with this run's home, return its exit code."""
    words = list(sys.argv[1:] if argv is None else argv)
    ours, extra = (words[: words.index("--")], words[words.index("--") + 1:]) \
        if "--" in words else (words, [])
    args = harnessing.parser("codex", "Codex", DEFAULT_PORT, DEFAULT_SLOTS).parse_args(ours)
    provenance.told(args.lease_for)
    args.seat_factory = seat_factory
    binary = args.codex or shutil.which("codex") or ""
    if not binary:
        say("error: no `codex` on PATH; install Codex or pass --codex PATH")
        return 2
    if args.on and args.model:
        say("error: --on names a server already running; do not name a model as well")
        return 2
    try:
        harnessing.check_role(args.role)
    except ValueError as why:
        say(f"error: {why}")
        return 2
    runner = run_codex or (lambda cmd, env: subprocess.call(cmd, env=env, cwd=args.project or None))
    if args.on:
        base_url = args.on.rstrip("/")
        alias = alias_of(base_url, "")
        if not alias:
            say(f"error: {base_url} did not say what model it serves")
            return 2
        return _run(args, [binary, *extra], (base_url, alias, harnessing.window_of(base_url)), say, runner)
    began = time.time()
    try:
        with harnessing.serving(
                args.model or harnessing.DEFAULT_MODEL,
                harnessing.Want(args.port, args.slots, args.ctx, args.no_profile, args.draft), say, "codex"
        ) as (base_url, config, found):
            alias = alias_of(base_url, found)
            say(f"codex on {base_url} as {alias!r}, up in {time.time() - began:.0f}s")
            return _run(args, [binary, *extra], (base_url, alias, config.serving.slot_context), say, runner)
    except ValueError as why:
        say(f"error: {why}")
        return 2


def _run(args: argparse.Namespace, command: Sequence[str], served: tuple[str, str, int],
         say: Callable[[str], None], runner: Callable[..., int]) -> int:
    """Write this run's ``CODEX_HOME`` and run ``command`` (``codex`` and its arguments) in it."""
    base_url, alias, window = served
    binary, *extra = command
    try:
        with harnessing.opened(args, "codex", served, say) as run:
            workspace = ["", "[mcp_servers.workspace]", f"command = {_q(sys.executable)}",
                         'args = ["-m", "ml_stack.mcp", "--builtin", "--workspace-only"]',
                         "startup_timeout_sec = 30", "required = true", "[mcp_servers.workspace.env]"]
            workspace += [f"{name} = {_q(value)}" for name, value in {
                TOKEN_ENV: tokens.load(run.seat.base, run.seat.parent or run.seat.name),
                "ML_STACK_HOME": str(run.seat.base.parent), "ML_STACK_WORKSPACE_HOME": str(run.seat.base),
                "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
                "ML_STACK_AGENT": "1", "ML_STACK_NONINTERACTIVE": "1"}.items()]
            path = run.files.write("config.toml", config_toml(base_url, alias, window,
                                    (run.pre, run.post, harnessing.WAIT_S)) + "\n".join(workspace) + "\n")
            run.files.write("AGENTS.md", run.brief + "\nUse the workspace MCP tools for inbox/send/thread/claim. "
                            "They already carry your assigned identity; never use shell workspace commands "
                            "inside the coding sandbox.\n")
            say(f"role {args.role}; Bash, apply_patch and MCP calls go through ml-stack's classifier "
                f"(CODEX_HOME {path.parent}, outside the working tree)")
            flags = [*policy_flags(args.role), "--dangerously-bypass-hook-trust", "--cd", str(run.cwd)]
            return int(runner([binary, *flags, *extra], environment(run.files.path, key=serverkeys.for_url(base_url))))
    except ValueError as why:
        say(f"error: {why}")
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(launch())
