"""Launch Claude Code against a broker-managed local model with authenticated harness hooks."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from ml_stack import harnessing, serverkeys
from ml_stack.client import reported_models
from ml_stack.log import say
from ml_stack.serve import provenance

__all__ = ["environment", "launch", "main", "settings"]

DEFAULT_PORT = 8080

MIN_COMPACT_WINDOW = 100000
DEFAULT_SLOTS = 1      # one conversation, one slot, the whole measured cache
OFFLINE = {
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
    "CLAUDE_CODE_DISABLE_1M_CONTEXT": "1",
    "CLAUDE_CODE_DISABLE_ADAPTIVE_THINKING": "1",
    "CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS": "1",
    "CLAUDE_CODE_SKIP_FAST_MODE_ORG_CHECK": "1",
    "DISABLE_TELEMETRY": "1",
}
MODEL_VARS = ("ANTHROPIC_MODEL", "ANTHROPIC_DEFAULT_MODEL", "ANTHROPIC_DEFAULT_OPUS_MODEL",
              "ANTHROPIC_DEFAULT_SONNET_MODEL", "ANTHROPIC_DEFAULT_HAIKU_MODEL",
              "ANTHROPIC_DEFAULT_FABLE_MODEL", "CLAUDE_CODE_SUBAGENT_MODEL")


def environment(base_url: str, alias: str, *, offline: bool = True, context: int = 0,
                base: Mapping[str, str] | None = None) -> dict[str, str]:
    """The process environment Claude Code runs with, over ``base`` (the caller's own)."""
    env = dict(os.environ if base is None else base)
    env.pop("ANTHROPIC_API_KEY", None)
    env["ANTHROPIC_BASE_URL"] = base_url.rstrip("/")
    env["ANTHROPIC_AUTH_TOKEN"] = serverkeys.for_url(base_url) or "local"
    if context:
        env["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] = str(int(context))
        if context >= MIN_COMPACT_WINDOW:
            env["CLAUDE_CODE_AUTO_COMPACT_WINDOW"] = str(int(context))
    for name in MODEL_VARS:
        env[name] = alias
    if offline:
        env.update(OFFLINE)
    return env


def settings(pre: str = "", post: str = "", wait: float = 0.0, stop: str = "", observe: str = "") -> str:
    """The ``--settings`` JSON: no web-fetch preflight (a call to the real API), no always-on
    thinking, and the PreToolUse and PostToolUse command hooks when given."""
    out: dict[str, object] = {"skipWebFetchPreflight": True, "alwaysThinkingEnabled": False}
    if pre:
        out["hooks"] = {
            "PreToolUse": [{"matcher": "*", "hooks": [
                {"type": "command", "command": pre, "timeout": int(wait) + 30}]}],
            "PostToolUse": [{"matcher": "*", "hooks": [{"type": "command", "command": post, "timeout": 10}]}]}
    if stop:
        completion = [{"hooks": [{"type": "command", "command": stop, "timeout": 30}]}]
        out.setdefault("hooks", {}).update(Stop=completion, SubagentStop=completion)
    if observe:
        for event in ('SessionStart', 'PostModelSwitch', 'PreToolUse', 'PostToolUse', 'Stop', 'SubagentStop'):
            out.setdefault('hooks', {}).setdefault(event, []).append({'hooks': [
                {'type': 'command', 'command': observe, 'timeout': 2, 'async': True}]})
    return json.dumps(out, sort_keys=True)


def alias_of(base_url: str, model: str) -> str:
    """The name the server serves the model under -- what every model variable must say."""
    names = reported_models(base_url)
    for name in names:
        said = str(name)
        if said and "/" not in said and not said.endswith(".gguf"):
            return said
    stem = Path(str(names[0]) if names else str(model)).name
    return stem[:-5] if stem.endswith(".gguf") else stem


def parser() -> argparse.ArgumentParser:
    ap = harnessing.parser("claude", "Claude Code", DEFAULT_PORT, DEFAULT_SLOTS)
    ap.add_argument("--online", action="store_true",
                    help="leave Claude Code's telemetry and feature-flag calls on")
    return ap


def launch(argv: Sequence[str] | None = None, *, say: Callable[[str], None] = say,
           run_claude: Callable[..., int] | None = None,
           seat_factory: Callable[..., Any] | None = None) -> int:
    """Lease the model, run ``claude`` inside the lease with the session's hooks, return its exit code."""
    words = list(sys.argv[1:] if argv is None else argv)
    ours, extra = (words[: words.index("--")], words[words.index("--") + 1:]) \
        if "--" in words else (words, [])
    args = parser().parse_args(ours)
    provenance.told(args.lease_for)
    args.seat_factory = seat_factory
    binary = args.claude or harnessing.binary_for("claude")
    if not binary:
        say("error: no `claude` on PATH; install Claude Code or pass --claude PATH")
        return 2
    if args.on and args.model:
        say("error: --on names a server already running; do not name a model as well")
        return 2
    try:
        harnessing.check_role(args.role)
    except ValueError as why:
        say(f"error: {why}")
        return 2
    runner = run_claude or (lambda cmd, env: subprocess.call(cmd, env=env, cwd=args.project or None))
    if args.on:
        base_url = args.on.rstrip("/")
        alias = alias_of(base_url, "")
        if not alias:
            say(f"error: {base_url} did not say what model it serves")
            return 2
        say(f"claude on {base_url} as {alias!r}; this server was already up and is left up")
        return _run(args, [binary, *extra], (base_url, alias, harnessing.window_of(base_url)), say, runner)
    began = time.time()
    try:
        with harnessing.serving(
                args.model or harnessing.DEFAULT_MODEL,
                harnessing.Want(args.port, args.slots, args.ctx, args.no_profile, args.draft), say, "claude") as (base_url, config, found):
            alias = alias_of(base_url, found)
            say(f"claude on {base_url} as {alias!r}, up in {time.time() - began:.0f}s")
            return _run(args, [binary, *extra], (base_url, alias, config.serving.slot_context), say, runner)
    except ValueError as why:
        say(f"error: {why}")
        return 2


def _run(args: argparse.Namespace, command: Sequence[str], served: tuple[str, str, int],
         say: Callable[[str], None], runner: Callable[..., int]) -> int:
    """Write the session's settings and brief, and run ``command`` (``claude`` and its arguments)
    with them; ``served`` is the base URL, the alias and the window."""
    base_url, alias, window = served
    binary, *extra = command
    try:
        with harnessing.opened(args, "claude-code", served, say) as run:
            path = run.files.write("settings.json", settings(run.pre, run.post, harnessing.WAIT_S, run.stop, run.observe))
            brief = run.files.write("brief.md", run.brief)
            run.files.lock()
            env = environment(base_url, alias, offline=not args.online, context=window)
            say(f"role {args.role}; every tool call goes through ml-stack's classifier "
                f"(settings {path}, outside the working tree)")
            return int(runner([binary, "--settings", str(path), "--append-system-prompt-file", str(brief),
                               *extra], env))
    except ValueError as why:
        say(f"error: {why}")
        return 2


def main(argv: Sequence[str] | None = None) -> int:
    return launch(argv)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
