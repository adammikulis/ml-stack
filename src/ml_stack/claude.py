"""Claude Code on a model this machine serves: one command, the settings it scored best with, nothing else.

Adam: "make a very clean/easy way for me to launch claude code with llama-server."
llama-server speaks the Messages API at ``/v1/messages`` -- streaming, tool use (with
``--jinja``, which every lease here carries), thinking, ``count_tokens`` -- so Claude Code
needs no bridge, only an environment that points every request at the served model and
keeps every other call off the network. ``ml-stack-claude MODEL [-- claude args]`` leases
the model in the settings it scored best with (a `Config` from its profile, the way the bench, the page and
the ingest lease), builds that environment, runs ``claude`` inside the lease, and lets the
server go when Claude Code exits.

What the environment does (from Claude Code's own gateway and environment references):

- ``ANTHROPIC_BASE_URL`` and ``ANTHROPIC_AUTH_TOKEN`` send every model call to the server as a
  bearer request; ``ANTHROPIC_API_KEY`` is left unset so nothing reaches for a real key.
- ``ANTHROPIC_MODEL``, ``ANTHROPIC_DEFAULT_MODEL``, the four ``ANTHROPIC_DEFAULT_*_MODEL``
  tiers and ``CLAUDE_CODE_SUBAGENT_MODEL`` all name the served alias, so a subagent or a
  "fast" side task goes to the same server rather than to a model that is not there.
- ``CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC``, ``CLAUDE_CODE_DISABLE_1M_CONTEXT``,
  ``CLAUDE_CODE_DISABLE_ADAPTIVE_THINKING``, ``CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS``,
  ``CLAUDE_CODE_SKIP_FAST_MODE_ORG_CHECK`` and ``DISABLE_TELEMETRY`` keep feature flags,
  telemetry, the fast-mode check and betas a local server does not have off the wire; the
  ``--settings`` handed to ``claude`` turns the web-fetch preflight (a call to the real API)
  and always-on thinking off.

Hooks and settings for the session: the ``--settings`` file is written to a directory under the
state root (outside the working tree, read-only), carrying a PreToolUse hook that sends every tool
call through ``ml_stack.harnesshook`` (classifier, role, Requests inbox) and a PostToolUse hook that
passes on the workspace nudge. The model is served at 262,144 tokens on one slot with a q8_0 KV
cache, and ``CLAUDE_CODE_MAX_CONTEXT_TOKENS`` and ``CLAUDE_CODE_AUTO_COMPACT_WINDOW`` say so.
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


def settings(pre: str = "", post: str = "", wait: float = 0.0) -> str:
    """The ``--settings`` JSON: no web-fetch preflight (a call to the real API), no always-on
    thinking, and the PreToolUse and PostToolUse command hooks when given."""
    out: dict[str, object] = {"skipWebFetchPreflight": True, "alwaysThinkingEnabled": False}
    if pre:
        out["hooks"] = {
            "PreToolUse": [{"matcher": "*", "hooks": [
                {"type": "command", "command": pre, "timeout": int(wait) + 30}]}],
            "PostToolUse": [{"matcher": "*", "hooks": [{"type": "command", "command": post, "timeout": 10}]}]}
    return json.dumps(out, sort_keys=True)


def alias_of(base_url: str, model: str) -> str:
    """The name the server serves the model under -- what every model variable must say."""
    try:
        from ml_stack.client import reported_models

        names = reported_models(base_url)
    except Exception:  # noqa: BLE001 - the file stem is a fine name when the server will not say
        names = []
    for name in names:
        said = str(name)
        # a server that answers with the file it loaded gives a path; a name is wanted
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
    binary = args.claude or shutil.which("claude") or ""
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
            path = run.files.write("settings.json", settings(run.pre, run.post, harnessing.WAIT_S))
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
