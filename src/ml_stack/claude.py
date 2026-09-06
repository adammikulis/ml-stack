"""Claude Code on a model this machine serves: one command, the measured shape, nothing else.

Adam: "make a very clean/easy way for me to launch claude code with llama-server."
llama-server speaks the Messages API at ``/v1/messages`` -- streaming, tool use (with
``--jinja``, which every lease here carries), thinking, ``count_tokens`` -- so Claude Code
needs no bridge, only an environment that points every request at the served model and
keeps every other call off the network. ``ml-stack-claude MODEL [-- claude args]`` leases
the model in its measured shape (a `Run` from its profile, the way the bench, the page and
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

from ml_stack import hub
from ml_stack.log import say

__all__ = ["environment", "launch", "main", "settings"]

DEFAULT_PORT = 8080

BARE_CONTEXT = 131072
"""What a model that will not say what it trained for is served with. The system prompt and
tool definitions Claude Code sends are tens of thousands of tokens before the conversation
starts, so the build's 4,096 default answers one request with an error."""
DEFAULT_SEATS = 1      # one conversation, one seat, the whole measured cache
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


def environment(base_url: str, alias: str, *, token: str = "local", offline: bool = True,
                context: int = 0,
                base: Mapping[str, str] | None = None) -> dict[str, str]:
    """The process environment Claude Code runs with, over ``base`` (the caller's own)."""
    env = dict(os.environ if base is None else base)
    env.pop("ANTHROPIC_API_KEY", None)
    env["ANTHROPIC_BASE_URL"] = base_url.rstrip("/")
    env["ANTHROPIC_AUTH_TOKEN"] = token
    if context:
        env["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] = str(int(context))
    for name in MODEL_VARS:
        env[name] = alias
    if offline:
        env.update(OFFLINE)
    return env


def settings() -> str:
    """The ``--settings`` JSON: no web-fetch preflight (a call to the real API), no
    always-on thinking (the served model's profile decided that)."""
    return json.dumps({"skipWebFetchPreflight": True, "alwaysThinkingEnabled": False})


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
    stem = os.path.basename(str(names[0]) if names else str(model))
    return stem[:-5] if stem.endswith(".gguf") else stem


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="ml-stack-claude", allow_abbrev=False,
        description="Claude Code on a model this machine serves, in its measured shape. "
                    "Everything after `--` goes to claude.",
        usage="ml-stack-claude {MODEL | --on URL} [--port N] [--seats N] [--no-profile] [--online] "
              "[--claude PATH] [-- claude arguments]")
    ap.add_argument("model", nargs="?", default="",
                    help="the model file, a name ml-stack-models finds, or hf:owner/repo/file")
    ap.add_argument("--on", metavar="URL", default="",
                    help="a server already running, e.g. http://127.0.0.1:8080 -- claude "
                         "talks to it as it stands and it is left running afterwards; "
                         "nothing is served and no model is named")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--seats", type=int, default=DEFAULT_SEATS,
                    help="conversations the server holds at once; one seat gets the whole "
                         "measured cache (default: %(default)s)")
    ap.add_argument("--no-profile", action="store_true", help="serve the model bare")
    ap.add_argument("--draft", default="auto", metavar="HEAD",
                    help="the draft head that guesses tokens ahead for the model to check "
                         "in one pass: 'auto' takes the smallest one on this machine, "
                         "'none' serves without one, or name one of the heads offered "
                         "(default: %(default)s)")
    ap.add_argument("--online", action="store_true",
                    help="leave Claude Code's telemetry and feature-flag calls on")
    ap.add_argument("--claude", default="", metavar="PATH",
                    help="the claude binary (default: the one on PATH)")
    return ap


def launch(argv: Sequence[str] | None = None, *, say: Callable[[str], None] = say,
           run_claude: Callable[..., int] | None = None) -> int:
    """Lease the model, run ``claude`` inside the lease, return its exit code."""
    words = list(sys.argv[1:] if argv is None else argv)
    # everything after `--` is claude's, untouched; before it, ours
    ours, extra = (words[: words.index("--")], words[words.index("--") + 1:]) \
        if "--" in words else (words, [])
    args = parser().parse_args(ours)
    binary = args.claude or shutil.which("claude") or ""
    if not binary:
        say("error: no `claude` on PATH; install Claude Code or pass --claude PATH")
        return 2
    if args.on and args.model:
        say("error: --on names a server already running; do not name a model as well")
        return 2
    if not args.on and not args.model:
        if not sys.stdin.isatty():
            say("error: name a model to serve, or --on URL for a server already running")
            return 2
        from ml_stack.serve.recent import choices, pick, pick_head

        chosen = pick(choices(), say=say)
        if chosen is None:
            return 1
        if chosen["kind"] == "server":
            args.on = chosen["url"]
        else:
            args.model = chosen["name"]
            if chosen.get("record") is None or args.no_profile:
                head = pick_head(chosen.get("heads") or [], say=say)
                args.draft = head.name if head is not None else "none"

    if args.on:
        base_url = args.on.rstrip("/")
        alias = alias_of(base_url, "")
        if not alias:
            say(f"error: {base_url} did not say what model it serves")
            return 2
        env = environment(base_url, alias, offline=not args.online)
        say(f"claude on {base_url} as {alias!r}; this server was already up and is left up")
        command = [binary, "--settings", settings(), *extra]
        runner = run_claude or (lambda cmd, env: subprocess.call(cmd, env=env))
        return int(runner(command, env))

    from ml_stack.serve import chat_template, manager, profile
    from ml_stack.serve.recent import note
    from ml_stack.serve.shape import Run, Shape, drafted

    found = str(hub.located(args.model, loose=True) or args.model)
    note(found, by="claude")
    measured = None if args.no_profile else profile.profile_for(found)
    if measured is not None:
        run = measured.run(port=args.port, seats=args.seats, model=found)
        if whole := chat_template.trained_context(found):
            from dataclasses import replace

            run = replace(run, shape=replace(run.shape, seat_context=whole // max(1, args.seats)))
        say(f"serving in its measured shape: {profile.said(measured)}")
        if whole:
            say(f"  with the model's whole {whole:,}-token window, not the measured cache")
        run = drafted(run, "none", say=say)
    else:
        seat = chat_template.trained_context(found) or BARE_CONTEXT
        run = Run(shape=Shape(model=found, port=args.port, seats=args.seats,
                              seat_context=seat))
        say(f"serving bare: no measured shape for this model, {seat:,} tokens a seat")
        try:
            run = drafted(run, args.draft, say=say)
        except ValueError as why:
            say(f"error: {why}")
            return 2
    patched = chat_template.written_beside(found)
    if patched is not None:
        say("this model's template refuses a system message after the first; serving with "
            f"one that renders it instead ({patched.name})")
    began = time.time()
    with manager.serve(run.model, manager=run.shape.manager(), **run.lease(), timeout=900.0,
               cache_reuse=256, warmup=False, escalate=True,
               chat_template_file=patched,
               on_event=lambda e: say(f"  {e.get('event')}: "
                                      + ", ".join(f"{k}={v}" for k, v in e.items()
                                                  if k != "event"))) as server:
        alias = alias_of(server.base_url, found)
        env = environment(server.base_url, alias, offline=not args.online,
                          context=run.shape.context)
        say(f"claude on {server.base_url} as {alias!r}, up in {time.time() - began:.0f}s; "
            f"the server goes when claude exits")
        command = [binary, "--settings", settings(), *extra]
        runner = run_claude or (lambda cmd, env: subprocess.call(cmd, env=env))
        return int(runner(command, env))


def main(argv: Sequence[str] | None = None) -> int:
    return launch(argv)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
