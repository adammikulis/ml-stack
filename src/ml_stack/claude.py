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

__all__ = ["environment", "launch", "main", "settings"]

DEFAULT_PORT = 8080
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
                base: Mapping[str, str] | None = None) -> dict[str, str]:
    """The process environment Claude Code runs with, over ``base`` (the caller's own)."""
    env = dict(os.environ if base is None else base)
    env.pop("ANTHROPIC_API_KEY", None)
    env["ANTHROPIC_BASE_URL"] = base_url.rstrip("/")
    env["ANTHROPIC_AUTH_TOKEN"] = token
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
    if names:
        return str(names[0])
    stem = os.path.basename(str(model))
    return stem[:-5] if stem.endswith(".gguf") else stem


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="ml-stack-claude", allow_abbrev=False,
        description="Claude Code on a model this machine serves, in its measured shape. "
                    "Everything after `--` goes to claude.",
        usage="ml-stack-claude MODEL [--port N] [--seats N] [--no-profile] [--online] "
              "[--claude PATH] [-- claude arguments]")
    ap.add_argument("model", help="the model file, a name ml-stack-models finds, or hf:owner/repo/file")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--seats", type=int, default=DEFAULT_SEATS,
                    help="conversations the server holds at once; one seat gets the whole "
                         "measured cache (default: %(default)s)")
    ap.add_argument("--no-profile", action="store_true", help="serve the model bare")
    ap.add_argument("--online", action="store_true",
                    help="leave Claude Code's telemetry and feature-flag calls on")
    ap.add_argument("--claude", default="", metavar="PATH",
                    help="the claude binary (default: the one on PATH)")
    return ap


def launch(argv: Sequence[str] | None = None, *, say: Callable[[str], None] = print,
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

    from ml_stack.graph.bench.serve import find_model
    from ml_stack.serve.manager import serve
    from ml_stack.serve.profile import profile_for, said

    found = str(find_model(args.model))
    measured = None if args.no_profile else profile_for(found)
    if measured is not None:
        run = measured.run(port=args.port, seats=args.seats, model=found)
        say(f"serving in its measured shape: {said(measured)}")
    else:
        from ml_stack.serve.shape import Run, Shape

        run = Run(shape=Shape(model=found, port=args.port, seats=args.seats))
        say("serving bare: no measured shape for this model")
    began = time.time()
    with serve(run.model, manager=run.shape.manager(), **run.lease(), timeout=900.0,
               cache_reuse=256, warmup=False, escalate=True,
               on_event=lambda e: say(f"  {e.get('event')}: "
                                      + ", ".join(f"{k}={v}" for k, v in e.items()
                                                  if k != "event"))) as server:
        alias = alias_of(server.base_url, found)
        env = environment(server.base_url, alias, offline=not args.online)
        say(f"claude on {server.base_url} as {alias!r}, up in {time.time() - began:.0f}s; "
            f"the server goes when claude exits")
        command = [binary, "--settings", settings(), *extra]
        runner = run_claude or (lambda cmd, env: subprocess.call(cmd, env=env))
        return int(runner(command, env))


def main(argv: Sequence[str] | None = None) -> int:
    return launch(argv)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
