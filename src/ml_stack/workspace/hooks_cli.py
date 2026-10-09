"""The `hook-snippet` and `install-hooks` commands of `ml-stack-workspace`."""

from __future__ import annotations

import argparse

from ml_stack import agent_hooks, authority, home
from ml_stack.command import flag
from ml_stack.log import say
from ml_stack.workspace import onboard

SNIPPET = [flag("tool", choices=agent_hooks.AGENTS)]
INSTALL = [flag("--settings", default="", help="the Claude Code settings file (default: the user's)"),
           flag("--codex-config", default="", help="the Codex config file (default: the user's)"),
           flag("--only", action="append", default=[], choices=agent_hooks.AGENTS,
                help="write one agent's hooks only")]


def snippet(args: argparse.Namespace, ws: object) -> int:
    """Print the setting a tool needs so it runs `nudge`; writes nothing."""
    say(onboard.hook_snippet(args.tool, args.agent or "NAME"), end="")
    return 0


def install(args: argparse.Namespace, ws: object) -> int:
    """Write the nudge hooks into each agent's settings; exit 1 when no agent was found."""
    authority.require("workspace.setup", "workspace install-hooks")
    where = {agent: home.expand(given) for agent, given in
             (("claude-code", args.settings), ("codex", args.codex_config)) if given}
    done = agent_hooks.install(tuple(args.only) or agent_hooks.AGENTS, where)
    for agent, events in done.items():
        say(f"{agent}: {', '.join(events)} written to {where.get(agent) or agent_hooks.paths()[agent]}")
    return 0 if done else 1
