"""Pi coding agent sessions on models served by ml-stack."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from typing import Any

from ml_stack import harnessing
from ml_stack.claude import DEFAULT_PORT, DEFAULT_SLOTS, alias_of
from ml_stack.client import families
from ml_stack.log import say
from ml_stack.serve import provenance

__all__ = ["extension", "launch", "models"]


def models(base_url: str, alias: str, window: int, max_output_tokens: int = 8192) -> str:
    """Pi provider configuration for the model already served by ml-stack."""
    model = {"id": alias, "name": alias, "reasoning": True, "input": ["text"],
             "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
             "contextWindow": window, "maxTokens": max_output_tokens}
    return json.dumps({"providers": {"mlstack": {"baseUrl": base_url.rstrip("/"),
        "api": "openai-completions", "apiKey": "local", "models": [model]}}}, sort_keys=True)


def extension(settings: dict[str, Any]) -> str:
    """Pi extension applying ml-stack's tool classifier and turn limit."""
    config = json.dumps(settings)
    return f'''import {{ spawnSync }} from "node:child_process";
const cfg = {config};
const names = {{bash:"Bash",powershell:"PowerShell",read:"Read",edit:"Edit",write:"Write",grep:"Grep",find:"Glob",ls:"LS"}};
function call(args, payload) {{
  const result = spawnSync(cfg.python, ["-m", "ml_stack.harnesshook", ...args], {{input:JSON.stringify(payload),encoding:"utf8",timeout:310000}});
  if (result.error || result.status !== 0) return {{failed:true}};
  try {{ return JSON.parse(result.stdout || "{{}}") }} catch {{ return {{failed:true}} }}
}}
export default function (pi) {{
  let turns = 0;
  pi.on("before_provider_request", (event) => {{
    const payload = {{...event.payload}};
    delete payload.max_completion_tokens;
    payload.max_tokens = cfg.maxOutputTokens;
    if (cfg.thinking) payload.chat_template_kwargs = {{...payload.chat_template_kwargs, ...cfg.thinking}};
    return payload;
  }});
  pi.on("turn_start", (_event, ctx) => {{
    if (++turns > cfg.maxTurns) {{
      console.log(JSON.stringify({{type:"error",message:"Pi turn budget exhausted"}}));
      ctx.abort(); ctx.shutdown();
    }}
  }});
  pi.on("tool_call", (event) => {{
    const mapped = names[event.toolName] || event.toolName;
    const input = {{...event.input}};
    if (input.path && !input.file_path) input.file_path = input.path;
    if (input.oldText !== undefined) input.old_string = input.oldText;
    if (input.newText !== undefined) input.new_string = input.newText;
    const result = call(["pre","--role",cfg.role,"--label",cfg.label,"--root",cfg.root,
      ...cfg.protected.flatMap(path => ["--protect",path]),"--wait","300"],
      {{tool_name:mapped,tool_input:input,cwd:cfg.root}});
    if (result.failed) return {{block:true,reason:"ml-stack: tool policy failed; blocked"}};
    const answer = result.hookSpecificOutput || {{}};
    if (answer.permissionDecision === "deny") return {{block:true,reason:"ml-stack: " + (answer.permissionDecisionReason || "blocked by policy")}};
  }});
  pi.on("tool_result", () => {{ call(["post","--label",cfg.label],{{}}); }});
}}
'''


def parser() -> argparse.ArgumentParser:
    ap = harnessing.parser("pi", "Pi", DEFAULT_PORT, DEFAULT_SLOTS)
    ap.add_argument("--max-turns", type=int, default=60)
    ap.add_argument("--max-output-tokens", type=int, default=8192)
    ap.add_argument("--effort", choices=("off", "low", "medium", "high"), default="off")
    return ap


def launch(argv: Sequence[str] | None = None, *, say: Callable[[str], None] = say,
           run_pi: Callable[..., int] | None = None, seat_factory: Callable[..., Any] | None = None) -> int:
    """Run Pi against a leased local model with per-tool ml-stack policy checks."""
    words = list(sys.argv[1:] if argv is None else argv)
    ours, extra = (words[:words.index("--")], words[words.index("--") + 1:]) if "--" in words else (words, [])
    args = parser().parse_args(ours)
    provenance.told(args.lease_for)
    args.seat_factory = seat_factory
    binary = args.pi or harnessing.binary_for("pi")
    if not binary:
        say("error: no `pi` on PATH; install @earendil-works/pi-coding-agent")
        return 2
    if args.on and args.model:
        say("error: --on names a server already running; do not name a model as well")
        return 2
    if args.max_output_tokens <= 0:
        say("error: --max-output-tokens must be positive")
        return 2
    if args.max_turns <= 0:
        say("error: --max-turns must be positive")
        return 2
    try:
        harnessing.check_role(args.role)
    except ValueError as why:
        say(f"error: {why}")
        return 2
    runner = run_pi or (lambda command, env: subprocess.call(command, env=env, cwd=args.project or None))
    if args.on:
        base_url = args.on.rstrip("/")
        alias = alias_of(base_url, "")
        if not alias:
            say(f"error: {base_url} did not say what model it serves")
            return 2
        return _run(args, [binary, *extra], (base_url, alias, harnessing.window_of(base_url)), say, runner)
    began = time.time()
    try:
        with harnessing.serving(args.model or harnessing.DEFAULT_MODEL,
                harnessing.Want(args.port,args.slots,args.ctx,args.no_profile,args.draft),say,"pi") as (base_url,config,found):
            alias = alias_of(base_url, found)
            say(f"pi on {base_url} as {alias!r}, up in {time.time()-began:.0f}s")
            return _run(args,[binary,*extra],(base_url,alias,config.serving.slot_context),say,runner)
    except ValueError as why:
        say(f"error: {why}")
        return 2


def _run(args, command, served, say, runner):
    base_url, alias, window = served
    binary, *extra = command
    try:
        with harnessing.opened(args,"pi",served,say) as run:
            config_dir = run.files.path
            (config_dir / "models.json").write_text(models(base_url,alias,window,args.max_output_tokens),encoding="utf-8")
            protected = harnessing.protected_paths(run.files)
            hook = run.files.write("ml-stack.ts", extension({"python": sys.executable,
                "role": args.role, "label": run.seat.name, "root": str(run.cwd),
                "protected": protected, "maxTurns": args.max_turns, "maxOutputTokens": args.max_output_tokens,
                "thinking": families.for_model_id(alias).think_kwargs(args.effort != "off")}))
            env = {**os.environ,"PI_CODING_AGENT_DIR":str(config_dir),"PI_OFFLINE":"1",
                   "ML_STACK_AGENT":"1","ML_STACK_NONINTERACTIVE":"1"}
            argv = [binary,"--provider","mlstack","--model",alias,"--mode","json","--print","--no-session",
                    "--no-mcp","--no-skills","--no-prompt-templates","--no-extensions","--extension",str(hook),
                    "--append-system-prompt",run.brief,"--thinking",args.effort,*extra]
            say(f"role {args.role}; Pi tools pass through ml-stack's classifier")
            return int(runner(argv,env))
    except ValueError as why:
        say(f"error: {why}")
        return 2


if __name__ == "__main__":
    raise SystemExit(launch())
