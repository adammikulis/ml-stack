"""The message-board nudge hooks written into each coding agent's own settings, and a check of them."""

from __future__ import annotations

import json
import os
import re
import shutil
import tomllib
from pathlib import Path

from ml_stack import home
from ml_stack.checks import Finding

__all__ = ["AGENTS", "findings", "install", "install_report", "paths"]

AGENTS = ("claude-code", "codex")
IDS = {"claude-code": "claude", "codex": "codex"}
BINARY = "ml-stack-workspace"
OWN = (f"{BINARY} nudge", "claude-nudge.sh")
CLAUDE_EVENTS = (("PostToolUse", "post", "*"), ("Stop", "stop", ""), ("UserPromptSubmit", "prompt", ""))
CODEX_EVENTS = (("PostToolUse", "post", ".*"), ("Stop", "stop", ""), ("UserPromptSubmit", "prompt", ""))
FIX = f"{BINARY} install-hooks"
BEGIN = "# >>> ml-stack nudge hooks (managed) >>>"
END = "# <<< ml-stack nudge hooks <<<"
CODEX_OTHER_POST = "ml_stack.harnesshook post"
_MANAGED = re.compile(rf"\n*{re.escape(BEGIN)}.*?{re.escape(END)}\n?", re.S)
_NOTIFY = re.compile(rf'(?m)^notify\s*=\s*\[[^\]\n]*"{re.escape(BINARY)}"[^\]\n]*\]\s*\n')
_FEATURES = re.compile(r"(?m)^\[features\][^\n]*\n")
_TABLE = re.compile(r"(?m)^\[")


def command(agent: str, hook: str) -> str:
    """The command an ``agent``'s ``hook`` event runs."""
    return f"{BINARY} nudge --agent {agent} --hook {hook}"


def paths() -> dict[str, Path]:
    """Where each agent keeps the settings the hooks are written to."""
    claude = os.environ.get("CLAUDE_CONFIG_DIR")
    codex = os.environ.get("CODEX_HOME")
    return {"claude-code": (Path(claude) if claude else home.user_home() / ".claude") / "settings.json",
            "codex": (Path(codex) if codex else home.user_home() / ".codex") / "config.toml"}


def present(agent: str, path: Path) -> bool:
    """Whether ``agent`` is on this machine: its settings folder exists or its command is on PATH."""
    return path.parent.is_dir() or shutil.which(agent.split("-")[0]) is not None


def _own(command_line: str) -> bool:
    return any(mark in command_line for mark in OWN)


def install_claude(settings: Path) -> list[str]:
    """Write the PostToolUse, Stop and UserPromptSubmit nudge hooks into Claude Code's ``settings``,
    replacing earlier nudge hooks and keeping every other setting; returns the events written."""
    data = json.loads(settings.read_text()) if settings.exists() else {}
    hooks = data.setdefault("hooks", {})
    for event, hook, matcher in CLAUDE_EVENTS:
        groups = []
        for group in hooks.get(event, []):
            kept = [h for h in group.get("hooks", []) if not _own(h.get("command", ""))]
            if kept:
                groups.append({**group, "hooks": kept})
        groups.append({**({"matcher": matcher} if matcher else {}),
                       "hooks": [{"type": "command", "command": command(IDS["claude-code"], hook)}]})
        hooks[event] = groups
    text = json.dumps(data, indent=2) + "\n"
    if not settings.exists() or settings.read_text() != text:
        settings.parent.mkdir(parents=True, exist_ok=True)
        settings.write_text(text)
    return [event for event, _, _ in CLAUDE_EVENTS]


def _enable_hooks(text: str) -> tuple[str, bool]:
    """``text`` with the Codex ``hooks`` feature on, and whether a features table is still to be written."""
    head = _FEATURES.search(text)
    if head is None:
        return text, True
    end = _TABLE.search(text, head.end())
    stop = end.start() if end else len(text)
    section = text[head.end():stop]
    line = re.search(r"(?m)^\s*hooks\s*=.*$", section)
    section = (section[:line.start()] + "hooks = true" + section[line.end():]) if line else "hooks = true\n" + section
    return text[:head.end()] + section + text[stop:], False


def _codex_block(agent: str, with_post: bool, with_features: bool) -> str:
    out = [BEGIN]
    if with_features:
        out += ["[features]", "hooks = true", ""]
    for event, hook, matcher in CODEX_EVENTS:
        if hook == "post" and not with_post:
            continue
        out += [f"[[hooks.{event}]]", *([f'matcher = "{matcher}"'] if matcher else []), "",
                f"[[hooks.{event}.hooks]]", 'type = "command"', f'command = "{command(IDS[agent], hook)}"',
                "timeout = 30", *(["additionalContextLimit = 4000"] if hook == "post" else []), ""]
    return "\n".join([*out[:-1], END]) + "\n"


def install_codex(config: Path) -> list[str]:
    """Write the UserPromptSubmit and Stop nudge hooks, and PostToolUse unless the launcher's hook
    already nudges, into Codex's ``config``, replacing earlier nudge hooks and keeping every other
    setting; returns the events written."""
    text = config.read_text() if config.exists() else ""
    base = _NOTIFY.sub("", _MANAGED.sub("\n", text)).rstrip("\n")
    base, with_features = _enable_hooks(base)
    others = tomllib.loads(base).get("hooks", {}).get("PostToolUse", []) if base.strip() else []
    with_post = not any(CODEX_OTHER_POST in h.get("command", "") for g in others for h in g.get("hooks", []))
    new = (base + "\n\n" if base else "") + _codex_block("codex", with_post, with_features)
    tomllib.loads(new)
    if new != text:
        config.parent.mkdir(parents=True, exist_ok=True)
        config.write_text(new)
    return [event for event, hook, _ in CODEX_EVENTS if with_post or hook != "post"]


def install(only: tuple[str, ...] = AGENTS, where: dict[str, Path] | None = None) -> dict[str, list[str]]:
    """Write the nudge hooks for each of ``only`` that is on this machine (or named in ``where``);
    returns the events written by agent."""
    found = {**paths(), **(where or {})}
    writers = {"claude-code": install_claude, "codex": install_codex}
    return {agent: writers[agent](found[agent]) for agent in only
            if agent in (where or {}) or present(agent, found[agent])}


def install_report() -> list[str]:
    """`install` for every agent on this machine, as the lines to show; a refusal is a line, not an error."""
    try:
        done = install()
    except (OSError, ValueError) as err:
        return [f"message-board hooks were not written: {err}"]
    return [f"{agent}: message-board hooks ({', '.join(events)}) are in {paths()[agent]}"
            for agent, events in done.items()]


def _commands(groups: object) -> list[str]:
    return [h.get("command", "") for g in groups or [] if isinstance(g, dict)
            for h in g.get("hooks", []) if isinstance(h, dict)]


def _problems(agent: str, data: dict, events: tuple[tuple[str, str, str], ...]) -> list[str]:
    hooks = data.get("hooks") if isinstance(data.get("hooks"), dict) else {}
    out = []
    for event, hook, _ in events:
        have = [c for c in _commands(hooks.get(event)) if _own(c)]
        if agent == "codex" and hook == "post" and any(CODEX_OTHER_POST in c for c in _commands(hooks.get(event))):
            continue
        if not have:
            out.append(f"{event} missing")
        elif have != [command(IDS[agent], hook)]:
            out.append(f"{event} stale")
    if agent == "codex" and (data.get("features") or {}).get("hooks") is not True:
        out.append("the hooks feature is off")
    return out


def findings(where: dict[str, Path] | None = None) -> list[Finding]:
    """One finding per agent on this machine: whether its nudge hooks are all present and current."""
    found = {**paths(), **(where or {})}
    events = {"claude-code": CLAUDE_EVENTS, "codex": CODEX_EVENTS}
    out = []
    for agent in AGENTS:
        path = found[agent]
        if agent not in (where or {}) and not present(agent, path):
            continue
        try:
            data = (json.loads(path.read_text()) if agent == "claude-code" else tomllib.loads(path.read_text())) \
                if path.exists() else {}
            problems = _problems(agent, data, events[agent])
        except (OSError, ValueError) as err:
            problems = [f"{path} cannot be read ({err})"]
        if shutil.which(BINARY) is None:
            problems.append(f"{BINARY} is not on PATH")
        out.append(Finding(name=f"{agent}: message-board hooks", good=not problems,
                           said="installed in " + str(path) if not problems else "; ".join(problems),
                           fix=FIX.split() if problems else []))
    return out
