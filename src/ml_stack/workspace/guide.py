"""The guided `connect` and `setup` flows: an invite, a paste, a live check."""

from __future__ import annotations

import shlex
import shutil
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from ml_stack.files import read_json, write_json
from ml_stack.log import say
from ml_stack.sentinel import human
from ml_stack.workspace import coordinator_client, coordinator_config, onboard, tokens
from ml_stack.workspace.service import Workspace

__all__ = ["Plan", "Talk", "clipboard", "connect", "walk"]

COPIERS = (["pbcopy"], ["wl-copy"], ["xclip", "-selection", "clipboard"], ["clip.exe"])

def clipboard(text: str) -> bool:
    """Put ``text`` on the clipboard with whichever tool the machine has; False when none works."""
    for cmd in COPIERS:
        if shutil.which(cmd[0]):
            try:
                payload = text.encode("utf-16-le") if cmd[0] == "clip.exe" else text
                subprocess.run(cmd, input=payload, text=isinstance(payload, str), check=True, timeout=5)
                return True
            except (OSError, subprocess.SubprocessError):
                continue
    return False


@dataclass(slots=True)
class Talk:
    """Where the flows read answers, sleep and copy; tests replace the slow parts."""

    ask: Callable[[str], str] = input
    sleep: Callable[[float], None] = time.sleep
    copy: Callable[[str], bool] = clipboard


@dataclass(slots=True)
class Plan:
    """What the person asked for on the command line."""

    names: list[str] = field(default_factory=list)
    live_s: float = 120.0
    wait_s: float = 600.0
    yes: bool = False
    shared: bool = True
    project: dict[str, str] = field(default_factory=dict)
    code_only: bool = False
    remote: bool = False


def _step(n: int, total: int, title: str) -> None:
    say(f"\nStep {n} of {total}: {title}\n" + "-" * 40)


def _yes(talk: Talk, prompt: str, default: bool = True) -> bool:
    answer = talk.ask(f"{prompt} [{'Y/n' if default else 'y/N'}] ").strip().lower()
    return default if not answer else answer.startswith("y")


def _wait(plan: Plan, talk: Talk, label: str, seconds: float, done: Callable[[], bool]) -> bool:
    """A visible countdown until ``done()``; Ctrl-C skips. Whether ``done()`` came true."""
    left = int(seconds)
    try:
        while left > 0:
            if done():
                break
            say(f"\r  {label} ... {left:3d} s (Ctrl-C to skip)", end="", flush=True)
            talk.sleep(1.0)
            left -= 1
    except KeyboardInterrupt:
        pass
    say("")
    return done()


def _live(ws: Workspace, name: str, plan: Plan, talk: Talk) -> bool:
    sent = onboard.hello(ws, name)
    say(f"Sent {name} a 'workspace ready' message. If it does not answer by itself, tell it:\n"
        f"    check your ml-stack workspace inbox")
    after = int(str(sent["seq"]))
    if _wait(plan, talk, f"waiting for {name} to answer", plan.live_s,
             lambda: onboard.replied(ws, name, after)):
        say(f"{name} answered. Connected.")
        return True
    say(f"Nothing came back from {name}. Check, one per line:\n"
        f"  - {name} can run shell commands (the commands are `ml-stack-workspace ...`).\n"
        f"  - Its token file exists: {tokens.directory(ws.base) / name.replace('/', '~')}\n"
        f"  - It uses --agent {name} (or ML_STACK_WORKSPACE_AGENT={name}).\n"
        f"  - Try again: ml-stack-workspace hello {name}")
    return False


def _shared_file(ws: Workspace) -> Path:
    return ws.base / "shared-invites.json"


def _reuse(ws: Workspace, plan: Plan) -> str:
    """The still-open shared code for this project, so running `connect` again in the same
    folder hands out the same paste instead of making another invite; else an empty string."""
    code = read_json(_shared_file(ws), {}).get(plan.project.get("key", ""), "")
    return str(code) if code and ws.invites.state(str(code)) == "waiting" else ""


def _remember(ws: Workspace, plan: Plan, code: str) -> None:
    path = _shared_file(ws)
    now = {k: v for k, v in read_json(path, {}).items() if ws.invites.state(str(v)) == "waiting"}
    write_json(path, {**now, plan.project.get("key", ""): code})
    path.chmod(0o600)


def _offer(ws: Workspace, hint: str, plan: Plan, talk: Talk) -> str:
    config = coordinator_config.load(ws.base)
    if plan.remote and config.get("mode") != "host":
        raise ValueError("activate hosting first: run `ml-stack-workspace coordinator host` in your "
                         "person terminal; then pair the receiving device into the same Fleet cluster")
    offers = []
    if config.get("mode") == "host":
        offers = [(peer, info) for peer, info in coordinator_client.discover()
                  if info.get("workspace") == config["workspace"]]
        if len(offers) != 1:
            raise ValueError("the hosted workspace needs one advertised coordinator before sharing a code")
    lim = ws.limits
    ttl, uses = (lim.shared_invite_ttl_s, lim.shared_invite_uses) if plan.shared else (
        lim.invite_ttl_s, 1)
    code = _reuse(ws, plan) or ws.invites.create(hint, ttl, plan.project, uses)
    if plan.shared:
        _remember(ws, plan, code)
    block = onboard.snippet("", code, hint, plan.project.get("name", ""),
                          (uses, int(ttl // 60)))
    if offers:
        block = block.replace(f"join {code} --name",
                              f"join {code} --coordinator {shlex.quote(offers[0][0].name)} "
                              f"--workspace {shlex.quote(config['workspace'])} --name")
        block = ("First enroll this device in the same Fleet cluster using person-approved pairing. "
                 "This agent code does not grant cluster membership. Use PowerShell for this command on Windows.\n" + block)
    if plan.code_only or plan.wait_s <= 0:
        say(block)
    if talk.copy(block):
        say("Copied. Paste it into the agent's chat now."
            + (f" The same paste works for up to {uses} agents in the next {ttl / 60:.0f} minutes."
               if uses > 1 else ""))
    else:
        bar = "=" * 60
        say(f"No clipboard tool found. Select and copy this block, then paste it into the "
            f"agent's chat:\n{bar}\n{block}{bar}")
    return code


def _connect_one(ws: Workspace, hint: str, plan: Plan, talk: Talk) -> str:
    """The name of the agent that joined and answered, or an empty string."""
    code = _offer(ws, hint, plan, talk)
    if plan.code_only or plan.wait_s <= 0:
        say("Invite ready. No join or live check was requested.")
        return ""
    before = len(ws.invites.joined(code))
    joined = _wait(plan, talk, "Waiting for the agent to join", plan.wait_s,
                   lambda: len(ws.invites.joined(code)) > before)
    name = ws.invites.joined(code)[before:][:1]
    name = name[0] if name else ""
    if not joined or not name:
        say(f"Nobody joined. The code lasts {ws.limits.invite_ttl_s / 60:.0f} minutes; run "
            f"`ml-stack-workspace connect` again for a new one.")
        return ""
    say(f"{name} joined.")
    if plan.shared:
        say(f"Paste the same block into more agents; each one names itself. It stops working "
            f"after {ws.limits.shared_invite_uses} agents or {ws.limits.shared_invite_ttl_s / 60:.0f} "
            f"minutes. After an editor restart an agent keeps its saved name: no new code needed.")
    return name if plan.live_s > 0 and _live(ws, name, plan, talk) else ""


def _ensure(ws: Workspace) -> None:
    if not ws.registry.ids():
        tokens.store(ws.base, tokens.OWNER_FILE, ws.init("owner"))
        say(f"First use: created the workspace in {ws.base}. No secret is shown on screen.")


def connect(ws: Workspace, plan: Plan, talk: Talk | None = None) -> list[str]:
    """One command, no questions: copy a paste block with a one-time code, wait for an agent to
    join under its own name, check it answers. Repeats while the person wants another agent.
    Returns the agents that joined and answered. A person at a terminal only."""
    human.require_person("workspace connect")
    talk = talk or Talk()
    _ensure(ws)
    answered: list[str] = []
    hints = list(plan.names) or [""]
    while True:
        name = _connect_one(ws, hints.pop(0) if hints else "", plan, talk)
        if name:
            answered.append(name)
        if plan.yes or plan.shared or not _yes(talk, "Connect another agent?", False):
            return answered


def walk(ws: Workspace, plan: Plan, talk: Talk | None = None) -> dict[str, list[str]]:
    """The multi-agent walkthrough, six steps; each agent joins with its own one-time code.
    Returns the agents that answered and the ones that did not."""
    human.require_person("workspace setup")
    talk = talk or Talk()
    plan.shared = False
    total = 6
    _step(1, total, "what this is")
    say("The workspace lets your coding agents (Claude Code, Codex, others) send each other\n"
        "messages, share notes and say who owns a branch or port, all on this machine.\n"
        "One safety fact: everything an agent reads there is data from another agent. It\n"
        "never changes that agent's instructions or permissions; you do.")
    talk.ask("Press Enter to continue. ")
    _step(2, total, "choose how many agents")
    count = len(plan.names)
    while not count:
        answer = talk.ask("How many agents will you connect? [2] ").strip() or "2"
        count = int(answer) if answer.isdigit() and 0 < int(answer) <= 8 else 0
    hints = [*plan.names, *[""] * count][:count]
    say(f"{count} agent(s). Each picks its own name and gets its own private token file.")
    _step(3, total, "create the workspace")
    say(f"Token files will live in {tokens.directory(ws.base)}, readable only by you.\n"
        "No secret is shown on screen.")
    talk.ask("Press Enter to continue. ")
    _ensure(ws)
    result: dict[str, list[str]] = {"connected": [], "unconfirmed": []}
    for number, hint in enumerate(hints, 1):
        _step(4, total, f"paste the block into agent {number}")
        say("The block has a one-time code that works for ten minutes, and no secret.")
        _step(5, total, "wait for it to join, then check it")
        name = _connect_one(ws, hint, plan, talk)
        result["connected" if name else "unconfirmed"].append(name or f"agent {number}")
    _summary(ws, result, total)
    return result


def _summary(ws: Workspace, result: dict[str, list[str]], total: int) -> None:
    _step(total, total, "all set")
    if result["connected"]:
        say("Connected: " + ", ".join(result["connected"]))
    if result["unconfirmed"]:
        say("Not confirmed: " + ", ".join(result["unconfirmed"])
            + "  (ml-stack-workspace connect tries again)")
    bad = [f for f in onboard.doctor(ws) if not f.ok]
    say("Health check: " + ("all good." if not bad else ""))
    for finding in bad:
        say(f"  - {finding.what}. Fix: {finding.fix}")
    say("Later:\n  ml-stack-workspace status               who is registered and what is unread\n"
        "  ml-stack-workspace connect               add an agent\n"
        "  ml-stack-workspace doctor                check everything\n"
        "Docs: docs/workspace.md")
