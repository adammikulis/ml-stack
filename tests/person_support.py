"""Fixtures for the person-record tests: transcripts in the shapes Claude Code writes, a repository on a development branch, and the hooks run as real child processes."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HOOKS = ROOT / "scripts" / "hooks"
VERSION = "2.1.293"
SESSION = "sess-person-1"
DEV = "0.9dev"
ENV_DROPPED = ("CLAUDECODE", "ML_STACK_NONINTERACTIVE", "CLAUDE_CODE_SESSION_ATTENDED", "ML_STACK_AGENT",
               "ML_STACK_SESSION_ID", "MLSTACK_GUARD", "CODEX_THREAD_ID", "CODEX_SESSION_ID")


def environment(state: Path, **more: str) -> dict[str, str]:
    """A child environment whose ml-stack state lives under ``state``."""
    env = {k: v for k, v in os.environ.items() if k not in ENV_DROPPED}
    env.update(ML_STACK_HOME=str(state), ML_STACK_NO_REAL_KEYSTORE="1", **more)
    return env


def human(prompt: str, prompt_id: str, *, source: str = "typed", version: str = VERSION,
          session: str = SESSION, stamp: str = "2026-10-08T18:36:54.961Z") -> dict:
    """A user entry the person typed."""
    return {"type": "user", "promptId": prompt_id, "uuid": uuid.uuid4().hex, "sessionId": session,
            "isSidechain": False, "origin": {"kind": "human"}, "promptSource": source, "turnOrigin": "human",
            "version": version, "timestamp": stamp,
            "message": {"role": "user", "content": prompt}}


def peer(prompt: str, prompt_id: str) -> dict:
    """A user entry another agent wrote."""
    return {**human(prompt, prompt_id), "promptSource": "system", "turnOrigin": "peer"}


def notification(prompt: str, prompt_id: str) -> dict:
    """A user entry the harness wrote for a finished background task."""
    return {**human(prompt, prompt_id), "origin": {"kind": "task-notification"},
            "turnOrigin": "task_notification", "promptSource": "system"}


def local_command(prompt: str, prompt_id: str) -> dict:
    """A user entry for a local command such as /model, whose origin is null."""
    return {**human(prompt, prompt_id), "origin": None}


def assistant(text: str, message_id: str = "msg_1", *, tool: str = "", stamp: str = "2026-10-08T18:36:00.000Z") -> dict:
    """An assistant entry holding one text block and optionally a tool call."""
    blocks: list[dict] = [{"type": "text", "text": text}]
    if tool:
        blocks.append({"type": "tool_use", "name": tool, "id": "tool_1", "input": {}})
    return {"type": "assistant", "uuid": uuid.uuid4().hex, "timestamp": stamp,
            "message": {"id": message_id, "role": "assistant", "content": blocks}}


def transcript(path: Path, *entries: dict) -> Path:
    """Write ``entries`` as a JSONL transcript."""
    path.write_text("".join(json.dumps(e) + "\n" for e in entries), encoding="utf-8")
    return path


def git(where: Path, *args: str) -> str:
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
           "GIT_COMMITTER_EMAIL": "t@t"}
    return subprocess.run(["git", "-C", str(where), *args], check=True, text=True, capture_output=True,
                          env=env).stdout.strip()


def repository(where: Path, branch: str = DEV) -> Path:
    """A repository whose primary checkout is on ``branch``."""
    subprocess.run(["git", "init", "-q", "-b", branch, str(where)], check=True)
    (where / "README.md").write_text("x\n")
    git(where, "add", "README.md")
    git(where, "commit", "-q", "-m", "x")
    return where


def prompt_event(prompt: str, prompt_id: str, path: Path, cwd: Path, *, session: str = SESSION, **more) -> dict:
    """The input Claude Code gives a UserPromptSubmit hook."""
    return {"hook_event_name": "UserPromptSubmit", "session_id": session, "transcript_path": str(path),
            "cwd": str(cwd), "prompt": prompt, "prompt_id": prompt_id, **more}


def run_hook(event: dict, state: Path, **env: str) -> subprocess.CompletedProcess:
    """Run scripts/hooks/claude-user-prompt on ``event`` as a child process."""
    return subprocess.run([sys.executable, str(HOOKS / "claude-user-prompt")], input=json.dumps(event),
                          text=True, capture_output=True, timeout=30, check=False,
                          env=environment(state, **env))


def consume(cwd: Path, state: Path, *, session: str = SESSION, remote: str = "origin",
            kind: str = "push-dev") -> subprocess.CompletedProcess:
    """Run scripts/hooks/person-consume in ``cwd`` as the session ``session``."""
    return subprocess.run([sys.executable, str(HOOKS / "person-consume"), kind, remote], cwd=cwd, text=True,
                          capture_output=True, timeout=30, check=False,
                          env=environment(state, ML_STACK_SESSION_ID=session))


def say(tmp: Path, state: Path, repo: Path, prompt: str, *, before: tuple[dict, ...] = (),
        entry=human, prompt_id: str = "p-1", **more) -> subprocess.CompletedProcess:
    """A turn: ``before`` entries, then ``prompt`` written by ``entry``, then the hook run on it."""
    path = transcript(tmp / f"{prompt_id}.jsonl", *before, entry(prompt, prompt_id))
    return run_hook(prompt_event(prompt, prompt_id, path, repo, **more), state)


def rows(state: Path) -> list[dict]:
    """Every record in the person log under ``state``."""
    log = state / "person" / "statements.log"
    return [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
