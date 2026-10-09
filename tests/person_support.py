"""Fixtures for the person-record tests: transcripts in the shapes Claude Code writes, a repository one commit ahead of its remote's main, and the hooks run as real child processes."""

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
PROPOSAL_TEXT = "The tests pass. I'll push main to origin now."
ZERO = "0" * 40
ENV_DROPPED = ("CLAUDECODE", "POOLHOUSE_NONINTERACTIVE", "CLAUDE_CODE_SESSION_ATTENDED", "POOLHOUSE_AGENT",
               "POOLHOUSE_SESSION_ID", "POOLHOUSE_GUARD", "CODEX_THREAD_ID", "CODEX_SESSION_ID", "CLAUDE_CONFIG_DIR")


def environment(state: Path, **more: str) -> dict[str, str]:
    """A child environment whose poolhouse state is ``state`` and whose Claude folder is beside it."""
    env = {k: v for k, v in os.environ.items() if k not in ENV_DROPPED}
    env.update(POOLHOUSE_HOME=str(state), POOLHOUSE_NO_REAL_KEYSTORE="1",
               CLAUDE_CONFIG_DIR=str(state.parent / "claude"), **more)
    return env


def project_dir(state: Path) -> Path:
    """Where the session transcripts of the test live."""
    where = state.parent / "claude" / "projects" / "-test-project"
    where.mkdir(parents=True, exist_ok=True)
    return where


def human(prompt: str, prompt_id: str, **fields) -> dict:
    """A user entry the person typed; ``fields`` replace entry keys such as promptSource, version or timestamp."""
    return {"type": "user", "promptId": prompt_id, "uuid": uuid.uuid4().hex, "sessionId": SESSION,
            "isSidechain": False, "origin": {"kind": "human"}, "promptSource": "typed", "turnOrigin": "human",
            "version": VERSION, "timestamp": "2026-10-08T18:36:54.961Z",
            "message": {"role": "user", "content": prompt}, **fields}


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


def commit(where: Path, name: str) -> str:
    """Commit a new file called ``name`` and return the commit."""
    (where / name).write_text(name + "\n")
    git(where, "add", name)
    git(where, "commit", "-q", "-m", f"add {name}")
    return git(where, "rev-parse", "HEAD")


def repository(where: Path) -> Path:
    """A repository whose ``main`` is one commit ahead of ``origin/main``; the first commit is ``base``."""
    remote = where.with_name(where.name + "-remote.git")
    subprocess.run(["git", "init", "-q", "--bare", "-b", "main", str(remote)], check=True)
    subprocess.run(["git", "init", "-q", "-b", "main", str(where)], check=True)
    commit(where, "base.txt")
    git(where, "remote", "add", "origin", str(remote))
    git(where, "push", "-q", "origin", "main")
    commit(where, "release.txt")
    return where


def tip(repo: Path, ref: str = "main") -> str:
    return git(repo, "rev-parse", ref)


def target(repo: Path, sha: str = "", remote: str = "origin") -> str:
    """The target a release-main approval of ``sha`` (default main) is bound to."""
    sha = sha or tip(repo)
    return f"{repo.resolve()}@{remote}:{sha}:{git(repo, 'rev-parse', sha + '^{tree}')}"


def prompt_event(prompt: str, prompt_id: str, path: Path, cwd: Path, **more) -> dict:
    """The input Claude Code gives a UserPromptSubmit hook."""
    return {"hook_event_name": "UserPromptSubmit", "session_id": SESSION, "transcript_path": str(path),
            "cwd": str(cwd), "prompt": prompt, "prompt_id": prompt_id, **more}


def run_hook(event: dict, state: Path, **env: str) -> subprocess.CompletedProcess:
    """Run scripts/hooks/claude-user-prompt on ``event`` as a child process."""
    return subprocess.run([sys.executable, str(HOOKS / "claude-user-prompt")], input=json.dumps(event),
                          text=True, capture_output=True, timeout=60, check=False,
                          env=environment(state, **env))


def person_consume(cwd: Path, state: Path, *args: str, stdin: str = "", **env: str) -> subprocess.CompletedProcess:
    """Run scripts/hooks/person-consume in ``cwd``."""
    return subprocess.run([sys.executable, str(HOOKS / "person-consume"), *args], cwd=cwd, input=stdin, text=True,
                          capture_output=True, timeout=60, check=False, env=environment(state, **env))


def say(tmp: Path, state: Path, repo: Path, prompt: str, **options) -> subprocess.CompletedProcess:
    """A turn: the ``before`` entries, then ``prompt`` written by ``entry`` (default `human`), then the hook
    run on it; ``prompt_id`` names the turn and any other option goes into the hook event."""
    before, entry = options.pop("before", ()), options.pop("entry", human)
    prompt_id = options.pop("prompt_id", "p-1")
    path = transcript(project_dir(state) / f"{SESSION}.jsonl", *before, entry(prompt, prompt_id))
    return run_hook(prompt_event(prompt, prompt_id, path, repo, **options), state)


def proposal(repo: Path, state: Path, sha: str = "") -> tuple[str, str, str]:
    """The approval question, the approving answer and the declining answer `person-consume propose` prints."""
    done = person_consume(repo, state, "propose", *([sha] if sha else []))
    assert done.returncode == 0, done.stderr
    text, _, options = done.stdout.strip().partition("\n\noptions: ")
    yes, _, no = options.partition(" | ")
    return text, yes, no


def answer_event(repo: Path, question: str, label: str, **more) -> dict:
    """The PostToolUse input for an AskUserQuestion the person answered."""
    return {"hook_event_name": "PostToolUse", "tool_name": "AskUserQuestion", "session_id": SESSION,
            "cwd": str(repo), "tool_response": {"answers": {question: label}, "annotations": {}}, **more}


def approve(tmp: Path, state: Path, repo: Path, sha: str = "") -> subprocess.CompletedProcess:
    """Bind the session (a typed prompt), then answer the approval question for ``sha`` with its approval."""
    say(tmp, state, repo, "hello", prompt_id="bind")
    question, yes, _ = proposal(repo, state, sha)
    return run_hook(answer_event(repo, question, yes), state)


def rows(state: Path) -> list[dict]:
    """Every record in the person log under ``state``."""
    log = state / "person" / "statements.log"
    return [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []


def authorized(state: Path) -> list[dict]:
    return [r for r in rows(state) if r["type"] == "authorization"]
