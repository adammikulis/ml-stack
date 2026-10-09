"""One unique readable name per main session: model family plus a suffix of the session identity.

The suffix is presentation, never authority: it is derived from the whole session identity
(harness and native session id) and lengthens when another session already holds the short form.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from ml_stack.files import writing
from ml_stack.workspace.chain import held
from ml_stack.workspace.identity import Denied

SHORT = 6
GENERIC = frozenset({"claude", "chatgpt", "qwen", "claude-code", "codex"})
"""The names `--agent` once carried for every session; each is refused in favour of the session's own."""
AGENT_ENV = "ML_STACK_WORKSPACE_AGENT"


def family_word(model: str) -> str:
    """The lowercase model family of an exact model id (`agent` when it names none)."""
    model = model.lower().rsplit("/", 1)[-1]
    if model.startswith("claude-"):
        return "claude"
    if model.startswith(("gpt-", "chatgpt-", "o1", "o3", "o4")):
        return "chatgpt"
    if model.startswith(("qwen", "thinkingcap-qwen")):
        return "qwen"
    return "agent"


def digest(harness: str, session: str) -> str:
    """The full identity of a native session, as the hash its suffix is cut from."""
    return hashlib.sha256(f"{harness}\0{session}".encode()).hexdigest()


def assign(root: Path, model: str, harness: str, session: str) -> str:
    """The unique name of this session, recorded so no later session can take the same one.

    The same session always gets the same name back; a session whose short suffix collides with
    another's gets a longer one."""
    if not session or not harness:
        raise Denied("a session name needs the native harness and session id")
    full, word = digest(harness, session), family_word(model)
    path = root / "session-names.json"
    with held(root / "session-names.lock"):
        names = json.loads(path.read_text()) if path.exists() else {}
        for name, owner in names.items():
            if owner == full:
                return name
        for width in range(SHORT, len(full) + 1, 2):
            name = f"{word}-{full[:width]}"
            if name not in names:
                break
        names[name] = full
        with writing(path) as tmp:
            tmp.write_text(json.dumps(names, sort_keys=True), encoding="utf-8")
    return name


def lookup(root: Path, harness: str, session: str) -> str:
    """The name already given to this session, or an empty string."""
    path = root / "session-names.json"
    full = digest(harness, session)
    names = json.loads(path.read_text()) if path.exists() else {}
    return next((name for name, owner in names.items() if owner == full), "")


def check_agent(given: str, environ: dict[str, str] | None = None) -> None:
    """Refuse a shared harness name where the session has its own; the message names the new id."""
    own = (environ if environ is not None else os.environ).get(AGENT_ENV, "")
    if own and given and given in GENERIC and given != own:
        raise Denied(f"`--agent {given}` is no longer an identity: every session has its own name, and "
                     f"yours is {own}. Run workspace commands as yourself: the environment names you, "
                     f"so drop --agent")
