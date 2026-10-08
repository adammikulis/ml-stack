"""The model-free reading of what the person typed. Typed words never authorize: they revoke, or say that the structured approval question is needed."""

from __future__ import annotations

import re
from dataclasses import dataclass

__all__ = ["KINDS", "Reading", "interpret", "proposal_kinds"]

KINDS = ("release-main",)
MAX_WORDS = 12

AFFIRM = {"yes", "yep", "yeah", "ok", "okay", "sure", "go", "do", "please", "approved", "confirmed", "sounds",
          "good", "looks", "it", "ahead"}
NEGATION = {"no", "not", "dont", "don", "never", "stop", "wait", "hold", "cancel", "without", "nothing",
            "nope", "undo", "revoke", "nevermind"}
NEVER = {"tag", "tags", "tagged", "force", "forced", "delete", "deletion", "mirror", "sudo", "keystore", "budget",
         "budgets", "token", "all"}
VERBS = r"push|pushing|publish|release|releasing|ship|merge"
ACTIONS = re.compile(r"\b(?:push|delete|remove|run|rerun|force|merge|release|tag|reset|rebase|drop|restart|restore|"
                     r"deploy|install|revert|kill|clean)\w*")
REVOKE_PHRASES = {"stop", "cancel that", "cancel", "never mind", "nevermind", "stop that", "revoke"}
RELEASE = re.compile(rf"\b(?:{VERBS})\b[^.?!]*\bmain\b|\bmain\b[^.?!]*\b(?:{VERBS})\b")
QUOTED = re.compile(r"[\"`“”]|^>|\n>")
PROPOSAL = re.compile(r"\b(?:i'?ll|i will|i'm going to|shall i|should i|want me to|ready to|about to|let me|"
                      r"going to|can i|may i)\b")


@dataclass(frozen=True, slots=True)
class Reading:
    """What a prompt means: ``revoke``, ``ask`` for the approval question for a ``kind``, ``refuse`` a request
    chat can never approve, or ``none``. No reading authorizes."""

    action: str
    kind: str = ""
    reason: str = ""


NONE = Reading("none")


def _words(text: str) -> list[str]:
    spaced = re.sub(r"[^a-z0-9./' -]", " ", text.lower()).replace("'", "").replace("-", " ")
    return [word.strip(".") for word in spaced.split() if word.strip(".")]


def proposal_kinds(message: str) -> tuple[str, ...]:
    """The kinds the assistant's last message proposes: ``release-main`` for pushing or releasing main, and
    ``other`` when it asks another question or names another action."""
    found: list[str] = []
    proposing = False
    for sentence in re.split(r"(?<=[.?!])\s+|\n", message.lower()):
        verbs = {m.group(0) for m in ACTIONS.finditer(sentence)}
        if not verbs:
            continue
        release = bool(RELEASE.search(sentence)) and PROPOSAL.search(sentence) is not None
        if release and not proposing and len(verbs) == 1:
            proposing = True
            found.append("release-main")
        else:
            found.append("other")
    return tuple(dict.fromkeys(found))


def interpret(prompt: str, proposal: tuple[str, ...] = ()) -> Reading:
    """Read ``prompt`` against the kinds the assistant's last message proposed."""
    text = prompt.strip()
    words = _words(text)
    if not words:
        return NONE
    normal = " ".join(words)
    if normal in REVOKE_PHRASES or text.lower() == "/revoke" or (
            len(words) <= MAX_WORDS and set(words) & NEGATION and set(words) & {"push", "release", "ship", "main"}):
        return Reading("revoke", reason="revoked")
    if text.lower().startswith("/allow"):
        return Reading("refuse", reason="an authorization comes from the approval question, not from /allow")
    if QUOTED.search(text) or len(words) > MAX_WORDS or "\n" in text or "?" in text or set(words) & NEGATION:
        return NONE
    if set(words) & NEVER and re.search(rf"\b(?:{VERBS})\b", normal):
        return Reading("refuse", reason="tags, forced pushes and deletions are the owner's")
    if RELEASE.search(normal):
        return Reading("ask", "release-main", "typed words do not approve a release")
    if words[0] in AFFIRM and set(words) <= AFFIRM and "release-main" in proposal:
        return Reading("ask", "release-main", "a typed yes does not approve a release")
    return NONE
