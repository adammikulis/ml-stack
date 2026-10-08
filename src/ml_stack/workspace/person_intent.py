"""The model-free reading of what the person typed: a closed lexicon decides whether it authorizes, revokes, needs a question or means nothing."""

from __future__ import annotations

import re
from dataclasses import dataclass

__all__ = ["KINDS", "Reading", "interpret", "is_affirmative", "proposal_kinds"]

KINDS = ("push-dev",)
MAX_WORDS = 12

AFFIRM = {"yes", "yep", "yeah", "ok", "okay", "sure", "go", "do", "please", "approved", "confirmed"}
AFFIRM_PHRASES = {"go ahead", "do it", "sounds good", "yes please", "looks good"}
FILLER = AFFIRM | {"it", "that", "ahead", "sounds", "good", "looks", "great", "perfect", "fine", "now",
                   "thanks", "thank", "you", "lets", "let", "s", "push", "the", "dev", "development",
                   "branch", "and"}
NEGATION = {"no", "not", "dont", "don", "never", "stop", "wait", "hold", "cancel", "without", "nothing",
            "nope", "undo", "revoke", "nevermind"}
CONDITIONAL = {"if", "unless", "once", "after", "when", "until", "provided", "depending", "assuming",
               "before", "first", "but"}
FORBIDDEN = {"main", "master", "tag", "tags", "tagged", "release", "releases", "force", "forced", "delete",
             "deletion", "mirror", "sudo", "keystore", "budget", "budgets", "token"}
OTHER_ACTIONS = {"restart", "restore", "deploy", "remove", "install", "merge", "reset", "daemon", "launcher",
                 "launchers", "worktree", "wheel"}
REVOKE_PHRASES = {"stop", "cancel that", "cancel", "never mind", "nevermind", "stop that", "revoke"}
LEAD_IN = r"(?:(?:ok|okay|yes|yeah|yep|sure|please|go ahead and|go ahead|you may|you can|you could|now|then|and) )*"
PUSH_OBJECT = r"(?:the )?(?:dev|development)(?: branch)?"
IMPERATIVE = re.compile(rf"^{LEAD_IN}push (?:{PUSH_OBJECT}|(?P<branch>[\w./-]+))(?: (?:now|please|to origin|for me))*$")
PRONOUN = re.compile(rf"^{LEAD_IN}push (?:it|that|this)(?: (?:now|please|to origin|for me))*$")
PROPOSAL_SENTENCE = re.compile(
    r"\b(?:i'?ll|i will|i'm going to|i am going to|shall i|should i|want me to|ready to|about to|let me|"
    r"going to|i can|can i|may i|ok to|okay to|next step is to)\b[^.?!\n]*\bpush(?:ing)?\b")
GENERIC_OBJECTS = {"the", "dev", "development", "origin", "to", "it", "that", "this", "now", "our", "my"}
QUOTED = re.compile(r"[\"`“”]|^>|\n>")


@dataclass(frozen=True, slots=True)
class Reading:
    """What a prompt means: ``authorize`` a ``kind``, ``revoke``, ``ask`` for a confirmation, ``refuse``
    a kind chat cannot authorize, or ``none``."""

    action: str
    kind: str = ""
    how: str = ""
    reason: str = ""


NONE = Reading("none")


def _words(text: str) -> list[str]:
    spaced = re.sub(r"[^a-z0-9./' -]", " ", text.lower()).replace("'", "").replace("-", " ")
    return [word.strip(".") for word in spaced.split() if word.strip(".")]


def _normal(text: str) -> str:
    return " ".join(_words(text))


def is_affirmative(text: str) -> bool:
    """Whether ``text`` is a bare yes from the closed lexicon, with no question, negation or condition."""
    words = _words(text)
    if not words or len(words) > MAX_WORDS or "?" in text:
        return False
    if set(words) & (NEGATION | CONDITIONAL | FORBIDDEN | OTHER_ACTIONS):
        return False
    return (words[0] in AFFIRM or " ".join(words[:2]) in AFFIRM_PHRASES) and set(words) <= FILLER


def proposal_kinds(message: str, dev_branch: str) -> tuple[str, ...]:
    """The kinds the assistant's last message proposes: ``push-dev`` when it names pushing the development
    branch, ``other`` for anything else it proposes, ordered and without duplicates."""
    found: list[str] = []
    text = message.lower()
    for sentence in re.split(r"(?<=[.?!])\s+|\n", text):
        if not PROPOSAL_SENTENCE.search(sentence):
            continue
        named = re.search(r"\bpush(?:ing)?\s+(?:origin\s+)?([\w./-]+)", sentence)
        branch = named.group(1) if named else ""
        dev = dev_branch.lower()
        names_dev = bool(re.search(r"\b(?:dev|development)\b", sentence)) or bool(dev and dev in sentence)
        other_branch = branch not in GENERIC_OBJECTS | {dev} and not names_dev
        kind = "other" if set(_words(sentence)) & FORBIDDEN or not names_dev or other_branch else "push-dev"
        if kind not in found:
            found.append(kind)
    for word in OTHER_ACTIONS:
        if re.search(rf"\b(?:i'?ll|shall i|want me to|going to)\b[^.?!\n]*\b{word}\b", text) and "other" not in found:
            found.append("other")
    return tuple(found)


def _revokes(text: str, words: list[str]) -> bool:
    if _normal(text) in REVOKE_PHRASES or text.strip().lower() == "/revoke":
        return True
    return len(words) <= MAX_WORDS and bool(set(words) & NEGATION) and "push" in words


def interpret(prompt: str, proposal: tuple[str, ...] = (), dev_branch: str = "") -> Reading:
    """Read ``prompt`` against the kinds the assistant's last message proposed; ``dev_branch`` is the
    development branch the guard derives, which a sentence may name but never change."""
    text = prompt.strip()
    words = _words(text)
    if not words:
        return NONE
    if _revokes(text, words):
        return Reading("revoke", reason="revoked")
    if text.lower().startswith("/allow"):
        asked = _normal(text[len("/allow"):])
        if asked == "push dev":
            return Reading("authorize", "push-dev", "explicit")
        return Reading("refuse", reason="only push-dev can be allowed from chat")
    if QUOTED.search(text) or len(words) > MAX_WORDS or re.search(r"[.!;]\s+\S", text.rstrip(".!")) or "\n" in text:
        return NONE
    mentions_push = "push" in words or "pushing" in words
    if mentions_push and set(words) & FORBIDDEN:
        return Reading("refuse", reason="pushing main, tags, releases and forced pushes are the owner's")
    if "?" in text or set(words) & NEGATION:
        return NONE
    if set(words) & CONDITIONAL and (mentions_push or (proposal and set(words) & AFFIRM)):
        kind = "push-dev" if mentions_push else (proposal[0] if proposal[0] in KINDS else "")
        return Reading("ask", kind, reason="conditional")
    normal = _normal(text)
    named = None if PRONOUN.match(normal) else IMPERATIVE.match(normal)
    if named:
        branch = named.group("branch") or ""
        if branch and branch != dev_branch.lower():
            return Reading("refuse", reason="the named branch is not the development branch")
        return Reading("authorize", "push-dev", "imperative")
    if PRONOUN.match(normal) or is_affirmative(text):
        return _reply(proposal)
    if words[0] in AFFIRM and proposal:
        return Reading("ask", proposal[0] if proposal[0] in KINDS else "", reason="not a bare confirmation")
    return NONE


def _reply(proposal: tuple[str, ...]) -> Reading:
    if not proposal:
        return NONE
    if len(proposal) > 1:
        return Reading("ask", reason="the last message proposed more than one action")
    if proposal[0] not in KINDS:
        return Reading("ask", reason="the last message proposed an action chat cannot authorize")
    return Reading("authorize", proposal[0], "reply")
