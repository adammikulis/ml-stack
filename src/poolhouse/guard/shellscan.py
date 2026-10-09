"""Splits a shell command line into the commands it runs, without running or expanding anything.

`segments` returns each command as a normalised argv (quotes and backslashes resolved) with the
redirections that follow it, or the findings that say the line cannot be read.
"""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass, field

from poolhouse.guard.harm import Finding

__all__ = ["MAX_COMMAND", "Segment", "has_expansion", "has_glob", "segments"]

MAX_COMMAND = 4000
CONTROL = frozenset("();&|")
REDIRECT = frozenset("<>&|")
EXPANSION = re.compile(r"\$[A-Za-z_{(0-9@*#?!$-]|`")
GLOB = re.compile(r"[*?\[]|\{[^{}]*,[^{}]*\}")
OBFUSCATION = (
    (re.compile(r"\$\(|`"), "runs a command substitution (the commands inside are not read)"),
    (re.compile(r"<\(|>\("), "uses process substitution"),
    (re.compile(r"\$['\"]"), "uses an escaped quoted string that hides what it spells"),
    (re.compile(r"\\x[0-9a-fA-F]{2}|\\[0-7]{3}|\\u[0-9a-fA-F]{4}"),
     "contains escape sequences that can spell a command"),
)


@dataclass(slots=True)
class Segment:
    """One command: its ``argv``, the ``redirects`` as (operator, target) and whether a pipe
    feeds it."""

    argv: list[str] = field(default_factory=list)
    redirects: list[tuple[str, str]] = field(default_factory=list)
    piped: bool = False


def has_expansion(token: str) -> bool:
    """Whether ``token`` holds a variable or substitution the shell would expand."""
    return EXPANSION.search(token) is not None


def has_glob(token: str) -> bool:
    """Whether ``token`` holds a wildcard or brace list the shell would expand."""
    return GLOB.search(token) is not None


def _lines_to_separators(text: str) -> str:
    out: list[str] = []
    quote = ""
    at = 0
    while at < len(text):
        c = text[at]
        if c == "\\" and quote != "'" and at + 1 < len(text):
            if text[at + 1] != "\n":
                out += [c, text[at + 1]]
            at += 2
            continue
        if quote:
            quote = "" if c == quote else quote
        elif c in "'\"":
            quote = c
        elif c in "\n\r":
            c = " ; "
        out.append(c)
        at += 1
    return "".join(out)


def segments(command: str) -> tuple[list[Segment], list[Finding]]:
    """The commands in ``command`` and the findings that make the line hard to read (empty when
    it parsed cleanly)."""
    if len(command) > MAX_COMMAND:
        return [], [Finding("unsure", f"the command is {len(command)} characters, too long to read")]
    bad = [Finding("unsure", why) for rx, why in OBFUSCATION if rx.search(command)]
    if "<<" in command.replace("<<<", ""):
        bad.append(Finding("unsure", "feeds a here-document to a command (the text is not read)"))
    try:
        lex = shlex.shlex(_lines_to_separators(command), posix=True, punctuation_chars=True)
        lex.whitespace_split = True
        lex.commenters = ""
        tokens = list(lex)
    except ValueError as exc:
        return [], [*bad, Finding("unsure", f"cannot be parsed as a shell command ({exc})")]
    out: list[Segment] = []
    cur = Segment()
    it = iter(tokens)
    for tok in it:
        if set(tok) <= CONTROL and not tok.startswith("&>") and tok != "|&":
            if cur.argv or cur.redirects:
                out.append(cur)
            cur = Segment(piped=tok == "|")
        elif tok == "|&":
            if cur.argv or cur.redirects:
                out.append(cur)
            cur = Segment(piped=True)
        elif tok[0] in "<>&" and set(tok) <= REDIRECT:
            target = next(it, "")
            if cur.argv and cur.argv[-1].isdigit() and not tok.startswith("&"):
                cur.argv.pop()
            cur.redirects.append((tok, target))
        else:
            cur.argv.append(tok)
    if cur.argv or cur.redirects:
        out.append(cur)
    return out, bad
