"""What the verbs in a tool's name or a subcommand say it does."""

from __future__ import annotations

import re
from collections.abc import Iterable

from poolhouse.guard.harm import Finding

__all__ = ["DESTRUCTIVE", "REVERSIBLE", "SAFE", "verb_finding", "words_of"]

DESTRUCTIVE = frozenset([
    "delete", "remove", "rm", "rmi", "rmdir", "drop", "destroy", "purge", "truncate", "wipe",
    "erase", "uninstall", "revoke", "kill", "terminate", "shred", "unlink", "prune", "reset",
    "overwrite", "send", "transfer", "pay", "publish", "push", "deploy", "release", "email",
    "tweet", "post", "upload", "share", "invite", "refund", "withdraw", "charge", "cancel",
    "approve", "submit", "unpublish", "force", "nuke", "decommission", "deprovision", "detach",
    "clean", "clear", "flush", "empty", "discard", "rollback", "down", "yank", "burn", "notify",
    "broadcast", "sms", "dispatch", "disconnect", "unsubscribe", "unregister", "deregister",
    "expire"
])
REVERSIBLE = frozenset([
    "write", "create", "edit", "update", "add", "set", "install", "commit", "checkout", "switch",
    "stash", "rename", "move", "mv", "copy", "cp", "restart", "start", "stop", "toggle", "enable",
    "disable", "assign", "comment", "save", "append", "insert", "put", "patch", "apply",
    "refactor", "generate", "build", "init", "clone", "download", "restore", "label", "tag",
    "mkdir", "touch", "pull", "merge", "rebase", "record", "schedule", "reschedule", "attach",
    "snooze", "mark", "close", "reopen", "resolve", "lock", "unlock", "pin", "star", "follow",
    "subscribe", "archive", "open", "reply", "draft", "format"
])
SAFE = frozenset([
    "get", "list", "read", "show", "describe", "search", "status", "stat", "ps", "log", "logs",
    "diff", "query", "find", "view", "check", "count", "inspect", "lookup", "browse", "ls", "cat",
    "head", "tail", "grep", "glob", "pwd", "whoami", "version", "info", "help", "health", "ping",
    "validate", "explain", "preview", "tree", "which", "history", "locate", "analyze", "scan",
    "top", "wc", "df", "du", "fetch", "plan", "select", "verify", "watch", "print", "echo",
    "diagnose", "summarize", "summary", "compare", "lint"
])
CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
SPLIT = re.compile(r"[^A-Za-z0-9]+")


def words_of(name: str) -> list[str]:
    """The lower-case words of a tool or command name: ``mcp__x__DeleteFile`` is
    ``['delete', 'file']``."""
    last = name.split("__")[-1] if "__" in name else name
    return [w.lower() for w in SPLIT.split(CAMEL.sub("_", last)) if w]


def _known(word: str) -> str:
    for table, label in ((DESTRUCTIVE, "destructive"), (REVERSIBLE, "reversible"), (SAFE, "safe")):
        if word in table or (len(word) > 3 and word.endswith("s") and not word.endswith("ss")
                             and word[:-1] in table):
            return label
    return ""


def verb_finding(words: Iterable[str], subject: str = "the tool") -> Finding | None:
    """What the verbs among ``words`` say: any destructive verb decides, else the first
    reversible or safe one; None when no word is a known verb."""
    words = list(words)
    found = [(w, _known(w)) for w in words]
    if "merge" in words and set(words) & {"pr", "pull", "request", "mr"}:
        return Finding("destructive", f"{subject} name says it will merge a pull request")
    for word, label in found:
        if label == "destructive":
            return Finding("destructive", f"{subject} name says it will {word} something")
    for word, label in found:
        if label:
            return Finding(label, f"{subject} name says it will {word}")
    return None
