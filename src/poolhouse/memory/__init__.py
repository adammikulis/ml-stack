"""What the chat agent keeps across sessions: encrypted per-user graphs of facts in two scopes (the
person's, and one project's), fenced recall over both, and the two tools that read and write them.
``docs/memory.md`` has the wiring contract."""

from poolhouse.memory.facts import KINDS, SOURCES, Fact, Refused
from poolhouse.memory.project import SCOPES, Project, detect
from poolhouse.memory.recall import describe, retrieve, session_context
from poolhouse.memory.store import Setup, Store, Tampered
from poolhouse.memory.tools import ACTING, GUIDE, READ, guidance, propose, tools
from poolhouse.memory.union import Memory, Merged
from poolhouse.memory.vault import KeyUnavailable

__all__ = ["ACTING", "GUIDE", "KINDS", "READ", "SCOPES", "SOURCES", "Fact", "KeyUnavailable", "Memory", "Merged",
           "Project", "Refused", "Setup", "Store", "Tampered", "describe", "detect", "guidance", "propose", "retrieve",
           "session_context", "tools"]
