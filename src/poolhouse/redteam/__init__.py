"""Red-teaming poolhouse's own model-facing surfaces with PyRIT: attacks sent to a served model,
its tool loop, the pages it reads and the daemon in front of it, scored by what they actually
did (a canary touched, a secret read back) rather than by a model's opinion.

Run with ``python -m poolhouse.redteam``. Everything in this package imports without PyRIT;
sending an attack needs the ``redteam`` extra.
"""

from __future__ import annotations

from poolhouse.redteam.egress import EgressRefused, local_only
from poolhouse.redteam.report import Attempt, Report, compare, markdown, summary
from poolhouse.redteam.targets import (
    Answer,
    Responder,
    ToolSpy,
    chat_endpoint,
    from_callable,
    mcp_tool_agent,
)

__all__ = ["Answer", "Attempt", "EgressRefused", "Report", "Responder", "ToolSpy",
           "chat_endpoint", "compare", "from_callable", "local_only", "markdown",
           "mcp_tool_agent", "summary"]
