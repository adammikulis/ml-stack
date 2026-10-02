"""Red-teaming ml-stack's own model-facing surfaces with PyRIT: attacks sent to a served model,
its tool loop, the pages it reads and the daemon in front of it, scored by what they actually
did (a canary touched, a secret read back) rather than by a model's opinion.

Run with ``python -m ml_stack.redteam``. Everything in this package imports without PyRIT;
sending an attack needs the ``redteam`` extra.
"""

from __future__ import annotations

from ml_stack.redteam.egress import EgressRefused, local_only
from ml_stack.redteam.report import Attempt, Report, compare, markdown, summary
from ml_stack.redteam.targets import (
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
