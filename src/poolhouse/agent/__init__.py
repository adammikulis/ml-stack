"""A bounded, streaming tool-calling loop for a local model, driven by Python callables or
an MCP server: schemas converted, malformed calls repaired or sent back, results trimmed,
and the conversation compacted before it overflows the context."""

from __future__ import annotations

from poolhouse.agent.auto import AutoCompact, Compacting
from poolhouse.agent.compact import (
    Compaction,
    CompactResult,
    Spill,
    compact,
    has_open_calls,
)
from poolhouse.agent.context import ContextUsage, Counter, context_limit, context_usage
from poolhouse.agent.events import (
    Compacted,
    ConfirmRequest,
    Context,
    Denied,
    Done,
    Event,
    Repair,
    Text,
    Thinking,
    ToolCall,
    ToolResult,
)
from poolhouse.agent.loop import Agent, Budget, Cancelled
from poolhouse.agent.schema import from_mcp, parse_arguments, validate
from poolhouse.agent.sources import FunctionTools, McpAuthError, McpTools, ToolOutput, ToolSource
from poolhouse.agent.summarise import model_summarizer
from poolhouse.agent.transcript import Transcript
from poolhouse.agent.watched import unwatched
from poolhouse.interventions import Confirm, Deny, Guide, Intervention, Proceed, Rewrite

__all__ = [
    "Agent",
    "AutoCompact",
    "Budget",
    "Cancelled",
    "CompactResult",
    "Compacted",
    "Compacting",
    "Compaction",
    "Confirm",
    "ConfirmRequest",
    "Context",
    "ContextUsage",
    "Counter",
    "Denied",
    "Deny",
    "Done",
    "Event",
    "FunctionTools",
    "Guide",
    "Intervention",
    "McpAuthError",
    "McpTools",
    "Proceed",
    "Repair",
    "Rewrite",
    "Spill",
    "Text",
    "Thinking",
    "ToolCall",
    "ToolOutput",
    "ToolResult",
    "ToolSource",
    "Transcript",
    "compact",
    "context_limit",
    "context_usage",
    "from_mcp",
    "has_open_calls",
    "model_summarizer",
    "parse_arguments",
    "unwatched",
    "validate",
]
