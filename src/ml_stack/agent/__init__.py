"""A bounded, streaming tool-calling loop for a local model, driven by Python callables or
an MCP server: schemas converted, malformed calls repaired or sent back, results trimmed,
and the conversation compacted before it overflows the context."""

from __future__ import annotations

from ml_stack.agent.auto import AutoCompact, Compacting
from ml_stack.agent.compact import (
    Compaction,
    CompactResult,
    Spill,
    compact,
    has_open_calls,
)
from ml_stack.agent.context import ContextUsage, Counter, context_limit, context_usage
from ml_stack.agent.events import (
    Compacted,
    Context,
    Done,
    Event,
    Repair,
    Text,
    Thinking,
    ToolCall,
    ToolResult,
)
from ml_stack.agent.loop import Agent, Budget, Cancelled
from ml_stack.agent.schema import from_mcp, parse_arguments, validate
from ml_stack.agent.sources import FunctionTools, McpTools, ToolOutput, ToolSource
from ml_stack.agent.summarise import model_summarizer
from ml_stack.agent.transcript import Transcript

__all__ = [
    "Agent", "AutoCompact", "Budget", "Cancelled", "CompactResult", "Compacted", "Compacting", "Compaction",
    "Context", "ContextUsage", "Counter", "Done", "Event", "FunctionTools", "McpTools",
    "Repair", "Spill", "Text", "Thinking", "ToolCall", "ToolOutput", "ToolResult",
    "ToolSource", "Transcript", "compact", "context_limit", "context_usage", "from_mcp",
    "has_open_calls", "model_summarizer", "parse_arguments", "validate",
]
