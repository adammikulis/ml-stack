"""A bounded, streaming tool-calling loop for a local model, driven by Python callables or
an MCP server: schemas converted, malformed calls repaired or sent back, results trimmed."""

from __future__ import annotations

from ml_stack.agent.loop import (
    Agent,
    Budget,
    Cancelled,
    Done,
    Event,
    Repair,
    Text,
    Thinking,
    ToolCall,
    ToolResult,
)
from ml_stack.agent.schema import from_mcp, parse_arguments, validate
from ml_stack.agent.sources import FunctionTools, McpTools, ToolOutput, ToolSource

__all__ = ["Agent", "Budget", "Cancelled", "Done", "Event", "FunctionTools", "McpTools",
           "Repair", "Text", "Thinking", "ToolCall", "ToolOutput", "ToolResult", "ToolSource",
           "from_mcp", "parse_arguments", "validate"]
