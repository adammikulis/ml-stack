"""Native Codex and Claude stream events exposed as conversation progress."""
from __future__ import annotations


def event(row: dict, harness: str) -> dict:
    if harness == "codex":
        kind = row.get("type", "")
        if kind == "thread.started":
            return {"session": row.get("thread_id", "")}
        if kind in ("error", "turn.failed"):
            return {"error": row.get("message") or row.get("error", {}).get("message", "The coding turn failed")}
        item = row.get("item", {})
        if kind == "item.completed" and item.get("type") == "agent_message":
            return {"text": item.get("text", "")}
        if kind.startswith("item."):
            return {"activity": item}
        if kind == "turn.completed":
            return {"usage": row.get("usage", {})}
    else:
        result = {"session": row["session_id"]} if row.get("session_id") else {}
        kind = row.get("type")
        if kind == "stream_event":
            delta = row.get("event", {}).get("delta", {})
            if delta.get("type") == "text_delta":
                result["delta"] = delta.get("text", "")
        elif kind == "assistant":
            blocks = row.get("message", {}).get("content", [])
            text = "".join(block.get("text", "") for block in blocks if block.get("type") == "text")
            if text:
                result["text"] = text
            result["activity"] = {"type": "assistant", "content": blocks}
        elif kind == "result":
            if row.get("is_error"):
                result["error"] = row.get("result") or "The coding turn failed"
            result["usage"] = row.get("usage", {})
        return result
    return {}
