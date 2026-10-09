"""Native harness stream events exposed as conversation progress."""
from __future__ import annotations


def event(row: dict, harness: str) -> dict:
    if harness == "pi":
        kind = row.get("type", "")
        if kind == "session":
            return {"session": row.get("id", "")}
        if kind == "message_update":
            delta = row.get("assistantMessageEvent", {})
            return {"delta": delta.get("delta", "")} if delta.get("type") == "text_delta" else {}
        if kind == "message_end":
            message = row.get("message", {})
            if message.get("stopReason") in ("error", "aborted"):
                return {"error": message.get("errorMessage") or "Pi coding turn failed"}
            content = message.get("content", [])
            text = "".join(block.get("text", "") for block in content if isinstance(block, dict)) \
                if isinstance(content, list) else ""
            if message.get("role") == "assistant":
                return {"text": text} if text else {}
        if kind == "tool_execution_start":
            return {"activity": {"type": "tool", "name": row.get("toolName", ""),
                                  "args": row.get("args", {})}}
        if kind == "tool_execution_end":
            result = row.get("result", {})
            content = result.get("content", []) if isinstance(result, dict) else []
            text = " ".join(str(block.get("text", "")) for block in content if isinstance(block, dict))
            if row.get("isError") and "poolhouse:" in text:
                return {"blocked": text[:1000]}
        if kind == "turn_end":
            usage = row.get("message", {}).get("usage", {})
            return {"usage": usage} if usage else {}
        if kind == "error":
            return {"error": row.get("message") or "Pi coding turn failed"}
        return {}
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
        elif kind == "user":
            blocks = row.get("message", {}).get("content", [])
            for block in blocks if isinstance(blocks, list) else []:
                content = block.get("content", "")
                if (block.get("type") == "tool_result" and block.get("is_error") and isinstance(content, str)
                        and "PreToolUse:" in content and "the person did not allow this call" in content):
                    result["blocked"] = content[:1000]
        elif kind == "result":
            if row.get("is_error"):
                result["error"] = row.get("result") or "The coding turn failed"
            result["usage"] = row.get("usage", {})
        return result
    return {}
