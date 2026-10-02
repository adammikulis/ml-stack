"""A two-tool MCP server over stdio for the agent tests, written against the wire protocol
so it runs whichever version of the ``mcp`` package is installed."""

import json
import sys

TOOLS = {
    "add": ("Add two integers.", {"a": {"type": "integer"}, "b": {"type": "integer"}},
            lambda a, b: str(a + b)),
    "shout": ("Upper-case some text.", {"text": {"type": "string"}}, lambda text: text.upper()),
    "fail": ("Always fails.", {}, None),
}


def listed() -> list[dict]:
    return [{"name": name, "description": doc,
             "inputSchema": {"type": "object", "properties": props,
                             "required": list(props)}}
            for name, (doc, props, _) in TOOLS.items()]


def answer(method: str, params: dict) -> dict | None:
    if method == "initialize":
        return {"protocolVersion": params.get("protocolVersion", "2025-06-18"),
                "capabilities": {"tools": {}}, "serverInfo": {"name": "toy", "version": "1"}}
    if method == "tools/list":
        return {"tools": listed()}
    if method == "tools/call":
        fn = TOOLS[params["name"]][2]
        if fn is None:
            return {"content": [{"type": "text", "text": "it failed"}], "isError": True}
        return {"content": [{"type": "text", "text": fn(**params["arguments"])}],
                "isError": False}
    return {}


for line in sys.stdin:
    message = json.loads(line)
    if "id" in message:
        sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": message["id"],
                                     "result": answer(message["method"],
                                                      message.get("params") or {})}) + "\n")
        sys.stdout.flush()
