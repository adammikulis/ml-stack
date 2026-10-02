"""An MCP server over stdio whose tools touch the machine, for the sandbox tests: read a file,
write a file, read an environment variable, open a loopback connection."""

import json
import os
import socket
import sys


def peek(path):
    with open(path) as handle:
        return handle.read()


def put(path, text):
    with open(path, "w") as handle:
        handle.write(text)
    return "written"


def env(name):
    return os.environ.get(name, "<unset>")


def dial(port):
    with socket.create_connection(("127.0.0.1", port), timeout=3) as conn:
        return conn.recv(4).decode()


TOOLS = {"peek": (peek, ["path"]), "put": (put, ["path", "text"]), "env": (env, ["name"]),
         "dial": (dial, ["port"])}


def listed():
    return [{"name": name, "description": name,
             "inputSchema": {"type": "object", "required": args,
                             "properties": {a: {"type": "integer" if a == "port" else "string"}
                                            for a in args}}}
            for name, (_, args) in TOOLS.items()]


def answer(method, params):
    if method == "initialize":
        return {"protocolVersion": params.get("protocolVersion", "2025-06-18"),
                "capabilities": {"tools": {}}, "serverInfo": {"name": "probe", "version": "1"}}
    if method == "tools/list":
        return {"tools": listed()}
    if method == "tools/call":
        try:
            text, bad = str(TOOLS[params["name"]][0](**params["arguments"])), False
        except OSError as exc:
            text, bad = f"{type(exc).__name__}: {exc}", True
        return {"content": [{"type": "text", "text": text}], "isError": bad}
    return None


for line in sys.stdin:
    message = json.loads(line)
    if "id" in message:
        result = answer(message["method"], message.get("params") or {})
        reply = ({"result": result} if result is not None else
                 {"error": {"code": -32601, "message": "method not found"}})
        sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": message["id"], **reply}) + "\n")
        sys.stdout.flush()
