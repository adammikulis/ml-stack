# Embedding ml-stack in another app

`ml_stack.serve` starts, adopts and stops `llama-server`; `ml_stack.client` talks to it over
HTTP. Both import nothing outside the standard library, read no file and open no socket when
imported. The package runs on Python 3.11 to 3.14; the app's own environment is 3.13.

```
pip install "ml-stack @ git+https://github.com/adammikulis/ml-stack"
pip install -e /path/to/ml-stack               # a local checkout
```

The base install adds `packaging` and `psutil`, which finds and stops the server's process
tree. Nothing else is installed: no torch, no MLX, no daemon.

## One conversation, pinned to a slot

```python
from ml_stack.client import Client, Request
from ml_stack.serve import serve

TOOLS = [{"type": "function", "function": {
    "name": "add", "description": "Add two integers.",
    "parameters": {"type": "object", "required": ["a", "b"],
                   "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}}}}]

with serve("/models/model.gguf", context=8192, parallel=2) as server:
    client = Client(server.base_url, request=Request(slot=0, n_predict=512))
    reply = client.chat([{"role": "user", "content": "What is 2 + 3?"}], tools=TOOLS)
    for call in reply.tool_calls or []:
        print(call["function"]["name"], call["function"]["arguments"])
    print(reply.content)
```

The server stops when the block ends. `Request(slot=N)` sends `id_slot=N` with every request, so
a conversation that keeps its slot keeps its KV cache; give each concurrent conversation its
own `N` below `parallel`.

## The surface

| Call | Returns |
|---|---|
| `serve(model, *, port=None, context=4096, timeout=None, manager=None, roam=True, **ServerSpec fields)` | a context manager yielding `ServerInfo(base_url, port, pid, backend, adopted, log_path, load_s, warmup_s)` |
| `ServerManager(backend=None, *, state_file=None)` | `lease(spec, ...)` and `release(info)`: the same lifecycle without the block |
| `ServerSpec(model, port, context, parallel, ...)` | every `llama-server` setting as a field |
| `find_binary()` / `require_binary()` | the `llama-server` path, or `None` / `BinaryNotFound` |
| `Client(base_url, *, model=None, family=None, request=None, transport=None)` | `chat(messages, *, tools=None, tool_choice="auto", on_delta=None)` returning `Reply(content, tool_calls, finish_reason, thinking, raw)` |
| `Request(temperature, top_p, top_k, min_p, n_predict, slot, ...)` | the settings every request from a client carries |

`find_binary` looks at `$LLAMA_CPP_SERVER`, `$LLAMA_CPP_DIR`, the build `ml-stack-serve build`
keeps, then the directories a login shell has and `PATH`. Leases are recorded under
`$ML_STACK_HOME` (default `~/.ml-stack`), so a second process asking for the same model and
port adopts the running server rather than starting another.

`ml_stack.hub.local.on_disk()` lists the GGUF files already in the Hub cache and the model
roots.
