# The agent loop

`ml_stack.agent` runs a model against tools until it answers, and streams what happens.

```python
from ml_stack.agent import Agent, Budget, Compaction, FunctionTools, McpTools
from ml_stack.client import Client

async with McpTools.stdio("python", ["-m", "my_server"]) as tools:      # or McpTools.http(url)
    agent = Agent(Client("http://127.0.0.1:8080"), tools,
                  budget=Budget(max_steps=8, max_tokens=4000),
                  auto_compact=Compaction())
    async for event in agent.run([{"role": "system", "content": "..."},
                                  {"role": "user", "content": "..."}]):
        ...
```

`McpTools` needs the `mcp` extra (mcp 2.2 or later, MCP revision 2026-07-28) and is built on `mcp.Client`, which negotiates the version itself: a stateless 2026-07-28 server and an older handshake server are both reached, and nothing here keeps a session id. `FunctionTools` takes `(schema, callable)` pairs or MCP-shaped
dicts with `{name: callable}`; any object with `list_tools()` and `call(name, arguments)`
coroutines is a `ToolSource`.

## MCP over HTTP

`McpTools.http(url, headers={...}, bearer="token" | callable)` sends the headers on every request;
a callable bearer is read again each time the connection is made. A 401 or 403 raises
`McpAuthError` saying the credentials were not accepted. The token is cut from everything the
server returns (tool text, structured results, descriptions) and is absent from `repr`; it is
never written anywhere. No `Origin` header is sent.

A server may ask the person a question while a tool runs (elicitation). It arrives as a
`ConfirmRequest(id="", name="elicitation", question, details)` and is answered by `agent.confirm`:
return a dict to accept with those form values, `True` to accept with none, anything false to
decline. With no handler it declines.

Not supported: the tasks extension (long-running calls with polling and cancel). The 2.2 client
has no tasks API; `McpTools.on_progress(progress, total, message)` receives progress
notifications for ordinary calls.

## Events

`Text` and `Thinking` deltas as they stream; `ToolCall`, then `ToolResult` per call; `Repair` for
a call that did not match its schema (the model is told the errors and nothing runs); `Denied`
and `ConfirmRequest` from interventions; `Context` and `Compacted` (see `compaction.md`); and a
final `Done(reason, text, steps, tool_calls, tokens, messages)` whose reason is `answer`,
`max_steps`, `max_tool_calls`, `max_tokens`, `repairs_exhausted` or `denied`.
Stopping the iteration or cancelling the task stops the model's stream.

## Calls

MCP `inputSchema`s become OpenAI function schemas (`from_mcp`, with the `lean` and `tiny`
profiles trimming descriptions and nesting for a small model). Arguments are repaired when strict
JSON fails (`parse_arguments`: code fences, prose around the object, trailing commas, Python
literals, single quotes, a cut-off tail), checked with `validate`, and dispatched together up to
`Budget.parallel` at a time. A call written as JSON in the reply text is run as a call. Results
longer than `Budget.max_result_chars` are cut, and `Budget.summarise(name, output)` may rewrite
them first.

## Interventions

`Agent(interventions=...)` runs the guard's rails and, when a local model can be leased, its model
tier unless a list is given (`interventions=guard.off(because=...)` for none; any other empty list
is refused). A list replaces them: `[*guard.default(), hook, ...]` keeps them. Each hook may define `before_invocation(context)`,
`before_model_call(context)`, `before_tool_call(call, context)` and `after_tool_call(call, result,
context)`, returning `Proceed()`, `Deny(reason)`, `Confirm(question, details)`, `Guide(message)` or
`Rewrite(text, tainted=...)`. These are the types of `ml_stack.interventions`, the one mechanism
the guard's rails, a decision model's tool-call check (`ml_stack.decide.guard`) and the
`ml_stack.do` loop share; `docs/guardrails.md` lists the rails.

- `Deny` on a tool call is answered to the model as a tool error carrying the reason; on a model
  call or the invocation it ends the run with `Done("denied")`; on a tool result it replaces the
  result with a withheld notice.
- `Confirm` yields a `ConfirmRequest` and waits for `agent.confirm(question, call)`, a function or
  coroutine function returning whether the person agrees. With no handler it refuses.
- `Guide` puts the message in the next user turn.
- `Rewrite` on a tool result replaces what the model reads; one marked `tainted` makes
  `Context.tainted` true for the rest of the run, which the tool-policy rail reads.
- A hook that raises, or returns anything else, refuses.

## Slot caches

`ml-stack-serve slots save|restore` and `ml_stack.serve.save_slot` / `restore_slot` write a
running server's slot caches to its `--slot-save-path`, each with a `.guard.json` naming the
model, per-slot context, slot count and llama-server build; a restore against a server that
differs, or of a dump with no guard file, is refused.

## Tool-calling models

`ml-stack-train-tools` writes `schema_hash` and `signatures` into a dataset's manifest, refuses
to reuse data made for other tools, and `ml-stack-train-tools eval --data DIR --url URL` scores a
served model on the held-out rows: tool name, required arguments, valid JSON, exact arguments and
all together.
