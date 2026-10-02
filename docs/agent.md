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

`McpTools` needs the `mcp` extra. `FunctionTools` takes `(schema, callable)` pairs or MCP-shaped
dicts with `{name: callable}`; any object with `list_tools()` and `call(name, arguments)`
coroutines is a `ToolSource`.

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

`Agent(interventions=[hook, ...])`: each hook may define `before_invocation(context)`,
`before_model_call(context)` and `before_tool_call(call, context)`, returning `Proceed()`,
`Deny(reason)`, `Confirm(question, details)` or `Guide(message)`.

- `Deny` on a tool call is answered to the model as a tool error carrying the reason; on a model
  call or the invocation it ends the run with `Done("denied")`.
- `Confirm` yields a `ConfirmRequest` and waits for `agent.confirm(decision, call)`, a function or
  coroutine function returning whether the person agrees. With no handler it refuses.
- `Guide` puts the message in the next user turn.
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
