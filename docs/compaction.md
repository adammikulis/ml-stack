# Compaction

`ml_stack.agent` keeps a conversation inside the model's context. It counts what the
conversation takes, and when it gets close to the limit it shrinks the oldest part, writes what
it removed to a file, and carries on.

## Counting

`context_usage(messages, tools, context_size, count=Counter(client))` returns the tokens used,
the limit and the fraction. `Counter(client)` asks the server's `/tokenize`, so llama-server's
own tokenizer does the counting; each distinct text is counted once. With no reachable server it
falls back to `ml_stack.client.tokens` scaled by the characters-per-token ratio of whatever the
server did count, and by a 15% margin when nothing was counted. Each message adds 4 tokens for
its role markers, and the tool schemas count as one block. The limit is `Compaction.context_size`
or the per-slot `n_ctx` in the server's `/props`.

## Stages

`compact(messages, budget=..., using=Compaction(...))` runs these until the messages fit:

1. `elide` cuts a tool result over `Spill.above` tokens (400) to its head and tail, with a marker
   that names the line of the transcript holding the full text.
2. `prune` drops a tool call, with its results, when a later call repeats it with the same
   arguments.
3. `summarise` replaces the oldest messages with one user message that starts
   `[Summary of the earlier conversation]`. `model_summarizer(client)` asks the served model for
   Goals, Decisions, Facts (identifiers copied exactly), Open tasks and State, in chunks of
   3,000 tokens with the earlier summary carried forward; any callable
   `(messages, prior_summary) -> str` works instead.
4. `truncate` removes the oldest messages outright, leaving a one-line note in the summary. It
   also runs when no summarizer is set or the summarizer fails.

What is never touched: the leading system messages, the last `keep_last` messages (extended to
a whole tool-call group), the last user message, and anything `preserve` names. An assistant
message with tool calls and its results always stay or go together. When the summary is
followed by a user message, a one-word assistant turn separates them so the roles alternate.
A conversation already inside its budget comes back unchanged, and compacting a compacted
conversation leaves one summary.

`CompactResult` carries `messages`, `summary_message`, `dropped_count`, `tokens_before`,
`tokens_after`, `strategy_used` (`elide+summarise`, `none`, ...) and `notes`.

## Automatic

`Agent(client, tools, auto_compact=Compaction(threshold=0.80, target=0.50, keep_last=6,
summarize=True))` checks the context before every request. Past `threshold` of the limit it
compacts to `target` of it, using the share left after the tool schemas. If the server refuses
a request as longer than its context, it compacts to the target and asks once more. It does
nothing while an assistant message still has tool calls without results. `Agent.compact_now(
messages)` compacts at once, in place, whatever the fill.

Events on `Agent.run`:

- `Context(used, limit, fraction)` before every request: a context meter.
- `Compacted(before, after, dropped, strategy, tokens_before, tokens_after, notes)`, with
  `as_dict()` giving `{"type": "compact", ...}`: "compacted 14 messages (62% -> 31%)".

Cancelling the task that drives `run` or `compact_now` leaves the message list as it was; the
list is replaced only when a compaction has finished.

`Compacting(client, Compaction(...), on_event=...)` wraps any client with a `chat` method and
compacts the `messages` it is handed before each call. `ml-stack-do --auto-compact
[--context-size N]` uses it. The MCP tool `conversation_compact` fits a chat saved as JSON.

## The transcript

Every removal and every cut result is appended to a JSONL file under
`$ML_STACK_HOME/compaction/` (`Transcript`, one file per conversation). Each line has `n`, `ts`,
`kind` (`summarised`, `superseded`, `truncated`, `elided`) and the `messages` or `text`.
`Transcript.fetch("<file>#<n>")` returns a line, which is how an elided result is read in full.

## Thresholds

Compacting at 80% leaves room for the reply and for the longest tool result a step is likely to
add; compacting to 50% leaves half the window free, so the next compaction is many steps away
rather than one. Every compaction rewrites the conversation after the system prompt, so
llama-server reuses its cached prefix only up to the end of the system messages and tool
schemas; the summary sits directly after them, and stays byte-identical until the next
compaction. A KV dump taken with `ml-stack-serve slots save` before a compaction holds the old
prefix, so save again afterwards.
