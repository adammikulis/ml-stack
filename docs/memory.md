# Agent memory

What the chat agent keeps across sessions: short facts about the person's preferences, this
machine, what worked or failed on it, and things the person asked it to remember. The code is
`src/ml_stack/memory/`; the command is `ml-stack-memory`.

## What is stored

One sealed JSON file, `<state>/memory/facts.json` (directory mode 0700, file mode 0600, written
atomically under a lock). Each fact has:

| field | meaning |
|---|---|
| `text` | one line, at most 400 characters |
| `kind` | `preference`, `machine`, `result` or `note` |
| `source` | `user-said`, `agent-observed` or `tool-result` |
| `created`, `last_confirmed`, `confirm_count` | when it was first stored, last checked, how often |
| `scope` | the machine, the managed llama.cpp build and (optionally) the model it was learned on |

The file is sealed with an HMAC (`sentinel.sealed.SealedFile`, key beside it, mode 0600). An
edit made outside the program fails the seal: the previous sealed copy is used if it holds
(`recovered`), otherwise the store reads as empty and refuses writes (`tampered`) until the
person runs `ml-stack-memory forget --all`. The facts are flat records, so a plain file is used
rather than a `GraphStore`.

Limits: 400 characters a fact, 300 facts a store, 10 facts added per session.

## Staleness

`machine` and `result` facts are marked `RE-CHECK` when the managed llama.cpp build in use
differs from the one they were learned on, or when they were last confirmed more than 90 days
(`machine`) or 30 days (`result`) ago. Preferences and notes are never marked. Storing the same
text again, or `ml-stack-memory confirm ID`, records a confirmation under the current build.

## Retrieval

`retrieve(store, query)` fuses a stemmed word ranking with a meaning ranking when an `embed`
callable is given (reciprocal rank fusion, as in `graph.search`), returns at most 5 facts and
stops at about 400 tokens. Without an embedder only the words vote.

## Safety

- `remember` is an acting tool. It shows the person the exact cleaned text, kind and the source
  the model claimed, and writes only after a yes. Tool output and model text are never stored
  otherwise.
- Every fact is cleaned on write (control, bidi and zero-width characters removed, one line)
  and refused when it holds a credential, reads like an instruction to a model, claims or
  waives a permission, a role or a confirmation, names sentinel's state, or (from a model)
  holds an email address, phone number or id number. The person's own `ml-stack-memory add`
  may hold personal details but never a credential or a permission claim.
- Facts come back only inside `<untrusted source='memory'>` fences, one line each with kind,
  source, confirmation count, date, scope and any re-check reason, under a header saying they
  are notes and not instructions. A fact cannot close the fence or start a line of its own.
- The store directory is registered with `sentinel.human.protect`, so a tool call whose
  arguments name it is refused. `recall` is the only way the model reads facts.
- `ml-stack-memory` refuses a process started by an agent (`CLAUDECODE`, `ML_STACK_AGENT`,
  `ML_STACK_NONINTERACTIVE`); commands that write also need a terminal on stdin and stdout.

## Wiring contract

```python
from ml_stack import memory
from ml_stack.interventions import Confirm

memory_tools = memory.tools(
    confirm=lambda question: person.confirm(Confirm(question, {}, "memory")))
offered = [*chat_tools, *memory_tools]          # recall, remember
context = memory.session_context(task_or_none)   # "" when there is nothing to say
```

- `memory.READ` (`recall`) only looks. `memory.ACTING` (`remember`) writes.
- `remember` asks through the `confirm` it was given. The roles code must not list it in
  `CONFIRM` (the person would be asked twice) and must not offer "always allow" for it: each
  fact is a separate yes.
- `recall` already returns a fenced block. Add it to the rail's fenced set; a second fence is
  harmless (the inner tag is replaced with `[tag removed]`).
- Put `session_context(task)` in front of the first user message of a session, as the
  `prefix` of the turn, not in the system prompt. It is already fenced.
- `memory.propose(fact, kind, source)` returns what `remember` would ask, or the reason it
  would refuse, and stores nothing. Code that wants to suggest a fact after a task can call it
  and then hand the text to the person; nothing is stored without the yes.
- `Store(path=None, clock=..., scope=...)` takes a path for tests. `embed` is any
  `str -> Sequence[float]`.

## Commands

```
ml-stack-memory list [--json]       show ID       add TEXT [--kind K --source S --model M]
ml-stack-memory confirm ID          forget ID     forget --all [--yes]
ml-stack-memory export              stats [--json]
```

## Not in v1

Ranking tools from outcomes with a decider, syncing across devices, proposing facts
automatically after a task, persisted embeddings, pinning the store through the sentinel
manifest.
