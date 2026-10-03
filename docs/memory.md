# Agent memory

What the chat agent keeps across sessions: short facts about the person's preferences, this
machine, what worked or failed on it, and things the person asked it to remember. The code is
`src/ml_stack/memory/`; the command is `ml-stack-memory`. Install the extra with
`pip install ml-stack[memory,store]` (`cryptography`, `keyring`, `ladybug`).

## The graph

Facts and the things they are about are nodes of ml-stack's own `GraphStore`
(`docs/graph.md`), so a fact comes back with its neighbourhood and the graph's search and path
queries apply to it.

| node | kind | holds |
|---|---|---|
| `fact:mNNNN` | `fact` | label = the text (one line, at most 400 characters); attrs `ident`, `fkind` (`preference`, `machine`, `result`, `note`), `source` (`user-said`, `agent-observed`, `tool-result`), `created`, `last_confirmed`, `confirm_count`, `scope` (machine, llama.cpp build), `state` (`current` or `superseded`) |
| `entity:KIND:KEY` | `model`, `build`, `setting`, `task`, `topic` | label = the display name; the key is the casefolded, hyphenated canonical name |

| edge | from, to | meaning |
|---|---|---|
| `about` | fact, entity | the model, setting, task or topic a fact is about |
| `learned_on` | fact, build entity | the build a `machine` or `result` fact was learned on (added automatically from the build in use) |
| `supersedes` | new fact, old fact | the old fact is marked `superseded` and no longer recalled |
| `contradicts` | new note, old note | both stay current and both are recalled, each showing the link |
| `related` | fact, fact | set by the person with `ml-stack-memory link` |

**Entities.** `remember` takes `entities`, a list of `kind:name` strings
(`model:Qwen3.8-Flash-Next`, `build:b11380`, `topic:slot count`). Each name goes through the
write checks (credentials, instruction-like text, permission claims, sentinel state names), is
at most 80 characters, and at most 6 are allowed per fact. Names are canonicalised so two
spellings are one node: a model is read as `hub.naming.pretty_name` does (path, `.gguf`, shard
suffix and `UD-` removed, quantisation in brackets), a build as `bNNNN` (`llama.cpp b11380`
and `B11380` are the same), anything else by case and spacing.

**Supersede and contradict.** A new fact of the same kind that has the same non-build entities
as a current fact, including at least one `topic`, replaces it: `supersedes` for `preference`,
`machine` and `result`, `contradicts` for `note`. At most 8 links are made per fact. A fact
with no topic is never linked automatically; facts that share an entity are neighbours anyway.

**Queries are parameterised.** No graph query is built from fact, entity or search text; the
text travels as a bound parameter or is never given to the engine (`tests/test_memory_graph.py`
runs hostile text through every path).

## Encryption, users, keys

- **At rest.** The whole graph is serialised and sealed with AES-256-GCM into one file per user
  and profile, `<state>/memory/u-<hash of uid:login>/<profile>/graph.enc` (directory 0700, file
  0600), plus `graph.enc.prev`, the previous good copy, sealed the same way. The header (magic,
  mode, salt, key id) and the owner (`uid:login|profile`) are authenticated data. No fact text,
  entity name, edge or embedding is in any file in plaintext: the engine runs in memory
  (`GraphStore(":memory:")`), is rebuilt from the sealed file when the file changes, and is
  never written to disk, not as a database file and not as a temporary one.
- **Trade-off.** The alternative, field-level encryption with a blind index, leaves node ids,
  relation names, graph structure and vectors in the clear and cannot use the engine's word
  index, vector index or path search over ciphertext. Sealing the whole graph keeps every graph
  feature and leaks only the file size. The cost: the graph is rebuilt on load (about 0.1 s for a
  few hundred facts) and plaintext exists in this process's memory, which swap or a core dump
  could reach.
- **Whose.** A store belongs to the OS account the process runs as (real uid and login name),
  and to a profile (`default`, or `$ML_STACK_MEMORY_PROFILE`, or `Setup(profile=...)`, 1-32 of
  `a-z0-9_-`). The user is never read from a model, a tool argument or a request; a daemon
  serving several people passes each its own `Setup(user=...)` from its own authentication.
  The key account, the path and the authenticated owner all differ per user and profile, so a
  copied file or a copied key does not open another user's store.
- **Key.** A subkey of the user's master key (`docs/keystore.md`): HKDF-SHA256 over the master
  with the purpose `memory`, the user, profile and directory, and the file's own salt, so a new
  salt is a new key. The master lives in the OS keystore, never in a file. A store written by
  an older version under a random key in the service `ml-stack-memory` is re-sealed on first
  open and that item is deleted once every file reads back. With no usable keystore the
  store fails closed: reads see an empty, `locked` store, writes raise `KeyUnavailable` naming
  the fallback. The fallback is explicit: `ML_STACK_MEMORY_KEYS=passphrase` derives the key with
  scrypt from `$ML_STACK_MEMORY_PASSPHRASE` or a prompt on a terminal. A file records which mode
  it was written in and is never opened by the other.
- **Integrity.** The AES-GCM tag covers the ciphertext and header. A file that fails it falls
  back to the previous copy (`recovered`), otherwise the store reads empty and refuses writes
  (`tampered`) until `ml-stack-memory forget --all`. Replacing the file with an older sealed
  copy made under the same key is not detected.
- **Rekey.** `ml-stack-memory rekey` (person only) chooses a new salt, re-seals the store and
  the previous copy under the key it gives, and the old key is no longer used; an interruption
  leaves a file that still opens under its own salt. Old ciphertext still opens for whoever
  holds the master key; `ml-stack-security keystore-reset` is what ends that. A version 1 file
  imported by the migration below is sealed under the same key.

Stats and `export` are the only places text leaves the store: `export` writes plain JSON to
stdout when the person asks.

## Staleness

`machine` and `result` facts are marked `RE-CHECK` when the managed llama.cpp build in use
differs from the one they were learned on, or when they were last confirmed more than 90 days
(`machine`) or 30 days (`result`) ago. Preferences and notes are never marked. Storing the same
text again, or `ml-stack-memory confirm ID`, records a confirmation under the current build.

## Retrieval

`retrieve(store, query)` fuses three rankings by reciprocal rank: stemmed words over the fact
text, the entities the query names (every word of an entity's name in the query) and the
graph's `hybrid` search (characters, word index and, when an `embed` callable is given,
meaning from the vectors stored with each fact). A hit found only by meaning must reach cosine
0.3. Superseded facts are not returned. `render` lays out at most 5 facts within about 400
tokens, each with up to 2 one-hop neighbours (`near [mNNNN]`, facts sharing an entity or linked
to it), their entities, links and any re-check reason. `session_context(task)` puts preferences
first, then the facts that match the task. Without an embedder only words and entities vote.

## Safety

- `remember` is an acting tool. It shows the person the exact cleaned text, kind, source the
  model claimed and entities, and writes only after a yes. Tool output and model text are never
  stored otherwise.
- Every fact and entity name is cleaned on write (control, bidi and zero-width characters
  removed, one line) and refused when it holds a credential, reads like an instruction to a
  model, claims or waives a permission, a role or a confirmation, names sentinel's state, or
  (from a model) holds an email address, phone number or id number. The person's own
  `ml-stack-memory add` may hold personal details but never a credential or a permission claim.
- Facts come back only inside `<untrusted source='memory'>` fences, one line each (neighbours
  indented under it) with kind, source, confirmation count, date, scope, entities, links and
  any re-check reason, under a header saying they are notes and not instructions. A fact cannot
  close the fence or start a line of its own.
- The store directory is registered with `sentinel.human.protect`, so a tool call whose
  arguments name it is refused. `recall` is the only way the model reads facts.
- `ml-stack-memory` refuses a process started by an agent (`CLAUDECODE`, `ML_STACK_AGENT`,
  `ML_STACK_NONINTERACTIVE`); commands that write also need a terminal on stdin and stdout.

Limits: 400 characters a fact, 80 an entity name, 300 facts a store, 6 entities and 8 links a
fact, 10 facts added per session.

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
- `remember(fact, kind, source, model, entities)` asks through the `confirm` it was given. The
  roles code must not list it in `CONFIRM` (the person would be asked twice) and must not offer
  "always allow" for it: each fact is a separate yes.
- `recall` already returns a fenced block. Add it to the rail's fenced set; a second fence is
  harmless (the inner tag is replaced with `[tag removed]`). A locked or tampered store returns
  a fenced line saying so.
- Put `session_context(task)` in front of the first user message of a session, as the
  `prefix` of the turn, not in the system prompt. It is already fenced.
- `memory.propose(fact, kind, source, entities=None)` returns what `remember` would ask, or the
  reason it would refuse, and stores nothing.
- `Store(path=None, clock=..., scope=..., setup=Setup(...))` takes a path for tests; `Setup`
  carries `user`, `profile`, `keys`, `legacy` and `embed`. `embed` is any `str -> Sequence[float]`;
  `tools(embed=...)` sets it on a store that has none.

## Migration from version 1

A version 1 store (`<state>/memory/facts.json`, one sealed JSON file) is imported the first
time the default store is opened: each fact goes through the write checks again (stricter, as a
model's fact, unless its source is `user-said`), keeps its id, dates and count, and gets a
`model` entity from its scope and a `build` entity for `machine` and `result` facts. The file
is then kept as `facts.json.v1`, sealed under the new key (so it holds no plaintext), and the
plain file, its previous copy and its key file are deleted. A file that fails its old seal is
left alone.

## Commands

```
ml-stack-memory list [--json]   show ID   add TEXT [--kind K --source S --model M --entity kind:name]
ml-stack-memory edit ID TEXT    confirm ID        forget ID | --all [--yes]
ml-stack-memory link ID supersedes|contradicts|related OTHER      unlink ID REL OTHER
ml-stack-memory rekey           export            stats [--json]
```

## Not yet

Ranking tools from outcomes with a decider, syncing across devices, proposing facts
automatically after a task, ingesting documents (`ml-stack-ingest`) into the same graph, the
chat session files (`~/.ml-stack/chat/ID.json`) under the same per-user encryption.
