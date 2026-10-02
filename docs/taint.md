# Taint tracking

Decision record and measurements. Untrusted content cannot reach the arguments of a privileged
tool unless a person confirms it or the value is one the system can vouch for. The check is
deterministic and does not depend on a judge model noticing an attack.

## Decision

Every agent loop gets a `taint.Ledger`, kept in the run's `Context.notes`. The loop feeds it
without the application doing anything: the person's messages are recorded as trusted, every tool
result and every message that arrives from outside as untrusted. A `TaintRail` (an intervention,
`ml_stack.taint.TaintRail`) answers `before_tool_call`. When the ledger says untrusted content has
entered the context, a call to a sink whose arguments are not individually vouched for is
`Deny` (proven flow into a hard sink) or `Confirm` (everything else). It is on by default in
`ml_stack.do.run` and `ml_stack.agent.Agent`, and turning it off needs a reason that is logged.

Status: accepted for the core. The dual-model extractor ships as a helper; the dual-model
architecture as the default is left for later (see "Dual model").

## Information-flow model

### Labels

A label is a trust level and an origin.

| Level | Meaning | Examples |
| --- | --- | --- |
| `system` | written by the operator or the library | system prompt, sink registry, schemas, the installed-model list |
| `user` | typed by the person in this session | the task, later user turns, the answer to `ask_user` |
| `untrusted` | anything else | web pages, documents, retrieved chunks, tool results from tools not declared trusted, peer messages, sub-agent output, summaries of any of these |

Join is the lowest level. An origin is a short id (`tool:web_fetch#3`, `subagent:2`,
`summary:1`) so a refusal can say which source a value came from and an event can name it without
quoting it.

### Sources

- A tool result is `untrusted` unless its tool is declared `user` (`ask_user`) or `local` (a
  deterministic read of this machine that is not attacker-influenced). Unknown tools are
  `untrusted`. The existing `UntrustedRail` classification (external source, injection phrase) also
  marks the run contaminated, so nothing it flagged is missed.
- Messages found in the conversation that the ledger did not see arrive (a resumed session, a
  `tool` role message, a peer message) are `untrusted`, except the person's own `user` messages.
- A compaction summary is `untrusted` whenever the ledger is contaminated, and always in a resumed
  conversation whose history the ledger did not see.
- A sub-agent's result is `untrusted` when the sub-agent was contaminated.

### Propagation

The ledger has two parts.

1. A **contamination flag** with the origins that set it. It is monotone for the run: nothing
   clears it, not compaction, not a long clean stretch, not a model answer.
2. A **value store**: the text of every untrusted item (so an argument can be traced to it), the
   person's text (so an argument can be recognised as typed by the person), and the values a
   validated extractor vouched for.

Propagation is by value where it can be seen and by contamination where it cannot.

- If an argument shares a verbatim, distinctive span with an untrusted item it is **proven
  tainted**, with the origin of the item.
- If it does not, it is **unproven**: a value the model generated. A model can launder taint
  through text it writes (paraphrase, re-encode, split across arguments, spell out). Value tracking
  cannot see that, and this design does not claim to. It bounds it: once the context holds
  untrusted content, *every* later argument of a privileged call counts as tainted unless it is
  vouched for. Unproven is treated like proven for the purpose of stopping the call; the difference
  is only whether the answer is `Deny` or `Confirm`, and what the person is shown.
- A value is **vouched for** (safe) when one of these holds, and no other way exists:
  1. it is a verbatim, word-bounded span of text the person typed (or the answer to `ask_user`);
  2. the sink's schema constrains it to an enum, a boolean, or a number inside a declared range;
  3. it is a member of a registry the system owns (installed model files, configured peers) read
     at the time of the call;
  4. it fully matches a pattern the sink declared for that argument;
  5. it came out of a validated extraction under the schema name the sink asks for.

### Sinks

Privileged tools are classified by what they can do.

| Capability | Examples | Tainted, proven | Tainted, unproven |
| --- | --- | --- | --- |
| `exec` | start a process, run a command, bench runs | Deny | Confirm |
| `egress` | download, fetch a URL, upload, send | Deny | Confirm |
| `credential` | join a fleet with a passphrase, use a key | Deny | Confirm |
| `fleet` | serve up or down, change a peer | Confirm | Confirm |
| `write` | write a file, synthesise to a path | Confirm | Confirm |
| `state` | any other change | Confirm | Confirm |
| `read` | search, list, show | Proceed | Proceed |

A tool the registry does not know is `state` with every argument free: it needs a confirmation
once the run is contaminated. Declaring a tool `read` is the application's decision and is
visible in one place (`Sinks`). `Confirm` goes to the person through the loop's existing confirm
callback; with no callback it is a refusal, which is the existing behaviour of `Run`.

### Policies

For a call to a sink other than `read`, when the run is contaminated:

1. Every string, number and boolean in the arguments is classified safe, proven or unproven.
2. All safe: `Proceed`.
3. Any proven on `exec`, `egress` or `credential`: `Deny` (reason names the argument and origin).
4. Otherwise `Confirm`, with the tool, capability, and for each unsafe argument its name, status,
   origin ids and a short preview.

When the run is not contaminated the rail proceeds without looking at the arguments; the other
rails (tool policy, secrets) still apply. The rail never relaxes another rail: `merge` takes the
most severe verdict.

## What taint tracking cannot stop

- **Steering through safe channels.** A page that persuades the model to pick the wrong value
  from an enum, the wrong installed model, or to word an answer to the person misleadingly passes
  every check, because the value is one the system vouches for. Constrained sinks shrink the damage
  to the choices the operator allowed.
- **Confirmation fatigue and misleading prompts.** A person who confirms everything defeats the
  `Confirm` path. The prompt shows origin and preview, not the model's justification.
- **Laundering inside the user's own text.** If the person pastes a hostile document into their
  message, its text is `user`. The application must label pasted documents
  (`Ledger.admit(text, origin, Level.untrusted)`); the ledger cannot tell.
- **Reads that exfiltrate without a sink.** A tool classified `read` that sends its arguments
  somewhere (a search engine taking a query) is an egress channel in disguise. Classifying it is
  the registry's job.
- **Side channels after the call.** The tool result of an allowed call is itself untrusted and is
  tracked, but a tool that has side effects not described by its arguments is outside the model.
- **A hostile tool or server.** Tracking assumes tool results are data. A malicious MCP server
  that lies about what a call did is the sentinel's problem (quarantine), not this one.
- **Denial of service.** Tainted calls are refused; the model can be made to waste turns.

## Multi-turn state, compaction, caches, sub-agents

- **Multi-turn.** The ledger lives with the run and serialises (`Ledger.to_dict`,
  `Ledger.from_dict`) so an application that keeps a conversation across processes keeps the
  contamination too. A ledger started on a conversation it did not see from the start (a
  `tool` message or a summary already in it) starts contaminated.
- **Compaction.** The ledger keeps what it saw whether or not the messages survive, so a value
  from a page that was compacted away still traces to it. The summary message is registered as
  untrusted when the run is contaminated; the person's text kept in the ledger is what recognises
  typed values after the original message is gone. A summary never counts as user text.
- **Caches.** An entry that stores model-visible content stores its label with it
  (`Labelled`, `LabelCache`); a hit admits the content to the ledger at the stored level. A cache
  that drops the label is a taint laundry.
- **Sub-agents.** `Ledger.fork` gives the child a copy of the contamination and the person's
  text. The task text the parent wrote is `user` only when the parent was clean. `Ledger.absorb`
  folds the child's result back: untrusted, and contaminating the parent, when the child was
  contaminated.

## Dual model

The pattern: a quarantined model reads untrusted text and may only return values that validate
against a schema; the privileged model never sees the raw text.

It is the strongest structural defence, because the privileged context never holds untrusted
words, so there is nothing to launder. It is not the default here, for these reasons:

- It changes the application, not just the loop: every tool that returns text a task needs to
  read (a page the person asked to summarise) has no schema to return. Summaries and answers are
  free text, and free text is exactly what the privileged model must not see.
- A small local model doing the extraction is itself steerable; the schema is what carries the
  security, not the model. Only enum, pattern-checked, bounded-number and boolean fields survive,
  which covers ids and choices and excludes prose.
- It costs a model call per untrusted read on a machine where model leases are scarce.

What ships: `taint.extract` runs a quarantined model (any callable from prompt to text, with no
tools) and returns only the fields that validate. Free strings are refused unless they are an
enum or a full-match pattern. The values are vouched under the schema name, so a sink can accept
them (rule 5 above) and a tool wrapped with `quarantined` returns only that JSON. What waits: a
wrapper that routes every untrusted tool through it, and a policy for the free-text case (the
privileged model sees only a summary produced by the quarantined one and is told it is data,
which is the existing fence and no stronger).

## Events

A blocked or confirm-gated tainted call writes a `TaintEvent` (`kind`, `severity`, `source`,
`subject`, `ts`, `evidence`) to subscribers (`taint.subscribe`). Evidence holds ids, digests and
counts, never argument text. The shape is the one `docs/sentinel.md` describes for its bus, so
sentinel subscribes with `taint.subscribe(bus.emit)`; the adapter is written on the sentinel
branch, where the bus lives. The `ml_stack.guard` logger gets a warning for each.

## Opting out

`guard.rails(without=["taint"], because="...")` for the loops that use `ml_stack.guard`;
`Agent(taint=False, because="...")` for the agent loop. Both need the reason, log it at warning
level and print it, the same as every other rail.

## Results

Written from the measurements; see the end of the commit that adds them.
