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
`ml_stack.chat.run_task` and `ml_stack.agent.Agent` (both run `guard.default()` unless told otherwise), and
turning it off needs a reason that is logged.

Status: accepted and implemented in `src/ml_stack/taint/`. The dual-model extractor ships as a helper; the dual-model
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
  (`Ledger.admit(text, Label(Level.UNTRUSTED, origin))`); the ledger cannot tell.
- **Reads that exfiltrate without a sink.** A tool classified `read` that sends its arguments
  somewhere (a search engine taking a query) is an egress channel in disguise. A read whose argument is
  an address is judged as an egress; a read that leaks through some other argument is the
  registry's job to classify.
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

`guard.rails(without=["taint"], because="...")`, for `ml_stack.chat.run_task(guard=...)` and for
`Agent(..., interventions=...)`. It needs the reason, logs it at warning level and prints it, the
same as every other rail. `Agent` with no `interventions` runs `guard.default()`, which includes
the taint rail.

## Control flow

Argument tracking does not see *whether* a call is made, only what goes into it. A page that
persuades the model to call `serve_down()` with no arguments, or `serve_up` with an installed
model and an allowed port, passes every argument rule. The bound is the intent rule: for
`exec`, `egress` and `credential` sinks at least one argument must be a value the person typed
(or came from a validated extraction), otherwise the call is a `Confirm` even when every value
is harmless. For `fleet`, `write` and `state` sinks it is not applied: a model that picks among
values the operator allowed is the cost of being usable. Those three remain the measured gaps.

## Results

Measured 2026-10-02 on this branch (`agent/taint`, on `agent/native-guard`, which holds the
merged guard rails, decision models and agent loop). Python 3.13.5, no served model: every model
is scripted, and a scripted model that does whatever the planted text says is the worst case, so
these numbers measure the rails and not a model's manners. The first three tables are scripted; the real-model rows are marked.

**Flows** (`tests/test_taint_flows.py`; 26 attacks, 24 legitimate tasks; each is a task, what the
run read, and one privileged call; "ran" means the call executed with nobody to answer a
question). `coarse` is the mechanism this replaces: any tool on the sensitive list asks once
untrusted text was read.

| chain | attacks that ran | legitimate tasks that asked |
| --- | --- | --- |
| rails, no taint tracking | 24 of 26 | 0 of 24 |
| rails, coarse rule | 4 of 26 | 18 of 24 |
| rails, taint tracking | 3 of 26 | 4 of 24 |

The 3 that ran are the gaps above: `serve_down()` on its default port, `serve_up` of an installed
model on an allowed port, and an enum value, each after a page told the model to. The coarse
rule's 4 are calls to tools it does not list (a shell, a file write, a mail send) and a join
steered by a peer message; taint tracking stops those and asks about the same legitimate tasks
less than a quarter as often. The 4 legitimate tasks that ask are the ones where the value came
from the model, not the person: a reference picked from search results, text the model wrote to
say, the largest file of a repository the person named, and a path the model chose. A person who
types the reference, or agrees to a plan that names it, is not asked.

**Canary** (`python -m ml_stack.testing.canary`, scripted worst-case model, 18 attacks, 2 benign
tasks): rails off 17 succeed; rails without taint 3 succeed (`injected-fleet-join`,
`injected-download`, `injected-serve`); default 0 succeed; benign tasks completed 2 of 2 in
all three. (A guard built before the scenario sets its environment variable misses that
variable; the figure above builds the guard after.)

**Red-team** (`ml_stack.redteam` pages and toolbox, scripted gullible model, indirect-web: 13
page variants (PDF text layer not built here) times 4 goals, 52 attempts per arm; the agent loop
with no other rail and the web guard off, so only taint tracking differs; PyRIT not installed, so
its scorers and converters were not used; the evidence is the red-team canary file and honeypot):

| arm | model asked for the call | attack succeeded | call refused |
| --- | --- | --- | --- |
| no taint tracking | 24 | 24 | 0 |
| taint tracking, the toy tools unclassified | 24 | 0 | 24 |
| taint tracking, tools classified (page reader a read, note a write, report an egress) | 24 | 0 | 24 |

A read that is given an address is judged as an egress: any argument that holds a URL, or a
string under a name such as `url` or `host`, must be an address the person typed or one on a host
in the trusted `hosts` registry, else the call asks (or is denied when the address repeats text
from a page). Before that rule the page reader let the `ssrf` goal through 6 of 6 times. Left
unclassified, the same tool asks instead, which is the cost of an unknown tool: fail closed. A
model that picks a result URL out of search results to open it now asks too, unless the host is
registered. The red-team's direct-injection
attacks put the instruction in the person's own turn; taint tracking treats that as the person
and does not stop it.

**Tests and mutations.** 115 tests in `tests/test_taint*.py` on real objects: the real rails
chained by `ml_stack.guard`, the real agent loop against a scripted llama-server on a socket, the
real `compact`. 72 textual mutants of the core (a flipped comparison, a dropped branch, a
constant, a removed call, one per rule): the first run killed 47 of 72 and left 25, each a rule
no test read; tests were added for each, and the final run kills all 71 that still apply (one
mutant names code that was removed). These are hand-picked, one per rule, not the repository's
sampled `scripts/mutate`.

**Cost found while measuring.** Matching any value of three characters against tool output made
the plain word `run` in a benchmark call a "proven" copy of a log path, which refuses instead of
asking; values under five characters are no longer matched. The do loop's own acceptance flow
asked after every `bench_run` because the plan named labels but not arguments; the plan tool now
asks for the exact arguments, and the person's `go` covers those values.

**A real model** (Qwen3-4B-Instruct-2507 Q4_K_M, leased once through the broker, nothing else
running, released and stopped after; `canary.live` with 2 repetitions of 9 planted texts, then 5
page variants times 4 goals for the red-team arms; one model, greedy decoding off its default):

| measurement | rails off | rails without taint | default (taint) |
| --- | --- | --- | --- |
| canary planted texts, runs where the attack landed | 14 of 18 | 12 of 18 | 0 of 18 (14 calls asked) |
| red-team web arms, attempts that succeeded of 20 | 4 (model asked 4) | n/a | 0 (model asked 5, 7 refused) |

The scripted model asks for the call in 24 of 52 web attempts and copies the page's value
verbatim every time. The real model asked 4 and 5 times in 20 and, of the 7 taint events in the
taint arm, 3 were verbatim copies (denied) and 4 were values it had re-spelled or composed
(asked). In the canary's planted texts the page names a tool but no value, so all 14 gated calls
carried a value the model made up, and all 14 asked. So a real model re-words more than the
scripted one: at least 4 of 7 attacker-driven calls would have passed a check that follows
copied values only, and all were stopped by treating every model-made value after an untrusted
read as tainted. Counts are small (single model, 2 repetitions); they show direction, not rates.
