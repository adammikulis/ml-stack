# Requests

One place where everything that waits for a person is raised, shown and answered. A tool call that
needs a yes, a fact the agent wants remembered, something sentinel holds: each is a request, and
the terminal, the browser page and the desktop dialog are three ways to answer the same one.

Code: `ml_stack.requests` (`raise_request`, `answer`, `list_requests`), the terminal command
`ml-stack-requests`, the route `ml_stack.inbox.route` with the element `ml-requests`
(`ui/assets/ml-requests.js`).

## What a request holds

An id, a kind, who raised it (agent, project, session), the thing it is about, a reason in plain
words, the choices with what each does, when it was raised and when it expires, a state and, once
answered, which way and when. The words come from fixed sentences and the destructive classifier's
labels; every text is escaped (control, bidi and invisible characters become visible escapes) and
cut to a bound before it is stored.

| kind | what waits | answered by a person only |
| --- | --- | --- |
| `tool_call` | a call that waits for a yes | |
| `tool_call_destructive` | a call the classifier labelled destructive or unsure; never offers always allow | |
| `memory_remember` | a fact the agent offered | |
| `hold`, `rule_suggestion` | a held call, a suggested rule | |
| `quarantine_release` | what sentinel holds | yes: the sentinel re-checks at the click |
| `host_approval`, `pairing`, `grant`, `keystore_unlock` | declared kinds; their sources still own their prompts | yes |

States: `pending`, `approved`, `denied`, `expired`, `cancelled`, `superseded`. Choices: allow this
time, always allow, never allow, no, cancel, release, keep held, later, approve. A request that
expires is `expired`, which is a denial. Only an approving answer approves.

## Answering

`answer(id, choice, fingerprint, via)` takes the fingerprint of the exact words and choices that
were shown. The request is read again under a lock; when any word changed since, or the id is
unknown, or the choice is not one the request offered, the answer is refused. The first answer
wins: a later one is refused with `already resolved: <state> by <way>`.

Only a person answers. A terminal answer needs a terminal on stdin and stdout; every way is
refused when an agent marker (`CLAUDECODE`, `ML_STACK_AGENT`, `ML_STACK_NONINTERACTIVE`) is set in
the process. No tool offered to a model, MCP tool, token or workspace message reaches `answer`: a
tool named for a request or an answer is on the person-only floor, a call whose text names
`ml-stack-requests` or the request store is refused, and `ml-stack-requests` refuses a process an
agent started. An answer that approves something human-only (a release, a host, a grant) is not
the permission: the action checks again, in the process that carries it out, that no agent
started it.

An expired or unanswered request is a denial. A store that cannot be read or written denies: the
handle `raise_request` returns answers `denied` at once, and nothing is approved.

## Where it is kept

`~/.ml-stack/requests/requests.enc` (under the state root): AES-256-GCM under the `requests` subkey
of the user's master key in the OS keystore, one file per user, no plaintext field. At most 100
requests wait and 400 are kept; a resolved request is dropped after seven days. Every raise,
answer and withdrawal is an activity record (`request.raised`, `request.answered`,
`request.cancelled`: kind, agent, project, choice and way, never the subject). Relations are fields
of the row (request, agent, project), not graph edges: the module sits in the core layer, below the
graph.

A process that cannot read the keystore (a background one) cannot keep requests. Sentinel's dialog
then keeps its own requests in memory for that process, so the dialog still shows and answers what
that process raised.

## Terminal

```
ml-stack-requests list [--state S] [--agent A] [--project P] [--kind K] [--json]
ml-stack-requests show ID
ml-stack-requests answer ID CHOICE [--fingerprint FP]
ml-stack-requests watch [--once]
```

`answer` shows the request and asks for a yes unless `--fingerprint` (the one `show` printed) is
given. `ml-stack-security status` and `chip` carry the number waiting.

## The chat, memory and sentinel

* `ml-stack-chat`: each confirmation is a `tool_call` request carrying the classifier's reason. The
  prompt is unchanged (1 allow this time, 2 always allow, 3 never allow, Enter no). An answer made
  in the UI or the dialog while the terminal waits settles the prompt. Always allow saves a rule
  only from the terminal; from the UI the call runs once and no rule is saved.
* `remember`: a `memory_remember` request with the same menu.
* Sentinel: a quarantine that needs a person raises one `quarantine_release` request listing what
  is held. The single-flight dialog shows the oldest waiting request and answers it through the same
  module (`ML_STACK_NOTIFY=off`, the cooldown and one dialog at a time are unchanged). Release,
  Keep held and Later are carried out by sentinel for the ids the answered request lists.

## The page

A shell mounts `RequestsApp.dispatch(method, path, headers, body)` from `ml_stack.inbox.route`
under `/requests` in its own server (no server of its own lives here). A person's terminal gets
`RequestsApp.launch_key`; the link `/requests?k=KEY` opens one browser session once; the session
cookie (`HttpOnly`, `SameSite=Strict`, path `/requests`) and a CSRF token are delivered only inside
the page. The element lists waiting requests grouped by agent and project, history below, filters
by agent, project and kind.

A POST that answers needs: a loopback `Host`, `application/json`, an `Origin` equal to the page's
origin, `Sec-Fetch-Site: same-origin` when sent, the session cookie, `X-Requests-CSRF`, a body of at
most 4096 bytes, fewer than 20 answers in 10 seconds, and the fingerprint. The page sends a content
security policy with no inline script, builds every node with `textContent`, shows no link, and
holds no token. "Answer all of this kind" is offered only for waiting requests of one kind in one
agent and project, none destructive or human-only, lists exactly what it covers before it sends,
and the server refuses a destructive or human-only request in it.

The shell puts `<ml-requests></ml-requests>` in its page with the `<meta>` from
`RequestsApp.meta(session)` (or serves `/requests` itself); the element emits `ml-requests-count`
(`{pending}`) for a chip, and takes the attributes `api`, `csrf`, `agent`, `project`, `kind` and
`history`.

## Limits

* The launch key and session protect the browser path from a program that is not given the link;
  a program that can read the person's terminal or browser can use it.
* Fleet join requests (`fleet/onboard/requests.py`) and `ml-stack-security approve-host` keep their
  own state machines and typed confirmations; they are not raised here yet.
* The reputation notice (`reputation/notice.py`) keeps its own dialog.
* A dialog raised for sentinel can show another component's older request first.
* Hold windows and undo (`docs/notes/undo-mode.md`) are not request states yet.
