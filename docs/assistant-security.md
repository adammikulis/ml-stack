# Security contract for an assistant that does things for a person

ml-stack is growing from a serving library into something closer to a personal assistant: it
reads mail and files, talks on chat channels, runs commands, remembers things. The first
generation of such assistants failed in a familiar way. Reports on OpenClaw list the same
structural faults: no permissions model, no sandboxing, credentials in plain text, unsigned
extensions (and many malicious ones), exposed instances, a web page able to hijack the local
agent, and one user's data reaching another's session. Those are design choices, so the fix
is a design contract that every new capability must meet before it ships. This page is that
contract. Where a rule is already built, the page says where; where it is not, it says so.

## The promise

Nothing happens that the person did not knowingly allow, and the person can always see what
the assistant can do right now, stop it, and take a grant back.

Prompt injection cannot be removed, only contained. Text the assistant reads (mail, pages,
chat messages, file names, model cards, memory) can always try to give it instructions. The
rules below therefore limit what a fooled assistant is able to do, instead of trusting a
filter to catch every attempt.

## The rules

1. **Off until granted.** Every integration (mail, calendar, files, shell, browser, a chat
   channel) is a named capability, off by default. A grant says in plain words what it can
   read, write and send, on which account or folder, for how long. It is the narrowest
   thing that works, it expires, and it is listed and revocable in one place.
   *Built:* roles (`read-only`, `approve-first`, `plan-and-go`) and saved Always/Never rules
   (`docs/agent-roles.md`). *Not built:* a grant ledger that covers connectors.

2. **No ambient authority.** The model never holds a credential. A separate connector process
   holds each secret in the OS keystore and makes the call on the model's behalf, with the
   narrowest scope the service offers. An injected instruction then finds nothing to steal.
   *Built:* secrets in the keystore for memory and fleet signing; redaction rails.
   *Not built:* the connector broker.

3. **Never combine private data, untrusted content and a way out in one session without a
   person.** Once a session has read outside text, any tool that sends data out, spends money
   or changes something durable asks the person again, and says why. *Built:* the taint rail
   asks again after outside text was read. *To do:* make it a hard rule for every
   outward-sending tool, not only the ones that ask today.

4. **Each integration runs in its own sandboxed process** with its own egress allow-list, so
   a compromised calendar connector cannot read files or reach an arbitrary host.
   *Built:* the sandbox (Seatbelt; bubblewrap argv builder) and the net pipeline's host
   allow-list. *Not built:* per-connector policies.

5. **Authenticate every way in.** A chat channel, web UI or local port accepts only a paired
   identity; messages from anyone else are untrusted data with no tool access. Local HTTP
   endpoints check origin and fetch metadata, so a web page cannot drive the agent.
   Nothing listens beyond loopback unless the person turned it on. *Built:* paired fleet
   peers with pinned TLS, capability tokens on the agent workspace, a loopback-only graph page that
   refuses another Host or Origin. *To do:* the same standard for any
   new channel.

6. **No marketplace.** An extension is data plus declared capabilities, with a manifest signed
   and pinned by hash. An update shows its diff and needs the person's approval; installing
   one is never triggered by text the assistant read, and fetched content never becomes
   code. *Built:* pinned artifacts and signed manifest lists for models.

7. **Small by construction, and checked.** Every dangerous call is inventoried and a gate
   fails when a new one is unlisted: the budgets ratchet (`scripts/budgets`) only lets the
   surface shrink, and the red-team coverage map fails when a surface has no attack test.

8. **Small blast radius per task.** A task gets a scoped folder or worktree, never the home
   directory by default. User memory and project memory are separate stores for the same
   reason. *Built:* per-user, per-project encrypted memory (`docs/memory.md`).

9. **Visible, stoppable, and honest in the UI.** One command lists everything the assistant
   can do right now, in plain words. Every grant, action and refusal is in a tamper-evident
   log, and one switch revokes everything. A notice the person can only dismiss is a bug:
   each prompt carries a button that does the thing. The person must never be shown a
   screen with nothing to do. *Built:* the single-flight click-to-release dialog and the
   hash-chained event log (`docs/sentinel.md`). *Not built:* the one command that lists
   everything the assistant can do now, and the one revoke-everything switch.

10. **Assume failure and detect it.** Canaries, decoys and honeytokens extend to each
    connector; rate limits and tool-mix anomaly checks watch for a steered agent.
    *Built:* sentinel, decoy endpoint, honeytokens, scan loop.

## Human-only floor

These are never offered to any model, role or connector, and no stored fact, saved rule or
channel message can unlock them: releasing a quarantine, approving a network host, minting a
grant, changing guard or sentinel policy, changing a role, creating a saved rule, and changing
the GPU wiring limit (`iogpu.wired_limit_mb`) or the boot-time daemon that keeps it. A person
does them at their own screen: the Settings slider and `ml-stack-serve memory` refuse a process
an agent started, an access token, another web page and another machine, and the password goes
only into macOS's own dialog or sudo.

## What a new integration must bring

An integration is not merged until it has all of these:

* a one-sentence description of what a grant gives it, in words a non-engineer can follow, and
  the worst thing it could do with that grant;
* its capabilities declared in the capability map, each with the reason it is safe;
* its own process, credential and egress list (rules 2 and 4);
* the outward-send behaviour under rule 3;
* red-team entries in the coverage map for every place it takes untrusted text (mail bodies,
  attachments, message senders, page text, file names), each mutation-checked;
* a way to see it in the "what can it do now" listing, and a way to revoke it.

## Limits we state plainly

* A filter, a judge model or a sentinel can miss an attack. They reduce the odds; rules 1 to 4
  decide how much a miss costs.
* A grant prompt protects only as far as its wording is short and true. Wording is reviewed
  like code.
* Same-user malware can press a button on screen. Release by click defends against an agent
  that is steered by text, not against a program already running as the person.
* Plaintext of decrypted memory exists in process memory while the assistant runs.

## Order of work

1. The capabilities listing and a grant ledger that the roles and rules already feed.
2. The connector credential broker, then the first connector (read-only), under rules 2 to 4.
3. The hard outward-send rule across all tools.
4. Channel authentication for the first chat channel.
5. The extension policy, before any extension mechanism exists.
