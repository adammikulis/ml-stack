# Workspace workflows

The daemon workspace groups interactive screens in its sidebar. Tools exposes every installed
`ml-stack-*` console entry point, its help, command review, monitored execution, logs, and cancellation.
Common controls remain visible; less common settings are grouped under collapsed Advanced
sections. Model serving labels its token-window setting **Context length**.
Setup and Settings use the same slider, from 2,048 through 1,048,576 tokens.
Qwen uses YaRN when the requested per-slot context exceeds its native 262,144 tokens.
The default coding-model server stays at 262,144; selecting a higher limit costs more memory.

Mac device cards show one physical Memory meter, using total installed RAM and current
usage. Metal’s recommended working-set allowance is a separate serving limit, never free
physical memory. Each device lists its verified loaded models, per-slot context and
speculative-decoding status; an empty device explicitly says no models are loaded.
Commands execute in the daemon files root with the existing job environment and scheduling gate.

| Capability | Destination |
| --- | --- |
| Conversations, model choice, generation | Chat |
| Model discovery, download, serving | Models |
| Dataset upload, file browsing, previews | Data |
| Supervised, tool caller, decision model, PPO training | Training |
| Sensors, controllers, live native environments | Gym |
| Measurement sweeps, history, capacity and cost charts | Benchmarks |
| Peers, availability, clusters, fleet coordination | Fleet |
| Decision ask, probabilities, evaluation, calibration, fetching | Tools decision lab |
| Graph store and graph server | Tools: ml-stack-store, ml-stack-graph |
| World model, ingestion, walking, retrieval surfaces | Tools: ml-stack-world, ml-stack-ingest, ml-stack-walk, ml-stack-surface |
| Speech providers, transcription and synthesis | Tools: ml-stack-speech |
| Security, audit and credentials | Tools: ml-stack-security, ml-stack-audit, ml-stack-credentials |
| Diagnostics and installation | Tools: ml-stack-doctor, ml-stack-setup |
| Workspace coordination, jobs, suites | Tools: ml-stack-workspace, ml-stack-jobs, ml-stack-suite |
| Agent and MCP workflows | Tools: ml-stack-agent, ml-stack-claude, ml-stack-mcp |
| Serving, drafts, native chat agents and memory | Tools: ml-stack-serve, ml-stack-draft, ml-stack-chat, ml-stack-memory |
| Libraries, machine preferences, updates | Settings |

## Native environment integration

Gym forwards JSON configuration to each environment's native adapter. Car uses MetaDrive's
native physics and sensors, displayed in the default sensor scene or its native camera.
The locally packaged Three.js and OrbitControls display simulator snapshots; the car
scene adds a follow camera, physical materials, environment lighting and native sensor
overlays. Display coordinates do not advance the simulator. Each session exposes numeric
observations, controller decisions, applied actions, reward, episode information and its trajectory
artifact path. Downloaded snapshots contain the browser-observed sequence; native trajectory
artifacts contain the authoritative episode record.

## Files and jobs

Data paths resolve under the daemon files root and reject traversal or escaped symlinks.
Uploads create new UTF-8 files up to 10 MB. Previews are limited to 1 MB. Use the peer file-transfer
CLI for larger files. Training paths are relative to the same files root. Gym training stores its
own artifacts under the Gym cache.

The specialist runner accepts installed ml-stack commands and a JSON array of CLI arguments.
Command review produces the exact argv without running it. Help is executed as a monitored job.
Detached execution is rejected so the runner retains lifecycle ownership. Jobs use the daemon's
existing capacity and resource gate; failures, output, structured metrics and cancellation remain
visible in the run inspector.

## Reviewed decision datasets

The Gym recording panel lists persisted native trajectories. Open a recording, inspect its exact
pre-action observation and decision, and choose a native action label. Uncertain steps can be
skipped. Labels are keyed by episode and sequence, and may be updated before export. Export
includes reviewed transitions only, with episode grouping preserved for train/test isolation.
The exported JSONL path is handed directly to the decision-model training form. Load the trained
checkpoint directory into a live decision controller to test it on fresh scenarios.

Gym dependencies install through Settings' managed environment and the environment catalog's
Install button. An existing simulator interpreter can be selected with `ml-stack-traind
--root ROOT --gym-python PYTHON`; keep the same root on later launches to retain completed
setup and the saved interpreter. `ML_STACK_GYM_PYTHON` takes precedence over that selection,
which takes precedence over the managed environment. Readiness reflects the selected
simulator interpreter's package versions and the wheel's selected extra requirements. The daemon's own packages are reported separately from job readiness.

See [Studio and live Gym](studio-gym.md) for native ownership, the winding-road stop task,
recordings, and the PPO/decision-model training loop. In the combined traffic-driving
example, MetaDrive IDM drives vehicles and learned policies control SUMO-RL signal phases
at one intersection.

## World construction

Traffic and combined traffic-driving worlds support native SUMO procedural generation
(one controlled intersection) and manual network/routes XML from Data. Warehouse worlds
support seeded native RWARE lattice parameters and manual rectangular ASCII layouts.
The per-example `world` configuration keeps native field names; world artifact manifests
record resolved definitions, seeds, native versions and output hashes. World construction
and simulation lifetime are separate controls. See [native world configuration](studio-gym.md#native-procedural-and-manual-worlds).

Persistent warehouse and traffic worlds keep their native instance across learning-task
boundaries. SUMO continues its clock and inserts continued demand through native TraCI;
warehouse robots retain their identities and request queue. `task_horizon` limits learning
steps, while `world_demand_period` sets continuing traffic demand in seconds. Construction,
world lifetime and model update policy are separate settings.

## Saved conversations

Chat keeps conversations on the daemon's machine. The conversation sidebar searches titles
and message text; its options menu renames or deletes a conversation. Deletion asks for
confirmation. Reloading restores the last open conversation, model choice and saved generation
settings. Temperature stays under collapsed Advanced options and defaults to the model server's
setting. An unavailable saved model remains identified until another running model is selected.

One composer handles messages: Enter sends, Shift + Enter inserts a new line, and Stop interrupts
generation. Existing saved messages retain their history when the conversation acquires versioned
settings. Settings updates preserve the title and messages and reject invalid values.

Conversations, messages, chosen models and project relationships live in the daemon's embedded
graph store, at `chats/conversations.db`. The first access imports existing JSON histories once;
the original files remain as backups. Subsequent edits and deletions use the graph exclusively.
Writes run in transactions under the existing file lock, including concurrent message appends.
Chat does not request a credential or encryption key when opening its history. The standalone
app bundles the graph engine; word and vector indexes load their extensions when used.

Open **History** in the navigation to inspect recorded actions grouped by agent. Each
agent card lists its newest actions first; open an action for its status, timestamp,
session, model, task and project references, and available issue links. Filter by
agent or search for a tool, model, task, or outcome. Refresh keeps expanded actions
open, and the page refreshes every ten seconds while visible.

History reads the maintained per-user activity log, with up to 500 recent actions.
It reports unreadable or dropped records. Prompts, tool arguments, outputs, and
message bodies are not stored there; use Board or the originating conversation
for message content. History is read-only and requires the normal Fleet UI access.

History also shows earned credit balances independently of recent activity. Open an
account to inspect completion credits, independently verified quality bonuses, and
verification evidence with the reviewer, checks and artifact hashes. Credits and
work reputation are separate: quality and reliability ratings show their review
sample count and confidence; an account without an independent review is not yet
rated. Compute usage appears only when recorded, with available token counts and
elapsed time. **Runs are free**; credits are not spent and ratings do not restrict
access. Model names in actions describe the work performed, not separate balances.

Structured tasks use the workspace coordination graph. The authenticated Tasks API at
`/ui/tasks` lists tasks and accepted-outcome metrics; `?id=task:…` returns the task's
resource lease, checkpoints, proposal artifacts and independent reviews. Person requests
can create a specification or review a submitted proposal. Browser requests cannot claim
worker resources or supply a worker identity. Worker claims require a live scheduler
allocation and expire without renewed heartbeats. Checkpoints and submitted checks remain
worker claims; completion requires independent passed checks covering the acceptance
criteria. Infrastructure blockage is distinct from rejection and earns no completion credit.

Designated peer assignees and reviewers need an existing person-set project grant matching
the task's permission metadata. Listing an identity in a task creates no new authority.
An actual worker cannot review its own proposal. Execution paths come from the verified
scheduler allocation, never from task descriptions or browser-supplied permission metadata.

Expired working leases require an explicit authorized recovery, which records a checkpoint
and consumes the task retry budget. Blocked tasks require an authorized resume with a
recorded reason after their blocking condition is addressed; the scheduler cannot silently
retry them. Recovery and resume do not grant native resources: the next claim still verifies
a fresh live allocation. Configured task limits currently cover model selection, wall time
and retry count; unsupported memory/context/token guarantees are not inferred from labels.
