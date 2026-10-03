# Workspace workflows

The daemon workspace groups interactive screens in its sidebar. Tools exposes every installed
`ml-stack-*` console entry point, its help, command review, monitored execution, logs, and cancellation.
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
| Agent and MCP workflows | Tools: ml-stack-agent, ml-stack-do, ml-stack-claude, ml-stack-mcp |
| Serving, drafts and terminal chat | Tools: ml-stack-serve, ml-stack-draft, ml-stack-chat |
| Libraries, machine preferences, updates | Settings |

## Native environment integration

Gym forwards JSON configuration to each environment's native adapter. Car uses MetaDrive's
native rendered frames. Warehouse and traffic visualizations render simulator snapshots in
locally packaged Three.js and OrbitControls; display coordinates do not advance the simulator. Each session exposes numeric
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
