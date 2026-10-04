# Send-to-wake latency

`tests/test_workspace_live.py::test_a_waiting_agent_wakes_within_milliseconds_of_a_send` starts N
real agent processes, each blocked in `Workspace.wait`, then sends and times (wall clock, same
machine) from just before the `send` call to the moment each process's `wait` returns. Python start-up
is excluded (each process says it is ready first). It asserts p50 under 50 ms and p99 under 150 ms.

Machine: 16 cores, 128 GB, macOS 26.6 (arm64), 2026-10-03, branch `feat/board`, a temporary
workspace on the local disk. Run: `python3 -m pytest -n 0 -q -s tests/test_workspace_live.py`.

| Agents | Delivery | p50 | p99 |
|---|---|---|---|
| 1 | direct message | 2.2 ms | 2.2 ms |
| 10 | direct message, one each | 2.0 ms | 2.5 ms |
| 40 | direct message, one each | 3.9 ms | 8.5 ms |
| 1 | board post, subscribed | 2.4 ms | 2.4 ms |
| 10 | one board post, 10 subscribers | 3.9 ms | 4.3 ms |
| 40 | one board post, 40 subscribers | 9.4 ms | 38.4 ms |

The person's page long poll (`/board/wait`), five direct messages: p50 1.5 ms, max 1.9 ms.

Before the change, with the agents as threads of one process (so the interpreter lock is part of
the number), one direct message woke its reader in 68 ms, almost all of it the `send` itself: 28 ms of
its 52 ms went to rewriting a durable JSON file for the rate limit (an atomic write and a sync per
message) before the message was appended. The rate counter is now one small file per sender,
appended without a sync, and the wake is signalled as soon as the row is on disk, ahead of the audit
row. A send costs about 24 ms on a disk where a sync takes 12 ms (two chained appends) and 2 to 6 ms
on a fast one. The wake pipes themselves were never the delay. A board post wakes its subscribers
(plain id) and everyone following it (`.follow`, `.chat`, `.web`) through the same call.
