# Workspace load: many agent processes on one machine

`scripts/experiments/workspace_load.py` starts N agent processes, each with its own token,
against one temporary workspace. Each sends to its neighbour, waits for mail, claims a port and a
branch and renews them, reads the keystore master once (through a counting fake keyring; the OS
keystore is never touched), and every tenth agent also reads threads. It reports latencies,
lock waits, failures by type, log growth, keystore backend calls and CPU, and exits 1 when a
budget is missed. No model is loaded. A small version runs in `tests/test_workspace_load.py`.

    python3 scripts/experiments/workspace_load.py --agents 40 --messages 30 --out run.json
    python3 scripts/experiments/workspace_load.py --default-limits   # keep the shipped rate limit

Machine: 16 cores, 128 GB, macOS 26.6 (arm64), 2026-10-03, working tree `feat/swarm-scale`. The
"before" run is the same script against `integration/dev` as it stood when this branch started
(`workspace-load-40x30-before.json`).

## Budgets

| Measure | Budget |
|---|---|
| send latency p99 | 1 s |
| delivery latency p99 (send to read, after the reader is up) | 2 s |
| inbox read p99 | 0.25 s |
| bus append lock wait p99 | 1 s |
| `KeystoreBusy` | 0 |
| failures other than the shipped rate limit | 0 |
| messages sent but not received | 0 |
| CPU per agent per message | 0.25 s |
| the bus chain verifies and holds every sent row | yes |

## Results, 40 agents x 30 messages (1200 messages), limits opened

| Measure | Before | After |
|---|---|---|
| verdict | fail: 22 `KeystoreBusy` | pass |
| wall, throughput | 15.6 s, 77 msg/s | 11.4 s, 106 msg/s |
| send p50 / p99 / max | 28 ms / 291 ms / 783 ms | 2.1 ms / 32 ms / 58 ms |
| delivery p50 / p99 | 48 ms / 189 ms | 1.1 ms / 41 ms |
| bus lock wait p99 | 245 ms | 0.06 ms |
| bus lock hold p50 / p99 | 5.0 ms / 9.9 ms | 0.19 ms / 0.32 ms |
| inbox read (1200-row log) | 9.2 ms | 0.13 ms |
| thread read p50 | 4.4 ms | 0.15 ms |
| keystore read per process p50 / max | 2.5 s / 6.0 s | 0.32 s / 4.2 s |
| keystore backend calls | 19 reads, 1 create, 22 refused | 41 reads, 1 create, none refused |
| CPU total, per agent per message | 44.7 s, 0.037 s | 5.1 s, 0.004 s |
| log | 428 KB bus, 378 KB audit, chain ok | 428 KB bus, 378 KB audit, chain ok |

## Results, 100 agents x 20 messages (2000 messages), after

Pass. 158 msg/s over 12.7 s. Send p50 3.4 ms, p99 58 ms. Delivery p50 1.3 ms, p99 151 ms. Bus lock
wait p99 0.15 ms, hold p99 0.42 ms. Inbox read 0.2 ms. 101 keystore reads and 1 create, none
refused. CPU 13.7 s total. Keystore read per process p50 1.1 s, max 6.7 s (100 processes
queue for one lock; each read is a Python start-up plus an HKDF, the waiting is the queue).
Bus 714 KB, audit 674 KB, chain ok (`workspace-load-100x20.json`).

## Component measurements

Taken with throwaway scripts on the same machine, before and after.

| Measure | Before | After |
|---|---|---|
| 30 writer processes, 40 appends each: throughput | 189/s | 1249/s |
| same: append p50 / p99 | 5.3 ms / 1039 ms | 0.3 ms / 177 ms (the p99 is each process's first append, a full read of the log) |
| one append to a 20 000-row log (5 MB) | 127 ms | 0.5 ms after the first |
| inbox, `get`, `thread` on that log | 119 ms each | 0.1 ms each after the first |
| 50 waiters, 5000-row log, 3 s idle: CPU per waiter | 0.68 s (23% of a core) | 0.04 s |
| same: wake latency p50 / p99 after a send | 193 ms / 311 ms | 45 ms / 97 ms (50 processes scheduling) |
| 50 waiters, tiny log: wake p50 | 68 ms (poll tick) | 22 ms |

## What the numbers say

* Before: every append, inbox, thread or pending check walked and hash-verified the whole log,
  so one append cost grew with the log (127 ms at 20 000 rows) and held the lock the whole time;
  50 idle waiters polling 10 times a second kept the CPU busy. After: a process remembers the
  rows it verified and reads only the bytes added since; a wait sleeps on a named pipe until a
  sender signals it, with a 0.1 s to 2 s jittered backoff as the fallback.
* The keystore ceiling of 20 per hour was the first thing a swarm hit: 22 of 40 starts failed.
* Not done: batching fsyncs. Lock hold is 0.2 ms with the fsync on this machine, so there is
  nothing to batch; on a filesystem where fsync costs milliseconds the hold time is the number
  to watch (`bus_lock_hold_s`).
