# Admission control: one broker, any number of servers, one request at a time

Any number of model servers may be up at once, as long as they fit the memory the machine
allows. Requests to them run one at a time per accelerator pool. Every server start, reuse
and release goes through the broker, and every generation request takes its turn in the
pool's queue. None of this is something a consumer opts into.

## How a lease flowed before

```
caller -> ServerManager.lease(spec)          serve/manager.py
            adopt(spec)                      a healthy server on spec.port is taken, whoever
                                             started it, if the model name matches
            _beside(spec)                    a busy port: start on a free one if free_memory()
                                             is 0.8 over the weights file size
            _launch(spec)                    backend.start(spec, lease=_pending(spec))
            _record(spec, info)              servers.json: port, pid, owner_pid, model
```

`Broker` (`serve/broker.py`, `serve/broker_wire.py`) was a second, separate way in: a daemon
that started servers for a purpose and recorded who held them in `broker-leases.json`. Only
`ml_stack.bench.tree` used it. `ServerManager.lease` did not know the broker existed, so a
caller that used the manager directly (every other caller) got:

- no memory check beyond the weights file against free memory, only for a busy port;
- no knowledge of the other servers up: three callers loading three models each started
  their own `llama-server -ngl 99`, and `servers.json` listed all three while
  `broker-leases.json` was empty;
- no limit on requests: nothing stopped three clients from generating on the one GPU at the
  same moment, which made each slower than running them one after another;
- any healthy llama-server on the port was adopted, including one somebody started by hand.

## How it flows now

```
caller -> ServerManager.lease(spec) -> broker.start(spec)
            (the machine's broker over its socket;
             an in-process Broker for a manager with its own backend or lease file)
          Broker.start -> ServerManager._start_server(spec)       private to both
            adopt(spec)              only a server on the lease registry, matching the spec
            _reusable / _compatible  a running server on another port that serves the spec
            _admitted                under servers.admission.lock:
                                       reuse, wait for a compatible server still loading,
                                       rate memory, stop servers a dead process left, write the
                                       pending record
            backend.start(spec, lease=...)
          Broker records the holder (pid, label) and returns ServerInfo(lease=...)

caller -> request_json / request_stream / Client -> ml_stack.http
            gate.turn(url)           FIFO queue of the pool the server is in
```

### The broker is the only way in

`ServerManager.lease`, `escalate`, `release` and `detach` call the broker. `_start_server`,
`_escalate`, `_detach` and `_stop_server` carry a leading underscore and are called by the
manager and by `Broker` only; the backends refuse to launch without the manager's `Lease`.
`tests/test_serve_no_bypass.py` reads the source and fails when:

- a module calls `backend.launch` that is not a backend, or a backend calls it without
  `claim_port`;
- a module starts a process and names the llama-server binary without being listed;
- `backend.start(..., lease=...)` or `Lease(...)` appears outside the manager;
- a private start is reached from any module but the manager and the broker;
- `Broker(...)` is built anywhere but `broker_wire.broker_for`;
- a module other than `ml_stack.http` and the fleet proxy opens a connection with `urlopen`.

`broker_for` picks the transport: the machine's broker when the manager keeps the machine's
lease file and uses the ordinary llama.cpp backend (its binary and build travel with the
call); otherwise a `Broker` in the process, over the same registry, admission lock and
request queue. `ML_STACK_BROKER_LOCAL=1` chooses the in-process broker for the machine's
manager too. No argument of `lease` turns the broker off.

`on_event` and `say` are called only for an in-process broker; the machine's broker runs
in another process and answers with the result.

### Memory

`admission.check` adds the estimate of each live server on the registry (its recorded
estimate, else its weights file times 1.1) and each unmanaged `llama-server` (its resident
size) to the estimate of the server being started: weights, draft and projector files, the
KV cache read off the GGUF header, 512 MiB for the runtime. It rates the sum against
`hub.room()` (the machine's wired limit, capped by `ml-stack-serve limits --memory`):

| share  | rating | what happens                                                       |
|--------|--------|--------------------------------------------------------------------|
| < 0.80 | green  | the server starts                                                  |
| < 0.95 | yellow | it starts and the share is said                                    |
| >= 0.95 | red   | the start waits, then raises `AdmissionRefused` naming the holders |

The wait is `ML_STACK_ADMISSION_WAIT_S` (default 60). Entries whose process has gone are not
counted, and are removed from the file the next time it is written. When the rating is red,
a server whose leasing process has gone (`owner_pid` dead, the server alive) is stopped
and the rating is taken again. There is no limit on the number of servers unless
`ml-stack-serve limits --servers N` sets one.

### Reuse

A lease for a model that a running server already serves, with at least the context per
slot and slots asked for, the same embedding setting, and a projector if one is asked for,
returns that server (`adopted=True`) instead of loading the weights again. A lease that
arrives while a compatible server is still loading waits for it. A caller that needs a
specific port (`roam=False`) does not get another.

### Requests

`ml_stack.gate` queues generation and embedding requests (`/v1/chat/completions`,
`/completion`, `/v1/embeddings`, `/embedding`, `/infill`, `/v1/messages`) to a loopback
port on the lease registry. The pool is `gpu` for a server with offload layers and `cpu`
for `-ngl 0`. A request takes a ticket in `<state>/gate/<pool>/`, named by the time it was
taken, and holds an exclusive file lock on it while it runs; the oldest live ticket runs.
A process that dies releases its lock, and the next request removes the ticket. A request
that has waited `ML_STACK_REQUEST_WAIT_S` (default 600) raises `ServerError` with status 429
naming the request ahead of it: pid, label, how long and which server. Requests to
different servers in one pool queue behind each other, requests to different pools do not,
and a thread that already holds the pool is not queued again. A streamed answer holds its
turn until the stream is read to the end or closed. Servers that are not on the registry
(including unmanaged ones that were not adopted) are not queued.

Parallel requests are the one escape hatch, and it is named: `ML_STACK_PARALLEL_REQUESTS=1`
for the process, or `with gate.parallel("bench sweep"):` for a block of one thread. The first
use of each name logs a warning. The benchmarks that measure streams in flight together
(`bench.speed.cell`, `bench.measure.concurrent`) name themselves and send in parallel.

`ml-stack-serve queue` (and `Broker.snapshot()["requests"]`) lists each pool's line of requests.

### Unmanaged servers

A `llama-server` the registry does not hold is reported as `unmanaged` (`ml-stack-serve
status`, `status --every`, `queue`, the bench's "beside the card" line, `Broker.snapshot()`), counted
against the memory budget, and never leased from, shared, stopped or restarted. Leasing a
port such an unmanaged server holds moves the new server to another port, or is refused for
`roam=False`.

`ML_STACK_ADOPT_UNMANAGED` (or `ml-stack-serve limits --adopt-unmanaged`) sets what the
broker does about them:

| setting | behaviour |
|---------|-----------|
| `off` (default) | report only |
| `ask` | adopt when the manager's `confirm` hook returns true for the description (pid, port, model); with no hook, nothing is adopted |
| `auto` | adopt every one that passes the checks below |

`unmanaged.examine` adopts a listener only when all of these hold, tested in this order and
before anything is sent to it:

1. a process the user can inspect listens on the port;
2. the address is loopback (`127.0.0.1`, `::1`);
3. the process belongs to this user;
4. its executable is `llama-server`;
5. `/health` answers, and `/props` has `default_generation_settings` and an integer
   `total_slots` (two GETs, no credentials, no prompt).

An adopted server is written to the registry with `unmanaged: true` and `owner_pid` set to its
own pid; its estimate is its resident size and its pool is `gpu`. Its requests queue like any
other, `release`, `stop_all`, `stop_all_servers`, `ml-stack-serve down`, idle reclaim and the
orphan sweep leave it alone, and it leaves the registry when its process ends. The adoption is
logged with pid, port and model.

## Behaviour on one machine with three callers

Three processes each call `ServerManager().lease` for a different model on a machine that
allows 10 GiB and the models are 2 GiB each:

- before: three `llama-server -ngl 99` processes, three records in `servers.json` with three
  different owners, an empty `broker-leases.json`, and requests from the three callers
  running on the GPU at once;
- now: three servers owned by the broker, `broker-leases.json` with three holders, memory
  rated green, and requests from the three callers run one after another.

A fourth caller asking for a 6 GiB model is refused with `AdmissionRefused` naming the three
servers and the sum; a fourth asking for one of the three models gets that server.

`tests/test_serve_three_callers.py` runs this against a real broker process
(`pytest --slow`).

## What is not covered

- Request queueing applies to servers on the registry. A server nobody registered is not
  queued, and neither is one on another machine.
- Pools are `gpu` and `cpu`; a machine with several GPUs has one `gpu` pool.
- A process that holds a turn and neither finishes nor dies holds the pool until its request
  times out.
- A server started by `ml-stack` in a process that has since exited without releasing is
  stopped when the memory is needed (local broker) or when the machine's broker has found
  no holder for it for `idle_s` (default 600 s, `ml-stack-serve limits --idle`).
