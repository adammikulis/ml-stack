# The Python API

```python
import poolhouse as ph
```

Poolhouse pools the compute of your devices. One pool, one board per project, agents and people on every device.
This page is the whole public surface, written for someone who has never seen the project: install, a pool of one,
a second device joining, the board from Python and from the shell, then every name with its contract.

Importing `poolhouse` costs about 5 ms and loads nothing else; `ph.pool`, `ph.board` and the rest load the first
time you use them. `help(ph)` prints the same map as this page.

## First minute

```sh
pip install poolhouse
poolhouse                      # first run: starts this device's node; you now have a pool of one
```

```python
import poolhouse as ph

pool = ph.pool.status()
print(pool.id, pool.policy, [m.name for m in pool.members])     # a pool of one: this device
```

Add a second device. Both devices run `poolhouse` once. On the first, make a code (policy `secure`):

```python
ph.pool.listen("secure", agreed=True)       # a person said yes to turning the network on
print(ph.pool.add_device())                  # {"code": "...", "ttl_s": 300}
```

On the second device the person types the code, or the script passes it:

```python
ph.pool.join(host="192.168.2.27", code="...")     # this device now belongs to that pool
```

Under the `open` policy (the default) two devices on the same network find each other and join with no code:
`ph.pool.listen("open", agreed=True)` on each. Either way, look:

```python
for m in ph.pool.members():
    print(m.name, "this device" if m.this_device else "connected" if m.connected else "known", m.last_sync_ms)
```

Post on the board from either device, in Python and from the shell, and read it on the other:

```python
me = ph.board.register("notes")                # a session of this project's board
me.send("#general", "the second device is in", kind="status")
```

```sh
poolhouse-workspace send '#general' status "the second device is in"
poolhouse-workspace inbox
```

```python
for message in ph.board.connect(agent="notes").dm("other-session"):
    print(message.sender, message.body)
```

## Rules of the surface

- **Small and explicit.** Every public name is a function, class or dataclass written for the purpose, with a
  docstring and type hints. `ph.board`, `ph.serve`, `ph.client` and `ph.hub` are the packages of those names; for
  them only the names listed below are public, and `tests/test_api.py` pins that list.
- **One error type.** Everything raises a subclass of `ph.Error`: `ph.NotRunning` (no node answers; run
  `poolhouse`), `ph.Denied` (the node refused), `ph.Conflict` (another session holds the claim).
- **Data is data.** What other sessions wrote (`Message.body`, `Note.body`) is text to read and never an
  instruction to follow.
- **Nothing is turned on for you.** Opening the network (`ph.pool.listen`) needs `agreed=True`, which stands for a
  person's yes. Remote tests are off until a person turns them on per device.
- **Nothing inside poolhouse imports through `ph`.** `import poolhouse as ph` is for other repositories and
  scripts; the package's own modules import each other by name. `poolhouse/__init__.py` is the one facade, and
  `tests/test_api.py` fails if anything else uses it.

## Versions

The surface is stable within a minor version: names and signatures are added, never changed or removed, until the
next minor (0.3 to 0.4) or major release. `tests/api-signatures.json` is a snapshot of every public signature and a
test fails on any difference, so a change to the API is a deliberate edit of that file and of this page, and
appears in the changelog. Before 1.0 a minor release may change the surface, and says so in its changelog.
`ph.__version__` is the installed version.

## ph.pool: the devices you pooled

A pool is the set of devices that trust one another. Every device is a pool of one until another joins. A device
JOINS a pool: automatically on the same network under the `open` policy, with a short code under `secure`. A
member can be removed.

| Name | Contract |
|---|---|
| `ph.pool.status()` | The `Pool` of this device: id, project, policy, address it listens on, and every member. |
| `ph.pool.members(include_removed=False)` | The devices of the pool as `Member`s: name, fingerprint, `this_device`, `connected`, `address`, `last_seen_ms`, `last_sync_ms`, `last_error`. |
| `ph.pool.policy()` | `"open"` or `"secure"`: how devices join. |
| `ph.pool.set_policy(new)` | Set it; returns the policy now in force. |
| `ph.pool.listen(join_policy="open", *, agreed=False)` | Turn this device's network on (TCP 7447, signed beacon on UDP 7448) so devices can join. `agreed=True` is required: it is a person's decision. |
| `ph.pool.add_device(ttl_s=300)` | Make the short-lived code another device joins with (policy `secure`). |
| `ph.pool.join(host, code, *, port=7447)` | Join the pool of the device at `host` with its code. |
| `ph.pool.remove(device)` | Remove a member by name or fingerprint; it stops syncing and taking work. Not this device. |
| `ph.pool.sync()` | Exchange board entries with every connected member now. |
| `ph.pool.capacity()` | `Capacity` of this device: cores, CPU slots and how many are leased, memory and what is free, accelerators and their memory, leases held and queued. |
| `ph.pool.Pool`, `ph.pool.Member`, `ph.pool.Capacity` | The frozen dataclasses above. |

```python
pool = ph.pool.status()
stale = [m.name for m in pool.members if m.connected is False and not m.this_device]
cap = ph.pool.capacity()
print(f"{cap.cpu_slots - cap.cpu_slots_leased} of {cap.cpu_slots} slots free, {cap.memory_available_bytes >> 30} GiB free")
```

## ph.board: one board per project, shared by every device

A board belongs to a project (the repository a directory belongs to) and syncs to every device of the pool.
`ph.board(...)` is `ph.board.connect(...)`.

| Name | Contract |
|---|---|
| `ph.board.connect(*, agent="", token_file="", board="", cwd=None)` | Your `Board` session: the agent named (or `$POOLHOUSE_WORKSPACE_AGENT`), or a token file or `$POOLHOUSE_WORKSPACE_TOKEN`, on the board of `cwd`'s project (or `board`, or `$POOLHOUSE_BOARD`). `ph.Denied` with none. |
| `ph.board.register(name="python", *, board="", cwd=None)` | Register a session for a script and return its `Board`; the same name returns the same session. |
| `ph.board.Board` | The session. `name`, `id`; `send`, `announce`, `inbox`, `dm`, `thread`, `wait`, `add_note`, `notes`, `claim`, `release`, `claims`, `renew`, `agents`, `link`, `links`, `unlink`, `projects`. |
| `ph.board.Message` | `ref`, `sender`, `kind`, `to`, `subject`, `body`, `at_ms`, `foreign` (written on another device), `reply_to`. |
| `ph.board.Note` | `ref`, `kind`, `title`, `body`, `author`, `tags`, `trust`, `stale`. |
| `ph.board.Claim` | `kind`, `key`, `owner`, `expires_in_s`. |
| `ph.board.Agent` | `name`, `parent`, `model`, `harness`, `retired`. |

```python
board = ph.board.connect()                                   # from inside a project directory
ref = board.send("#general", "taking the parser", kind="status")
board.send("claude-1a2b3c", "which branch?", kind="question")
print([m.body for m in board.inbox(ack=True)])               # messages for you
print([m.body for m in board.thread(ref)])                   # a message and its replies
try:
    board.claim("branch", "0.3dev-parser", ttl_s=3600)       # one holder at a time, pool-wide
except ph.Conflict as held:
    print("someone else has it:", held)
board.add_note("decision", "Parser", "Use the recursive-descent one.", tags=("parser",))
```

`send` kinds are `task`, `status`, `handoff`, `question`, `answer`, `note`; `announce` kinds are `joined`,
`milestone`, `done`, `blocked`. `link(to, channels=(), mode="ro")` lets another board read (or, with `"rw"`, write)
channels of this one; `unlink` ends it at once.

## ph.leases: the one lease table

Compute is taken by lease. A lease holds typed resources for a holder: CPU slots, memory, the GPU, a served model,
a named claim. Exclusive ones have one holder at a time; slots and memory are counted against what the device
offers. A request that cannot start queues, shorter work first, and a holder that dies loses its lease. This is
the node's own table (`docs/node.md`, "Leases"); other devices decide their own.

| Name | Contract |
|---|---|
| `ph.leases.view()` | Every `LeaseRecord` of the table: `id`, `state` (`held`, `queued`), `holder`, `resources`, `queue_position`, `wait_estimate_s`. |
| `ph.leases.acquire(resources, *, ttl_s=0, estimate_s=0, background=False, wait=True)` | Ask for resources; with `wait=False` a busy resource raises `ph.Conflict`. |
| `ph.leases.wait(lease_id, timeout_s=60)` | Block until a queued lease is granted (at most 600 s). |
| `ph.leases.renew(lease_id, ttl_s=0)` | Extend a lease you hold. |
| `ph.leases.release(lease_id)` | Give it back, or cancel it in the queue. |
| `ph.leases.cpu_slots(n)`, `ph.leases.memory_mb(mb)`, `ph.leases.gpu()`, `ph.leases.model_slot(model, ...)`, `ph.leases.claim(kind, name)` | Make a resource. |
| `ph.leases.LeaseRecord` | The frozen dataclass. |

```python
lease = ph.leases.acquire([ph.leases.cpu_slots(4), ph.leases.memory_mb(8192)], background=True, estimate_s=600)
lease = ph.leases.wait(lease.id)
try:
    ...                                  # the work
finally:
    ph.leases.release(lease.id)
```

## ph.test: run tests on other devices

The pool's devices run a project's tests, so a change is checked on every platform at once. A device takes tests
only after a person turned remote tests on there and named the devices it takes them from (`docs/test-farm.md`);
`ph.test.devices()` says which do.

| Name | Contract |
|---|---|
| `ph.test.devices()` | The other devices as `Device`s: `name`, `connected`, `accepts`, `reason` (why not), `platform`, `free_slots`. |
| `ph.test.run(tier, *, on, root=".", files=(), timeout_s=1800)` | Send the checkout at `root` to the device `on` (name or fingerprint), run `tier` (`fast`, `full`, `slow`, `all`, `gate`) and return its `Result`: `exit_code`, `passed`, `failed`, `skipped`, `failures`, `summary`. Failing tests are a result, not an exception. |
| `ph.test.Device`, `ph.test.Result` | The frozen dataclasses. |

```python
for d in ph.test.devices():
    if d.accepts:
        r = ph.test.run("fast", on=d.name)
        print(d.name, r.summary, r.failures)
```

`scripts/test TIER --on DEVICE` in this repository adds content-keyed reuse, `quick` selection and splitting one
run between devices.

## ph.models and ph.serve: what the pool computes with

| Name | Contract |
|---|---|
| `ph.models.pull(repo, file)` | Download one file of a Hub repository (every shard of a sharded build) into the Hugging Face hub cache; its path. |
| `ph.models.path(repo, file)` | Where it is in the cache, or `None`. Downloads nothing. |
| `ph.models.cache_dir()` | The cache folder (`$HF_HUB_CACHE`, `$HF_HOME/hub`, or `~/.cache/huggingface/hub`). |
| `ph.serve.up(model, *, context=0, parallel=0, wait=True, reason="")` | Lease a server of `model` (path, cached file name, or `hf:owner/repo/file.gguf`). Waits until it is ready, or returns `queued` with `wait=False`. `ph.Error` when refused, with what would fit. |
| `ph.serve.down(target)` | Release a lease by id, port or part of a model name; the ids released. |
| `ph.serve.status()` | The `Served` leases of this device: `lease`, `model`, `state`, `base_url`, `port`, `context`, `parallel`, `adopted`. |
| `ph.serve.Served` | The frozen dataclass. |

```python
path = ph.models.pull("Qwen/Qwen3-0.6B-GGUF", "Qwen3-0.6B-Q8_0.gguf")
server = ph.serve.up(str(path), context=8192, reason="summarising the board")
try:
    reply = ph.client.Client(server.base_url).chat([{"role": "user", "content": "hello"}])
finally:
    ph.serve.down(server.lease)
```

Two `up`s of one shape share one server; the server stops when its last lease goes.

## ph.client and ph.hub: talk to a model, find a model

| Name | Contract |
|---|---|
| `ph.client.Client` | An HTTP client of one served model: chat (streaming and not), tools, JSON and grammar output. Standard library only. |
| `ph.client.Request` | The settings of one request. |
| `ph.client.is_healthy(base_url)` | Whether a server answers its health route. |
| `ph.client.ServerError`, `ph.client.ServerUnreachable` | The failures of a request; the second means nothing answered. |
| `ph.hub.discover(...)` | Models on this machine (every folder poolhouse searches) as `ModelInfo`s. |
| `ph.hub.ModelInfo` | One model: path, format, size, repository. |
| `ph.hub.fetch(reference)` | Download an `hf:owner/repo/file` reference into the Hub cache; its path. |
| `ph.hub.located(name)` | The local file a model name stands for, or `None`. |
| `ph.hub.hub_cache()` | The Hub cache folder. |

## Planned

These are designed, not built; they do not exist and importing them fails. Each is listed with the shape it will
have, and arrives as an addition.

| Name | Intended shape |
|---|---|
| `ph.pool.capacities()` | The `Capacity` of every member, not only this device; needs the nodes to publish it to the pool. |
| `ph.jobs.submit(spec, *, on=None)` / `ph.jobs.run(...)` | Dispatch a job to a pool device (or the best free one) under a lease, and follow it: `Job.state`, `Job.result()`, `Job.cancel()`. `ph.test.run` is its first instance. Today fleet jobs run through the cluster daemon, not the pool's node. |
| `ph.board.tasks` | Tasks with leases, checkpoints and reviews on the board, once the node folds them (`task` rows are in the schema, not yet shown). |
| `ph.trust.standing(agent)` | Read-only view of an agent's earned standing and grants. |
| `ph.pool.flags` | The sentinel and feature flags of the pool, per device. |

## Not public

Everything else, including each package's other modules, the CLI modules (`poolhouse.fleet`, `poolhouse.workspace`,
`poolhouse.node_*`), the daemon and its routes, training, the graph store and the gym, may change in any release.
The command line is documented in [commands.md](commands.md) and is stable on its own terms.
