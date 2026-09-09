# The fleet

## Driving it from Python

`ml-stack` has no dependencies, so the machine you drive from needs no CUDA, no MLX
and no training stack.

```python
from ml_stack.fleet import Peer, Requires, Unit, run

peers = Peer.discover()
report = run(
    [Unit(id=f"shard{i}",
          argv=["python", "-m", "ml_stack.train.run", "--recipe", "text-lm",
                "--data", f"shards/{i}.jsonl", "--out", f"out/{i}"],
          requires=Requires(labels=("train",), min_vram_gb=8))
     for i in range(8)],
    peers, kind="text-lm")

for place in report:
    print(place.unit_id, place.peer, place.state, f"{place.elapsed_s:.0f}s")
```

Work waits for capacity rather than failing, is retried on a *different* machine if it
fails, and a machine that fails several in a row is set aside rather than draining the
queue. A unit no machine can run fails at once, naming every machine and why:

```
gpubox: has 23.0 GB VRAM, needs 80.0
pi-rack: does not report 'cuda'; has no backends
radeon: this machine is in use (mon tue wed thu fri 09:00-17:00); work resumes Mon 17:00
```

One machine at a time, when that is what you want:

```python
rtx = Peer.find_one(require="cuda")     # refuses to guess between two
rtx.push("data/train.jsonl", "data/train.jsonl")
job = rtx.submit(["python", "-m", "ml_stack.train.run", "--recipe", "text-lm",
                  "--data", "data/train.jsonl", "--out", "out/lm"])
rtx.wait(job["id"], on_metric=print)
rtx.pull("out/lm/step_000010000/model.safetensors", "local/model.safetensors")
```

Uploads and downloads resume and are verified by digest.

A model sweep goes over the fleet the same way. `ml_stack.fleet.bench` is the fleet side
of `ml-stack-bench sweep --fleet`: `plan(models, peers)` sends each model, largest first,
to the idle peer with the most room for it -- `room_bytes` is what the daemon announces,
`hub.room()` rather than free memory -- spreading models over machines rather than
stacking them, and names every model that fits nowhere and why on each peer. `jobs_from`
turns the plan into one `Job` per peer (the sweep's line with only that peer's `--serve`s),
`dispatch` posts them, `wait` polls until each is `done` or `failed` and prints the log's
tail, and `gather` brings every peer's runs home into one store as `bench:` docs with
`server["host"]` and `server["commit"]` set, never overwriting and skipping what is already
there; `import_runs(FILE.json, into, host=...)` does the same by hand from a
`show --export` file for a peer with no daemon. A daemon refuses a bench job (409, with
`refused` saying which) when its checkout's commit is not the dispatcher's, when its
`measuring.lock` is held or a bench of its own is still running, or when a model's
estimated bytes exceed its room; otherwise it runs `ml-stack-bench ... --detach` on itself
and adopts the pid into its job list, so `ml-stack-peers ls` shows it `measuring` and no
training job starts beside it. The dispatcher counts as a peer through `here()`.

## Joining the fleet

Three lines on a new machine:

```
pip install ml-stack
ml-stack-fleet join --persist
ml-stack-fleet status
```

`join` runs the checks serving depends on (the memory a model may use, a llama-server --
downloaded if there is none), asks for the passphrase every machine shares (or takes
`--passphrase WORDS`), starts the daemon, installs it at logon with `--persist`, announces on
the discovery port, and prints the peers that answered. `status` is that listing on its own:

```
NAME             URL                          ROOM             STATE        COMMIT       UPDATES      SERVING
studio           http://192.168.2.44:8770     96.0G            idle         0ce5bc5 3h   main 4m      quince-2b.gguf:8099
larch            http://192.168.2.27:8770     20.5/24.0 GB     measuring    0ce5bc5 3h   releases 2h  -
harrowgate       http://192.168.2.31:8770     20.5/24.0 GB     idle         9f2c1ab 6d   off          -
```

COMMIT is what each peer is running and how old that commit is; UPDATES is how it keeps
current and when it last looked. A fleet half on one commit and half on another is the
thing those two columns exist to make visible -- `harrowgate` above is six days behind and
following nothing, which is a machine somebody has to visit.

### Seating users

`plan` says which model each peer should serve, and with how many seats, for a number of
conversations at once:

```
ml-stack-fleet plan --users 36 --context 16384
PEER             MODEL                                            SEATS  CONTEXT     USED     ROOM
studio           Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf    31    16384   109.9G   110.0G
larch            gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf                   5    16384     5.7G    24.0G
36 of 36 user(s) seated at 16384 tokens each
--prefer quality: the best measured model that fits each peer
```

Models are taken in the order `docs/model-ranking.md` ranks them, and each goes to every
peer with room for its loaded weights and at least one seat's cache at `--context`, taking
as many seats as fit or as are still wanted; so the best model reaches the most users, and
a smaller machine serves a smaller model to the rest. A peer serves one model.

`--prefer seats` reads the other way: each peer, roomiest first, serves whichever model
seats the most of the users still waiting, and a tie goes to the better-ranked model. A
machine with room for the best model at one seat and a smaller one at six gives six people
a smaller model:

```
ml-stack-fleet plan --users 36 --context 16384 --prefer seats
```

A user who gets no seat is counted, with every peer's reason; so is a model with no memory
measurement and a memory measurement for a model nobody has scored. `--apply` serves the
plan: each placed peer's daemon runs its model with those seats (`POST /serve`), and what
each is serving afterwards is printed. `--json` is the same as data.

### Following main

A machine that is a git checkout with an editable install can follow a branch instead of
waiting for a release:

```
ml-stack-fleet join --persist --track main     # or: ml-stack-traind --track main
```

Every five minutes it asks `git ls-remote` for the head of `main`, and when it has moved:
`git pull --ff-only` (never a merge -- a checkout holding commits `main` does not have is
reported and left alone, because resetting somebody's work in progress at three in the
morning is unforgivable), `pip install -e .` only if `pyproject.toml` or a lock file moved,
and then a restart -- `launchctl kickstart` or `systemctl restart` where a login service is
installed, a re-exec where there is not. A pull that fails changes nothing, so the daemon
keeps running the code it started with and says so on the next `ml-stack-fleet status`.

Neither this nor the release update ever interrupts work: both wait for no job running, no
benchmark measuring (the same lock `ml-stack-bench status` reads, so a run started at the
keyboard counts) and no model loaded. A machine part way through a sweep is left alone
until it is not, however new the code is.

Be honest about what this is: **it runs code nobody reviewed, minutes after it is pushed**.
That is the right trade for a machine in the next room that you would otherwise have to walk
over to, and the wrong one for anything else. `--track off` goes back to releases, and it is
off unless asked for. It is remembered in the daemon's settings, so it is asked for once and
survives a reboot.

Discovery is multicast on UDP port **8771** (`239.255.77.70`, TTL 1 -- it never leaves the
segment), one above the daemon's HTTP port 8770 so one firewall rule covers both;
`$ML_STACK_DISCOVERY_PORT` moves it. Beacons are signed with the key the passphrase derives,
so a machine that does not hold it hears nothing and is heard by nobody. There is one
discovery mechanism: `ml-stack-peers ls`, the app's Cluster view and `ml-stack-fleet status`
all read the same beacons.

The app's Cluster view has the same Join button, and a "Run across the fleet" form that
builds `ml-stack-bench sweep --fleet --serve MODEL ...` from the models the peers hold,
starts it detached, and shows `status` and `history` beside it.

**For an agent**, `ml-stack-mcp` serves the same functions as MCP tools over stdio. In
Claude Code:

```
claude mcp add ml-stack -- ml-stack-mcp
```

or in a project's `.mcp.json`: `{"mcpServers": {"ml-stack": {"command": "ml-stack-mcp"}}}`.
The tools are `serve_status`, `serve_up`, `serve_down`, `models_find`, `models_files`,
`models_fetch`, `bench_run`, `bench_status`, `bench_history`, `bench_show`, `fleet_peers`,
`fleet_join`, `world_make`, `setup_look` and `doctor`; a model load, a download and a
measurement never block the call -- each returns a log path and a pid, and `bench_status`
follows it. With `pip install 'ml-stack[mcp]'` the SDK's server is used; without it the
command speaks the protocol itself.

