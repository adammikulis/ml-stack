# ml-stack

Primitives for running local AI models: serving them, talking to them, converting them,
and training them.

The packages are small and separately installable, so a project takes only what it needs.
A voice assistant on a single-board computer should not have to install PyTorch to say a
sentence, and a training script should not have to reimplement checkpoint rotation.

## Tiers

Packages are organised by **where the code runs**, not by what it is about. This is the
constraint that shapes everything else.

| Tier | Runs on | May depend on |
|---|---|---|
| **device** | embedded boards, phones, host applications | **standard library only** |
| **host** | a desktop that serves models | + psutil, hf_hub, httpx |
| **lab** | a desktop that trains | + mlx, torch, transformers, datasets |

A package may import its own tier and any lower one, never a higher one.

The rule is enforced, not documented. `scripts/check_tiers.py --live` imports every device
package in a subprocess with site-packages stripped from `sys.path`, which is the only
check a lazy `import torch` inside a function body cannot slip past.

```
python scripts/check_tiers.py --live
```

## Packages

| Package | Tier | What it is |
|---|---|---|
| `ml-stack-contracts` | device | Reader for `contracts/`; the RAM→model ladder and the fitting rule |
| `ml-stack-media` | device | WAV containers, image format sniffing, resumable asset download |
| `ml-stack-client` | device | HTTP client: chat, completion, embeddings, health, token estimate |
| `ml-stack-fleet` | device | Find the other boxes, run jobs on them, move files between them |
| `ml-stack-serve` | host | Start, adopt and tear down a model server |
| `ml-stack-gguf` | host | Converter/quantiser discovery, export, tokenizer-metadata repair |
| `ml-stack-speech` | host | ASR / TTS / VAD behind three protocols and one resolver |
| `ml-stack-vision` | host | Image payloads, and a gate that verifies a model can see |
| `ml-stack-backend` | lab | One array API over MLX and PyTorch, so math is written once |
| `ml-stack-graph` | lab | Graphs as tensors: message passing, DAG sweeps, topology |
| `ml-stack-train` | lab | Atomic checkpoints, schedules, guards, metrics, leak-safe splits, tokenizer fertility |
| `ml-stack-testing` | lab | Cross-backend numerical parity harness |

## Using it

```python
from ml_stack.serve import serve
from ml_stack.client import Client

with serve("model.gguf", port=8899) as server:
    client = Client(server.base_url)
    client.assert_grammar_support()          # fail now if constrained decoding is broken
    print(client.chat([{"role": "user", "content": "hello"}]).content)
```

`serve` adopts a healthy server that is already running rather than starting a second one,
and leaves an adopted server alone on exit. It only stops what it started.

## The cluster

Training happens on whatever box has the card, which is usually not the one you are
working on. `ml-stack-fleet` is how you reach it, and it is standard library only --
the laptop you drive the fleet from needs no CUDA, no MLX and no training stack.

Mint a key once, and copy it to every machine that should join:

```
ml-stack-peers init          # prints the one-line command to run elsewhere
```

Then run the daemon on each box, telling it what it is:

```
ml-stack-traind                                            # a box with one card
ml-stack-traind --slots 8                                  # a box that preps data
ml-stack-traind --report ml_stack.train.accelerator:report  # ...and can see its GPU
```

```
$ ml-stack-peers ls
NAME             URL                          FREE    STATE      DEVICE
gpubox           http://192.168.2.27:8770     1/1     idle       RTX 4090  23.1/24.0 GB free
prepbox          http://192.168.2.31:8770     6/8     busy +2    16 cpu
```

A peer you can find is a peer you can already drive: the bearer token is derived from
the cluster key rather than transmitted, so discovery and authentication are one step.

```python
from ml_stack.fleet import Peer

rtx = Peer.find_one(require="cuda")
rtx.push("data/packed/train.npy", "data/train.npy")
job = rtx.submit(["python", "-m", "train.run", "--steps", "30000"])
rtx.wait(job["id"], on_metric=print)
rtx.pull(f"jobs/{job['id']}/ckpt/best/model.safetensors", "local/model.safetensors")
```

```python
from ml_stack.contracts import largest_that_fits
import psutil

tier = largest_that_fits(psutil.virtual_memory().total)
print(tier.id, tier.gguf_repo, tier.context)
```

Write model math once, against the array protocol, and run it on either framework:

```python
from ml_stack.backend import get_backend

def rms_norm(backend, x, weight, eps=1e-6):
    ops = backend.ops
    scale = ops.rsqrt(ops.mean(x * x, axis=-1, keepdims=True) + eps)
    return x * scale * weight

rms_norm(get_backend("mlx"), x, w)      # same function
rms_norm(get_backend("torch"), x, w)    # same numbers
```

`ml_stack.testing` proves the two agree, forward and backward:

```python
from ml_stack.testing import needs_both, run_pair

@needs_both
def test_layer_matches():
    run_pair(build_torch, build_mlx, forward_torch, forward_mlx, (6, 8))
```

## `contracts/` is data, not code

`contracts/` holds JSON describing things a runtime and a non-Python host both need to
agree on: the RAM→model tier ladder, the sampler surface, GBNF grammars. It contains no
code, so a native or scripting host can read it directly.

There is exactly one copy on disk. `ml-stack-contracts` pulls it into its wheel at build time,
so there is no synced duplicate in the source tree to drift.

Resolution order at runtime: `$ML_STACK_CONTRACTS` → the copy inside the installed wheel → a
`contracts/` found by walking up from the source file. The walk-up is last on purpose: if a
wheel is installed *and* a repo happens to be an ancestor, the wheel's own data should win,
because that is what its version was tested against.

## Design notes

A few decisions that are easy to reverse by accident:

- **Never demote on ignorance.** If a model's size cannot be measured, `fits()` returns
  `"unknown"`, never `"too_big"`. A missing measurement is not evidence.
- **`wait_for_health` takes an `is_alive` callable**, not a process handle. That keeps the
  process-death watch available to the device tier without it importing `subprocess`, and
  means a bad model path fails in under a second instead of timing out.
- **Adopt, but prove the model matches.** "Something answers on this port" and "the thing
  on this port serves the model I asked for" are different facts.
- **`embed()` raises on a dimension mismatch** rather than padding to fit. Silently
  coercing a vector turns a model mismatch into a store of garbage that still
  cosine-compares.
- **Verify before the atomic rename.** A download lands in `.part` and only moves into
  place once its size and digest check out. A checkpoint is built in `.partial`, and the
  state file that makes a directory *count* as a checkpoint is written **last** — so a
  half-written save is ignored rather than resumed from.
- **A resume restores everything or fails.** Weights without optimizer state is a warm
  restart, not a resume, and it shows up as a loss spike that gets blamed on the LR.
- **Schedules are plain functions returning floats.** A framework schedule object captured
  by a compiled function freezes the learning rate for the rest of the run, silently.
- **Graph algorithms are networkx's job.** `ml-stack-graph` keeps a graph in the array
  backend so it can go through a model; for shortest path or components, call
  `Graph.to_networkx()`.
- **Constructing a provider proves nothing.** Speech auto-detection *starts* each
  candidate, because the weights load in `start()` — which is where a missing model or a
  wheel built for another architecture actually shows up.
- **One job per slot, and slots default to one.** A GPU is not shareable in any useful
  sense: two jobs on one card contend for memory and both get slower, and the failure is
  silent. A box tokenizing text has no such contention and says so with `--slots`. The
  default is the safe one, so raising it is a decision someone made rather than one they
  inherited.
- **A beacon describes right now, not boot time.** `busy`, `queued` and free VRAM are
  refreshed on the way out. A daemon that advertised the state it started with would have
  every scheduler reading a constant.
- **A device-tier package cannot see a GPU, and must not guess.** The daemon reports what
  the standard library knows; a lab-tier probe passed with `--report` adds the card. An
  inferred "no accelerator here" would be read as "not a training box", so an unverified
  negative would park every run on the wrong machine.
- **`find_one` refuses to pick between two matching peers.** Two GPU boxes on one LAN and
  a coin-flip between them is how a run ends up on the wrong card with nobody noticing.
  Fanning out is a different entry point, not a relaxation of this one.
- **Model servers stay on loopback.** `DEFAULT_HOST` is `127.0.0.1` because a model server
  on `0.0.0.0` is an unauthenticated inference endpoint on every network the machine is
  attached to. Reaching one across the LAN goes through the daemon, which already demands
  a token.
- **Verify a vision model can see.** A model served without its projector doesn't error;
  it describes the picture from the prompt, fluently. `VisionGate` shows it a known image
  first, using a palette that isn't primary colours precisely because those are what a
  blind model guesses.

## Testing

```
python -m pytest tests/ -q
```

Nothing here mocks the transport. Every client test runs a real `http.server` on a real
socket, because the failures these modules exist to prevent are transport-shaped: a server
that answers `/health` while still loading, one that ignores a `Range` header, one that
returns 500 on a concurrent request. A mocked `urlopen` reproduces none of them.

The fleet tests go further: they boot real `ml-stack-traind` subprocesses and speak real
UDP on a real interface, on randomised ports so a run never answers -- or gets answered
by -- a daemon you actually have running on your LAN. A forged beacon, a replayed reply,
a multicast group a router quietly drops: a fake socket reproduces none of those either.

## Licence

Apache License 2.0. See [LICENSE](LICENSE).
