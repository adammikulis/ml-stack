# ml-stack

Primitives for running local AI models: serving them, talking to them, converting them,
and training them.

The packages are small and separately installable, so a project takes only what it needs.
A voice assistant on a single-board computer should not have to install PyTorch to say a
sentence, and a training script should not have to reimplement checkpoint rotation.

## Packages

| Package | What it is |
|---|---|
| `ml-stack-contracts` | Reader for `contracts/`; the RAM→model ladder and the fitting rule |
| `ml-stack-media` | WAV containers, image format sniffing, resumable asset download |
| `ml-stack-client` | HTTP client: chat, completion, embeddings, health, token estimate |
| `ml-stack-fleet` | Find the other boxes, run jobs on them, move files between them |
| `ml-stack-serve` | Start, adopt and tear down a model server |
| `ml-stack-gguf` | Converter/quantiser discovery, export, tokenizer-metadata repair |
| `ml-stack-speech` | ASR / TTS / VAD behind three protocols and one resolver |
| `ml-stack-vision` | Image payloads, and a gate that verifies a model can see |
| `ml-stack-backend` | One array API over MLX and PyTorch, so math is written once |
| `ml-stack-graph` | Graphs as tensors: message passing, DAG sweeps, topology |
| `ml-stack-train` | Atomic checkpoints, schedules, guards, metrics, leak-safe splits, tokenizer fertility |
| `ml-stack-testing` | Cross-backend numerical parity harness |

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
working on. `ml-stack-fleet` is how you reach it. It has no dependencies.

Run this on every machine, and type the same passphrase into each:

```
ml-stack-peers setup
```

That is the whole join story. Machines that derived their key from the same words find
each other; machines that did not are invisible to each other, so several groups share a
network without any of them being configured to. There is no key to copy and no address
to write down anywhere.

Anyone who knows the passphrase can run commands on every machine in the group, so it is
closer to the password to your house than to a wifi password. It is stretched with scrypt
before it becomes a key, because everyone on the network can hear the beacons and grind
guesses against them offline.

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
