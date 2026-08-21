# mainspring

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
| `mainspring-contracts` | device | Reader for `contracts/`; the RAM→model ladder and the fitting rule |
| `mainspring-media` | device | WAV containers, image format sniffing, resumable asset download |
| `mainspring-client` | device | HTTP client: chat, completion, embeddings, health, token estimate |
| `mainspring-serve` | host | Start, adopt and tear down a model server |
| `mainspring-gguf` | host | Converter/quantiser discovery, export, tokenizer-metadata repair |
| `mainspring-speech` | host | *(planned)* ASR / TTS / VAD behind one resolver |
| `mainspring-vision` | host | *(planned)* VLM, OCR, image generation |
| `mainspring-backend` | lab | One array API over MLX and PyTorch, so math is written once |
| `mainspring-graph` | lab | Graphs as tensors: message passing, DAG sweeps, topology |
| `mainspring-train` | lab | Atomic checkpoints, schedules, guards, metrics, leak-safe splits |
| `mainspring-testing` | lab | Cross-backend numerical parity harness |

## Using it

```python
from mainspring.serve import serve
from mainspring.client import Client

with serve("model.gguf", port=8899) as server:
    client = Client(server.base_url)
    client.assert_grammar_support()          # fail now if constrained decoding is broken
    print(client.chat([{"role": "user", "content": "hello"}]).content)
```

`serve` adopts a healthy server that is already running rather than starting a second one,
and leaves an adopted server alone on exit. It only stops what it started.

```python
from mainspring.contracts import largest_that_fits
import psutil

tier = largest_that_fits(psutil.virtual_memory().total)
print(tier.id, tier.gguf_repo, tier.context)
```

Write model math once, against the array protocol, and run it on either framework:

```python
from mainspring.backend import get_backend

def rms_norm(backend, x, weight, eps=1e-6):
    ops = backend.ops
    scale = ops.rsqrt(ops.mean(x * x, axis=-1, keepdims=True) + eps)
    return x * scale * weight

rms_norm(get_backend("mlx"), x, w)      # same function
rms_norm(get_backend("torch"), x, w)    # same numbers
```

`mainspring.testing` proves the two agree, forward and backward:

```python
from mainspring.testing import needs_both, run_pair

@needs_both
def test_layer_matches():
    run_pair(build_torch, build_mlx, forward_torch, forward_mlx, (6, 8))
```

## `contracts/` is data, not code

`contracts/` holds JSON describing things a runtime and a non-Python host both need to
agree on: the RAM→model tier ladder, the sampler surface, GBNF grammars. It contains no
code, so a native or scripting host can read it directly.

There is exactly one copy on disk. `mainspring-contracts` pulls it into its wheel at build time,
so there is no synced duplicate in the source tree to drift.

Resolution order at runtime: `$MAINSPRING_CONTRACTS` → the copy inside the installed wheel → a
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
- **Graph algorithms are networkx's job.** `mainspring-graph` keeps a graph in the array
  backend so it can go through a model; for shortest path or components, call
  `Graph.to_networkx()`.

## Testing

```
python -m pytest tests/ -q
```

Nothing here mocks the transport. Every client test runs a real `http.server` on a real
socket, because the failures these modules exist to prevent are transport-shaped: a server
that answers `/health` while still loading, one that ignores a `Range` header, one that
returns 500 on a concurrent request. A mocked `urlopen` reproduces none of them.

## Licence

Apache License 2.0. See [LICENSE](LICENSE).
