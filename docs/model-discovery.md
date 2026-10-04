# Models on this machine

`ml_stack.hub` finds every model installed by Hugging Face, llama.cpp, LM Studio, Ollama and
the other tools below, downloads others with progress and resume, and says what serving one
costs in memory.

```python
from ml_stack import hub
from ml_stack.serve import estimate, suggest

for m in hub.discover():                              # id, name, path, format, size_bytes, ...
    print(m.id, m.size_bytes, m.quantization, m.source)
machine = hub.machine_memory()                        # RAM, unified-memory limit, VRAM
for r in hub.search("qwen3 instruct", hub.Filters(max_bytes=8_000_000_000, quant="Q4_K_M")):
    print(r.id, r.builds())                           # (build, bytes, shards, quant) per build
path = hub.pull("hf:owner/repo:Q4_K_M", on_progress=lambda p: print(f"{p.fraction:.0%}"))
pick = suggest.suggest(path, machine, goal="agent")   # context, kv type, offload, reasons[]
print(pick.verdict, pick.context, pick.lease())       # lease() are ServerSpec keywords
print(estimate.estimate(path, context=16384).breakdown)  # bytes per part, for a meter
```

`ml-stack-models` (or `ml-stack models`) has the same as commands, each with `--json`:
`list`, `where`, `info`, `which`, `fit`, `suggest`, `pull`, `search`, `recommend`, `machine`.

## Where models are searched

`hub.places.places()` is the table; `ml-stack-models where` prints it for this machine with
what each folder holds. An environment variable that moves a tool's folder replaces that
tool's default folders, as it does for the tool. A folder reached twice is read once.

| Tool | Folder (macOS / Linux / Windows) | Moved by | Verified |
| --- | --- | --- | --- |
| ml-stack store | `<state>/models` | `ML_STACK_HOME` | yes |
| Hugging Face hub | `~/.cache/huggingface/hub` (all three) | `HF_HUB_CACHE`, `HUGGINGFACE_HUB_CACHE`, `HF_HOME`/hub, `TRANSFORMERS_CACHE`, `XDG_CACHE_HOME`/huggingface/hub | macOS |
| llama.cpp (`llama-server -hf`) | `~/Library/Caches/llama.cpp` / `$XDG_CACHE_HOME/llama.cpp` or `~/.cache/llama.cpp` / `%LOCALAPPDATA%\llama.cpp` | `LLAMA_CACHE` | macOS |
| Ollama | `~/.ollama/models` (Linux service: `/usr/share/ollama/.ollama/models`) | `OLLAMA_MODELS` | macOS |
| LM Studio | `~/.lmstudio/models`, older `~/.cache/lm-studio/models`, and `downloadsFolder` in its `settings.json` | | no |
| GPT4All | `~/Library/Application Support/nomic.ai/GPT4All` / `~/.local/share/nomic.ai/GPT4All` / `%LOCALAPPDATA%\nomic.ai\GPT4All` | | no |
| Jan | `~/jan/models`, `.../Jan/data/models`, `.../Jan/data/llamacpp/models` under Application Support / `~/.config` / `%APPDATA%` | | no |
| ModelScope | `~/.cache/modelscope/hub` | `MODELSCOPE_CACHE` | no |
| KaggleHub | `~/.cache/kagglehub/models` | `KAGGLEHUB_CACHE` | no |
| Manual | `~/models`, `~/Models`, `/opt/models` (`/srv/models` on Linux, `%LOCALAPPDATA%\models` on Windows), `~/Downloads` one level deep | | yes |
| Extra | each folder in `ML_STACK_MODEL_PATHS` (path-separator list), any layout | | yes |
| Volumes | `/Volumes/*/models` and `/Volumes/*/Models` on macOS, only with `ML_STACK_SCAN_VOLUMES=1` | | no |

"Verified" means the layout was read off a real install on the machine named; the other
rows follow each tool's documentation and have not been run against an install. Msty,
koboldcpp, text-generation-webui and Open WebUI keep models in a folder the person chose or
(Open WebUI) in Ollama; put that folder in `ML_STACK_MODEL_PATHS`. Torch Hub holds no
language models and is not searched.

Layouts read:

- **Hugging Face**: `models--owner--name/snapshots/<rev>/...`, each file a symlink into
  `blobs/`. The link is the model's path (llama.cpp finds a sharded model's other shards from
  that name). A folder with `config.json` and `*.safetensors` is one model, `mlx` when the
  repository is under `mlx-community`, has `mlx` in its name, or its config has a
  `quantization` key.
- **Ollama**: `manifests/<registry>/<namespace>/<model>/<tag>` names an
  `application/vnd.ollama.image.model` layer in `blobs/sha256-<hex>`, a plain GGUF that
  llama.cpp loads by path; `...image.projector` is its vision projector. The name is
  `model:tag`, with the registry and namespace shown only when they are not
  `registry.ollama.ai/library`. A blob smaller than its manifest says is incomplete.
  Manifests of per-tensor layers (`...image.tensor`, Ollama's MLX models) are not GGUF and
  are not listed.
- **llama.cpp**: flat files beside a `<file>.etag` sidecar. The build checked here stores
  files without the repository in the name, so an `hf:` reference cannot be matched to them;
  a `<file>.json` sidecar, which older builds wrote with the download URL, supplies the
  repository when present.
- **LM Studio**: `<publisher>/<repo>/<file>.gguf`; the first two parts are the repository id.

Symlinks are followed only when the target is under the folder holding the link or under
another searched folder; a link out of every searched folder is skipped. Files still arriving
(`.part`, `.downloading`, `.incomplete`, `.crdownload`) mark the model incomplete, and so does
a missing shard, a short Ollama blob or a file without the GGUF magic.

## What `discover()` returns

`discover(roots=None, formats=("gguf", "safetensors", "mlx"), include=None, *, companions=False,
refresh=False) -> list[ModelInfo]`. `ModelInfo` has `id` (an `hf:owner/repo/file` reference
where the repository is known, `ollama:name:tag`, else `file:<name>`), `name`, `path`, `format`,
`size_bytes` (all shards), `quantization`, `parameters`, `architecture`, `context_length`,
`mmproj`, `source`, `repo`, `mtime`, `is_complete`, `shards`, `verified`, `copies` and `files` (the exact discovered members). A model
installed twice is one row: complete before incomplete, then ml-stack store, extra folders,
Hugging Face, llama.cpp, LM Studio, Jan, GPT4All, Ollama, then the rest, then newest; the other
paths are in `copies`. Copies are the same when size, architecture and name in the header match;
nothing is hashed. Header facts are read from the front of each file and kept under
`<cache>/models/headers.json` by path, size and modification time. 93 models on the machine of
the table below: 0.13 s cold, 0.03 s warm.

### Kinds of model

Every row carries a `kind`, one of `hub.KINDS` (defined once, in `hub/kinds.py`): `chat`,
`embedding`, `vision`, `speech`, `decision`. `discover(kind=...)` and
`ml-stack-models list --kind decision` filter on it, and the listing has a KIND column.
The label comes from the header's architecture (`bert` embeds, `whisper` hears), a vision
projector beside the weights, and, for `decision`, either an entry in the decider registry
(`decide/registry.py`: the model sits in a registered decider's directory or is its local
base) or the words `decision` / `decider` in the model's name or repository (the Strands
decision models). Only scalar header keys are read, so `general.tags` arrays are not consulted.
The decide package asks the same library (`decide/library.py`) for `kind="decision"` models
when a named decider is not a directory or a registry name. The fleet catalogue's suggestions
carry the same label.

A kind is a label and nothing else: a `decision` model is still loaded only through a Broker
lease, and admission and the memory fit treat it like any other model of its size.

Serving reuses what is installed: `ServerManager.lease` replaces an `hf:owner/repo/file`
model, projector or draft that `installed_for` finds with the file, and `hub.located` resolves a
name such as `llama3:latest` to its Ollama blob.

## Pulling

`pull(ref, dest=None, on_progress=None, cancel=None) -> Path` downloads `hf:owner/repo/file.gguf`
(every shard of that build), `hf:owner/repo:Q4_K_M`, or `hf:owner/repo` (the `Q4_K_M` build, else
the largest under 85% of this machine's room), into `dest` or `<state>/models/owner/repo`.
`on_progress` gets a `Progress` (`phase` of `downloading`, `verifying` or `done`, byte counts,
`bytes_per_second`, `fraction`); a `CancelToken` stops it and the partial `<file>.part` is
continued by the next pull with an HTTP range request. Two processes pulling one file take
turns. A finished file is checked against the sha256 the endpoint lists; the free space on the
destination is checked first (`NotEnoughSpace`). A gated or private repository raises
`GatedRepo` saying to accept the licence and set `HF_TOKEN`. `HF_ENDPOINT` moves the endpoint; the
token goes to the endpoint and not to the CDN host it redirects to.

### Peers first

By default a pull asks the devices you paired before it goes to the Hub, on the local network
or at an address you stored (a VPN or overlay such as a tailnet; never a public one). The
stored devices are the peer book (`<state>/onboard/peers.json`, the `PeerBook` of
`ml_stack.hub.peerbook`). **Pairing fills it**: when two devices pair (`fleet listen` / `fleet
pair`), each keeps the other's share address (the address the pairing used and the share port,
`--share-port`, default 8773), pinned certificate, owner signing key and request key, learned from
the grant and from an offer sealed under the exchange, so it is as authentic as the pairing; a
device started with `--no-share` has no share address and no row is made for it. Revoking a
device removes its rows. `ml-stack fleet peers add NAME --url https://host:port --certificate ...
--signing-key ... [--device-secret ...]` stays as the manual path for a device paired some other
way, with `remove`, `on` and `off`. A row that came from pairing names its device, and its route
is chosen when the pull starts: the address on this network, else its tailnet address
(`fleet/onboard/routes.py`), each answering with the pinned certificate; a revoked device is not
asked. For each file of the pull the session asks every peer for its signed
manifest (over TLS pinned to that device's certificate), keeps the peers whose manifest lists
that very file (name, size and digest), orders them by how fast they answered, and fetches the
file chunk by chunk, resumably, from several at once. `on_progress`, `CancelToken` and the
sentinel events work as for a Hub pull.

**What is trusted.** Not the peer. The digest a file must have is the Hub's own listing
(fetched through `ml_stack.net`); a peer whose manifest lists the file with another digest is
not used, and a digest a peer states never replaces the pin. Only when the listing carries no
digest does an entry of a manifest signed by your own pinned manifest key supply it. Every
chunk is checked against the signed manifest, the whole file against the pin, and the result
goes through the same format check, scan and quarantine as an internet download. A peer that
sends bytes failing a digest is dropped for the rest of the pull, the partial bytes are deleted,
and a critical `onboard.peer.bad_copy` event goes to the bus and to sentinel; the Hub is used.

**Who may have what** is decided by the device that serves the file, not by the one asking:
`never` (the licence forbids copies) is refused, `owner` (gated or licensed models) goes only to
a device you marked as yours and only after you recorded accepting that licence on the serving
device (who, when, which licence; keep it with `ml-stack fleet share`), `open` goes to any
paired device. A copy that sentinel holds in quarantine on the serving device is not served.
Credentials such as `HF_TOKEN` never leave a device. A peer that has no copy, refuses or is
unreachable costs a few seconds at most (each answer has 4 s) and the pull continues from the Hub.

**Speed.** Each peer's rate is measured (bytes completed per second, kept in the peer book and
blended with earlier transfers); peers are ranked by it, unmeasured ones first so that they get
measured. A peer gets at most two requests at a time, and an optional bandwidth cap:
`ml-stack fleet peers limit [NAME] [--rate 20MiB/s] [--streams N] [--metered on|off]` (no NAME
means every peer; `--rate off` and `--streams 0` remove a limit; `--metered on` means the link to
that device costs per byte, so it is not asked and the Hub is used). When the transfer as a whole
stays under 2 MiB/s for a 15 s window it is given up, the partial bytes are kept for a resume and
the Hub is used.

**Serving the model store.** `ml-stack fleet share --models [--models-dir DIR ...]` serves the
GGUF and safetensors files `ml-stack-models list` shows (or only those under the `--models-dir`
folders), by file name with the repository, size and sha256, under the sharing levels above.
Nothing outside the model roots is served: each path is resolved at every request and a symlink
or `..` that leaves the roots is refused; a copy sentinel holds is neither listed nor served. A
model is `owner` unless `--sharing NAME=open` (or `REPO=open`) says it is not gated, so by default
it goes only to your own devices and only once you accepted its licence on this machine (the
licence is the repository unless `--licence` says otherwise; you confirm it once at a terminal).
The manifest is signed with the owner key and its serial only rises.

**What the downloading device keeps.** An append-only local record
(`<state>/onboard/peer-downloads.jsonl`, `ml_stack.hub.origins`): when, which file (name, size,
sha256), which peer, and for an `owner` model the acceptance it relied on (the licence, who
accepted it and when, as the serving device reported it). `ml-stack-models list` shows it in the
FROM column and in `--json` as `from_peer`.

**Turning it off.** `ml-stack-models pull --no-peers` or `fetch --no-peers` (or
`pull(..., peers=False)`) for one pull; `ML_STACK_NO_PEERS=1` for a shell; `ml-stack fleet peers off`
for the machine. Nothing is contacted when it is off.

`search(query, Filters(max_bytes, quant, owner, gated, limit, files))` returns `Repo` rows with
their files; `Repo.builds()` is `(build, bytes, shards, quantization)` per build. Tests run
against `ml_stack.testing.fakehub`.

## Memory estimates

`serve.estimate.estimate(model, setup=None, **changes)` takes the fields of `Setup` as keywords
(`context=4096, parallel=1, n_gpu_layers="auto", kv_cache_type="q8_0", flash_attn=None, batch=512,
mmproj=None, draft=None`) and returns an `Estimate` (`weights_bytes`, `kv_cache_bytes`, `compute_buffer_bytes`, `mmproj_bytes`,
`draft_bytes`, `state_bytes`, `total_bytes`, `gpu_bytes`, `cpu_bytes`, `breakdown`, `confidence`,
`notes`) from the header and file sizes. `context` is per slot. `verdict(estimate, machine,
reserve_bytes=None)` is `green` below 80% of the memory the estimate lands in, `yellow` from 80% to
95%, `red` from 95% or when it does not fit, and `none` when the machine is unknown; the memory is the
unified working-set limit or available RAM less the reserve (the larger of 2 GiB and a tenth of RAM), or
a card's free VRAM for the GPU part, so 95% of it is the real limit. The two numbers are the ones
`ml_stack.ui.verdict` uses. `meters(estimate, machine)` gives one `Meter` per pool with `segments`
(`[{label, value}]` in bytes), `capacity_bytes` and `verdict`, which `<ml-meter>` takes as `segments`,
`capacity` and `verdict`. `max_context(model, machine, setup, max_verdict="yellow")` is the longest
context in steps of 256.

The KV cache is `q8_0` by default: 34 bytes per 32 values, 53% of f16, near-lossless for chat and
tool use. `q4_0` saves more and costs quality, so it is a last resort. llama.cpp accepts a
quantised V cache only with flash attention; where the head size or the build lacks it, V stays f16
and K stays q8_0 (`"q8_0/f16"`).

`serve.suggest.suggest(model, machine=None, goal="agent", max_verdict="green")` picks context
(`agent` up to 32k and one slot, `chat` 16k, `long-context` the largest, `fast` 4k; never past the
trained context), offload, cache types and batch (2048 for `agent` and `long-context` when it keeps
the rating), with `reasons` and a smaller and a bigger `alternatives`. `lease()` gives `ServerSpec`
keywords. `suggest_model(candidates, machine, goal)` ranks models: rating first, then parameters
times a quantisation quality factor (a mixture of experts counts half); `fast` takes the smaller of
the models over 1.5 billion parameters. No benchmark scores are used. `recommend(machine, goal,
query)` ranks installed models and, with `query`, what the Hub offers.

### Measured accuracy

2026-10-02, Apple M4 Max 128 GB, llama.cpp 0.3.0 (build 10621, c1d0e7a), flash attention on, models
from the Hugging Face cache. Each row serves the model with `ml_stack.serve.measuring.measure` and
compares `estimate()` with the file size plus the KV, recurrent and compute buffers in the load log
(`tests/fixtures/estimate_logs.json` holds the logs and headers; `tests/test_serve_estimate.py` checks
them).

| Model | Context / KV | Log total | Estimate | Error |
| --- | --- | --- | --- | --- |
| Qwen3-4B-Instruct-2507 Q4_K_M | 2k / q8_0 | 2623 MiB | 2620 | -0.1% |
| | 8k / q8_0 | 3094 | 3107 | +0.4% |
| | 8k / f16 | 3647 | 3647 | 0.0% |
| | 32k / q8_0 | 5073 | 5056 | -0.3% |
| Qwen3-VL-2B-Instruct Q8_0 | 2k / q8_0 | 1956 | 1941 | -0.7% |
| | 32k / q8_0 | 3910 | 3867 | -1.1% |
| Qwen3.5-0.8B Q4_K_M (hybrid) | 2k / q8_0 | 588 | 589 | +0.1% |
| | 32k / q8_0 | 897 | 921 | +2.6% |
| gemma-4-E2B QAT UD-Q4_K_XL | 2k / q8_0 | 2639 | 2596 | -1.6% |
| | 32k / q8_0 | 2819 | 2832 | +0.5% |

Process resident memory for Qwen3-4B (`psutil`, after load): 2628, 3137 and 4967 MiB at 2k, 8k and 32k
against estimates of 2620, 3107 and 5056 (-0.3%, -1.0%, +1.8%). The KV cache and the recurrent state
match the log to the byte; the compute buffer is a fit and is off by up to 40 MiB. The vision projector
is estimated at 1.5 times its file (one measurement: 784 MiB file, 1165 MiB reported by llama.cpp).

Not measured, so treat as approximate (`confidence` says `approx`): flash attention off (the score
matrix is added from the formula), partial offload, a draft model, CUDA, ROCm and Windows, and a
batch other than 512 beyond one 4B run (`-ub 2048`: 404 MiB in the log, 456 predicted). On unified memory
a partial offload does not save memory, because the whole file is mapped; on the CPU side llama.cpp
repacks weights, which takes up to one more copy of those layers.

The Fleet Models page groups installed builds under their maintained model-family labels.
A sharded GGUF build appears once, with the combined weight size and completeness status;
its individual filenames appear in **Files and model details**. Search, family, format,
quantization, capability and status filters combine, with name, size and recent-installation
sorting. Projectors and draft heads are companions rather than independently runnable
models. Incomplete builds and unsupported formats remain visible with serving disabled.

`GET /ui/models` includes a `library` of grouped discovery entries alongside the existing
file-level `here` rows used for network copies. Starting a library model submits its exact
canonical path to `POST /ui/serving`; the daemon rechecks that it is a complete supported
primary entry within its configured model roots. A display label cannot select a different
format or build. Every expected GGUF shard number, including shard 1, must be present.
