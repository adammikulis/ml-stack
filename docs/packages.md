# Packages

| Module | What it is |
|---|---|
| `poolhouse.contracts` | Reader for `contracts/`; the RAM→model ladder and the fitting rule |
| `poolhouse.messages` | One thing somebody said, and the product ids and timestamps that go with it |
| `poolhouse.media` | WAV containers, image format sniffing, resumable asset download |
| `poolhouse.http` | One HTTP client: JSON, bytes, streams, resumed downloads, retries |
| `poolhouse.client` | Talking to a model server: chat, completion, embeddings, health, token estimate |
| `poolhouse.fleet` | Find the other boxes, run jobs on them, move files between them |
| `poolhouse.serve` | Start, adopt and tear down a model server |
| `poolhouse.gguf` | Converter/quantiser discovery, export, tokenizer-metadata repair |
| `poolhouse.speech` | ASR / TTS / VAD behind three protocols and one resolver |
| `poolhouse.vision` | Image payloads, and a gate that verifies a model can see |
| `poolhouse.graph` | A graph: stored, searched, asked about, drawn — and as tensors: a converter, message passing, spatial topology, DAG sweeps |
| `poolhouse.bench` | Timing and scoring a model's answers, and comparing what served them |
| `poolhouse.entities` | Resolving names, planning edits, spelling, paths through a graph |
| `poolhouse.scrape` | Reading a site you are signed in to, with presets to start from |
| `poolhouse.sources` | A PDF, a Slack export, an mbox or scraper rows into one document and message shape |
| `poolhouse.ingest` | Documents read into a store section by section, with the page behind every claim |
| `poolhouse.world` | An invented organisation that talks, and questions about it with known answers |
| `poolhouse.web` | Search, read and screenshot the web as tools, refused against private addresses |
| `poolhouse.redact` | Reading a file for a real person's details — the hook's reader and `poolhouse-audit` |
| `poolhouse.train` | Atomic checkpoints, schedules, guards, metrics, leak-safe splits, tokenizer fertility |
| `poolhouse.backend` | One array API over MLX and PyTorch, so math is written once |
| `poolhouse.testing` | Cross-backend numerical parity harness, behind `poolhouse-train-run parity`, and the fakes the suite shares |

`poolhouse.serve` and `poolhouse.client` can be used alone from another application:
[embedding.md](embedding.md).

Everything above ships in one package. Its dependencies are `packaging` and `psutil`; the
extras carry what a module needs beyond them: `[claude] [train] [train-lora] [gguf] [graph] [store]
[scrape] [web] [hub] [vision] [pdf] [viz] [plot] [testing] [telemetry] [mcp] [standard]
[privacy] [arrays] [test]`, plus `[torch]` and `[mlx]`, and `[all]`. `[all]` includes
the standard PyPI runtime extras. `[wsl-dev]` adds the compatible optional runtime and test packages
for WSL. The setup script installs MetaDrive from its documented pinned source revision. The mutually exclusive `[viz]`
and `[gym-warehouse]` extras, `[privacy-transformers]` and `[train-lora]`, `[mlx]` (Apple silicon
only), and `[pdf-agpl]` (an explicit AGPL engine selection) stay separate.
