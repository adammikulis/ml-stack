# Packages

| Module | What it is |
|---|---|
| `ml_stack.contracts` | Reader for `contracts/`; the RAM→model ladder and the fitting rule |
| `ml_stack.media` | WAV containers, image format sniffing, resumable asset download |
| `ml_stack.http` | One HTTP client: JSON, bytes, streams, resumed downloads, retries |
| `ml_stack.client` | Talking to a model server: chat, completion, embeddings, health, token estimate |
| `ml_stack.fleet` | Find the other boxes, run jobs on them, move files between them |
| `ml_stack.serve` | Start, adopt and tear down a model server |
| `ml_stack.gguf` | Converter/quantiser discovery, export, tokenizer-metadata repair |
| `ml_stack.speech` | ASR / TTS / VAD behind three protocols and one resolver |
| `ml_stack.vision` | Image payloads, and a gate that verifies a model can see |
| `ml_stack.graph` | A graph: stored, searched, asked about, drawn — and as tensors: a converter, message passing, spatial topology, DAG sweeps |
| `ml_stack.bench` | Timing and scoring a model's answers, and comparing what served them |
| `ml_stack.entities` | Resolving names, planning edits, spelling, paths through a graph |
| `ml_stack.scrape` | Reading a site you are signed in to, with presets to start from |
| `ml_stack.sources` | A PDF, a Slack export, an mbox or scraper rows into one document and message shape |
| `ml_stack.ingest` | Documents read into a store section by section, with the page behind every claim |
| `ml_stack.world` | An invented organisation that talks, and questions about it with known answers |
| `ml_stack.web` | Search, read and screenshot the web as tools, refused against private addresses |
| `ml_stack.redact` | Reading a file for a real person's details — the hook's reader and `ml-stack-audit` |
| `ml_stack.train` | Atomic checkpoints, schedules, guards, metrics, leak-safe splits, tokenizer fertility |
| `ml_stack.train.backend` | One array API over MLX and PyTorch, so math is written once |
| `ml_stack.testing` | Cross-backend numerical parity harness, behind `ml-stack-train-run parity`, and the fakes the suite shares |

Everything above ships in one package. The extras carry what a module needs beyond the
standard library: `[claude] [train] [train-lora] [serve] [gguf] [graph] [store]
[scrape] [web] [hub] [vision] [pdf] [viz] [plot] [testing] [telemetry] [mcp] [standard]
[privacy] [arrays] [test]`, plus `[torch]` and `[mlx]`, and `[all]`.

