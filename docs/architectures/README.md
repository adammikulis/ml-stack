# Architectures worth notes

One file per architecture that behaves unlike a dense transformer when served -- what the
GGUF header says it is, and what that meant when it was measured. Written when a model
surprised us; kept because the next release of the same family will not.

- [qwen4exp](qwen4exp.md) -- Qwen3.8-Flash-Next: hybrid recurrent, 1-in-4 compressed sparse
  attention, 512 experts, and a 51B-parameter 3-gram lookup table in one tensor that is
  paged on demand (why Real Mem is 90G for a 104G file).
- [gemma4](gemma4.md) -- gemma-4 E2B/E4B/26B: sliding windows on five layers in six, the
  last 18 layers share KV, per-layer embeddings; the header's KV formula overstates 14x.
- [gpt-oss](gpt-oss.md) -- alternating 128-token sliding layers; harmony template with
  `reasoning_effort`, and it needs its thinking (60% -> 24% F1 without).

`ml-stack-serve fit --tensors MODEL` and the header dump behind it are how a note starts.
