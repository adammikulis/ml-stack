# Reranker comparison, 2026-10-08

Which small reranker reorders the top hits of `graph.search.hybrid` best. Text only: the multimodal
side (the embedding model's mmproj, Qwen3-VL-Reranker) was not measured.

## What was run

- Command: `poolhouse-bench retrieval --store ~/.poolhouse/bench/graph.ladybug --embed-url http://127.0.0.1:8081 --embed-model embeddinggemma-2-Q8_0.gguf --rerank-url http://127.0.0.1:8082 --rerank-model <name> -k 6`
- Store: `poolhouse-bench prepare` over the invented community (134 nodes, 207 edges), all 134 nodes
  embedded with `embeddinggemma-2-Q8_0.gguf` (`ggml-org/embeddinggemma-2-GGUF`, Apache-2.0).
- Questions: `poolhouse.graph.community.QUESTIONS`, 100 scored; 10 expect nobody and are left out.
- Rerankers, each served alone through the lease with `--reranking` (llama.cpp b11514, `de7fa0a3c`):
  `ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF` (639 MB, Apache-2.0) and
  `gpustack/bge-reranker-v2-m3-GGUF` Q8_0 (636 MB, Apache-2.0).
- One run each, no confidence bands.

## Result (window = the top 6 fused rows, `RERANK`)

| order | recall@6 | MRR | nDCG@6 |
| --- | --- | --- | --- |
| fused | 0.471 | 0.388 | 0.372 |
| vector similarity (today) | 0.471 | 0.403 | 0.384 |
| bge-reranker-v2-m3 | 0.471 | 0.408 | 0.393 |
| Qwen3-Reranker-0.6B | 0.471 | 0.435 | 0.406 |

## Reading it

- Qwen3-Reranker-0.6B is ahead of today's vector order by 0.032 MRR and 0.022 nDCG, and of
  bge-reranker-v2-m3 by 0.027 MRR, at the same file size. bge is barely ahead of the vector order.
- Recall@6 cannot move: the seam reorders the first six rows and never changes which rows are in
  them. Only 47% of the right answers are in those six, so the other 53% are out of a reranker's reach.
  The next experiment is a wider window (rerank the top 24, then cut to 6), where recall can move.
- Not measured: latency, image inputs, and jina-reranker-v1-tiny-en, which llama.cpp b11514 does not
  load (`architecture: 'jina-bert-v2'` is not read by the build).
- The community is invented and the sample is 100 questions from one run, so a 0.03 difference is a
  lead, not a proof.
