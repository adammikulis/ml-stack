# Draft trees for `llama-server`

A patch for `ggml-org/llama.cpp`, carried in `patches/llama.cpp/0003-speculative-tree.patch`
and applied by `ml-stack-serve build --from source`. Not sent upstream; this note is what a
pull request would say.

## What it enables

A chain draft proposes one continuation: the first token the target disagrees with ends the
pass, so a pass is worth the length of the drafter's longest correct run. A tree proposes
several continuations at once — the drafter's top few tokens at each depth, each with its own
children — and the target verifies all of them in a single forward pass, keeping the one path
it agrees with.

```
ml-stack-serve up MODEL --parallel 1 --spec draft-mtp --spec-n-max 12 --spec-tree 3
```

`--spec-tree W` expands `W` branches per depth, each proposing `W` children, ranked by path
probability, up to `--spec-n-max` nodes. `0` (the default) drafts a chain, and the rest of the
speculative machinery is untouched.

## What the patch adds

- **A tree-shaped batch.** `llama_set_tree(ctx, parent, n)` marks the next decode: `parent[i]`
  is the node token `i` hangs off, `-1` for the committed prefix. Node positions are their
  depth past the prefix, so siblings share a position.
- **An ancestor mask.** The KV cache records the cells a tree batch was placed in, and masks
  every tree cell that is not on the token's own path to the root. The prefix stays visible.
- **Per-path recurrent state** (Qwen3.5/3.6/3.8 DeltaNet layers, and every other
  `llm_build_delta_net_base` model). Each node's convolution window is gathered along its own
  path, and the gated delta state is carried from parent to child: the batch is a set of
  one-token sequences whose initial states are gathered from their parents, run
  `max depth + 1` times, which is what it takes for every node to read its parent's final
  state. Node `n`'s state is written to recurrent snapshot group `n_nodes - 1 - n`, so a
  commit selects a node's state the way a chain rollback selects a token's.
- **A commit.** `llama_tree_commit(ctx, seq, path, n)` keeps a root-to-node path: the KV cells
  of the other nodes are freed and the recurrent state and convolution window of the last kept
  node become the sequence's. `n = 0` drops a whole tree, which is how a drafter clears its own
  scratch tree.
- **Two tree drafters.** The MTP head expands the tree by its own probabilities, one decode per
  depth over that depth's frontier. The ngram lookup builds a trie of the continuations that
  followed earlier occurrences of the last n-gram, most recent first.
- **Acceptance through the sampler that was already there.** The server samples at the root,
  follows the child holding that token, and stops at the first node with no matching child, so
  the tokens it emits are the tokens plain sampling would have emitted.

## Limits

- One conversation at a time: the server refuses `--spec-tree` unless `--parallel 1`.
- A hybrid model needs one recurrent snapshot group per drafted node
  (`n_rs_seq = --spec-n-max`), which is memory: on Qwen3.8-27B a group is about 150 MB.
- A model whose memory cannot roll back a single token (`COMMON_CONTEXT_SEQ_RM_TYPE_FULL`)
  is refused rather than served through checkpoints.

## Checking it

`llama-tree-verify MODEL [n_predict] [prompt] [max_nodes]` decodes greedily one token at a
time, then again verifying trees built around that continuation — the true tokens mixed with
wrong siblings, in shuffled order, one branch going wrong partway down — and reports the first
token that differs and the tokens kept per pass. It needs no drafter, so it tests the mask,
the state and the commit rather than the quality of a draft.
