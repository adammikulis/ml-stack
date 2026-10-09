"""Data that ships with poolhouse, and is read rather than computed.

`fit.json` is the one source of truth for per-model KV-cache measurements -- what
llama.cpp actually allocated at load, not what a formula over a GGUF header predicts.
`poolhouse.serve.fit` reads it, `poolhouse-serve fit --measure` adds to it, and
`~/.poolhouse/fit.json` layers a machine's own additions over it.

`profiles.json` is the same idea for the *shape* rather than the memory: one record per
model of the serving and the asking that measured best, and the row of the bench store
that set it. `poolhouse.serve.profile` reads it, `poolhouse-bench report --profile` writes it
from the store, and `~/.poolhouse/profiles.json` layers a machine's own over it.
"""
