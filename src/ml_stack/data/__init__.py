"""Data that ships with ml-stack, and is read rather than computed.

`fit.json` is the one source of truth for per-model KV-cache measurements -- what
llama.cpp actually allocated at load, not what a formula over a GGUF header predicts.
`ml_stack.serve.fit` reads it, `ml-stack-serve fit --measure` adds to it, and
`~/.ml-stack/fit.json` layers a machine's own additions over it.
"""
