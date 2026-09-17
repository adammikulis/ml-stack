"""ml-stack-train-tools: a project's tool schemas, turned into a model that calls them.

One command does three things, each skipped when its output is already in ``--out``:
`synth` reads the worked examples out of the tool descriptions and templates them into
chat conversations, `train` runs the ``tool-calls`` recipe over them, and `export` writes
the checkpoint back out as a GGUF. One subcommand, ``from-bench``, seeds itself instead
from what a model actually did: every traced question a benchmark kept, above a score,
as one training example per model turn.

`schemas` reads the tools and the examples in their descriptions, `synthesise` writes the
conversations, `dataset` splits them and writes the files, `from_bench` reads a bench
trace, and `cli` is the command.
"""

from __future__ import annotations

from ml_stack.train.tools.cli import load_tools, main
from ml_stack.train.tools.dataset import split, write_dataset
from ml_stack.train.tools.from_bench import examples_from, from_bench, traced_rows, would_yield
from ml_stack.train.tools.schemas import CHAT, Example, examples_in, schemas_of
from ml_stack.train.tools.synthesise import SYSTEM, synthesise

__all__ = [
           "CHAT",
           "SYSTEM",
           "Example",
           "examples_from",
           "examples_in",
           "from_bench",
           "load_tools",
           "main",
           "schemas_of",
           "split",
           "synthesise",
           "traced_rows",
           "would_yield",
           "write_dataset",
]
