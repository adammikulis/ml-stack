"""poolhouse-train-tools: a project's tool schemas, turned into a model that calls them.

One command does three things, each skipped when its output is already in ``--out``:
`synth` reads the worked examples out of the tool descriptions and templates them into
chat conversations, `train` runs the ``tool-calls`` recipe over them, and `export` writes
the checkpoint back out as a GGUF. ``from-bench`` seeds the data from what a model did
instead, and ``eval`` scores a served model on the held-out rows.

`schemas` reads the tools and the examples in their descriptions, `synthesise` writes the
conversations, `dataset` splits them and writes the files, `drift` pins the schemas a dataset was made
for, `evaluate` scores a served model on the held-out rows, `from_bench` reads a bench
trace, and `cli` is the command.
"""

from __future__ import annotations

from poolhouse.train.tools.cli import load_tools, main
from poolhouse.train.tools.dataset import split, write_dataset
from poolhouse.train.tools.drift import SchemaDrift, fingerprint, schema_hash, signatures
from poolhouse.train.tools.evaluate import evaluate, score
from poolhouse.train.tools.from_bench import examples_from, from_bench, traced_rows, would_yield
from poolhouse.train.tools.schemas import CHAT, Example, examples_in, schemas_of
from poolhouse.train.tools.synthesise import SYSTEM, synthesise

__all__ = [
           "CHAT",
           "SYSTEM",
           "Example",
           "SchemaDrift",
           "evaluate",
           "examples_from",
           "examples_in",
           "fingerprint",
           "from_bench",
           "load_tools",
           "main",
           "schema_hash",
           "schemas_of",
           "score",
           "signatures",
           "split",
           "synthesise",
           "traced_rows",
           "would_yield",
           "write_dataset",
]
