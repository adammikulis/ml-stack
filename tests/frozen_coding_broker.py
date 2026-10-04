"""PyInstaller runtime hook replacing only the model-serving boundary in Coding proofs."""
from contextlib import contextmanager
from types import SimpleNamespace

from ml_stack import codex, harnessing


@contextmanager
def serving(model, want, say, by):
    yield "http://127.0.0.1:1", SimpleNamespace(serving=SimpleNamespace(slot_context=want.ctx)), model


harnessing.serving = serving
codex.alias_of = lambda base, found: "fixture"
