"""A prompt-injection classifier run on this machine through onnxruntime.

Needs the ``guard-model`` extra (onnxruntime, tokenizers, numpy, huggingface_hub) and the
ONNX export of ``protectai/deberta-v3-base-prompt-injection-v2`` (Apache-2.0, 738 MB), which
:func:`fetch` downloads once; after that nothing here touches the network.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ml_stack.guard.untrusted import unfenced
from ml_stack.guard.verdict import ALLOW, ToolCall, Verdict, deny, modify

__all__ = ["MODEL", "PROSE", "InjectionClassifierRail", "cached", "fetch"]

MODEL = "protectai/deberta-v3-base-prompt-injection-v2"
FILES = ["onnx/*", "config.json"]
PROSE = frozenset({"tool:speech_transcribe", "tool:web_search", "tool:web_fetch",
                   "tool:WebFetch", "tool:WebSearch"})
"""Sources whose results are free text. The model scores a list, table or JSON of eight or more
records as an injection (0.95 and up), so it is not run on structured tool output."""
WINDOW = 400
STRIDE = 200


def fetch(cache_dir: str | None = None) -> Path:
    """Download the model into the Hugging Face cache and return its folder."""
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(MODEL, allow_patterns=FILES, cache_dir=cache_dir)) / "onnx"


def cached(cache_dir: str | None = None) -> Path | None:
    """The folder holding the model if it is already on this machine, else None."""
    from huggingface_hub import snapshot_download

    try:
        return Path(snapshot_download(MODEL, allow_patterns=FILES, local_files_only=True,
                                      cache_dir=cache_dir)) / "onnx"
    except OSError:
        return None


class InjectionClassifierRail:
    """Scores each result from a ``sources`` tool for text that gives the assistant orders.
    Over ``taint`` the result is tainted; over ``withhold`` it is replaced by a notice."""

    name = "injection-model"

    def __init__(self, folder: Path, *, taint: float = 0.5, withhold: float = 0.98,
                 sources: frozenset[str] = PROSE) -> None:
        import onnxruntime
        from tokenizers import Tokenizer

        self.taint, self.withhold, self.sources = taint, withhold, sources
        self.tokenizer = Tokenizer.from_file(str(folder / "tokenizer.json"))
        self.session = onnxruntime.InferenceSession(str(folder / "model.onnx"),
                                                    providers=["CPUExecutionProvider"])
        self.inputs = {i.name for i in self.session.get_inputs()}

    def score(self, text: str) -> float:
        """The probability that ``text`` is an injection; the highest over its windows."""
        ids = self.tokenizer.encode(text, add_special_tokens=False).ids
        windows = [ids[i: i + WINDOW] for i in range(0, max(len(ids), 1), STRIDE)] or [[]]
        return max(self._window(w) for w in windows)

    def _window(self, ids: list[int]) -> float:
        import numpy as np

        cls, sep = self.tokenizer.token_to_id("[CLS]"), self.tokenizer.token_to_id("[SEP]")
        full = np.array([[cls, *ids, sep]], dtype=np.int64)
        feed: dict[str, Any] = {"input_ids": full, "attention_mask": np.ones_like(full)}
        logits = self.session.run(None, {k: v for k, v in feed.items() if k in self.inputs})[0][0]
        exp = np.exp(logits - logits.max())
        return float((exp / exp.sum())[1])

    def on_input(self, text: str, source: str) -> Verdict:
        if source not in self.sources:
            return ALLOW
        score = self.score(unfenced(text))
        if score >= self.withhold:
            return deny(self.name, f"reads as an instruction to the assistant (score {score:.2f})")
        if score >= self.taint:
            return modify(self.name, text, f"may be an instruction (score {score:.2f})", tainted=True)
        return ALLOW

    def on_output(self, text: str, source: str) -> Verdict:
        return ALLOW

    def on_tool_call(self, call: ToolCall) -> Verdict:
        return ALLOW
