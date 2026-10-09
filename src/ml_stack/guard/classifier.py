"""A prompt-injection classifier run on this machine through onnxruntime.

Needs the ``guard-model`` extra (onnxruntime, tokenizers, numpy) and the
ONNX export of ``protectai/deberta-v3-base-prompt-injection-v2`` (Apache-2.0, 738 MB), which
:func:`fetch` downloads once; after that nothing here touches the network.
"""

from __future__ import annotations

import os
import urllib.parse
from pathlib import Path
from typing import Any

from ml_stack import home, net
from ml_stack.guard.untrusted import unfenced
from ml_stack.interventions import Base, Call, Context, Deny, Proceed, Rewrite, Verdict
from ml_stack.safenames import safe_filename

__all__ = ["MODEL", "PROSE", "InjectionClassifierRail", "cached", "fetch"]

MODEL = "protectai/deberta-v3-base-prompt-injection-v2"
ENDPOINT = "https://huggingface.co"
PROSE = frozenset({"speech_transcribe", "web_search", "web_fetch", "WebFetch", "WebSearch"})
"""Sources whose results are free text. The model scores a list, table or JSON of eight or more
records as an injection (0.95 and up), so it is not run on structured tool output."""
WINDOW = 128
STRIDE = 64


def _root(cache_dir: str | Path | None) -> Path:
    return Path(cache_dir) if cache_dir else home.cache("guard", "classifier", MODEL.replace("/", "--"))


def _wanted(path: str) -> bool:
    return path == "config.json" or path.startswith("onnx/")


def fetch(cache_dir: str | Path | None = None) -> Path:
    """Download the model through the net pipeline (allow-listed host, the Hub's own size and
    SHA-256 for each file, format check, scan) and return its folder."""
    base = (os.environ.get("HF_ENDPOINT") or ENDPOINT).rstrip("/")
    tree = net.default().json(f"{base}/api/models/{MODEL}/tree/main?recursive=true",
                              net.Ask(purpose="injection classifier", tries=3))
    root = _root(cache_dir)
    for row in tree if isinstance(tree, list) else ():
        if not isinstance(row, dict) or row.get("type") != "file" or not _wanted(str(row.get("path"))):
            continue
        path = str(row["path"])
        lfs = row.get("lfs") if isinstance(row.get("lfs"), dict) else {}
        size = int(lfs.get("size") or row.get("size") or 0)
        want = net.Want(sha256=str(lfs.get("oid") or ""), size=size, max_bytes=size + (1 << 20)
                        if size else 1 << 30, purpose="injection classifier")
        final = root.joinpath(*(safe_filename(part) for part in path.split("/")))
        if not final.is_file():
            net.download(f"{base}/{MODEL}/resolve/main/{urllib.parse.quote(path)}", final, want)
    folder = root / "onnx"
    if not (folder / "model.onnx").is_file():
        raise FileNotFoundError(f"{MODEL} has no onnx/model.onnx on {base}")
    return folder


def cached(cache_dir: str | Path | None = None) -> Path | None:
    """The folder holding the model if it is already on this machine, else None."""
    folder = _root(cache_dir) / "onnx"
    return folder if (folder / "model.onnx").is_file() else None


class InjectionClassifierRail(Base):
    """Scores each result from a ``sources`` tool for text that gives the assistant orders.
    Over ``taint`` the result is tainted; over ``withhold``, when one is given, it is replaced
    by a notice."""

    name = "injection-model"

    def __init__(self, folder: Path, *, taint: float = 0.5, withhold: float | None = None,
                 sources: frozenset[str] = PROSE) -> None:
        import onnxruntime
        from tokenizers import Tokenizer

        self.folder = folder
        self.taint, self.withhold, self.sources = taint, withhold, sources
        self.tokenizer = Tokenizer.from_file(str(folder / "tokenizer.json"))
        self.session = onnxruntime.InferenceSession(str(folder / "model.onnx"),
                                                    providers=["CPUExecutionProvider"])
        self.inputs = {i.name for i in self.session.get_inputs()}

    def score(self, text: str) -> float:
        """The probability that ``text`` is an injection; the highest over its token windows."""
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

    def after_tool_call(self, call: Call, result: str, context: Context) -> Verdict:
        if call.name not in self.sources:
            return Proceed()
        score = self.score(unfenced(result))
        if self.withhold is not None and score >= self.withhold:
            return Deny(f"reads as an instruction to the assistant (score {score:.2f})", self.name)
        if score >= self.taint:
            return Rewrite(result, f"may be an instruction (score {score:.2f})", True, self.name)
        return Proceed()
