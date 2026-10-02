"""The embed backend: sentence embeddings and a small trained head per decision.

With no head it ranks options by cosine similarity to the question and state. A head is
trained from a few dozen labelled cases: ``classes`` learns one weight vector per option of a
fixed set, ``pairwise`` learns one vector shared by all options and so scores option sets it
has not seen.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ml_stack.client.embed import embed as server_embed
from ml_stack.decide.base import Asked, BaseDecider
from ml_stack.decide.calibrate import Calibration
from ml_stack.decide.cases import Case
from ml_stack.decide.types import BackendUnavailable, DecideError, Option

FORMAT = "ml-stack-embed-head/1"
Embedder = Callable[[list[str]], list[list[float]]]
CACHE_MAX = 4096


def _numpy() -> Any:
    try:
        import numpy
    except ImportError as exc:
        raise BackendUnavailable("the embed backend needs numpy: pip install 'ml-stack[arrays]'"
                                 ) from exc
    return numpy


def server_embedder(base_url: str = "http://127.0.0.1:8080", *, model: str | None = None,
                    prefix: str = "", token: str = "") -> Embedder:
    """An `Embedder` that asks a server's ``/v1/embeddings``; ``prefix`` leads every text."""
    def run(texts: list[str]) -> list[list[float]]:
        return server_embed([prefix + t for t in texts], base_url=base_url, model=model,
                            api_key=token or None)
    return run


def option_text(option: Option) -> str:
    """What an option is embedded as."""
    return f"{option.name}: {option.description}" if option.description else option.name


def state_text(question: str, state: str) -> str:
    """What a question about a state is embedded as."""
    return f"{question}\n{state}"


@dataclass(frozen=True, slots=True)
class Head:
    """Trained weights: ``classes`` has ``W`` [options, dim] and ``b``; ``pairwise`` has ``w``."""

    kind: str
    dim: int
    options: tuple[str, ...]
    w: Any = None
    W: Any = None
    b: Any = None
    embedder: str = ""
    data_hash: str = ""
    n: int = 0

    def save(self, path: Path | str) -> None:
        """Write the head as a safetensors file with its description in the metadata."""
        from safetensors.numpy import save_file
        tensors = ({"w": self.w} if self.kind == "pairwise" else {"W": self.W, "b": self.b})
        meta = {"format": FORMAT, "kind": self.kind, "dim": str(self.dim),
                "options": json.dumps(list(self.options)), "embedder": self.embedder,
                "data_hash": self.data_hash, "n": str(self.n)}
        save_file({k: v.astype("float32") for k, v in tensors.items()}, str(path), metadata=meta)

    @classmethod
    def load(cls, path: Path | str) -> Head:
        """Read a head written by `save`; anything not matching its own description raises."""
        from safetensors import safe_open
        with safe_open(str(path), framework="numpy") as f:
            meta = f.metadata() or {}
            if meta.get("format") != FORMAT:
                raise DecideError(f"{path} is not an embed head ({meta.get('format')!r})")
            kind, dim = meta["kind"], int(meta["dim"])
            options = tuple(json.loads(meta["options"]))
            got = {k: f.get_tensor(k) for k in f.keys()}  # noqa: SIM118
        shapes = {"pairwise": {"w": (dim,)}, "classes": {"W": (len(options), dim),
                                                          "b": (len(options),)}}
        if kind not in shapes or {k: v.shape for k, v in got.items()} != shapes[kind]:
            raise DecideError(f"{path}: tensors do not match a {kind!r} head of dim {dim}")
        return cls(kind, dim, options, w=got.get("w"), W=got.get("W"), b=got.get("b"),
                   embedder=meta.get("embedder", ""), data_hash=meta.get("data_hash", ""),
                   n=int(meta.get("n", 0)))


def _unit(np: Any, rows: Sequence[Sequence[float]]) -> Any:
    m = np.asarray(rows, dtype="float64")
    return m / np.maximum(np.linalg.norm(m, axis=-1, keepdims=True), 1e-12)


def _adam(np: Any, grad: Callable[[Any], Any], start: Any, steps: int, lr: float) -> Any:
    x, m, v = start.copy(), np.zeros_like(start), np.zeros_like(start)
    for t in range(1, steps + 1):
        g = grad(x)
        m = 0.9 * m + 0.1 * g
        v = 0.999 * v + 0.001 * g * g
        x = x - lr * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
    return x


def _softmax(np: Any, z: Any) -> Any:
    e = np.exp(z - z.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


@dataclass(frozen=True, slots=True)
class Training:
    """How a head is trained and what it records about itself."""

    kind: str = "pairwise"
    l2: float = 1e-3
    steps: int = 300
    lr: float = 0.05
    scale: float = 20.0
    embedder_id: str = ""
    data_hash: str = ""


def fit_head(embedder: Embedder, cases: Sequence[Case], training: Training | None = None) -> Head:
    """Train a head on ``cases``; ``classes`` needs every case to offer the same options."""
    np = _numpy()
    training = training or Training()
    kind, l2, steps, lr, scale = (training.kind, training.l2, training.steps, training.lr,
                                  training.scale)
    embedder_id, data_hash = training.embedder_id, training.data_hash
    if kind not in ("pairwise", "classes"):
        raise ValueError(f"kind must be 'pairwise' or 'classes', got {kind!r}")
    if not cases:
        raise ValueError("no cases to train on")
    X = _unit(np, embedder([state_text(c.question, c.state if isinstance(c.state, str)
                                       else json.dumps(c.state, default=str)) for c in cases]))
    names = tuple(o.name for o in cases[0].options)
    y = np.array([c.label_index for c in cases])
    if kind == "classes":
        if any(tuple(o.name for o in c.options) != names for c in cases):
            raise ValueError("classes training needs the same options, in order, in every case")
        n = len(names)

        def grad_c(p: Any) -> Any:
            W, b = p[:, :-1], p[:, -1]
            P = _softmax(np, X @ W.T * scale + b)
            P[np.arange(len(y)), y] -= 1.0
            return np.concatenate([(P.T @ X) * scale / len(y) + 2 * l2 * W,
                                   (P.sum(0) / len(y))[:, None]], axis=1)

        p = _adam(np, grad_c, np.zeros((n, X.shape[1] + 1)), steps, lr)
        return Head("classes", X.shape[1], names, W=p[:, :-1] * scale, b=p[:, -1],
                    embedder=embedder_id, data_hash=data_hash, n=len(cases))
    texts = sorted({option_text(o) for c in cases for o in c.options})
    table = dict(zip(texts, _unit(np, embedder(texts)), strict=True))
    vectors = [np.stack([table[option_text(o)] for o in c.options]) for c in cases]

    def grad_p(w: Any) -> Any:
        total = 2 * l2 * (w - 1.0)
        for x, opts, label in zip(X, vectors, y, strict=True):
            F = opts * x
            P = _softmax(np, F @ w * scale)
            P[label] -= 1.0
            total = total + (P @ F) * scale / len(y)
        return total

    w = _adam(np, grad_p, np.ones(X.shape[1]), steps, lr)
    return Head("pairwise", X.shape[1], names, w=w * scale, embedder=embedder_id,
                data_hash=data_hash, n=len(cases))


class EmbedDecider(BaseDecider):
    """Scores options with embeddings and, when given, a trained `Head`."""

    name = "embed"

    def __init__(self, embedder: Embedder, *, head: Head | None = None,
                 temperature: float = 0.05, calibration: Calibration | None = None,
                 model: str = "") -> None:
        self.embedder = embedder
        self.head = head
        self.temperature = temperature
        self.calibration = calibration
        self.model = model or (head.embedder if head else "")
        self._cache: dict[str, Any] = {}
        self._lock = threading.Lock()

    def _options(self, np: Any, options: tuple[Option, ...]) -> Any:
        texts = [option_text(o) for o in options]
        with self._lock:
            missing = [t for t in dict.fromkeys(texts) if t not in self._cache]
        if missing:
            vectors = _unit(np, self.embedder(missing))
            with self._lock:
                if len(self._cache) + len(missing) > CACHE_MAX:
                    self._cache.clear()
                self._cache.update(zip(missing, vectors, strict=True))
        with self._lock:
            return np.stack([self._cache[t] for t in texts])

    def probabilities(self, asked: Asked) -> tuple[list[float], dict[str, Any]]:
        np = _numpy()
        x = _unit(np, self.embedder([state_text(asked.question, asked.state)]))[0]
        head = self.head
        if head is not None and x.shape[0] != head.dim:
            raise DecideError(f"the embedder returns {x.shape[0]} dimensions; the head "
                              f"was trained on {head.dim}")
        if head is not None and head.kind == "classes":
            names = [o.name for o in asked.options]
            if sorted(names) != sorted(head.options):
                raise DecideError(f"this head decides among {list(head.options)}, "
                                  f"not {names}")
            z = head.W @ x + head.b
            by = dict(zip(head.options, z, strict=True))
            logits = np.array([by[n] for n in names])
        else:
            w = head.w if head is not None else np.full(x.shape[0], 1.0 / self.temperature)
            logits = (self._options(np, asked.options) * x) @ w
        return [float(p) for p in _softmax(np, logits)], {"head": head.kind if head else "zero-shot"}
