"""What a model file's name says: its build, its quantisation, whether it is a head."""

from __future__ import annotations

import re
import sys
from pathlib import Path

WEIGHT_SUFFIXES = (".gguf", ".safetensors", ".bin", ".pt", ".onnx")
"""The file extensions a model's weights come in."""

DRAFT_MARK = ".draft"
"""The extra suffix a draft head carries: ``thing-Q4.draft.gguf``."""

# How a draft head names itself, and the `--spec-type` each one needs. A head is named by
# the *method* it implements, not by the fact that it is a draft: `mtp-` for
# multi-token prediction, `eagle3-` for EAGLE3. A rule that knew only one of them reported
# "no draft" for gpt-oss-20b, which ships two EAGLE3 heads and no mtp- file at all.
DRAFT_KINDS = {"mtp-": "draft-mtp", "eagle3-": "draft-eagle3"}

# The quantisation token in a GGUF file name: unsloth's dynamic builds prefix it with UD-.
QUANT = re.compile(r"-(?:UD-)?((?:IQ|Q)\d(?:_[A-Z0-9]+)+|mxfp4|BF16|F16|F32)", re.IGNORECASE)
SHARD = re.compile(r"-\d{5}-of-\d{5}$")
_SHARD = re.compile(r"-\d{5}-of-\d{5}(?=\.gguf$)", re.IGNORECASE)

# How a file says which precision it is, best first. A repository that ships more than one
# companion ships one per precision, and which to take depends on what the companion is.
_QUANTS = ("f32", "bf16", "f16", "q8_0", "q6_k", "q5_k_m", "q5_k", "q4_k_xl", "q4_k_m",
           "q4_k", "q4_0", "iq4_nl", "q3_k", "q2_k")

# Words that name a build rather than the model, dropped from both sides of a match.
_NOT_THE_MODEL = frozenset({"shared", "qat", "ud"})


def _precision(name: str) -> int:
    """Where a file sits in ``_QUANTS``; lower is more precise. Unmarked sorts last."""
    plain = name.casefold()
    for n, quant in enumerate(_QUANTS):
        if quant in plain:
            return n
    return len(_QUANTS)


def aside(name: str) -> int:
    """0 for the weights themselves, 1 for what merely travels alongside them.

    A subdirectory does not make something a companion. A large model is published one
    directory per quantisation -- `UD-Q4_K_XL/thing-00001-of-00004.gguf` -- and calling
    those "alongside" buries the weights under the projector and prints the model itself as
    an afterthought. What a file *is* is in its name, not its folder.
    """
    plain = name.lower().rsplit("/", 1)[-1]
    return 1 if plain.startswith(("mmproj", "imatrix")) or is_head(plain) else 0


def spec_for(draft: str) -> str:
    """The `--spec-type` a draft head needs, read from what it is called, or ''.

    A head implements one method and only that method: an EAGLE3 head served as
    `draft-simple` is not slower, it is wrong about what it is being asked to do.
    """
    plain = str(draft).lower().rsplit("/", 1)[-1]
    for prefix, kind in DRAFT_KINDS.items():
        if plain.startswith(prefix):
            return kind
    return ""


def is_head(name: str | Path) -> bool:
    """Whether a file name is a draft head rather than a model to serve."""
    plain = Path(str(name)).name
    return plain.lower().startswith(tuple(DRAFT_KINDS)) or DRAFT_MARK in Path(plain).suffixes


def borrowed_head(name: str | Path) -> bool:
    """Whether a head borrows its target's embeddings, which only a fork build loads."""
    return "shared" in Path(str(name)).name.lower()


def base_words(name: str | Path) -> tuple[str, ...]:
    """The words that name the model a file is for, with quantisation and shard dropped.

    ``mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf`` and
    ``Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf`` both give
    ``('qwen3', '8', 'flash', 'next')``, which is how a head on disk is matched to the
    model it drafts for.
    """
    text = Path(str(name)).name
    if text.lower().endswith(".gguf"):
        text = text[:-5]
    if text.lower().endswith(DRAFT_MARK):
        text = text[: -len(DRAFT_MARK)]
    text = QUANT.sub("", SHARD.sub("", text))
    for prefix in DRAFT_KINDS:
        if text.lower().startswith(prefix):
            text = text[len(prefix):]
            break
    words = (w for w in re.split(r"[-_. ]+", text.lower()) if w)
    return tuple(w for w in words if w not in _NOT_THE_MODEL)


def iq_on_metal(name: str, platform: str | None = None) -> bool:
    """Whether ``name`` is an IQ-quantised build being considered on Apple silicon, where
    its lookup-table kernels run slower than a K-quant's (measured 2026-09-02: the 87 GB
    IQ4_XS took 70 s a question where the 104 GB Q4_K_XL took 44). ``platform`` overrides
    ``sys.platform`` for a test."""
    where = platform if platform is not None else sys.platform
    return where == "darwin" and bool(re.search(r"(^|[-_.])IQ\d", name, re.IGNORECASE))


def pretty_name(name: str) -> str:
    """A model file's name as a person would say it: ``Qwen3.8-Flash-Next (IQ4_XS)`` for
    ``hf:.../Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf``. The path, the ``.gguf``,
    the shard suffix and the ``UD-`` prefix go; the quantisation moves into brackets. A
    head keeps its ``mtp-``/``eagle3-`` prefix so a legend still says what it is. The
    same rule is written once in JavaScript for the pages (``fit.html``, ``graph.html``)
    and a test holds the two to the same answers."""
    text = str(name or "").split("/")[-1].split("\\")[-1]
    if text.lower().endswith(".gguf"):
        text = text[:-5]
    text = SHARD.sub("", text)
    found = QUANT.search(text)
    if not found:
        return text
    quant = found.group(1)
    text = text[:found.start()] + text[found.end():]
    return f"{text.strip('-')} ({quant})"
