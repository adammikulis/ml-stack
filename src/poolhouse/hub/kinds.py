"""What a model is for: the one list of kinds, and the rules that label a model with one.

A kind is a label for listings and filters. It is never a grant: a model labelled
``decision`` is loaded through the same Broker lease, admission and memory fit as any other,
and a file's own words (its name, its repository) can claim a kind but cannot widen what the
model is allowed to do.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

from poolhouse import deciders

CHAT = "chat"
EMBEDDING = "embedding"
VISION = "vision"
SPEECH = "speech"
DECISION = "decision"

KINDS = (CHAT, EMBEDDING, VISION, SPEECH, DECISION)
"""Every kind of model, in the order a listing shows them. The only definition."""

_DECISION_WORDS = re.compile(r"(?<![a-z0-9])(decision|decider|deciders)(?![a-z0-9])", re.I)
_EMBEDDING_ARCH = ("bert",)
_SPEECH_ARCH = ("whisper", "wav2vec", "parakeet")
_VISION_ARCH = ("llava", "fastvlm", "mllama", "qwen2_vl", "qwen2_5_vl", "qwen3_vl", "smolvlm")


def valid(kind: str) -> str:
    """``kind`` when it is one of `KINDS`, else a `ValueError` naming them."""
    if kind not in KINDS:
        raise ValueError(f"unknown model kind {kind!r}; one of {', '.join(KINDS)}")
    return kind


def says_decision(*words: str) -> bool:
    """Whether a name, repository or tag calls itself a decision model or decider."""
    return any(_DECISION_WORDS.search(w.replace("_", " ").replace("-", " ").replace("/", " "))
               for w in words if w)


def registered_paths() -> list[Path]:
    """The directories of every registered decider, and the local base models they sit on."""
    return deciders.roots()


def registered(path: Path | str, held: Iterable[Path] | None = None) -> bool:
    """Whether ``path`` is, or is inside, a registered decider's directory."""
    where = Path(path).expanduser()
    try:
        # a Hub snapshot is a link into blobs/: the folder the link sits in counts as well
        # as the file it points at
        names = {where.parent.resolve() / where.name, where.resolve()}
    except OSError:
        return False
    return any(here == one or one in here.parents for here in names
               for one in (registered_paths() if held is None else [Path(h).resolve() for h in held]))


def classify(  # noqa: PLR0913 - independent facts, all optional
        *, architecture: str = "", name: str = "", repo: str = "", tags: Iterable[str] = (),
        path: Path | str = "", has_projector: bool = False,
        held: Iterable[Path] | None = None) -> str:
    """The kind a model's header, names and registry entry say it is: ``decision`` first
    (a registered decider, or words saying so), then ``speech``, ``embedding`` and ``vision``
    from the architecture and projector, else ``chat``."""
    if (path and registered(path, held)) or says_decision(name, repo, *tags):
        return DECISION
    arch = architecture.lower()
    if any(a in arch for a in _SPEECH_ARCH):
        return SPEECH
    if any(a in arch for a in _EMBEDDING_ARCH):
        return EMBEDDING
    if has_projector or any(arch.startswith(a) for a in _VISION_ARCH):
        return VISION
    return CHAT
