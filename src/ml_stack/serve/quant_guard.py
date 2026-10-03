"""The refusal of IQ-family GGUF quantisations on Apple silicon, and the recorded override."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

from ml_stack import sentinel
from ml_stack.hub import header, naming
from ml_stack.hub.modelfile import NotAModelFile
from ml_stack.platform import is_apple_silicon
from ml_stack.sentinel.events import Event, Severity
from ml_stack.serve import guarded, tensors
from ml_stack.serve.backend import ServerFailed

logger = logging.getLogger(__name__)

__all__ = [
    "BROKER_ENV",
    "ENV",
    "BlockedQuant",
    "IqQuant",
    "blocked_message",
    "enforce",
    "iq_quant",
    "overridden",
    "override_note",
    "refusal",
    "wire_allows",
]

ENV = "ML_STACK_ALLOW_IQ"
BROKER_ENV = "ML_STACK_BROKER_ALLOW_IQ_LEASES"

# Measured in docs/architectures/qwen4exp.md; the same K-quant build also ran faster in docs/report-2026-09-23.md.
EVIDENCE = ("The one measurement here is a pair of small runs on this Mac "
            "(docs/architectures/qwen4exp.md): Qwen3.8-Flash-Next UD-IQ4_XS took 70.1 s a "
            "question at 54% F1 over ten questions, UD-Q4_K_XL 43.7 s at 64% over nine.")

# IQ tensors holding more than this share of a file's weight bytes make it an IQ file.
IQ_SHARE = 0.5

_told = False


class BlockedQuant(ServerFailed):
    """A lease was refused because the model is an IQ quantisation on Apple silicon."""


@dataclass(frozen=True, slots=True)
class IqQuant:
    """The IQ quantisation a model file is: its name, what said so, and the share of weight
    bytes held by IQ tensors (0 when only the header or the file name said it)."""

    name: str
    basis: str
    share: float = 0.0


def iq_quant(model: object) -> IqQuant | None:
    """The IQ quantisation ``model`` is, or ``None``. The header's ``general.file_type`` and
    the tensor types over every shard decide; the file name decides only for a model that is
    not a file on this machine."""
    path = guarded.file_of(model)
    if path is None:
        found = naming.QUANT.search(str(model))
        quant = found.group(1).upper() if found else ""
        return IqQuant(quant, "file name") if quant.startswith("IQ") else None
    try:
        meta = header.meta(path) or {}
        label = header.FILE_TYPES.get(meta.get("general.file_type"), "")
        if label.startswith("IQ"):
            return IqQuant(label, "file_type")
        found_tensors = tensors.tensors_of(path)
    except (OSError, NotAModelFile):
        return None
    total = sum(one.bytes for one in found_tensors)
    by_type: dict[str, int] = {}
    for one in found_tensors:
        if one.type.startswith("iq"):
            by_type[one.type] = by_type.get(one.type, 0) + one.bytes
    held = sum(by_type.values())
    if total and held / total > IQ_SHARE:
        return IqQuant(max(by_type, key=lambda k: by_type[k]).upper(), "tensors", held / total)
    return None


def _alternatives(model: object, quant: IqQuant) -> list[str]:
    """Non-IQ builds of the same model on disk beside it, as paths."""
    path = guarded.file_of(model)
    if path is None:
        return []
    family = naming.QUANT.sub("", naming.SHARD.sub("", path.stem)).casefold()
    seen: dict[str, str] = {}
    for root in (path.parent, path.parent.parent):
        for one in [*root.glob("*.gguf"), *root.glob("*/*.gguf")]:
            found = naming.QUANT.search(one.name)
            later_shard = "-of-" in one.name and "-00001-of-" not in one.name
            if (found is None or found.group(1).upper().startswith("IQ") or one == path
                    or later_shard
                    or naming.QUANT.sub("", naming.SHARD.sub("", one.stem)).casefold() != family):
                continue
            seen.setdefault(found.group(1).upper(), str(one))
    return list(seen.values())[:5]


def _message(model: object, quant: IqQuant) -> str:
    named = naming.pretty_name(Path(str(model)).name) or str(model)
    how = {"file_type": f"its header says file type {quant.name}",
           "tensors": f"{quant.share:.0%} of its weights are {quant.name} tensors",
           "file name": f"its name says {quant.name} and it is not on disk to check"}[quant.basis]
    lines = [f"blocked: {named} is an IQ quantisation ({how}). On Apple silicon llama.cpp "
             "runs it through Metal, where IQ quantisations are expected to be slower and "
             "less accurate than a K-quant of the same model.", EVIDENCE]
    alternatives = _alternatives(model, quant)
    if alternatives:
        lines.append("On disk, not IQ, same model: " + ", ".join(alternatives))
    lines.append(f"To serve it anyway, the person runs it with {ENV}=1 in the environment, "
                 "`ml-stack-serve up --allow-iq`, or allow_iq=True on the lease. The "
                 "override is recorded and warned about.")
    return "\n".join(lines)


def overridden() -> bool:
    """Whether this process's own environment allows IQ quantisations on Metal."""
    return os.environ.get(ENV) == "1"


def wire_allows() -> bool:
    """Whether this process, as a broker, takes ``allow_iq`` from a client's request."""
    return os.environ.get(BROKER_ENV) == "1"


def override_note(model: object) -> str:
    """The warning ``status`` repeats for a server that was started under the override."""
    named = naming.pretty_name(Path(str(model)).name) or str(model)
    return (f"{named} is an IQ quantisation served on Apple silicon under the IQ override: "
            "expected to be slower and less accurate than a K-quant.")


def _record(model: object, quant: IqQuant, who: str) -> None:
    global _told
    if not _told:
        _told = True
        logger.warning("%s", override_note(model))
    node = sentinel.default()
    node.bus.emit(Event("serve.iq_override", Severity.WARNING, "serve", f"model:{model}",
                        {"model": str(model), "quant": quant.name, "basis": quant.basis,
                         "who": who}, node.clock()))


def refusal(model: object, *, gpu_layers: object = "auto") -> tuple[IqQuant, str] | None:
    """``(quant, message)`` when ``model`` is an IQ quantisation on Apple silicon, else ``None``."""
    if not is_apple_silicon() or str(gpu_layers) == "0":
        return None
    quant = iq_quant(model)
    return None if quant is None else (quant, _message(model, quant))


def blocked_message(model: object, *, gpu_layers: object = "auto") -> str:
    """The refusal text for ``model`` as this process's environment sees it, or an empty
    string when the lease would go ahead."""
    found = refusal(model, gpu_layers=gpu_layers)
    return "" if found is None or overridden() else found[1]


def enforce(model: object, *, gpu_layers: object = "auto", allow: bool = False,
            who: str = "") -> IqQuant | None:
    """Raise `BlockedQuant` for an IQ quantisation on Apple silicon unless ``allow`` or the
    environment allows it; an override is recorded and its quant returned. Returns ``None``
    when the model is not IQ or the machine is not Apple silicon."""
    found = refusal(model, gpu_layers=gpu_layers)
    if found is None:
        return None
    if not (allow or overridden()):
        raise BlockedQuant(found[1])
    _record(model, found[0], who)
    return found[0]
