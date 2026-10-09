"""The warning, or in strict mode the refusal, for IQ-family GGUF quantisations on Apple silicon."""

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

__all__ = ["DEFAULT_MODE", "ENV", "MODES", "BlockedQuant", "IqQuant", "blocked_message", "enforce",
           "iq_quant", "mode", "refusal", "status_note"]

ENV = "ML_STACK_IQ"
MODES = ("off", "warn", "block")
DEFAULT_MODE = "warn"

EVIDENCE = ("Measured once on this Mac (docs/architectures/qwen4exp.md, "
            "docs/report-2026-09-23.md): Qwen3.8-Flash-Next UD-IQ4_XS took 70.1 s a question "
            "at 54% F1 over ten questions, UD-Q4_K_XL 43.7 s at 64% over nine. The same report "
            "also holds UD-Q4_K_XL at 27.6 s and 81% F1 on nine questions with the same asking, "
            "and UD-IQ4_XS at 85% F1 in 36.6 s with thinking off, so the evidence is thin and "
            "points both ways. A controlled test (docs/experiments/iq-vs-kquant-metal.md) is "
            "pending.")

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


def _text(model: object, quant: IqQuant, *, blocked: bool) -> str:
    named = naming.pretty_name(Path(str(model)).name) or str(model)
    how = {"file_type": f"its header says file type {quant.name}",
           "tensors": f"{quant.share:.0%} of its weights are {quant.name} tensors",
           "file name": f"its name says {quant.name} and it is not on disk to check"}[quant.basis]
    lines = [f"{'blocked' if blocked else 'warning'}: {named} is an IQ quantisation ({how}). "
             "On Apple silicon llama.cpp runs it through Metal, where IQ quantisations MAY be "
             "slower and less accurate than a K-quant of the same model.", EVIDENCE]
    alternatives = _alternatives(model, quant)
    if alternatives:
        lines.append("On disk, not IQ, same model: " + ", ".join(alternatives))
    lines.append(f"Set {ENV}=warn or `--iq warn` to serve it with this warning, {ENV}=off or "
                 f"`--iq off` for no warning, {ENV}=block or `--iq block` to refuse it."
                 if blocked else
                 f"{ENV}=block or `--iq block` refuses IQ quantisations on Apple silicon, "
                 f"{ENV}=off silences this.")
    return "\n".join(lines)


def mode(asked: str = "") -> str:
    """``asked`` when it is a mode, else ``$ML_STACK_IQ`` when that is one, else ``warn``."""
    for one in (asked, os.environ.get(ENV, "")):
        if one in MODES:
            return one
    return DEFAULT_MODE


def status_note(quant: str) -> str:
    """The line ``status`` shows for a server started with the IQ quantisation ``quant``."""
    return (f"{quant} is an IQ quantisation on Apple silicon: it MAY be slower and less "
            "accurate than a K-quant; the evidence is thin (docs/serving.md).")


def _record(model: object, quant: IqQuant, who: str, message: str) -> None:
    global _told
    if not _told:
        _told = True
        logger.warning("%s", message)
    node = sentinel.default()
    node.bus.emit(Event("serve.iq_warning", Severity.NOTICE, "serve", f"model:{model}",
                        {"model": str(model), "quant": quant.name, "basis": quant.basis,
                         "who": who}, node.clock()))


def refusal(model: object, *, gpu_layers: object = "auto") -> IqQuant | None:
    """The IQ quantisation ``model`` is when it is served on Apple silicon, else ``None``."""
    if not is_apple_silicon() or str(gpu_layers) == "0":
        return None
    return iq_quant(model)


def blocked_message(model: object, *, gpu_layers: object = "auto", asked: str = "") -> str:
    """The refusal text when strict mode refuses ``model`` here, else an empty string."""
    found = refusal(model, gpu_layers=gpu_layers)
    return _text(model, found, blocked=True) if found and mode(asked) == "block" else ""


def enforce(model: object, *, gpu_layers: object = "auto", asked: str = "",
            who: str = "") -> IqQuant | None:
    """Apply the IQ mode to a lease: `BlockedQuant` in ``block``, one warning per process and
    a ``serve.iq_warning`` event in ``warn``, nothing in ``off``. Returns the quantisation
    when it was warned about."""
    found = refusal(model, gpu_layers=gpu_layers)
    chosen = mode(asked)
    if found is None or chosen == "off":
        return None
    if chosen == "block":
        raise BlockedQuant(_text(model, found, blocked=True))
    _record(model, found, who, _text(model, found, blocked=False))
    return found
