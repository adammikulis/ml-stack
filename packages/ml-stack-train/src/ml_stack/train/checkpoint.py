"""Checkpoints that survive being killed, and resumes that are actually exact.

Two failure modes this is shaped around, both of which cost days rather than minutes:

**A half-written checkpoint that looks complete.** A process killed during a save leaves a
directory containing some of the files. A later resume reads it, gets a partial model, and
trains on from there -- producing a run that is subtly wrong with nothing in the logs. So:
every checkpoint is built in a ``.partial`` directory and moved into place with
``os.replace``, and the metadata file that makes a directory *count* as a checkpoint is
written **last**. A directory without it is ignored, which is exactly what a half-written
one will be.

**A resume that silently restores less than it saved.** Restoring the weights but not the
optimizer state is not a resume; it is a fresh run starting from a warm initialisation,
and it shows up as a loss spike that is easy to mistake for a bad learning rate. So the
loader is strict: every tensor the checkpoint holds must land somewhere, and anything
missing or extra raises rather than being tolerated.

The array serialisation itself is delegated: this module owns the *directory protocol*
(atomicity, validity, rotation, ``latest``), and a backend supplies ``save``/``load`` for
whatever it stores tensors in.
"""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

STATE_FILE = "state.json"
"""Written last. Its presence is what makes a directory a valid checkpoint."""

LATEST = "latest"
WEIGHTS_FILE = "model.safetensors"
OPTIMIZER_FILE = "optimizer.safetensors"


class CheckpointError(RuntimeError):
    """A checkpoint could not be written, or could not be trusted."""


@dataclass
class CheckpointState:
    """Everything needed to resume that is not a tensor."""

    step: int
    epoch: int = 0
    best_metric: float | None = None
    rng: dict[str, Any] | None = None
    """Bit-generator state. Without it, a resumed run re-draws the same batches the
    original run already trained on, which quietly turns a fresh epoch into a repeat."""
    config: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)


def save(
    directory: Path | str,
    *,
    state: CheckpointState,
    tensors: dict[str, Any],
    optimizer: dict[str, Any] | None = None,
    write_tensors: Callable[[Path, dict[str, Any]], None],
) -> Path:
    """Write a checkpoint atomically. Returns the finished directory.

    ``write_tensors(path, mapping)`` is the backend's serialiser -- typically
    ``mlx.core.save_safetensors`` or ``safetensors.torch.save_file``.
    """
    directory = Path(directory)
    staging = directory.with_name(directory.name + ".partial")

    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    try:
        _write_and_verify(write_tensors, staging / WEIGHTS_FILE, tensors)
        if optimizer:
            _write_and_verify(write_tensors, staging / OPTIMIZER_FILE, optimizer)

        # Last. Everything above must have succeeded for this file to exist, which is
        # precisely the property `is_valid` relies on.
        (staging / STATE_FILE).write_text(
            json.dumps(asdict(state), indent=2, default=str), encoding="utf-8"
        )
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    if directory.exists():
        shutil.rmtree(directory)
    os.replace(staging, directory)
    return directory


def _write_and_verify(
    write_tensors: Callable[[Path, dict[str, Any]], None],
    path: Path,
    tensors: dict[str, Any],
) -> None:
    """Call the serialiser, then check it actually produced the file it was given.

    Some serialisers append their own extension (``numpy.savez`` is the common one). The
    save then appears to succeed, ``state.json`` gets written, and the checkpoint passes
    ``is_valid`` while being unloadable -- which is exactly the class of failure the
    write-state-last protocol exists to prevent, reintroduced one layer down.
    """
    write_tensors(path, tensors)
    if not path.is_file():
        near = sorted(p.name for p in path.parent.iterdir() if p.name.startswith(path.name))
        raise CheckpointError(
            f"the tensor writer did not create {path.name}"
            + (f"; it wrote {near} instead -- it is renaming the file" if near else "")
            + ". Pass a writer that honours the path it is given."
        )


def is_valid(directory: Path | str) -> bool:
    """Whether this directory is a complete checkpoint."""
    return (Path(directory) / STATE_FILE).is_file()


def load_state(directory: Path | str) -> CheckpointState:
    """Read the non-tensor state. Raises if the checkpoint is incomplete."""
    directory = Path(directory)
    path = directory / STATE_FILE
    if not path.is_file():
        raise CheckpointError(
            f"{directory} has no {STATE_FILE}, so it is not a complete checkpoint "
            "(most likely a save that was interrupted)"
        )
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CheckpointError(f"{path} is not valid JSON: {exc}") from exc
    return CheckpointState(**raw)


def load_tensors(
    directory: Path | str,
    *,
    read_tensors: Callable[[Path], dict[str, Any]],
    optimizer: bool = False,
) -> dict[str, Any]:
    """Read a checkpoint's tensors. ``optimizer=True`` reads the optimizer file instead."""
    directory = Path(directory)
    if not is_valid(directory):
        raise CheckpointError(f"{directory} is not a complete checkpoint")

    path = directory / (OPTIMIZER_FILE if optimizer else WEIGHTS_FILE)
    if not path.is_file():
        if optimizer:
            raise CheckpointError(
                f"{directory} has no {OPTIMIZER_FILE}. Resuming from it would restore the "
                "weights but not the optimizer -- which is a warm restart, not a resume, "
                "and shows up as a loss spike."
            )
        raise CheckpointError(f"{directory} has no {WEIGHTS_FILE}")
    return read_tensors(path)


def assert_exact_restore(saved: dict[str, Any], target: dict[str, Any]) -> None:
    """Fail unless the checkpoint's tensors exactly match what is being restored into.

    A partial restore is the failure that eats a multi-day run: it does not raise, the loss
    is merely worse, and by the time that is noticeable the original run is long gone.
    """
    only_saved = sorted(set(saved) - set(target))
    only_target = sorted(set(target) - set(saved))
    if only_saved or only_target:
        raise CheckpointError(
            "checkpoint does not match the model:\n"
            f"  in the checkpoint but not the model: {only_saved}\n"
            f"  in the model but not the checkpoint: {only_target}\n"
            "Refusing a partial restore."
        )

    mismatched = [
        name
        for name in saved
        if tuple(getattr(saved[name], "shape", ())) != tuple(getattr(target[name], "shape", ()))
    ]
    if mismatched:
        detail = ", ".join(
            f"{n}: {tuple(saved[n].shape)} vs {tuple(target[n].shape)}" for n in mismatched[:5]
        )
        raise CheckpointError(f"shape mismatch on {len(mismatched)} tensor(s): {detail}")


def point_latest_at(root: Path | str, directory: Path | str) -> Path:
    """Repoint ``root/latest`` at ``directory``, atomically.

    Through a temporary symlink and ``os.replace``: removing the old link and creating a
    new one leaves a window in which ``latest`` does not exist, and a resume that lands in
    that window reports no checkpoint at all.
    """
    root, directory = Path(root), Path(directory)
    link = root / LATEST
    staging = root / f"{LATEST}.tmp"

    staging.unlink(missing_ok=True)
    staging.symlink_to(directory.name, target_is_directory=True)
    os.replace(staging, link)
    return link


def find_latest(root: Path | str) -> Path | None:
    """The most recent valid checkpoint, or ``None``.

    Follows ``latest`` when it resolves to a valid checkpoint, and otherwise scans -- so a
    dangling or stale symlink degrades to "find it the slow way" rather than to "there is
    nothing here".
    """
    root = Path(root)
    if not root.is_dir():
        return None

    link = root / LATEST
    if link.is_symlink() or link.is_dir():
        resolved = link.resolve()
        if is_valid(resolved):
            return resolved

    candidates = sorted(
        (d for d in root.iterdir() if d.is_dir() and not d.name.startswith(".") and is_valid(d)),
        key=lambda d: d.name,
    )
    return candidates[-1] if candidates else None


def rotate(
    root: Path | str,
    *,
    keep_last: int = 3,
    milestone_every: int = 0,
    protected: tuple[str, ...] = ("best",),
) -> list[Path]:
    """Delete old checkpoints, keeping the last N plus every Nth milestone.

    Returns what was removed. Checkpoints whose name is in ``protected`` are never touched,
    and neither is whatever ``latest`` points at.

    Milestones exist because "keep the last 3" alone means a run that goes wrong at step
    50k has nothing left from step 10k to go back to.
    """
    root = Path(root)
    if not root.is_dir():
        return []

    checkpoints = sorted(
        (
            d
            for d in root.iterdir()
            if d.is_dir() and is_valid(d) and d.name not in protected and not d.is_symlink()
        ),
        key=lambda d: d.name,
    )

    keep: set[Path] = set(checkpoints[-keep_last:] if keep_last > 0 else [])

    link = root / LATEST
    if link.is_symlink():
        keep.add(link.resolve())

    if milestone_every > 0:
        for directory in checkpoints:
            try:
                step = load_state(directory).step
            except CheckpointError:
                continue
            if step % milestone_every == 0:
                keep.add(directory)

    removed: list[Path] = []
    for directory in checkpoints:
        if directory in keep:
            continue
        shutil.rmtree(directory, ignore_errors=True)
        removed.append(directory)
    return removed


def checkpoint_name(step: int, *, width: int = 9) -> str:
    """``step_000001000``. Zero-padded so lexical order is numeric order.

    That property is load-bearing: ``find_latest`` and ``rotate`` both sort by name, and
    unpadded names put ``step_9`` after ``step_10000``.
    """
    return f"step_{step:0{width}d}"
