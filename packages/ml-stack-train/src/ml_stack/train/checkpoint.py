"""Checkpoints that survive being killed, and resumes that are actually exact."""

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
    """Bit-generator state. Without it, a resumed run re-draws the same batches the"""
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
    """Write a checkpoint atomically. Returns the finished directory."""
    directory = Path(directory)
    staging = directory.with_name(directory.name + ".partial")

    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    try:
        _write_and_verify(write_tensors, staging / WEIGHTS_FILE, tensors)
        if optimizer:
            _write_and_verify(write_tensors, staging / OPTIMIZER_FILE, optimizer)

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
    """Call the serialiser, then check it actually produced the file it was given."""
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
    """Fail unless the checkpoint's tensors exactly match what is being restored into."""
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
    """Repoint ``root/latest`` at ``directory``, atomically."""
    root, directory = Path(root), Path(directory)
    link = root / LATEST
    staging = root / f"{LATEST}.tmp"

    staging.unlink(missing_ok=True)
    staging.symlink_to(directory.name, target_is_directory=True)
    os.replace(staging, link)
    return link


def find_latest(root: Path | str) -> Path | None:
    """The most recent valid checkpoint, or ``None``."""
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
    """Delete old checkpoints, keeping the last N plus every Nth milestone."""
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
    """``step_000001000``. Zero-padded so lexical order is numeric order."""
    return f"step_{step:0{width}d}"
