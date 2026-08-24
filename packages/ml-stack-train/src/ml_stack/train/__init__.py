"""Training: the loop, and the parts of it that are easy to get wrong.

Lab tier. ``Trainer`` runs the loop, on PyTorch or MLX, with checkpointing, resume,
schedules, divergence guards and metrics already attached:

    from ml_stack.train import Trainer, warmup_stable_decay

    report = Trainer(model, optimizer, loss, out="runs/small").fit(
        batches, steps=100_000,
        schedule=warmup_stable_decay(3e-4, total_steps=100_000, warmup_steps=2_000),
        eval_data=holdout, eval_every=1_000, checkpoint_every=1_000)

``loss(model, batch)`` is yours; everything around it is not. The framework is detected
from the model, so the same call trains on a Mac and on a CUDA box.

Every piece is also usable on its own, for a loop you want to write yourself --
``CheckpointState``, ``MetricsLog``, ``RunLock``, the schedules, the guards, the leak-safe
splits. ``Trainer`` is an assembly of them, not a replacement for them.
"""

from __future__ import annotations

from ml_stack.fleet import (
    Advertiser,
    Beacon,
    DaemonError,
    DiscoveryError,
    Job,
    JobRunner,
    Peer,
    PeerError,
    create_cluster_key,
    derive_token,
    discover,
    load_cluster_key,
    safe_relpath,
    serve_forever,
)
from ml_stack.train.checkpoint import (
    CheckpointError,
    CheckpointState,
    assert_exact_restore,
    checkpoint_name,
    find_latest,
    is_valid,
    load_state,
    load_tensors,
    point_latest_at,
    rotate,
    save,
)
from ml_stack.train.guard import (
    NonFiniteBudget,
    RunLock,
    RunLockError,
    StallWatchdog,
    StepTimer,
    TrainingDiverged,
)
from ml_stack.train.holdout import (
    GUARD,
    LeakageError,
    Split,
    assert_no_duplicates,
    by_group,
    contiguous_tail,
    spread_order,
)
from ml_stack.train.metrics import MetricsLog, Throughput, read
from ml_stack.train.step import MLXStep, Step, TorchStep, step_for
from ml_stack.train.trainer import Trainer, TrainReport, batches_from
from ml_stack.train.schedule import (
    Schedule,
    constant,
    linear_warmup,
    warmup_cosine,
    warmup_stable_decay,
)

__all__ = [
    "Trainer",
    "TrainReport",
    "Step",
    "TorchStep",
    "MLXStep",
    "step_for",
    "batches_from",
    "serve_forever",
    "safe_relpath",
    "Peer",
    "PeerError",
    "Advertiser",
    "Beacon",
    "DiscoveryError",
    "create_cluster_key",
    "derive_token",
    "discover",
    "load_cluster_key",
    "JobRunner",
    "Job",
    "DaemonError",
    "GUARD",
    "CheckpointError",
    "CheckpointState",
    "LeakageError",
    "MetricsLog",
    "NonFiniteBudget",
    "RunLock",
    "RunLockError",
    "Schedule",
    "Split",
    "StallWatchdog",
    "StepTimer",
    "Throughput",
    "TrainingDiverged",
    "assert_exact_restore",
    "assert_no_duplicates",
    "by_group",
    "checkpoint_name",
    "constant",
    "contiguous_tail",
    "find_latest",
    "is_valid",
    "linear_warmup",
    "load_state",
    "load_tensors",
    "point_latest_at",
    "read",
    "rotate",
    "save",
    "spread_order",
    "warmup_cosine",
    "warmup_stable_decay",
    "Fertility", "embedding_params", "measure", "report_markdown",
]

from ml_stack.train.fertility import (  # noqa: E402
    Fertility, embedding_params, measure, report_markdown,
)
