"""The parts of a training loop that are the same everywhere and easy to get wrong.

Lab tier. Deliberately **not** a Trainer class: the loop body is where a project's actual
work lives, and wrapping it costs more than it saves. What is here is the scaffolding
around the loop -- checkpointing, schedules, guards, metrics, splitting -- each usable on
its own.

    from mainspring.train import CheckpointState, MetricsLog, RunLock, warmup_stable_decay

    with RunLock(out), MetricsLog(out / "metrics.jsonl") as log:
        log.start(config)
        lr = warmup_stable_decay(3e-4, total_steps=100_000, warmup_steps=2_000)
        for step in range(100_000):
            opt.learning_rate = lr(step)     # a float, assigned outside any compiled region
            ...
"""

from __future__ import annotations

from mainspring.train.checkpoint import (
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
from mainspring.train.guard import (
    NonFiniteBudget,
    RunLock,
    RunLockError,
    StallWatchdog,
    StepTimer,
    TrainingDiverged,
)
from mainspring.train.holdout import (
    GUARD,
    LeakageError,
    Split,
    assert_no_duplicates,
    by_group,
    contiguous_tail,
    spread_order,
)
from mainspring.train.metrics import MetricsLog, Throughput, read
from mainspring.train.schedule import (
    Schedule,
    constant,
    linear_warmup,
    warmup_cosine,
    warmup_stable_decay,
)

__all__ = [
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
]
