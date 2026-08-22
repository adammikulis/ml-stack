"""The parts of a training loop that are the same everywhere and easy to get wrong.

Lab tier. Deliberately **not** a Trainer class: the loop body is where a project's actual
work lives, and wrapping it costs more than it saves. What is here is the scaffolding
around the loop -- checkpointing, schedules, guards, metrics, splitting -- each usable on
its own.

    from ml_stack.train import CheckpointState, MetricsLog, RunLock, warmup_stable_decay

    with RunLock(out), MetricsLog(out / "metrics.jsonl") as log:
        log.start(config)
        lr = warmup_stable_decay(3e-4, total_steps=100_000, warmup_steps=2_000)
        for step in range(100_000):
            opt.learning_rate = lr(step)     # a float, assigned outside any compiled region
            ...
"""

from __future__ import annotations

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
from ml_stack.train.daemon import (
    DaemonError,
    Job,
    JobRunner,
    safe_relpath,
    serve_forever,
)
from ml_stack.train.discovery import (
    Advertiser,
    Beacon,
    DiscoveryError,
    create_cluster_key,
    derive_token,
    discover,
    load_cluster_key,
)
from ml_stack.train.remote import RemoteError, RemoteTrainer
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
from ml_stack.train.schedule import (
    Schedule,
    constant,
    linear_warmup,
    warmup_cosine,
    warmup_stable_decay,
)

__all__ = [
    "serve_forever",
    "safe_relpath",
    "RemoteTrainer",
    "RemoteError",
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
