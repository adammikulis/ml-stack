"""Train a model, with the things that are easy to get wrong already wired in."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ml_stack.train.checkpoint import (
    CheckpointState,
    checkpoint_name,
    find_latest,
    load_state,
    load_tensors,
    point_latest_at,
    rotate,
    save,
    tensor_reader,
    tensor_writer,
)
from ml_stack.train.guard import (
    NonFiniteBudget,
    RunLock,
    StallWatchdog,
    StepTimer,
    TrainingDiverged,
)
from ml_stack.train.metrics import MetricsLog, Throughput
from ml_stack.train.schedule import Schedule, constant
from ml_stack.train.step import Step, step_for

__all__ = ["Hook", "TrainReport", "Trainer", "batches_from"]

FEW_STEPS = 10
"""A time budget that fits fewer steps than this after the first one is noted in the log."""


@dataclass
class Hook:
    """``run(step)`` after every ``every``-th applied step; each mapping it returns is a note."""

    name: str
    every: int
    run: Callable[[int], Mapping[str, Any] | Sequence[Mapping[str, Any]] | None]


@dataclass
class TrainReport:
    """What happened. Returned rather than printed, so a caller can assert on it."""

    steps: int = 0
    resumed_from: int = 0
    final_loss: float = float("nan")
    best_metric: float | None = None
    best_checkpoint: Path | None = None
    last_checkpoint: Path | None = None
    skipped: int = 0
    stalls: list[str] = field(default_factory=list)
    seconds: float = 0.0
    history: list[dict[str, Any]] = field(default_factory=list)
    stop_reason: str = "completed"
    """``completed``, ``stopped``, ``time_limit`` or ``already complete``."""

    @property
    def steps_per_second(self) -> float:
        return self.steps / self.seconds if self.seconds > 0 else 0.0


def batches_from(data: Any, *, batch_size: int = 0) -> Callable[[int], Any]:
    """Turn whatever the caller has into ``step -> batch``."""
    if callable(data):
        return data

    if hasattr(data, "__len__") and batch_size:
        rows: Sequence[Any] = data
        n = len(rows)

        def slice_batch(step: int) -> Any:
            start = (step * batch_size) % max(1, n)
            chunk = rows[start:start + batch_size]
            if len(chunk) < batch_size and n >= batch_size:
                chunk = list(chunk) + list(rows[:batch_size - len(chunk)])
            return chunk

        return slice_batch

    iterator: Iterator[Any] = iter(data)
    source: Iterable[Any] = data

    def next_batch(step: int) -> Any:
        nonlocal iterator
        try:
            return next(iterator)
        except StopIteration:
            iterator = iter(source)
            return next(iterator)

    return next_batch


def run_hooks(hooks: Sequence[Hook], done: int, log: MetricsLog) -> None:
    """Call each hook due at ``done`` applied steps and write what it returns as notes."""
    for hook in hooks:
        if hook.every <= 0 or done % hook.every:
            continue
        got = hook.run(done)
        if got is None:
            continue
        for fields in [got] if isinstance(got, Mapping) else got:
            log.note(hook.name, **{"step": done, **fields})


class Trainer:
    """A training loop with the scaffolding already attached."""

    def __init__(self, model: Any, optimizer: Any,
                 loss: Callable[[Any, Any], Any], *,
                 out: Path | str = "runs/default",
                 clip_grad_norm: float = 0.0,
                 step: Step | None = None) -> None:
        self.model = model
        self.optimizer = optimizer
        self.out = Path(out).expanduser()
        self.step: Step = step or step_for(model, optimizer, loss,
                                           clip_grad_norm=clip_grad_norm)

    @property
    def framework(self) -> str:
        return str(getattr(self.step, "name", ""))

    # -- checkpoints -----------------------------------------------------
    def save_checkpoint(self, step: int, *, state: CheckpointState,
                        name: str | None = None) -> Path:
        directory = self.out / (name or checkpoint_name(step))
        saved = save(directory, state=state,
                     tensors=self.step.parameters(),
                     optimizer=self.step.optimizer_state() or None,
                     write_tensors=tensor_writer(self.framework))
        if name is None:
            point_latest_at(self.out, saved)
        return saved

    def resume(self) -> CheckpointState | None:
        """Restore weights *and* optimizer state, or return None if there is nothing."""
        latest = find_latest(self.out)
        if latest is None:
            return None
        state = load_state(latest)
        read = tensor_reader(self.framework)
        tensors = load_tensors(latest, read_tensors=read)
        wants_opt = (latest / "optimizer.safetensors").exists()
        opt = load_tensors(latest, read_tensors=read, optimizer=True) if wants_opt else None
        self.step.restore(tensors, opt)
        return state

    def _stop_reason(self, should_stop: Callable[[], bool] | None, max_seconds: float,
                     began: float | None) -> str:
        """Why the loop ends before the next step, or ``""`` when it goes on."""
        if should_stop is not None and should_stop():
            return "stopped"
        if max_seconds > 0 and began is not None and time.monotonic() - began >= max_seconds:
            return "time_limit"
        return ""

    # -- the loop --------------------------------------------------------
    def fit(self, data: Any, *, steps: int,
            batch_size: int = 0,
            schedule: Schedule | float = 1e-3,
            eval_data: Any = None,
            eval_every: int = 0,
            eval_batches: int = 1,
            checkpoint_every: int = 0,
            keep_last: int = 3,
            milestone_every: int = 0,
            resume: bool = True,
            write_checkpoints: bool = True,
            log_every: int = 1,
            max_skipped: int = 50,
            stall_factor: float = 3.0,
            config: dict[str, Any] | None = None,
            on_step: Callable[[int, float], None] | None = None,
            should_stop: Callable[[], bool] | None = None,
            max_seconds: float = 0.0,
            hooks: Sequence[Hook] = (),
            continue_from: int | None = None,
            ) -> TrainReport:
        """Train until ``steps`` steps are done, a stop is asked, or time is up."""
        lr: Schedule = schedule if callable(schedule) else constant(float(schedule))
        next_batch = batches_from(data, batch_size=batch_size)
        next_eval = batches_from(eval_data, batch_size=batch_size) if eval_data is not None else None

        report = TrainReport()
        budget = NonFiniteBudget(max_skipped=max_skipped)
        watchdog = StallWatchdog(factor=stall_factor)
        throughput = Throughput()
        started_at = time.time()
        continuing = continue_from is not None

        self.out.mkdir(parents=True, exist_ok=True)
        with RunLock(self.out), MetricsLog(self.out / "metrics.jsonl",
                                           resume=resume or continuing) as log:
            state = self.resume() if resume and not continuing else None
            start = int(continue_from or 0) if continuing else (state.step if state else 0)
            report.resumed_from = report.steps = start
            best = state.best_metric if state else None

            log.start({"steps": steps, "resumed_from": start,
                       "framework": self.framework, **(config or {})})
            if start >= steps:
                report.stop_reason = "already complete"
                report.seconds = time.time() - started_at
                log.finish(reason=report.stop_reason, step=start)
                return report

            began: float | None = None
            for step in range(start, steps):
                reason = self._stop_reason(should_stop, max_seconds, began)
                if reason:
                    report.stop_reason = reason
                    break
                began = time.monotonic() if began is None else began
                rate = lr(step)
                self.step.learning_rate(rate)

                with StepTimer() as timer:
                    loss, applied = self.step(next_batch(step))

                if not applied:
                    budget.record_skip(step)
                    report.skipped += 1
                    log.note("non-finite loss", step=step, skipped=budget.skipped)
                    if budget.exhausted:
                        log.finish(reason="diverged", step=step)
                        raise TrainingDiverged(
                            f"{budget.skipped} non-finite steps by step {step}; "
                            f"the last was at {budget.last_step}. Stopping rather than "
                            "writing checkpoints of a model that is already NaN.")
                    continue

                if max_seconds > 0 and report.steps == start and timer.elapsed > 0:
                    fits = int(max_seconds / timer.elapsed)
                    if fits < FEW_STEPS:
                        log.note("time limit fits few steps", step=step,
                                 step_seconds=round(timer.elapsed, 3), steps_that_fit=fits,
                                 max_seconds=max_seconds)
                throughput.record(1, timer.elapsed)
                stall = watchdog.record(timer.elapsed)
                if stall:
                    report.stalls.append(f"step {step}: {stall}")
                    log.note(stall, step=step)

                report.steps = step + 1
                report.final_loss = loss
                if on_step is not None:
                    on_step(step, loss)
                if log_every and step % log_every == 0:
                    log.step(step, loss=loss, lr=rate,
                             steps_per_s=round(throughput.per_second, 3))
                    report.history.append({"step": step, "loss": loss, "lr": rate})

                if next_eval is not None and eval_every and (step + 1) % eval_every == 0:
                    scores = [self.step.eval_loss(next_eval(i)) for i in range(eval_batches)]
                    score = sum(scores) / len(scores)
                    log.eval(step, loss=score)
                    if best is None or score < best:
                        best = score
                        report.best_metric = best
                    if write_checkpoints and report.best_metric == score:
                        report.best_checkpoint = self.save_checkpoint(
                            step + 1,
                            state=CheckpointState(step=step + 1, best_metric=best,
                                                  config=dict(config or {})),
                            name="best")

                if checkpoint_every and (step + 1) % checkpoint_every == 0:
                    report.last_checkpoint = self.save_checkpoint(
                        step + 1,
                        state=CheckpointState(step=step + 1, best_metric=best,
                                              config=dict(config or {})))
                    rotate(self.out, keep_last=keep_last,
                           milestone_every=milestone_every)
                run_hooks(hooks, step + 1, log)

            if write_checkpoints:
                report.last_checkpoint = self.save_checkpoint(
                    report.steps,
                    state=CheckpointState(step=report.steps, best_metric=best,
                                          config=dict(config or {})))
                rotate(self.out, keep_last=keep_last,
                       milestone_every=milestone_every)
            report.best_metric = best
            report.seconds = time.time() - started_at
            log.finish(reason=report.stop_reason, step=report.steps,
                       loss=report.final_loss, seconds=round(report.seconds, 2))
        return report
