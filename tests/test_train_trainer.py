"""The training loop, against real models on whichever frameworks are installed.

Nothing is stubbed. A fake model with a fake optimizer would pass every test here and
prove nothing, because the failures this exists to prevent are framework-shaped: an MLX
loop that never calls ``mx.eval`` and therefore trains nothing while looking busy, a
resume that restores weights but not Adam's moments, a NaN that reaches every parameter
before anyone notices.

The models are tiny and the task is linear regression with noise, so "the loss went
down" is a fact with a known answer rather than a hope.
"""

from __future__ import annotations

import pytest
from ml_stack.testing import needs_mlx, needs_torch
from ml_stack.train import (
    RunLock,
    RunLockError,
    Trainer,
    TrainingDiverged,
    batches_from,
    constant,
    find_latest,
    is_valid,
    load_state,
    read,
    warmup_cosine,
)


# -- torch fixtures ------------------------------------------------------
def _torch_problem(seed: int = 0, n: int = 512, d: int = 8):
    import torch

    torch.manual_seed(seed)
    x = torch.randn(n, d)
    w = torch.randn(d, 1)
    y = x @ w + 0.1 * torch.randn(n, 1)
    return x, y


def _torch_setup(lr: float = 1e-3, seed: int = 0):
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(8, 32), nn.ReLU(), nn.Linear(32, 1))
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    def loss(m, batch):
        xs, ys = batch
        return ((m(xs) - ys) ** 2).mean()

    return model, opt, loss


def _batcher(x, y, size: int = 64):
    n = len(x) - size

    def batches(step: int):
        i = (step * size) % max(1, n)
        return x[i:i + size], y[i:i + size]

    return batches


# -- it actually trains --------------------------------------------------
@needs_torch
def test_the_loss_actually_goes_down_on_torch(tmp_path):
    x, y = _torch_problem()
    model, opt, loss = _torch_setup()

    report = Trainer(model, opt, loss, out=tmp_path / "run").fit(
        _batcher(x, y), steps=200, schedule=warmup_cosine(3e-3, total_steps=200,
                                                          warmup_steps=20))

    assert report.steps == 200
    assert report.final_loss < report.history[0]["loss"] / 5, report.history[0]


@needs_mlx
def test_the_loss_actually_goes_down_on_mlx(tmp_path):
    """MLX is lazy: a loop that never evaluates builds a graph of every step it has
    taken, trains nothing, and looks perfectly busy while doing it."""
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim

    mx.random.seed(0)
    x = mx.random.normal((512, 8))
    y = x @ mx.random.normal((8, 1)) + 0.1 * mx.random.normal((512, 1))
    model = nn.Sequential(nn.Linear(8, 32), nn.ReLU(), nn.Linear(32, 1))

    def loss(m, batch):
        xs, ys = batch
        return mx.mean((m(xs) - ys) ** 2)

    report = Trainer(model, optim.Adam(learning_rate=1e-3), loss,
                     out=tmp_path / "run").fit(
        _batcher(x, y), steps=200,
        schedule=warmup_cosine(3e-3, total_steps=200, warmup_steps=20))

    assert report.final_loss < report.history[0]["loss"] / 5, report.history[0]


@needs_torch
def test_the_framework_is_detected_from_the_model(tmp_path):
    """A caller who has to name their framework can name it wrongly, and the error then
    arrives deep inside a backward pass looking like a bug in the loss."""
    model, opt, loss = _torch_setup()
    assert Trainer(model, opt, loss, out=tmp_path / "r").step.name == "torch"


def test_a_model_from_no_known_framework_is_refused(tmp_path):
    with pytest.raises(TypeError, match="which framework"):
        Trainer(object(), None, lambda m, b: 0.0, out=tmp_path / "r")


# -- resuming ------------------------------------------------------------
@needs_torch
class TestResume:
    def test_steps_is_a_total_not_a_remainder(self, tmp_path):
        """Re-running the same call after a crash must finish the run, not double it."""
        x, y = _torch_problem()
        model, opt, loss = _torch_setup()
        Trainer(model, opt, loss, out=tmp_path / "run").fit(
            _batcher(x, y), steps=100, checkpoint_every=50)

        model2, opt2, loss2 = _torch_setup(seed=1)
        again = Trainer(model2, opt2, loss2, out=tmp_path / "run").fit(
            _batcher(x, y), steps=250, checkpoint_every=50)

        assert again.resumed_from == 100
        assert again.steps == 250

    def test_a_finished_run_re_run_does_nothing(self, tmp_path):
        x, y = _torch_problem()
        model, opt, loss = _torch_setup()
        Trainer(model, opt, loss, out=tmp_path / "run").fit(
            _batcher(x, y), steps=60, checkpoint_every=30)

        model2, opt2, loss2 = _torch_setup(seed=1)
        again = Trainer(model2, opt2, loss2, out=tmp_path / "run").fit(
            _batcher(x, y), steps=60, checkpoint_every=30)

        assert again.resumed_from == 60
        assert again.steps == 60

    def test_a_resume_restores_the_optimizer_not_just_the_weights(self, tmp_path):
        """Weights without Adam's moments is a warm restart, not a resume, and it shows
        up as a loss spike that gets blamed on the learning rate."""
        x, y = _torch_problem()
        model, opt, loss = _torch_setup()
        Trainer(model, opt, loss, out=tmp_path / "run").fit(
            _batcher(x, y), steps=150, checkpoint_every=150,
            schedule=constant(1e-3))

        model2, opt2, loss2 = _torch_setup(seed=99)
        trainer = Trainer(model2, opt2, loss2, out=tmp_path / "run")
        state = trainer.resume()

        assert state is not None and state.step == 150
        assert opt2.state, "the optimizer came back empty -- this is a warm restart"
        moments = [v for s in opt2.state.values() for k, v in s.items()
                   if k in ("exp_avg", "exp_avg_sq")]
        assert moments and any(float(m.abs().sum()) > 0 for m in moments)

    def test_a_resumed_run_keeps_improving_rather_than_spiking(self, tmp_path):
        x, y = _torch_problem()
        model, opt, loss = _torch_setup()
        first = Trainer(model, opt, loss, out=tmp_path / "run").fit(
            _batcher(x, y), steps=200, checkpoint_every=200, schedule=constant(2e-3))

        model2, opt2, loss2 = _torch_setup(seed=7)
        second = Trainer(model2, opt2, loss2, out=tmp_path / "run").fit(
            _batcher(x, y), steps=400, checkpoint_every=200, schedule=constant(2e-3))

        assert second.history, "no steps ran after the resume"
        assert second.history[0]["loss"] < first.final_loss * 3, (
            f"loss spiked on resume: {first.final_loss} -> {second.history[0]['loss']}")
        assert second.final_loss <= first.final_loss

    def test_resume_can_be_turned_off(self, tmp_path):
        x, y = _torch_problem()
        model, opt, loss = _torch_setup()
        Trainer(model, opt, loss, out=tmp_path / "run").fit(
            _batcher(x, y), steps=60, checkpoint_every=60)

        model2, opt2, loss2 = _torch_setup(seed=3)
        fresh = Trainer(model2, opt2, loss2, out=tmp_path / "run").fit(
            _batcher(x, y), steps=60, checkpoint_every=60, resume=False)

        assert fresh.resumed_from == 0


# -- refusing to keep going ---------------------------------------------
@needs_torch
class TestGuards:
    def test_a_diverged_run_stops_instead_of_checkpointing_nan(self, tmp_path):
        """A NaN reaches every parameter within a step or two, and every checkpoint
        after that point is worthless."""
        x, y = _torch_problem()
        model, opt, loss = _torch_setup()

        with pytest.raises(TrainingDiverged, match="non-finite"):
            Trainer(model, opt, lambda m, b: loss(m, b) * float("nan"),
                    out=tmp_path / "run").fit(
                _batcher(x, y), steps=500, max_skipped=5, checkpoint_every=1)

        assert find_latest(tmp_path / "run") is None, \
            "a checkpoint of a NaN model was written"

    def test_one_bad_batch_is_skipped_rather_than_fatal(self, tmp_path):
        """A single non-finite step happens. A pattern of them is a diverged run, and
        the budget is what tells them apart."""
        x, y = _torch_problem()
        model, opt, loss = _torch_setup()
        bad = {3}

        def sometimes_nan(m, b):
            value = loss(m, b)
            return value * float("nan") if len(bad & {sometimes_nan.step}) else value
        sometimes_nan.step = 0

        def batches(step):
            sometimes_nan.step = step
            return _batcher(x, y)(step)

        report = Trainer(model, opt, sometimes_nan, out=tmp_path / "run").fit(
            batches, steps=50, max_skipped=10)

        assert report.skipped == 1
        assert report.steps == 50

    def test_two_runs_cannot_share_an_output_directory(self, tmp_path):
        """They overwrite each other's checkpoints and the survivor is whichever
        process happened to write last."""
        x, y = _torch_problem()
        model, opt, loss = _torch_setup()
        out = tmp_path / "run"
        out.mkdir(parents=True)

        with RunLock(out), pytest.raises(RunLockError):
            Trainer(model, opt, loss, out=out).fit(_batcher(x, y), steps=5)


# -- what it records -----------------------------------------------------
@needs_torch
class TestRecords:
    def test_the_schedule_sets_the_learning_rate_each_step(self, tmp_path):
        x, y = _torch_problem()
        model, opt, loss = _torch_setup()
        seen = []

        report = Trainer(model, opt, loss, out=tmp_path / "run").fit(
            _batcher(x, y), steps=40,
            schedule=warmup_cosine(1e-2, total_steps=40, warmup_steps=10),
            on_step=lambda s, v: seen.append(opt.param_groups[0]["lr"]))

        assert seen[0] < seen[9], "the warmup did not ramp"
        assert seen[-1] < seen[9], "the cosine did not decay"
        assert report.history[0]["lr"] == pytest.approx(seen[0])

    def test_every_step_reaches_the_metrics_file(self, tmp_path):
        x, y = _torch_problem()
        model, opt, loss = _torch_setup()
        Trainer(model, opt, loss, out=tmp_path / "run").fit(_batcher(x, y), steps=30)

        rows = read(tmp_path / "run" / "metrics.jsonl")
        kinds = [r.get("event") for r in rows]
        assert kinds.count("step") == 30
        assert "start" in kinds and "finish" in kinds

    def test_the_best_checkpoint_tracks_the_best_score_not_the_last(self, tmp_path):
        x, y = _torch_problem()
        model, opt, loss = _torch_setup()

        report = Trainer(model, opt, loss, out=tmp_path / "run").fit(
            _batcher(x, y), steps=120, eval_data=_batcher(x, y),
            eval_every=40, checkpoint_every=40)

        assert report.best_checkpoint is not None
        assert is_valid(report.best_checkpoint)
        assert load_state(report.best_checkpoint).best_metric == report.best_metric

    def test_a_checkpoint_is_written_even_with_no_checkpoint_every(self, tmp_path):
        """Training for an hour and saving nothing is not a defensible default."""
        x, y = _torch_problem()
        model, opt, loss = _torch_setup()
        report = Trainer(model, opt, loss, out=tmp_path / "run").fit(
            _batcher(x, y), steps=20)

        assert report.last_checkpoint is not None
        assert is_valid(report.last_checkpoint)


# -- batching ------------------------------------------------------------
class TestBatchesFrom:
    def test_a_callable_is_left_alone(self):
        def fn(step):
            return step

        assert batches_from(fn) is fn

    def test_a_sequence_is_sliced_and_wraps(self):
        rows = list(range(10))
        take = batches_from(rows, batch_size=4)
        assert take(0) == [0, 1, 2, 3]
        assert len(take(2)) == 4, "a short final batch triggers a recompile every epoch"

    def test_an_iterable_is_cycled_rather_than_exhausted(self):
        """A run is specified in steps, not epochs. Running out of data mid-run and
        stopping silently would make `steps=10_000` mean something else."""
        take = batches_from([1, 2, 3])
        assert [take(i) for i in range(5)] == [1, 2, 3, 1, 2]
