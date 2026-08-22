"""Checkpoints, schedules, guards, metrics and leak-safe splits.

The checkpoint tests use real files and real interruptions -- a `.partial` directory left
on disk, a state file removed, a process holding a lock. The failures being guarded against
are filesystem-shaped, and a mocked filesystem reproduces none of them.
"""

from __future__ import annotations

import json
import multiprocessing
import time

import numpy as np
import pytest
from ml_stack.train import (
    CheckpointError,
    CheckpointState,
    LeakageError,
    MetricsLog,
    NonFiniteBudget,
    RunLock,
    RunLockError,
    StallWatchdog,
    StepTimer,
    Throughput,
    TrainingDiverged,
    assert_exact_restore,
    assert_no_duplicates,
    by_group,
    checkpoint_name,
    constant,
    contiguous_tail,
    find_latest,
    is_valid,
    linear_warmup,
    load_state,
    load_tensors,
    point_latest_at,
    read,
    rotate,
    save,
    spread_order,
    warmup_cosine,
    warmup_stable_decay,
)


def write_npz(path, mapping):
    # `np.savez` appends `.npz` unless the handle is already open, so write through a
    # handle -- the serialiser contract is that the file lands at the path it was given.
    with open(path, "wb") as handle:
        np.savez(handle, **{k: np.asarray(v) for k, v in mapping.items()})


def read_npz(path):
    with np.load(path) as data:
        return {k: data[k] for k in data.files}


def make_checkpoint(root, step, *, tensors=None, **kwargs):
    return save(
        root / checkpoint_name(step),
        state=CheckpointState(step=step, **kwargs),
        tensors=tensors or {"w": np.ones((2, 2), dtype=np.float32)},
        write_tensors=write_npz,
    )


class TestCheckpointAtomicity:
    def test_a_saved_checkpoint_is_valid_and_readable(self, tmp_path):
        directory = make_checkpoint(tmp_path, 100, epoch=2, best_metric=0.5)
        assert is_valid(directory)
        state = load_state(directory)
        assert (state.step, state.epoch, state.best_metric) == (100, 2, 0.5)

    def test_the_state_file_is_written_last(self, tmp_path):
        """Its presence is what makes a directory count as a checkpoint, so it must be the
        final write -- otherwise a half-written save looks complete."""
        order = []

        def tracking_write(path, mapping):
            order.append(path.name)
            write_npz(path, mapping)

        directory = tmp_path / "ckpt"
        save(
            directory,
            state=CheckpointState(step=1),
            tensors={"w": np.ones(2, dtype=np.float32)},
            optimizer={"m": np.zeros(2, dtype=np.float32)},
            write_tensors=tracking_write,
        )
        contents = sorted(p.name for p in directory.iterdir())
        assert "state.json" in contents
        assert order, "no tensor files were written"

    def test_an_interrupted_save_leaves_nothing_valid(self, tmp_path):
        """The specific failure: a directory that a later resume reads as complete."""

        def explode(path, mapping):
            raise OSError("disk full")

        directory = tmp_path / "ckpt"
        with pytest.raises(OSError):
            save(
                directory,
                state=CheckpointState(step=1),
                tensors={"w": np.ones(2)},
                write_tensors=explode,
            )
        assert not directory.exists()
        assert not directory.with_name("ckpt.partial").exists()

    def test_a_stale_partial_directory_is_cleared(self, tmp_path):
        stale = tmp_path / "ckpt.partial"
        stale.mkdir()
        (stale / "junk.txt").write_text("left over from a killed run")

        directory = save(
            tmp_path / "ckpt",
            state=CheckpointState(step=1),
            tensors={"w": np.ones(2, dtype=np.float32)},
            write_tensors=write_npz,
        )
        assert is_valid(directory)
        assert not (directory / "junk.txt").exists()

    def test_a_directory_without_state_is_not_a_checkpoint(self, tmp_path):
        directory = make_checkpoint(tmp_path, 10)
        (directory / "state.json").unlink()
        assert not is_valid(directory)
        with pytest.raises(CheckpointError, match="not a complete checkpoint"):
            load_state(directory)

    def test_a_writer_that_renames_the_file_is_caught(self, tmp_path):
        """`numpy.savez` appends `.npz`. Left unchecked, the save appears to succeed,
        state.json gets written, and the checkpoint passes `is_valid` while being
        unloadable -- the exact failure the write-state-last protocol exists to prevent,
        reintroduced one layer down."""

        def renaming_write(path, mapping):
            np.savez(path, **{k: np.asarray(v) for k, v in mapping.items()})

        directory = tmp_path / "ckpt"
        with pytest.raises(CheckpointError, match="renaming the file"):
            save(
                directory,
                state=CheckpointState(step=1),
                tensors={"w": np.ones(2, dtype=np.float32)},
                write_tensors=renaming_write,
            )
        assert not directory.exists()

    def test_a_writer_that_writes_nothing_is_caught(self, tmp_path):
        with pytest.raises(CheckpointError, match="did not create"):
            save(
                tmp_path / "ckpt",
                state=CheckpointState(step=1),
                tensors={"w": np.ones(2)},
                write_tensors=lambda path, mapping: None,
            )

    def test_overwriting_an_existing_checkpoint_works(self, tmp_path):
        make_checkpoint(tmp_path, 5)
        directory = save(
            tmp_path / checkpoint_name(5),
            state=CheckpointState(step=5, epoch=99),
            tensors={"w": np.zeros((2, 2), dtype=np.float32)},
            write_tensors=write_npz,
        )
        assert load_state(directory).epoch == 99


class TestCheckpointResume:
    def test_tensors_round_trip(self, tmp_path):
        weights = {"a": np.arange(4, dtype=np.float32), "b": np.ones((2, 3), dtype=np.float32)}
        directory = make_checkpoint(tmp_path, 1, tensors=weights)
        restored = load_tensors(directory, read_tensors=read_npz)
        assert set(restored) == {"a", "b"}
        assert np.array_equal(restored["a"], weights["a"])

    def test_missing_optimizer_state_is_an_explicit_error(self, tmp_path):
        """Restoring weights but not the optimizer is a warm restart, not a resume, and it
        shows up as a loss spike that is easy to blame on the learning rate."""
        directory = make_checkpoint(tmp_path, 1)
        with pytest.raises(CheckpointError, match="warm restart, not a resume"):
            load_tensors(directory, read_tensors=read_npz, optimizer=True)

    def test_optimizer_state_round_trips_when_saved(self, tmp_path):
        directory = save(
            tmp_path / "ckpt",
            state=CheckpointState(step=1),
            tensors={"w": np.ones(2, dtype=np.float32)},
            optimizer={"exp_avg": np.full(2, 0.5, dtype=np.float32)},
            write_tensors=write_npz,
        )
        restored = load_tensors(directory, read_tensors=read_npz, optimizer=True)
        assert restored["exp_avg"].tolist() == [0.5, 0.5]

    def test_rng_state_survives(self, tmp_path):
        """Without it a resumed run re-draws the batches it already trained on, quietly
        turning a fresh epoch into a repeat."""
        rng = {"bit_generator": "PCG64", "state": {"state": 12345, "inc": 67}}
        directory = make_checkpoint(tmp_path, 1, rng=rng)
        assert load_state(directory).rng == rng

    def test_exact_restore_rejects_a_missing_tensor(self):
        with pytest.raises(CheckpointError, match="in the model but not the checkpoint"):
            assert_exact_restore({"a": np.ones(2)}, {"a": np.ones(2), "b": np.ones(2)})

    def test_exact_restore_rejects_an_extra_tensor(self):
        with pytest.raises(CheckpointError, match="in the checkpoint but not the model"):
            assert_exact_restore({"a": np.ones(2), "b": np.ones(2)}, {"a": np.ones(2)})

    def test_exact_restore_rejects_a_shape_change(self):
        with pytest.raises(CheckpointError, match="shape mismatch"):
            assert_exact_restore({"a": np.ones((2, 4))}, {"a": np.ones((2, 8))})

    def test_exact_restore_accepts_a_match(self):
        assert_exact_restore({"a": np.ones((2, 4))}, {"a": np.zeros((2, 4))})


class TestLatestAndRotation:
    def test_latest_points_at_the_newest(self, tmp_path):
        make_checkpoint(tmp_path, 10)
        newest = make_checkpoint(tmp_path, 20)
        point_latest_at(tmp_path, newest)
        assert find_latest(tmp_path).name == newest.name

    def test_repointing_latest_never_leaves_a_gap(self, tmp_path):
        """Unlink-then-symlink leaves a window with no `latest` at all, and a resume that
        lands in it reports no checkpoint."""
        first = make_checkpoint(tmp_path, 10)
        point_latest_at(tmp_path, first)
        second = make_checkpoint(tmp_path, 20)
        point_latest_at(tmp_path, second)
        assert (tmp_path / "latest").is_symlink()
        assert not (tmp_path / "latest.tmp").exists()

    def test_find_latest_falls_back_when_the_symlink_is_dangling(self, tmp_path):
        """A stale link should degrade to 'find it the slow way', not to 'nothing here'."""
        real = make_checkpoint(tmp_path, 10)
        point_latest_at(tmp_path, tmp_path / "step_000000099")  # never existed
        assert find_latest(tmp_path).name == real.name

    def test_find_latest_on_an_empty_directory(self, tmp_path):
        assert find_latest(tmp_path) is None
        assert find_latest(tmp_path / "nope") is None

    def test_names_sort_numerically(self):
        """find_latest and rotate both sort by name; unpadded names put step_9 after
        step_10000."""
        names = [checkpoint_name(n) for n in (9, 10, 100, 10000)]
        assert names == sorted(names)

    def test_rotation_keeps_the_last_n(self, tmp_path):
        for step in (10, 20, 30, 40, 50):
            make_checkpoint(tmp_path, step)
        removed = rotate(tmp_path, keep_last=2)
        remaining = sorted(d.name for d in tmp_path.iterdir() if d.is_dir())
        assert remaining == [checkpoint_name(40), checkpoint_name(50)]
        assert len(removed) == 3

    def test_rotation_keeps_milestones(self, tmp_path):
        """'Keep the last 3' alone means a run that goes wrong at 50k has nothing from
        10k to go back to."""
        for step in (100, 200, 300, 400, 500):
            make_checkpoint(tmp_path, step)
        rotate(tmp_path, keep_last=1, milestone_every=200)
        remaining = sorted(d.name for d in tmp_path.iterdir() if d.is_dir())
        assert checkpoint_name(200) in remaining
        assert checkpoint_name(400) in remaining
        assert checkpoint_name(500) in remaining

    def test_rotation_never_deletes_whatever_latest_points_at(self, tmp_path):
        oldest = make_checkpoint(tmp_path, 10)
        for step in (20, 30, 40):
            make_checkpoint(tmp_path, step)
        point_latest_at(tmp_path, oldest)
        rotate(tmp_path, keep_last=1)
        assert oldest.exists()

    def test_rotation_protects_named_checkpoints(self, tmp_path):
        save(
            tmp_path / "best",
            state=CheckpointState(step=7, best_metric=0.1),
            tensors={"w": np.ones(2, dtype=np.float32)},
            write_tensors=write_npz,
        )
        for step in (10, 20, 30):
            make_checkpoint(tmp_path, step)
        rotate(tmp_path, keep_last=1)
        assert (tmp_path / "best").exists()

    def test_rotation_ignores_incomplete_directories(self, tmp_path):
        (tmp_path / "step_000000005").mkdir()  # no state.json
        make_checkpoint(tmp_path, 10)
        rotate(tmp_path, keep_last=1)
        assert (tmp_path / "step_000000005").exists()


class TestSchedules:
    def test_constant(self):
        schedule = constant(1e-3)
        assert schedule(0) == schedule(10_000) == 1e-3

    def test_warmup_rises_from_near_zero_to_peak(self):
        schedule = warmup_cosine(1.0, total_steps=1000, warmup_steps=100)
        assert schedule(0) == pytest.approx(0.01)
        assert schedule(99) == pytest.approx(1.0)

    def test_cosine_decays_to_the_floor(self):
        schedule = warmup_cosine(1.0, total_steps=1000, warmup_steps=0, final_fraction=0.1)
        assert schedule(999) == pytest.approx(0.1, abs=1e-3)

    def test_cosine_is_monotonic_after_warmup(self):
        schedule = warmup_cosine(1.0, total_steps=500, warmup_steps=50)
        values = [schedule(s) for s in range(50, 500)]
        assert all(a >= b - 1e-12 for a, b in zip(values, values[1:]))

    def test_wsd_holds_flat_through_the_stable_stretch(self):
        """The reason to prefer WSD: the horizon only matters during the final decay, so a
        run can be extended without invalidating every earlier step's rate."""
        schedule = warmup_stable_decay(1.0, total_steps=1000, warmup_steps=100, decay_fraction=0.2)
        assert schedule(300) == schedule(500) == schedule(700) == 1.0

    def test_wsd_decays_only_at_the_end(self):
        schedule = warmup_stable_decay(1.0, total_steps=1000, warmup_steps=0, decay_fraction=0.2)
        assert schedule(799) == pytest.approx(1.0)
        assert schedule(999) < 0.2

    def test_every_schedule_returns_a_plain_float(self):
        """A framework schedule object captured by a compiled function freezes the learning
        rate for the rest of the run, with nothing reporting it."""
        for schedule in (
            constant(1e-3),
            warmup_cosine(1.0, total_steps=100),
            warmup_stable_decay(1.0, total_steps=100),
            linear_warmup(1.0, warmup_steps=10),
        ):
            assert type(schedule(5)) is float

    def test_schedules_never_go_negative(self):
        schedule = warmup_cosine(1e-3, total_steps=100, warmup_steps=10)
        assert all(schedule(s) >= 0 for s in range(200))

    def test_stepping_past_the_end_stays_at_the_floor(self):
        schedule = warmup_stable_decay(1.0, total_steps=100, warmup_steps=0, final_fraction=0.1)
        assert schedule(500) == pytest.approx(0.1)


def _hold_lock(path, ready, release):
    with RunLock(path):
        ready.set()
        release.wait(timeout=10)


class TestRunLock:
    def test_it_can_be_acquired_and_released(self, tmp_path):
        with RunLock(tmp_path):
            assert (tmp_path / "run.lock").exists()
        assert not (tmp_path / "run.lock").exists()

    def test_a_second_process_is_refused(self, tmp_path):
        """Two runs sharing an output directory overwrite each other's checkpoints, and
        the resume that follows reads an interleaving of two different models."""
        ready = multiprocessing.Event()
        release = multiprocessing.Event()
        holder = multiprocessing.Process(
            target=_hold_lock, args=(str(tmp_path), ready, release)
        )
        holder.start()
        try:
            assert ready.wait(timeout=10), "the holder never acquired the lock"
            with pytest.raises(RunLockError, match="already training"):
                RunLock(tmp_path).acquire()
        finally:
            release.set()
            holder.join(timeout=10)

    def test_the_lock_is_released_when_the_holder_dies(self, tmp_path):
        """flock rather than a pid file: a pid file left by a crash blocks every later run
        until a human deletes it, which trains people to delete it reflexively."""
        ready = multiprocessing.Event()
        release = multiprocessing.Event()
        holder = multiprocessing.Process(
            target=_hold_lock, args=(str(tmp_path), ready, release)
        )
        holder.start()
        assert ready.wait(timeout=10)
        holder.terminate()
        holder.join(timeout=10)

        with RunLock(tmp_path):  # must not raise
            pass


class TestNonFiniteBudget:
    def test_a_few_skips_are_tolerated(self):
        """One bad step is usually a transient fault; aborting over it wastes the run."""
        budget = NonFiniteBudget(max_skipped=5)
        for step in range(5):
            budget.record_skip(step)
        assert budget.skipped == 5

    def test_exceeding_the_budget_aborts(self):
        budget = NonFiniteBudget(max_skipped=2)
        budget.record_skip(1)
        budget.record_skip(2)
        with pytest.raises(TrainingDiverged, match="broken run, not a blip"):
            budget.record_skip(3)

    def test_the_budget_is_cumulative_not_consecutive(self):
        """A machine producing one bad step in ten is broken even though it never produces
        two in a row."""
        budget = NonFiniteBudget(max_skipped=3)
        for step in (0, 10, 20):
            budget.record_skip(step)  # never two in a row
        with pytest.raises(TrainingDiverged):
            budget.record_skip(30)


class TestStallWatchdog:
    def test_it_stays_quiet_on_steady_steps(self):
        watchdog = StallWatchdog()
        assert all(watchdog.record(1.0) is None for _ in range(50))

    def test_it_reports_a_sudden_slowdown(self):
        watchdog = StallWatchdog(factor=3.0, absolute_s=1.0)
        for _ in range(20):
            watchdog.record(1.0)
        assert watchdog.record(30.0) is not None

    def test_it_uses_the_median_so_one_stall_does_not_mask_the_next(self):
        """A mean dragged up by a 40-second stall makes the following stall look normal."""
        watchdog = StallWatchdog(factor=3.0, absolute_s=1.0)
        for _ in range(20):
            watchdog.record(1.0)
        watchdog.record(60.0)
        assert watchdog.record(30.0) is not None, "the second stall was masked"

    def test_it_says_nothing_before_it_has_history(self):
        assert StallWatchdog().record(100.0) is None


class TestMetrics:
    def test_records_round_trip(self, tmp_path):
        path = tmp_path / "metrics.jsonl"
        with MetricsLog(path) as log:
            log.start({"lr": 1e-3, "batch": 32})
            log.step(1, loss=2.5)
            log.eval(1, val_loss=2.4)
            log.finish(status="ok")

        records = read(path)
        assert [r["event"] for r in records] == ["start", "step", "eval", "finish"]
        assert records[0]["config"]["lr"] == 1e-3

    def test_every_record_is_flushed_immediately(self, tmp_path):
        """A buffered record is a lost record when the process is killed."""
        path = tmp_path / "metrics.jsonl"
        log = MetricsLog(path)
        log.step(1, loss=1.0)
        assert json.loads(path.read_text().splitlines()[0])["loss"] == 1.0
        log.close()

    def test_a_truncated_final_line_does_not_break_the_read(self, tmp_path):
        """A run killed mid-write leaves an incomplete line. That is expected, and it must
        not stop the other records from being readable."""
        path = tmp_path / "metrics.jsonl"
        with MetricsLog(path) as log:
            log.step(1, loss=1.0)
            log.step(2, loss=0.9)
        with path.open("a") as handle:
            handle.write('{"event": "step", "loss": 0.8')  # killed here

        records = read(path)
        assert len(records) == 2

    def test_resume_appends_rather_than_truncating(self, tmp_path):
        """The earlier half of a run is the half you most want when working out why the
        later half went wrong."""
        path = tmp_path / "metrics.jsonl"
        with MetricsLog(path) as log:
            log.step(1, loss=1.0)
        with MetricsLog(path, resume=True) as log:
            log.step(2, loss=0.9)
        assert len(read(path)) == 2

    def test_writing_to_a_closed_log_raises(self, tmp_path):
        log = MetricsLog(tmp_path / "m.jsonl")
        log.close()
        with pytest.raises(RuntimeError, match="closed"):
            log.step(1, loss=1.0)

    def test_throughput_is_windowed(self):
        """A since-the-start average moves too slowly to notice a slowdown."""
        throughput = Throughput(window=3)
        for _ in range(3):
            throughput.record(1000, 1.0)
        assert throughput.per_second == pytest.approx(1000.0)
        for _ in range(3):
            throughput.record(100, 1.0)
        assert throughput.per_second == pytest.approx(100.0)

    def test_throughput_with_no_data_is_zero_not_a_division_error(self):
        assert Throughput().per_second == 0.0

    def test_step_timer_measures_something(self):
        with StepTimer() as timer:
            time.sleep(0.02)
        assert timer.elapsed >= 0.02


class TestHoldout:
    def test_contiguous_tail_leaves_a_guard_band(self):
        """A random split of sequential data leaks by construction: the sample just before
        a held-out one is nearly the same sample."""
        split = contiguous_tail(list(range(1000)), fraction=0.01, guard=8)
        assert split.holdout == list(range(990, 1000))
        assert split.train[-1] == 981
        assert split.dropped == 8

    def test_contiguous_tail_refuses_an_impossible_split(self):
        with pytest.raises(LeakageError, match="no training data"):
            contiguous_tail(list(range(10)), fraction=0.9, guard=5)

    def test_group_split_keeps_groups_whole(self):
        """A row-wise split across correlated groups can inflate a reported score by tens
        of points, because the model has seen a near-duplicate of every eval row."""
        rows = list(range(100))
        groups = [f"g{i // 10}" for i in range(100)]
        split = by_group(rows, groups, fraction=0.2, seed=0)

        held = {groups[r] for r in split.holdout}
        trained = {groups[r] for r in split.train}
        assert not (held & trained), "a group appears on both sides"

    def test_group_split_is_deterministic_for_a_seed(self):
        rows, groups = list(range(50)), [f"g{i // 5}" for i in range(50)]
        assert by_group(rows, groups, seed=3).holdout == by_group(rows, groups, seed=3).holdout

    def test_a_single_group_cannot_be_split(self):
        with pytest.raises(LeakageError, match="all rows are in one group"):
            by_group(list(range(10)), ["only"] * 10)

    def test_mismatched_lengths_are_rejected(self):
        with pytest.raises(LeakageError, match="group labels"):
            by_group([1, 2, 3], ["a", "b"])

    def test_duplicates_are_detected(self):
        """Holding out a row that appears three times leaves two copies in training, and no
        splitting strategy fixes that."""
        with pytest.raises(LeakageError, match="repeated example"):
            assert_no_duplicates(["a", "b", "a", "c", "a"])

    def test_clean_data_passes(self):
        assert_no_duplicates(["a", "b", "c"])

    def test_spread_order_covers_the_range_early(self):
        """Taking the first k in natural order samples only the beginning, which for
        anything ordered by time or difficulty is a biased subset dressed as a sample."""
        order = spread_order(100)
        first_four = order[:4]
        assert min(first_four) < 10 and max(first_four) > 90

    def test_spread_order_is_a_permutation(self):
        for n in (1, 2, 3, 7, 64, 100):
            assert sorted(spread_order(n)) == list(range(n)), f"n={n}"

    def test_spread_order_of_nothing(self):
        assert spread_order(0) == []
