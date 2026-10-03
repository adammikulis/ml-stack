"""Native road task compliance changes reward once per checkpoint."""

from types import SimpleNamespace

from ml_stack.gym.road_rules import StopCheckpoint, front_progress


def test_required_stop_hold_resets_on_motion_and_rewards_once():
    stop = StopCheckpoint(50, [1, 2], 0.)
    assert stop.update(48, 0, .6) == 0
    assert stop.update(48, 1, .1) == 0
    assert stop.held == 0
    assert stop.update(47, 0, 1.) == 5
    assert stop.update(47, 0, 1.) == 0
    assert stop.update(51, 2, .1) == 0
    assert stop.completed and stop.passed and not stop.violated


def test_crossing_stop_without_hold_penalizes_once():
    stop = StopCheckpoint(50, [1, 2], 0.)
    assert stop.update(51, 3, .1) == -10
    assert stop.update(52, 3, .1) == 0
    state = stop.state(52)
    assert state["violated"]
    assert state["distance_m"] == -2


def test_front_bumper_crossing_is_violation_before_center_crosses():
    native = SimpleNamespace(agent=SimpleNamespace(LENGTH=4., navigation=SimpleNamespace(travelled_length=49.)))
    stop = StopCheckpoint(50, [1, 2], 0.)
    assert stop.update(front_progress(native), 2., .1) == -10
    assert stop.violated


def test_stop_outside_zone_cannot_satisfy_hold():
    stop = StopCheckpoint(50, [1, 2], 0.)
    assert stop.update(42, 0, 2.) == 0
    assert stop.held == 0 and not stop.completed
    assert stop.update(48, 0, .6) == 0
    assert stop.update(46, 0, .6) == 0
    assert stop.held == 0 and not stop.completed
