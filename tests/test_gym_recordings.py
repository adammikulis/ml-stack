"""Reviewed trajectories preserve model inputs and episode split groups."""

import json

import pytest

from poolhouse.decide.cases import read_cases
from poolhouse.gym.recordings import export_reviewed


def test_export_uses_reviewed_label_and_previous_observation(tmp_path):
    trajectory = tmp_path / "trajectory.jsonl"
    reviews = tmp_path / "reviews.jsonl"
    output = tmp_path / "cases.jsonl"
    trajectory.write_text(json.dumps({"environment": "car", "actions": ["brake", "go"],
                                      "transition": {"episode_id": 3, "sequence": 5,
                                                     "observation": [1], "next_observation": [2]}}) + "\n")
    reviews.write_text('{"episode_id":3,"sequence":5,"label":"brake"}\n')
    assert export_reviewed(trajectory, reviews, output) == 1
    case = read_cases(output)[0]
    assert case.label == "brake"
    assert case.state == [1]
    assert case.group == f"{tmp_path.name}:3"


def test_export_rejects_review_for_absent_transition(tmp_path):
    trajectory = tmp_path / "trajectory.jsonl"
    reviews = tmp_path / "reviews.jsonl"
    trajectory.write_text("")
    reviews.write_text('{"episode_id":3,"sequence":5,"label":"brake"}\n')
    with pytest.raises(ValueError, match="absent"):
        export_reviewed(trajectory, reviews, tmp_path / "cases.jsonl")


def test_export_preserves_exact_named_controller_state(tmp_path):
    trajectory, reviews, output = (tmp_path / name for name in ("trajectory.jsonl", "reviews.jsonl", "cases.jsonl"))
    state = {"ego": {"speed_km_h": 2.}, "native_observation": [1]}
    trajectory.write_text(json.dumps({"environment": "car", "actions": ["brake", "go"],
                                      "decision": {"state": state},
                                      "transition": {"episode_id": 1, "sequence": 2,
                                                     "observation": [1], "next_observation": [2]}}) + "\n")
    reviews.write_text('{"episode_id":1,"sequence":2,"label":"brake"}\n')
    export_reviewed(trajectory, reviews, output)
    assert read_cases(output)[0].state == state
