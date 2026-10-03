"""Reviewed trajectories preserve model inputs and episode split groups."""

import json

import pytest

from ml_stack.decide.cases import read_cases
from ml_stack.gym.recordings import export_reviewed


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
