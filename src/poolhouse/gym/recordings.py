"""Reviewed simulator trajectories as labelled decision datasets."""

from pathlib import Path

from poolhouse import jsonl
from poolhouse.decide.cases import Case, write_cases
from poolhouse.decide.types import options_of


def export_reviewed(trajectory, reviews, output):
    """Export explicitly labelled transitions with episode split groups."""
    labels = {}
    for row in jsonl.rows(Path(reviews)):
        key = int(row["episode_id"]), int(row["sequence"])
        if key in labels:
            raise ValueError(f"Duplicate review for episode/sequence {key}")
        labels[key] = str(row["label"])
    cases = []
    session = Path(trajectory).parent.name
    for row in jsonl.rows(Path(trajectory)):
        transition = row["transition"]
        key = int(transition["episode_id"]), int(transition["sequence"])
        if key not in labels:
            continue
        cases.append(Case(question="Choose the next safe environment action",
                          state=row.get("decision", {}).get("state", transition["observation"]),
                          options=options_of(row["actions"]),
                          label=labels.pop(key), id=f"{session}:{key[0]}:{key[1]}",
                          group=f"{session}:{key[0]}", tags=("gym", str(row["environment"]))))
    if labels:
        raise ValueError("Reviews reference transitions absent from the trajectory")
    if not cases:
        raise ValueError("No reviewed transitions were selected")
    return write_cases(output, cases)
