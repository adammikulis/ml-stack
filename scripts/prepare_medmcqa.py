"""Convert a pinned MedMCQA snapshot to the decision-model JSONL format."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from ml_stack import files, worktreerules
from ml_stack.safenames import safe_join

REPOSITORY = "openlifescienceai/medmcqa"
REVISION = "91c6572c454088bf71b679ad90aa8dffcd0d5868"
LETTERS = ("A", "B", "C", "D")


def convert(row: dict[str, Any]) -> dict[str, Any] | None:
    """Convert one labelled, single-answer row; return None for unsupported rows."""
    if row.get("choice_type") != "single":
        return None
    question = row.get("question")
    choices = [row.get(key) for key in ("opa", "opb", "opc", "opd")]
    answer = row.get("cop")
    case_id = row.get("id")
    if (not isinstance(question, str) or not question.strip()
            or any(not isinstance(choice, str) or not choice.strip() for choice in choices)
            or not isinstance(case_id, str) or not case_id
            or not isinstance(answer, int) or isinstance(answer, bool) or answer not in range(4)):
        return None
    return {"id": case_id, "group": case_id, "kind": "choice",
            "question": "Which option correctly answers this medical exam question?",
            "state": question.strip(),
            "options": dict(zip(LETTERS, (choice.strip() for choice in choices))),
            "label": LETTERS[answer]}


def convert_split(snapshot: Path, split: str, output: Path) -> tuple[int, str]:
    """Write one split as JSONL and return its count and SHA-256."""
    checked = safe_join(output.parent, output.name)
    if output.is_symlink() or worktreerules.checkouts(checked) is not None:
        raise ValueError("generated cases must be outside a Git checkout and cannot replace links")
    output = checked
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise RuntimeError("install pyarrow to convert the cached Parquet files") from exc
    parts = sorted((snapshot / "data").glob(f"{split}-*.parquet"))
    if not parts:
        raise FileNotFoundError(f"no Parquet files for {split} under {snapshot / 'data'}")
    rows = (row for part in parts for row in parquet.read_table(part).to_pylist())
    digest = hashlib.sha256()
    count = 0
    with files.writing(output) as temporary:
        with temporary.open("wb") as stream:
            for row in rows:
                case = convert(row)
                if case is None:
                    continue
                line = (json.dumps(case, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
                stream.write(line)
                digest.update(line)
                count += 1
        if count < 40:
            raise ValueError(f"only {count} supported cases in {split}; expected at least 40")
    return count, digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path, help="cached dataset snapshot at the pinned revision")
    parser.add_argument("out_dir", type=Path, help="cache directory for generated JSONL files")
    args = parser.parse_args()
    if args.snapshot.name != REVISION or not (args.snapshot / "README.md").is_file():
        parser.error(f"snapshot must be {REPOSITORY} at revision {REVISION}")
    if worktreerules.checkouts(args.out_dir) is not None:
        parser.error("generated cases must be outside a Git checkout")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "validation"):
        count, digest = convert_split(args.snapshot, split, args.out_dir / f"{split}.jsonl")
        print(f"{split}: {count} cases sha256={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
