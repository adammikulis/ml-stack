"""Checking a labelled file before training on it: schema, label balance, duplicates, size, and
whether the training and evaluation sets share cases."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from poolhouse import jsonl
from poolhouse.decide.base import text_of
from poolhouse.decide.cases import Case, case_from

MIN_CASES = 40
"""Under this many cases there are too few to hold out calibration and evaluation splits."""
WARN_CASES = 500
"""Under this many training cases a run checks the pipeline and nothing else."""
MIN_PER_LABEL = 5
DOMINANT = 0.9
MAX_FILE_BYTES = 64 * 1024 * 1024
MAX_CASES = 200_000
MAX_LABELS = 255

_SPACE = re.compile(r"\s+")


class DataError(ValueError):
    """The labelled file cannot be trained on."""


@dataclass(slots=True)
class Findings:
    """What `check` found: ``errors`` stop a run, ``warnings`` do not."""

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    labels: dict[str, int] = field(default_factory=dict)
    kinds: dict[str, int] = field(default_factory=dict)

    def raise_for_errors(self) -> None:
        """Raise `DataError` listing every error."""
        if self.errors:
            raise DataError("; ".join(self.errors))


def load(path: Path | str) -> list[Case]:
    """The cases in a JSONL file, refusing a file over `MAX_FILE_BYTES` or `MAX_CASES` and
    reporting a bad line by file and line number."""
    where = Path(path)
    if where.stat().st_size > MAX_FILE_BYTES:
        raise DataError(f"{where} is over {MAX_FILE_BYTES >> 20} MB")
    out: list[Case] = []
    for number, row in enumerate(jsonl.rows(where), start=1):
        try:
            out.append(case_from(row))
        except ValueError as exc:
            raise DataError(f"{where}, case {number}: {exc}") from None
        if len(out) > MAX_CASES:
            raise DataError(f"{where} holds more than {MAX_CASES} cases")
    return out


def _norm(text: str) -> str:
    return _SPACE.sub(" ", text.strip().lower())


def key_of(case: Case) -> str:
    """A digest of what the model sees: the question, the state and the option names, ignoring
    case, spacing and option order."""
    body = json.dumps([_norm(case.question), _norm(text_of(case.state)),
                       sorted(_norm(o.name) for o in case.options)], ensure_ascii=False)
    return hashlib.sha256(body.encode()).hexdigest()


def check(cases: Sequence[Case], *, label: str = "data") -> Findings:
    """Findings for one set of cases."""
    f = Findings()
    if not cases:
        f.errors.append(f"{label} has no cases")
        return f
    f.labels = dict(Counter(c.label for c in cases))
    f.kinds = dict(Counter(c.kind_name for c in cases))
    if len(cases) < MIN_CASES:
        f.errors.append(f"{label} has {len(cases)} cases; at least {MIN_CASES} are needed to "
                        "hold out calibration and evaluation splits")
    elif len(cases) < WARN_CASES:
        f.warnings.append(f"{label} has {len(cases)} cases: a run checks the pipeline, it does "
                          f"not support a quality claim (aim for {WARN_CASES}+)")
    if len(f.labels) > MAX_LABELS:
        f.errors.append(f"{len(f.labels)} distinct labels; the limit is {MAX_LABELS}")
    top, count = max(f.labels.items(), key=lambda kv: kv[1])
    if len(f.labels) < 2:
        f.errors.append(f"every case has the label {top!r}: nothing to learn")
    elif count / len(cases) > DOMINANT:
        f.warnings.append(f"label {top!r} is {count / len(cases):.0%} of {label}: always "
                          "answering it scores that well")
    rare = sorted(k for k, n in f.labels.items() if n < MIN_PER_LABEL)
    if rare and len(f.labels) >= 2:
        f.warnings.append(f"labels with under {MIN_PER_LABEL} cases: {rare[:8]}")
    f.warnings.extend(_duplicates(cases, label, f))
    ids = Counter(c.id for c in cases if c.id)
    clash = sorted(i for i, n in ids.items() if n > 1)
    if clash:
        f.warnings.append(f"{len(clash)} ids appear more than once in {label}: {clash[:5]}")
    return f


def _duplicates(cases: Sequence[Case], label: str, f: Findings) -> list[str]:
    seen: dict[str, str] = {}
    exact = 0
    for c in cases:
        k = key_of(c)
        if k in seen and seen[k] != c.label:
            f.errors.append(f"{label} gives the same case two labels ({seen[k]!r} and "
                            f"{c.label!r}): {c.id or c.question[:40]!r}")
        elif k in seen:
            exact += 1
        seen.setdefault(k, c.label)
    return [f"{exact} cases in {label} repeat an earlier one"] if exact else []


def leaks(train: Sequence[Case], held: Sequence[Case], *, held_label: str = "evaluation"
          ) -> list[str]:
    """Where ``held`` shares a case or a group with ``train``: the same question, state and
    options, or one group name in both. Empty when the sets are apart."""
    seen = {key_of(c) for c in train}
    same = [c for c in held if key_of(c) in seen]
    out = []
    if same:
        out.append(f"{len(same)} of {len(held)} {held_label} cases are also in the training "
                   f"data (first: {same[0].id or same[0].question[:40]!r})")
    groups = {c.group for c in train if c.group}
    shared = sorted({c.group for c in held if c.group} & groups)
    if shared:
        out.append(f"{len(shared)} groups are in both the training and {held_label} sets "
                   f"(first: {shared[0]!r})")
    return out


def raw_rows(path: Path | str) -> list[Any]:
    """The parsed lines of a JSONL file, unchecked."""
    return list(jsonl.rows(path))
