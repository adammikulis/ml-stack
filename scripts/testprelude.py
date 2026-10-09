"""What scripts/test works out before admission: the duration estimate, the class, and what to tell the caller."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import testclass
import testhistory
import testslots


def begin(args: argparse.Namespace, rest: list[str], root: Path, reused: set[str] = frozenset()) -> testclass.Verdict:
    """Estimate and classify the run (files the reuse store serves cost nothing), export its class, estimate and history path, and print the lines."""
    history = testhistory.history_path(root)
    budget = testslots.budget()
    workers = min(args.workers, budget) if args.workers > 0 else budget
    verdict = testclass.classify(args.tier, rest, root,
                                 testclass.Basis(testhistory.load(history), workers, frozenset(reused)), args.background)
    os.environ["DEV_TEST_HISTORY"] = str(history)
    os.environ["DEV_TEST_CLASS"] = verdict.klass
    if verdict.estimate_s is not None:
        os.environ["DEV_TEST_ESTIMATE_S"] = f"{verdict.estimate_s * workers:.1f}"
        print(testclass.estimate_line(verdict), flush=True)
    if verdict.reason and verdict.klass == "background":
        print(testclass.notice(verdict), flush=True)
    return verdict
