#!/usr/bin/env python3
"""Draw the share of each commit type over time as a 100% stacked area chart.

    python3 scripts/commit_types_chart.py                 # week buckets, writes commit-types.png
    python3 scripts/commit_types_chart.py --bucket month --smooth 5 --show

Reads the full history of a branch with `git log` (no network) and writes a PNG to --out.
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import re
import subprocess
import sys
from itertools import pairwise
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TYPES = ("feat", "fix", "perf", "refactor", "docs", "test", "chore", "merge", "other")
KNOWN = set(TYPES) - {"other"}
COLOURS = {
    "feat": "#0072B2", "fix": "#D55E00", "perf": "#CC79A7", "refactor": "#E69F00",
    "docs": "#009E73", "test": "#56B4E9", "chore": "#F0E442", "merge": "#332288", "other": "#BBBBBB",
}
PREFIX = re.compile(r"([A-Za-z]+)(?:\([^()]*\))?!?:")
SHORT_SPAN_DAYS = 60
BUCKET_LABEL = {"day": "Day", "week": "Week starting", "month": "Month starting"}


def commit_type(subject: str) -> str:
    """The type of a commit subject: its conventional prefix, `merge` for merges, else `other`."""
    if subject.lower().startswith("merge"):
        return "merge"
    match = PREFIX.match(subject)
    if not match:
        return "other"
    word = match.group(1).lower()
    if word == "chore" and subject[match.end():].lstrip().lower().startswith("merge"):
        return "merge"
    return word if word in KNOWN else "other"


def read_commits(repo: Path, branch: str, since: str | None) -> list[tuple[dt.date, str]]:
    """Every commit on the branch as (date, type), oldest first."""
    command = ["git", "log", branch, "--reverse", "--format=%ad%x1f%s", "--date=short"]
    if since:
        command.append(f"--since={since}")
    output = subprocess.run(command, cwd=repo, capture_output=True, text=True, check=True).stdout
    commits = []
    for line in output.splitlines():
        date_text, _, subject = line.partition("\x1f")
        commits.append((dt.date.fromisoformat(date_text), commit_type(subject)))
    return commits


def pick_bucket(commits: list[tuple[dt.date, str]], requested: str | None) -> str:
    """The requested bucket, else `day` for histories under 60 days and `week` for longer ones."""
    if requested:
        return requested
    span = (commits[-1][0] - commits[0][0]).days
    return "day" if span < SHORT_SPAN_DAYS else "week"


def bucket_start(day: dt.date, unit: str) -> dt.date:
    """The first day of the day, ISO week (Monday) or month that holds `day`."""
    if unit == "week":
        return day - dt.timedelta(days=day.weekday())
    if unit == "month":
        return day.replace(day=1)
    return day


def next_bucket(start: dt.date, unit: str) -> dt.date:
    """The start of the bucket after the one starting on `start`."""
    if unit == "day":
        return start + dt.timedelta(days=1)
    if unit == "week":
        return start + dt.timedelta(days=7)
    return (start.replace(day=28) + dt.timedelta(days=4)).replace(day=1)


def count_buckets(commits: list[tuple[dt.date, str]], unit: str) -> tuple[list[dt.date], list[dict[str, int]]]:
    """Consecutive bucket starts from the first commit to the last, with per-type counts (zero when empty)."""
    counts: dict[dt.date, dict[str, int]] = {}
    for day, kind in commits:
        row = counts.setdefault(bucket_start(day, unit), dict.fromkeys(TYPES, 0))
        row[kind] += 1
    starts = []
    current = bucket_start(commits[0][0], unit)
    last = bucket_start(commits[-1][0], unit)
    while current <= last:
        starts.append(current)
        current = next_bucket(current, unit)
    return starts, [counts.get(start, dict.fromkeys(TYPES, 0)) for start in starts]


def normalise(row: list[float]) -> list[float]:
    """The row scaled so its values sum to 1."""
    total = sum(row)
    return [value / total for value in row]


def to_shares(counts: list[dict[str, int]]) -> list[list[float] | None]:
    """Each type's share of a bucket's commits, in TYPES order; None for a bucket with no commits."""
    rows: list[list[float] | None] = []
    for row in counts:
        total = sum(row.values())
        rows.append([row[kind] / total for kind in TYPES] if total else None)
    return rows


def fill_empty(rows: list[list[float] | None]) -> list[list[float]]:
    """Replace each empty bucket's shares by linear interpolation between its non-empty neighbours.

    The first and last buckets always hold a commit, so every empty bucket has both neighbours.
    """
    known = [i for i, row in enumerate(rows) if row is not None]
    filled = [row if row is not None else [0.0] * len(TYPES) for row in rows]
    for left, right in pairwise(known):
        for i in range(left + 1, right):
            t = (i - left) / (right - left)
            filled[i] = [(1 - t) * a + t * b for a, b in zip(rows[left], rows[right], strict=True)]
    return filled


def smooth(rows: list[list[float]], window: int) -> list[list[float]]:
    """A centred rolling mean of each type over `window` buckets (fewer at the ends), renormalised.

    For an even window the extra bucket falls on the left. A window of 1 leaves the rows unchanged.
    """
    if window <= 1:
        return rows
    left, right = (window - 1) // 2, window // 2
    means = []
    for i in range(len(rows)):
        lo, hi = max(0, i - left), min(len(rows), i + right + 1)
        means.append([sum(row[k] for row in rows[lo:hi]) / (hi - lo) for k in range(len(TYPES))])
    return [normalise(row) for row in means]


def build_matrix(commits: list[tuple[dt.date, str]], unit: str, window: int) -> tuple[list[dt.date], list[list[float]]]:
    """Bucket starts and one row of shares per bucket, each row summing to 1."""
    starts, counts = count_buckets(commits, unit)
    return starts, smooth(fill_empty(to_shares(counts)), window)


def print_table(commits: list[tuple[dt.date, str]]) -> None:
    """Print the commit total of each type and the overall total."""
    totals = dict.fromkeys(TYPES, 0)
    for _, kind in commits:
        totals[kind] += 1
    print(f"{'type':<10}{'commits':>8}")
    for kind in TYPES:
        print(f"{kind:<10}{totals[kind]:>8}")
    print(f"{'total':<10}{len(commits):>8}")


def draw(starts: list[dt.date], rows: list[list[float]], title: str, unit: str, show: bool):
    """Draw the stacked areas (feat at the bottom, other on top) and return the figure."""
    import matplotlib

    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.dates import AutoDateLocator, ConciseDateFormatter, date2num
    from matplotlib.ticker import PercentFormatter

    x = [date2num(dt.datetime.combine(start, dt.time())) for start in starts]
    columns = [[row[k] for row in rows] for k in range(len(TYPES))]
    fig, ax = plt.subplots(figsize=(12, 6.5), dpi=150)
    ax.stackplot(x, columns, colors=[COLOURS[kind] for kind in TYPES], labels=TYPES, linewidth=0)
    ax.set_xlim(x[0], x[-1] if len(x) > 1 else x[0] + 1)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    locator = AutoDateLocator()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(ConciseDateFormatter(locator))
    ax.set_xlabel(BUCKET_LABEL[unit])
    ax.set_ylabel("Share of commits in the bucket")
    ax.set_title(title)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    return fig


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    """The command line."""
    parser = argparse.ArgumentParser(description="Plot commit types over time as a 100%% stacked area chart.")
    parser.add_argument("--branch", default="0.2dev", help="branch to read (default 0.2dev)")
    parser.add_argument("--since", help="only commits after this git date, e.g. 2026-01-01")
    parser.add_argument("--bucket", choices=("day", "week", "month"), help="default: week, or day under 60 days")
    parser.add_argument("--smooth", type=int, default=3, help="centred rolling mean over N buckets; 1 is off")
    parser.add_argument("--out", type=Path, default=Path("commit-types.png"), help="PNG to write")
    parser.add_argument("--show", action="store_true", help="open a window")
    args = parser.parse_args(argv)
    if args.smooth < 1:
        parser.error("--smooth must be 1 or more")
    return args


def main(argv: list[str] | None = None, repo: Path = ROOT) -> int:
    """Read the branch's history under `repo`, print the totals and write the chart."""
    args = parse_args(argv)
    if importlib.util.find_spec("matplotlib") is None:
        print("matplotlib is missing: install it with pip install 'ml-stack[plot]'", file=sys.stderr)
        return 1
    try:
        commits = read_commits(repo, args.branch, args.since)
    except subprocess.CalledProcessError as error:
        print(f"git log {args.branch} failed: {error.stderr.strip()}", file=sys.stderr)
        return 2
    if not commits:
        print(f"no commits on {args.branch}", file=sys.stderr)
        return 1
    unit = pick_bucket(commits, args.bucket)
    starts, rows = build_matrix(commits, unit, args.smooth)
    title = f"Commit types on {args.branch}, {commits[0][0]} to {commits[-1][0]}, {len(commits)} commits"
    print_table(commits)
    fig = draw(starts, rows, title, unit, args.show)
    fig.savefig(args.out, bbox_inches="tight")
    print(f"wrote {args.out.resolve()}")
    if args.show:
        import matplotlib.pyplot as plt

        plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
