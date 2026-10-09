"""The commit-type chart parses real git history and keeps every bucket's shares summing to one.

The history comes from a real temporary git repository, not from mocks.
"""

from __future__ import annotations

import datetime as dt
import importlib.util
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DAY = dt.date(2026, 3, 2)  # a Monday


def chart():
    spec = importlib.util.spec_from_file_location("commit_types_chart", ROOT / "scripts" / "commit_types_chart.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def git(repo: Path, *args: str, date: str | None = None) -> None:
    env = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.invalid",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.invalid", "PATH": "/usr/bin:/bin:/usr/local/bin"}
    if date:
        env.update(GIT_AUTHOR_DATE=f"{date}T12:00:00+0000", GIT_COMMITTER_DATE=f"{date}T12:00:00+0000")
    subprocess.run(["git", *args], cwd=repo, env=env, check=True, capture_output=True)


def make_repo(tmp_path: Path, commits: list[tuple[str, str]]) -> Path:
    """A repository with one commit per (ISO date, subject), on branch `main`."""
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q", "-b", "main")
    for index, (date, subject) in enumerate(commits):
        (repo / "f.txt").write_text(str(index))
        git(repo, "add", "f.txt")
        git(repo, "commit", "-q", "-m", subject, date=date)
    return repo


def test_subjects_map_to_types():
    module = chart()
    cases = {
        "feat: add a thing": "feat",
        "fix(api): handle empty input": "fix",
        "Fix!: breaking": "fix",
        "chore: gates cache works": "chore",
        "test: cover parser": "test",
        "docs: explain it": "docs",
        "perf: faster scan": "perf",
        "refactor: split module": "refactor",
        "Merge branch '0.2dev' into x": "merge",
        "chore: merge main": "merge",
        "Update the readme": "other",
        "wip": "other",
    }
    for subject, expected in cases.items():
        assert module.commit_type(subject) == expected, subject


def test_every_bucket_sums_to_one_before_and_after_smoothing(tmp_path):
    module = chart()
    dates = ["2026-01-05", "2026-01-06", "2026-02-10", "2026-02-11", "2026-03-20", "2026-04-01"]
    subjects = ["feat: a", "fix: b", "feat: c", "chore: d", "docs: e", "Merge x"]
    repo = make_repo(tmp_path, list(zip(dates, subjects, strict=True)))
    commits = module.read_commits(repo, "main", None)
    for unit in ("day", "week", "month"):
        starts, counts = module.count_buckets(commits, unit)
        filled = module.fill_empty(module.to_shares(counts))
        for row in filled:
            assert sum(row) == pytest.approx(1.0, abs=1e-9), unit
        for window in (1, 2, 3, 5):
            _, rows = module.build_matrix(commits, unit, window)
            assert len(rows) == len(starts)
            for row in rows:
                assert sum(row) == pytest.approx(1.0, abs=1e-9), (unit, window)


def test_an_empty_middle_bucket_is_interpolated(tmp_path):
    module = chart()
    repo = make_repo(tmp_path, [
        (DAY.isoformat(), "feat: one"),
        ((DAY + dt.timedelta(days=14)).isoformat(), "fix: two"),
    ])
    commits = module.read_commits(repo, "main", None)
    starts, counts = module.count_buckets(commits, "week")
    assert len(starts) == 3
    assert sum(counts[1].values()) == 0
    rows = module.fill_empty(module.to_shares(counts))
    feat, fix = module.TYPES.index("feat"), module.TYPES.index("fix")
    assert rows[1][feat] == pytest.approx(0.5)
    assert rows[1][fix] == pytest.approx(0.5)


def test_the_script_writes_a_png(tmp_path):
    pytest.importorskip("matplotlib")
    repo = make_repo(tmp_path, [("2026-01-05", "feat: a"), ("2026-01-09", "fix: b"), ("2026-02-02", "docs: c")])
    out = tmp_path / "chart.png"
    assert chart().main(["--branch", "main", "--out", str(out)], repo=repo) == 0
    data = out.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    assert len(data) > 1000


def white_pixels_in_plot(fig, ax) -> int:
    """White pixels inside the axes box, not counting a 3-pixel band at each edge."""
    import matplotlib.pyplot as plt
    import numpy as np

    fig.canvas.draw()
    rgb = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
    height = rgb.shape[0]
    box = ax.get_window_extent()
    x0, x1 = int(box.x0) + 3, int(box.x1) - 3
    top, bottom = height - int(box.y1) + 3, height - int(box.y0) - 3
    inner = rgb[top:bottom, x0:x1]
    plt.close(fig)
    return int((inner.min(axis=2) >= 250).sum())


def test_the_areas_fill_the_axes_edge_to_edge(tmp_path):
    pytest.importorskip("matplotlib")
    module = chart()
    dates = ["2026-01-05", "2026-01-20", "2026-02-10", "2026-03-01"]
    repo = make_repo(tmp_path, list(zip(dates, ["feat: a", "fix: b", "docs: c", "chore: d"], strict=True)))
    commits = module.read_commits(repo, "main", None)
    starts, rows = module.build_matrix(commits, "week", 3)
    fig = module.draw(starts, rows, "t", "week", False)
    assert white_pixels_in_plot(fig, fig.axes[0]) == 0
