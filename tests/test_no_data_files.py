"""The pre-commit hook that keeps graphs, datasets and weights out of this public repository."""

import os
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
HOOK = REPO / "scripts" / "hooks" / "no-data-files"


def run(where: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run([str(HOOK), *args], cwd=where, capture_output=True, text=True)


@pytest.fixture
def staging(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)

    def stage(name: str, content: bytes) -> subprocess.CompletedProcess:
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_bytes(content)
        subprocess.run(["git", "-C", str(tmp_path), "add", name], check=True)
        return run(tmp_path)
    return stage


@pytest.mark.parametrize("name", ["g.lbug", "g.lbug.wal", "k.kgx", "m.gguf", "m.safetensors",
                                  "t.parquet", "notes.jsonl", "chat.db"])
def test_a_data_or_weight_file_is_refused(staging, name):
    done = staging(name, b"x")
    assert done.returncode == 1
    assert name in done.stderr


def test_anything_under_a_data_directory_is_refused(staging):
    assert staging("data/people.json", b"{}").returncode == 1
    assert staging("examples/graphs/g.json", b"{}").returncode == 1


def test_a_file_over_the_size_limit_is_refused(staging):
    assert staging("docs/big.png", b"0" * ((1 << 20) + 1)).returncode == 1


def test_code_and_package_resources_pass(staging):
    assert staging("src/ml_stack/tool.py", b"x = 1\n").returncode == 0
    assert staging("src/ml_stack/data/fit.json", b"{}").returncode == 0


def test_the_tracked_tree_passes():
    done = run(REPO, "--tracked")
    assert done.returncode == 0, done.stderr


def test_pre_commit_runs_it():
    assert '"$here/no-data-files"' in (REPO / "scripts" / "hooks" / "pre-commit").read_text()
    assert os.access(HOOK, os.X_OK)
