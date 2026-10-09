"""The token rules of the rename script: each old spelling has its own rule, the kept spellings
survive, and running the rule over its own output changes nothing."""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("rename_to_poolhouse", ROOT / "scripts" / "rename_to_poolhouse.py")
rename = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rename)


@pytest.mark.parametrize(("old", "new"), [
    ("from ml_stack.home import state", "from poolhouse.home import state"),
    ("pip install ml-stack", "pip install poolhouse"),
    ("ml-stack-workspace inbox", "poolhouse-workspace inbox"),
    ("ML_STACK_HOME and MLSTACK_GUARD", "POOLHOUSE_HOME and POOLHOUSE_GUARD"),
    ("ML_STACK", "POOLHOUSE"),
    ("X-ML-Stack-Token", "X-Poolhouse-Token"),
    ("x-ml-stack-ui", "x-poolhouse-ui"),
    ("window.mlStackNative", "window.poolhouseNative"),
    ("Remove-MlStack and MLStackLlamaBuild", "Remove-Poolhouse and PoolhouseLlamaBuild"),
    ("ML stack, ml stack and ML-Stack", "Poolhouse, poolhouse and Poolhouse"),
    ("Poolside, poolside and POOLSIDE", "Poolhouse, poolhouse and POOLHOUSE"),
    ("poolside-node and --poolside-navy", "poolhouse-node and --poolhouse-navy"),
    ("com.ml-stack.traind", "com.poolhouse.traind"),
    ("~/.ml-stack/llama.cpp", "~/.poolhouse/llama.cpp"),
    ("an ml-stack install", "a poolhouse install"),
    ("tmp/mlstack-shims-1", "tmp/poolhouse-shims-1"),
])
def test_each_old_spelling_has_a_rule(old, new):
    assert rename.rewrite(old) == new


@pytest.mark.parametrize("kept", [
    "https://github.com/adammikulis/ml-stack/releases",
    "https://raw.githubusercontent.com/adammikulis/ml-stack/main/install.sh",
    "git clone git@github.com:adammikulis/ml-stack.git",
    "~/Documents/repos/ml-stack",
    "package com.mlstack.companion;",
    "src/main/java/com/mlstack/companion",
    'b"ml-stack/keystore/v1"',
    'b"ml-stack/activity/v1"',
    'b"ml-stack/v1"',
    'b"ml-stack-join-v1/"',
    'b"ml-stack-memory-key-id"',
])
def test_the_kept_spellings_survive(kept):
    assert rename.rewrite(kept) == kept


def test_a_kept_spelling_beside_a_renamed_one_keeps_only_itself():
    text = 'url = "https://github.com/adammikulis/ml-stack"  # ml-stack-bench\n'
    assert rename.rewrite(text) == 'url = "https://github.com/adammikulis/ml-stack"  # poolhouse-bench\n'


def test_the_rules_are_idempotent():
    text = "ml_stack ml-stack ML_STACK MLSTACK_X Poolside poolside X-ML-Stack-Id adammikulis/ml-stack"
    once = rename.rewrite(text)
    assert rename.rewrite(once) == once
    assert "ml_stack" not in once.replace("adammikulis/ml-stack", "")


def test_a_document_capitalises_the_name_only_where_it_is_a_word():
    text = "ml-stack runs here. Run `ml-stack-serve up` or ml-stack's daemon; see [ml-stack](docs/ml-stack.md).\n"
    assert rename.rewrite(text, prose=True) == (
        "Poolhouse runs here. Run `poolhouse-serve up` or Poolhouse's daemon; see [Poolhouse](docs/poolhouse.md).\n")


def test_a_fenced_block_keeps_code_in_lower_case():
    text = "```\nml-stack --list\n```\nml-stack lists them.\n"
    assert rename.rewrite(text, prose=True) == "```\npoolhouse --list\n```\nPoolhouse lists them.\n"


def test_a_message_that_opens_with_the_name_and_a_verb_is_a_sentence():
    assert rename.rewrite('print("ml-stack will ask")') == 'print("Poolhouse will ask")'
    assert rename.rewrite('warn(f"ml-stack security: {exc}")') == 'warn(f"poolhouse security: {exc}")'
    assert rename.rewrite("run `ml-stack runtime ensure`") == "run `poolhouse runtime ensure`"


def test_paths_follow_the_same_names():
    assert rename.target("src/ml_stack/home.py") == "src/poolhouse/home.py"
    assert rename.target("app/poolside-node/Cargo.toml") == "app/poolhouse-node/Cargo.toml"
    assert rename.target("packaging/ml-stack.spec") == "packaging/poolhouse.spec"
    assert rename.target("app/android/src/main/java/com/mlstack/companion/Invite.java") == (
        "app/android/src/main/java/com/mlstack/companion/Invite.java")


def test_history_and_the_rename_s_own_records_are_not_moved_or_rewritten(tmp_path):
    assert rename.target("CHANGELOG.md") == "CHANGELOG.md"
    for path in ("CHANGELOG.md", "scripts/rename_to_poolhouse.py", "src/poolhouse/legacy.py", "src/poolhouse/migrate.py"):
        assert rename.changed(ROOT, path) is None


def test_the_files_only_a_person_edits_are_listed_and_not_rewritten():
    for path in ("src/ml_stack/workspace/person_hook.py", "src/poolhouse/workspace/person_store.py",
                 "scripts/hooks/claude-bash-guard", "scripts/hooks/rules_loader.py", ".claude/settings.json"):
        assert rename.PROTECTED.search(path), path
    assert not rename.PROTECTED.search("src/ml_stack/workspace/personnel.py")


def test_a_run_over_a_scratch_repository_renames_paths_and_text_and_then_changes_nothing(tmp_path):
    def git(*args):
        subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)

    git("init", "-q")
    (tmp_path / "src" / "ml_stack").mkdir(parents=True)
    (tmp_path / "src" / "ml_stack" / "a.py").write_text("from ml_stack import b\n", encoding="utf-8")
    (tmp_path / "blob.bin").write_bytes(b"\xff\xfe ml_stack \x00")
    (tmp_path / "CHANGELOG.md").write_text("ml-stack 0.1\n", encoding="utf-8")
    git("add", "src/ml_stack/a.py", "blob.bin", "CHANGELOG.md")

    assert rename.main(["--root", str(tmp_path), "--apply"]) == 0
    assert (tmp_path / "src" / "poolhouse" / "a.py").read_text(encoding="utf-8") == "from poolhouse import b\n"
    assert not (tmp_path / "src" / "ml_stack").exists()
    assert (tmp_path / "blob.bin").read_bytes() == b"\xff\xfe ml_stack \x00"
    assert (tmp_path / "CHANGELOG.md").read_text(encoding="utf-8") == "ml-stack 0.1\n"

    before = subprocess.run(["git", "status", "--short"], cwd=tmp_path, capture_output=True, text=True).stdout
    assert rename.main(["--root", str(tmp_path), "--apply"]) == 0
    assert subprocess.run(["git", "status", "--short"], cwd=tmp_path, capture_output=True, text=True).stdout == before


def test_a_dry_run_changes_nothing(tmp_path, capsys):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / "a.txt").write_text("ml-stack\n", encoding="utf-8")
    subprocess.run(["git", "add", "a.txt"], cwd=tmp_path, check=True)
    assert rename.main(["--root", str(tmp_path)]) == 0
    assert "rewrite a.txt" in capsys.readouterr().out
    assert (tmp_path / "a.txt").read_text(encoding="utf-8") == "ml-stack\n"
