"""Taking it off a machine.

An uninstall is a path someone walks once, so there is no second chance to notice it
took the wrong thing. What a person made -- their models, their datasets -- must
survive unless they say otherwise.
"""

from __future__ import annotations

import pytest
from ml_stack.fleet.uninstall import plan, remove


@pytest.fixture
def installed(tmp_path):
    """A machine with everything an install leaves behind."""
    home = tmp_path / ".ml-stack"
    root = home / "traind"
    for name in ("chats", "files", "models", "env", "llama"):
        (root / name).mkdir(parents=True)
    (home / "cluster.key").write_text("k" * 44)
    (home / "cluster.group").write_text("home")
    (home / "traind.log").write_text("log")
    (root / "settings.json").write_text("{}")
    (root / "token").write_text("t")
    (root / "models" / "big.gguf").write_bytes(b"m" * 4096)
    (root / "files" / "dataset.jsonl").write_bytes(b"d" * 512)
    (root / "chats" / "a.json").write_text('{"id": "a"}')
    (root / "env" / "pyvenv.cfg").write_text("home = /x")
    (root / "llama" / "llama-server").write_bytes(b"x" * 128)
    return root, home / "cluster.key"


class TestWhatIsOffered:
    def test_models_and_your_files_are_not_ticked(self, installed):
        root, key = installed
        by_key = {i.key: i for i in plan(root, key_path=key)}
        assert by_key["models"].default is False
        assert by_key["datasets"].default is False
        assert by_key["settings"].default is True
        assert by_key["conversations"].default is True
        assert by_key["environment"].default is True
        assert by_key["llama"].default is True

    def test_each_one_says_why(self, installed):
        root, key = installed
        for item in plan(root, key_path=key):
            assert item.why, f"{item.key} offers no reason"

    def test_it_reports_what_each_takes_up(self, installed):
        root, key = installed
        by_key = {i.key: i for i in plan(root, key_path=key)}
        assert by_key["models"].bytes == 4096
        assert by_key["datasets"].bytes == 512

    def test_nothing_absent_is_offered(self, tmp_path):
        bare = tmp_path / "traind"
        bare.mkdir()
        assert plan(bare, key_path=tmp_path / "cluster.key") == []


class TestRemoving:
    def test_leaving_models_unticked_keeps_them(self, installed):
        root, key = installed
        out = remove(root, ["settings", "conversations", "environment", "llama"],
                     key_path=key)

        assert (root / "models" / "big.gguf").exists(), "it took the models"
        assert (root / "files" / "dataset.jsonl").exists(), "it took the datasets"
        assert not (root / "chats").exists()
        assert not (root / "env").exists()
        assert not (root / "llama").exists()
        assert not key.exists()
        assert not (key.parent / "traind.log").exists()
        assert set(out["removed"]) == {"settings", "conversations", "environment",
                                       "llama"}
        assert out["freed"] > 0

    def test_ticking_models_takes_them(self, installed):
        root, key = installed
        remove(root, ["models"], key_path=key)
        assert not (root / "models").exists()
        assert key.exists(), "it took the key when only models were asked for"

    def test_removing_nothing_removes_nothing(self, installed):
        root, key = installed
        out = remove(root, [], key_path=key)
        assert out["removed"] == []
        assert out["freed"] == 0
        assert (root / "models" / "big.gguf").exists()
        assert key.exists()

    def test_everything_leaves_no_directory_behind(self, installed):
        root, key = installed
        keys = [i.key for i in plan(root, key_path=key)]
        remove(root, keys, key_path=key)
        assert not root.exists()

    def test_it_can_be_run_twice(self, installed):
        root, key = installed
        remove(root, ["conversations"], key_path=key)
        out = remove(root, ["conversations"], key_path=key)
        assert out["removed"] == []
        assert out["failed"] == {}
