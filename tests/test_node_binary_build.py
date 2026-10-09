"""The node build starts cargo with a fixed command line; a hostile source path stays a directory."""

from __future__ import annotations

import json
import stat

import pytest

from ml_stack import node_binary


def _cargo_shim(directory, record):
    shim = directory / "cargo"
    shim.write_text(
        "#!/bin/sh\n"
        f"printf '%s\\n' \"$@\" > '{record}.argv'\n"
        f"pwd > '{record}.cwd'\n"
        "mkdir -p \"${CARGO_TARGET_DIR:-target}/release\"\n"
        "printf x > \"${CARGO_TARGET_DIR:-target}/release/poolside-node\"\n",
        encoding="utf-8")
    shim.chmod(shim.stat().st_mode | stat.S_IXUSR)
    return shim


def _tree(root, name):
    source = root / name
    (source / "app" / "poolside-node").mkdir(parents=True)
    (source / "app" / "poolside-node" / "Cargo.toml").write_text("[package]\n", encoding="utf-8")
    return source


@pytest.mark.parametrize("name", ["plain", "with space", "semi;colon", "dollar$(touch pwned)", "back`tick`", "-leading-dash"])
def test_a_hostile_source_path_reaches_cargo_as_a_directory_and_nothing_else(tmp_path, monkeypatch, name):
    record = tmp_path / "record"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _cargo_shim(bin_dir, record)
    monkeypatch.setenv("PATH", f"{bin_dir}:/usr/bin:/bin")
    source = _tree(tmp_path, name)
    made = node_binary.build(source, cache=tmp_path / "target")
    assert made.is_file()
    argv = (tmp_path / "record.argv").read_text(encoding="utf-8").split("\n")[:-1]
    assert argv == ["build", "--release", "--locked", "-p", "poolside-node"]
    assert (tmp_path / "record.cwd").read_text(encoding="utf-8").strip().endswith(f"{name}/app")
    assert not list(tmp_path.rglob("pwned"))


def test_a_tree_without_the_node_crate_is_refused_before_cargo_starts(tmp_path, monkeypatch):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _cargo_shim(bin_dir, tmp_path / "record")
    monkeypatch.setenv("PATH", f"{bin_dir}:/usr/bin:/bin")
    with pytest.raises(node_binary.NodeBinaryError):
        node_binary.build(tmp_path)
    assert not (tmp_path / "record.argv").exists()
    assert json.dumps({}) == "{}"
