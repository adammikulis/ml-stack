"""Hostile input to the two places the board client starts a process: a note's re-derive command and a git query about a path."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from poolhouse.board import client as board_client, place
from poolhouse.workspace import limits

pytest_plugins = ["node_kit"]
pytestmark = pytest.mark.redteam


def allow(*commands):
    root = limits.root()
    limits.save(root, replace(limits.load(root), verify_allow=[list(c) for c in commands]))


def test_a_notes_command_is_never_run_through_a_shell(workspace_node, tmp_path):
    me = workspace_node.member("author")
    marker = tmp_path / "pwned"
    allow(["true"])
    for hostile in (f"true && touch {marker}", f"true; touch {marker}", f"true $(touch {marker})", f"true `touch {marker}`",
                    f"true | tee {marker}", f"true > {marker}"):
        note = json.loads(workspace_node.cli("notes-add", "fact", "x", "y", "--verify-cmd", hostile, "--json", who=me).stdout)
        workspace_node.cli("notes-verify", note["id"], "--json", who=me)
        assert not marker.exists(), hostile


def test_a_command_off_the_allow_list_is_refused_whatever_the_note_says(workspace_node, tmp_path):
    me = workspace_node.member("author")
    marker = tmp_path / "ran"
    allow(["true"])
    for hostile in (f"touch {marker}", f"/bin/sh -c 'touch {marker}'", f"env touch {marker}", f"./true {marker}"):
        note = json.loads(workspace_node.cli("notes-add", "fact", "x", "y", "--verify-cmd", hostile, "--json", who=me).stdout)
        done = workspace_node.cli("notes-verify", note["id"], "--json", who=me)
        assert done.returncode == 3 and "verify_allow" in done.stdout and not marker.exists(), hostile


def test_a_peers_note_cannot_choose_the_command_a_verification_records(workspace_node):
    me = workspace_node.member("author")
    allow(["true"])
    note = json.loads(workspace_node.cli("notes-add", "fact", "x", "y", "--verify-cmd", "true", "--json", who=me).stdout)
    forged = workspace_node.client.call("note_verify", workspace_node.board, me.token, note=note["id"], exit=0, out_sha="0" * 64)
    assert forged["notes"][0]["last_verified"]["exit"] == 0
    with pytest.raises(board_client.Invalid):
        workspace_node.client.call("note_verify", workspace_node.board, me.token, note=note["id"], exit=0, out_sha="0" * 64, cmd="rm -rf /")


@pytest.mark.parametrize("name", ["-evil", "--git-dir=x", "a b", "line\nfeed", "quote'\"", "$(touch x)"])
def test_a_directory_named_like_an_option_or_a_command_is_only_a_path_to_git(tmp_path, name):
    hostile = tmp_path / name
    hostile.mkdir()
    assert place.board_id(hostile) is None
    assert not (tmp_path / "x").exists() and not Path("/tmp/x/HEAD").exists()
