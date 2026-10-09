"""A machine that joined the node's pool lists it with `ml-stack-peers ls` and needs no cluster key file."""

import argparse
import json

import pytest

from ml_stack import node_pool
from ml_stack.fleet import peers
from ml_stack.fleet.discovery import DiscoveryError

POOL = {"pool": "0123456789abcdef", "project": "ab" * 8, "policy": "open", "members": [
    {"fingerprint": "a" * 64, "name": "desk", "status": "active", "self": True, "connected": False, "addr": None},
    {"fingerprint": "b" * 64, "name": "", "status": "active", "self": False, "connected": True, "addr": "192.0.2.9:7447"},
    {"fingerprint": "c" * 64, "name": "gone", "status": "revoked", "self": False, "connected": False, "addr": None}]}


def args(tmp_path, *, as_json=False):
    return argparse.Namespace(cluster_key=str(tmp_path / "no-such.key"), timeout=0.1, json=as_json)


def test_only_active_members_are_listed_with_who_is_this_machine():
    rows = node_pool.rows(POOL)
    assert [(r["name"], r["state"]) for r in rows] == [("desk", "this machine"), ("b" * 12, "connected")]
    assert rows[1]["addr"] == "192.0.2.9:7447"


def test_the_network_arguments_ask_for_the_lan_and_pass_a_project_on():
    assert node_pool.network_args() == ["--lan"]
    assert node_pool.network_args(project="team", project_dir="/r") == ["--lan", "--project", "team", "--project-dir", "/r"]


def test_ls_without_a_key_file_prints_the_pool_of_the_running_node(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(node_pool, "status", lambda state=None: POOL)
    assert peers.cmd_ls(args(tmp_path)) == 0
    out = capsys.readouterr().out
    assert "0123456789abcdef" in out and "desk" in out and "192.0.2.9:7447" in out and "gone" not in out
    assert peers.cmd_ls(args(tmp_path, as_json=True)) == 0
    assert [m["name"] for m in json.loads(capsys.readouterr().out)["members"]] == ["desk", "b" * 12]


def test_ls_without_a_key_file_and_without_a_node_still_says_what_is_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(node_pool, "status", lambda state=None: None)
    with pytest.raises(DiscoveryError, match="no cluster key"):
        peers.cmd_ls(args(tmp_path))


def test_status_is_none_when_no_node_answers(tmp_path):
    assert node_pool.status(tmp_path / "empty") is None
