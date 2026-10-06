"""The page's list of clusters to join: what a daemon with the UI mounted answers."""

from __future__ import annotations

import socket

import pytest

from ml_stack.fleet.discovery import Advertiser, Beacon, mint_cluster
from tests.test_fleet_ui import Serving, primary_ip


@pytest.fixture
def serving(tmp_path):
    s = Serving(tmp_path)
    try:
        yield s
    finally:
        s.close()


def test_first_run_lists_the_clusters_the_network_offers(serving, tmp_path, monkeypatch):
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind(("127.0.0.1", 0))
        udp = s.getsockname()[1]
    monkeypatch.setenv("ML_STACK_DISCOVERY_PORT", str(udp))
    key = mint_cluster("lab", tmp_path / "other.key").key
    with Advertiser(Beacon(name="larch", port=9), key, port=udp, cluster="lab", interval_s=5.0):
        status, body, _ = serving.call("/ui/setup/clusters")
    assert status == 200
    assert body["found"] == [{"name": "lab", "machines": ["larch"], "mine": False, "method": "passphrase"}]


def test_a_machine_in_a_cluster_asks_for_the_list_behind_its_sign_in(serving):
    mint_cluster("home", serving.keyfile)
    assert serving.call("/ui/clusters/found")[0] == 401


def test_first_run_refuses_the_list_from_another_machine(serving):
    status, body, _ = serving.call("/ui/setup/clusters", host=primary_ip())
    assert status == 403 and "ssh" in body["error"]
