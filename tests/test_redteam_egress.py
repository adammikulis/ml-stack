"""The guard that keeps red-team traffic on this machine."""

from __future__ import annotations

import socket

import pytest

from ml_stack.http import request_bytes
from ml_stack.redteam.egress import EgressRefused, local_only
from ml_stack.testing.fakes import fake_llama_server


def test_a_connection_to_an_address_off_this_machine_is_refused_before_it_is_sent():
    with local_only(), pytest.raises(EgressRefused), socket.socket() as sock:
        sock.connect(("192.0.2.1", 9))


def test_a_name_that_is_not_an_address_is_refused_rather_than_resolved():
    with local_only(), pytest.raises(EgressRefused), socket.socket() as sock:
        sock.connect(("example.com", 80))


def test_connect_ex_is_guarded_as_well():
    with local_only(), socket.socket() as sock, pytest.raises(EgressRefused):
        sock.connect_ex(("198.51.100.7", 80))


def test_a_server_on_loopback_is_still_reachable_inside_the_block():
    with fake_llama_server() as server, local_only():
        assert request_bytes(f"{server.base_url}/health", timeout=5).status == 200


def test_the_socket_methods_are_the_originals_again_afterwards():
    before = (socket.socket.connect, socket.socket.connect_ex)
    with local_only():
        assert socket.socket.connect is not before[0]
    assert (socket.socket.connect, socket.socket.connect_ex) == before


def test_a_block_that_raises_still_restores_the_socket_methods():
    before = socket.socket.connect
    with pytest.raises(RuntimeError), local_only():
        raise RuntimeError("boom")
    assert socket.socket.connect is before
