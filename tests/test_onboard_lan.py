"""Onboarding reaches machines on this network and nothing on the public internet."""

import ssl

import pytest

from ml_stack.fleet.onboard import transfer
from ml_stack.fleet.onboard.lan import NotLocal, require_local
from ml_stack.fleet.onboard.pairing import PairError, PairingClient


@pytest.mark.parametrize("host", ["127.0.0.1", "192.168.1.20", "10.1.2.3", "172.16.0.9",
                                  "169.254.1.1", "100.64.0.7", "::1", "localhost"])
def test_a_machine_on_this_network_is_reachable(host):
    require_local(host, 8772)


@pytest.mark.parametrize("host", ["8.8.8.8", "1.1.1.1", "93.184.216.34", "2606:4700:4700::1111"])
def test_a_public_address_is_refused(host):
    with pytest.raises(NotLocal, match="public internet"):
        require_local(host, 8772)


def test_pairing_with_a_public_address_is_refused_before_a_connection():
    with pytest.raises(PairError, match="public internet"):
        PairingClient("8.8.8.8", 8772, fingerprint="ab" * 32, timeout=0.5)._call("GET", "/x")


def test_fetching_from_a_public_peer_is_refused_before_a_connection():
    peer = transfer.PeerSource("https://8.8.8.8:8772", "", ssl.create_default_context())
    with pytest.raises(NotLocal):
        transfer.fetch_manifest(peer, timeout=0.5)


def test_a_peer_needs_pinned_tls_and_plain_http_is_for_this_machine_only():
    with pytest.raises(transfer.TransferError, match="pinned"):
        transfer.PeerSource("https://192.168.1.2:8772")
    with pytest.raises(transfer.TransferError, match="this machine only"):
        transfer.PeerSource("http://192.168.1.2:8772")
    with pytest.raises(transfer.TransferError, match="https"):
        transfer.PeerSource("ftp://192.168.1.2")
    transfer.PeerSource("http://127.0.0.1:8772")
