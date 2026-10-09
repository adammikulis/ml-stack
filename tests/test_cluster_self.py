"""The local machine remains visible when LAN beacons are absent."""

from unittest.mock import patch

from poolhouse.fleet.discovery import Beacon, Membership
from poolhouse.fleet.ui import UI


def test_joined_cluster_includes_local_machine_without_beacons(tmp_path):
    interface = UI(name='local-box', cluster_key_path=tmp_path / 'absent.key')
    with patch('poolhouse.fleet.ui.memberships', return_value=[Membership('default', b'a' * 43, selection='manual')]), \
            patch('poolhouse.fleet.ui.discover', return_value=[]):
        rows = interface.peers(force=True)
    assert len(rows) == 1
    assert rows[0]['name'] == 'local-box'
    assert rows[0]['is_self']
    assert rows[0]['clusters'] == ['default']


def test_local_machine_beacons_on_multiple_addresses_do_not_duplicate_it(tmp_path):
    interface = UI(name='local-box', cluster_key_path=tmp_path / 'absent.key')
    local = interface.itself()
    beacons = [Beacon(name='local-box', machine=local['machine'], host=host, port=8770)
               for host in ('192.0.2.1', '192.0.2.2')]
    with patch('poolhouse.fleet.ui.memberships', return_value=[Membership('default', b'a' * 43, selection='manual')]), \
            patch('poolhouse.fleet.ui.discover', return_value=beacons):
        rows = interface.peers(force=True)
    assert len(rows) == 1
    assert rows[0]['base_url'] == 'http://127.0.0.1:8770'
