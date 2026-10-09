"""Seat renewal preserves identity and refuses foreign or revoked delegates."""

import pytest
from workspace_kit import Kit

from poolhouse.workspace import child_renewal, tokens
from poolhouse.workspace.identity import Denied


def test_parent_renews_same_token_and_identity(tmp_path):
    kit = Kit(tmp_path)
    parent = kit.agent('parent')
    child = kit.ws.delegate(parent, 'child', ttl_s=60)
    secret = tokens.load(kit.base, child['id'])
    info = kit.ws.registry.info(child['id'])
    renewed = child_renewal.renew(kit.ws, parent, child['id'])
    assert renewed['expires'] > info['expires']
    assert tokens.load(kit.base, child['id']) == secret
    assert kit.ws.auth(secret).id == child['id']
    with pytest.raises(Denied):
        child_renewal.renew(kit.ws, kit.agent('foreign'), child['id'])
    with pytest.raises(Denied):
        child_renewal.renew(kit.ws, secret, child['id'])


@pytest.mark.parametrize('expired', [False, True])
def test_revoked_or_expired_seat_is_not_restored(tmp_path, expired):
    now = [1000.0]
    kit = Kit(tmp_path, clock=lambda: now[0])
    parent = kit.agent('parent')
    child = kit.ws.delegate(parent, 'child', ttl_s=60)
    if expired:
        now[0] += 61
    else:
        kit.ws.registry.revoke(kit.ws.auth(parent), child['id'])
    with pytest.raises(Denied, match='live child'):
        child_renewal.renew(kit.ws, parent, child['id'])
