"""Completion checks retain checkout provenance after ephemeral ownership expires."""

import pytest
from workspace_kit import Kit, clean_env

from ml_stack.workspace import Denied, integration_git as repo, worktree_lifecycle as lifecycle


@pytest.fixture
def setup(tmp_path, monkeypatch):
    kit = Kit(clean_env(monkeypatch, tmp_path))
    kit.sender = kit.agent('worker')
    kit.agent('receiver')
    kit.primary = tmp_path / 'repository'
    kit.primary.mkdir()
    repo.git(kit.primary, 'init', '-b', 'development')
    repo.git(kit.primary, 'config', 'user.name', 'Fixture')
    repo.git(kit.primary, 'config', 'user.email', 'fixture@example.test')
    (kit.primary / 'source.py').write_text('value = 1\n')
    repo.git(kit.primary, 'add', 'source.py')
    repo.git(kit.primary, 'commit', '-m', 'chore: fixture')
    kit.checkout = tmp_path / 'checkout'
    repo.git(kit.primary, 'worktree', 'add', '-b', 'worker/change', str(kit.checkout))
    return kit


def claim(kit, label='helper'):
    return kit.ws.claim(kit.sender, 'worktree', str(kit.checkout), label=label)


def test_scoped_done_blocks_retained_checkout_after_claim_release(setup):
    kit = setup
    claim(kit)
    kit.ws.release(kit.sender, 'worktree', str(kit.checkout))
    with pytest.raises(Denied, match=r'checkout.*worker/change'):
        kit.ws.announce(kit.sender, 'done', 'Complete', label='helper')
    assert kit.checkout.exists()
    assert kit.ws.announce(kit.sender, 'milestone', 'Review ready', label='helper')
    assert kit.ws.announce(kit.sender, 'done', 'Read-only audit', label='other')
    with pytest.raises(Denied, match='verified cleanup'):
        kit.ws.announce(kit.sender, 'done', 'Everything complete')


def test_missing_checkout_does_not_hide_surviving_branch(setup):
    kit = setup
    claim(kit)
    repo.git(kit.primary, 'worktree', 'remove', str(kit.checkout))
    with pytest.raises(Denied, match='branches remain: worker/change'):
        kit.ws.announce(kit.sender, 'done', 'Complete', label='helper')
    assert repo.git(kit.primary, 'rev-parse', 'worker/change')
    repo.git(kit.primary, 'branch', '-d', 'worker/change')
    assert kit.ws.announce(kit.sender, 'done', 'Complete', label='helper')
    assert lifecycle.pending(kit.base, 'worker', 'helper') == []


def test_foreign_checkout_and_primary_are_not_completion_scopes(setup):
    kit = setup
    foreign = kit.agent('foreign')
    kit.ws.claim(foreign, 'worktree', str(kit.checkout), label='helper')
    kit.ws.claim(kit.sender, 'worktree', str(kit.primary))
    assert kit.ws.announce(kit.sender, 'done', 'Read-only audit')
    assert kit.checkout.exists()
    assert lifecycle.pending(kit.base, 'worker') == []


@pytest.mark.redteam
def test_completion_never_deletes_dirty_unique_or_ignored_work(setup):
    kit = setup
    claim(kit)
    (kit.checkout / 'source.py').write_text('value = 2\n')
    (kit.checkout / 'private.txt').write_text('preserved')
    (kit.checkout / '.gitignore').write_text('private.txt\n')
    before = {name: (kit.checkout / name).read_bytes() for name in ('source.py', 'private.txt', '.gitignore')}
    with pytest.raises(Denied, match='checkout remains'):
        kit.ws.announce(kit.sender, 'done', 'Complete', label='helper')
    assert {name: (kit.checkout / name).read_bytes() for name in before} == before


def test_claim_expiry_does_not_erase_durable_checkout_attribution(setup):
    kit = setup
    now = [kit.ws.clock()]
    kit.ws.claims.clock = lambda: now[0]
    kit.ws.claim(kit.sender, 'worktree', str(kit.checkout), ttl_s=1, label='helper')
    now[0] += 2
    assert kit.ws.claims.listing(owner='worker') == []
    with pytest.raises(Denied, match='verified cleanup'):
        kit.ws.announce(kit.sender, 'done', 'Complete', label='helper')


def test_branch_changes_remain_tracked_until_every_attributed_branch_is_removed(setup):
    kit = setup
    claim(kit)
    repo.git(kit.checkout, 'switch', '-c', 'worker/second')
    claim(kit)
    repo.git(kit.primary, 'worktree', 'remove', str(kit.checkout))
    repo.git(kit.primary, 'branch', '-d', 'worker/second')
    with pytest.raises(Denied, match='worker/change'):
        kit.ws.announce(kit.sender, 'done', 'Complete', label='helper')


@pytest.mark.redteam
def test_cached_done_rechecks_reappeared_branch(setup):
    kit = setup
    claim(kit)
    repo.git(kit.primary, 'worktree', 'remove', str(kit.checkout))
    repo.git(kit.primary, 'branch', '-d', 'worker/change')
    kit.ws.announce(kit.sender, 'done', 'Complete', label='helper')
    repo.git(kit.primary, 'branch', 'worker/change')
    with pytest.raises(Denied, match='branches remain'):
        kit.ws.announce(kit.sender, 'done', 'Complete', label='helper')


def test_native_mutation_records_its_unmanaged_checkout(setup):
    from ml_stack import harness_claims
    from ml_stack.workspace import tokens
    from ml_stack.workspace.project import describe
    kit = setup
    tokens.store(kit.base, 'worker', kit.sender)
    kit.ws.registry.set_project(kit.ws.auth(kit.owner), 'worker', describe(str(kit.checkout)))
    harness_claims.reserve('Write', {'file_path': str(kit.checkout / 'source.py')},
                           str(kit.checkout), 'worker', [str(kit.checkout)])
    assert lifecycle.pending(kit.base, 'worker')[0]['branch'] == 'worker/change'
    with pytest.raises(Denied, match='verified cleanup'):
        kit.ws.announce(kit.sender, 'done', 'Complete')
