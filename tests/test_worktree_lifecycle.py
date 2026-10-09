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
    # another label's retained checkout never blocks this one's completion
    assert kit.ws.announce(kit.sender, 'done', 'Everything complete')
    with pytest.raises(Denied, match='verified cleanup'):
        lifecycle.require_clean(kit.base, 'worker')
    lifecycle.require_clean(kit.base, 'worker', within=(str(kit.base),))
    with pytest.raises(Denied, match='verified cleanup'):
        lifecycle.require_clean(kit.base, 'worker', within=(str(kit.checkout),))


def test_missing_checkout_does_not_hide_surviving_branch(setup):
    kit = setup
    claim(kit)
    repo.git(kit.primary, 'worktree', 'remove', str(kit.checkout))
    with pytest.raises(Denied, match='branches remain: worker/change'):
        kit.ws.announce(kit.sender, 'done', 'Complete', label='helper')
    assert repo.git(kit.primary, 'rev-parse', 'worker/change')
    repo.git(kit.primary, 'branch', '-d', 'worker/change')
    with pytest.raises(Denied, match='cleanup proof is missing'):
        kit.ws.announce(kit.sender, 'done', 'Complete', label='helper')


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
    lifecycle.cleanup(kit.base, 'worker', str(kit.checkout), kit.ws.claims)
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
    repo.git(kit.primary, 'remote', 'add', 'origin', 'https://example.test/fixture.git')
    kit.ws.registry.set_project(kit.ws.auth(kit.owner), 'worker', describe(str(kit.checkout)))
    harness_claims.reserve('Write', {'file_path': str(kit.checkout / 'source.py')},
                           str(kit.checkout), 'worker', [str(kit.checkout)])
    assert lifecycle.pending(kit.base, 'worker')[0]['branch'] == 'worker/change'
    with pytest.raises(Denied, match='verified cleanup'):
        kit.ws.announce(kit.sender, 'done', 'Complete')


@pytest.mark.redteam
def test_deleted_unique_branch_does_not_prove_recorded_work_landed(setup):
    kit = setup
    (kit.checkout / 'source.py').write_text('value = 2\n')
    repo.git(kit.checkout, 'add', 'source.py')
    repo.git(kit.checkout, 'commit', '-m', 'feat: unique work')
    claim(kit)
    unique = repo.git(kit.checkout, 'rev-parse', 'HEAD')
    repo.git(kit.primary, 'worktree', 'remove', str(kit.checkout))
    repo.git(kit.primary, 'branch', '-D', 'worker/change')
    with pytest.raises(Denied, match='recorded commits are not landed'):
        kit.ws.announce(kit.sender, 'done', 'Complete', label='helper')
    repo.git(kit.primary, 'merge', '--ff-only', unique)
    with pytest.raises(Denied, match='cleanup proof is missing'):
        kit.ws.announce(kit.sender, 'done', 'Complete', label='helper')


@pytest.mark.redteam
def test_label_cannot_hide_unlabeled_native_scope(setup):
    kit = setup
    claim(kit, '')
    with pytest.raises(Denied, match='verified cleanup'):
        kit.ws.announce(kit.sender, 'done', 'Complete', label='helper')


@pytest.mark.redteam
def test_harness_stop_uses_authenticated_owner_and_refuses_lingering_scope(setup):
    from ml_stack import harnesshook
    from ml_stack.workspace import tokens
    kit = setup
    tokens.store(kit.base, 'worker', kit.sender)
    claim(kit)
    rail = harnesshook.Rail('plan-and-go', 'worker', roots=[str(kit.checkout)])
    result = harnesshook.stop(rail)
    assert result['decision'] == 'block' and str(kit.checkout) in result['reason']
    lifecycle.cleanup(kit.base, 'worker', str(kit.checkout), kit.ws.claims)
    assert harnesshook.stop(rail) == {}


def test_claude_settings_wire_stop_and_subagent_stop():
    import json

    from ml_stack import claude
    hooks = json.loads(claude.settings('PRE', 'POST', 300, 'STOP'))['hooks']
    assert hooks['Stop'][0]['hooks'][0]['command'] == 'STOP'
    assert hooks['SubagentStop'][0]['hooks'][0]['command'] == 'STOP'


@pytest.mark.redteam
def test_launcher_refuses_success_without_done_announcement(setup, monkeypatch):
    import argparse

    from ml_stack import harnessid, harnessing
    kit = setup
    claim(kit)
    seat = harnessid.Seat('worker', base=kit.base)
    args = argparse.Namespace(project=str(kit.checkout), parent='', name='worker',
                              role='plan-and-go', orders_from='owner',
                              seat_factory=lambda *args: seat)
    messages = []
    with pytest.raises(ValueError, match='verified cleanup'), \
            harnessing.opened(args, 'codex', ('http://127.0.0.1:9', 'fixture', 0), messages.append):
        pass
    assert any('unfinished checkout:' in line for line in messages)
    assert kit.checkout.exists()


def test_precreation_claim_catches_checkout_without_another_mutation(setup):
    kit = setup
    target = kit.checkout.parent / 'reserved-checkout'
    kit.ws.claim(kit.sender, 'worktree', str(target), label='helper')
    assert lifecycle.pending(kit.base, 'worker', 'helper') == []
    repo.git(kit.primary, 'worktree', 'add', '-b', 'worker/reserved', str(target))
    with pytest.raises(Denied, match=r'reserved-checkout.*worker/reserved'):
        kit.ws.announce(kit.sender, 'done', 'Complete', label='helper')
    assert target.exists()


@pytest.mark.redteam
def test_materialized_reservation_retains_branch_and_commit_after_removal(setup):
    kit = setup
    target = kit.checkout.parent / 'reserved-checkout'
    kit.ws.claim(kit.sender, 'worktree', str(target), label='helper')
    repo.git(kit.primary, 'worktree', 'add', '-b', 'worker/reserved', str(target))
    (target / 'source.py').write_text('value = 2\n')
    repo.git(target, 'add', 'source.py')
    repo.git(target, 'commit', '-m', 'feat: reserved work')
    unique = repo.git(target, 'rev-parse', 'HEAD')
    assert lifecycle.pending(kit.base, 'worker', 'helper')[0]['branch'] == 'worker/reserved'
    repo.git(kit.primary, 'worktree', 'remove', str(target))
    with pytest.raises(Denied, match='branches remain: worker/reserved'):
        kit.ws.announce(kit.sender, 'done', 'Complete', label='helper')
    repo.git(kit.primary, 'branch', '-D', 'worker/reserved')
    with pytest.raises(Denied, match='recorded commits are not landed'):
        kit.ws.announce(kit.sender, 'done', 'Complete', label='helper')
    repo.git(kit.primary, 'merge', '--ff-only', unique)
    with pytest.raises(Denied, match='cleanup proof is missing'):
        kit.ws.announce(kit.sender, 'done', 'Complete', label='helper')


@pytest.mark.redteam
def test_reserved_path_materialized_as_primary_repository_remains_pending(setup):
    kit = setup
    target = kit.checkout.parent / 'reserved-primary'
    kit.ws.claim(kit.sender, 'worktree', str(target), label='helper')
    target.mkdir()
    repo.git(target, 'init', '-b', 'development')
    assert lifecycle.pending(kit.base, 'worker', 'helper')[0]['reasons'] == ['reserved checkout path remains']


@pytest.mark.redteam
def test_native_mutation_records_target_checkout_instead_of_working_directory(setup):
    from ml_stack import harness_claims
    from ml_stack.workspace import tokens
    from ml_stack.workspace.project import describe
    kit = setup
    target = kit.checkout.parent / 'other-checkout'
    repo.git(kit.primary, 'worktree', 'add', '-b', 'worker/other', str(target))
    tokens.store(kit.base, 'worker', kit.sender)
    repo.git(kit.primary, 'remote', 'add', 'origin', 'https://example.test/fixture.git')
    kit.ws.registry.set_project(kit.ws.auth(kit.owner), 'worker', describe(str(kit.checkout)))
    harness_claims.reserve('Write', {'file_path': str(target / 'source.py')},
                           str(kit.checkout), 'worker', [str(kit.checkout), str(target)])
    scopes = lifecycle.pending(kit.base, 'worker')
    assert [scope['path'] for scope in scopes] == [str(target)]


@pytest.mark.redteam
def test_launcher_exception_reports_pending_scope_and_preserves_failure(setup):
    import argparse

    from ml_stack import harnessid, harnessing
    kit = setup
    claim(kit)
    seat = harnessid.Seat('worker', base=kit.base)
    args = argparse.Namespace(project=str(kit.checkout), parent='', name='worker',
                              role='plan-and-go', orders_from='owner',
                              seat_factory=lambda *args: seat)
    messages = []
    with pytest.raises(OSError, match='fixture process failed'), \
            harnessing.opened(args, 'codex', ('http://127.0.0.1:9', 'fixture', 0), messages.append):
        messages.clear()
        raise OSError('fixture process failed')
    assert any('unfinished checkout:' in line for line in messages)
    assert kit.checkout.exists()


@pytest.mark.redteam
def test_nested_reservation_retains_checkout_provenance(setup):
    kit = setup
    target = kit.checkout.parent / 'reserved-checkout'
    kit.ws.claim(kit.sender, 'worktree', str(target / 'nested'), label='helper')
    repo.git(kit.primary, 'worktree', 'add', '-b', 'worker/reserved', str(target))
    assert lifecycle.pending(kit.base, 'worker', 'helper')[0]['path'] == str(target)
    repo.git(kit.primary, 'worktree', 'remove', str(target))
    with pytest.raises(Denied, match='branches remain: worker/reserved'):
        kit.ws.announce(kit.sender, 'done', 'Complete', label='helper')


@pytest.mark.redteam
def test_launcher_exception_survives_failed_exit_inspection(setup, monkeypatch):
    import argparse

    from ml_stack import harnessid, harnessing
    kit = setup
    seat = harnessid.Seat('worker', base=kit.base)
    args = argparse.Namespace(project=str(kit.checkout), parent='', name='worker',
                              role='plan-and-go', orders_from='owner',
                              seat_factory=lambda *args: seat)
    messages = []
    with pytest.raises(OSError, match='fixture process failed'), \
            harnessing.opened(args, 'codex', ('http://127.0.0.1:9', 'fixture', 0), messages.append):
        monkeypatch.setattr(harnessid.Seat, 'pending_worktrees',
                            lambda self: (_ for _ in ()).throw(RuntimeError('fixture inspection failed')))
        raise OSError('fixture process failed')
    assert any('checkout inspection failed:' in line for line in messages)


@pytest.mark.redteam
def test_claim_before_final_commit_cannot_complete_after_lost_checkout_and_branch(setup):
    kit = setup
    claim(kit)
    (kit.checkout / 'source.py').write_text('value = 2\n')
    repo.git(kit.checkout, 'add', 'source.py')
    repo.git(kit.checkout, 'commit', '-m', 'feat: uncaptured final commit')
    repo.git(kit.primary, 'worktree', 'remove', str(kit.checkout))
    repo.git(kit.primary, 'branch', '-D', 'worker/change')
    lifecycle.remember(kit.base, 'worker', 'helper', str(kit.checkout))
    with pytest.raises(Denied, match='cleanup proof is missing'):
        kit.ws.announce(kit.sender, 'done', 'Complete', label='helper')


@pytest.mark.redteam
def test_pretool_invalidates_old_cleanup_proof_when_tool_removes_its_final_commit(setup, monkeypatch):
    from ml_stack import harness_claims, harnesshook
    from ml_stack.workspace import tokens
    from ml_stack.workspace.project import describe

    kit = setup
    claim(kit)
    lifecycle.cleanup(kit.base, 'worker', str(kit.checkout), kit.ws.claims)
    repo.git(kit.primary, 'worktree', 'add', '-b', 'worker/change', str(kit.checkout))
    tokens.store(kit.base, 'worker', kit.sender)
    repo.git(kit.primary, 'remote', 'add', 'origin', 'https://example.test/fixture.git')
    kit.ws.registry.set_project(kit.ws.auth(kit.owner), 'worker', describe(str(kit.checkout)))
    harness_claims.reserve('Write', {'file_path': str(kit.checkout / 'source.py')},
                           str(kit.checkout), 'worker', [str(kit.checkout)])
    (kit.checkout / 'source.py').write_text('value = 2\n')
    repo.git(kit.checkout, 'add', 'source.py')
    repo.git(kit.checkout, 'commit', '-m', 'feat: vanished tool commit')
    repo.git(kit.primary, 'worktree', 'remove', str(kit.checkout))
    repo.git(kit.primary, 'branch', '-D', 'worker/change')
    monkeypatch.setattr(harnesshook, 'nudge', lambda *_, **__: '')
    assert harnesshook.post('worker', harnesshook.Rail('plan-and-go', 'worker', [str(kit.checkout)])) == {}
    with pytest.raises(Denied, match='cleanup proof is missing'):
        kit.ws.announce(kit.sender, 'done', 'Complete')


def test_posttool_captures_final_commit_and_cleanup_proves_it_landed(setup, monkeypatch):
    from ml_stack.workspace import notification_reader, tokens

    kit = setup
    claim(kit)
    tokens.store(kit.base, 'worker', kit.sender)
    (kit.checkout / 'source.py').write_text('value = 2\n')
    repo.git(kit.checkout, 'add', 'source.py')
    repo.git(kit.checkout, 'commit', '-m', 'feat: captured final commit')
    commit = repo.git(kit.checkout, 'rev-parse', 'HEAD')
    # the post hook's notification reader checkpoints under the authenticated identity
    notification_reader.checkpoint(kit.base, 'worker')
    assert commit in lifecycle.scopes(kit.base, 'worker')[0]['commits']
    with pytest.raises(Denied, match='commits outside'):
        lifecycle.cleanup(kit.base, 'worker', str(kit.checkout), kit.ws.claims)
    assert kit.checkout.exists()
    repo.git(kit.primary, 'merge', '--ff-only', commit)
    result = lifecycle.cleanup(kit.base, 'worker', str(kit.checkout), kit.ws.claims)
    assert result['cleanup_verified'] and result['commit'] == commit
    assert kit.ws.announce(kit.sender, 'done', 'Complete', label='helper')


@pytest.mark.redteam
def test_cleanup_never_takes_a_foreign_live_claim(setup):
    kit = setup
    claim(kit)
    kit.ws.release(kit.sender, 'worktree', str(kit.checkout))
    foreign = kit.agent('foreign')
    kit.ws.claim(foreign, 'worktree', str(kit.checkout))
    with pytest.raises(Denied, match='live checkout claim'):
        lifecycle.cleanup(kit.base, 'worker', str(kit.checkout), kit.ws.claims)
    assert kit.checkout.exists()
    assert kit.ws.who_owns('worktree', str(kit.checkout))['owner'] == 'foreign'


def test_authenticated_worktree_cleanup_cli_records_the_exact_landed_commit(setup):
    from types import SimpleNamespace

    from ml_stack.workspace import cli

    kit = setup
    claim(kit)
    args = SimpleNamespace(cleanup=str(kit.checkout), label='helper')
    result = cli._worktrees(args, kit.ws, kit.sender)
    assert result['cleanup_verified']
    assert lifecycle.pending(kit.base, 'worker') == []
