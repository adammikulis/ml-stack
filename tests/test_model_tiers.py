"""The tier table decides which main sessions may coordinate: real registry, real table file."""
import copy
import json

import pytest
from workspace_kit import Kit, clean_env

from ml_stack.workspace.agent_display import SPAWN_NOTICE, metadata
from ml_stack.workspace.model_tiers import (
    LOWEST,
    NOT_LISTED,
    NOT_VERIFIED,
    TABLE,
    TierTableError,
    load_table,
    tier_of,
)
from ml_stack.workspace.modelid import CLAIMED, INHERITED, VERIFIED

PERSON = {'terminal': (True, True), 'env': {}}
SHIPPED = json.loads(TABLE.read_text(encoding='utf-8'))
EXPECTED = {
    'claude-haiku-5-5': ('anthropic', 0), 'anthropic.claude-haiku-5-5': ('anthropic', 0),
    'claude-haiku-4-5': ('anthropic', 0), 'claude-haiku-4-5-20251001': ('anthropic', 0),
    'claude-sonnet-5': ('anthropic', 1), 'claude-sonnet-5-5': ('anthropic', 1),
    'anthropic.claude-sonnet-5-5': ('anthropic', 1), 'claude-sonnet-4-6': ('anthropic', 1),
    'claude-opus-5': ('anthropic', 2), 'claude-opus-5-5': ('anthropic', 2),
    'anthropic.claude-opus-5-5': ('anthropic', 2), 'claude-opus-4-5': ('anthropic', 2),
    'claude-opus-4-6': ('anthropic', 2), 'claude-opus-4-7': ('anthropic', 2), 'claude-opus-4-8': ('anthropic', 2),
    'claude-fable-5': ('anthropic', 3), 'claude-fable-5-1': ('anthropic', 3),
    'anthropic.claude-fable-5-1': ('anthropic', 3),
    'gpt-6-luna': ('openai', 0), 'gpt-6-sol': ('openai', 1), 'gpt-6.1-sol': ('openai', 1),
    'gpt-6-astra': ('openai', 2),
}


@pytest.fixture
def kit(monkeypatch, tmp_path):
    return Kit(clean_env(monkeypatch, tmp_path))


def main(kit, name, model=None, verified=True):
    token = kit.agent(name)
    kit.ws.register_session(token)
    if model:
        kit.ws.set_model(name, model, verified=verified, **PERSON)
    return token


def shown(kit, name, label=''):
    return metadata(kit.ws.registry, name, label)


@pytest.mark.parametrize(('model', 'place'), EXPECTED.items())
def test_every_listed_id_lands_in_its_tier(model, place):
    tier = tier_of(model, VERIFIED)
    assert (tier.family, tier.index) == place
    assert tier.lowest == (place[1] == 0)
    assert tier.coordinator_eligible == (place[1] > 0)


def test_the_shipped_table_lists_exactly_the_ids_the_tests_name():
    literal = {g for f in SHIPPED['families'] for t in f['tiers'] for g in t['id_globs'] if '*' not in g}
    assert literal <= set(EXPECTED)
    assert load_table(SHIPPED)


@pytest.mark.parametrize('model', ['claude-haiku-5-5', 'gpt-6-luna', 'anthropic.claude-haiku-5-5'])
def test_lowest_tier_main_sessions_are_not_eligible(kit, model):
    main(kit, 'lowest', model)
    row = shown(kit, 'lowest')
    assert not row['coordinator_eligible']
    assert row['coordinator_reason'] == LOWEST


@pytest.mark.parametrize('model', ['claude-sonnet-5-5', 'claude-opus-5-5', 'claude-fable-5-1',
                                   'gpt-6-sol', 'gpt-6-astra'])
def test_higher_tier_verified_main_sessions_are_eligible(kit, model):
    main(kit, 'higher', model)
    row = shown(kit, 'higher')
    assert row['coordinator_eligible']
    assert row['coordinator_reason'] == ''


def test_unknown_and_unverified_models_are_not_eligible(kit):
    main(kit, 'nomodel')
    main(kit, 'stranger', 'frontier-9000')
    main(kit, 'claimed', 'claude-opus-5-5', verified=False)
    assert shown(kit, 'nomodel')['coordinator_reason'] == NOT_LISTED
    assert shown(kit, 'stranger')['coordinator_reason'] == NOT_LISTED
    assert shown(kit, 'claimed')['coordinator_reason'] == NOT_VERIFIED
    assert not any(shown(kit, n)['coordinator_eligible'] for n in ('nomodel', 'stranger', 'claimed'))
    assert tier_of('claude-opus-5-5', CLAIMED).reason == NOT_VERIFIED
    assert tier_of('claude-opus-5-5', INHERITED).reason == NOT_VERIFIED


def test_a_claimed_id_stays_ineligible_after_the_session_claims_a_listed_model(kit):
    token = main(kit, 'claimer')
    kit.ws.claim_model(token, 'claude-opus-5-5', 'claude-code')
    assert kit.ws.registry.model_of('claimer') == ('claude-opus-5-5', CLAIMED)
    assert not shown(kit, 'claimer')['coordinator_eligible']


def test_names_and_labels_never_set_the_tier(kit):
    main(kit, 'sonnet-lead', 'claude-haiku-5-5')
    assert not shown(kit, 'sonnet-lead')['coordinator_eligible']
    assert 'sonnet' not in shown(kit, 'sonnet-lead')['display_name'].lower()
    labelled = shown(kit, 'sonnet-lead', 'claude-sonnet-5-5')
    assert not labelled['coordinator_eligible']
    assert tier_of('Claude Sonnet 5.5', VERIFIED).reason == NOT_LISTED
    assert tier_of('claude-sonnet-5-5 ', VERIFIED).reason == NOT_LISTED
    assert tier_of('CLAUDE-SONNET-5-5', VERIFIED).reason == NOT_LISTED
    assert tier_of('claude-sonnet-5-5-haiku', VERIFIED).reason == NOT_LISTED


def test_a_subagent_never_becomes_eligible_by_model(kit):
    token = main(kit, 'parent', 'claude-sonnet-5-5')
    child = kit.ws.registry.delegate(kit.ws.auth(token), 'worker', 600, ('read',), 10)
    kit.ws.claim_model(child, 'claude-fable-5-1')
    kit.ws.set_model('parent/worker', 'claude-fable-5-1', **PERSON)
    row = shown(kit, 'parent/worker')
    assert row['session_kind'] == 'subagent'
    assert not row['coordinator_eligible']
    assert shown(kit, 'parent')['coordinator_eligible']


def test_registered_rows_and_status_carry_the_reason(kit):
    main(kit, 'low', 'claude-haiku-5-5')
    main(kit, 'high', 'claude-sonnet-5-5')
    rows = {row['id']: row for row in kit.ws.registered()}
    assert rows['low']['coordinator_reason'] == LOWEST
    assert rows['high']['coordinator_eligible']
    assert kit.ws.status()['coordinator_notice'] == ''


def test_only_lowest_tier_mains_are_told_to_spawn_a_coordinator(kit):
    main(kit, 'haiku', 'claude-haiku-5-5')
    main(kit, 'luna', 'gpt-6-luna')
    assert kit.ws.status()['coordinator_notice'] == SPAWN_NOTICE
    assert 'Sonnet 5.5' in SPAWN_NOTICE
    main(kit, 'sonnet', 'claude-sonnet-5-5')
    assert kit.ws.status()['coordinator_notice'] == ''


def test_unknown_only_mains_get_no_spawn_notice(kit):
    main(kit, 'stranger', 'frontier-9000')
    assert kit.ws.status()['coordinator_notice'] == ''


def mutated(change):
    table = copy.deepcopy(SHIPPED)
    change(table)
    return table


def _first(table):
    return table['families'][0]['tiers']


MALFORMED = {
    'unknown top field': lambda t: t.update(extra=1),
    'unknown family field': lambda t: t['families'][0].update(extra=1),
    'unknown tier field': lambda t: _first(t)[0].update(coordinator_eligible=True),
    'no version': lambda t: t.pop('version'),
    'no families': lambda t: t.update(families=[]),
    'duplicate glob across tiers': lambda t: _first(t)[1]['id_globs'].append('claude-haiku-5-5'),
    'duplicate glob across families': lambda t: t['families'][1]['tiers'][1]['id_globs'].append('claude-fable-5'),
    'literal matched by another tier glob': lambda t: _first(t)[2]['id_globs'].append('claude-haiku-4-5-x'),
    'no lowest tier': lambda t: _first(t)[0].update(lowest=False),
    'two lowest tiers': lambda t: _first(t)[1].update(lowest=True),
    'lowest tier not first': lambda t: _first(t).reverse(),
    'empty glob list': lambda t: _first(t)[1].update(id_globs=[]),
    'empty glob': lambda t: _first(t)[1]['id_globs'].append(''),
    'non-boolean lowest': lambda t: _first(t)[1].update(lowest='no'),
    'duplicate family': lambda t: t['families'][1].update(family='anthropic'),
    'family without tiers': lambda t: t['families'][1].update(tiers=[]),
}


@pytest.mark.parametrize('name', MALFORMED)
def test_a_malformed_table_is_refused(name):
    with pytest.raises(TierTableError):
        load_table(mutated(MALFORMED[name]))
    with pytest.raises(TierTableError):
        tier_of('claude-sonnet-5-5', VERIFIED, mutated(MALFORMED[name]))


def test_a_table_that_is_not_an_object_is_refused():
    with pytest.raises(TierTableError, match='not an object'):
        load_table([])


def test_a_substitute_table_classifies_by_its_own_ids():
    table = {'version': 1, 'families': [{'family': 'acme', 'tiers': [
        {'lowest': True, 'id_globs': ['acme-small-*']}, {'lowest': False, 'id_globs': ['acme-big']}]}]}
    assert tier_of('acme-small-2', VERIFIED, table).lowest
    assert tier_of('acme-big', VERIFIED, table).coordinator_eligible
    assert tier_of('claude-sonnet-5-5', VERIFIED, table).reason == NOT_LISTED
