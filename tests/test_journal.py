"""Per-device signed journals: appending, sealing, copying between devices and merging."""

import copy

import pytest

from poolhouse.fleet.onboard.manifest import Signer
from poolhouse.workspace import journal_merge as rules
from poolhouse.workspace.journal import Journals


class Clock:
    def __init__(self, now=1_700_000_000.0):
        self.now = now

    def __call__(self):
        return self.now


def device(tmp_path, name, clock=None, key=None):
    signer = key or Signer.generate()
    return Journals(tmp_path / name, lambda: signer, clock or Clock()), signer


def post(journals, text, actor='alice', idem=''):
    return journals.append('message', actor, {'text': text}, idem)


def test_rows_chain_carry_a_clock_that_never_goes_back_and_idem_repeats_return_the_first(tmp_path):
    clock = Clock()
    a, _ = device(tmp_path, 'a', clock)
    first = post(a, 'one', idem='req-1')
    clock.now -= 60
    second = post(a, 'two')
    again = post(a, 'one again', idem='req-1')
    assert again == first
    assert [r['seq'] for r in a.rows(a.origin)] == [1, 2]
    assert first['hlc'][2] == a.origin and rules.row_id(first) == f'{a.origin}:1'
    assert rules.total_order(second) > rules.total_order(first)
    assert second['prev'] == first['hash']


def test_a_sealed_journal_copies_through_its_last_head_and_a_second_copy_adds_nothing(tmp_path):
    a, _ = device(tmp_path, 'a')
    b, _ = device(tmp_path, 'b')
    for text in ('one', 'two', 'three'):
        post(a, text)
    assert a.seal()['kind'] == 'head' and a.seal() is None
    sent = a.rows(a.origin)
    stored = b.ingest(a.origin, sent)
    assert [r['seq'] for r in stored] == [1, 2, 3, 4]
    assert b.vector() == a.vector()
    assert b.ingest(a.origin, sent) == []
    assert b.rows(a.origin) == sent


def test_rows_after_the_last_head_are_not_stored(tmp_path):
    a, _ = device(tmp_path, 'a')
    b, _ = device(tmp_path, 'b')
    post(a, 'one')
    a.seal()
    post(a, 'unsigned tail')
    assert [r['seq'] for r in b.ingest(a.origin, a.rows(a.origin))] == [1, 2]
    assert [r['seq'] for r in a.trusted(a.origin)] == [1, 2]


def test_a_forged_head_signature_marks_the_origin_damaged_and_stores_nothing(tmp_path):
    a, _ = device(tmp_path, 'a')
    b, _ = device(tmp_path, 'b')
    post(a, 'one')
    a.seal()
    rows = copy.deepcopy(a.rows(a.origin))
    outsider = Signer.generate()
    head = rows[-1]
    head['body']['sig'] = rules.base64.b64encode(outsider.sign_bytes(
        rules.head_message('', a.origin, 1, rows[0]['hash']))).decode()
    from poolhouse.workspace.chain import _digest
    head['hash'] = _digest(head['prev'], head)
    with pytest.raises(rules.Damaged, match='bad signature'):
        b.ingest(a.origin, rows, authoritative=True)
    assert b.rows(a.origin) == [] and a.origin in b.damaged()
    with pytest.raises(rules.Damaged, match='refused'):
        b.ingest(a.origin, a.rows(a.origin), authoritative=True)


def test_a_head_signed_by_another_key_than_the_pinned_one_is_refused(tmp_path):
    a, _ = device(tmp_path, 'a')
    b, _ = device(tmp_path, 'b')
    post(a, 'one')
    a.seal()
    b.ingest(a.origin, a.rows(a.origin))
    outsider = Signer.generate()
    a.signer = lambda: outsider
    post(a, 'two')
    a.seal()
    with pytest.raises(rules.Damaged, match='different key'):
        b.ingest(a.origin, a.rows(a.origin))
    assert len(b.rows(a.origin)) == 2


def test_a_truncated_or_gapped_journal_is_not_stored(tmp_path):
    a, _ = device(tmp_path, 'a')
    b, _ = device(tmp_path, 'b')
    for text in ('one', 'two'):
        post(a, text)
    a.seal()
    post(a, 'three')
    a.seal()
    rows = a.rows(a.origin)
    assert len(rows) == 5
    with pytest.raises(rules.Gap):
        b.ingest(a.origin, rows[2:])
    assert [r['seq'] for r in b.ingest(a.origin, rows[:4])] == [1, 2, 3]
    assert [r['seq'] for r in b.ingest(a.origin, rows)] == [4, 5]
    c, _ = device(tmp_path, 'c')
    with pytest.raises(rules.Damaged):
        c.ingest(a.origin, [rows[0], *rows[2:]])


def test_replaying_an_older_journal_changes_nothing_and_a_fork_is_refused(tmp_path):
    a, _ = device(tmp_path, 'a')
    b, _ = device(tmp_path, 'b')
    for text in ('one', 'two'):
        post(a, text)
    a.seal()
    old = a.rows(a.origin)
    post(a, 'three')
    a.seal()
    b.ingest(a.origin, a.rows(a.origin))
    before = b.rows(a.origin)
    assert b.ingest(a.origin, old) == []
    assert b.rows(a.origin) == before
    forked = copy.deepcopy(old)
    forked[1]['body']['text'] = 'rewritten'
    from poolhouse.workspace.chain import _digest
    forked[1]['hash'] = _digest(forked[1]['prev'], forked[1])
    with pytest.raises(rules.Damaged, match='differs'):
        b.ingest(a.origin, forked)
    assert b.rows(a.origin) == before


def row(origin, seq, wall, counter, **fields):
    return {'origin': origin, 'seq': seq, 'hlc': [wall, counter, origin], 'kind': 'message',
            'actor': 'x', 'idem': '', 'body': {}, **fields}


def test_merge_orders_by_clock_then_origin_then_sequence_whatever_order_it_is_given():
    a = [row('a', 1, 10, 0), row('a', 2, 20, 0), row('a', 3, 20, 0)]
    b = [row('b', 1, 10, 0), row('b', 2, 20, 1)]
    forward = rules.merge({'a': a, 'b': b})
    backward = rules.merge({'b': list(reversed(b)), 'a': list(reversed(a))})
    assert forward == backward
    assert [(r['origin'], r['seq']) for r in forward] == [('a', 1), ('b', 1), ('a', 2), ('a', 3), ('b', 2)]


def test_merge_keeps_one_row_per_origin_actor_and_idem_and_drops_heads():
    a = [row('a', 1, 10, 0, idem='r1'), row('a', 2, 11, 0, kind='head'), row('a', 3, 12, 0, idem='r1')]
    b = [row('b', 1, 12, 0, idem='r1'), row('b', 2, 13, 0, actor='y', idem='r1')]
    merged = rules.merge({'a': a, 'b': b})
    assert [(r['origin'], r['seq']) for r in merged] == [('a', 1), ('b', 1), ('b', 2)]


def test_a_row_far_ahead_of_the_local_clock_is_held_back_and_does_not_move_the_local_clock(tmp_path):
    clock = Clock()
    a, _ = device(tmp_path, 'a', clock)
    b, _ = device(tmp_path, 'b', clock)
    post(a, 'one')
    a.seal()
    far = clock.now + 3600
    a.clock = lambda: far
    post(a, 'from the future')
    a.seal()
    rows = a.rows(a.origin)
    b.ingest(a.origin, rows)
    assert rules.held_back(rows[2], int(clock.now * 1000)) and not rules.held_back(rows[0], int(clock.now * 1000))
    mine = post(b, 'now')
    assert mine['hlc'][0] < int(far * 1000)


def forge(journals, genuine, extra):
    """Rows a relay builds after ``genuine``: the chain hashes are unkeyed, so anyone can."""
    from poolhouse.workspace.chain import _digest
    out, prev = [], genuine[-1]
    for seq, (kind, body) in enumerate(extra, start=len(genuine) + 1):
        made = {'v': 1, 'origin': prev['origin'], 'hlc': [prev['hlc'][0], 0, prev['origin']], 'kind': kind,
                'actor': 'claude' if kind != 'head' else '', 'idem': '', 'body': body, 'seq': seq,
                'prev': prev['hash'], 'ts': 1.0}
        made['hash'] = _digest(prev['hash'], made)
        out.append(made)
        prev = made
    return out


def test_a_relay_cannot_append_rows_under_a_replayed_head(tmp_path):
    v, _ = device(tmp_path, 'v')
    victim, _ = device(tmp_path, 'victim')
    post(v, 'genuine1')
    post(v, 'genuine2')
    v.seal()
    genuine = v.rows(v.origin)
    old_head = dict(genuine[-1]['body'])
    forged = forge(v, genuine, [('message', {'body': 'FORGED'}), ('head', old_head)])
    with pytest.raises(rules.Damaged, match='does not sign the row before'):
        victim.ingest(v.origin, genuine + forged)
    assert victim.rows(v.origin) == []
    assert victim.ingest(v.origin, genuine, authoritative=True) == genuine


def test_a_head_must_sign_the_row_directly_before_it(tmp_path):
    v, key = device(tmp_path, 'v')
    victim, _ = device(tmp_path, 'victim')
    post(v, 'one')
    v.seal()
    genuine = v.rows(v.origin)
    between = forge(v, genuine, [('message', {'body': 'unsigned row between'})])
    target = genuine[-2]
    body = {'head': target['seq'], 'hash': target['hash'], 'public': genuine[-1]['body']['public'],
            'sig': rules.base64.b64encode(key.sign_bytes(rules.head_message('', v.origin, target['seq'], target['hash']))).decode()}
    late = forge(v, genuine + between, [('head', body)])
    with pytest.raises(rules.Damaged):
        victim.ingest(v.origin, genuine + between + late)


@pytest.mark.parametrize('name', ['../../../escaped', '/tmp/absolute-origin', 'A' * 32, 'g' * 32, 'a' * 31, 'a' * 33, ''])
def test_an_origin_that_is_not_32_lower_case_hex_never_reaches_the_file_system(tmp_path, name):
    victim, _ = device(tmp_path / 'one' / 'two', 'victim')
    with pytest.raises(ValueError, match='not a journal id'):
        victim.ingest(name, [{'seq': 1}])
    with pytest.raises(ValueError, match='not a journal id'):
        victim.log(name)
    assert sorted(p.name for p in tmp_path.rglob('*') if p.suffix == '.jsonl') == []


def test_the_upper_case_form_of_this_devices_origin_cannot_reach_its_journal(tmp_path):
    a, _ = device(tmp_path, 'a')
    post(a, 'mine')
    before = a.rows(a.origin)
    with pytest.raises(ValueError, match='not a journal id'):
        a.ingest(a.origin.upper(), before)
    assert a.rows(a.origin) == before and [p.name for p in a.directory.glob('*.jsonl')] == [a.origin + '.jsonl']


def test_a_forged_copy_marks_an_origin_damaged_only_when_its_owner_presented_it(tmp_path):
    a, _ = device(tmp_path, 'a')
    b, _ = device(tmp_path, 'b')
    post(a, 'one')
    a.seal()
    rows = copy.deepcopy(a.rows(a.origin))
    rows[-1]['body']['sig'] = rules.base64.b64encode(b'\x00' * 64).decode()
    from poolhouse.workspace.chain import _digest
    rows[-1]['hash'] = _digest(rows[-1]['prev'], rows[-1])
    with pytest.raises(rules.Damaged):
        b.ingest(a.origin, rows, authoritative=False)
    assert b.damaged() == {}
    with pytest.raises(rules.Damaged):
        b.ingest(a.origin, rows, authoritative=True)
    assert list(b.damaged()) == [a.origin]
    b.forgive(a.origin)
    assert b.damaged() == {} and len(b.ingest(a.origin, a.rows(a.origin), authoritative=True)) == 2


def test_an_origin_bound_by_a_relay_is_rebound_to_the_key_its_owner_presents(tmp_path):
    a, _ = device(tmp_path, 'a')
    squatter = Signer.generate()
    owner = Signer.generate()
    a.signer = lambda: squatter
    post(a, 'squatted')
    a.seal()
    squat_rows = a.rows(a.origin)
    victim, _ = device(tmp_path, 'victim')
    victim.ingest(a.origin, squat_rows, authoritative=False)
    real, _ = device(tmp_path, 'real', key=owner)
    real._origin = a.origin
    post(real, 'the owner')
    real.seal()
    took = victim.ingest(a.origin, real.rows(a.origin), authoritative=True)
    assert [r['body'].get('text') for r in took if r['kind'] == 'message'] == ['the owner']
    assert [r['body'].get('text') for r in victim.rows(a.origin) if r['kind'] == 'message'] == ['the owner']


def test_a_device_holds_no_more_journals_than_the_origin_cap(tmp_path, monkeypatch):
    from poolhouse.workspace import journal
    monkeypatch.setattr(journal, 'MAX_ORIGINS', 2)
    victim, _ = device(tmp_path, 'victim')
    post(victim, 'own')
    for name in ('x', 'y'):
        other, _ = device(tmp_path, name)
        post(other, name)
        other.seal()
        if name == 'x':
            victim.ingest(other.origin, other.rows(other.origin))
        else:
            with pytest.raises(rules.Quota):
                victim.ingest(other.origin, other.rows(other.origin))
