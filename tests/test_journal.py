"""Per-device signed journals: appending, sealing, copying between devices and merging."""

import copy

import pytest

from ml_stack.fleet.onboard.manifest import Signer
from ml_stack.workspace import journal_merge as rules
from ml_stack.workspace.journal import Journals


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
    from ml_stack.workspace.chain import _digest
    head['hash'] = _digest(head['prev'], head)
    with pytest.raises(rules.Damaged, match='bad signature'):
        b.ingest(a.origin, rows)
    assert b.rows(a.origin) == [] and a.origin in b.damaged()
    with pytest.raises(rules.Damaged, match='refused'):
        b.ingest(a.origin, a.rows(a.origin))


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
    from ml_stack.workspace.chain import _digest
    forked[1]['hash'] = _digest(forked[1]['prev'], forked[1])
    with pytest.raises(rules.Damaged, match='differs'):
        b.ingest(a.origin, forked)
    assert b.rows(a.origin) == before


def row(origin, seq, wall, counter, actor='x', idem='', kind='message'):
    return {'origin': origin, 'seq': seq, 'hlc': [wall, counter, origin], 'kind': kind,
            'actor': actor, 'idem': idem, 'body': {}}


def test_merge_orders_by_clock_then_origin_then_sequence_whatever_order_it_is_given():
    a = [row('a', 1, 10, 0), row('a', 2, 20, 0), row('a', 3, 20, 0)]
    b = [row('b', 1, 10, 0), row('b', 2, 20, 1)]
    forward = rules.merge({'a': a, 'b': b})
    backward = rules.merge({'b': list(reversed(b)), 'a': list(reversed(a))})
    assert forward == backward
    assert [(r['origin'], r['seq']) for r in forward] == [('a', 1), ('b', 1), ('a', 2), ('a', 3), ('b', 2)]


def test_merge_keeps_one_row_per_actor_and_idem_and_drops_heads():
    a = [row('a', 1, 10, 0, idem='r1'), row('a', 2, 11, 0, kind='head')]
    b = [row('b', 1, 12, 0, idem='r1'), row('b', 2, 13, 0, actor='y', idem='r1')]
    merged = rules.merge({'a': a, 'b': b})
    assert [(r['origin'], r['seq']) for r in merged] == [('a', 1), ('b', 2)]


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
