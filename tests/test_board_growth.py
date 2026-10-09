"""Message page bounds and contiguous cursor behavior."""

import json

from poolhouse.workspace.board_pages import PAGE_BYTES, page


def test_large_history_pages_preserve_contiguous_forward_and_older_windows():
    rows = [{'seq': seq, 'body': 'message'} for seq in range(1, 100001)]
    shown = []

    def show(row):
        shown.append(row['seq'])
        return row

    latest = page(rows, show, 100)
    assert latest['older'] == 99901 and latest['newer'] == 100000
    assert latest['has_older'] and not latest['has_newer']
    older = page(rows, show, 100, before=latest['older'])
    assert [row['seq'] for row in older['messages']] == list(range(99801, 99901))
    forward = page(rows, show, 100, after=99800)
    assert [row['seq'] for row in forward['messages']] == list(range(99801, 99901))
    assert forward['has_newer']
    assert len(shown) == 300
    shown.clear()
    assert page(rows, show, 0)['messages'] == []
    assert not shown


def test_page_byte_cap_keeps_cursor_contiguous_for_unicode_messages():
    rows = [{'seq': seq, 'body': '🌊' * 16000} for seq in range(1, 20)]
    result = page(rows, lambda row: row)
    assert len(json.dumps(result).encode()) < PAGE_BYTES + 1024
    assert result['messages'] and result['has_older']
    assert result['newer'] == 19
    earlier = page(rows, lambda row: row, before=result['older'])
    assert earlier['newer'] == result['older'] - 1


class Counted:
    """A million-row log that counts how many rows a page asks it for."""

    def __init__(self, size):
        self.size, self.reads = size, 0

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        if not 0 <= index < self.size:
            raise IndexError(index)
        self.reads += 1
        return {'seq': index + 1, 'body': 'message'}


def test_choosing_a_page_touches_only_the_rows_near_its_cursor():
    rows = Counted(1_000_000)
    for cursor in ({}, {'before': 500_000}, {'after': 500_000}, {'after': 999_950}, {'before': 7}):
        rows.reads = 0
        got = page(rows, lambda row: row, 100, **cursor)
        assert got['messages'] and rows.reads <= 100 + 2 * 21 + 2, (cursor, rows.reads)
    first = page(rows, lambda row: row, 100, before=7)
    assert [row['seq'] for row in first['messages']] == [1, 2, 3, 4, 5, 6] and not first['has_older']
    assert first['has_newer']
    last = page(rows, lambda row: row, 100, after=999_950)
    assert last['newer'] == 1_000_000 and not last['has_newer'] and last['has_older']
