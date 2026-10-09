"""Cursor and byte bounded person Board pages.

``rows`` must be ordered by ``seq``, which every log read already is. A page is found by
binary search on the cursor and only the rows it returns are shown, so choosing a page
costs O(log N + limit) however long the conversation has grown; whether an older or newer
page exists is read off the page's position, never by scanning for it.
"""

import json
from bisect import bisect_left, bisect_right

PAGE_BYTES = 256 * 1024
PAGE_ITEMS = 200


def _seq(row):
    return row['seq']


def _size(message):
    return len(json.dumps(message, ensure_ascii=True, separators=(',', ':')).encode()) + 1


def _window(rows, after, before):
    """The ``[start, end)`` indexes of the rows a page may draw on, and the side it fills from."""
    start = bisect_right(rows, after, key=_seq) if after else 0
    end = bisect_left(rows, before, key=_seq) if before else len(rows)
    return start, max(start, end)


def page(rows, show, limit=100, after=0, before=0):
    """Select a contiguous message window and expose its navigation cursors."""
    if after and before:
        raise ValueError('choose after or before, not both')
    limit = min(max(limit, 0), PAGE_ITEMS)
    empty = {'messages': [], 'older': 0, 'newer': 0, 'has_older': False, 'has_newer': False}
    if not limit:
        return empty
    start, end = _window(rows, after, before)
    newest = not after
    indexes = range(end - 1, max(start, end - limit) - 1, -1) if newest else range(start, min(end, start + limit))
    messages, used, taken = [], 0, []
    for index in indexes:
        message = show(rows[index])
        used += _size(message)
        if used > PAGE_BYTES:
            break
        messages.append(message)
        taken.append(index)
    if newest:
        messages.reverse()
    if not messages:
        return empty
    low, high = min(taken), max(taken)
    return {'messages': messages, 'older': messages[0]['seq'], 'newer': messages[-1]['seq'],
            'has_older': low > 0, 'has_newer': high < len(rows) - 1}
