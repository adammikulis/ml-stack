"""The person's redacted activity history behind Fleet authorization."""

from collections import deque
from dataclasses import asdict

from poolhouse.activity import writer
from poolhouse.activity.schema import Entry


def route(request) -> bool:
    if request.path != '/ui/history/events':
        return False
    if request.method != 'GET':
        request.send(405, {'error': 'history is read-only'})
        return True
    try:
        rows = deque(maxlen=500)
        unreadable = 0
        for entry in writer.log().entries():
            if isinstance(entry, Entry):
                rows.append({'id': entry.id, **asdict(entry)})
            else:
                unreadable += 1
        request.send(200, {'events': list(reversed(rows)), 'limit': 500,
                           'unreadable': unreadable, 'drops': writer.drops()})
    except writer.FAILURES:
        request.send(503, {'error': 'Activity history cannot be read. Check the activity log and keystore status.'})
    return True
