"""Native task tool admission counters."""

import json
import sys
from pathlib import Path

from poolhouse.workspace.chain import held


def admit(path):
    """Count one native tool admission or refuse an exhausted task limit."""
    with held(path.with_suffix('.lock')):
        state = json.loads(path.read_text(encoding='utf-8'))
        if (type(state) is not dict or set(state) != {'version', 'calls', 'limit'}
                or type(state['version']) is not int or state['version'] != 1
                or type(state.get('calls')) is not int
                or state['calls'] < 0 or (state.get('limit') is not None and
                    (type(state['limit']) is not int or state['limit'] <= 0))):
            raise ValueError('invalid native tool admission counter')
        if state['limit'] is not None and state['calls'] >= state['limit']:
            return False
        state['calls'] += 1
        path.write_text(json.dumps(state), encoding='utf-8')
        return True


def run_detached():
    if len(sys.argv) != 2:
        return 2
    try:
        allowed = admit(Path(sys.argv[1]))
    except (OSError, ValueError):
        allowed = False
    if not allowed:
        sys.stderr.write('Native task tool call limit reached\n')
    return 0 if allowed else 2


if __name__ == '__main__':
    raise SystemExit(run_detached())
