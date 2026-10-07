"""Device membership reads serialize with graph writers across processes."""

import json
import subprocess
import sys

import pytest

HOLDER = """
import sys
from pathlib import Path
from ml_stack.graph.store import GraphStore
from ml_stack.workspace.chain import held
base = Path(sys.argv[1])
with held(base / 'device-accounts.lock'), GraphStore(base / 'device-accounts.db') as graph:
    graph.upsert_node({'id': 'device:test', 'kind': 'device-account', 'label': 'worker',
                       'attrs': {'base_id': 'worker', 'device_id': 'test'}})
    graph.upsert_node({'id': 'agent:worker', 'kind': 'agent', 'label': 'worker'})
    graph.upsert_edge({'source': 'agent:worker', 'target': 'device:test', 'rel': 'device-member'})
    print('ready', flush=True)
    sys.stdin.readline()
"""

READER = """
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from ml_stack.workspace.device_accounts import account_for
print('reading', flush=True)
print(json.dumps(account_for(SimpleNamespace(base=Path(sys.argv[1])), 'worker')), flush=True)
"""


@pytest.mark.slow
def test_account_read_waits_for_the_device_graph_writer(tmp_path):
    with subprocess.Popen([sys.executable, '-c', HOLDER, str(tmp_path)], stdin=subprocess.PIPE,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as holder:
        try:
            assert holder.stdout.readline().strip() == 'ready'
            with subprocess.Popen([sys.executable, '-c', READER, str(tmp_path)],
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as reader:
                try:
                    assert reader.stdout.readline().strip() == 'reading'
                    with pytest.raises(subprocess.TimeoutExpired):
                        reader.communicate(timeout=0.2)
                    holder.stdin.write('release\n')
                    holder.stdin.flush()
                    output, error = reader.communicate(timeout=20)
                    assert reader.returncode == 0, error
                    assert json.loads(output) == {'base_id': 'worker', 'device_id': 'test',
                                                  'members': ['worker']}
                finally:
                    if reader.poll() is None:
                        reader.kill()
                        reader.wait()
            _, error = holder.communicate(timeout=20)
            assert holder.returncode == 0, error
        finally:
            if holder.poll() is None:
                holder.kill()
                holder.wait()
