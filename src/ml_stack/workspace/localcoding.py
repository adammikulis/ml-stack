"""The maintained detached coding entry point runs canonical task assignments."""

import os
import sys

from ml_stack.workspace import localagent, task_worker
from ml_stack.workspace.service import Workspace

__all__ = ['run_detached']


def run_detached(argv=None):
    """Run one saved coding worker against the canonical TaskBoard."""
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        sys.stderr.write('usage: python -m ml_stack.workspace.localcoding NAME\n')
        return 2
    os.environ['ML_STACK_AGENT'] = '1'
    os.environ['ML_STACK_NONINTERACTIVE'] = '1'
    return task_worker.run(Workspace(), localagent.check_name(args[0])) or 0


if __name__ == '__main__':
    raise SystemExit(run_detached())
