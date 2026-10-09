"""Entry point for the frozen device daemon and Python subprocess commands."""

import multiprocessing
import sys

if __name__ == "__main__":
    multiprocessing.freeze_support()
    from ml_stack.fleet.frozen_dispatch import main
    sys.exit(main())
