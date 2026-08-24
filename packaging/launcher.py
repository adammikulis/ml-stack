"""Entry point for the bundled build."""

import multiprocessing
import sys

if __name__ == "__main__":
    multiprocessing.freeze_support()
    from ml_stack.fleet.launch import main
    sys.exit(main())
