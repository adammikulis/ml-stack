"""Entry point for the headless bundle: daemon only, no window."""

import multiprocessing
import sys

if __name__ == "__main__":
    multiprocessing.freeze_support()
    if sys.argv[1:3] == ["-m", "ml_stack.harnesshook"]:
        from ml_stack import harnesshook
        sys.excepthook = harnesshook._block
        sys.exit(harnesshook.run(sys.argv[3:]))
    from ml_stack.fleet.launch import main
    sys.exit(main())
