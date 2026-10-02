"""The package root: a version, and a logger that is quiet until the embedder listens."""

import importlib.metadata
import logging
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def test_the_package_states_its_version():
    import ml_stack

    assert ml_stack.__version__ == importlib.metadata.version("ml-stack")


def test_the_package_logger_has_a_null_handler():
    import ml_stack  # noqa: F401

    handlers = logging.getLogger("ml_stack").handlers
    assert any(isinstance(one, logging.NullHandler) for one in handlers)


def test_importing_the_package_prints_nothing():
    program = f"import sys; sys.path.insert(0, {str(REPO / 'src')!r}); import ml_stack"
    done = subprocess.run([sys.executable, "-c", program], capture_output=True, text=True)
    assert done.returncode == 0 and done.stdout == "" and done.stderr == ""


def test_a_record_with_no_listener_does_not_reach_the_last_resort_handler():
    program = (f"import sys, logging; sys.path.insert(0, {str(REPO / 'src')!r}); import ml_stack\n"
               "logging.getLogger('ml_stack.serve').warning('quiet')")
    done = subprocess.run([sys.executable, "-c", program], capture_output=True, text=True)
    assert done.stderr == ""
