"""The package root: a version, and a logger that is quiet until the embedder listens."""

import importlib.metadata
import logging
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def test_the_package_states_its_version():
    import poolhouse

    assert poolhouse.__version__ == importlib.metadata.version("poolhouse")


def test_the_package_logger_has_a_null_handler():
    import poolhouse  # noqa: F401

    handlers = logging.getLogger("poolhouse").handlers
    assert any(isinstance(one, logging.NullHandler) for one in handlers)


def test_importing_the_package_prints_nothing():
    program = f"import sys; sys.path.insert(0, {str(REPO / 'src')!r}); import poolhouse"
    done = subprocess.run([sys.executable, "-c", program], capture_output=True, text=True)
    assert done.returncode == 0 and done.stdout == "" and done.stderr == ""


def test_a_record_with_no_listener_does_not_reach_the_last_resort_handler():
    program = (f"import sys, logging; sys.path.insert(0, {str(REPO / 'src')!r}); import poolhouse\n"
               "logging.getLogger('poolhouse.serve').warning('quiet')")
    done = subprocess.run([sys.executable, "-c", program], capture_output=True, text=True)
    assert done.stderr == ""


def test_importing_serve_builds_no_manager():
    program = (f"import sys; sys.path.insert(0, {str(REPO / 'src')!r})\n"
               "import poolhouse.serve, poolhouse.serve.manager as m\n"
               "assert m._DEFAULT is None\n"
               "first = m.default_manager()\n"
               "assert m.default_manager() is first and m._DEFAULT is first\n")
    done = subprocess.run([sys.executable, "-c", program], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr


def test_threads_asking_together_share_one_default_manager():
    import threading

    from poolhouse.serve import manager as manager_module

    manager_module._DEFAULT = None
    got = []
    threads = [threading.Thread(target=lambda: got.append(manager_module.default_manager()))
               for _ in range(8)]
    for one in threads:
        one.start()
    for one in threads:
        one.join()
    assert len({id(one) for one in got}) == 1
