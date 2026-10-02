"""Process control is not optional: a missing psutil is an import error, never a no-op."""

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _blocked(module: str, body: str) -> subprocess.CompletedProcess:
    program = (
        "import sys\n"
        "sys.modules['psutil'] = None\n"
        f"sys.path.insert(0, {str(REPO / 'src')!r})\n"
        f"import {module}\n"
        f"{body}\n"
    )
    return subprocess.run([sys.executable, "-c", program], capture_output=True, text=True)


def test_process_control_refuses_to_import_without_psutil():
    done = _blocked("ml_stack.serve.process", "print('imported')")
    assert done.returncode != 0 and "psutil" in done.stderr
    assert "imported" not in done.stdout


def test_port_scanning_refuses_to_run_without_psutil():
    done = _blocked("ml_stack.serve.ports", "print(ml_stack.serve.ports.server_pids_on_port(1))")
    assert done.returncode != 0 and "psutil" in done.stderr
    assert done.stdout.strip() != "[]"


def test_a_live_child_is_killed_with_its_own_children():
    parent = subprocess.Popen([sys.executable, "-c",
                               "import subprocess,sys,time;"
                               "subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)']);"
                               "time.sleep(60)"])
    try:
        import time

        import psutil

        from ml_stack.serve.process import kill_process_tree
        deadline = time.monotonic() + 10
        while not psutil.Process(parent.pid).children() and time.monotonic() < deadline:
            time.sleep(0.05)
        acted = kill_process_tree(parent.pid, grace_s=5)
        assert parent.pid in acted and len(acted) == 2
        parent.wait(timeout=10)
    finally:
        if parent.poll() is None:
            parent.kill()
