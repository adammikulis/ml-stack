"""A write under the real state root made by another agent's process is not a test's escape."""

import subprocess
import sys
import time

from tests import conftest
from tests.foreign_writers import held_by_foreign

HOLD = "import sys,time\nf=open(sys.argv[1],'a')\nprint('up',flush=True)\ntime.sleep(60)\n"


def _orphan(path):
    """A process outside this session's tree: its parent exits at once, so init adopts it."""
    code = ("import subprocess,sys;"
            f"subprocess.Popen([sys.executable,'-c',{HOLD!r},sys.argv[1]],start_new_session=True,"
            "stdout=subprocess.DEVNULL)")
    subprocess.run([sys.executable, "-c", code, str(path)], check=True)


def test_a_file_another_process_holds_open_is_not_this_runs_write(tmp_path):
    held, mine = tmp_path / "taskd" / "held.json", tmp_path / "taskd" / "mine.json"
    held.parent.mkdir()
    before = conftest.file_mtimes(tmp_path, attribute_external=False)
    mine.write_text("x")
    _orphan(held)
    for _ in range(100):
        if held.exists() and held_by_foreign(tmp_path, [held.relative_to(tmp_path)]):
            break
        time.sleep(0.1)
    after = conftest.file_mtimes(tmp_path, attribute_external=False)
    assert conftest.real_state_changes(tmp_path, before, after) == ["taskd/mine.json"]
    held.write_text("again")
    assert held_by_foreign(tmp_path, [held.relative_to(tmp_path)])
