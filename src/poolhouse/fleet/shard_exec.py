"""The process the node starts for one accepted shard, from this device's own checkout.

The node passes one folder. In it: ``spec.json`` (the job: tier, test files, time limit) and
``tree.tgz`` (the sender's tree, whose digest the node already checked). This process checks both
again, unpacks the tree into a scratch checkout, runs the test runner there and writes
``result.json``. It leads the process group the node kills, so the runner stays in it; SIGTERM or
SIGINT cancels the run the way a time limit does.
"""

from __future__ import annotations

import json
import signal
import threading
from pathlib import Path

from . import shard_result, shard_run, shard_spec, shard_tree

REFUSED = 70


def refuse(folder: Path, job: object, why: str) -> int:
    """Write a result that says the job never ran, and why."""
    digest = job.get("tree_sha256", "") if isinstance(job, dict) else ""
    shard = {"id": job.get("id", "") if isinstance(job, dict) else "", "files": [], "tree_sha256": digest}
    result = {**shard_result.build(shard, folder / "none.xml", REFUSED, (0.0, 0.0), why), "state": "failed"}
    (folder / shard_run.RESULT).write_text(json.dumps(result))
    return REFUSED


def run(folder: str) -> int:
    """Run the job in ``folder``; the exit status of the run, or 70 when it was refused."""
    job_dir = Path(folder)
    cancel = threading.Event()
    for name in ("SIGTERM", "SIGINT"):
        signal.signal(getattr(signal, name), lambda *_: cancel.set())
    try:
        job = json.loads((job_dir / "spec.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return refuse(job_dir, None, f"the job is unreadable: {exc}")
    try:
        shard = shard_spec.check_header(job)
        data = (job_dir / "tree.tgz").read_bytes()
        names = [member.name for member in shard_tree.members(data, shard["tree_sha256"])]
        shard_spec.check_present(shard["files"], names)
    except (shard_spec.Refused, shard_tree.TreeError, OSError) as exc:
        return refuse(job_dir, job, str(exc))
    return int(shard_run.run_shard(shard, data, job_dir, cancel=cancel)["exit"])
