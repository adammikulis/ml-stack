"""The other machines on this LAN, and how to run work on them.

Device tier: standard library only. That is load-bearing rather than incidental -- the
box you *drive* the fleet from is usually a laptop with no CUDA, no MLX and no training
stack at all, and it should not need one to ask a GPU box what it is doing.

Three pieces, each usable on its own:

**Discovery** finds peers without configuration. A daemon holding the cluster key
announces itself over UDP; a client holding the same key finds it and derives the bearer
token from that key rather than being handed one. So a peer you can find is a peer you
can already drive, and the secret never crosses the wire.

**The daemon** runs jobs, and moves files in and out. It executes commands you send it,
which is the point and also remote code execution: a token is mandatory, and this belongs
on a trusted LAN and nowhere else.

**The peer client** drives one daemon. Uploads and downloads resume and are verified by
digest, because a 2GB dataset over a home network will be interrupted eventually.

    from ml_stack.fleet import Peer

    rtx = Peer.find_one(require="cuda")
    rtx.push("data/packed/train.npy", "data/train.npy")
    job = rtx.submit(["python", "-m", "train.run", "--steps", "30000"])
    rtx.wait(job["id"], on_metric=print)
"""

from __future__ import annotations

from ml_stack.fleet.daemon import (
    REPORT_GROUP,
    DaemonError,
    Job,
    JobRunner,
    device_report,
    load_or_create_token,
    make_handler,
    registered_reports,
    resolve_report,
    safe_relpath,
    serve_forever,
    stdlib_device_report,
)
from ml_stack.fleet.discovery import (
    Advertiser,
    Beacon,
    DiscoveryError,
    MIN_PASSPHRASE,
    create_cluster_key,
    derive_token,
    discover,
    check_passphrase,
    cluster_group,
    group_path,
    in_cluster,
    join_cluster,
    key_from_passphrase,
    key_path,
    load_cluster_key,
)
from ml_stack.fleet.bench import BENCH_KIND, calibrate, measure
from ml_stack.fleet.pool import (
    Candidate,
    Requires,
    Score,
    candidates,
    choose,
    eligible,
    soonest,
)
from ml_stack.fleet.rates import Rates
from ml_stack.fleet.remote import Peer, PeerError, sha256_file
from ml_stack.fleet.work import Placement, Unit, run

__all__ = [
    "BENCH_KIND",
    "MIN_PASSPHRASE",
    "REPORT_GROUP",
    "Advertiser",
    "Beacon",
    "Candidate",
    "DaemonError",
    "DiscoveryError",
    "Job",
    "JobRunner",
    "Peer",
    "PeerError",
    "Placement",
    "Rates",
    "Requires",
    "Score",
    "Unit",
    "calibrate",
    "candidates",
    "choose",
    "create_cluster_key",
    "derive_token",
    "device_report",
    "discover",
    "eligible",
    "check_passphrase",
    "cluster_group",
    "group_path",
    "in_cluster",
    "join_cluster",
    "key_from_passphrase",
    "key_path",
    "load_cluster_key",
    "load_or_create_token",
    "make_handler",
    "measure",
    "registered_reports",
    "resolve_report",
    "run",
    "safe_relpath",
    "serve_forever",
    "sha256_file",
    "soonest",
    "stdlib_device_report",
]
