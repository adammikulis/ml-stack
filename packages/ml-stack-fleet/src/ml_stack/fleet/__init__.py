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
    create_cluster_key,
    derive_token,
    discover,
    key_path,
    load_cluster_key,
)
from ml_stack.fleet.remote import Peer, PeerError, sha256_file

__all__ = [
    "REPORT_GROUP",
    "Advertiser",
    "Beacon",
    "DaemonError",
    "DiscoveryError",
    "Job",
    "JobRunner",
    "Peer",
    "PeerError",
    "create_cluster_key",
    "derive_token",
    "device_report",
    "discover",
    "key_path",
    "load_cluster_key",
    "registered_reports",
    "resolve_report",
    "safe_relpath",
    "serve_forever",
    "sha256_file",
    "stdlib_device_report",
]
