"""Serving an MLX model with tree speculative decoding, behind the same lease as llama-server.

A model named ``mlx:owner/repo``, or a local directory of MLX weights, is served by
`MlxTreeBackend`: a Python process running `ml_stack.serve.mlx_tree_server` with the weights
and a drafter. The spec's ``draft`` names the drafter -- a DFlash or DSpark head, an MTP
head, ``ngram``, or nothing -- and ``spec_draft_max`` the most tree nodes a pass verifies.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from ml_stack import home
from ml_stack.hub import room
from ml_stack.serve.backend import (
    Lease,
    ServerBackend,
    ServerFailed,
    ServerInfo,
    ServerSpec,
    claim_port,
    launch,
    log_dir,
)
from ml_stack.serve.preflight import Check, Report
from ml_stack.spec import LAYOUTS

__all__ = ["PREFIX", "MlxTreeBackend", "drafter_of", "is_mlx", "located", "report_for",
           "resident_bytes"]

PREFIX = "mlx:"
#: the node cap when a spec names none
MAX_NODES = 32


def is_mlx(model: str | Path) -> bool:
    """Whether ``model`` names MLX weights: ``mlx:owner/repo`` or a directory with a config."""
    text = str(model)
    if text.startswith(PREFIX):
        return True
    where = home.expand(text)
    return where.is_dir() and (where / "config.json").is_file()


def located(model: str | Path) -> Path | None:
    """The weights directory ``model`` names, when it is on this machine."""
    text = str(model).removeprefix(PREFIX)
    where = home.expand(text)
    if where.is_dir():
        return where
    try:
        from huggingface_hub import snapshot_download
        from huggingface_hub.errors import LocalEntryNotFoundError
    except ImportError:
        return None
    try:
        return Path(snapshot_download(text, local_files_only=True))
    except (LocalEntryNotFoundError, ValueError, OSError):
        return None


def drafter_of(draft: str | Path | None) -> str:
    """The drafter kind a spec's ``draft`` names: ``dflash``, ``mtp``, ``ngram`` or ``none``."""
    text = str(draft or "").strip()
    if not text:
        return "none"
    if text == "ngram":
        return "ngram"
    name = Path(text).name.lower()
    if "dflash" in name or "dspark" in name:
        return "dflash"
    if "mtp" in name:
        return "mtp"
    raise ServerFailed(f"no tree drafter reads {text!r}: name a DFlash, DSpark or MTP head, "
                       "or 'ngram'")


#: the tensors a load memory-maps from disk instead of holding in memory
MAPPED = ".ple.ple_embedding.ngram_embedding."


def resident_bytes(where: Path | None) -> int:
    """Bytes of the safetensors under ``where`` a load holds in memory."""
    if where is None:
        return 0
    held = 0
    for shard in where.glob("*.safetensors"):
        with shard.open("rb") as stream:
            header = json.loads(stream.read(int.from_bytes(stream.read(8), "little")))
        held += sum(entry["data_offsets"][1] - entry["data_offsets"][0]
                    for name, entry in header.items()
                    if name != "__metadata__" and MAPPED not in name)
    return held


def report_for(spec: ServerSpec, *, limit_bytes: int = 0) -> Report:
    """What can be known before the load: weights present, a layout for them, and the fit."""
    report = Report()
    weights = located(spec.model)
    report.checks.append(Check("weights", True,
                               str(weights) if weights else
                               f"{spec.model} is not on this machine; the load fetches it"))
    if weights is not None:
        kind = json.loads((weights / "config.json").read_text(encoding="utf-8")).get(
            "model_type", "")
        known = kind in LAYOUTS
        report.checks.append(Check("layout", known, f"model_type {kind!r}"
                                   + ("" if known else " has no tree-verification layout")))
    try:
        kind = drafter_of(spec.draft)
        report.checks.append(Check("drafter", True, kind))
    except ServerFailed as why:
        report.checks.append(Check("drafter", False, str(why)))
        kind = "none"
    report.weights_bytes = resident_bytes(weights)
    drafted = resident_bytes(located(spec.draft)) if kind in ("dflash", "mtp") else 0
    if limit_bytes and report.weights_bytes:
        wanted = report.weights_bytes + drafted
        report.checks.append(Check("fit", wanted <= limit_bytes,
                                   f"{wanted / 2**30:.1f}G of weights against "
                                   f"{limit_bytes / 2**30:.1f}G this machine allows"))
    return report


class MlxTreeBackend(ServerBackend):
    """`ml_stack.serve.mlx_tree_server` in a child process: one model, one drafter, one slot."""

    name = "mlx-tree"

    def command(self, spec: ServerSpec) -> list[str]:
        kind = drafter_of(spec.draft)
        argv = [sys.executable, "-m", "ml_stack.serve.mlx_tree_server", str(spec.model),
                "--port", str(spec.port), "--context", str(spec.context),
                "--drafter", kind,
                "--max-nodes", str(spec.spec_draft_max or MAX_NODES)]
        if kind in ("dflash", "mtp"):
            argv += ["--drafter-model", str(spec.draft).removeprefix(PREFIX)]
        return argv

    def start(self, spec: ServerSpec, *, lease: Lease, timeout: float = 300.0,
              **starting: bool) -> ServerInfo:
        """Launch and wait until the model has loaded and answers. Raises ``ServerFailed``.

        Of the lease's ``starting`` switches only ``preflight`` applies; the load measures
        its own cost curve, which is the warm-up.
        """
        argv = self.command(spec)
        claim_port(spec, lease)
        if starting.get("preflight", True):
            report = report_for(spec, limit_bytes=room())
            if not report.ok:
                raise ServerFailed(report.said())
        logs = log_dir()
        logs.mkdir(parents=True, exist_ok=True)
        log_path = logs / f"mlx-tree-{spec.port}.log"
        process, base_url, load_s = launch(argv, port=spec.port, log_path=log_path,
                                           timeout=timeout, env=dict(os.environ))
        return ServerInfo(base_url=base_url, port=spec.port, pid=process.pid, backend=self.name,
                          log_path=log_path, load_s=load_s, process=process)
