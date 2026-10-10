"""The upstream commit the llama.cpp patches were last verified against, and a trial apply that
cannot leave a checkout conflicted.

A patch set is a rebase of a few files onto upstream's master, and master moves faster than the
patches do: the day this was written ``git apply --3way`` failed on ``common/speculative.cpp``
and left the managed ``src`` tree half-patched, with conflict markers in it. So the patch
directory carries ``verified.json`` -- the upstream commit the set was last checked to apply to
-- a build lands on that commit unless asked to float (``--upstream-head``), and every build
applies the patches into a throwaway worktree first. Only a set that applies there touches the
real checkout, so a failure leaves ``src`` and ``current`` exactly as they were.
"""

from __future__ import annotations

import json
import re
import shutil
import tempfile
from pathlib import Path

from poolhouse.log import warn
from poolhouse.serve.build_paths import BuildFailed, patch_files, patch_stamp, patches_dir, root

__all__ = ["MANIFEST", "check_patches", "manifest_path", "rebase_help", "verified_upstream"]

MANIFEST = "verified.json"
FULL_SHA = re.compile(r"[0-9a-f]{40}")


def manifest_path() -> Path:
    """The file beside the patches naming the upstream commit they were verified against."""
    return patches_dir() / MANIFEST


def verified_upstream() -> str:
    """The upstream commit the patch set was last verified to apply to, ``""`` when the
    directory has no patches or no manifest (nothing is then pinned: master's tip is built).

    A manifest whose ``patches`` stamp is not the current set's is still followed -- it names
    the best known base -- but is warned about, since the trial apply is what decides.
    """
    path = manifest_path()
    if not patch_files() or not path.is_file():
        return ""
    try:
        data = json.loads(path.read_text())
        commit = str(data["commit"])
    except (ValueError, KeyError, TypeError) as exc:
        raise BuildFailed(f"{path} is not a verified-commit manifest ({exc}); it needs "
                          '{"commit": "<40 hex>", "patches": "<stamp>"}') from None
    if not FULL_SHA.fullmatch(commit):
        raise BuildFailed(f"{path} names {commit!r}, which is not a full upstream commit SHA")
    if data.get("patches") != patch_stamp():
        warn(f"the patches changed since {commit[:9]} was verified: after checking them against "
             f"it, set \"patches\" to \"{patch_stamp()}\" in {path}")
    return commit


def rebase_help(source: Path, rev: str, failed: Path, detail: str) -> str:
    """What failed, what was left alone, and the commands that rebase the patch set."""
    work = f"{source}.rebase"
    order = " ".join(item.name for item in patch_files())
    return "\n".join([
        f"{failed.name} does not apply to upstream {rev}: {detail}",
        f"  {source} and the current build were not touched. To rebase the patches onto it:",
        f"    git -C {source} worktree add {work} {rev}",
        f"    cd {work}",
        f"    for each of {order} (in {patches_dir()}), in that order:",
        "      git apply --3way PATCH     # resolve the conflict markers, then git add the files",
        "      git diff --cached > PATCH && git commit -qm PATCH",
        f"  then set \"commit\" to {rev} and \"patches\" to the stamp the next build warns about, in "
        f"{manifest_path()}.",
        f"  Until then `poolhouse-serve build` lands on the commit in {MANIFEST}, not upstream's head.",
    ])


def _worktree(source: Path, rev: str, git) -> Path:
    base = root()
    base.mkdir(parents=True, exist_ok=True)
    trial = Path(tempfile.mkdtemp(prefix="trial-", dir=base))
    git("worktree", "add", "--detach", str(trial), rev, cwd=source)
    return trial


def check_patches(source: Path, rev: str, git) -> None:
    """Apply every patch, in order, to a throwaway worktree of ``source`` at ``rev``.

    ``git`` is the build's own runner (it raises `BuildFailed`). Raises `BuildFailed` with the
    rebase commands when one does not apply; ``source`` itself is never written to either way.
    """
    files = patch_files()
    if not files:
        return
    trial = _worktree(source, rev, git)
    try:
        for item in files:
            try:
                git("apply", "--3way", str(item.resolve()), cwd=trial)
            except BuildFailed as exc:
                raise BuildFailed(rebase_help(source, rev, item, str(exc))) from None
    finally:
        try:
            git("worktree", "remove", "--force", str(trial), cwd=source)
        except BuildFailed:
            shutil.rmtree(trial, ignore_errors=True)
            git("worktree", "prune", cwd=source)
