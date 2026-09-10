"""Build llama-server from its own master, since a release lags it by an architecture or two.

Measured on one machine the day this was written: the newest homebrew bottle
(``llama.cpp 0.3.0``, `brew outdated` empty) read ``gemma4`` and ``qwen3moe`` but not
``qwen4exp`` -- Qwen3.8-Flash-Next's architecture -- and lacked ``--kv-unified-per-slot``
besides. A hand-built binary in one person's ``~/.local/llama-next`` fixed that for one
machine, selected only by one application setting ``LLAMA_CPP_SERVER`` for one model name.
Every other bench run, on every other machine, kept loading the stale bottle without saying
so.

``ml-stack-serve build`` clones or fast-forwards llama.cpp's own master into a managed
directory and builds it (``--from source``), or downloads the newest GitHub release with an
asset for this machine (``--from release``, the fallback when no compiler is on PATH -- the
only path on a machine with no toolchain, such as most Windows installs). Either way, the
new binary is installed into its own flat, versioned directory and is trusted only once it
answers ``--help`` and reads every architecture the previous build did; only then does
``current`` -- a symlink (a junction on Windows, where a symlink needs a privilege a plain
account may not have) -- point at it. A build that fails verification leaves ``current``
untouched, which is what makes ``--persist``'s weekly, unattended rerun safe.

The steps live beside this file: `build_paths` (where a build lives, and its patches),
`build_platform` (the small platform facts a build needs), `build_source` (building
master or a fork's ref from source), `build_release` (downloading a release instead),
`build_verify` (trusting a build, then switching to it), `build_report` (what is built,
and rolling back), `build_persist` (a weekly unattended rerun). This module dispatches
``ml-stack-serve build`` to them.
"""

from __future__ import annotations

from ml_stack.log import say, warn
from ml_stack.serve.build_paths import (
    BuildFailed,
    builds_dir,
    current_link,
    named_dir,
    named_src_dir,
    patch_files,
    patch_stamp,
    patches_dir,
    root,
    src_dir,
)
from ml_stack.serve.build_persist import PERSIST_PLIST, PERSIST_TASK, WEEK_SECONDS, cmd_persist
from ml_stack.serve.build_platform import can_build_from_source
from ml_stack.serve.build_release import build_from_release, build_from_release_named
from ml_stack.serve.build_report import cmd_list, do_rollback, report
from ml_stack.serve.build_source import build_from_source, build_from_source_named
from ml_stack.serve.build_verify import cmd_adopt, verify_and_switch

__all__ = [
    "PERSIST_PLIST",
    "PERSIST_TASK",
    "WEEK_SECONDS",
    "BuildFailed",
    "builds_dir",
    "cmd_build",
    "current_link",
    "named_dir",
    "named_src_dir",
    "patch_files",
    "patch_stamp",
    "patches_dir",
    "root",
    "src_dir",
]


def cmd_build(args) -> int:
    if args.check:
        return report(args)
    if getattr(args, "list", False):
        return cmd_list()
    if args.rollback:
        return do_rollback()
    if args.persist:
        return cmd_persist()
    if getattr(args, "adopt", ""):
        return cmd_adopt(args)

    name = getattr(args, "name", "") or ""
    if name and not getattr(args, "repo", ""):
        warn("error: --name requires --repo OWNER/REPO")
        return 2

    kind = args.source_kind or ("source" if can_build_from_source() else "release")
    if kind == "release" and patch_files():
        warn(f"a downloaded release carries none of the {len(patch_files())} patches in "
             f"{patches_dir()}; build --from source for those")
    try:
        if name:
            if kind == "release":
                dest, commit = build_from_release_named(args)
            else:
                dest, commit = build_from_source_named(args)
            say("verifying")
            verify_and_switch(dest, commit, named=name)
        else:
            if kind == "release":
                dest, commit = build_from_release(args)
            else:
                dest, commit = build_from_source(args)
            say("verifying")
            verify_and_switch(dest, commit)
    except BuildFailed as exc:
        warn(f"error: {exc}")
        return 2
    return 0
