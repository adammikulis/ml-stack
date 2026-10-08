"""``land finish``: fast-forward the target from the verified batch and remove landed branches."""

from __future__ import annotations

import json
import os
from pathlib import Path

import land_git as lg
import land_run

from ml_stack.activity.gate import tree_hash


def load(root: Path) -> dict:
    """The last batch's state, or an empty dict."""
    path = land_run.state_file(root)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def inside(path: Path) -> bool:
    """Whether the current directory is ``path`` or below it."""
    here = Path.cwd().resolve()
    return here == path.resolve() or path.resolve() in here.parents


def full_running(state: dict) -> bool:
    """Whether the background full run started for this batch is still alive."""
    full = state.get("full")
    pid = full.get("pid") if isinstance(full, dict) else None
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
    except (OSError, ValueError):
        return False
    return True


def readiness(root: Path, state: dict) -> tuple[str, str]:
    """A reason the batch cannot be fast-forwarded, and the command a lead would run instead."""
    wt, target = Path(state["worktree"]), state["target"]
    top = lg.primary(root)
    command = f"git -C {top.path} merge --ff-only {state['integration']}"
    if state.get("status") != "verified":
        return f"batch status is {state.get('status')}", ""
    if not wt.exists() or lg.dirty(wt) or tree_hash(wt) != state["tree"]:
        return "the integration worktree is missing, dirty or no longer the verified tree", ""
    if lg.git(root, "merge-base", "--is-ancestor", target, state["integration"], check=False).returncode:
        return f"{target} moved since the batch was cut; run land again", ""
    if top.branch != target:
        return f"the primary checkout is on {top.branch or 'a detached head'}, not {target}", command
    if lg.dirty(top.path):
        return "the primary checkout has uncommitted changes", command
    return "", command


def removable(root: Path, upstream: str, name: str) -> str:
    """Why ``name`` must stay (unique patches, a dirty or occupied tree), or an empty string."""
    if lg.unique(root, upstream, name):
        return "unique patches remain"
    tree = lg.tree_of(root, name)
    if tree is None:
        return ""
    if tree.path == lg.primary(root).path:
        return "primary checkout"
    if lg.dirty(tree.path):
        return "dirty worktree"
    if inside(tree.path):
        return "current directory is inside it"
    return ""


def remove(root: Path, name: str) -> str:
    """Remove the worktree and branch of ``name``; an empty string on success, else why not."""
    tree = lg.tree_of(root, name)
    top = lg.primary(root).path
    if tree is not None:
        done = lg.git(top, "worktree", "remove", str(tree.path), check=False)
        if done.returncode:
            return (done.stderr.strip().splitlines() or ["worktree remove failed"])[-1]
    done = lg.git(top, "branch", "-d", name, check=False)
    return "" if done.returncode == 0 else "branch -d refused (not merged by ancestry)"


def cleanup(root: Path, state: dict, apply: bool) -> tuple[list[str], list[dict]]:
    """Remove (or list) the landed branches and the integration branch."""
    gone, kept = [], []
    ejected = {e["branch"] for e in state.get("ejected", [])}
    for name in [n for n in state.get("candidates", state["sources"]) if n not in ejected]:
        problem = removable(root, state["integration"], name)
        problem = problem or (remove(root, name) if apply else "")
        (kept.append({"branch": name, "reason": problem}) if problem else gone.append(name))
    if full_running(state):
        kept.append({"branch": state["integration"], "reason": "background full run in progress"})
    else:
        problem = remove(root, state["integration"]) if apply else ""
        if problem:
            kept.append({"branch": state["integration"], "reason": problem})
    if apply:
        lg.git(root, "worktree", "prune")
    return gone, kept


def finish(root: Path, apply: bool) -> tuple[int, dict]:
    """Fast-forward the target and clean up; the exit status and the summary."""
    state = load(root)
    if not state:
        print("finish: no batch recorded; run `land run` first")
        return 1, {"command": "finish", "status": "none"}
    target = state["target"]
    why, command = readiness(root, state)
    summary: dict = {"command": "finish", "target": target, "apply": apply,
                     "integration": state["integration"]}
    if why:
        print(f"finish: not landing: {why}")
        if command:
            print(f"finish: for the lead: {command}")
        return 3, {**summary, "status": "blocked", "reason": why, "lead_command": command}
    if apply:
        lg.git(lg.primary(root).path, "merge", "--ff-only", state["integration"])
    print(f"finish: {'fast-forwarded' if apply else 'would fast-forward'} {target} to "
          f"{lg.out(root, 'rev-parse', '--short', state['integration'])}; the push is the lead's")
    gone, kept = cleanup(root, state, apply)
    for name in gone:
        print(f"finish: {'removed' if apply else 'would remove'} {name}")
    for row in kept:
        print(f"finish: kept {row['branch']}: {row['reason']}")
    return 0, {**summary, "status": "landed" if apply else "dry-run", "removed": gone, "kept": kept}
