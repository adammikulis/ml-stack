"""The landing runner: takes reviewed requests from the board, lands them as one verified batch and
pushes only the development branch.

One pass (``Runner.once``) claims the runner role, drops requests that no longer stand, runs
``scripts/land run`` over the rest in an integration worktree, then ``scripts/land finish --apply``
and a normal push. Every step is an audit row; outcomes are board messages to the requesters.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import land_entries
import land_git as lg

from ml_stack.devbranch import development_branch
from ml_stack.workspace import landing
from ml_stack.workspace.claims import Conflict

HERE = Path(__file__).resolve().parent
MAX_BATCH = 8
STALL_S = 20 * 60.0
KILL_GRACE_S = 60.0
TEST_LINES = 6


@dataclass
class Runner:
    """A landing runner for one checkout, acting as the authenticated identity ``token``."""

    ws: object
    token: str
    root: Path
    target: str = ""
    remote: str = "origin"
    stall_s: float = STALL_S
    env: dict = field(default_factory=lambda: dict(os.environ))
    poll_s: float = 0.5

    def __post_init__(self) -> None:
        """An unnamed target is the development branch of this repository."""
        self.target = self.target or development_branch(self.root)

    def say(self, text: str, **fields: object) -> None:
        """One audit row for a runner step."""
        self.ws.audit("land.run", self.ws.auth(self.token).id, step=text, **fields)

    def tell(self, req: dict, text: str, kind: str = "status") -> None:
        """A direct message to the requester of ``req``."""
        self.ws.send(self.token, req["by"], kind, f"{req['id']} {req['branch']}: {text}"[:1500])

    def settle(self, req: dict, status: str, detail: str, **evidence: object) -> None:
        """Record the new state of ``req`` and tell its requester."""
        landing.transition(self.ws, self.token, req["id"], status, detail, **evidence)
        self.tell(req, f"{status}: {detail}", "handoff" if status in ("failed", "needs-human", "refused") else "status")

    def screen(self, queue: landing.Queue) -> list[dict]:
        """The requests that may be landed now; each other one is settled or left waiting with a reason."""
        ready = []
        for req in queue.waiting():
            reason = landing.standing(self.ws, req)
            if reason:
                if not req.get("evidence", {}).get("seen"):
                    landing.transition(self.ws, self.token, req["id"], "needs-review", reason, seen=True)
                    self.tell(req, f"waiting: {reason}")
                continue
            try:
                land_entries.check(self.root, req["target"], req["branch"], req["sha"])
            except ValueError as error:
                self.settle(req, "refused", str(error))
                continue
            if req["target"] != self.target:
                self.settle(req, "refused", f"this runner lands {self.target}, not {req['target']}")
                continue
            ready.append(req)
        return ready

    def stream(self, argv: list[str], batch: list[dict]) -> tuple[int, list[str], str]:
        """Run ``argv``, watch it for progress and cancellation; the status, its lines and a stop reason."""
        proc = subprocess.Popen(argv, cwd=self.root, env=self.env, stdin=subprocess.DEVNULL,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                start_new_session=True)
        lines: list[str] = []
        last = [time.monotonic()]

        def pump() -> None:
            for line in proc.stdout:
                lines.append(line.rstrip("\n"))
                last[0] = time.monotonic()

        reader = threading.Thread(target=pump, daemon=True)
        reader.start()
        stop, ids, marked = "", {r["id"] for r in batch}, 0.0
        while proc.poll() is None:
            time.sleep(self.poll_s)
            now = time.monotonic()
            if now - marked > 10:
                marked = now
                landing.beat(self.ws, self.token, lines[-1] if lines else "starting")
                self.ws.heartbeat(self.token)
                if any(r["status"] == "cancelled" for r in landing.fold(self.ws).requests.values() if r["id"] in ids):
                    stop = "cancelled"
            if now - last[0] > self.stall_s:
                stop = "stuck"
            if stop:
                self.halt(proc)
        reader.join(5)
        return proc.returncode, lines, stop

    def halt(self, proc: subprocess.Popen) -> None:
        """Ask the gate's process group to stop; kill it only when it ignores that for a minute."""
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(KILL_GRACE_S)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()

    def land(self, *args: str, batch: list[dict] | None = None) -> tuple[int, dict, str]:
        """``scripts/land ARGS``: its exit status, the JSON summary on its last line and any stop reason."""
        argv = [sys.executable, str(HERE / "land"), *args, "--repo", str(self.root), "--target", self.target]
        status, lines, stop = self.stream(argv, batch or [])
        last = lines[-1] if lines else "{}"
        try:
            summary = json.loads(last)
        except ValueError:
            summary = {"status": "error", "error": "no summary: " + " | ".join(lines[-3:])}
        return status, summary, stop

    def push(self) -> str:
        """Push the development branch with a normal push; an empty string on success, else why not."""
        lg.refuse_protected(self.target)
        top = lg.primary(self.root)
        if top.branch != self.target:
            return f"the primary checkout is on {top.branch or 'a detached head'}, not {self.target}"
        if self.remote not in lg.lines(self.root, "remote"):
            return f"no remote named {self.remote}"
        done = lg.git(top.path, "push", self.remote, f"{self.target}:refs/heads/{self.target}", check=False,
                      env=self.env)
        return "" if done.returncode == 0 else (done.stderr.strip().splitlines() or ["push failed"])[-1]

    def once(self) -> dict:
        """One pass over the queue; a summary of what happened."""
        who = self.ws.auth(self.token).id
        try:
            self.ws.claim(self.token, *landing.runner_claim(self.ws, self.target))
        except Conflict as held:
            return {"status": "runner-held", "owner": held.owner}
        queue = landing.fold(self.ws)
        if queue.paused:
            return {"status": "paused", "by": queue.paused_by}
        batch = self.screen(queue)[:MAX_BATCH]
        if not batch:
            return {"status": "idle"}
        self.say("batch", ids=[r["id"] for r in batch])
        for req in batch:
            landing.transition(self.ws, self.token, req["id"], "running", f"in a batch of {len(batch)}")
        try:
            return self.run_batch(batch, who)
        finally:
            self.ws.heartbeat(self.token)

    def run_batch(self, batch: list[dict], who: str) -> dict:
        """Merge, gate, land and push one batch, then report to every requester."""
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
            json.dump([{"branch": r["branch"], "tip": r["sha"]} for r in batch], handle)
        try:
            _, summary, stop = self.land("run", "--entries", handle.name, batch=batch)
        finally:
            Path(handle.name).unlink(missing_ok=True)
        if stop:
            return self.stopped(batch, stop)
        self.report_ejected(batch, summary)
        if summary.get("status") != "verified":
            self.report_unverified(batch, summary)
            return {"status": summary.get("status", "error"), "batch": [r["id"] for r in batch]}
        _, fin, _ = self.land("finish", "--apply")
        if fin.get("status") != "landed":
            return self.not_landed(batch, summary, fin)
        return self.landed(batch, summary, who)

    def stopped(self, batch: list[dict], stop: str) -> dict:
        """A cancelled or stuck gate: report it, keep the cancelled out and settle the rest."""
        now = landing.fold(self.ws)
        for req in batch:
            if now.requests[req["id"]]["status"] == "cancelled":
                continue
            if stop == "stuck":
                self.settle(req, "needs-human", f"the gate made no progress for {int(self.stall_s)}s and was aborted")
            else:
                landing.transition(self.ws, self.token, req["id"], "queued", "its batch was cancelled; queued again")
        if stop == "stuck":
            self.ws.announce(self.token, "blocked", f"landing gate stuck {int(self.stall_s)}s, aborted: "
                             + ", ".join(r["branch"] for r in batch))
        return {"status": stop, "batch": [r["id"] for r in batch]}

    def report_ejected(self, batch: list[dict], summary: dict) -> None:
        """Settle each ejected request: a merge conflict needs a person, a red check goes back to its author."""
        by_branch = {r["branch"]: r for r in batch}
        for row in summary.get("ejected", []):
            req = by_branch.get(row["branch"])
            if req is None:
                continue
            if row["check"] == "merge":
                self.settle(req, "needs-human", "merge conflict: " + row["evidence"])
            else:
                base = ", ".join(summary.get("baseline", [])) or "none"
                self.settle(req, "failed", f"{row['check']} failed on this branch but not on a clean "
                            f"{self.target}: {row['evidence']} (also red on {self.target}: {base})")

    def report_unverified(self, batch: list[dict], summary: dict) -> None:
        """No push: the requests still running had a red gate that no ejection explains."""
        now = landing.fold(self.ws)
        base = ", ".join(summary.get("baseline", [])) or "none"
        for req in batch:
            if now.requests[req["id"]]["status"] == "running":
                self.settle(req, "failed", f"the combined gate was red; red on {self.target} too: {base}; "
                            f"{summary.get('status')}: {summary.get('error', '')}")
        self.ws.announce(self.token, "blocked", f"landing batch not landed ({summary.get('status')}): "
                         + ", ".join(r["branch"] for r in batch))

    def not_landed(self, batch: list[dict], summary: dict, fin: dict) -> dict:
        """The batch verified but the fast-forward was blocked; hand it to a person."""
        for req in batch:
            if req["branch"] in summary.get("merged", []):
                self.settle(req, "needs-human", f"verified but not fast-forwarded: {fin.get('reason', fin)}")
        self.ws.announce(self.token, "blocked", "landing verified a batch but could not fast-forward "
                         f"{self.target}: {fin.get('reason', 'see land finish')}")
        return {"status": "blocked", "reason": fin.get("reason", "")}

    def landed(self, batch: list[dict], summary: dict, who: str) -> dict:
        """Push the fast-forwarded development branch and tell each merged requester."""
        head = lg.out(self.root, "rev-parse", self.target)
        problem = self.push()
        done = "landed" if not problem else "landed-unpushed"
        for req in batch:
            if req["branch"] in summary.get("merged", []):
                self.settle(req, done, f"{self.target} is at {head[:12]}" + (f"; push failed: {problem}" if problem else ""),
                            commit=head)
        text = f"landed {len(summary.get('merged', []))} on {self.target} at {head[:12]}"
        self.ws.announce(self.token, "milestone" if not problem else "blocked",
                         text if not problem else f"{text} but the push failed: {problem}")
        self.say("landed", head=head, pushed=not problem, by=who)
        return {"status": done, "head": head, "merged": summary.get("merged", []), "push": problem}


def serve(root: Path, target: str, flags: dict) -> int:
    """Run passes as the identity in this environment until interrupted, or one pass with ``once``."""
    from ml_stack.workspace import Workspace, tokens

    ws = Workspace()
    token = tokens.resolve(ws.base)
    if not token:
        raise ValueError("land serve acts as an authenticated workspace identity; no token found")
    runner = Runner(ws, token, root, target, flags["remote"], flags["stall_minutes"] * 60.0)
    while True:
        result = runner.once()
        print(json.dumps(result, sort_keys=True), flush=True)
        if flags["once"]:
            return 0
        time.sleep(flags["interval"])
