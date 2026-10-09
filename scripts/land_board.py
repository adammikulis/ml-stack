"""The landing runner: takes reviewed requests from the board, lands them as one verified batch and
pushes only the development branch.

One pass (``Runner.once``) takes the runner lock and claim, settles what a crashed pass left behind,
drops requests that no longer stand, runs ``scripts/land run`` over the rest in an integration
worktree, then ``scripts/land finish --apply`` and a normal push. Every step is an audit row;
outcomes are board messages to the requesters.

One request never fails another: a request that is stale, conflicts, is red, hangs or makes the
runner raise is settled alone with its reason, and the others in the batch are run again without it
(one by one when nothing names the culprit).
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
import land_recover as recover

from poolhouse.devbranch import development_branch
from poolhouse.workspace import landing
from poolhouse.workspace.claims import Conflict
from poolhouse.workspace.identity import Denied

HERE = Path(__file__).resolve().parent
MAX_BATCH = 8
STALL_S = 20 * 60.0
BATCH_S = 2 * 3600.0
REQUEST_S = 3600.0
KILL_GRACE_S = 60.0
RETRY = ("stuck", "timeout", "error", "unverified")
# what a request's own step can raise: git, the filesystem, the board, a missing key
STEP_ERRORS = (RuntimeError, ValueError, OSError, LookupError, TypeError, AttributeError,
               subprocess.SubprocessError, Denied, Conflict)


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
    batch_s: float = BATCH_S
    request_s: float = REQUEST_S

    def __post_init__(self) -> None:
        """An unnamed target is the development branch of this repository."""
        self.target = self.target or development_branch(self.root)
        self.lock = recover.RunnerLock(self.root)

    def say(self, text: str, **fields: object) -> None:
        """One audit row for a runner step."""
        self.ws.audit("land.run", self.ws.auth(self.token).id, step=text, **fields)

    def tell(self, req: dict, text: str, kind: str = "status") -> None:
        """A direct message to the requester of ``req``, by name."""
        self.ws.send(self.token, req["by"], kind, f"{req['id']} {req['branch']}: {text}"[:1500])

    def settle(self, req: dict, status: str, detail: str, **evidence: object) -> None:
        """Record the new state of ``req`` and tell its requester."""
        landing.transition(self.ws, self.token, req["id"], status, detail, **evidence)
        self.tell(req, f"{status}: {detail}", "handoff" if status in ("failed", "needs-human", "refused") else "status")

    def guard(self, req: dict, what: str, step):
        """Run ``step`` for one request; an exception settles that request alone and is logged.

        The value of ``step``, or None when it raised.
        """
        try:
            return step()
        except STEP_ERRORS as error:
            text = f"runner error while {what}: {type(error).__name__}: {error}"[:300]
            try:
                self.say("error", id=req["id"], what=what, error=text)
                self.settle(req, "needs-human", text)
            except STEP_ERRORS as again:
                print(f"land: could not record {text}: {again}", file=sys.stderr, flush=True)
            return None

    def vet(self, req: dict) -> bool:
        """Whether ``req`` may be landed now; otherwise it is settled or left waiting with a reason."""
        reason = landing.standing(self.ws, req)
        if reason:
            if not req.get("evidence", {}).get("seen"):
                landing.transition(self.ws, self.token, req["id"], "needs-review", reason, seen=True)
                self.tell(req, f"waiting: {reason}")
            return False
        if recover.present(self.root, req["sha"], self.target):
            self.settle(req, "landed", f"{self.target} already holds {req['sha'][:12]}")
            return False
        try:
            land_entries.check(self.root, req["target"], req["branch"], req["sha"])
        except ValueError as error:
            self.settle(req, "refused", str(error))
            return False
        if req["target"] != self.target:
            self.settle(req, "refused", f"this runner lands {self.target}, not {req['target']}")
            return False
        return True

    def screen(self, queue: landing.Queue) -> list[dict]:
        """The requests that may be landed now; each other one is settled or left waiting."""
        return [r for r in queue.waiting() if self.guard(r, "screening it", lambda r=r: self.vet(r)) is True]

    def limit(self, batch: list[dict]) -> float:
        """The longest a gate may run for ``batch``: one request has its own, shorter limit."""
        return self.request_s if len(batch) == 1 else self.batch_s

    def stream(self, argv: list[str], batch: list[dict]) -> tuple[int, list[str], str]:
        """Run ``argv``, watch it for progress and cancellation; the status, its lines and a stop reason."""
        proc = subprocess.Popen(argv, cwd=self.root, env=self.env, stdin=subprocess.DEVNULL,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                start_new_session=True)
        lines: list[str] = []
        began = last = time.monotonic()
        mark = [last]

        def pump() -> None:
            for line in proc.stdout:
                lines.append(line.rstrip("\n"))
                mark[0] = time.monotonic()

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
            if now - mark[0] > self.stall_s:
                stop = "stuck"
            elif now - began > self.limit(batch):
                stop = "timeout"
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

    def close(self) -> None:
        """Give up the branch claim and the runner lock."""
        try:
            self.ws.release(self.token, *landing.runner_claim(self.ws, self.target))
        except STEP_ERRORS as error:
            print(f"land: claim not released: {error}", file=sys.stderr, flush=True)
        self.lock.release()

    def once(self) -> dict:
        """One pass over the queue; a summary of what happened."""
        who = self.ws.auth(self.token).id
        if not self.lock.acquire():
            return {"status": "runner-held", "owner": f"another runner process ({recover.holder(self.root)})"}
        try:
            self.ws.claim(self.token, *landing.runner_claim(self.ws, self.target), pid=os.getpid())
        except Conflict as held:
            return {"status": "runner-held", "owner": held.owner}
        if landing.fold(self.ws).paused:
            return {"status": "paused", "by": landing.fold(self.ws).paused_by}
        self.resume()
        batch = self.screen(landing.fold(self.ws))[:MAX_BATCH]
        if not batch:
            return {"status": "idle"}
        self.say("batch", ids=[r["id"] for r in batch])
        batch = [r for r in batch if self.guard(r, "starting it", lambda r=r: landing.transition(
            self.ws, self.token, r["id"], "running", f"in a batch of {len(batch)}") or True)]
        try:
            return self.drive(batch, who) if batch else {"status": "idle"}
        finally:
            recover.sweep(self.root)
            self.ws.heartbeat(self.token)

    def resume(self) -> None:
        """Settle what a crashed pass left: remove its trees, then settle each request by its SHA.

        A request already on the development branch is landed (pushed when origin lacks it); the
        others that were running are queued again. Nothing is merged a second time.
        """
        gone = recover.sweep(self.root)
        if gone:
            self.say("swept", trees=gone)
        held = []
        for req in landing.fold(self.ws).requests.values():
            if req["status"] == "running":
                self.guard(req, "resuming it", lambda r=req: self.resume_one(r, held))
            elif req["status"] == "landed-unpushed":
                held.append(req)
        held = [r for r in held if not recover.on_origin(self.root, r["sha"], self.target, self.remote)]
        if held:
            problem = recover.push_with_resync(self)
            for req in held:
                self.guard(req, "pushing it", lambda r=req: self.after_push(r, problem))

    def resume_one(self, req: dict, held: list[dict]) -> None:
        """A request left ``running``: landed already, or queued again."""
        if recover.on_target(self.root, req["sha"], self.target):
            held.append(req)
        else:
            landing.transition(self.ws, self.token, req["id"], "queued", "the runner restarted mid-batch; queued again")

    def after_push(self, req: dict, problem: str) -> None:
        """Settle a request found on the development branch after a restart."""
        head = lg.out(self.root, "rev-parse", self.target)
        if problem and req["status"] == "landed-unpushed":
            return
        if problem:
            self.settle(req, "landed-unpushed", f"{self.target} holds it at {head[:12]}; push failed: {problem}", commit=head)
        else:
            self.settle(req, "landed", f"{self.target} is at {head[:12]}", commit=head)

    def drive(self, batch: list[dict], who: str) -> dict:
        """Land ``batch``; a failure no request is blamed for runs the requests one at a time."""
        out = self.attempt(batch, who)
        if out["status"] not in RETRY:
            return out
        live = [r for r in batch if landing.fold(self.ws).requests[r["id"]]["status"] == "running"]
        if len(live) > 1:
            self.say("split", why=out["status"], ids=[r["id"] for r in live])
            return self.singly(live, who, out["status"])
        for req in live:
            self.give_up(req, out)
        return out

    def attempt(self, batch: list[dict], who: str) -> dict:
        """One try at landing ``batch``; an exception inside the runner becomes an ``error`` outcome."""
        try:
            return self.run_batch(batch, who)
        except STEP_ERRORS as error:
            text = f"{type(error).__name__}: {error}"[:300]
            self.say("batch-error", ids=[r["id"] for r in batch], error=text)
            return {"status": "error", "error": text, "batch": [r["id"] for r in batch]}
        finally:
            recover.sweep(self.root)

    def singly(self, live: list[dict], who: str, why: str) -> dict:
        """Run each request alone so the one that fails is the only one that does."""
        results = []
        for req in live:
            if landing.fold(self.ws).requests[req["id"]]["status"] != "running":
                continue
            out = self.attempt([req], who)
            if out["status"] in RETRY:
                self.give_up(req, out)
            results.append(out["status"])
        return {"status": "split", "reason": why, "results": results, "batch": [r["id"] for r in live]}

    def give_up(self, req: dict, out: dict) -> None:
        """Settle ``req`` as the one a failed run is attributed to."""
        status = out["status"]
        if status == "stuck":
            why = f"the gate made no progress for {int(self.stall_s)}s and was aborted"
        elif status == "timeout":
            why = f"the gate ran past its {int(self.request_s)}s limit and was aborted"
        elif status == "error":
            why = f"the runner raised: {out.get('error', '')}"
        else:
            base = ", ".join(out.get("baseline", [])) or "none"
            why = f"the gate was red on this request alone; red on {self.target} too: {base}; {out.get('detail', '')}"
        self.guard(req, "giving it up", lambda: self.settle(req, "failed" if status == "unverified" else "needs-human", why))
        self.ws.announce(self.token, "blocked", f"landing could not land {req['branch']} ({status})")

    def run_batch(self, batch: list[dict], who: str) -> dict:
        """Merge, gate, land and push one batch, then report to every requester."""
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
            json.dump([{"branch": r["branch"], "tip": r["sha"]} for r in batch], handle)
        try:
            _, summary, stop = self.land("run", "--entries", handle.name, batch=batch)
        finally:
            Path(handle.name).unlink(missing_ok=True)
        if stop in ("stuck", "timeout"):
            return {"status": stop, "batch": [r["id"] for r in batch]}
        if stop:
            return self.cancelled(batch)
        self.report_ejected(batch, summary)
        if summary.get("status") == "empty":
            return self.landed(batch, {"merged": []}, who)
        if summary.get("status") != "verified":
            bad = ", ".join(summary.get("blocked", [])) or summary.get("error", "") or summary.get("status", "")
            return {"status": "unverified", "batch": [r["id"] for r in batch], "detail": bad,
                    "baseline": summary.get("baseline", [])}
        _, fin, _ = self.land("finish", "--apply")
        if fin.get("status") != "landed":
            return self.not_landed(batch, summary, fin)
        return self.landed(batch, summary, who)

    def cancelled(self, batch: list[dict]) -> dict:
        """A cancelled batch: keep the cancelled out and queue the rest again."""
        now = landing.fold(self.ws)
        for req in batch:
            if now.requests[req["id"]]["status"] != "cancelled":
                landing.transition(self.ws, self.token, req["id"], "queued", "its batch was cancelled; queued again")
        return {"status": "cancelled", "batch": [r["id"] for r in batch]}

    def report_ejected(self, batch: list[dict], summary: dict) -> None:
        """Settle each ejected request: a merge that did not complete needs a person, a red check goes back."""
        by_branch = {r["branch"]: r for r in batch}
        base = ", ".join(summary.get("baseline", [])) or "none"
        for row in summary.get("ejected", []):
            req = by_branch.get(row["branch"])
            if req is None:
                continue
            if row["check"] == "merge":
                self.guard(req, "reporting it", lambda r=req, w=row: self.settle(
                    r, "needs-human", ("merge " if w["evidence"].startswith("conflicts") else "merge failed: ")
                    + w["evidence"]))
            else:
                self.guard(req, "reporting it", lambda r=req, w=row: self.settle(
                    r, "failed", f"{w['check']} failed on this branch but not on a clean "
                    f"{self.target}: {w['evidence']} (also red on {self.target}: {base})"))

    def not_landed(self, batch: list[dict], summary: dict, fin: dict) -> dict:
        """The batch verified but the fast-forward was blocked: queue it again if the target moved."""
        reason = fin.get("reason", "")
        for req in batch:
            if req["branch"] in summary.get("merged", []):
                if "moved since" in reason:
                    landing.transition(self.ws, self.token, req["id"], "queued", f"{reason}; queued again")
                else:
                    self.guard(req, "reporting it", lambda r=req: self.settle(
                        r, "needs-human", f"verified but not fast-forwarded: {reason or fin}"))
        self.ws.announce(self.token, "blocked", f"landing verified a batch but could not fast-forward {self.target}: "
                         f"{reason or 'see land finish'}")
        return {"status": "blocked", "reason": reason, "batch": [r["id"] for r in batch]}

    def landed(self, batch: list[dict], summary: dict, who: str) -> dict:
        """Push the fast-forwarded development branch and tell each requester how its request ended."""
        problem = recover.push_with_resync(self)
        head = lg.out(self.root, "rev-parse", self.target)
        merged = summary.get("merged", [])
        for req in batch:
            if req["branch"] in merged or recover.present(self.root, req["sha"], self.target):
                self.guard(req, "reporting it", lambda r=req: self.after_push(r, problem))
            elif landing.fold(self.ws).requests[req["id"]]["status"] == "running":
                landing.transition(self.ws, self.token, req["id"], "queued", "not merged in this batch; queued again")
        text = f"landed {len(merged)} on {self.target} at {head[:12]}"
        self.ws.announce(self.token, "milestone" if not problem else "blocked",
                         text if not problem else f"{text} but the push failed: {problem}")
        self.say("landed", head=head, pushed=not problem, by=who)
        return {"status": "landed" if not problem else "landed-unpushed", "head": head, "merged": merged,
                "push": problem}


def serve(root: Path, target: str, flags: dict) -> int:
    """Run passes as the identity in this environment until interrupted, or one pass with ``once``."""
    from poolhouse.workspace import Workspace, tokens

    ws = Workspace()
    token = tokens.resolve(ws.base)
    if not token:
        raise ValueError("land serve acts as an authenticated workspace identity; no token found")
    runner = Runner(ws, token, root, target, flags["remote"], flags["stall_minutes"] * 60.0,
                    request_s=flags["request_minutes"] * 60.0, batch_s=flags["batch_minutes"] * 60.0)
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    try:
        while True:
            try:
                result = runner.once()
            except STEP_ERRORS as error:
                result = {"status": "error", "error": f"{type(error).__name__}: {error}"[:300]}
            print(json.dumps(result, sort_keys=True), flush=True)
            if flags["once"]:
                return 0
            time.sleep(flags["interval"])
    except KeyboardInterrupt:
        return 0
    finally:
        runner.close()
