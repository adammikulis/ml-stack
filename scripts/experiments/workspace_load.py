#!/usr/bin/env python3
"""Load test for the agent workspace: many processes sending, waiting, claiming and renewing.

    workspace_load.py                       30 agents, 20 messages each, limits opened for the run
    workspace_load.py --agents 60 --messages 40 --out run.json
    workspace_load.py --default-limits      keep the shipped rate limits and count what they refuse

Every agent is its own process with its own token against one workspace in a temporary
directory. It sends to its neighbour, waits for what its other neighbour sent, claims a port and
a branch and renews them, and every tenth agent also reads threads. Each process reads the
keystore master once through a counting fake keyring (the OS keystore is never touched). The
report is JSON on stdout (and ``--out``); the exit status is 1 when a budget is missed. No model
is loaded.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import resource
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

BUDGETS = {
    "send_p99_s": 1.0,
    "delivery_p99_s": 2.0,
    "inbox_read_p99_s": 0.25,
    "lock_wait_p99_s": 1.0,
    "keystore_busy": 0,
    "unexpected_failures": 0,
    "lost_messages": 0,
    "cpu_s_per_agent_per_message": 0.25,
}
WALL_BUDGETS = ("send_p99_s", "delivery_p99_s", "inbox_read_p99_s", "lock_wait_p99_s")
EXPECTED_WITH_DEFAULT_LIMITS = {"rate-limited", "refused"}
RING_NAME = "LoadRing"
FAILURES = (RuntimeError, ValueError, OSError)


def pct(values: list[float], q: float) -> float:
    """The ``q`` quantile of ``values``; 0 for none."""
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def summary(values: list[float]) -> dict[str, float]:
    """Count, p50, p99 and max of ``values``."""
    return {"n": len(values), "p50": round(pct(values, 0.5), 5), "p99": round(pct(values, 0.99), 5),
            "max": round(max(values, default=0.0), 5)}


# -- the child -----------------------------------------------------------------------------------


def install_ring(path: Path) -> None:
    """Replace the keyring with a file-backed fake that logs each call; refuses anything else."""
    import keyring
    from keyring.backend import KeyringBackend

    class LoadRing(KeyringBackend):
        priority = 5  # type: ignore[assignment]

        def _note(self, name: str) -> None:
            with Path(f"{path}.calls").open("a") as out:
                out.write(name + "\n")

        def _all(self) -> dict[str, str]:
            try:
                return json.loads(path.read_text())
            except (OSError, ValueError):
                return {}

        def get_password(self, service: str, username: str) -> str | None:
            self._note("get")
            return self._all().get(f"{service}/{username}")

        def set_password(self, service: str, username: str, password: str) -> None:
            self._note("set")
            path.write_text(json.dumps({**self._all(), f"{service}/{username}": password}))

        def delete_password(self, service: str, username: str) -> None:
            self._note("delete")

    keyring.set_keyring(LoadRing())
    if type(keyring.get_keyring()).__name__ != RING_NAME:
        raise SystemExit("the fake keyring is not in force; refusing to continue")


def timed_locks(waits: list[float], holds: list[float]) -> None:
    """Wrap the workspace's file lock so the bus log's wait and hold times are recorded."""
    from contextlib import contextmanager

    from ml_stack.workspace import board_graph, chain, claims, rates

    original = chain.held

    @contextmanager
    def held(path: Path, **options):
        start = time.perf_counter()
        with original(path, **options) as got:
            taken = time.perf_counter()
            if path.name == "board-graph.lock":
                waits.append(taken - start)
            try:
                yield got
            finally:
                if path.name == "board-graph.lock":
                    holds.append(time.perf_counter() - taken)

    board_graph.held = chain.held = claims.held = rates.held = held


class Agent:
    """One agent process: its identity, its measurements and the steps of its day."""

    def __init__(self, spec: dict[str, Any]) -> None:
        from ml_stack.workspace import Workspace

        self.spec, self.name, self.token = spec, spec["name"], spec["token"]
        self.ws = Workspace(Path(spec["base"]))
        self.lock_wait: list[float] = []
        self.lock_hold: list[float] = []
        self.out: dict[str, Any] = {key: [] for key in ("send", "inbox", "delivery", "heartbeat",
                                                        "thread", "claim", "keystore")}
        self.out.update({"failures": {}, "received": 0, "sent": 0})
        self.ready = 0.0
        self.mine = (("port", str(20000 + spec["index"])), ("branch", f"load/{self.name}"))

    def fail(self, err: BaseException) -> None:
        """Count a failure by its kind."""
        names = {"RateLimited": "rate-limited", "Refused": "refused", "Denied": "denied",
                 "KeystoreBusy": "busy", "Conflict": "conflict"}
        kind = names.get(type(err).__name__, f"other:{type(err).__name__}")
        self.out["failures"][kind] = self.out["failures"].get(kind, 0) + 1

    def timed(self, key: str, call: Any, *args: Any) -> Any:
        """Call ``call`` and record its duration under ``key``; a failure is counted, not raised."""
        start = time.perf_counter()
        try:
            return call(*args)
        except FAILURES as err:
            self.fail(err)
            return None
        finally:
            self.out[key].append(time.perf_counter() - start)

    def take(self, got: list[dict[str, Any]]) -> None:
        """Record what a read returned and when it was sent."""
        now = time.time()
        self.out["received"] += len(got)
        self.out["delivery"] += [max(0.0, now - max(m["ts"], self.ready)) for m in got]

    def keystore(self) -> None:
        """Read the master once, as a new process would."""
        from ml_stack import keystore

        def read() -> None:
            keystore.Keystore(wires=keystore.Wires(interactive=lambda: True, say=lambda _m: None)).subkey(
                "load", self.name)

        self.timed("keystore", read)

    def exchange(self) -> None:
        """Send to the neighbour, renew claims, wait for mail; every tenth agent reads a thread."""
        first = 0
        for i in range(self.spec["messages"]):
            sent = self.timed("send", self.ws.send, self.token, self.spec["peer"], "status", f"{self.name} {i}")
            if sent:
                self.out["sent"] += 1
                first = first or sent["seq"]
            self.timed("heartbeat", self.ws.heartbeat, self.token)
            self.take(self.ws.wait(self.token, 0.3, ack=True))
            if self.spec["index"] % 10 == 0 and first:
                self.timed("thread", self.ws.thread, self.token, first)

    def drain(self) -> None:
        """Wait for the rest of the neighbour's messages, then read the inbox once more."""
        deadline = time.time() + self.spec["drain_s"]
        while self.out["received"] < self.spec["expect"] and time.time() < deadline:
            self.take(self.ws.wait(self.token, 1.0, ack=True))
        self.timed("inbox", self.ws.inbox, self.token, False)

    def run(self) -> dict[str, Any]:
        """The whole day; returns the measurements."""
        timed_locks(self.lock_wait, self.lock_hold)
        while not Path(self.spec["go"]).exists():
            time.sleep(0.01)
        started, cpu0 = time.perf_counter(), resource.getrusage(resource.RUSAGE_SELF)
        self.keystore()
        self.ready = time.time()
        for kind, key in self.mine:
            self.timed("claim", self.ws.claim, self.token, kind, key)
        self.exchange()
        self.drain()
        for kind, key in self.mine:
            self.timed("claim", self.ws.release, self.token, kind, key)
        cpu1 = resource.getrusage(resource.RUSAGE_SELF)
        self.out["cpu_s"] = cpu1.ru_utime + cpu1.ru_stime - cpu0.ru_utime - cpu0.ru_stime
        self.out["wall_s"] = time.perf_counter() - started
        self.out["lock_wait"], self.out["lock_hold"] = self.lock_wait, self.lock_hold
        return self.out


def child(spec: dict[str, Any]) -> dict[str, Any]:
    """One agent process: returns its measurements."""
    os.environ["ML_STACK_HOME"] = spec["home"]
    install_ring(Path(spec["ring"]))
    return Agent(spec).run()


# -- the parent ----------------------------------------------------------------------------------


def run(agents: int = 30, messages: int = 20, *, default_limits: bool = False,
        drain_s: float = 60.0, workdir: Path | None = None) -> dict[str, Any]:
    """Run the load and return the report, including ``pass`` and ``missed``."""
    root = Path(workdir or tempfile.mkdtemp(prefix="ml-stack-load-"))
    home, base, ring, go = root / "home", root / "ws", root / "ring.json", root / "go"
    os.environ["ML_STACK_HOME"] = str(home)
    for marker in ("CLAUDECODE", "ML_STACK_AGENT", "ML_STACK_NONINTERACTIVE", "ML_STACK_WORKSPACE_TOKEN"):
        os.environ.pop(marker, None)
    install_ring(ring)
    from ml_stack import keystore
    from ml_stack.workspace import Workspace

    keystore.Keystore(wires=keystore.Wires(interactive=lambda: True, say=lambda _m: None)).subkey("load", "seed")
    base.mkdir(parents=True, exist_ok=True)
    if not default_limits:
        (base / "limits.json").write_text(json.dumps({"version": 1, "sends_per_window": 10_000_000,
                                                      "inbox_pending": 10_000_000}))
    ws = Workspace(base)
    owner = ws.init("owner")
    names = [f"agent{i:03d}" for i in range(agents)]
    tokens = {n: ws.mint(owner, n) for n in names}
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "ML_STACK_NOTIFY": "off"}
    kids = []
    for i, n in enumerate(names):
        spec = {"index": i, "name": n, "token": tokens[n], "peer": names[(i + 1) % agents],
                "messages": messages, "expect": messages, "base": str(base), "home": str(home),
                "ring": str(ring), "go": str(go), "drain_s": drain_s}
        kids.append(subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--child",
                                      json.dumps(spec)], env=env, stdout=subprocess.PIPE, text=True))
    time.sleep(1.0 + agents * 0.05)
    start = time.perf_counter()
    go.write_text("go")
    outs: list[dict[str, Any]] = []
    for kid in kids:
        text, _ = kid.communicate(timeout=drain_s + 600)
        outs.append(json.loads(text.strip().splitlines()[-1]) if kid.returncode == 0 and text.strip()
                    else {"failures": {"child-died": 1}, "send": [], "inbox": [], "delivery": [],
                          "heartbeat": [], "thread": [], "claim": [], "keystore": [], "lock_wait": [], "lock_hold": [],
                          "received": 0, "sent": 0, "cpu_s": 0.0, "wall_s": 0.0})
    wall = time.perf_counter() - start
    shape = {"agents": agents, "messages": messages, "wall": wall, "default_limits": default_limits}
    return report(outs, ws, ring, shape)


def report(outs: list[dict[str, Any]], ws: Any, ring: Path, shape: dict[str, Any]) -> dict[str, Any]:
    """Merge the agents' measurements, check the logs and judge against ``BUDGETS``."""
    agents, messages, wall = shape["agents"], shape["messages"], shape["wall"]
    default_limits, base = shape["default_limits"], ws.base

    def merged(key: str) -> list[float]:
        return [x for o in outs for x in o[key]]

    failures: dict[str, int] = {}
    for o in outs:
        for kind, n in o["failures"].items():
            failures[kind] = failures.get(kind, 0) + n
    sent, received = sum(o["sent"] for o in outs), sum(o["received"] for o in outs)
    verdict = ws.bus.log.verify()
    calls_file = Path(f"{ring}.calls")
    calls = calls_file.read_text().split() if calls_file.exists() else []
    cpu = sum(o["cpu_s"] for o in outs)
    expected = EXPECTED_WITH_DEFAULT_LIMITS if default_limits else set()
    result: dict[str, Any] = {
        "machine": {"cpus": os.cpu_count(), "platform": platform.platform(),
                    "memory_gb": round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30)},
        "date": time.strftime("%Y-%m-%d"), "agents": agents, "messages_each": messages,
        "default_limits": default_limits, "wall_s": round(wall, 2),
        "sent": sent, "received": received,
        "throughput_msgs_per_s": round(sent / wall, 1) if wall else 0.0,
        "send_s": summary(merged("send")), "delivery_s": summary(merged("delivery")),
        "inbox_read_s": summary(merged("inbox")), "keystore_read_s": summary(merged("keystore")), "heartbeat_s": summary(merged("heartbeat")),
        "claim_s": summary(merged("claim")), "thread_read_s": summary(merged("thread")),
        "bus_lock_wait_s": summary(merged("lock_wait")), "bus_lock_hold_s": summary(merged("lock_hold")),
        "failures": failures,
        "log": {"bus_rows": verdict.rows, "graph_bytes": ws.bus.log.graph.path.stat().st_size,
                "audit_bytes": (base / "audit.jsonl").stat().st_size, "chain_ok": verdict.ok},
        "keystore_backend_calls": {k: calls.count(k) for k in ("get", "set", "delete")},
        "cpu_s_total": round(cpu, 2),
        "cpu_s_per_agent_per_message": round(cpu / max(agents * messages, 1), 4),
    }
    unexpected = sum(n for k, n in failures.items() if k not in expected)
    lost = 0 if default_limits else max(0, sent - received)
    measured = {"send_p99_s": result["send_s"]["p99"], "delivery_p99_s": result["delivery_s"]["p99"],
                "inbox_read_p99_s": result["inbox_read_s"]["p99"],
                "lock_wait_p99_s": result["bus_lock_wait_s"]["p99"],
                "keystore_busy": failures.get("busy", 0), "unexpected_failures": unexpected,
                "lost_messages": lost,
                "cpu_s_per_agent_per_message": result["cpu_s_per_agent_per_message"]}
    # Agents beyond the cores take turns on them, so every waiting time stretches by that many turns;
    # CPU seconds and counts do not.
    # Other work already running on the host takes turns too: its runnable queue counts as agents.
    busy = os.getloadavg()[0] if hasattr(os, "getloadavg") else 0.0
    turns = max(1, math.ceil((agents + busy) / (os.cpu_count() or 1)))
    budgets = {k: limit * turns if k in WALL_BUDGETS else limit for k, limit in BUDGETS.items()}
    result["cpu_turns"] = turns
    result["budgets"] = budgets
    result["missed"] = sorted(k for k, limit in budgets.items() if measured[k] > limit)
    if not verdict.ok or verdict.rows < sent:
        result["missed"].append("chain")
    result["pass"] = not result["missed"]
    return result


def main(argv: list[str] | None = None) -> int:
    """Command line; exit 1 when a budget is missed."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--agents", type=int, default=30)
    parser.add_argument("--messages", type=int, default=20)
    parser.add_argument("--default-limits", action="store_true")
    parser.add_argument("--drain", type=float, default=60.0, help="seconds an agent waits for its last messages")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--child", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.child:
        sys.stdout.write(json.dumps(child(json.loads(args.child))) + "\n")
        return 0
    result = run(args.agents, args.messages, default_limits=args.default_limits, drain_s=args.drain)
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.out:
        args.out.write_text(text + "\n")
    sys.stdout.write(text + "\n")
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
