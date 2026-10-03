"""Run, train, evaluate, and replay specialist simulations."""

import json
import time
from pathlib import Path

from ml_stack.command import Group, flag
from ml_stack.gym.catalog import catalogue
from ml_stack.gym.recordings import export_reviewed
from ml_stack.gym.runtime import manager
from ml_stack.gym.training import evaluate, train
from ml_stack.log import say


def commands():
    group = Group("ml-stack-gym", "Run, train, and evaluate specialist simulators")
    group.add("catalogue", handle, help="List environments and installation diagnostics")
    common = (flag("environment", choices=[entry["id"] for entry in catalogue()]),
              flag("--config", default="{}", help="Native environment configuration as JSON"),
              flag("--seed", type=int, default=0))
    group.add("run", handle, help="Run a recorded session", options=(*common, flag("--controller", choices=["manual", "random", "decider"], default="manual"), flag("--steps", type=int, default=100), flag("--action", type=int, default=4)))
    group.add("train", handle, help="Train CPU PPO", options=(*common, flag("--timesteps", type=int, default=2048), flag("--checkpoint")))
    group.add("evaluate", handle, help="Evaluate a local PPO checkpoint", options=(*common, flag("checkpoint"), flag("--episodes", type=int, default=5)))
    group.add("replay", handle, help="Read a recorded trajectory", options=(flag("trajectory", type=Path),))
    group.add("export", handle, help="Export reviewed decisions", options=(
        flag("trajectory", type=Path), flag("reviews", type=Path), flag("output", type=Path)))
    return group


def argument_parser():
    return commands().parser()


def handle(args):
    if args.cmd == "catalogue":
        result = catalogue()
    elif args.cmd == "replay":
        for line in args.trajectory.read_text().splitlines():
            say(json.dumps(json.loads(line)))
        return
    elif args.cmd == "export":
        result = {"cases": export_reviewed(args.trajectory, args.reviews, args.output),
                  "output": str(args.output)}
    else:
        config = json.loads(args.config)
        if args.cmd == "train":
            result = train(args.environment, config, args.timesteps, args.seed, args.checkpoint)
        elif args.cmd == "evaluate":
            result = evaluate(args.environment, args.checkpoint, config, args.episodes, args.seed)
        else:
            result = run_session(args, config)
    say(json.dumps(result, indent=2))


def run_session(args, config):
    snapshot = manager.create(args.environment, config, args.controller, args.seed)
    identifier = snapshot["id"]
    try:
        deadline = time.monotonic() + 60
        while snapshot["status"] == "starting" and time.monotonic() < deadline:
            time.sleep(.1)
            snapshot = manager.get(identifier)
        if snapshot["status"] == "starting":
            raise RuntimeError("Simulation startup timed out")
        manager.control(identifier, "action", {"action": args.action})
        for _ in range(args.steps):
            sequence = snapshot["sequence"]
            manager.control(identifier, "step")
            deadline = time.monotonic() + 60
            while snapshot["sequence"] == sequence and not snapshot.get("error"):
                if time.monotonic() > deadline:
                    raise RuntimeError("Simulation step timed out")
                time.sleep(.05)
                snapshot = manager.get(identifier)
            say(json.dumps(snapshot))
            if snapshot.get("error") or snapshot.get("terminated") or snapshot.get("truncated"):
                break
        return snapshot
    finally:
        manager.close(identifier)


main = commands().run

if __name__ == "__main__":
    main()
