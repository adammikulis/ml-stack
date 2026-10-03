#!/usr/bin/env python3
"""The pre-registered IQ against K-quant experiment on Metal (docs/experiments/iq-vs-kquant-metal.md).

    iq_vs_kquant.py            print the plan and the exact commands; runs nothing
    iq_vs_kquant.py --check    validate the config, the model files and every command's flags;
                               loads no model
    iq_vs_kquant.py --run      run the cells one at a time, writing results as it goes
    iq_vs_kquant.py --analyse  read the results and write the table and the verdict

Every cell is one call of the existing machinery: `ml-stack-decide jevbench` (the logprob
backend), `ml-stack-bench sweep` (the question set) or `ml-stack-bench speed`.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

CONFIG = ROOT / "docs/experiments/iq-vs-kquant-metal.json"
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}
QUIET = r"llama-server|decide_cli train"


@dataclass(frozen=True)
class Cell:
    pair: str
    quant: str
    task: str
    thinking: str
    repeat: int

    @property
    def id(self) -> str:
        return f"{self.pair}-{self.quant}-{self.task}-think{self.thinking}-r{self.repeat}"


def load(path: Path = CONFIG) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def expand(path: str) -> Path:
    return Path(path).expanduser()


def cells(config: dict[str, Any]) -> list[Cell]:
    """Repeat-major, the two builds of a pair in alternating order, so drift in the machine
    lands on both builds alike."""
    out: list[Cell] = []
    for repeat in range(1, int(config["repeats"]) + 1):
        for pair in config["pairs"]:
            order = ("iq", "kquant") if repeat % 2 else ("kquant", "iq")
            for quant in order:
                for task in pair["tasks"]:
                    for thinking in (pair["thinking"] if task == "generation" else ["off"]):
                        out.append(Cell(pair["id"], quant, task, thinking, repeat))
    return out


def pair_of(config: dict[str, Any], cell: Cell) -> dict[str, Any]:
    return next(p for p in config["pairs"] if p["id"] == cell.pair)


def label(cell: Cell) -> str:
    return f"iqx-{cell.id}"


def model_of(ref: str) -> str:
    """The file ``ref`` names when it is installed, else ``ref``."""
    from ml_stack import hub

    found = hub.installed_for(ref) if ref.startswith("hf:") else None
    return str(found.path) if found else ref


def command(config: dict[str, Any], cell: Cell, out_dir: Path) -> list[str]:
    """The exact argv of one cell."""
    model = model_of(pair_of(config, cell)[cell.quant])
    seed = ["--serve-arg=-s", f"--serve-arg={config['seed']}"]
    common = ["--serve", model, "--serve-label", label(cell), "--no-profile", "--no-draft",
              "--context", str(config["context"]), "--kept", str(expand(config["store"])),
              "--no-queue", "--yes"]
    bench = [sys.executable, "-m", "ml_stack.bench.run"]
    if cell.task == "jevbench":
        spec = config["tasks"]["jevbench"]
        return [sys.executable, "-m", "ml_stack.decide_cli", "jevbench", "--yes", "--backend",
                "logprob", "--gguf", model, "--context", str(spec["context"]), "--tier",
                spec["tier"], "--limit", str(spec["limit"]),
                "--out", str(out_dir / f"{cell.id}.json")]
    if cell.task == "generation":
        thinking = [] if cell.thinking == "on" else ["--reasoning-budget", "0"]
        return [*bench, "sweep", *common, "--plain-only", "--temperature",
                str(config["temperature"]), "--sample", str(config["sample"]),
                *seed, *thinking]
    spec = config["tasks"]["speed"]
    return [*bench, "speed", *common, "--prompts", ",".join(map(str, spec["prompts"])),
            "--streams", ",".join(map(str, spec["streams"])), "--generate", str(spec["generate"]),
            "--reasoning-budget", "0"]


def problems(config: dict[str, Any]) -> list[str]:
    """Everything wrong with the config, without loading a model."""
    found: list[str] = []
    if int(config.get("repeats", 0)) < 3:
        found.append("repeats must be at least 3")
    rule = config.get("decision_rule", {})
    for key in ("accuracy_points", "speed_ratio", "repeats_needed", "confidence"):
        if not isinstance(rule.get(key), (int, float)):
            found.append(f"decision_rule.{key} is missing")
    if float(config.get("temperature", 1)) != 0:
        found.append("temperature must be 0")
    if config.get("draft") != "none":
        found.append("the draft head must be held at none")
    for pair in config["pairs"]:
        for task in pair["tasks"]:
            if task not in config["tasks"]:
                found.append(f"{pair['id']}: task {task} has no settings")
        found += _files(pair)
    found += _flags(config)
    return found


def _files(pair: dict[str, Any]) -> list[str]:
    from ml_stack import hub
    from ml_stack.serve import quant_guard

    out: list[str] = []
    for side, want_iq in (("iq", True), ("kquant", False)):
        found = hub.installed_for(pair[side])
        where = found.path if found else None
        if not where:
            out.append(f"{pair['id']}: {side} model {pair[side]} is not on this machine")
        elif (quant_guard.iq_quant(where) is not None) != want_iq:
            out.append(f"{pair['id']}: {side} model {where} is {'not ' if want_iq else ''}an IQ file")
    return out


def _flags(config: dict[str, Any]) -> list[str]:
    from ml_stack import decide_cli
    from ml_stack.bench import run as bench_run

    out: list[str] = []
    for cell in cells(config):
        argv = command(config, cell, Path("/nonexistent"))[3:]
        group = decide_cli.COMMANDS if cell.task == "jevbench" else bench_run.COMMANDS
        try:
            with contextlib.redirect_stderr(io.StringIO()):
                group.parser().parse_args(argv)
        except SystemExit:
            out.append(f"{cell.id}: the command's flags do not parse: {' '.join(argv)}")
    return out


def busy() -> str:
    """What holds the GPU: the lines of running llama-server and training processes."""
    done = subprocess.run(["pgrep", "-fl", QUIET], capture_output=True, text=True)
    return "\n".join(line for line in done.stdout.splitlines()
                     if "pgrep" not in line and "iq_vs_kquant" not in line)


def build_id() -> str:
    from ml_stack.serve import binary, build_platform

    try:
        return build_platform.version_of(binary.require_binary())
    except Exception as exc:  # noqa: BLE001 - recorded, not fatal
        return f"unknown ({exc})"


def harvest(config: dict[str, Any], cell: Cell, out_dir: Path) -> dict[str, Any]:
    """The numbers one finished cell produced."""
    if cell.task == "jevbench":
        got = json.loads((out_dir / f"{cell.id}.json").read_text())["runs"]["logprob"]["all"]
        return {"accuracy": got["accuracy"] * 100, "n": got["n"], "errors": got["errors"],
                "ms_per_decision": got["p50_ms"]}
    from ml_stack import bench
    from ml_stack.bench.peer_runs import exported

    store = expand(config["store"])
    if cell.task == "generation":
        run = next(r for r in exported(store)["runs"] if cell.id in r["label"])
        low, high = run["f1_band"] or (None, None)
        return {"accuracy": run["f1"] * 100, "n": run["questions"],
                "half_width": None if low is None else (high - low) * 50,
                "seconds_per_question": run["seconds"] / max(1, run["questions"])}
    run = next(r for r in bench.runs(store) if cell.id in str(r.get("label", "")))
    rows = [r for r in run["rows"] if int(r.get("streams", 1)) == 1]
    return {"speed": {str(r["prompt_tokens"]): {k: r.get(k) for k in
                                                ("prefill_tps", "decode_tps", "ttft_s", "wall_s")}
                      for r in rows}}


def run_all(config: dict[str, Any], *, resume: bool) -> int:
    where = expand(config["results"])
    out_dir = where.parent / "iq-vs-kquant-raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    done = json.loads(where.read_text()) if where.exists() else {"cells": {}}
    done["build"] = build_id()
    done["models"] = {f"{p['id']}-{q}": model_of(p[q]) for p in config["pairs"] for q in ("iq", "kquant")}
    for cell in cells(config):
        if resume and cell.id in done["cells"]:
            continue
        if held := busy():
            print(f"stopping before {cell.id}: the machine is not quiet\n{held}")
            return 3
        began = time.time()
        code = subprocess.run(command(config, cell, out_dir)).returncode
        entry: dict[str, Any] = {"cell": cell.__dict__, "exit": code, "wall_s": time.time() - began}
        if code == 0:
            entry["metrics"] = harvest(config, cell, out_dir)
        done["cells"][cell.id] = entry
        where.write_text(json.dumps(done, indent=1, sort_keys=True) + "\n")
    return 0


# -- analysis -------------------------------------------------------------------------------

def interval(values: list[float], item_half_width: float) -> tuple[float, float, float]:
    """``(mean, low, high)``: the mean of the repeats, widened to the larger of the item-level
    half-width and the t interval of the repeats."""
    mean = statistics.fmean(values)
    t_half = (T95.get(len(values) - 1, 1.96) * statistics.stdev(values) / math.sqrt(len(values))
              if len(values) > 1 else 0.0)
    half = max(item_half_width, t_half)
    return mean, mean - half, mean + half


def wilson_half_width(accuracy: float, n: int) -> float:
    """Half the 95% Wilson interval, in points, of ``accuracy`` (0-100) over ``n`` items."""
    p, z = accuracy / 100, 1.96
    if n <= 0:
        return 0.0
    spread = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return spread * 100


def _series(results: dict[str, Any], pair: str, quant: str, task: str, thinking: str
            ) -> list[dict[str, Any]]:
    wanted = [c for c in results["cells"].values()
              if c["cell"]["pair"] == pair and c["cell"]["quant"] == quant
              and c["cell"]["task"] == task and c["cell"]["thinking"] == thinking and "metrics" in c]
    return [c["metrics"] for c in sorted(wanted, key=lambda c: c["cell"]["repeat"])]


def judge(config: dict[str, Any], results: dict[str, Any], pair: str, task: str, thinking: str
          ) -> dict[str, Any]:
    """The verdict for one (pair, task, thinking) group under the pre-registered rule."""
    rule = config["decision_rule"]
    iq, kq = (_series(results, pair, q, task, thinking) for q in ("iq", "kquant"))
    out: dict[str, Any] = {"pair": pair, "task": task, "thinking": thinking,
                           "repeats": [len(iq), len(kq)]}
    if min(len(iq), len(kq)) < int(config["repeats"]):
        return {**out, "verdict": "incomplete"}
    if task == "speed":
        ratios = [k["speed"][s]["decode_tps"] / i["speed"][s]["decode_tps"]
                  for i, k in zip(iq, kq, strict=True) for s in i["speed"]]
        slower = sum(r >= float(rule["speed_ratio"]) for r in ratios)
        worse = slower >= int(rule["repeats_needed"])
        return {**out, "ratios": ratios, "verdict": "worse" if worse else "no worse"}
    half = [(wilson_half_width(m["accuracy"], m["n"]) if task == "jevbench"
             else (m.get("half_width") or 0.0)) for m in iq + kq]
    acc = {name: interval([m["accuracy"] for m in series], statistics.fmean(
        half[offset:offset + len(series)])) for name, series, offset in
        (("iq", iq, 0), ("kquant", kq, len(iq)))}
    gap = acc["kquant"][1] - acc["iq"][2]
    out["accuracy"] = acc
    out["accuracy_gap_points"] = gap
    seconds = ("ms_per_decision" if task == "jevbench" else "seconds_per_question")
    ratios = [i[seconds] / k[seconds] for i, k in zip(iq, kq, strict=True)]
    slower = sum(r >= float(rule["speed_ratio"]) for r in ratios)
    out["time_ratios"] = ratios
    worse_accuracy = gap >= float(rule["accuracy_points"])
    worse_speed = slower >= int(rule["repeats_needed"])
    level = abs(acc["iq"][0] - acc["kquant"][0]) < float(rule["accuracy_points"])
    out["verdict"] = ("worse" if worse_accuracy or worse_speed
                      else "no worse" if level and slower == 0 else "inconclusive")
    return out


def analyse(config: dict[str, Any], results: dict[str, Any]) -> dict[str, Any]:
    groups = [judge(config, results, p["id"], task, thinking)
              for p in config["pairs"] for task in p["tasks"]
              for thinking in (p["thinking"] if task == "generation" else ["off"])]
    return {"build": results.get("build"), "groups": groups}


def table(analysis: dict[str, Any]) -> str:
    """The analysis as a markdown table."""
    lines = [f"Build: {analysis['build']}", "",
             "| pair | task | thinking | IQ accuracy (95% CI) | K-quant accuracy (95% CI) | "
             "gap, points | time ratio IQ/K per repeat | verdict |", "|---|---|---|---|---|---|---|---|"]
    for g in analysis["groups"]:
        if "accuracy" in g:
            a = g["accuracy"]
            cols = [f"{a[q][0]:.1f} ({a[q][1]:.1f}-{a[q][2]:.1f})" for q in ("iq", "kquant")]
            cols += [f"{g['accuracy_gap_points']:.1f}", ", ".join(f"{r:.2f}" for r in g["time_ratios"])]
        else:
            ratios = ", ".join(f"{r:.2f}" for r in g.get("ratios", []))
            cols = ["-", "-", "-", f"decode K/IQ: {ratios}"]
        lines.append(f"| {g['pair']} | {g['task']} | {g['thinking']} | " + " | ".join(cols)
                     + f" | {g['verdict']} |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", default=str(CONFIG))
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--analyse", action="store_true")
    args = ap.parse_args(argv)
    config = load(Path(args.config))
    if args.check:
        found = problems(config)
        print("\n".join(found) if found else f"ok: {len(cells(config))} cells, nothing loaded")
        return 1 if found else 0
    if args.analyse:
        analysis = analyse(config, json.loads(expand(config["results"]).read_text()))
        base = expand(config["results"]).with_suffix("")
        base.with_suffix(".analysis.json").write_text(json.dumps(analysis, indent=1) + "\n")
        base.with_suffix(".md").write_text(table(analysis) + "\n")
        print(table(analysis))
        return 0
    if args.run:
        if found := problems(config):
            print("\n".join(found))
            return 1
        return run_all(config, resume=args.resume)
    out_dir = expand(config["results"]).parent / "iq-vs-kquant-raw"
    print(f"{len(cells(config))} cells; nothing is run. Before --run: pgrep -fl '{QUIET}' "
          "must print nothing.\n")
    for cell in cells(config):
        print(f"# {cell.id}\n{' '.join(command(config, cell, out_dir))}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
