"""An evening's measurements as a file, run one at a time through the bench's own lock.

A night of benching was a zsh script in a scratch directory: a `step()` function, `&&`
between a smoke and the run it guards, a `--yes` typed onto the long ones, and a
`grep | tail` to keep the log readable. Nine of them were written in one evening
(2026-09-02), each one a little different from the last, and none of them could say what
was running or what was left. That is a queue file, and this is the thing that reads it:

    # what this evening is for
    set FX=hf:unsloth/Some-Model-GGUF/UD-Q4_K_XL/Some-Model-UD-Q4_K_XL.gguf
    set SHAPE=--serve-kv q8_0 --context 65536 --parallel 2

    smoke: sweep --serve ${FX} ${SHAPE} --label-suffix=-v2 --smoke
    then:  sweep --serve ${FX} ${SHAPE} --label-suffix=-v2 --sample 10

    show --rank docs/model-ranking.md

One `ml-stack-bench` invocation per line, `#` comments, `${VAR}` from a `set` line (or
from the environment), and a `smoke:` whose failure skips the `then:` under it and says
so. Every step is a separate `ml-stack-bench` process, so each takes the measuring lock
itself and two steps never share the GPU; the queue holds no lock of its own and is only
the thing that waits. `queue --detach` puts the whole evening in the background the way a
run detaches, `ml-stack-bench status` says which step is running and what is left, and
`ml-stack-bench stop` ends the queue and the step inside it.

Not a second scheduler: the steps are `ml-stack-bench` lines, and everything that decides
whether a measurement may run -- the self-check, the estimate against the ceiling, the
smoke, the lock -- stays in the step, where it already is.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from ml_stack.graph import bench

#: what a summary line looks like, so the log can be read by eye and by `grep`
SUMMARY = "=== {clock} step {n}/{total}: {words} -- {state} ({seconds:.0f}s)"

_SET = re.compile(r"^set\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$")
_VAR = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")

#: how many words of a line stand for it in the summary
WORDS = 4


class QueueError(ValueError):
    """A queue file that cannot be run as written -- said with the line it is on, before
    anything is measured, because a typo found an hour in has cost an hour."""


@dataclass
class Step:
    """One `ml-stack-bench` invocation out of the file."""

    n: int                          # 1-based, over the whole queue
    line: int                       # the line of the file it came from
    kind: str                       # "step", "smoke", or "then"
    argv: list[str]                 # what follows `ml-stack-bench`
    label: str = ""                 # what `--resume` looks for in the runs store
    state: str = "queued"           # queued | ok | failed | skipped
    seconds: float = 0.0
    why: str = ""                   # why it was skipped

    @property
    def words(self) -> str:
        """The first few words of the line, which is what a person calls the step."""
        said = " ".join(self.argv[:WORDS])
        return said if len(said) <= 56 else said[:53] + "..."

    def public(self) -> dict[str, Any]:
        return {"n": self.n, "line": self.line, "kind": self.kind, "words": self.words,
                "label": self.label, "state": self.state,
                "seconds": round(self.seconds, 1), "why": self.why}


# -- reading the file ----------------------------------------------------------------------

def expand(text: str, variables: dict[str, str], *, line: int) -> str:
    """``${NAME}`` replaced from ``variables``, else from the environment.

    An unset name is refused rather than expanded to nothing: a queue whose `${FX}` went
    missing would otherwise serve the default model for six hours and call it a result.
    """
    missing: list[str] = []

    def one(m: "re.Match[str]") -> str:
        name = m.group(1)
        if name in variables:
            return variables[name]
        if name in os.environ:
            return os.environ[name]
        missing.append(name)
        return ""

    said = _VAR.sub(one, text)
    if missing:
        raise QueueError(f"line {line}: nothing sets " + ", ".join(f"${{{n}}}" for n in missing))
    return said


def parse(text: str, *, validate: bool = True) -> list[Step]:
    """The steps a queue file names, in order, with its `set` lines expanded.

    Every line is checked against ``ml-stack-bench``'s own parser as it is read, so a flag
    that does not exist is a refusal before the first model is loaded rather than a
    traceback after the fourth.
    """
    variables: dict[str, str] = {}
    steps: list[Step] = []
    pending_smoke: Step | None = None
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        named = _SET.match(line)
        if named:
            variables[named.group(1)] = expand(named.group(2).strip(), variables, line=lineno)
            continue
        kind = "step"
        if line.lower().startswith("smoke:"):
            kind, line = "smoke", line[len("smoke:"):].strip()
        elif line.lower().startswith("then:"):
            kind, line = "then", line[len("then:"):].strip()
        line = expand(line, variables, line=lineno)
        try:
            argv = shlex.split(line, comments=True)
        except ValueError as why:                       # an unbalanced quote
            raise QueueError(f"line {lineno}: {why}") from None
        if not argv:
            continue
        if argv[0] == "ml-stack-bench":                 # written out in full: allowed, ignored
            argv = argv[1:]
        if not argv:
            raise QueueError(f"line {lineno}: nothing to run")
        if kind == "then" and pending_smoke is None:
            raise QueueError(f"line {lineno}: `then:` with no `smoke:` above it")
        if kind == "smoke" and pending_smoke is not None:
            raise QueueError(f"line {lineno}: `smoke:` on line {pending_smoke.line} has no "
                             f"`then:` under it")
        step = Step(n=len(steps) + 1, line=lineno, kind=kind, argv=argv)
        if validate:
            step.label = label_of(argv, where=f"line {lineno}")
        steps.append(step)
        pending_smoke = step if kind == "smoke" else None
    if pending_smoke is not None:
        raise QueueError(f"line {pending_smoke.line}: `smoke:` with no `then:` under it")
    return steps


def read(path: str | Path, *, validate: bool = True) -> list[Step]:
    """`parse` over a file, said with the file's name when it will not read."""
    where = Path(path).expanduser()
    try:
        text = where.read_text(encoding="utf-8")
    except OSError as why:
        raise QueueError(f"{where}: {why}") from None
    try:
        return parse(text, validate=validate)
    except QueueError as why:
        raise QueueError(f"{where}: {why}") from None


def _parsed(argv: Sequence[str], *, where: str = "") -> Any:
    """``argv`` through ``ml-stack-bench``'s own parser, with argparse's exit turned into a
    `QueueError` and its usage kept out of the log."""
    import contextlib
    import io

    from ml_stack.graph.bench.run import _parser

    said = io.StringIO()
    try:
        with contextlib.redirect_stderr(said), contextlib.redirect_stdout(said):
            return _parser().parse_args(list(argv))
    except SystemExit:
        first = next((ln for ln in reversed(said.getvalue().splitlines()) if ln.strip()), "")
        raise QueueError(f"{where}: ml-stack-bench {' '.join(argv)}\n  {first.strip()}") from None


def label_of(argv: Sequence[str], *, where: str = "") -> str:
    """What the runs this step keeps are labelled with, as far as the line can say.

    `run` and `concurrent` are told their label; a `sweep` builds one per model out of the
    model's file name (the first fourteen characters, as `sweep` does) and `--label-suffix`,
    and every way it asks is that stem with something appended. So the stem is what
    `--resume` matches on, and a step that keeps nothing under a name -- `show`, `report`,
    `prepare` -- returns "" and is never skipped, being the cheap part and the conclusion.
    """
    args = _parsed(argv, where=where or "queue")
    told = str(getattr(args, "label", "") or "")
    if told:
        return told
    model = (next(iter(getattr(args, "serve", []) or []), "")
             or str(getattr(args, "model", "") or ""))
    if not model:
        return ""
    stem = str(model).rsplit("/", 1)[-1].removesuffix(".gguf")[:14]
    return stem + str(getattr(args, "label_suffix", "") or "")


# -- what the store already holds ----------------------------------------------------------

def kept_since(started: str, *, store: str | Path | None = None) -> list[str]:
    """The labels of every run kept at or after ``started``, for `--resume`."""
    where = Path(store) if store else bench.HOME / "runs.ladybug"
    try:
        if not Path(str(where)).expanduser().exists():
            return []
        rows = bench.runs(where)
    except Exception:                            # noqa: BLE001 - a store that will not open
        return []                                #   has nothing to skip a step on
    return [str(r.get("label", "")) for r in rows if str(r.get("at", "")) >= started[:19]]


def _already(step: Step, labels: Sequence[str]) -> bool:
    """Whether the store holds a run this step would only measure again."""
    return bool(step.label) and any(one.startswith(step.label) for one in labels)


# -- the state a queue leaves behind, for `status` -------------------------------------------

def state_file() -> Path:
    """Where the running queue writes what it is on. Beside `measuring.json`, and read the
    same way: a record whose pid has gone is a queue that ended, not one that is running."""
    return bench.HOME / "queue.json"


def _write_state(state: dict[str, Any]) -> None:
    where = state_file()
    where.parent.mkdir(parents=True, exist_ok=True)
    where.write_text(json.dumps(state, indent=1), encoding="utf-8")


def _state() -> dict[str, Any]:
    try:
        held = json.loads(state_file().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return held if isinstance(held, dict) else {}


def queue_status() -> str:
    """The `queue` block of ``ml-stack-bench status``: which step is running and what is
    left, or one line about the queue that last ran. Empty when no queue has ever run.

    `status` calls this; nothing else here knows about `status`, so the two stay apart.
    """
    from ml_stack.serve.process import pid_exists

    held = _state()
    if not held:
        return ""
    steps = [s for s in held.get("steps") or [] if isinstance(s, dict)]
    done = [s for s in steps if s.get("state") in ("ok", "failed", "skipped")]
    tally = (f"{sum(1 for s in done if s['state'] == 'ok')} ok, "
             f"{sum(1 for s in done if s['state'] == 'failed')} failed, "
             f"{sum(1 for s in done if s['state'] == 'skipped')} skipped")
    if held.get("ended") or not pid_exists(held.get("pid")):
        return (f"queue: {held.get('file', '?')} -- ended {held.get('ended', 'without a word')}; "
                f"{len(steps)} step(s), {tally}")
    running = next((s for s in steps if s.get("state") == "running"), None)
    lines = [f"queue: {held.get('file', '?')} (pid {held.get('pid')}), started "
             f"{held.get('started', '?')}"]
    if running:
        lines.append(f"  step {running.get('n')}/{len(steps)}: {running.get('words', '')}")
    lines.append(f"  done: {tally}")
    left = [s for s in steps if s.get("state") in ("queued", None)]
    lines.append(f"  left: {len(left)}" + (f" -- next {left[0].get('words', '')}" if left else ""))
    return "\n".join(lines)


# -- running it ------------------------------------------------------------------------------

_running: "subprocess.Popen[bytes] | None" = None


def run_step(argv: Sequence[str]) -> int:
    """One `ml-stack-bench argv` as its own process, its output going where the queue's
    goes -- the terminal, or the detached queue's log.

    Its own process, not a function call: that is what puts each step through the measuring
    lock, the self-check, the estimate and the smoke it already has, and what lets `stop`
    take a step down without taking the queue's own interpreter with it.
    """
    global _running
    command = [sys.executable, "-m", "ml_stack.graph.bench", *argv]
    sys.stdout.flush()
    _running = subprocess.Popen(command, stdin=subprocess.DEVNULL,
                                env={**os.environ, "PYTHONUNBUFFERED": "1"})
    try:
        return _running.wait()
    finally:
        _running = None


def _stop_running(grace: float = 60.0) -> None:
    """SIGTERM the step in flight and wait for it, so the model it put up comes down."""
    child = _running
    if child is None or child.poll() is not None:
        return
    print("[killed] the queue was told to stop; asking the step in flight to stop too",
          flush=True)
    try:
        child.terminate()
        child.wait(timeout=grace)
    except (OSError, subprocess.TimeoutExpired):
        pass


def _for_step(argv: list[str], *, yes: bool, ceiling: float) -> list[str]:
    """The line as written, plus the go-ahead and the ceiling the queue was given -- but
    only for a step that takes them, and never over what the line says itself."""
    args = _parsed(argv, where="queue")
    out = list(argv)
    if yes and hasattr(args, "yes") and "--yes" not in out:
        out.append("--yes")
    if ceiling and hasattr(args, "ceiling") and not any(
            a == "--ceiling" or a.startswith("--ceiling=") for a in out):
        out += ["--ceiling", str(ceiling)]
    return out


def run_queue(path: str | Path, *, dry_run: bool = False, resume: bool = False,
              yes: bool = False, ceiling: float = 0.0,
              runner: Callable[[Sequence[str]], int] | None = None) -> int:
    """Run every step of the queue file at ``path``, one at a time. 0 unless a step failed.

    A step that fails does not end the queue -- the evening's other measurements are still
    worth having -- except that a failed `smoke:` skips the `then:` it guards, which is
    what the `&&` in the shell scripts was for.
    """
    steps = read(path)
    where = str(Path(path).expanduser())
    total = len(steps)
    if dry_run:
        print(f"{where}: {total} step(s), "
              f"{sum(1 for s in steps if s.kind == 'smoke')} smoke/then pair(s)")
        for step in steps:
            told = _for_step(step.argv, yes=yes, ceiling=ceiling)
            print(f"  {step.n:>2}  {step.kind:<5}  ml-stack-bench {' '.join(told)}"
                  + (f"    [label {step.label}]" if step.label else ""))
        print("nothing ran: --dry-run")
        return 0

    was = _state()
    started = (str(was.get("started") or "") if resume and was.get("file") == where
               else "") or time.strftime("%FT%T")
    labels = kept_since(started) if resume else []
    if resume:
        print(f"resuming {where}: skipping what has been kept since {started}")
    state = {"file": where, "started": started, "pid": os.getpid(), "total": total,
             "steps": [s.public() for s in steps]}
    _write_state(state)

    def note(step: Step, state_name: str) -> None:
        step.state = state_name
        state["steps"][step.n - 1] = step.public()
        _write_state(state)

    previous = None
    try:
        previous = signal.signal(signal.SIGTERM, lambda *_: (_stop_running(), sys.exit(143)))
    except ValueError:
        pass                                 # not the main thread: nothing to hand a signal
    failed_smoke: Step | None = None
    try:
        for step in steps:
            if step.kind == "then" and failed_smoke is not None:
                step.why = f"its smoke (step {failed_smoke.n}) failed"
                _say(step, "skipped", total)
                note(step, "skipped")
                failed_smoke = None
                continue
            failed_smoke = None
            if resume and _already(step, labels):
                step.why = f"{step.label} is already kept since {started}"
                _say(step, "skipped", total)
                note(step, "skipped")
                continue
            note(step, "running")
            print(f"--- {time.strftime('%H:%M:%S')} step {step.n}/{total}: "
                  f"ml-stack-bench {' '.join(step.argv)}", flush=True)
            began = time.monotonic()
            code = (runner or run_step)(_for_step(step.argv, yes=yes, ceiling=ceiling))
            step.seconds = time.monotonic() - began
            if code != 0:
                step.why = f"exit {code}"
            _say(step, "ok" if code == 0 else "failed", total)
            note(step, "ok" if code == 0 else "failed")
            if code != 0 and step.kind == "smoke":
                failed_smoke = step
    finally:
        state["ended"] = time.strftime("%FT%T")
        state["steps"] = [s.public() for s in steps]
        _write_state(state)
        if previous is not None:
            signal.signal(signal.SIGTERM, previous)
    bad = [s for s in steps if s.state == "failed"]
    print(f"=== {time.strftime('%H:%M:%S')} queue done: {total} step(s), "
          f"{sum(1 for s in steps if s.state == 'ok')} ok, {len(bad)} failed, "
          f"{sum(1 for s in steps if s.state == 'skipped')} skipped", flush=True)
    return 1 if bad else 0


def _say(step: Step, state: str, total: int) -> None:
    """The one line per step the log is read by."""
    line = SUMMARY.format(clock=time.strftime("%H:%M:%S"), n=step.n, total=total,
                          words=step.words, state=state, seconds=step.seconds)
    print(line + (f": {step.why}" if step.why else ""), flush=True)
