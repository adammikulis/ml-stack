"""Supervise bounded pytest collection and release idle worker CPU permits."""
from __future__ import annotations

import contextlib
import contextvars
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import testqueue
import testslots
import testslots_policy as policy
import testslots_rpc
import testwritedeny

ROOT = Path(__file__).resolve().parent.parent
ARTIFACT_CONTEXT = contextvars.ContextVar("test_artifact_context", default=None)


@dataclass
class Preparation:
    want: int = 0
    terminals: object | None = None
    confined: object | None = None
    browser: object | None = None


def confining(environment: dict[str, str]) -> bool:
    """Whether the run asked for the macOS Seatbelt confinement kernel."""
    return sys.platform == "darwin" and environment.get("DEV_TEST_CONFINE") == "1"


def environment_for(env: dict[str, str] | None) -> dict[str, str]:
    environment = dict(os.environ if env is None else env)
    for name in ("PYTEST_XDIST_WORKER", "PYTEST_XDIST_WORKER_COUNT", "PYTEST_XDIST_TESTRUNUID"):
        environment.pop(name, None)
    if confining(environment):
        from test_kernel_isolation import validate_control_roots
        validate_control_roots(environment)
    return environment


def run_artifact_pytest(command: list[str], want: int, environment: dict[str, str], artifacts) -> int:
    from test_kernel_outputs import ArtifactOutputs
    if not confining(environment) or type(artifacts) is not ArtifactOutputs:
        raise RuntimeError("test confinement: strict artifact mode requires its macOS supervisor context")
    artifacts.validate()
    token = ARTIFACT_CONTEXT.set(artifacts)
    try:
        return run_pytest(command, want, env=environment)
    finally:
        ARTIFACT_CONTEXT.reset(token)


def open_admission(environment: dict[str, str], label: str, want: int, unix: bool):
    """The run's class, its admission endpoint and its run record."""
    run_class = policy.environment_class(environment)
    admission = testslots_rpc.UnixAdmission(False, run_class) if unix else testslots_rpc.Admission(False, run_class)
    estimate = environment.get("DEV_TEST_ESTIMATE_S")
    record = testqueue.RunRecord(testslots.slots_dir(), label, run_class, float(estimate) if estimate else None, want)
    return run_class, admission, record


def run_pytest(command: list[str], want: int = 0, label: str = "pytest", env: dict[str, str] | None = None,
               *, container: bool = False) -> int:
    """Run pytest admitted by the broker; on a host that can, the tests that do not run
    `sandbox-exec` themselves run under a kernel denial of writes to the real state root
    (scripts/testwritedeny.py), the rest beside them without it."""
    environment = environment_for(env)
    plan = None if container or confining(environment) else testwritedeny.passes(command, environment, ROOT)
    environment.pop(testwritedeny.ENV, None)
    if plan is None:
        return run_once(command, want, label, environment, container=container)
    given = testwritedeny.junit_path(command)
    statuses, extra = [], []
    for index, one in enumerate(plan):
        shown = list(one.command)
        if given is not None and index:
            descriptor, name = tempfile.mkstemp(suffix=".xml")
            os.close(descriptor)
            extra.append(Path(name))
            shown = testwritedeny.with_junit(shown, Path(name))
        statuses.append(run_once(shown, want, label, {**environment, **({testwritedeny.WRAP_ENV: "1"} if one.denied else {})}))
    try:
        for path in extra:
            if given is not None:
                testwritedeny.merge_junit(given, path)
    finally:
        for path in extra:
            path.unlink(missing_ok=True)
    return testwritedeny.combined(statuses)


def run_once(command: list[str], want: int = 0, label: str = "pytest", env: dict[str, str] | None = None,
             *, container: bool = False) -> int:
    environment = environment_for(env)
    testslots._reject_nested()
    launch = None
    if container:
        from test_container_launch import ContainerRun
        launch = ContainerRun(command, environment)
    run_class, admission, record = open_admission(environment, label, want, confining(environment) or launch is not None)
    process = None
    confined = None
    output = None
    prepared = Preparation(want=want)
    try:
        with testslots.lease(1, 1, label=f"{label}: coordinator", run_class=run_class):
            if launch is not None:
                launch.prepare()
            command, environment = prepare_pytest(command, environment, admission, launch, prepared)
            confined = prepared.confined
            record.update(state=testqueue.RUNNING, granted=int(environment["DEV_TEST_WORKERS"]), started=time.time())
            process = subprocess.Popen(policy.lower_priority(command, run_class), env=environment, cwd=Path(__file__).resolve().parent.parent,
                                       close_fds=True, pass_fds=confined.bootstrap.pass_fds if confined is not None else (),
                                       stdin=subprocess.DEVNULL, start_new_session=True,
                                       stdout=subprocess.PIPE if confined is not None else None,
                                       stderr=subprocess.STDOUT if confined is not None else None)
            if confined is not None:
                output = threading.Thread(target=relay, args=(process, confined), daemon=True)
                output.start()
                confined.bootstrap.launched(process)
            deadline = time.monotonic() + float(environment.get("DEV_TEST_WAIT_S", "3600"))
            while not admission.ready.wait(.01):
                if process.poll() is not None:
                    return process.returncode
                if time.monotonic() > deadline:
                    raise TimeoutError("testslots: pytest collection did not finish before the wait limit")
        admission.admitted.set()
        return process.wait()
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            with contextlib.suppress(subprocess.TimeoutExpired):
                process.wait(timeout=10)
            if process.poll() is None:
                process.kill()
                process.wait()
        record.close()
        try:
            if launch is not None:
                launch.close()
        finally:
            try:
                finish(output, admission, prepared.terminals, prepared.confined)
            finally:
                if prepared.browser is not None:
                    prepared.browser.close()


def relay(process, confined):
    try:
        while block := process.stdout.read1(65536):
            confined.relays[0].write(block)
    except (OSError, RuntimeError) as exc:
        confined.relay_error = exc
        while process.stdout.read1(65536):
            pass


def finish(output, admission, terminals, confined):
    try:
        if output is not None:
            output.join(timeout=10)
            if output.is_alive():
                raise RuntimeError("test confinement: descendants retained the output pipe")
    finally:
        try:
            admission.finish()
        finally:
            try:
                if terminals is not None:
                    terminals.close()
            finally:
                if confined is not None:
                    try:
                        from test_kernel_lifecycle import held_shutdown, retired_probe
                        held_shutdown()
                        retired_probe(confined)
                    finally:
                        confined.finish()


def prepare_pytest(command, environment, admission, launch, prepared):
    confined = terminals = None
    maximum = int(environment.get("DEV_TEST_BUDGET", "0")) or testslots.base_budget()
    workers = min(prepared.want or maximum, maximum)
    environment.update(DEV_TEST_SLOTS_DIR=str(testslots.slots_dir().resolve()), DEV_TEST_WORKERS=str(workers),
                       DEV_TEST_PYTEST_ENDPOINT=admission.endpoint, DEV_TEST_PYTEST_TOKEN=admission.token,
                       DEV_TEST_REMOTE_BROKER=str(testslots.slots_dir().resolve()))
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        environment.setdefault(name, "1")
    command = [part.replace("{workers}", str(workers)) for part in command]
    if launch is not None:
        environment["DEV_TEST_PYTEST_IDENTITY"] = json.dumps(admission.identity)
        command, environment = launch.command(environment)
    elif confining(environment):
        import test_browser_admission
        from test_kernel_isolation import ConfinedRun, check_selectors
        from test_terminal_bank import TerminalBank
        plan = check_selectors(command)
        if plan is not None:
            prepared.browser = test_browser_admission.BrowserRun(command, environment, admission, workers, plan=plan)
        terminals = TerminalBank(64, admission.terminal_admission)
        prepared.terminals = terminals
        admission.terminals = terminals
        environment.update(DEV_TEST_PTY_TOKEN=terminals.token, DEV_TEST_PTY_ENDPOINT=terminals.endpoint,
                           DEV_TEST_PTY_IDENTITY=json.dumps(terminals.identity), DEV_TEST_PYTEST_IDENTITY=json.dumps(admission.identity))
        with test_browser_admission.bind(prepared.browser):
            confined = ConfinedRun(command, environment, admission.endpoint,
                                   terminal_endpoint=terminals.endpoint, artifacts=ARTIFACT_CONTEXT.get())
        prepared.confined = confined
        command, environment = confined.wrapped.argv, confined.environment
        confined.recheck_images()
    if environment.pop(testwritedeny.WRAP_ENV, "") == "1":
        wrapped = testwritedeny.wrapper(testwritedeny.protected(ROOT), command)
        if wrapped is None:
            testwritedeny.say("real-state write denial could not wrap this pass; only the after-the-fact check applies")
        else:
            command, environment[testwritedeny.ENV] = wrapped, "1"
    return command, environment
