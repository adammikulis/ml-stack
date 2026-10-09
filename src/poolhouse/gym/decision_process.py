"""Bounded decision inference outside the native physics process."""

import contextlib
import json
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

from poolhouse.decide.pointer import PointerDecider, device_name
from poolhouse.decide.sources import local_source
from poolhouse.gym.transport import interpreter, python_environment
from poolhouse.platform import start_process, terminate_process_group
from poolhouse.serve import broker_wire
from poolhouse.serve.exit_guard import protect, release
from poolhouse.serve.gpu import hold


def decision_controller(checkpoint=None, device="cpu"):
    """Load a default or verified trained pointer decider on the selected device."""
    if not checkpoint:
        return PointerDecider(device=device)
    source = Path(checkpoint).expanduser().resolve()
    if not source.is_dir():
        raise ValueError("Decision checkpoint must be a local trained model directory")
    local_source(source, download=False)
    return PointerDecider(source, device=device)



class DecisionProcess:
    """Own one model process and one outstanding decision request."""

    def __init__(self, checkpoint=None, *, module="poolhouse.gym.decision_process", device="cpu"):
        self.events = queue.Queue(maxsize=1)
        self.requests = queue.Queue(maxsize=1)
        self.pending = None
        self.status, self.error = "loading", None
        self.device = device
        self.handle = start_process(
            [interpreter(), "-m", module, json.dumps(checkpoint), device],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=sys.stderr,
            text=True, bufsize=1, env={**python_environment(), "OMP_NUM_THREADS": "1",
                                      "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"})
        protect(self.handle.pid)
        threading.Thread(target=self.read, daemon=True).start()
        threading.Thread(target=self.write, daemon=True).start()

    def read(self):
        for line in self.handle.stdout:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            self.events.put(event)
        self.handle.stdout.close()

    def write(self):
        try:
            while (request := self.requests.get()) is not None:
                self.handle.stdin.write(json.dumps(request) + "\n")
                self.handle.stdin.flush()
        except (BrokenPipeError, OSError, ValueError):
            return

    def poll(self):
        try:
            event = self.events.get_nowait()
        except queue.Empty:
            if self.handle.poll() is not None and self.status != "error":
                self.status, self.error = "error", "Decision model process exited"
            return None
        self.status = event["status"]
        self.error = event.get("error")
        self.device = event.get("device", self.device)
        if "result" in event:
            self.pending = None
        return event

    def submit(self, request):
        if self.status != "ready" or self.pending is not None:
            return False
        self.pending = request
        self.requests.put_nowait(request)
        return True

    def close(self):
        try:
            if self.handle.poll() is None:
                terminate_process_group(self.handle)
        except (ProcessLookupError, PermissionError):
            if self.handle.poll() is None:
                raise
        try:
            self.handle.wait(timeout=.4)
        except subprocess.TimeoutExpired:
            terminate_process_group(self.handle, force=True)
            self.handle.wait(timeout=.4)
        release(self.handle.pid)
        with contextlib.suppress(queue.Full):
            self.requests.put_nowait(None)
        with contextlib.suppress(BrokenPipeError, OSError):
            self.handle.stdin.close()


def emit(event, stream):
    stream.write(json.dumps(event) + "\n")
    stream.flush()


def serve():
    """Load verified local model files and emit typed decision events."""
    output = sys.stdout
    try:
        device = device_name(sys.argv[2] if len(sys.argv) > 2 else "cpu")
        lease = contextlib.nullcontext()
        if device != "cpu":
            emit({"status": "queued", "device": device}, output)
            if not broker_wire.status(start=True).get("exclusive_gpu_claims"):
                raise RuntimeError("Decision accelerator requires an updated exclusive-GPU broker")
            lease = hold("Gym decision model", wait_s=600)
        with lease:
            run_policy(json.loads(sys.argv[1]), device, output)
    except (RuntimeError, ValueError, OSError, ImportError, KeyError, TypeError) as exc:
        emit({"status": "error", "error": str(exc)}, output)


def run_policy(checkpoint, device, output):
    """Keep the device lease through model loading and bounded inference."""
    emit({"status": "loading", "device": device}, output)
    with contextlib.redirect_stdout(sys.stderr):
        policy = decision_controller(checkpoint, device)
        policy.load()
    emit({"status": "ready", "model": policy.model, "device": device}, output)
    for line in sys.stdin:
        request = json.loads(line)
        began = time.perf_counter()
        with contextlib.redirect_stdout(sys.stderr):
            answer = policy.decide("Choose the next safe environment action",
                                   request.get("model_state", request["state"]), request["options"])
        result = {**request, "choice": answer.choice, "probabilities": dict(answer.scores),
                  "abstained": answer.abstained, "model": answer.model, "backend": answer.backend,
                  "device": device, "latency_ms": (time.perf_counter() - began) * 1000}
        emit({"status": "ready", "device": device, "result": result}, output)


if __name__ == "__main__":
    serve()
