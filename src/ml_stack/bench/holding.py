"""What the server holding the model costs, and whether it is free to be timed.

`Watching` samples what the process tree holds while the questions are asked; `footprint`
folds those peaks into the run's record and `beyond_weights` splits them into the weights
and everything else. `machine_memory`, `footprint_of`, `serving_pids` and `process_tree`
are what one sample reads. `slot_count` says how many conversations the server holds,
`busy` how many it is already working on, and `_idle` refuses to time one somebody else
is using.
"""

from __future__ import annotations

import sys
import threading
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

# The package is the namespace the tests and `selfcheck` patch -- `bench.footprint`,
# `bench.busy` -- so anything patchable is looked up there at call time, never bound here
# at import.
from ml_stack import bench
from ml_stack.bench.backends import http_of, processes, served_by
from ml_stack.log import warn

# -------------------------------------------------------- what the server held, at its most
#
# `footprint` reads memory once, when the questions are over, and by then the cache that was
# full while four conversations were in flight has been let go. Flash-Next was sized at ~90G
# from readings like that and nobody could say whether 90G was the peak or the trough -- so
# "how many of these fit on this machine" had no number behind it.
#
# Two figures per sample, because macOS shows two and they are not the same thing:
#
#   `resident_peak`  -- Activity Monitor's **Real Mem**. The resident set: every physical
#                       page mapped in, shared and file-backed pages included. `ps -o rss`,
#                       `psutil.Process.memory_info().rss`, `ri_resident_size`.
#   `footprint_peak` -- Activity Monitor's **Memory**. The phys_footprint: dirty and
#                       compressed pages this process is charged for, clean file-backed
#                       pages excluded. `vmmap --summary`'s footprint, `ps -o footprint`,
#                       `ri_phys_footprint`. psutil has no field for it, so it is read
#                       through `proc_pid_rusage` -- see `_rusage_footprint`.
#
# They diverge exactly where it matters: llama.cpp mmaps its weights, so an 87G model can be
# most of the resident set and almost none of the footprint. Memory pressure is charged on
# the footprint and eviction is felt on the resident set, so both are kept and neither is
# called "the memory".
#
# Beside them the machine's own: `wired_peak` (what nothing can page out) against
# `wired_baseline` (the same, before the server came up -- the difference is the server's
# own wired cost), and `available_low`, free plus inactive at its lowest, which is the
# pressure proxy `vm_stat` gives.

# How often the sampler looks, in seconds. Once a second: a runner Ollama spawns on the
# first request holds the weights within a second of it, and a llama-server's resident
# set moves on the scale of a prompt being read.
SAMPLE_EVERY = 1.0

# `proc_pid_rusage(pid, RUSAGE_INFO_V4, &buf)`, and where `ri_phys_footprint` sits in
# `rusage_info_v4`: sixteen bytes of uuid, then user and system time, two wakeup counts,
# pageins, wired size, resident size, and the footprint -- the eighth `uint64_t`. Verified
# against `ps -o rss` on this machine rather than counted off the header, because counting
# it off the header put it one slot late and read the process's start time as a footprint
# of eight terabytes.
RUSAGE_INFO_V4 = 4
PHYS_FOOTPRINT_AT = 16 + 7 * 8


def _rusage_footprint(pid: int) -> int:
    """macOS's phys_footprint for ``pid`` -- Activity Monitor's "Memory" -- or 0.

    The seam: everything else about the sampler is ordinary Python, and this one function
    reaches into libSystem through ctypes. `footprint_of` calls it as ``bench.``, the way
    everything patchable in this package is called, so a test replaces this one function and
    nothing anywhere else has to pretend to be a kernel. 0 for a process that is gone, a platform without the
    call, or any failure at all -- a memory reading is never worth a run not finishing.
    """
    if sys.platform != "darwin":
        return 0
    try:
        import ctypes
        import ctypes.util

        lib = ctypes.CDLL(ctypes.util.find_library("System") or "libSystem.dylib")
        buf = ctypes.create_string_buffer(1024)
        if lib.proc_pid_rusage(ctypes.c_int(int(pid)), ctypes.c_int(RUSAGE_INFO_V4),
                               ctypes.byref(buf)) != 0:
            return 0
        return int.from_bytes(buf.raw[PHYS_FOOTPRINT_AT:PHYS_FOOTPRINT_AT + 8], sys.byteorder)
    except Exception:  # noqa: BLE001 - a number we could not get is not a failed run
        return 0


def footprint_of(process: Any) -> int:
    """One process's phys_footprint in bytes -- Activity Monitor's "Memory".

    psutil's own field where a build has one (some expose it in ``memory_full_info``), the
    `proc_pid_rusage` read where it does not, and the resident set on every platform that
    has no such distinction -- Linux and Windows charge a process for what is resident, so
    there the two figures are the same number and the table says so by printing it twice.
    """
    try:
        info = process.memory_info()
        for name in ("phys_footprint", "footprint"):
            got = int(getattr(info, name, 0) or 0)
            if got:
                return got
    except Exception:  # noqa: BLE001
        pass
    through_kernel = bench._rusage_footprint(int(getattr(process, "pid", 0) or 0))
    if through_kernel:
        return through_kernel
    try:
        return int(process.memory_info().rss)
    except Exception:  # noqa: BLE001
        return 0


def machine_memory() -> dict[str, int]:
    """``wired`` and ``unpressured`` for the whole machine, as far as this platform says.

    ``wired`` is what nothing can page out -- the figure a server's own load moves, and the
    one that decides whether a second model fits beside the first. ``unpressured`` is free
    plus inactive: the pages that are there for the asking, which is the closest thing to
    `vm_stat`'s pressure that a portable call gives. Missing keys on a platform whose
    ``virtual_memory`` has no such field, rather than a zero that reads as measured.
    """
    out: dict[str, int] = {}
    try:
        import psutil

        vm = psutil.virtual_memory()
    except Exception:  # noqa: BLE001
        return out
    if hasattr(vm, "wired"):
        out["wired"] = int(vm.wired)
    free, inactive = int(getattr(vm, "free", 0) or 0), int(getattr(vm, "inactive", 0) or 0)
    if free or inactive:
        out["unpressured"] = free + inactive
    elif getattr(vm, "available", 0):
        out["unpressured"] = int(vm.available)
    return out


def said_by(client: Any) -> dict[str, Any] | None:
    """What a client says served it (`Client.served_by`), for a program only the client
    can ask; None for a llama-server's, which `footprint` reads from ``/props`` itself."""
    from ml_stack.bench.backends import speaks_llama

    if client is None or not hasattr(client, "served_by") or speaks_llama(client):
        return None
    return served_by(client)


def serving_pids(base_url: str, client: Any = None) -> list[int]:
    """The pids holding the weights behind ``base_url``: what the client says
    (`Client.processes`), else the llama-server with ``--port N`` on its command line.

    Empty for a run against a ``--base-url`` somebody else put up on another machine:
    nothing here owns that port, and a run that samples nothing must say nothing rather
    than report zeroes.
    """
    named = processes(client) if client is not None else []
    if named:
        return named
    try:
        import psutil

        port = int(str(http_of(base_url)).rsplit(":", 1)[-1].strip("/"))
    except Exception:  # noqa: BLE001
        return []
    try:
        for process in psutil.process_iter(["pid", "cmdline"]):
            line = " ".join(process.info.get("cmdline") or ())
            if "llama-server" in line and f"--port {port}" in line:
                return [int(process.info.get("pid") or getattr(process, "pid", 0) or 0)]
    except Exception:  # noqa: BLE001
        return []
    return []


def serving_process(base_url: str, client: Any = None) -> Any | None:
    """The first process holding the weights behind ``base_url``, as a psutil process,
    or None -- see `serving_pids`."""
    pids = serving_pids(base_url, client)
    if not pids:
        return None
    try:
        import psutil

        return psutil.Process(pids[0])
    except Exception:  # noqa: BLE001
        return None


def _every(psutil: Any) -> list[Any]:
    try:
        return list(psutil.process_iter(["pid", "cmdline"]))
    except Exception:  # noqa: BLE001
        return []


def process_tree(pids: Sequence[int]) -> list[Any]:
    """Every process under ``pids`` -- each one and its children, read now -- as psutil
    processes, each once. Ollama spawns the runner that holds the weights after the first
    request, so the tree is re-read on every call rather than kept."""
    try:
        import psutil
    except Exception:  # noqa: BLE001
        return []
    out: dict[int, Any] = {}
    for pid in pids:
        try:
            parent = psutil.Process(int(pid))
        except Exception:  # noqa: BLE001 - gone, not ours to read, or no such call here
            parent = next((p for p in _every(psutil)
                           if int((getattr(p, "info", None) or {}).get("pid")
                                  or getattr(p, "pid", 0) or 0) == int(pid)), None)
            if parent is None:
                continue
        out.setdefault(int(pid), parent)
        try:
            for child in parent.children(recursive=True):
                out.setdefault(int(getattr(child, "pid", 0) or 0), child)
        except Exception:  # noqa: BLE001
            continue
    return list(out.values())


class Watching:
    """A thread that reads what the server holds every `SAMPLE_EVERY` seconds, and keeps
    the worst of it.

    ``peaks`` is what goes into the run's record: ``resident_peak`` (summed over the
    process tree, with ``resident_peak_at`` the sample it was seen on and ``processes`` the
    most the tree held), ``footprint_peak``, ``wired_peak`` and ``wired_baseline`` for the
    machine, ``available_low``, ``sampled_every`` and ``samples``. A run with no process to
    watch reports ``sampled: "no served process on this machine"`` and no figures. The pids
    come from the ``client`` (`Client.processes`) when it can say, else from the port, and
    the tree under them is re-read on every sample.
    """

    def __init__(self, base_url: str, *, every: float = SAMPLE_EVERY,
                 baseline: Mapping[str, int] | None = None, start: bool = True,
                 client: Any = None) -> None:
        self.base_url = base_url
        self.every = float(every)
        self.client = client
        self.pids = serving_pids(base_url, client)
        tree = process_tree(self.pids) if self.pids else []
        self.process = tree[0] if tree else None
        # what served it, read once here so `footprint` finds it beside the peaks: the
        # client is the one thing that can say, and `footprint` is not handed the client
        self.served = said_by(client)
        # taken before the server came up when a caller has one; otherwise the first thing
        # this thread sees, which includes the server and is honest about saying so
        self.baseline = dict(baseline) if baseline is not None else machine_memory()
        self.baseline_before_load = baseline is not None
        self.resident = self.footprint = self.wired = 0
        self.resident_at = 0
        self.processes = 0
        self.available: int | None = None
        self.samples = 0
        self._stop = threading.Event()
        # ``start=False`` leaves the thread unstarted and `_once` the only way it samples,
        # which is how it is tested: a series read at times a test chooses says exactly what
        # the maxima are, where a thread and a clock say something near them.
        self._thread = threading.Thread(target=self._watch, daemon=True)
        if start:
            self._thread.start()

    def _once(self) -> None:
        if not self.pids:
            # a runner that was not there when the watch began: asked again each tick
            self.pids = serving_pids(self.base_url, self.client)
        tree = process_tree(self.pids) if self.pids else []
        if tree:
            resident = footprint = 0
            counted = 0
            for process in tree:
                try:
                    resident += int(process.memory_info().rss)
                    footprint += footprint_of(process)
                    counted += 1
                except Exception:  # noqa: BLE001 - one that has gone is not an error here
                    continue
            if counted:
                self.process = tree[0]
                self.processes = max(self.processes, counted)
                if resident > self.resident:
                    self.resident, self.resident_at = resident, self.samples + 1
                self.footprint = max(self.footprint, footprint)
        elif self.pids:
            self.process = None
        held = machine_memory()
        if "wired" in held:
            self.wired = max(self.wired, held["wired"])
        if "unpressured" in held:
            self.available = (held["unpressured"] if self.available is None
                              else min(self.available, held["unpressured"]))
        self.samples += 1

    def _watch(self) -> None:
        while not self._stop.is_set():
            self._once()
            self._stop.wait(self.every)

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=30)
        self._once()                     # one last look, so a short run samples at all
        return self.peaks

    @property
    def peaks(self) -> dict[str, Any]:
        out: dict[str, Any] = {"sampled_every": self.every, "samples": self.samples}
        if self.served:
            out["served_by"] = dict(self.served)
        if self.pids:
            out["pids"] = list(self.pids)
        if self.process is None and not self.resident:
            out["sampled"] = "no served process on this machine"
        if self.resident:
            out["resident_peak"] = self.resident
            out["resident_peak_at"] = self.resident_at
            out["processes"] = self.processes
        if self.footprint:
            out["footprint_peak"] = self.footprint
        if self.wired:
            out["wired_peak"] = self.wired
        if "wired" in self.baseline:
            out["wired_baseline"] = self.baseline["wired"]
            # False when the baseline was taken with the server already up, so the
            # difference is not the server's own wired cost and nothing should read it as one
            out["wired_baseline_before_load"] = self.baseline_before_load
        if self.available is not None:
            out["available_low"] = self.available
        return out


# What the sampler saw, by the server it was watching, for `footprint` to fold into the
# record. The two belong together -- both are "what the server costs" -- and putting the
# handover here rather than through every caller is what lets a run keep peaks without the
# serving path having to carry them. One bench measures one server at a time, under the
# measuring lock, so the key is enough.
_WATCHED: dict[str, dict[str, Any]] = {}


def watched(base_url: str, peaks: Mapping[str, Any]) -> None:
    """Leave what `Watching` recorded where `footprint` will find it."""
    _WATCHED[http_of(base_url)] = dict(peaks)


def watching(base_url: str, *, every: float = SAMPLE_EVERY,
             baseline: Mapping[str, int] | None = None, client: Any = None) -> Any:
    """`Watching` over ``base_url``, as a context manager that files its peaks on exit."""
    from contextlib import contextmanager

    @contextmanager
    def held() -> Any:
        watcher = Watching(base_url, every=every, baseline=baseline, client=client)
        try:
            yield watcher
        finally:
            watched(base_url, watcher.stop())

    return held()


class _Peak:
    """The most a server held while a run was going.

    `footprint` reads resident memory once, after the fact, and a cache that was full while
    four conversations were in flight has been let go by then. Sampled every second on a
    thread; ``stop`` returns the footprint with the largest resident figure seen.
    """

    def __init__(self, base_url: str, every: float = 1.0) -> None:
        self.base_url = base_url
        self.every = every
        self.most = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._watch, daemon=True)
        self._thread.start()

    def _watch(self) -> None:
        while not self._stop.is_set():
            self.most = max(self.most,
                            int(bench.footprint(self.base_url).get("resident_bytes") or 0))
            self._stop.wait(self.every)

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        self._thread.join(timeout=30)
        out = bench.footprint(self.base_url)
        if self.most > int(out.get("resident_bytes") or 0):
            out["resident_bytes"] = self.most
            for derived_key in ("kv_and_run_bytes", "bytes_per_1k_context", "mmapped"):
                out.pop(derived_key, None)
            try:
                out = beyond_weights(out)
            except Exception:  # noqa: BLE001 - a summary line is never worth a run's answers
                pass
        return out


def footprint(base_url: str, client: Any = None) -> dict[str, Any]:
    """What the server holding this model costs to keep up, and what it is.

    The resident figure is summed over the process tree the ``client`` names
    (`serving_pids`), which is how Ollama's runner is counted beside its listener, and the
    peaks a `Watching` left behind are folded in. ``served_by`` is the program, its
    version, the format, the runtime and the quant (`backends.served_by`); the weights are
    its when the program can say, else the size of what ``/props`` names on disk.
    """
    from ml_stack.bench.backends import llama_served_by, props_of

    out: dict[str, Any] = {"base_url": base_url}
    # what the sampler saw while the questions were asked, and what it learnt of the
    # server from the client it was handed -- see `Watching.peaks`
    seen = _WATCHED.pop(http_of(base_url), {})
    pids = [int(p) for p in seen.pop("pids", ()) or ()]
    props = props_of(base_url)
    if props:
        out["model"] = str(props.get("model_path") or "").rsplit("/", 1)[-1]
        out["slots"] = int(props.get("total_slots") or 0)
        out["context"] = int((props.get("default_generation_settings") or {}).get("n_ctx") or 0)
    record = (said_by(client) or seen.pop("served_by", None)
              or llama_served_by(base_url, props=props))
    if record:
        out["served_by"] = dict(record)
        if not out.get("model") and record.get("model"):
            out["model"] = str(record["model"])
        if record.get("weights_bytes"):
            out["weights_bytes"] = int(record["weights_bytes"])
    try:
        tree = process_tree(serving_pids(base_url, client) or pids)
        held = 0
        for process in tree:
            try:
                held += int(process.memory_info().rss)
            except Exception:  # noqa: BLE001 - one that has gone holds nothing
                continue
        if tree and held:
            out["resident_bytes"] = held
    except Exception:  # noqa: BLE001 - a number we could not get is not a failed run
        pass
    # The weights come from /props, not the command line: a model served by `hf:` reference
    # is on the command line as a repository, and only the server knows where it landed.
    if "weights_bytes" not in out:
        try:
            where = Path(str(props.get("model_path") or ""))
            if where.exists():
                shards = sorted(where.parent.glob(where.name.replace("00001", "*"))) or [where]
                out["weights_bytes"] = sum(s.stat().st_size for s in shards if s.is_file())
        except Exception:  # noqa: BLE001 - a number we could not get is not a failed run
            pass
    # Resident minus weights, when that means anything. It does not always: llama.cpp mmaps
    # the weights, so a page is resident only once it has been touched, and an MoE that uses
    # ten experts of five hundred never touches most of them. Qwen3.8-Flash-Next sat at 63G
    # resident against 87G of weights on disk -- the subtraction goes negative, clamps to
    # zero and prints as a dash, which reads as "not measured" rather than "not meaningful".
    #
    # So it is only reported when the model really is fully resident, and `resident_bytes`
    # is carried either way, because what the process actually holds is the number that
    # decides how many of them fit.
    #
    # And what it held at its most, if a `Watching` was running over this server while the
    # questions were asked: a reading taken here is taken after the last answer, when the
    # cache that was full during it has been let go. `beyond_weights` prefers the peak.
    out.update(seen)
    try:
        return beyond_weights(out)
    except Exception:  # noqa: BLE001 - a summary line is never worth a run's answers
        return out


def beyond_weights(out: dict[str, Any]) -> dict[str, Any]:
    """Split what the process holds into the weights and everything else.

    Its own function because it is the part that can be wrong without a server: reading
    `kv_and_run_bytes` when a model is mmapped raised at the *end* of a run, after every
    question had been answered, and threw away a quarter hour of GPU for a summary line.
    """
    # The peak where a run has one, because what decides how many conversations fit is what
    # the machine was asked for while it was answering, not what was left over afterwards.
    if out.get("resident_peak"):
        out["resident_bytes"] = max(int(out["resident_peak"]),
                                    int(out.get("resident_bytes") or 0))
    if "resident_bytes" in out and "weights_bytes" in out:
        beyond = out["resident_bytes"] - out["weights_bytes"]
        if beyond > 0:
            out["kv_and_run_bytes"] = beyond
        else:
            # Less resident than the weights on disk means llama.cpp mapped the file and
            # never paged all of it in -- an MoE's unused experts, most often. The
            # subtraction then says nothing about the cache, so no number is better than a
            # wrong one.
            out["mmapped"] = True
        # What one more conversation costs, which is the question a number like this is
        # asked for. Held tokens are the context times the slots holding one each; dividing
        # by them makes two models comparable however each happened to be configured.
        held = (out.get("context") or 0) * (out.get("slots") or 0)
        if held and "kv_and_run_bytes" in out:
            out["bytes_per_1k_context"] = int(out["kv_and_run_bytes"] / (held / 1024))
    return out

def slot_count(base_url: str) -> int:
    """How many slots that server has, read from ``/slots`` as `busy` reads it.

    -1 when it will not say. It is what decides whether concurrent conversations queue:
    more of them than slots, and a turn waits for a slot before a token is read.
    """
    from ml_stack.http import request_json

    try:
        slots = request_json(f"{base_url.rstrip('/')}/slots", timeout=5.0, method="GET")
    except Exception:  # noqa: BLE001 - a server that will not answer has an unknown count
        return -1
    return len(slots) if isinstance(slots, list) else -1

def _idle(url: str, args: Any) -> bool:
    """Refuse to time a server somebody else is using, unless told not to care."""
    working = bench.busy(url)
    if working <= 0:
        if working < 0:
            warn(f"note: {url} would not say whether it is busy; timings may not be alone")
        return True
    warn(f"error: {url} is already working on {working} request(s). A timing taken while "
         f"another run has the same GPU is not a timing.\n"
         f"       Wait for it, or pass --anyway to measure regardless.")
    return bool(getattr(args, "anyway", False))


def busy(base_url: str) -> int:
    """How many requests that server is already working on.

    A timing taken while somebody else is using the same GPU is not a timing. This is the
    cheapest way to know: llama.cpp's /slots says what each slot is doing, and anything
    above zero means the number about to be measured belongs to two callers at once.

    -1 when the server will not say, which is not the same as idle and is not treated as it.
    """
    from ml_stack.http import request_json

    try:
        slots = request_json(f"{base_url.rstrip('/')}/slots", timeout=5.0, method="GET")
    except Exception:  # noqa: BLE001 - a server that will not answer is not known to be idle
        return -1
    if not isinstance(slots, list):
        return -1
    return sum(1 for one in slots if isinstance(one, Mapping) and one.get("is_processing"))
