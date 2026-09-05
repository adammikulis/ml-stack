"""Run every claim in FEATURES.md and report whether it holds.

    python docs/verify_release.py
"""

from __future__ import annotations

import json
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

# claims must hold for this tree, not whatever an older install left on the path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

RESULTS: list[tuple[str, str, bool, str]] = []


def check(area: str, claim: str):
    def wrap(fn):
        try:
            detail = fn() or ""
            RESULTS.append((area, claim, True, str(detail)))
        except Exception as exc:                      # noqa: BLE001
            RESULTS.append((area, claim, False, f"{type(exc).__name__}: {exc}"))
        return fn
    return wrap


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


TMP = Path(tempfile.mkdtemp(prefix="ml-stack-verify-"))
WORDS = "correct horse battery staple"


# -- setup ---------------------------------------------------------------
@check("Setup", "the same passphrase gives the same cluster key")
def _():
    from ml_stack.fleet import key_from_passphrase
    a = key_from_passphrase(WORDS)
    assert a == key_from_passphrase(WORDS)
    assert a != key_from_passphrase("something else entirely")
    return f"{len(a)}-byte key"


@check("Setup", "two groups on one network cannot see each other")
def _():
    from ml_stack.fleet import Advertiser, Beacon, discover, join_cluster
    port = free_port()
    ours = join_cluster(WORDS, path=TMP / "a.key")
    theirs = join_cluster("a completely different phrase", path=TMP / "b.key")
    with Advertiser(Beacon(name="ours", port=8770), ours, port=port, interval_s=0.2), \
         Advertiser(Beacon(name="theirs", port=8771), theirs, port=port, interval_s=0.2):
        we = {b.name for b in discover(ours, timeout_s=2.0, port=port)}
        they = {b.name for b in discover(theirs, timeout_s=2.0, port=port)}
    assert we == {"ours"} and they == {"theirs"}, (we, they)
    return "each sees only its own"


@check("Setup", "a passphrase shorter than the minimum is refused")
def _():
    from ml_stack.fleet import DiscoveryError, MIN_PASSPHRASE, key_from_passphrase
    try:
        key_from_passphrase("x" * (MIN_PASSPHRASE - 1))
    except DiscoveryError as exc:
        return str(exc)[:60]
    raise AssertionError("accepted a short passphrase")


@check("Setup", "the group is remembered, so a passphrase can be checked later")
def _():
    from ml_stack.fleet import check_passphrase, cluster_group, join_cluster
    key = TMP / "grp.key"
    join_cluster(WORDS, group="garage", path=key)
    assert cluster_group(key) == "garage"
    assert check_passphrase(WORDS, path=key)
    assert not check_passphrase("wrong words here", path=key)
    return "group 'garage'"


# -- the daemon ----------------------------------------------------------
class Box:
    def __init__(self, name="box", slots=1, labels=(), extra=None, host="127.0.0.1"):
        from ml_stack.fleet import JobRunner, Peer, device_report, load_or_create_token, make_handler
        from ml_stack.fleet.settings import Settings
        from ml_stack.fleet.ui import UI
        root = TMP / name
        self.files = root / "files"
        self.files.mkdir(parents=True)
        token = load_or_create_token(root)
        self.runner = JobRunner(root, self.files, slots=slots)
        self.port = free_port()
        report = lambda: {**device_report(lambda: dict(extra or {})), "labels": list(labels)}
        self.ui = UI(name=name, cluster_key_path=root / "cluster.key")
        self.ui.runner, self.ui.settings = self.runner, Settings()
        self.ui.settings_path, self.ui.report = root / "settings.json", report
        self.httpd = ThreadingHTTPServer(
            (host, self.port),
            make_handler(self.runner, self.files, token, name, report, ui=self.ui))
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.peer = Peer(f"http://127.0.0.1:{self.port}", token)

    def close(self):
        self.runner.shutdown(); self.httpd.shutdown(); self.httpd.server_close()


@check("Daemon", "a job runs and its output can be pulled back")
def _():
    box = Box("pull")
    try:
        job = box.peer.submit([sys.executable, "-c",
            "import os,pathlib;d=pathlib.Path(os.environ['ML_STACK_OUT']);"
            "d.mkdir(parents=True,exist_ok=True);(d/'r.txt').write_text('done')"])
        box.peer.wait(job["id"], poll_s=0.1, timeout_s=30)
        got = box.peer.pull(f"jobs/{job['id']}/out/r.txt", TMP / "pulled.txt")
        assert got.read_text() == "done"
        return "argv -> job -> file back"
    finally:
        box.close()


@check("Daemon", "slots let several jobs run at once, and default to one")
def _():
    one, many = Box("one", slots=1), Box("many", slots=4)
    try:
        assert one.peer.health()["slots"] == 1
        for i in range(4):
            many.peer.submit([sys.executable, "-c", "import time;time.sleep(2)"])
        end = time.time() + 8
        while time.time() < end and many.runner.status()["free"] > 0:
            time.sleep(0.1)
        assert many.runner.status()["free"] == 0
        return f"{len(many.runner.status()['running'])} at once"
    finally:
        one.close(); many.close()


@check("Daemon", "stopping one job leaves its neighbour running")
def _():
    box = Box("stop", slots=2)
    try:
        keep = box.peer.submit([sys.executable, "-c", "import time;time.sleep(4)"])
        doomed = box.peer.submit([sys.executable, "-c", "import time;time.sleep(4)"])
        end = time.time() + 6
        while time.time() < end and len(box.runner.status()["running"]) < 2:
            time.sleep(0.1)
        box.peer.stop(doomed["id"])
        time.sleep(0.5)
        assert box.peer.job(keep["id"])["state"] == "running"
        return "neighbour survived"
    finally:
        box.close()


@check("Daemon", "an empty file uploads without error")
def _():
    box = Box("empty")
    try:
        p = TMP / "empty.bin"; p.write_bytes(b"")
        box.peer.push(p, "empty.bin")
        assert (box.files / "empty.bin").read_bytes() == b""
        return "0 bytes round-tripped"
    finally:
        box.close()


# -- placement -----------------------------------------------------------
@check("Placement", "vendors are distinguished: cuda, rocm and any-GPU differ")
def _():
    from ml_stack.fleet import Requires
    amd = {"backends": ["torch"], "vendor": "amd", "rocm": True, "cuda": False,
           "accelerator": True, "labels": []}
    why = Requires(backend="cuda").why_not("amd", amd, 1, 1)
    assert why and "ROCm" in why
    assert Requires(backend="rocm").admits("amd", amd, 1, 1)
    assert Requires(backend="accelerator").admits("amd", amd, 1, 1)
    return why[:70]


@check("Placement", "an unmeasured machine is tried, not skipped")
def _():
    from ml_stack.fleet import Candidate, choose
    fast = Candidate(peer=None, name="fast", report={}, slots=1, free=1, rate=1000.0)
    new = Candidate(peer=None, name="new", report={}, slots=1, free=1, rate=None)
    assert choose([fast, new]).name == "new"
    return "unmeasured wins while it has capacity"


@check("Placement", "work is refused with every machine's reason")
def _():
    from ml_stack.fleet import Rates, Requires, Unit, run
    gpu, pi = Box("gpu-lbl", labels=("train",)), Box("pi-lbl", labels=("prep",))
    try:
        unit = Unit(id="needs-cuda", argv=["true"], requires=Requires(backend="cuda"))
        [place] = run([unit], [gpu.peer, pi.peer], rates=Rates(TMP / "r.json"))
        assert place.state == "unplaceable", place
        assert "gpu-lbl" in place.error and "pi-lbl" in place.error
        return place.error[:70]
    finally:
        gpu.close(); pi.close()


@check("Placement", "labels keep prep work off the training machines")
def _():
    from ml_stack.fleet import Rates, Requires, Unit, run
    gpu, pi = Box("g2", labels=("train",)), Box("p2", slots=2, labels=("prep",))
    try:
        units = [Unit(id=f"u{i}", argv=[sys.executable, "-c", "pass"],
                      requires=Requires(labels=("prep",))) for i in range(4)]
        places = run(units, [gpu.peer, pi.peer], rates=Rates(TMP / "r2.json"), poll_s=0.05)
        assert all(p.ok for p in places), [p.error for p in places]
        assert {p.peer for p in places} == {"p2"}
        return "all four landed on the prep box"
    finally:
        gpu.close(); pi.close()


# -- scheduling ----------------------------------------------------------
@check("Scheduling", "working hours block new work, including past midnight")
def _():
    from datetime import datetime
    from ml_stack.fleet.availability import Availability
    day = Availability.from_specs(busy=["mon-fri 09:00-17:00"])
    assert not day.open_at(datetime.fromisoformat("2026-08-24 10:00"))
    assert day.open_at(datetime.fromisoformat("2026-08-24 18:00"))
    night = Availability.from_specs(busy=["22:00-06:00"])
    assert not night.open_at(datetime.fromisoformat("2026-08-25 03:00"))
    assert night.open_at(datetime.fromisoformat("2026-08-25 07:00"))
    return "09:00-17:00 and 22:00-06:00 both hold"


@check("Scheduling", "pausing stops running work and requeues it")
def _():
    from ml_stack.fleet.availability import Availability
    box = Box("pause")
    sched = Availability()
    box.runner.gate = lambda: sched.may_start()
    try:
        job = box.peer.submit([sys.executable, "-c", "import time;time.sleep(30)"])
        end = time.time() + 5
        while time.time() < end and not box.runner.status()["running"]:
            time.sleep(0.1)
        sched.pause(reason="gaming")
        box.runner.stop_running()
        time.sleep(0.5)
        assert box.peer.job(job["id"])["state"] == "queued"
        return "job requeued, machine free"
    finally:
        box.close()


@check("Scheduling", "a pause survives a restart")
def _():
    from ml_stack.fleet.availability import Availability
    a = Availability(); a.pause(reason="gaming"); a.save(TMP / "av.json")
    back = Availability.load(TMP / "av.json")
    assert back.paused and "gaming" in back.may_start()[1]
    return back.may_start()[1][:50]


# -- training ------------------------------------------------------------
@check("Training", "a language model trains and the loss falls")
def _():
    import math
    from ml_stack.train.run import run as train_run
    data = TMP / "corpus"; data.mkdir()
    (data / "c.jsonl").write_text("\n".join(json.dumps(
        {"text": f"The quick brown fox jumps over the lazy dog {i}."}) for i in range(300)))
    got = train_run("text-lm", {"size": "small", "steps": 120, "context": 64,
                                "batch_size": 8}, data, TMP / "lm")
    assert got["final_loss"] < math.log(256) * 0.7, got
    return f"loss {got['final_loss']:.2f} vs {math.log(256):.2f} untrained"


@check("Training", "a classifier generalises to rows it never saw")
def _():
    import math
    from ml_stack.train.run import run as train_run
    data = TMP / "reviews"; data.mkdir()
    rows = [{"text": f"This was {'great' if i % 2 else 'awful'}, truly.",
             "label": "good" if i % 2 else "bad"} for i in range(300)]
    (data / "r.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    got = train_run("classify-text", {"size": "small", "steps": 200, "context": 48},
                    data, TMP / "cls")
    assert got["best_metric"] < math.log(2), got
    return f"held-out {got['best_metric']:.4f} vs {math.log(2):.2f} for guessing"


@check("Training", "a dry run leaves no checkpoint behind")
def _():
    from ml_stack.train.run import run as train_run
    out = TMP / "dry"
    got = train_run("classify-text", {"size": "small"}, TMP / "reviews", out, dry=True)
    assert got["dry_run"] and not [p for p in out.iterdir() if p.is_dir()]
    return f"{got['steps']} steps, nothing written"


@check("Training", "resuming continues rather than restarting")
def _():
    from ml_stack.train.run import run as train_run
    train_run("text-lm", {"size": "small", "steps": 60, "context": 64},
              TMP / "corpus", TMP / "resume")
    again = train_run("text-lm", {"size": "small", "steps": 120, "context": 64},
                      TMP / "corpus", TMP / "resume")
    assert again["steps"] == 120
    return "60 -> 120, not 60 -> 180"


@check("Training", "an undeclared setting is refused, not ignored")
def _():
    from ml_stack.train.recipes import validate
    try:
        validate("text-lm", {"lr_schedule": "cosine"})
    except ValueError as exc:
        return str(exc)[:70]
    raise AssertionError("a silently dropped setting")


# -- the interface -------------------------------------------------------
@check("Interface", "the web assets are served")
def _():
    box = Box("ui")
    try:
        for path, kind in (("/ui/", "text/html"), ("/ui/static/app.js", "text/javascript"),
                           ("/ui/static/style.css", "text/css")):
            req = urllib.request.Request(f"http://127.0.0.1:{box.port}{path}")
            with urllib.request.urlopen(req, timeout=5) as r:
                assert r.status == 200 and r.headers["Content-Type"] == kind, path
        return "index, script and stylesheet"
    finally:
        box.close()


@check("Interface", "setup is refused from another machine")
def _():
    from ml_stack.fleet.discovery import primary_ip
    box = Box("guard", host="0.0.0.0")
    try:
        req = urllib.request.Request(
            f"http://{primary_ip()}:{box.port}/ui/setup/join",
            data=json.dumps({"passphrase": WORDS}).encode(), method="POST")
        req.add_header("X-ML-Stack-UI", "1")
        req.add_header("Content-Type", "application/json")
        try:
            urllib.request.urlopen(req, timeout=5)
        except urllib.error.HTTPError as e:
            assert e.code == 403, e.code
            return json.loads(e.read())["error"][:70]
        raise AssertionError("a stranger could claim this machine")
    finally:
        box.close()
    return ""


@check("Interface", "the passphrase signs you in, and one typo does not lock you out")
def _():
    box = Box("login")
    try:
        def post(body):
            req = urllib.request.Request(
                f"http://127.0.0.1:{box.port}/ui/session",
                data=json.dumps(body).encode(), method="POST")
            req.add_header("X-ML-Stack-UI", "1")
            req.add_header("Content-Type", "application/json")
            try:
                with urllib.request.urlopen(req, timeout=10) as r:
                    return r.status
            except urllib.error.HTTPError as e:
                return e.code
        urllib.request.urlopen(urllib.request.Request(
            f"http://127.0.0.1:{box.port}/ui/setup/join",
            data=json.dumps({"passphrase": WORDS}).encode(), method="POST",
            headers={"X-ML-Stack-UI": "1", "Content-Type": "application/json"}), timeout=20)
        assert post({"passphrase": "wrong words here"}) == 401
        assert post({"passphrase": WORDS}) == 200
        return "typo then correct -> signed in"
    finally:
        box.close()


@check("Interface", "settings are suggested from the machine's own hardware")
def _():
    from ml_stack.fleet.settings import suggest
    gpu = suggest({"accelerator": True, "gpu": "RTX 4090", "cpus": 16})
    cpu = suggest({"accelerator": False, "cpus": 12})
    assert gpu["labels"].value == ["train"] and "RTX 4090" in gpu["labels"].why
    assert cpu["labels"].value == ["prep"]
    return "GPU -> train, CPU -> prepare data"


@check("Interface", "closing asks once, then remembers")
def _():
    from ml_stack.fleet.app import Bridge
    from ml_stack.fleet.settings import Settings
    path = TMP / "close.json"

    class W:
        hidden = destroyed = False
        def hide(self): self.hidden = True
        def destroy(self): self.destroyed = True

    b = Bridge(path); b.window = W()
    b.close_choice("background", remember=False)
    assert b.window.hidden and Settings.load(path).on_close == ""
    b.window = W(); b.close_choice("background", remember=True)
    assert Settings.load(path).on_close == "background"
    return "unticked -> ask again; ticked -> remembered"


# -- telemetry -----------------------------------------------------------
@check("Telemetry", "this machine reports its own temperature and clocks")
def _():
    from ml_stack.train.accelerator import report
    got = report()
    have = [k for k in ("temp_c", "clock_mhz", "power_w", "gpu_util_pct") if k in got]
    if not have:
        return "no probe available on this machine (optional)"
    return ", ".join(f"{k}={got[k]}" for k in have)


@check("Telemetry", "a machine says how much memory is in use and how busy it is")
def _():
    from ml_stack.fleet.daemon import stdlib_device_report

    got = stdlib_device_report()
    assert got.get("ram_gb"), "no memory total"
    assert "ram_used_gb" in got, "the card cannot say how much memory is in use"
    assert 0 <= got["ram_used_gb"] <= got["ram_gb"], got
    assert "cpu_pct" in got, "the card cannot say how busy the processors are"
    assert 0 <= got["cpu_pct"] <= 100, got
    return (f"{got['ram_used_gb']} of {got['ram_gb']} GB in use, "
            f"{got['cpu_pct']}% busy")


@check("Telemetry", "a machine with no framework still reports what it is")
def _():
    from ml_stack.fleet.daemon import stdlib_device_report
    got = stdlib_device_report()
    assert got["cpus"] >= 1 and got["arch"]
    assert "cuda" not in got and "gpu" not in got
    return f"{got['cpus']} cpus, {got['arch']}, no guess about a GPU"


# -- packaging -----------------------------------------------------------
@check("Packaging", "the interface ships inside the wheel")
def _():
    import zipfile
    out = TMP / "wheels"
    done = subprocess.run(
        [sys.executable, "-m", "build", "--wheel", "--outdir", str(out),
         str(Path(__file__).resolve().parent.parent)],
        capture_output=True, text=True)
    assert done.returncode == 0, done.stderr[-200:]
    names = zipfile.ZipFile(sorted(out.glob("*.whl"))[-1]).namelist()
    for asset in ("index.html", "style.css", "app.js"):
        assert f"ml_stack/fleet/web/{asset}" in names, asset
    return "index.html, style.css, app.js"


@check("Packaging", "installing it brings in nothing")
def _():
    import tomllib

    root = Path(__file__).resolve().parent.parent
    meta = tomllib.load((root / "pyproject.toml").open("rb"))["project"]
    assert meta["dependencies"] == [], meta["dependencies"]
    extras = sorted(meta["optional-dependencies"])
    assert {"app", "train", "serve", "all"} <= set(extras)
    return "no dependencies; " + ", ".join(extras)


# -- models --------------------------------------------------------------
@check("Models", "a model is copied from another machine rather than downloaded again")
def _():
    import os
    import threading
    from http.server import ThreadingHTTPServer

    from ml_stack.fleet import JobRunner, Models, join_cluster, make_handler
    from ml_stack.fleet.daemon import load_or_create_token

    key = join_cluster(WORDS, path=TMP / "models.key")
    where = TMP / "haver"
    where.mkdir(parents=True, exist_ok=True)
    payload = os.urandom(3 * 1024 * 1024)
    (where / "tiny-test.gguf").write_bytes(payload)

    root = TMP / "haver-daemon"
    files = root / "files"
    files.mkdir(parents=True, exist_ok=True)
    runner = JobRunner(root, files)
    port = free_port()
    httpd = ThreadingHTTPServer(("127.0.0.1", port), make_handler(
        runner, files, load_or_create_token(root, key), "haver",
        models=Models([where], where), cluster_key_path=TMP / "models.key"))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        want = TMP / "wanter"
        want.mkdir(parents=True, exist_ok=True)
        # Several daemons on one box share the discovery port, so the address that
        # discovery would supply is given directly.
        got = Models([want], want)._from_peer(
            "tiny-test.gguf", f"http://127.0.0.1:{port}", key, None)
        assert got.path.read_bytes() == payload, "the copy does not match"
        assert got.path.parent == want
    finally:
        runner.shutdown()
        httpd.shutdown()
        httpd.server_close()
    return f"{len(payload) // 1024} KB over the network"


@check("Models", "a part-file left by a different download is discarded, not resumed")
def _():
    import http.server
    import json as js
    import threading

    from ml_stack.fleet import Models

    store = TMP / "parts"
    store.mkdir(parents=True, exist_ok=True)
    part = store / "m.gguf.part"
    part.write_bytes(b"\x00" * 8192)
    (store.parent / "parts" / "m.gguf.part.from").write_text(
        js.dumps({"url": "http://elsewhere.invalid/other.gguf", "validator": "x"}))
    payload = b"R" * 4096
    asked = []

    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            asked.append(self.headers.get("Range"))
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", free_port()), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        got = Models([store], store).ensure(
            "m.gguf", source=f"http://127.0.0.1:{srv.server_address[1]}/m.gguf")
    finally:
        srv.shutdown()
    assert asked == [None], f"asked to resume another file's part: {asked}"
    assert got.path.read_bytes() == payload
    return "the stale part was thrown away"


@check("Models", "a download that stopped is listed, and can be discarded")
def _():
    import os
    import time as clock

    from ml_stack.fleet import Models

    store = TMP / "stopped"
    store.mkdir(parents=True, exist_ok=True)
    part = store / "half.gguf.part"
    part.write_bytes(b"x" * 4096)
    stamp = Path(str(part) + ".from")
    stamp.write_text('{"url": "http://x/half.gguf", "validator": "t"}')

    models = Models([store], store)
    assert models.unfinished() == [], "an active download must not be offered"
    old = clock.time() - 7200
    os.utime(part, (old, old))

    listed = models.unfinished()
    assert [r["name"] for r in listed] == ["half.gguf.part"], listed
    assert listed[0]["size"] == 4096
    assert models.discard("half.gguf.part") == ["half.gguf.part"]
    assert not part.exists() and not stamp.exists()
    return "4096 bytes offered, then discarded with what it recorded"


@check("Models", "getting a model reports how far along it is")
def _():
    import http.server
    import os
    import threading
    import time as clock

    from ml_stack.fleet.models import CHUNK, Downloads, Models

    payload = os.urandom(3 * CHUNK)

    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            for i in range(0, len(payload), 65536):
                self.wfile.write(payload[i:i + 65536])
                self.wfile.flush()
                clock.sleep(0.005)

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", free_port()), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()

    store = TMP / "progress"
    store.mkdir(parents=True, exist_ok=True)
    downloads = Downloads(Models([store], store))
    try:
        began = clock.monotonic()
        row = downloads.start(
            "big.gguf", source=f"http://127.0.0.1:{srv.server_address[1]}/big.gguf")
        answered = clock.monotonic() - began
        assert answered < 1.0, f"starting it waited {answered:.1f}s"

        partial = False
        for _ in range(400):
            now = next(g for g in downloads.active() if g.id == row.id)
            if 0 < now.done < now.total:
                partial = True
            if now.state != "getting":
                break
            clock.sleep(0.02)
    finally:
        srv.shutdown()

    assert now.state == "done", now.error
    assert partial, "it only ever reported nothing, then everything"
    assert (store / "big.gguf").read_bytes() == payload
    return f"{len(payload) // 1024 // 1024} MB, counted as it arrived"


@check("Models", "the model list is what Hugging Face says is popular now")
def _():
    """A list written into the code is out of date the day it ships."""
    from ml_stack.fleet.models import SUGGESTED, popular

    got = popular(free_gb=1024.0, ram_gb=1024.0, limit=12)
    assert len(got) >= 5, f"only {len(got)} came back"
    shipped = {s.name for s in SUGGESTED}
    assert not {p.name for p in got} <= shipped, (
        "the list matches the one written into the code, so it is not live")
    for pick in got:
        assert pick.gb > 0, f"{pick.name} has no size"
        assert pick.ref.startswith("hf:") and pick.file.endswith(".gguf")
        assert pick.family, f"{pick.name} has no family"
        assert "mmproj" not in pick.file.lower(), f"{pick.name} is a vision projector"
        assert "mtp" not in pick.file.lower(), f"{pick.name} is a draft head"
    families = sorted({p.family for p in got})
    return f"{len(got)} models across {len(families)}: {', '.join(families)}"


@check("Models", "the list it falls back to when offline still downloads")
def _():
    """A reference that stopped resolving is a button that does nothing."""
    import urllib.error
    import urllib.request

    from ml_stack.fleet.models import SUGGESTED, _resolve

    checked = []
    for pick in SUGGESTED:
        req = urllib.request.Request(_resolve(pick.ref), method="HEAD",
                                     headers={"User-Agent": "ml-stack"})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                size = int(r.headers.get("Content-Length") or 0)
        except (urllib.error.HTTPError, urllib.error.URLError, OSError) as exc:
            raise AssertionError(f"{pick.name}: {pick.ref} -> {exc}") from None
        assert size > 0, f"{pick.name}: {pick.ref} is empty"
        claimed = pick.gb * 2**30
        assert abs(size - claimed) < max(claimed * 0.25, 64 * 2**20), (
            f"{pick.name} is {size / 2**30:.2f} GB, listed as {pick.gb} GB")
        checked.append(pick.name)
    return f"{len(checked)} live: {', '.join(checked)}"


# -- chat ----------------------------------------------------------------
@check("Chat", "a machine that can run nothing itself still talks to a peer's model")
def _():
    """The install with no model server is the common one. If this fails, the
    product does not work for the people it is for."""
    import json as js
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    from ml_stack.fleet import JobRunner, Serving, join_cluster, make_handler
    from ml_stack.fleet.chat import find, reply_text, stream, targets
    from ml_stack.fleet.daemon import load_or_create_token
    from ml_stack.fleet.discovery import derive_token

    class Model(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"{}")

        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", "0")))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for piece in ("It ", "works"):
                self.wfile.write(
                    f"data: {js.dumps({'choices': [{'delta': {'content': piece}}]})}\n\n"
                    .encode())
                self.wfile.flush()
            self.wfile.write(b"data: [DONE]\n\n")

    model = ThreadingHTTPServer(("127.0.0.1", free_port()), Model)
    threading.Thread(target=model.serve_forever, daemon=True).start()

    key = join_cluster(WORDS, path=TMP / "chat.key")
    root = TMP / "host-daemon"
    files = root / "files"
    files.mkdir(parents=True, exist_ok=True)
    serving = Serving(root / "serving.json")
    serving.register(model.server_address[1], ["tiny-test.gguf"])
    runner = JobRunner(root, files)
    port = free_port()
    httpd = ThreadingHTTPServer(("127.0.0.1", port), make_handler(
        runner, files, load_or_create_token(root, key), "host", serving=serving))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()

    try:
        # This machine: no model store, nothing serving, no way to run one.
        beacon = {"name": "host", "base_url": f"http://127.0.0.1:{port}",
                  "is_self": False, "device": {"serving": serving.public()}}
        reachable = targets([beacon], serving=None, token=derive_token(key))
        assert reachable, "a peer is serving and this machine cannot see it"
        target = find(reachable, "tiny-test.gguf")
        assert target is not None and not target.local
        said = b"".join(stream(target, {"model": target.model, "stream": True,
                                        "messages": [{"role": "user", "content": "hi"}]}))
    finally:
        runner.shutdown()
        httpd.shutdown()
        httpd.server_close()
        model.shutdown()

    assert reply_text(said) == "It works", reply_text(said)
    return "no model store, nothing serving, still answered"


@check("Chat", "a conversation is still there after a restart")
def _():
    from ml_stack.fleet import Conversations

    where = TMP / "chats"
    first = Conversations(where)
    made = first.start(model="tiny-test.gguf")
    first.append(made.id, "user", "how tall is everest")
    first.append(made.id, "assistant", "8849 metres")

    again = Conversations(where).get(made.id)
    assert [m.content for m in again.messages] == ["how tall is everest",
                                                   "8849 metres"]
    assert again.title == "how tall is everest"
    assert [c.id for c in Conversations(where).search("everest")] == [made.id]
    return "kept and found again by what was said in it"


@check("Updating", "a machine that is working is not updated out from under a job")
def _():
    import threading as th
    import time as clock

    from ml_stack.fleet import updates

    tried = th.Event()
    real = updates.apply_if_newer
    updates.apply_if_newer = lambda: (tried.set(), {"installed": False})[1]
    try:
        updates.watch(wanted=lambda: True, idle=lambda: False,
                      every_s=0.02, first_after_s=0.0)
        clock.sleep(0.4)
        assert not tried.is_set(), "it updated while a job was running"

        ran = th.Event()
        updates.apply_if_newer = lambda: (ran.set(), {"installed": False})[1]
        updates.watch(wanted=lambda: True, idle=lambda: True,
                      every_s=0.02, first_after_s=0.0)
        assert ran.wait(3.0), "it never checked on an idle machine"
    finally:
        updates.apply_if_newer = real
    return "left alone while busy, checked when idle"


# -- removing it ---------------------------------------------------------
@check("Removing", "an uninstall leaves your models and your own files alone")
def _():
    from ml_stack.fleet.uninstall import plan, remove

    home = TMP / "leaving" / ".ml-stack"
    root = home / "traind"
    for name in ("chats", "files", "models", "env"):
        (root / name).mkdir(parents=True, exist_ok=True)
    key = home / "cluster.key"
    key.write_text("k" * 44)
    (root / "models" / "big.gguf").write_bytes(b"m" * 8192)
    (root / "files" / "mine.jsonl").write_bytes(b"d" * 256)
    (root / "chats" / "a.json").write_text('{"id": "a"}')

    offered = {i.key: i for i in plan(root, key_path=key)}
    assert offered["models"].default is False, "models were ticked for removal"
    assert offered["datasets"].default is False, "your files were ticked for removal"

    remove(root, [k for k, i in offered.items() if i.default], key_path=key)
    assert (root / "models" / "big.gguf").exists(), "it took the models"
    assert (root / "files" / "mine.jsonl").exists(), "it took the datasets"
    assert not (root / "chats").exists()
    assert not key.exists()
    return "chats and the key gone, the model and the dataset kept"


# -- report --------------------------------------------------------------
# -- graphs --------------------------------------------------------------
@check("Graphs", "a store keeps everything a graph carries, across a reopen")
def _():
    import pytest
    pytest.importorskip("ladybug")
    from ml_stack.graph import GraphStore
    path = TMP / "graph" / "g"
    graph = {"nodes": [{"id": "p:a", "kind": "person", "label": "Ada", "mentions": 2,
                        "attrs": {"role": "analyst"}, "messages": ["m1"]},
                       {"id": "t:c", "kind": "topic", "label": "compilers", "mentions": 1,
                        "attrs": {}, "messages": ["m1"]}],
             "edges": [{"source": "p:a", "target": "t:c", "rel": "interested_in", "weight": 2,
                        "messages": ["m1"]}],
             "stats": {"nodes": 2}}
    with GraphStore(path) as store:
        store.write(graph)
    with GraphStore(path) as reopened:
        back = reopened.read()
    assert back["nodes"][0]["messages"] == ["m1"], "a node lost what it carried"
    assert back["stats"] == {"nodes": 2, "edges": 1}, \
        f"the stats document does not count what the store holds: {back['stats']}"
    assert back["edges"][0]["messages"] == ["m1"]
    return "nodes, edges and documents round-trip"


@check("Graphs", "a write that would take most of a store is refused")
def _():
    import pytest
    pytest.importorskip("ladybug")
    from ml_stack.graph import GraphStore, WouldLoseTooMuch, count_store, replace
    path = TMP / "guard" / "g"
    with GraphStore(path) as store:
        store.write({"nodes": [{"id": f"n{i}", "kind": "t", "label": str(i), "mentions": 1,
                                "attrs": {}} for i in range(10)], "edges": []})
    try:
        replace(path, {"nodes": [], "edges": []})
    except WouldLoseTooMuch:
        pass
    else:
        raise AssertionError("it emptied the store")
    assert count_store(path)["nodes"] == 10
    return "10 of 10 refused; the store is intact"


@check("Graphs", "a snapshot is verified by reopening it, and a restore is undoable")
def _():
    import pytest
    pytest.importorskip("ladybug")
    from ml_stack.graph import GraphStore, count_store, roll_back, snapshot
    from ml_stack.graph.snapshots import snapshots
    path = TMP / "snap" / "g"
    with GraphStore(path) as store:
        store.write({"nodes": [{"id": f"n{i}", "kind": "t", "label": str(i), "mentions": 1,
                                "attrs": {}} for i in range(6)], "edges": []})
    kept = snapshot(path, reason="verifying the release")
    with GraphStore(path) as store:
        store.drop([f"n{i}" for i in range(6)], force=True)
    assert count_store(path)["nodes"] == 0
    roll_back(kept.path)
    assert count_store(path)["nodes"] == 6
    assert any("before restoring" in r.reason for r in snapshots(path))
    return f"{kept.method}, restored 6 nodes"


@check("Graphs", "finding things fuses characters, words and meaning")
def _():
    from ml_stack.graph.search import hybrid, lexical, rrf
    graph = {"nodes": [{"id": "t:r", "kind": "topic", "label": "robotics", "mentions": 2,
                        "attrs": {}, "messages": []},
                       {"id": "p:b", "kind": "person", "label": "Bea", "mentions": 1,
                        "attrs": {}, "messages": []}],
             "edges": [], "messages": {}}

    class Store:
        def search(self, text, limit=10):
            return [{"id": "t:r"}]

        def similar(self, vector, model="", limit=10):
            return [{"id": "p:b"}]

    assert rrf(["a", "b"], ["c", "b"], limit=1) == ["b"], "fusion did not prefer the agreed one"
    assert lexical(graph, "robotics") == ["t:r"]
    got = [h["id"] for h in hybrid(graph, "robotics", store=Store(), vector=[0.1])]
    assert set(got) == {"t:r", "p:b"}
    return "all three vote"


@check("Graphs", "the model reads a graph with tools, and invented ids are refused")
def _():
    from dataclasses import dataclass
    from ml_stack.graph.ask import converse
    graph = {"nodes": [{"id": "p:a", "kind": "person", "label": "Ada", "mentions": 1,
                        "attrs": {}, "messages": []}], "edges": [], "messages": {}}

    @dataclass
    class Reply:
        content: str = ""
        tool_calls: list | None = None

    class Model:
        def __init__(self):
            self.turn = 0

        def chat(self, messages, tools=None, **_):
            self.turn += 1
            if tools and self.turn == 1:
                return Reply(tool_calls=[{"id": "1", "function": {
                    "name": "look_at", "arguments": '{"ids": ["p:a", "p:ghost"]}'}}])
            return Reply(content="Ada is here.")

    out = converse("who?", graph, Model())
    assert out.ids == ["p:a"], f"an invented id got through: {out.ids}"
    return "one real id kept, one invented id refused"


# -- reading documents into a graph ---------------------------------------
@check("Documents", "a gold set of passages turns the reading into a number")
def _():
    import pytest
    pytest.importorskip("ladybug")
    from ml_stack.ingest import gold
    root = Path(__file__).resolve().parent.parent
    passages = gold.read_gold(root / "tests" / "fixtures" / "extraction-gold.json")
    shape = json.loads((root / "contracts" / "extraction-document.schema.json").read_text())

    class SaysNothing:
        def extract(self, text, schema, **_):
            return {"concepts": [], "relations": []}

    got = gold.gold_score(SaysNothing(), passages, shape)
    assert len(passages) >= 20 and got.wanted >= 100, (len(passages), got.wanted)
    assert got.matched == 0 and len(got.misses) == got.wanted, "the scorer missed nothing to miss"
    return (f"{len(passages)} passages, {got.wanted} triples: an extractor that says "
            f"nothing misses every one, each named")


# -- measuring models ------------------------------------------------------
@check("Measuring", "a model's measured shape is on file, not remembered")
def _():
    from ml_stack.serve import profile_for
    found = profile_for("Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf")
    assert found is not None, "no shipped profile for the flagship"
    assert found.right > 0 and found.questions > 0, "the record does not say what measured it"
    return (f"one seat at {found.seat_context}, F1 {found.right:.0%} over "
            f"{found.questions} questions, on {found.host or 'a measured host'}")


@check("Measuring", "an evening of measurement is a file checked before the first model loads")
def _():
    from ml_stack.graph.bench import queue
    steps = queue.read(Path(__file__).resolve().parent / "examples" / "flash-next-restart.queue")
    assert steps and all(s.argv for s in steps), "the shipped queue parsed to nothing"
    try:
        queue.parse("sweep --serve x.gguf --sampel 3\n")
    except queue.QueueError:
        pass
    else:
        raise AssertionError("a misspelled flag parsed as a step")
    return f"{len(steps)} steps parse; a misspelled flag is refused before anything loads"


# -- seating a fleet -------------------------------------------------------
@check("Fleet", "a plan seats the wanted conversations and names what fits nowhere")
def _():
    from ml_stack.fleet.plan import Room, place
    from ml_stack.serve.fit import Fit
    from ml_stack.serve.profile import Profile
    gb, kb = 2**30, 2**10
    big = Profile(model="big-120b-UD-Q4_K_XL.gguf", right=0.85, questions=100,
                  seconds_per_question=26.0)
    small = Profile(model="small-2b-UD-Q4_K_XL.gguf", right=0.40, questions=100,
                    seconds_per_question=3.0)
    fits = [Fit(model="big-120b-UD-Q4_K_XL.gguf", weights=70 * gb, per_token=150 * kb,
                per_seq=200 * kb, compute=gb),
            Fit(model="small-2b-UD-Q4_K_XL.gguf", weights=2 * gb, per_token=12 * kb,
                per_seq=100 * kb, compute=gb // 4)]
    peers = [Room(name="studio", room=110 * gb), Room(name="larch", room=20 * gb)]
    got = place(30, 16384, peers, [big, small], fits)
    assert got.seated == 30 and got.unplaced == 0, got.as_dict()
    assert got.rows[0].model == big.model, "the better model did not reach the roomiest peer"
    tight = place(400, 16384, [Room(name="pi", room=4 * gb)], [big, small], fits)
    assert tight.unplaced > 0 and any(p == "pi" and m == big.model for p, m, _ in tight.why), \
        "the model that fits nowhere was not named"
    return "30 seated over two peers; on a 4 GB peer the big model is refused by name"


# -- agents and the web ----------------------------------------------------
@check("Agents", "the same functions are MCP tools an agent can call")
def _():
    from ml_stack.mcp import TOOLS
    names = {t.name for t in TOOLS}
    need = {"serve_up", "serve_down", "serve_status", "models_find", "models_files",
            "models_fetch", "bench_run", "bench_status", "fleet_peers", "world_make",
            "setup_look", "doctor"}
    missing = need - names
    assert not missing, f"missing MCP tools: {sorted(missing)}"
    return f"{len(names)} tools: serve, models, bench, fleet, world, setup, doctor"


@check("Web", "the web tools refuse the machine they run on")
def _():
    from ml_stack.web import Refused, check
    for url in ("file:///etc/passwd", "http://localhost/x", "http://127.0.0.1:8080/v1/chat",
                "http://10.1.2.3/x", "http://192.168.2.44:8770/", "http://[::1]:8080/"):
        try:
            check(url)
        except Refused:
            continue
        raise AssertionError(f"{url} was not refused")
    return "file:, loopback and private hosts refused before a byte is fetched"


# -- reading a site ------------------------------------------------------
@check("Reading a site", "a virtualised list is read all the way, not one screenful")
def _():
    from ml_stack.scrape import SLACK, preset, read_all
    rows = [{"key": f"17879371{i:02d}.000000", "author": "x", "text": f"row {i}"}
            for i in range(9)]

    class Virtual:
        def __init__(self):
            self.top = 6

        def evaluate(self, js, arg=None):
            if "scrollTop" in js:
                was, self.top = self.top, max(0, self.top - 3)
                return self.top != was
            return [dict(r) for r in rows[self.top:self.top + 3]]

    seen = read_all(Virtual(), SLACK)
    assert len(seen) == 9, f"only {len(seen)} of 9 rows were read"
    assert preset("discord").key_pattern == ""
    return "9 of 9 rows, three at a time"



def main() -> int:
    width = max(len(c) for _, c, _, _ in RESULTS) + 2
    area = ""
    for a, claim, ok, detail in RESULTS:
        if a != area:
            print(f"\n{a}")
            area = a
        mark = "PASS" if ok else "FAIL"
        print(f"  [{mark}] {claim:<{width}} {detail}")
    failed = [r for r in RESULTS if not r[2]]
    print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} verified")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
