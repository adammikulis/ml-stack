"""The wiring limit: the plan for a model, the one privileged change, and who may ask for it.

The privileged call is a recording stand-in: no test runs `sysctl -w` or a macOS password
dialog. Models are real GGUF headers over a sparse file, so the estimator reads real bytes.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
from conftest import write_gguf

from ml_stack.sentinel.human import HumanRequired
from ml_stack.serve import wired

GIB = 1024**3
SRC = str(Path(__file__).resolve().parent.parent / "src")


def dense(layers=32, train=262144) -> dict:
    return {"general.architecture": "llama", "llama.block_count": layers,
            "llama.context_length": train, "llama.embedding_length": 4096,
            "llama.attention.head_count": 32, "llama.attention.head_count_kv": 8,
            "llama.attention.key_length": 128, "llama.attention.value_length": 128}


def model(tmp_path: Path, size: int, name="m-Q4_K_M.gguf") -> Path:
    path = write_gguf(tmp_path / name, dense())
    with path.open("ab") as f:
        f.truncate(size)
    return path


class Recorder:
    """A stand-in for the privileged call that remembers what it was asked and moves a fake
    limit the way sysctl would."""

    def __init__(self, start=0, code=0, err=""):
        self.calls: list[list[str]] = []
        self.limit = start
        self.code, self.err = code, err

    def __call__(self, argv, capture):
        self.calls.append(list(argv))
        if self.code == 0:
            self.limit = int(argv[-1].split("=")[-1].split('"')[0])
        return self.code, "", self.err

    def read(self):
        return self.limit


@pytest.fixture(autouse=True)
def state_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("ML_STACK_HOME", str(tmp_path / "home"))
    for name in wired.AGENT_MARKERS:
        monkeypatch.delenv(name, raising=False)


# -- the plan --------------------------------------------------------------------------------
def test_the_plan_counts_weights_cache_and_a_margin_and_proposes_a_whole_number(tmp_path):
    path = model(tmp_path, 20 * GIB)
    plan = wired.plan(str(path), context=65536, total=32 * GIB, current=0, others=0)
    sizes = dict(plan.counted)
    assert sizes["Weights"] == 20 * GIB
    assert sizes["KV cache"] > 0
    assert plan.need_bytes >= 20 * GIB + sizes["KV cache"]
    assert plan.needed_mb * 1024**2 >= plan.need_bytes + plan.margin_bytes
    assert plan.needed_mb % 256 == 0
    assert plan.fits and plan.proposed_mb == plan.needed_mb


def test_a_longer_context_and_a_wider_cache_need_more(tmp_path):
    path = str(model(tmp_path, 8 * GIB))
    short = wired.plan(path, context=8192, total=64 * GIB, current=0, others=0)
    long = wired.plan(path, context=262144, total=64 * GIB, current=0, others=0)
    f16 = wired.plan(path, context=262144, kv="f16", total=64 * GIB, current=0, others=0)
    assert short.need_bytes < long.need_bytes < f16.need_bytes


def test_the_rest_of_the_machine_keeps_8_gib_and_a_plan_past_it_does_not_fit(tmp_path):
    path = str(model(tmp_path, 28 * GIB))
    plan = wired.plan(path, context=262144, total=32 * GIB, current=0, others=0)
    assert plan.max_mb == (32 - 8) * 1024
    assert not plan.fits and plan.proposed_mb == 0
    assert plan.largest_context < 262144
    fitting = [r for r in plan.table if r.fits]
    assert all(r.left_bytes >= 8 * GIB for r in fitting)
    assert any(not r.fits for r in plan.table)


def test_under_12_gib_left_is_allowed_with_a_swap_warning(tmp_path):
    path = str(model(tmp_path, 19 * GIB))
    plan = wired.plan(path, context=8192, total=32 * GIB, current=0, others=0)
    assert plan.fits and 8 * GIB <= plan.left_bytes < 12 * GIB
    assert "swap" in plan.warn
    roomy = wired.plan(str(model(tmp_path, 8 * GIB, "b.gguf")), context=8192,
                       total=32 * GIB, current=0, others=0)
    assert roomy.warn == ""


def test_a_limit_already_high_enough_proposes_nothing_to_change(tmp_path):
    path = str(model(tmp_path, 4 * GIB))
    plan = wired.plan(path, context=8192, total=32 * GIB, current=24 * 1024, others=0)
    assert plan.enough_now


def test_the_growing_parts_are_host_ram_with_their_caps_and_the_uncapped_one_says_so(tmp_path):
    path = str(model(tmp_path, 4 * GIB))
    plan = wired.plan(path, total=64 * GIB, current=0, others=0, cache_ram_mb=4096)
    by = {g.label.split(" (")[0]: g for g in plan.grows}
    assert by["prompt cache"].bytes == 4096 * 1024**2 and "4096" in by["prompt cache"].cap
    assert all(g.where == "host RAM" for g in plan.grows)
    nocap = next(g for g in plan.grows if "dynamic" in g.label)
    assert nocap.bytes == 0 and "no cap" in nocap.cap
    assert plan.need_bytes == sum(v for _, v in plan.counted)


def test_the_mtp_head_adds_only_its_own_layer_when_it_lives_in_the_weights(tmp_path, monkeypatch):
    from ml_stack.serve import mtp

    path = str(model(tmp_path, 4 * GIB))
    monkeypatch.setattr(mtp, "embeds_head", lambda _p: True)
    on = wired.plan(path, mtp=True, total=64 * GIB, current=0, others=0)
    off = wired.plan(path, mtp=False, total=64 * GIB, current=0, others=0)
    extra = dict(on.counted)
    assert "MTP head weights" not in extra
    assert 0 < extra["MTP head KV cache"] < dict(on.counted)["KV cache"]
    assert on.need_bytes > off.need_bytes
    assert any("shares the main model's weights" in n for n in on.notes)


@pytest.mark.parametrize("bad", [{"kv": "q5_0"}, {"context": 0}, {"context": True}])
def test_an_unknown_cache_type_or_context_is_refused(tmp_path, bad):
    with pytest.raises(ValueError):
        wired.plan(str(model(tmp_path, GIB)), total=64 * GIB, current=0, others=0, **bad)


def test_a_model_that_is_not_installed_is_an_error_not_a_command(tmp_path):
    with pytest.raises(FileNotFoundError):
        wired.plan("x; rm -rf ~ $(id)", total=64 * GIB, current=0, others=0)


# -- the change ------------------------------------------------------------------------------
def test_apply_runs_exactly_one_argv_built_from_the_integer_and_records_the_original(tmp_path):
    ran = Recorder(start=98304)
    done = wired.apply(116736, via="osascript", total=128 * GIB, runner=ran, system="Darwin",
                       read=ran.read)
    assert done.ok and (done.before_mb, done.after_mb) == (98304, 116736)
    assert ran.calls == [["osascript", "-e",
                          'do shell script "/usr/sbin/sysctl -w iogpu.wired_limit_mb=116736" '
                          'with administrator privileges']]
    wired.apply(120000, via="osascript", total=128 * GIB, runner=ran, system="Darwin",
                read=ran.read)
    assert wired.original_mb() == 98304


def test_sudo_is_an_argv_list_so_sudo_asks_for_the_password(tmp_path):
    ran = Recorder()
    wired.apply(65536, via="sudo", total=128 * GIB, runner=ran, system="Darwin",
                read=ran.read, terminal=(True, True))
    assert ran.calls == [["sudo", "/usr/sbin/sysctl", "-w", "iogpu.wired_limit_mb=65536"]]


def test_reset_puts_back_the_recorded_original_or_the_default_share(tmp_path):
    ran = Recorder(start=98304)
    wired.apply(116736, via="osascript", total=128 * GIB, runner=ran, system="Darwin",
                read=ran.read)
    wired.reset(via="osascript", total=128 * GIB, runner=ran, system="Darwin", read=ran.read)
    assert ran.limit == 98304
    fresh = Recorder(start=114688)
    os.remove(wired._record())
    wired.reset(via="osascript", total=128 * GIB, runner=fresh, system="Darwin", read=fresh.read)
    assert fresh.limit == 0


@pytest.mark.parametrize("bad", ["65536; reboot", "65536", -1, 0, 1.5, True, None, 10**9,
                                 130 * 1024, [65536], {"mb": 1}])
def test_a_value_that_is_not_an_integer_in_range_never_reaches_the_runner(bad):
    ran = Recorder()
    with pytest.raises(ValueError):
        wired.apply(bad, via="osascript", total=128 * GIB, runner=ran, system="Darwin",
                    read=ran.read)
    assert ran.calls == []


def test_the_highest_limit_leaves_8_gib():
    ran = Recorder()
    wired.apply(120 * 1024, via="osascript", total=128 * GIB, runner=ran, system="Darwin",
                read=ran.read)
    with pytest.raises(ValueError):
        wired.apply(120 * 1024 + 1, via="osascript", total=128 * GIB, runner=ran,
                    system="Darwin", read=ran.read)


@pytest.mark.parametrize("marker", wired.AGENT_MARKERS)
def test_a_process_an_agent_started_cannot_change_or_reset_the_limit(marker):
    ran = Recorder()
    env = {marker: "1"}
    with pytest.raises(HumanRequired):
        wired.apply(65536, via="osascript", total=128 * GIB, runner=ran, system="Darwin",
                    env=env, read=ran.read)
    with pytest.raises(HumanRequired):
        wired.reset(via="sudo", total=128 * GIB, runner=ran, system="Darwin", env=env,
                    terminal=(True, True), read=ran.read)
    assert ran.calls == [] and wired.original_mb() is None


def test_sudo_without_a_terminal_is_refused():
    ran = Recorder()
    with pytest.raises(HumanRequired):
        wired.apply(65536, via="sudo", total=128 * GIB, runner=ran, system="Darwin",
                    terminal=(False, False), read=ran.read)
    assert ran.calls == []


def test_a_cancelled_password_dialog_changes_nothing_and_says_so():
    ran = Recorder(code=1, err="execution error: User canceled. (-128)")
    done = wired.apply(65536, via="osascript", total=128 * GIB, runner=ran, system="Darwin",
                       read=ran.read)
    assert not done.ok and done.message == "cancelled" and done.after_mb == 0


def test_the_command_line_refuses_an_agent_process_before_anything_runs(tmp_path):
    env = {**os.environ, "PYTHONPATH": SRC, "ML_STACK_HOME": str(tmp_path / "home"),
           "ML_STACK_NO_REAL_KEYSTORE": "1", "CLAUDECODE": "1"}
    code = ("import sys; from ml_stack.serve import cli; "
            "sys.exit(cli.COMMANDS.run(['memory','--reset']))")
    got = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True,
                         text=True, stdin=subprocess.DEVNULL, timeout=60)
    assert got.returncode != 0
    assert "agent" in (got.stdout + got.stderr)
    assert not (tmp_path / "home" / "wired-limit.json").exists()


def test_only_the_command_and_the_ui_route_reach_the_privileged_change():
    root = Path(SRC) / "ml_stack"
    users = sorted(p.relative_to(root).as_posix() for p in root.rglob("*.py")
                   if re.search(r"serve import wired$|import wired$",
                                p.read_text(encoding="utf-8"), re.M))
    assert users == ["fleet/room_routes.py", "serve/wired_cli.py"]
