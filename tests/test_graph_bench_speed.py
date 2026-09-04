"""``ml-stack-bench speed``: prefill, decode and the first token by prompt size and streams.

Every fixture here is invented. Nothing reads a real store, a real graph, or a real server.
"""

from __future__ import annotations

import json
import threading
import time

import pytest

from ml_stack.graph.bench import runs
from ml_stack.graph.bench.speed import (
    KIND,
    calibrated,
    cell,
    count_tokens,
    grid,
    only,
    prompt_text,
    speed_table,
)


class _Reply:
    def __init__(self, raw, content="a summary"):
        self.raw, self.content, self.tool_calls = raw, content, None
        self.thinking, self.finish_reason = None, "stop"


class _Llama:
    """A llama-server as far as the grid asks one: counts by /tokenize at four characters a
    token, answers with timings, and remembers how many requests were in flight together."""

    base_url = "http://127.0.0.1:1"

    def __init__(self, *, prefill_tps=1000.0, decode_tps=40.0, drafted=True):
        self.prefill_tps, self.decode_tps, self.drafted = prefill_tps, decode_tps, drafted
        self.calls: list[dict] = []
        self.in_flight = 0
        self.most_in_flight = 0
        self._lock = threading.Lock()
        self.sampling = {"temperature": 0.0}

    def tokenize(self, text):
        return list(range(len(text) // 4))

    def chat(self, messages, **kw):
        with self._lock:
            self.in_flight += 1
            self.most_in_flight = max(self.most_in_flight, self.in_flight)
        try:
            n = len(messages[-1]["content"]) // 4 + 12
            wrote = 1 if kw.get("n_predict") == 1 else 256
            if wrote > 1:
                time.sleep(0.05)          # long enough for the streams to overlap
            self.calls.append({"prompt_n": n, "kw": dict(kw)})
            timings = {"prompt_n": n, "cache_n": 0, "prompt_ms": n / self.prefill_tps * 1000,
                       "predicted_n": wrote, "predicted_ms": wrote / self.decode_tps * 1000}
            if self.drafted:
                timings.update(draft_n=wrote, draft_n_accepted=int(wrote * 0.8))
            return _Reply({"timings": timings,
                           "usage": {"prompt_tokens": n, "completion_tokens": wrote}})
        finally:
            with self._lock:
                self.in_flight -= 1


class _Ollama:
    """Ollama as far as the grid asks it: no /tokenize, counts by reply, no cache and no
    draft figure, durations in nanoseconds."""

    base_url = "http://127.0.0.1:11434"

    def __init__(self):
        self.calls = []
        self.sampling = {"temperature": 0.0}

    def chat(self, messages, **kw):
        n = len(messages[-1]["content"]) // 5 + 9
        wrote = 1 if kw.get("n_predict") == 1 else 256
        self.calls.append(dict(kw))
        return _Reply({"model": "thornfell:125b-mlx", "prompt_eval_count": n,
                       "prompt_eval_duration": int(n / 600 * 1e9), "eval_count": wrote,
                       "eval_duration": int(wrote / 30 * 1e9), "load_duration": 4_000_000_000})

    def served_by(self):
        return {"program": "ollama", "version": "0.33.3", "format": "safetensors",
                "runtime": "mlx", "quant": "nvfp4", "model": "thornfell:125b-mlx",
                "weights_bytes": 60 * 2**30}

    def processes(self):
        return []


class _Silent:
    """A program that reports no timings and no usage at all."""

    base_url = "http://127.0.0.1:1"
    sampling = {}

    def chat(self, messages, **kw):
        return _Reply({})


def test_the_prompt_is_deterministic_and_differs_by_seed():
    a, b = prompt_text(512, seed=1), prompt_text(512, seed=1)
    assert a == b
    assert prompt_text(512, seed=2) != a, "two streams never share a prefix"
    assert prompt_text(512, seed=2).split("\n")[0] != a.split("\n")[0]
    assert len(prompt_text(512, chars=1600)) in (1599, 1600), "chars fixes the length"


def test_the_count_comes_from_tokenize_when_the_server_has_it_and_the_reply_when_not():
    assert count_tokens(_Llama(), "x" * 400) == (100, "tokenize")
    got, how = count_tokens(_Ollama(), "x" * 400)
    assert how == "reply" and got == 89
    assert count_tokens(_Silent(), "x" * 400) == (None, "unknown")


def test_the_prompt_is_scaled_until_it_lands_within_tolerance():
    """At four characters a token the first guess is right; at five it is a fifth short,
    and the text is scaled by the miss and counted again."""
    text, measured, built = calibrated(_Llama(), 512, seed=3)
    assert measured == 512 and built["method"] == "tokenize"
    assert len(built["steps"]) == 1
    client = _Ollama()
    text, measured, built = calibrated(client, 1000, seed=3)
    assert built["method"] == "reply"
    assert abs(measured - 1000) <= 20, built
    assert len(built["steps"]) >= 2, "scaled at least once"
    assert all(kw.get("n_predict") == 1 for kw in client.calls), "counted by a one-token reply"
    assert "counted by the server" in built["how"]
    text, measured, built = calibrated(_Silent(), 512)
    assert measured is None and built["method"] == "unknown" and text


def test_a_cell_sends_its_streams_together_and_reads_the_rates_off_the_replies():
    client = _Llama(prefill_tps=1000.0, decode_tps=40.0)
    got = cell(client, tokens=512, streams=4, generate=256, seed=1)
    assert got["streams"] == 4 and got["prompt_tokens"] == 512
    assert client.most_in_flight == 4, "four requests in flight together"
    assert got["prefill_tps"] == pytest.approx(4 * 1000.0, rel=0.05), "read over the cell"
    assert got["prefill_tps_per_stream"] == pytest.approx(1000.0, rel=0.05)
    assert got["decode_tps"] == pytest.approx(160.0, rel=0.05), "summed for throughput"
    assert got["decode_tps_per_stream"] == pytest.approx(40.0, rel=0.05)
    assert got["ttft_s"] == pytest.approx(0.524, rel=0.05) and got["ttft_from"] == "prompt_ms"
    assert got["draft_tokens"] == 4 * 256 and got["draft_taken"] == 4 * int(256 * 0.8)
    assert got["errors"] == 0 and len(got["requests"]) == 4
    assert got["prompt_measured"] == pytest.approx(512, rel=0.05)
    assert all(c["kw"].get("think") is False for c in client.calls), "thinking off"
    texts = {json.dumps(c["prompt_n"]) for c in client.calls[-4:]}
    assert len(texts) >= 1


def test_a_cell_on_a_program_with_no_cache_or_draft_figure_says_none_not_zero():
    got = cell(_Ollama(), tokens=256, streams=2, generate=256, seed=2)
    assert got["prefill_tps"] is not None and got["decode_tps"] == pytest.approx(60.0, rel=0.05)
    assert got["cached_tokens"] is None and got["draft_tokens"] is None
    assert got["draft_taken"] is None


def test_a_cell_on_a_program_that_reports_nothing_keeps_the_wall_and_nothing_else():
    got = cell(_Silent(), tokens=64, streams=1, generate=8)
    assert got["prefill_tps"] is None and got["decode_tps"] is None and got["ttft_s"] is None
    assert got["ttft_from"] is None and got["prompt_measured"] is None
    assert got["wall_s"] >= 0 and got["errors"] == 0


def test_a_smoke_grid_is_one_cell_and_a_full_grid_is_every_pair():
    client = _Llama()
    assert len(grid(client, prompts=[64, 128], streams=[1, 2], generate=8, smoke=True)) == 1
    cells = grid(client, prompts=[64, 128], streams=[1, 2], generate=8)
    assert [(c["prompt_tokens"], c["streams"]) for c in cells] == [(64, 1), (64, 2), (128, 1),
                                                                    (128, 2)]


def test_the_speed_subcommand_on_a_standing_server_keeps_one_run_per_label(tmp_path, monkeypatch,
                                                                        capsys):
    import ml_stack.graph.bench as bench
    from ml_stack.graph.bench import backends

    client = _Ollama()
    monkeypatch.setattr(bench, "HOME", tmp_path / "home")
    monkeypatch.setattr(bench, "busy", lambda url: 0)
    monkeypatch.setattr(bench, "footprint",
                        lambda url, client=None: {"base_url": url, "resident_bytes": 65 * 2**30,
                                                  "served_by": client.served_by()})
    seen = {}

    def fake_client_for(url, **kw):
        seen.update(url=url, **kw)
        return client

    monkeypatch.setattr("ml_stack.graph.bench.speed.client_for", fake_client_for)
    kept = tmp_path / "runs.ladybug"
    code = bench._main(["speed", "--on", "flash-ollama=ollama://127.0.0.1:11434/thornfell:125b-mlx",
                        "--prompts", "64,128", "--streams", "1,2", "--generate", "8",
                        "--kept", str(kept), "--no-smoke", "--context", "8192"])
    assert code == 0
    assert seen["url"] == "ollama://127.0.0.1:11434/thornfell:125b-mlx"
    assert seen["n_predict"] == 8 and seen["temperature"] == 0.0 and seen["context"] == 8192
    kept_runs = runs(kept)
    assert len(kept_runs) == 1 and kept_runs[0]["kind"] == KIND
    assert kept_runs[0]["label"] == "flash-ollama-speed"
    assert len(kept_runs[0]["rows"]) == 4
    assert kept_runs[0]["server"]["served_by"]["program"] == "ollama"
    said = capsys.readouterr().out
    assert "flash-ollama-speed" in said and "ollama 0.33.3 · mlx · nvfp4" in said
    assert "prefill" in said and "decode" in said
    assert only(kept_runs) == kept_runs


def test_a_speed_run_that_is_not_a_smoke_smokes_one_cell_first(tmp_path, monkeypatch, capsys):
    import ml_stack.graph.bench as bench

    client = _Llama()
    monkeypatch.setattr(bench, "HOME", tmp_path / "home")
    monkeypatch.setattr(bench, "busy", lambda url: 0)
    monkeypatch.setattr(bench, "footprint", lambda url, client=None: {"base_url": url})
    monkeypatch.setattr("ml_stack.graph.bench.speed.client_for", lambda url, **kw: client)
    kept = tmp_path / "runs.ladybug"
    assert bench._main(["speed", "--on", "flash=http://127.0.0.1:1", "--prompts", "64,128",
                        "--streams", "1", "--generate", "8", "--kept", str(kept)]) == 0
    got = runs(kept)
    assert [len(r["rows"]) for r in got] == [1, 2], "the smoke cell, then the grid"
    assert "smoke: ok" in capsys.readouterr().out


def test_a_speed_run_whose_every_request_fails_stops_at_the_smoke(tmp_path, monkeypatch):
    import ml_stack.graph.bench as bench
    from ml_stack.graph.bench.serve import SmokeFailed

    class Down:
        base_url = "http://127.0.0.1:1"
        sampling = {}

        def chat(self, messages, **kw):
            raise ConnectionError("nothing there")

    monkeypatch.setattr(bench, "HOME", tmp_path / "home")
    monkeypatch.setattr(bench, "busy", lambda url: 0)
    monkeypatch.setattr(bench, "footprint", lambda url, client=None: {"base_url": url})
    monkeypatch.setattr("ml_stack.graph.bench.speed.client_for", lambda url, **kw: Down())
    with pytest.raises(SmokeFailed, match="every request failed"):
        bench._main(["speed", "--on", "flash=http://127.0.0.1:1", "--prompts", "64",
                     "--streams", "1", "--generate", "8", "--kept", str(tmp_path / "r.ladybug")])


def test_the_speed_subcommand_serves_a_model_without_its_head_and_labels_it_so(tmp_path,
                                                                               monkeypatch,
                                                                               capsys):
    """``--serve`` goes through the same lease the sweep uses: the profile's shape, minus
    the head with ``--no-draft``, labelled ``<stem>-nodraft-speed``."""
    from contextlib import contextmanager
    from dataclasses import replace

    import ml_stack.client
    import ml_stack.serve
    import ml_stack.graph.bench as bench
    from ml_stack.serve import ServerInfo
    from ml_stack.serve.profile import record

    from test_graph_bench import _preflight_ok

    measured = record("tiny.gguf", seat_context=4096, cache_type="q8_0",
                      draft="/models/mtp-tiny.gguf", spec_type="draft-mtp", spec_draft_max=4)
    monkeypatch.setattr("ml_stack.serve.profile.profile_for",
                        lambda m: replace(measured, served=str(m)))
    seen = {"kwargs": [], "clients": []}
    _preflight_ok(monkeypatch)

    @contextmanager
    def fake_serve(model, **kw):
        seen["kwargs"].append(dict(kw))
        yield ServerInfo(base_url="http://127.0.0.1:1", port=1, pid=None, backend="fake",
                         load_s=3.0, warmup_s=None)

    class Built(_Llama):
        def __init__(self, base_url, **settings):
            super().__init__()
            self.settings = dict(settings)
            seen["clients"].append(self)

    monkeypatch.setattr(bench, "HOME", tmp_path / "home")
    monkeypatch.setattr(bench, "find_model", lambda named: named)
    monkeypatch.setattr(bench, "footprint",
                        lambda url, client=None: {"base_url": url, "model": "tiny.gguf"})
    monkeypatch.setattr(ml_stack.serve, "serve", fake_serve)
    monkeypatch.setattr(ml_stack.client, "Client", Built)
    kept = tmp_path / "runs.ladybug"
    assert bench._main(["speed", "--serve", "tiny.gguf", "--no-draft", "--serve-label", "flash",
                        "--prompts", "64", "--streams", "1,2", "--generate", "8",
                        "--kept", str(kept), "--serve-port", "1", "--smoke"]) == 0
    kw = seen["kwargs"][0]
    assert not kw.get("draft") and not kw.get("spec_type"), "served without the head"
    assert kw.get("cache_type_k") == "q8_0", "the rest of the measured shape stays"
    assert kw.get("parallel") == 2, "a slot per stream"
    got = runs(kept)
    assert [r["label"] for r in got] == ["flash-nodraft-speed"]
    assert got[0]["server"]["load_s"] == 3.0 and got[0]["server"]["binary"]
    assert "draft_model" not in got[0]["server"]
    assert seen["clients"][-1].settings.get("n_predict") == 8
    assert seen["clients"][-1].settings.get("temperature") == 0.0
    # and with the head, the label says nothing extra
    assert bench._main(["speed", "--serve", "tiny.gguf", "--serve-label", "flash",
                        "--prompts", "64", "--streams", "1", "--generate", "8",
                        "--kept", str(kept), "--serve-port", "1", "--smoke"]) == 0
    assert seen["kwargs"][-1].get("draft") == "/models/mtp-tiny.gguf"
    assert [r["label"] for r in runs(kept)][-1] == "flash-speed"
    assert runs(kept)[-1]["server"]["draft_model"] == "mtp-tiny.gguf"


def test_show_speed_prints_the_speed_runs_and_the_answering_table_leaves_them_out(tmp_path,
                                                                                capsys):
    from ml_stack.graph.bench import save, table

    kept = tmp_path / "runs.ladybug"
    save(kept, [{"prompt_tokens": 512, "streams": 1, "prefill_tps": 900.0, "decode_tps": 31.5,
                 "decode_tps_per_stream": 31.5, "ttft_s": 0.57, "ttft_from": "prompt_ms",
                 "wall_s": 9.1, "draft_tokens": 200, "draft_taken": 150, "errors": 0},
                {"prompt_tokens": 4096, "streams": 2, "prefill_tps": None, "decode_tps": None,
                 "ttft_s": None, "wall_s": 20.0, "draft_tokens": None, "errors": 1}],
         held={"served_by": {"program": "llama.cpp", "format": "gguf", "quant": "Q4_K_XL"},
               "build": "unsloth", "resident_peak": 70 * 2**30, "load_s": 40.0},
         kind=KIND, label="flash-speed")
    speed_table(runs(kept))
    said = capsys.readouterr().out
    assert "flash-speed" in said and "llama.cpp (unsloth) · gguf · Q4_K_XL" in said
    assert "peak 70.0G" in said and "load 40s" in said
    first = next(ln for ln in said.splitlines() if ln.split()[:2] == ["512", "1"])
    assert "900" in first and "31.5" in first and "0.57s*" in first and "75%" in first
    second = next(ln for ln in said.splitlines() if ln.split()[:2] == ["4096", "2"])
    assert second.split()[2:5] == ["-", "-", "-"], "not measured is a dash"
    table(runs(kept))
    assert "flash-speed" not in capsys.readouterr().out


def test_the_selfcheck_drives_speed_through_the_whole_path():
    from ml_stack.graph.bench.selfcheck import selfcheck

    said = selfcheck(["speed", "--serve", "tiny.gguf", "--serve-label", "tiny"])
    assert said.startswith("speed: ") and "tiny-speed" in said and "read back" in said
    said = selfcheck(["speed", "--on", "flash=http://127.0.0.1:8080", "--smoke"])
    assert "flash-speed" in said
