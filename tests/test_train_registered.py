"""A recipe defined outside the library, trained through `train.run` on real torch and mlx.

The recipe is a module written into ``tmp_path`` and imported, the way an app's own recipe
is: registered by import, named on the command line with ``--import``.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path

import numpy as np
import pytest

from ml_stack.contracts import ContractError
from ml_stack.testing import needs_a_backend, needs_mlx, needs_torch
from ml_stack.train import find_latest, load_state, read
from ml_stack.train.probes import INBOX
from ml_stack.train.recipes import build, known, load, register, spec, specs, validate
from ml_stack.train.run import main, run

SRC = Path(__file__).resolve().parents[1] / "src"

MODULE = '''
import time

import numpy as np

from ml_stack.train.recipes import Built, Hook, Phase, register

SPEC = {
    "id": "RECIPE_ID", "title": "A line through points", "blurb": "Fits y = w.x.",
    "data": {"formats": ["none"]}, "requires": {"backend": ""}, "sizes": {},
    "fields": [
        {"name": "steps", "type": "int", "default": 30, "min": 1, "max": 100000,
         "label": "Steps"},
        {"name": "learning_rate", "type": "float", "default": 0.05, "min": 0.0, "max": 1.0,
         "label": "Learning rate"},
        {"name": "pause", "type": "float", "default": 0.0, "min": 0.0, "max": 1.0,
         "label": "Seconds each batch waits"},
        {"name": "phases", "type": "int", "default": 0, "min": 0, "max": 1,
         "label": "Whether to run the follow-on phase"},
    ],
}


def build_line(spec, config, data, framework):
    rng = np.random.default_rng(0)
    w = rng.normal(size=(4, 1)).astype(np.float32)
    prepared = []
    if framework == "mlx":
        import mlx.core as mx
        import mlx.nn as nn
        import mlx.optimizers as optim

        class Line(nn.Module):
            def __init__(self):
                super().__init__()
                self.inner = nn.Linear(4, 1)
                self.widths = [4, 1]
                self.label = None

            def __call__(self, x):
                return self.inner(x)

        model = Line()
        optimizer = optim.Adam(learning_rate=config["learning_rate"], bias_correction=True)
        array = mx.array

        def loss(m, batch):
            return mx.mean((m(batch[0]) - batch[1]) ** 2)

        def forward(x):
            return np.array(model(mx.array(x))).tolist()
    else:
        import torch

        model = torch.nn.Linear(4, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
        array = torch.as_tensor

        def loss(m, batch):
            return ((m(batch[0]) - batch[1]) ** 2).mean()

        def forward(x):
            with torch.no_grad():
                return model(torch.as_tensor(np.asarray(x, dtype=np.float32))).tolist()

    def batches(step):
        time.sleep(config["pause"])
        x = np.random.default_rng(step).normal(size=(16, 4)).astype(np.float32)
        return array(x), array(x @ w)

    def export(directory):
        (directory / "prepared.json").write_text(str(prepared))
        return {"file": str(directory / "prepared.json")}

    def grow():
        prepared.append("grow")
        return "nothing to grow; kept training"

    phases = [Phase("grow", grow, steps=5, learning_rate=0.01),
              Phase("never", lambda: None, steps=5, learning_rate=0.01)]
    return Built(model=model, optimizer=optimizer, loss=loss,
                 batches=None if data is None else batches,
                 predict=lambda inputs: forward(np.asarray(inputs, dtype=np.float32)),
                 hooks=[Hook("every-five", 5, lambda step: {"seen": step})],
                 phases=phases if config["phases"] else [], export=export,
                 config={**config, "framework": framework or "torch"})


register(SPEC, build_line)
'''


def a_recipe(tmp_path: Path, monkeypatch) -> tuple[str, str]:
    """(module, recipe id) for a freshly written module that registers a recipe on import."""
    tag = uuid.uuid4().hex[:8]
    name, recipe_id = f"line_recipe_{tag}", f"line-{tag}"
    (tmp_path / f"{name}.py").write_text(MODULE.replace("RECIPE_ID", recipe_id))
    monkeypatch.syspath_prepend(str(tmp_path))
    return name, recipe_id


@pytest.fixture
def line(tmp_path, monkeypatch):
    import importlib

    name, recipe_id = a_recipe(tmp_path, monkeypatch)
    importlib.import_module(name)
    return recipe_id


@pytest.fixture
def data(tmp_path):
    (tmp_path / "data").mkdir()
    return tmp_path / "data"


FRAMEWORKS = [pytest.param("torch", marks=needs_torch), pytest.param("mlx", marks=needs_mlx)]


def notes(out: Path, message: str) -> list[dict]:
    return [r for r in read(out / "metrics.jsonl")
            if r["event"] == "note" and r["message"] == message]


class TestRegistering:
    def test_a_registered_recipe_is_known_and_validated_like_a_shipped_one(self, line):
        assert line in known() and "text-lm" in known()
        assert [s["id"] for s in specs()] == sorted(known())
        assert spec(line)["title"] == "A line through points"
        got = validate(line, {"steps": 7})
        assert got["steps"] == 7 and got["framework"] == "" and got["max_minutes"] == 0.0

    def test_an_id_that_exists_is_refused(self, line):
        with pytest.raises(ValueError, match="already exists"):
            register(spec(line), lambda *a: None)
        with pytest.raises(ValueError, match="already exists"):
            register({**spec("text-lm")}, lambda *a: None)

    def test_an_unknown_recipe_names_the_registered_ones_too(self, line):
        with pytest.raises(ContractError, match=line):
            spec("nonesuch")

    @pytest.mark.parametrize("bad", [{"framework": "jax"}, {"max_minutes": -1}])
    def test_a_bad_framework_or_budget_is_refused(self, line, bad):
        with pytest.raises(ValueError):
            validate(line, bad)


class TestRun:
    @pytest.mark.parametrize("framework", FRAMEWORKS)
    def test_a_registered_recipe_trains_hooks_phases_and_exports(self, line, data, tmp_path,
                                                                 framework):
        out = tmp_path / "run"
        got = run(line, {"framework": framework, "steps": 30, "phases": 1}, data, out)

        assert got["stop_reason"] == "completed"
        assert got["phases"] == [{"name": "grow", "prepared": "nothing to grow; kept training",
                                  "steps": 5}]
        assert got["steps"] == 35
        records = read(out / "metrics.jsonl")
        assert [r["step"] for r in records if r["event"] == "step"] == list(range(35))
        assert records[0]["config"]["framework"] == framework
        assert [n["step"] for n in notes(out, "every-five")] == [5, 10, 15, 20, 25, 30, 35]
        [phase] = notes(out, "phase")
        assert phase["step"] == 30 and phase["phase"] == "grow"
        assert load_state(find_latest(out)).step == 35
        assert json.loads((out / "manifest.json").read_text())["phases"] == got["phases"]
        assert Path(got["export"]["file"]).read_text() == "['grow']"
        steps = [r["loss"] for r in records if r["event"] == "step"]
        assert steps[-1] < steps[0] / 10

    def test_a_stop_ends_the_run_with_a_checkpoint_and_no_phases(self, line, data, tmp_path):
        out = tmp_path / "run"
        asked = []
        got = run(line, {"steps": 1000, "phases": 1}, data, out,
                  should_stop=lambda: asked.append(1) or len(asked) > 12)

        assert got["stop_reason"] == "stopped" and got["steps"] == 12
        assert got["phases"] == [] and "export" not in got
        assert load_state(find_latest(out)).step == 12

    def test_a_tiny_time_budget_ends_the_run_and_says_so(self, line, data, tmp_path):
        out = tmp_path / "run"
        got = run(line, {"steps": 1000, "pause": 0.05, "max_minutes": 0.004}, data, out)

        assert got["stop_reason"] == "time_limit" and got["steps"] < 10
        assert notes(out, "time limit fits few steps")
        assert load_state(find_latest(out)).step == got["steps"]

    def test_a_probe_is_answered_and_removed_and_a_bad_one_is_a_note(self, line, data,
                                                                      tmp_path):
        out = tmp_path / "run"
        inbox = out / INBOX
        inbox.mkdir(parents=True)
        (inbox / "a.json").write_text(json.dumps({"id": "ones", "inputs": [[1, 1, 1, 1]]}))
        (inbox / "b.json").write_text(json.dumps({"id": "wrong", "inputs": [[1, 2]]}))

        got = run(line, {"steps": 10, "eval_every": 5}, data, out)

        assert got["stop_reason"] == "completed"
        answered = notes(out, "probe")
        assert [n["id"] for n in answered] == ["ones", "wrong"]
        assert answered[0]["step"] == 5
        assert np.asarray(answered[0]["output"]).shape == (1, 1)
        assert "error" in answered[1] and "output" not in answered[1]
        assert not list(inbox.iterdir())


class TestLoad:
    @pytest.mark.parametrize("framework", FRAMEWORKS)
    def test_a_trained_text_lm_is_rebuilt_from_its_checkpoint(self, tmp_path, framework):
        from ml_stack.train import Trainer

        corpus = tmp_path / "corpus"
        corpus.mkdir()
        (corpus / "a.jsonl").write_text("\n".join(
            json.dumps({"text": f"a small line of text number {i}"}) for i in range(200)))
        built = build("text-lm", {"size": "small", "steps": 20, "context": 32,
                                  "batch_size": 4}, corpus, framework=framework)
        Trainer(built.model, built.optimizer, built.loss, out=tmp_path / "run").fit(
            built.batches, steps=20, config=built.config)

        loaded = load(tmp_path / "run")

        assert loaded.batches is None
        assert loaded.config["framework"] == framework
        assert loaded.predict("a small") == built.predict("a small")
        window = np.arange(32, dtype=np.int64)[None] % 256
        if framework == "torch":
            import torch

            with torch.no_grad():
                same = [m(torch.as_tensor(window)).numpy() for m in (built.model, loaded.model)]
        else:
            import mlx.core as mx

            same = [np.array(m(mx.array(window))) for m in (built.model, loaded.model)]
        np.testing.assert_allclose(same[0], same[1], rtol=1e-5, atol=1e-5)

    @needs_a_backend
    def test_a_classifier_is_rebuilt_with_the_classes_it_trained_on(self, tmp_path):
        rows = [{"text": f"row {i} is {'good' if i % 2 else 'bad'}",
                 "label": "good" if i % 2 else "bad"} for i in range(60)]
        (tmp_path / "rows").mkdir()
        (tmp_path / "rows" / "r.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
        got = run("classify-text", {"size": "small", "steps": 20, "context": 32},
                  tmp_path / "rows", tmp_path / "run")

        loaded = load(tmp_path / "run")

        assert got["stop_reason"] == "completed"
        assert set(loaded.predict("row 3 is good")["scores"]) == {"bad", "good"}

    def test_nothing_to_load_is_an_error(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load(tmp_path)


class TestCommandLine:
    def test_import_makes_a_registered_recipe_a_valid_recipe(self, tmp_path, monkeypatch,
                                                            data, capsys):
        name, recipe_id = a_recipe(tmp_path, monkeypatch)
        argv = ["--recipe", recipe_id, "--data", str(data), "--out", str(tmp_path / "run"),
                "--set", "steps=10"]
        with pytest.raises(SystemExit):
            main(argv)
        capsys.readouterr()

        assert main(["--import", name, *argv]) == 0
        assert json.loads(capsys.readouterr().out)["stop_reason"] == "completed"

    @pytest.mark.slow
    @needs_torch
    def test_sigterm_ends_a_run_with_its_checkpoint(self, tmp_path, monkeypatch, data):
        name, recipe_id = a_recipe(tmp_path, monkeypatch)
        out = tmp_path / "run"
        env = {**os.environ,
               "PYTHONPATH": os.pathsep.join([str(SRC), str(tmp_path)])}
        child = subprocess.Popen(
            [sys.executable, "-m", "ml_stack.train.run", "--import", name,
             "--recipe", recipe_id, "--data", str(data), "--out", str(out),
             "--set", "steps=100000", "--set", "pause=0.02", "--set", "framework=torch"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
        try:
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline:
                metrics = out / "metrics.jsonl"
                if metrics.is_file() and any(r["event"] == "step" for r in read(metrics)):
                    break
                time.sleep(0.1)
            child.send_signal(signal.SIGTERM)
            said, err = child.communicate(timeout=60)
        finally:
            child.kill()

        assert child.returncode == 0, err
        result = json.loads(said)
        assert result["stop_reason"] == "stopped" and 0 < result["steps"] < 100000
        assert load_state(find_latest(out)).step == result["steps"]
