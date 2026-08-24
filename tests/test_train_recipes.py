"""Recipes: a config a form can produce, turned into a real trained model."""

from __future__ import annotations

import json
import math

import pytest
from ml_stack.contracts import ContractError, recipe, recipes
from ml_stack.train.holdout import LeakageError, stratified
from ml_stack.train.recipes import build, known, validate
from ml_stack.train.recipes.models import suggest_size


@pytest.fixture
def corpus(tmp_path):
    rows = [{"text": f"The quick brown fox jumps over the lazy dog number {i}. "
                     f"It was a bright cold day in April."} for i in range(400)]
    d = tmp_path / "corpus"
    d.mkdir()
    (d / "a.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    return d


@pytest.fixture
def reviews(tmp_path):
    good = ["excellent", "wonderful", "great", "loved it"]
    bad = ["terrible", "awful", "hated it", "dreadful"]
    rows = []
    for i in range(300):
        word = (good if i % 2 else bad)[i % 4]
        rows.append({"text": f"This was {word}, and I mean it.",
                     "label": "good" if i % 2 else "bad"})
    d = tmp_path / "reviews"
    d.mkdir()
    (d / "r.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    return d


class TestContracts:
    def test_every_recipe_contract_has_a_builder(self):
        for spec in recipes():
            assert spec["id"] in known()

    def test_every_recipe_describes_itself_to_a_person(self):
        for spec in recipes():
            assert spec.get("title") and spec.get("blurb")
            assert spec["data"]["formats"]
            for f in spec.get("fields", []):
                assert f.get("label"), f"{spec['id']}.{f['name']} has no label"
                assert "default" in f

    def test_an_unknown_recipe_names_the_ones_that_exist(self):
        with pytest.raises(ContractError, match="text-lm"):
            recipe("nonesuch")


class TestValidate:
    def test_defaults_fill_in(self):
        got = validate("text-lm", {})
        assert got["steps"] == recipe("text-lm")["fields"][0]["default"]

    def test_an_undeclared_setting_is_refused_not_ignored(self):
        """A silently dropped hyperparameter is a run that trained on defaults while
        its config said otherwise."""
        with pytest.raises(ValueError, match="lr_schedule"):
            validate("text-lm", {"lr_schedule": "cosine"})

    @pytest.mark.parametrize("bad", [{"steps": 1}, {"steps": 10 ** 9},
                                     {"context": 2}, {"learning_rate": 10}])
    def test_an_out_of_range_setting_is_refused(self, bad):
        with pytest.raises(ValueError):
            validate("text-lm", bad)

    def test_an_unknown_size_lists_the_real_ones(self):
        with pytest.raises(ValueError, match="small"):
            validate("text-lm", {"size": "enormous"})


class TestSizeFitsTheMachine:
    SIZES = {"small": {"needs_gb": 1}, "medium": {"needs_gb": 4},
             "large": {"needs_gb": 12}}

    @pytest.mark.parametrize("have, want", [(0.5, "small"), (2, "small"), (5, "medium"),
                                            (24, "large"), (None, "small")])
    def test_the_biggest_that_fits_is_chosen(self, have, want):
        assert suggest_size(self.SIZES, have) == want


class TestStratified:
    def test_both_classes_appear_on_both_sides(self):
        """by_group on the label puts a whole class in the holdout and none of it in
        training, so the model never sees it and the held-out score is meaningless."""
        rows = list(range(200))
        labels = ["good" if i % 2 else "bad" for i in rows]
        split = stratified(rows, labels, fraction=0.2)

        train_labels = {labels[i] for i in split.train}
        holdout_labels = {labels[i] for i in split.holdout}
        assert train_labels == {"good", "bad"}
        assert holdout_labels == {"good", "bad"}

    def test_nothing_is_in_both_halves(self):
        rows = list(range(100))
        labels = ["a"] * 50 + ["b"] * 50
        split = stratified(rows, labels, fraction=0.3)
        assert not set(split.train) & set(split.holdout)
        assert len(split.train) + len(split.holdout) == 100

    def test_mismatched_lengths_are_refused(self):
        with pytest.raises(LeakageError, match="line up"):
            stratified([1, 2, 3], ["a", "b"])


class TestBuild:
    def test_a_language_model_trains_on_real_text(self, corpus, tmp_path):
        from ml_stack.train.run import run

        got = run("text-lm", {"size": "small", "steps": 120, "context": 64,
                              "batch_size": 8}, corpus, tmp_path / "run")

        assert got["steps"] == 120
        assert got["parameters"] > 0
        # Byte-level: an untrained model sits at ln(256).
        assert got["final_loss"] < math.log(256) * 0.7, got

    def test_a_classifier_generalises_to_held_out_rows(self, reviews, tmp_path):
        from ml_stack.train.run import run

        got = run("classify-text", {"size": "small", "steps": 200, "context": 48},
                  reviews, tmp_path / "run")

        assert got["best_metric"] is not None
        assert got["best_metric"] < math.log(2), (
            "the held-out score is no better than guessing between two labels")

    def test_a_dry_run_leaves_no_checkpoint_behind(self, reviews, tmp_path):
        from ml_stack.train.run import run

        out = tmp_path / "dry"
        got = run("classify-text", {"size": "small"}, reviews, out, dry=True)

        assert got["dry_run"] and got["steps"] <= 20
        assert not [p for p in out.iterdir() if p.is_dir()]

    def test_an_empty_dataset_says_what_it_wanted(self, tmp_path):
        empty = tmp_path / "nothing"
        empty.mkdir()
        with pytest.raises(ValueError, match="jsonl"):
            build("text-lm", {"size": "small"}, empty)

    def test_a_single_label_is_refused(self, tmp_path):
        d = tmp_path / "one"
        d.mkdir()
        (d / "r.jsonl").write_text("\n".join(
            json.dumps({"text": f"row {i}", "label": "same"}) for i in range(50)))
        with pytest.raises(ValueError, match="nothing to learn"):
            build("classify-text", {"size": "small"}, d)

    def test_the_run_records_what_it_actually_trained_on(self, corpus, tmp_path):
        from ml_stack.train import read
        from ml_stack.train.run import run

        run("text-lm", {"size": "small", "steps": 40, "context": 64}, corpus,
            tmp_path / "run")
        start = next(r for r in read(tmp_path / "run" / "metrics.jsonl")
                     if r.get("event") == "start")["config"]

        assert start["recipe"] == "text-lm"
        assert start["documents"] == 400
        assert start["parameters"] > 0
        assert start["holdout_bytes"] > 0


class TestCommandLine:
    def test_the_cli_trains_and_prints_json(self, corpus, tmp_path, capsys):
        from ml_stack.train.run import main

        code = main(["--recipe", "text-lm", "--data", str(corpus),
                     "--out", str(tmp_path / "run"), "--set", "size=small",
                     "--set", "steps=40", "--set", "context=64"])

        assert code == 0
        assert json.loads(capsys.readouterr().out)["steps"] == 40

    def test_a_bad_setting_fails_with_a_message_not_a_traceback(self, corpus, tmp_path,
                                                               capsys):
        from ml_stack.train.run import main

        code = main(["--recipe", "text-lm", "--data", str(corpus),
                     "--out", str(tmp_path / "run"), "--set", "steps=1"])

        assert code == 2
        assert "steps" in capsys.readouterr().err
