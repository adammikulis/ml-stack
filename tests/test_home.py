"""One state directory and one cache directory, both read when they are asked for."""

from __future__ import annotations

from pathlib import Path

from ml_stack import home


class TestTheStateRoot:
    def test_it_defaults_under_the_account_home(self, monkeypatch):
        monkeypatch.delenv("ML_STACK_HOME", raising=False)
        monkeypatch.setattr(home, "user_home", lambda: Path("/somewhere/anne"))
        assert home.home() == Path("/somewhere/anne/.ml-stack")

    def test_one_variable_moves_everything_under_it(self, monkeypatch, tmp_path):
        monkeypatch.setenv("ML_STACK_HOME", str(tmp_path / "elsewhere"))
        assert home.state("jobs", "queue.json") == tmp_path / "elsewhere/jobs/queue.json"

    def test_the_variable_is_read_when_asked_not_at_import(self, monkeypatch, tmp_path):
        monkeypatch.delenv("MLSTACK_BENCH_HOME", raising=False)
        monkeypatch.setenv("ML_STACK_HOME", str(tmp_path / "first"))
        assert home.state("bench") == tmp_path / "first" / "bench"
        monkeypatch.setenv("ML_STACK_HOME", str(tmp_path / "second"))
        assert home.state("bench") == tmp_path / "second" / "bench"

    def test_a_tilde_in_the_variable_is_expanded(self, monkeypatch):
        monkeypatch.setenv("ML_STACK_HOME", "~/somewhere")
        assert home.home() == Path.home() / "somewhere"


class TestTheSubsystemVariables:
    def test_each_one_still_moves_its_own_corner(self, monkeypatch, tmp_path):
        monkeypatch.setenv("ML_STACK_HOME", str(tmp_path / "root"))
        monkeypatch.setenv("MLSTACK_BENCH_HOME", str(tmp_path / "runs"))
        assert home.state("bench", "kept.json") == tmp_path / "runs" / "kept.json"
        assert home.state("train") == tmp_path / "root" / "train"

    def test_a_file_variable_names_the_file_itself(self, monkeypatch, tmp_path):
        monkeypatch.setenv("MLSTACK_FIT_FILE", str(tmp_path / "mine.json"))
        assert home.state("fit.json") == tmp_path / "mine.json"

    def test_an_empty_variable_is_no_variable(self, monkeypatch, tmp_path):
        monkeypatch.setenv("ML_STACK_HOME", str(tmp_path / "root"))
        monkeypatch.setenv("MLSTACK_JOBS_HOME", "")
        assert home.state("jobs") == tmp_path / "root" / "jobs"


class TestTheCacheRoot:
    def test_it_defaults_beside_the_other_caches(self, monkeypatch):
        monkeypatch.delenv("ML_STACK_CACHE", raising=False)
        monkeypatch.setattr(home, "user_home", lambda: Path("/somewhere/anne"))
        assert home.cache("logs") == Path("/somewhere/anne/.cache/ml_stack/logs")

    def test_it_moves_on_its_own_variable(self, monkeypatch, tmp_path):
        monkeypatch.setenv("ML_STACK_CACHE", str(tmp_path / "c"))
        assert home.cache("logs", "one.log") == tmp_path / "c" / "logs" / "one.log"

    def test_the_state_root_does_not_move_the_cache(self, monkeypatch, tmp_path):
        monkeypatch.setenv("ML_STACK_HOME", str(tmp_path / "state"))
        monkeypatch.setenv("ML_STACK_CACHE", str(tmp_path / "cache"))
        assert home.state("x") == tmp_path / "state" / "x"
        assert home.cache("x") == tmp_path / "cache" / "x"


class TestAPathSomebodyNamed:
    def test_a_leading_tilde_is_resolved(self):
        assert home.expand("~/models") == Path.home() / "models"

    def test_anything_else_is_left_alone(self):
        assert home.expand("relative/one") == Path("relative/one")
