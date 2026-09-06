"""One state directory and one cache directory, both read when they are asked for."""

from __future__ import annotations

from pathlib import Path

from ml_stack import bench, home, hub, ingest, jobs
from ml_stack.serve import binary, build, limits, manager, reclaim


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


class TestOneVariableMovesEveryPathTheProjectWrites:
    """``ML_STACK_HOME`` set after import moves every state path, not only some."""

    def test_every_home_and_record_lands_under_it(self, monkeypatch, tmp_path):
        from ml_stack.train import run as train_run

        root = tmp_path / "elsewhere"
        monkeypatch.setenv("ML_STACK_HOME", str(root))
        monkeypatch.setenv("HF_HOME", str(root / "huggingface"))

        under = {
            "managed root": binary.managed_root(),
            "current build": binary.managed_current(),
            "named builds": binary.managed_named(),
            "build root": build.root(),
            "build source": build.src_dir(),
            "builds": build.builds_dir(),
            "current link": build.current_link(),
            "named dir": build.named_dir(),
            "named source": build.named_src_dir(),
            "hub cache": hub.hub_cache(),
            "limits": limits.where(),
            "idle": reclaim.state_path(),
            "lease": manager.lease_file(),
            "jobs": jobs.home_dir(),
            "bench": bench.home_dir(),
            "ingest": ingest.home_dir(),
            "train": train_run.home_dir(),
        }
        strays = {what: path for what, path in under.items()
                  if root not in path.parents}
        assert strays == {}

    def test_it_is_read_again_when_the_variable_moves(self, monkeypatch, tmp_path):
        monkeypatch.setenv("ML_STACK_HOME", str(tmp_path / "first"))
        assert build.builds_dir() == tmp_path / "first" / "llama.cpp" / "builds"
        monkeypatch.setenv("ML_STACK_HOME", str(tmp_path / "second"))
        assert binary.managed_current() == tmp_path / "second" / "llama.cpp" / "current"
        assert build.builds_dir() == tmp_path / "second" / "llama.cpp" / "builds"
        assert limits.where() == tmp_path / "second" / "limits.json"
        assert reclaim.state_path() == tmp_path / "second" / "idle.json"

    def test_the_hub_cache_is_read_again_when_hf_home_moves(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HF_HOME", str(tmp_path / "first"))
        assert hub.hub_cache() == tmp_path / "first" / "hub"
        monkeypatch.setenv("HF_HOME", str(tmp_path / "second"))
        assert hub.hub_cache() == tmp_path / "second" / "hub"


class TestAStatePathThatUsedToLiveInTheCache:
    def test_the_older_copy_is_taken_across_the_first_time_it_is_asked_for(
            self, monkeypatch, tmp_path):
        monkeypatch.setenv("ML_STACK_HOME", str(tmp_path / "state"))
        monkeypatch.setenv("ML_STACK_CACHE", str(tmp_path / "cache"))
        older = tmp_path / "cache" / "limits.json"
        older.parent.mkdir(parents=True)
        older.write_text('{"servers": 2}')

        assert home.moved("limits.json") == tmp_path / "state" / "limits.json"
        assert (tmp_path / "state" / "limits.json").read_text() == '{"servers": 2}'
        assert not older.exists()

    def test_what_is_already_under_the_state_root_wins(self, monkeypatch, tmp_path):
        monkeypatch.setenv("ML_STACK_HOME", str(tmp_path / "state"))
        monkeypatch.setenv("ML_STACK_CACHE", str(tmp_path / "cache"))
        current = tmp_path / "state" / "idle.json"
        current.parent.mkdir(parents=True)
        current.write_text("{}")
        older = tmp_path / "cache" / "idle.json"
        older.parent.mkdir(parents=True)
        older.write_text('{"8080": {}}')

        assert home.moved("idle.json") == current
        assert older.read_text() == '{"8080": {}}'

    def test_nothing_anywhere_names_the_state_root(self, monkeypatch, tmp_path):
        monkeypatch.setenv("ML_STACK_HOME", str(tmp_path / "state"))
        monkeypatch.setenv("ML_STACK_CACHE", str(tmp_path / "cache"))
        assert home.moved("limits.json") == tmp_path / "state" / "limits.json"
        assert not (tmp_path / "state").exists()

    def test_a_move_that_cannot_be_made_reads_the_older_copy_where_it_is(
            self, monkeypatch, tmp_path):
        monkeypatch.setenv("ML_STACK_HOME", str(tmp_path / "state"))
        monkeypatch.setenv("ML_STACK_CACHE", str(tmp_path / "cache"))
        older = tmp_path / "cache" / "limits.json"
        older.parent.mkdir(parents=True)
        older.write_text('{"seats": 3}')
        (tmp_path / "state").write_text("not a directory")

        assert home.moved("limits.json") == older
        assert older.read_text() == '{"seats": 3}'

    def test_the_limits_and_the_idle_record_carry_across(self, monkeypatch, tmp_path):
        monkeypatch.setenv("ML_STACK_HOME", str(tmp_path / "state"))
        monkeypatch.setenv("ML_STACK_CACHE", str(tmp_path / "cache"))
        (tmp_path / "cache").mkdir(parents=True)
        (tmp_path / "cache" / "limits.json").write_text('{"servers": 2, "seats": 4}')
        (tmp_path / "cache" / "idle.json").write_text('{"8100": {"idle": 90.0}}')

        assert limits.where() == tmp_path / "state" / "limits.json"
        held = limits.read()
        assert (held.servers, held.seats) == (2, 4)
        assert reclaim.state_path() == tmp_path / "state" / "idle.json"
        assert (tmp_path / "state" / "idle.json").read_text() == '{"8100": {"idle": 90.0}}'
