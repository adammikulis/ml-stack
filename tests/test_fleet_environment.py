"""The environment the app builds for training jobs."""

from __future__ import annotations

import sys

import pytest
from ml_stack.fleet.environment import CATALOG, Environment, catalog_for


class TestCatalog:
    def test_the_right_pytorch_is_offered_per_vendor(self):
        """torch for CUDA and torch for ROCm are different downloads from different
        indexes, and installing the wrong one produces a card that is never used."""
        nvidia = {lib.name for lib in catalog_for("nvidia")}
        amd = {lib.name for lib in catalog_for("amd")}
        assert "torch-cuda" in nvidia and "torch-rocm" not in nvidia
        assert "torch-rocm" in amd and "torch-cuda" not in amd

    def test_the_rocm_build_comes_from_the_rocm_index(self):
        rocm = next(lib for lib in CATALOG if lib.name == "torch-rocm")
        assert "rocm" in rocm.index

    def test_mlx_is_only_offered_on_apple_silicon(self):
        mlx = next(lib for lib in CATALOG if lib.name == "mlx")
        assert mlx.platforms == ("darwin",)

    def test_every_library_says_what_it_is_for_and_what_it_costs(self):
        for lib in CATALOG:
            assert lib.title and lib.blurb and lib.packages
            assert lib.size_mb > 0, lib.name

    def test_the_essentials_include_ml_stack_itself(self):
        """A machine can hold torch and still not run a job without these."""
        core = next(lib for lib in CATALOG if lib.name == "core")
        assert any("ml-stack-train" in p for p in core.packages)


class TestEnvironment:
    def test_a_fresh_machine_has_no_environment(self, tmp_path):
        assert not Environment(tmp_path).exists

    def test_it_finds_a_python_to_build_with(self, tmp_path):
        assert Environment(tmp_path).host_python() is not None

    def test_the_interpreter_lands_where_jobs_will_look(self, tmp_path):
        env = Environment(tmp_path)
        assert env.python.parent.parent == env.path
        assert env.path.parent == tmp_path

    def test_it_finds_the_wheels_for_ml_stacks_own_packages(self, tmp_path):
        """They are not on any index, so without them the environment can hold torch
        and still not import ml_stack."""
        found = Environment(tmp_path).wheels()
        if found is None:
            pytest.skip("no wheels built; run packaging/build.py")
        assert any(found.glob("ml_stack_train-*.whl"))

    def test_an_unknown_library_is_reported_not_ignored(self, tmp_path):
        env = Environment(tmp_path)
        env.path.mkdir(parents=True, exist_ok=True)
        (env.python.parent).mkdir(parents=True, exist_ok=True)
        env.python.write_text("")
        got = env.uninstall(["nonesuch"])
        assert not got["nonesuch"]["ok"]

    def test_the_state_names_every_library_and_whether_it_is_there(self, tmp_path):
        state = Environment(tmp_path).state("apple" if sys.platform == "darwin" else "cpu")
        assert state["ready"] is False
        assert state["libraries"]
        for lib in state["libraries"]:
            assert {"name", "title", "blurb", "size_mb", "installed"} <= set(lib)


@pytest.mark.slow
class TestBuildingItForReal:
    def test_it_builds_and_installs_and_a_job_can_use_it(self, tmp_path):
        env = Environment(tmp_path)
        if env.wheels() is None:
            pytest.skip("no wheels built; run packaging/build.py")
        done = env.install(["core"])
        assert done["core"]["ok"], done

        assert env.exists
        have = env.installed()
        assert "numpy" in have and "safetensors" in have
        assert "ml-stack-train" in have, "a job could not import ml_stack.train"

        import subprocess
        out = subprocess.run([str(env.python), "-c",
                              "import ml_stack.train, numpy; print('ok')"],
                             capture_output=True, text=True, timeout=120)
        assert out.returncode == 0, out.stderr[-300:]
