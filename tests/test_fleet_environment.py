"""The environment the app builds for training jobs."""

from __future__ import annotations

import sys

import pytest

from ml_stack.fleet.environment import CATALOG, METADRIVE_SOURCE, Environment, catalog_for


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
        assert any(p.startswith("ml-stack[") for p in core.packages), core.packages


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
        assert any(found.glob("ml_stack-*.whl"))

    def test_an_unknown_library_is_reported_not_ignored(self, tmp_path):
        env = Environment(tmp_path)
        env.path.mkdir(parents=True, exist_ok=True)
        (env.python.parent).mkdir(parents=True, exist_ok=True)
        env.python.write_text("")
        got = env.uninstall(["nonesuch"])
        assert not got["nonesuch"]["ok"]

    def test_a_package_the_daemons_interpreter_can_import_shows_installed(self, tmp_path):
        """pytest is only ever importable through the interpreter running this test."""
        env = Environment(tmp_path)
        assert not env.exists
        have = env.daemon_installed()
        assert "pytest" in have
        assert env.installed() == {}

    def test_a_frozen_app_reports_only_what_its_environment_holds(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sys, "frozen", True, raising=False)
        env = Environment(tmp_path)
        assert env.installed() == {}
        env.python.parent.mkdir(parents=True)
        env.python.write_text('#!/bin/sh\necho \'{"installed":[{"metadata": {"name": "Numpy", "version": "2.0.0"}}]}\'\n')
        env.python.chmod(0o755)
        assert env.installed() == {"numpy": "2.0.0"}

    def test_a_checkout_reports_the_managed_interpreter_separately(self, tmp_path):
        env = Environment(tmp_path)
        env.python.parent.mkdir(parents=True)
        env.python.write_text('#!/bin/sh\necho \'{"installed":[{"metadata": {"name": "Nonesuch", "version": "9.9"}}]}\'\n')
        env.python.chmod(0o755)
        have = env.installed()
        assert have["nonesuch"] == "9.9" and "pytest" not in have
        assert "pytest" in env.daemon_installed()

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
        assert "ml-stack" in have, "a job could not import ml_stack.train"

        import subprocess
        out = subprocess.run([str(env.python), "-c",
                              "import ml_stack.train, numpy; print('ok')"],
                             capture_output=True, text=True, timeout=120)
        assert out.returncode == 0, out.stderr[-300:]



class TestFetchingPython:
    def _release(self, monkeypatch, tmp_path, assets):
        import json

        from ml_stack.fleet import environment as mod
        from tests.net_site import Site

        env = Environment(tmp_path)
        listed = [{**a, "name": f"cpython-{mod.PYTHON}.1{env._asset_name()}.tar.gz"} for a in assets]
        site = Site()
        site.__enter__()
        site.add("/release", json.dumps({"assets": listed}).encode(),
                 headers={"Content-Type": "application/json"})
        monkeypatch.setattr(mod, "STANDALONE", site.base + "/release")
        return env, site

    def test_a_build_with_no_digest_is_refused_before_it_is_downloaded(
            self, monkeypatch, tmp_path, loopback_net):
        env, site = self._release(monkeypatch, tmp_path, [
            {"browser_download_url": "http://127.0.0.1:1/python.tar.gz", "size": 10}])
        try:
            with pytest.raises(OSError, match="SHA-256"):
                env.fetch_python()
        finally:
            site.__exit__(None, None, None)
        assert env.standalone_python() is None


def test_requirement_names_support_direct_urls():
    from ml_stack.fleet.environment import _base
    assert _base('MetaDrive-Simulator @ git+https://example.invalid/project.git@abc') == 'metadrive-simulator'
    assert _base('stable_baselines3>=2.0') == 'stable-baselines3'


def test_extra_readiness_requires_selected_dependencies(monkeypatch):
    from email.message import Message
    from types import SimpleNamespace

    from ml_stack.fleet import environment
    metadata = Message()
    metadata['Provides-Extra'] = 'gym-driving'
    dist = SimpleNamespace(metadata=metadata, requires=[
        'packaging>=24.2', 'gymnasium>=1.0; extra == "gym-driving"',
        'rware; extra == "gym-warehouse"'])
    monkeypatch.setattr(environment.metadata, 'distribution', lambda name: dist)
    library = next(lib for lib in CATALOG if lib.name == 'gym-driving')
    have = {'ml-stack': '0.3', 'packaging': '25.0', 'gymnasium': '1.2'}
    assert not environment._library_installed(library, have)
    have['metadrive-simulator'] = '0.4'
    base, revision = METADRIVE_SOURCE.split('git+')[1].rsplit('@', 1)
    cache = {'direct_urls': {'metadrive-simulator': {'url': base, 'vcs_info': {'vcs': 'git', 'commit_id': revision}}}}
    assert environment._library_installed(library, have, cache)
    have['gymnasium'] = '0.29'
    assert not environment._library_installed(library, have)


def test_pinned_git_readiness_requires_exact_installed_commit():
    from packaging.requirements import Requirement

    from ml_stack.fleet.environment import _direct_matches
    requirement = Requirement('native @ git+https://example.invalid/simulator.git@abc')
    urls = {'native': {'url': 'https://example.invalid/simulator.git',
                       'vcs_info': {'vcs': 'git', 'requested_revision': 'abc', 'commit_id': 'def'}}}
    assert not _direct_matches(requirement, urls)
    urls['native']['vcs_info']['commit_id'] = 'abc'
    assert _direct_matches(requirement, urls)


def test_managed_metadata_controls_extra_readiness(tmp_path, monkeypatch):
    import json
    import subprocess

    env = Environment(tmp_path)
    env.python.parent.mkdir(parents=True)
    env.python.write_text('managed')
    base, revision = METADRIVE_SOURCE.split('git+')[1].rsplit('@', 1)
    payload = {'installed': [
        {'metadata': {'name': 'ml-stack', 'version': '0.3', 'provides_extra': ['gym-driving'],
                      'requires_dist': []}},
        {'metadata': {'name': 'metadrive-simulator', 'version': '0.4'},
         'direct_url': {'url': base,
                        'vcs_info': {'vcs': 'git', 'commit_id': revision}}}]}
    monkeypatch.setattr(env, 'pip', lambda args, **kwargs: subprocess.CompletedProcess(
        args, 0, stdout=json.dumps(payload)))
    library = next(lib for lib in CATALOG if lib.name == 'gym-driving')
    assert env.has(library)
    payload['installed'][1]['direct_url']['vcs_info']['commit_id'] = 'def'
    assert not env.has(library)


@pytest.mark.parametrize('name', ['gym', 'gym-driving'])
def test_managed_driving_install_supplies_source_outside_package_metadata(tmp_path, monkeypatch, name):
    import subprocess

    env = Environment(tmp_path)
    monkeypatch.setattr(env, 'create', lambda **kwargs: env.python)
    calls = []

    def pip(args, **kwargs):
        calls.append(args)
        return subprocess.CompletedProcess(args, 0, stdout='')

    monkeypatch.setattr(env, 'pip', pip)
    result = env.install([name])
    assert result[name]['ok']
    assert calls == [['install', '--upgrade', f'ml-stack[{name}]', METADRIVE_SOURCE]]


def test_every_published_extra_uses_index_dependencies():
    import tomllib
    from pathlib import Path

    from packaging.requirements import Requirement

    project = tomllib.loads((Path(__file__).parents[1] / 'pyproject.toml').read_text())
    assert not project.get('tool', {}).get('hatch', {}).get('metadata', {}).get('allow-direct-references')
    for requirements in project['project']['optional-dependencies'].values():
        assert all(Requirement(requirement).url is None for requirement in requirements)


def test_built_wheel_metadata_has_no_direct_dependencies():
    from zipfile import ZipFile

    from packaging.requirements import Requirement

    wheels = Environment('.').wheels()
    if wheels is None:
        pytest.skip('build wheels with packaging/build.py')
    for wheel in wheels.glob('ml_stack-*.whl'):
        with ZipFile(wheel) as archive:
            metadata_file = next(name for name in archive.namelist() if name.endswith('.dist-info/METADATA'))
            metadata = archive.read(metadata_file).decode()
        requirements = [line.removeprefix('Requires-Dist: ') for line in metadata.splitlines() if line.startswith('Requires-Dist: ')]
        assert requirements and all(Requirement(requirement).url is None for requirement in requirements)


def test_unavailable_pyenv_shim_is_not_an_environment_builder(tmp_path, monkeypatch):
    import subprocess

    from ml_stack.fleet import environment
    version = '3.12' if sys.version_info[:2] != (3, 12) else '3.13'
    monkeypatch.setattr(environment.shutil, 'which', lambda _name: '/tmp/shims/python')
    monkeypatch.setattr(environment.subprocess, 'run', lambda *args, **kwargs:
                        subprocess.CompletedProcess(args[0], 127, '', 'pyenv: command not found'))
    assert Environment(tmp_path, python_version=version).host_python() is None


def test_python_builder_uses_verified_executable_not_shim(tmp_path, monkeypatch):
    import json
    import subprocess

    from ml_stack.fleet import environment
    version = '3.12' if sys.version_info[:2] != (3, 12) else '3.13'
    monkeypatch.setattr(environment.shutil, 'which', lambda _name: '/tmp/shims/python')
    monkeypatch.setattr(environment.subprocess, 'run', lambda *args, **kwargs:
                        subprocess.CompletedProcess(args[0], 0,
                            json.dumps(['/tmp/real/python', [int(v) for v in version.split('.')]]), ''))
    assert str(Environment(tmp_path, python_version=version).host_python()) == '/tmp/real/python'


def test_installed_pyenv_python_is_reused_when_current_shim_cannot_run(tmp_path, monkeypatch):
    import json
    import subprocess

    from ml_stack.fleet import environment

    version = '3.12' if sys.version_info[:2] != (3, 12) else '3.13'
    monkeypatch.setattr(environment.shutil, 'which', lambda name:
                        '/tmp/pyenv' if name == 'pyenv' else '/tmp/shims/python')

    def run(argv, **_kwargs):
        if argv[1] == 'whence':
            return subprocess.CompletedProcess(argv, 0, '/tmp/installed/python\n', '')
        if argv[0] == '/tmp/shims/python':
            return subprocess.CompletedProcess(argv, 127, '', 'pyenv: command not found')
        return subprocess.CompletedProcess(argv, 0, json.dumps([
            '/tmp/installed/python', [int(v) for v in version.split('.')]]), '')

    monkeypatch.setattr(environment.subprocess, 'run', run)
    assert str(Environment(tmp_path, python_version=version).host_python()) == '/tmp/installed/python'
