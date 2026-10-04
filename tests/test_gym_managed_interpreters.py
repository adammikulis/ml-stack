"""Version-scoped installations, saved interpreter reuse and actual process routing."""

import json
import queue
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from ml_stack.fleet import gym_interpreters
from ml_stack.fleet.environment import CATALOG, Environment, _requirements
from ml_stack.fleet.settings import Settings
from ml_stack.gym.catalog import catalogue
from ml_stack.gym.transport import Process, interpreter


@pytest.fixture
def isolated_routing(monkeypatch):
    monkeypatch.delenv('ML_STACK_GYM_PYTHON', raising=False)
    monkeypatch.setenv('ML_STACK_GYM_PYTHONS', '{}')


def test_child_python_markers_do_not_use_host_python():
    metadata = {'ml-stack': {'provides_extra': ['gym-drone'], 'requires_dist': [
        'PyFlyt==0.29.0; extra == "gym-drone" and python_version < "3.13"']}}
    child = _requirements('ml-stack[gym-drone]', metadata, {'python_version': '3.12'})
    host = _requirements('ml-stack[gym-drone]', metadata, {'python_version': '3.13'})
    assert {requirement.name for requirement in child} == {'ml-stack', 'PyFlyt'}
    assert {requirement.name for requirement in host} == {'ml-stack'}


def test_drone_install_is_scoped_and_does_not_change_default_packages(monkeypatch, tmp_path):
    environment = Environment(tmp_path)
    drone = next(lib for lib in CATALOG if lib.name == 'gym-drone')
    made, calls = [], []
    monkeypatch.setattr(Environment, 'create', lambda self, **kwargs: made.append((self.path, self.python_version)))

    def pip(self, args, **kwargs):
        calls.append((self.path, args))
        return subprocess.CompletedProcess(args, 0, stdout='')

    monkeypatch.setattr(Environment, 'pip', pip)
    assert environment.install(['gym-drone']) == {'gym-drone': {'ok': True}}
    target = environment.for_library(drone)
    assert made == [(tmp_path / 'simulators/gym-drone/env', '3.12')]
    assert calls == [(target.path, ['install', 'numpy<2', 'wheel']),
                     (target.path, ['install', '--upgrade', 'PyFlyt==0.29.0', 'ml-stack[gym-drone,gym-rl]'])]
    assert not environment.exists and environment.python_version == '3.13'


def test_incompatible_version_uses_matching_host_or_standalone(monkeypatch, tmp_path):
    environment = Environment(tmp_path, python_version='3.12')
    monkeypatch.setattr(sys, 'version_info', SimpleNamespace(major=3, minor=13))
    monkeypatch.setattr('ml_stack.fleet.environment.shutil.which', lambda name: None)
    assert environment.host_python() is None
    paths = []
    executable = tmp_path / 'standalone/python'
    monkeypatch.setattr(environment, 'fetch_python', lambda **kwargs: executable)
    monkeypatch.setattr('ml_stack.fleet.environment.subprocess.run',
                        lambda args, **kwargs: paths.append(args) or subprocess.CompletedProcess(args, 0))
    environment.create()
    assert paths == [[str(executable), '-m', 'venv', str(environment.path)]]


def test_scoped_readiness_and_uninstall_preserve_other_environment(monkeypatch, tmp_path):
    environment = Environment(tmp_path)
    drone = next(lib for lib in CATALOG if lib.name == 'gym-drone')
    target = environment.for_library(drone)
    target.python.parent.mkdir(parents=True)
    target.python.write_text('native environment')
    calls = []
    payload = {'environment': {'python_version': '3.12'}, 'installed': [
        {'metadata': {'name': 'PyFlyt', 'version': '0.29.0'}},
        {'metadata': {'name': 'ml-stack', 'version': '0.3', 'provides_extra': ['gym-drone', 'gym-rl'],
                      'requires_dist': ['pillow>=10; extra == "gym-drone" and python_version < "3.13"',
                                        'stable-baselines3>=2.7; extra == "gym-rl"']}},
        {'metadata': {'name': 'pillow', 'version': '12.0'}},
        {'metadata': {'name': 'stable-baselines3', 'version': '2.7'}}]}

    def pip(self, args, **kwargs):
        calls.append((self.path, args))
        return subprocess.CompletedProcess(args, 0, stdout=json.dumps(payload))

    monkeypatch.setattr(Environment, 'pip', pip)
    assert environment.has(drone)
    state = environment.state()
    entry = next(row for row in state['libraries'] if row['name'] == 'gym-drone')
    assert entry['installed'] and entry['python_version'] == '3.12'
    assert environment.uninstall(['gym-drone'])['gym-drone']['ok']
    assert all(path == target.path for path, _ in calls)
    assert calls[-1][1] == ['uninstall', '-y', 'ml-stack', 'pillow', 'pyflyt', 'stable-baselines3']


def test_managed_scope_is_rediscovered_after_settings_reload(isolated_routing, tmp_path):
    environment = Environment(tmp_path)
    target = environment.for_library(next(lib for lib in CATALOG if lib.name == 'gym-drone'))
    target.python.parent.mkdir(parents=True)
    target.python.write_text('installed Python')
    settings_file = tmp_path / 'settings.json'
    Settings(setup_done=True).save(settings_file)
    settings = Settings.load(settings_file)
    gym_interpreters.configure_interpreters(SimpleNamespace(settings=settings, environment=environment))
    assert interpreter('drone') == str(target.python)
    assert interpreter('warehouse') == sys.executable
    assert settings.setup_done and not settings.gym_pythons


def test_custom_per_example_python_persists_and_precedes_managed(isolated_routing, tmp_path):
    custom = tmp_path / 'custom Python'
    custom.write_text('installed Python')
    path = tmp_path / 'settings.json'
    Settings(setup_done=True, gym_pythons={'drone': str(custom)}).save(path)
    gym_interpreters.configure_interpreters(SimpleNamespace(settings=Settings.load(path), environment=None))
    assert interpreter('drone') == str(custom)
    assert interpreter('car') == sys.executable


@pytest.mark.skipif(sys.platform == 'win32', reason='Executable fixture uses a POSIX shebang')
def test_catalogue_probes_chosen_drone_python_without_recursive_mapping(isolated_routing, monkeypatch, tmp_path):
    python = tmp_path / 'native Python'
    python.write_text(f'#!{sys.executable}\nimport json,os\n'
                      'assert "ML_STACK_GYM_PYTHON" not in os.environ\n'
                      'assert "ML_STACK_GYM_PYTHONS" not in os.environ\n'
                      'print(json.dumps([{"id":"drone","available":True,"library":"native fixture"}]))\n')
    python.chmod(0o755)
    monkeypatch.setenv('ML_STACK_GYM_PYTHONS', json.dumps({'drone': str(python)}))
    entries = catalogue()
    drone = next(row for row in entries if row['id'] == 'drone')
    assert drone['available'] and drone['interpreter'] == str(python)
    assert next(row for row in entries if row['id'] == 'car')['library'] == 'MetaDrive'


@pytest.mark.skipif(sys.platform == 'win32', reason='Executable fixture uses a POSIX shebang')
def test_worker_launch_uses_saved_drone_python_without_parent_routes(isolated_routing, tmp_path):
    python = tmp_path / 'worker Python'
    python.write_text(f'#!{sys.executable}\nimport json,os,sys\n'
                      'assert "ML_STACK_GYM_PYTHON" not in os.environ\n'
                      'assert "ML_STACK_GYM_PYTHONS" not in os.environ\n'
                      'settings=json.loads(sys.argv[-1])\n'
                      'print(json.dumps({"environment":settings["environment"],"status":"ready"}),flush=True)\n')
    python.chmod(0o755)
    path = tmp_path / 'settings.json'
    Settings(setup_done=True, gym_pythons={'drone': str(python)}).save(path)
    gym_interpreters.configure_interpreters(SimpleNamespace(settings=Settings.load(path), environment=None))
    updates = queue.Queue()
    with (tmp_path / 'worker.log').open('w') as log:
        process = Process({'environment': 'drone'}, updates, log)
        process.start()
        try:
            snapshot = updates.get(timeout=5)
            assert snapshot == {'environment': 'drone', 'status': 'ready'}
            process.join(5)
        finally:
            if process.is_alive():
                process.terminate()
            process.commands.close()


@pytest.mark.slow
def test_installed_native_drone_relaunches_from_saved_settings(isolated_routing, monkeypatch, tmp_path):
    import os
    from pathlib import Path

    from ml_stack.gym.runtime import SessionManager
    configured = os.environ.get('ML_STACK_TEST_DRONE_PYTHON')
    if not configured or not Path(configured).is_file():
        pytest.skip('Set ML_STACK_TEST_DRONE_PYTHON to a native PyFlyt interpreter')
    monkeypatch.setenv('ML_STACK_CACHE', str(tmp_path / 'cache'))
    settings_path = tmp_path / 'settings.json'
    Settings(setup_done=True, gym_pythons={'drone': configured}).save(settings_path)
    for _ in range(2):
        settings = Settings.load(settings_path)
        gym_interpreters.configure_interpreters(SimpleNamespace(settings=settings, environment=None))
        manager = SessionManager()
        try:
            state = manager.create('drone', {'simulation_mode': 'world', 'world': {'trees': 4}},
                                   controller='native-patrol', seed=2)
            identifier = state['id']
            deadline = time.monotonic() + 20
            while state['status'] == 'starting' and time.monotonic() < deadline:
                time.sleep(.02)
                state = manager.get(identifier)
            assert state['status'] != 'error', state
            assert state['info']['native_pid_mode'] == 7
            initial_sequence = state['sequence']
            manager.control(identifier, 'step')
            while state['sequence'] == initial_sequence and time.monotonic() < deadline:
                time.sleep(.02)
                state = manager.get(identifier)
            assert state['sequence'] == initial_sequence + 1
            assert state['info']['world_time'] == pytest.approx(.2)
            assert state['decision']['model'] == 'PyFlyt PID patrol'
            assert settings.setup_done
        finally:
            manager.close_all()


def test_macos_bullet_build_selects_sdk_without_shell(monkeypatch, tmp_path):
    import os
    monkeypatch.setattr(sys, 'platform', 'darwin')
    monkeypatch.delenv('SDKROOT', raising=False)
    sdk = tmp_path / 'SDK with spaces'
    (sdk / 'usr/include').mkdir(parents=True)
    (sdk / 'usr/include/math.h').write_text('system header')
    calls = []

    def run(args, **kwargs):
        calls.append((args, kwargs))
        return subprocess.CompletedProcess(args, 0, stdout=str(sdk) + '\n')

    monkeypatch.setattr('ml_stack.fleet.environment.subprocess.run', run)
    environment = Environment(tmp_path, python_version='3.12').build_environment()
    assert calls[0][0] == ['xcrun', '--show-sdk-path']
    assert not calls[0][1].get('shell', False)
    assert environment['SDKROOT'] == str(sdk)
    assert '-Dfdopen=fdopen' in environment['CFLAGS']
    assert str(sdk) in environment['CXXFLAGS']
    assert os.environ.get('SDKROOT') is None
