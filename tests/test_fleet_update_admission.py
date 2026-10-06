import threading
from types import SimpleNamespace

import pytest

from ml_stack.fleet import updates
from ml_stack.fleet.daemon import DaemonRuntime
from ml_stack.fleet.setup_jobs import Jobs
from ml_stack.lock import only_one


@pytest.mark.parametrize('kind', ['release', 'branch'])
def test_update_holds_admission_through_restart(tmp_path, monkeypatch, kind):
    jobs = Jobs(tmp_path)
    entered, release = threading.Event(), threading.Event()
    calls = []

    def install(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        calls.append('installed')
        if kind == 'release':
            return {'installed': True}
        assert rejected()
        return updates.Pulled('dev', restarted='restarted')

    def rejected():
        with pytest.raises(ValueError, match='runtime update'):
            jobs.start('libraries', {}, lambda progress: calls.append('orphan'))
        return True

    def restart():
        assert rejected()
        calls.append('restarted')
        return True

    def admission():
        return only_one(tmp_path / 'runtime-install.lock', wait=False)
    schedule = updates.UpdateSchedule(interval=0, first_after_s=0, rounds=1)
    if kind == 'release':
        monkeypatch.setattr(updates, 'apply_if_newer', install)
        thread = updates.watch(wanted=lambda: True, idle=lambda: not jobs.active(), restart=restart,
                               admission=admission, schedule=schedule)
    else:
        monkeypatch.setattr(updates, 'track_once', install)
        thread = updates.track(updates.TrackedBranch('https://example.test/repo', 'dev', tmp_path),
                               idle=lambda: not jobs.active(), runtime=updates.UpdateRuntime(admission=admission),
                               schedule=schedule)
    assert entered.wait(5)
    assert rejected()
    assert not jobs.all()
    release.set()
    thread.join(5)
    assert not thread.is_alive()
    assert 'installed' in calls and 'orphan' not in calls


@pytest.mark.parametrize('kind', ['release', 'branch'])
def test_queued_jobs_prevent_update_and_admit_more_jobs(tmp_path, monkeypatch, kind):
    jobs = Jobs(tmp_path)
    entered, release = threading.Event(), threading.Event()

    def install(progress):
        entered.set()
        assert release.wait(5)
        return {}

    jobs.start('libraries', {'add': 'one'}, install)
    assert entered.wait(5)
    second = jobs.start('libraries', {'add': 'two'}, lambda progress: {})
    assert second['state'] == 'queued'
    calls = []
    monkeypatch.setattr(updates, 'apply_if_newer', lambda: calls.append('release'))
    monkeypatch.setattr(updates, 'track_once', lambda *a, **k: calls.append('branch'))
    def admission():
        return only_one(tmp_path / 'runtime-install.lock', wait=False)
    schedule = updates.UpdateSchedule(interval=0, first_after_s=0, rounds=1)
    if kind == 'release':
        thread = updates.watch(wanted=lambda: True, idle=lambda: not jobs.active(), admission=admission, schedule=schedule)
    else:
        thread = updates.track(updates.TrackedBranch('https://example.test/repo', 'dev', tmp_path),
                               idle=lambda: not jobs.active(), runtime=updates.UpdateRuntime(admission=admission), schedule=schedule)
    thread.join(5)
    assert not calls
    release.set()
    jobs._worker.join(5)
    assert not jobs.active()


@pytest.mark.parametrize('tracked', [False, True])
def test_daemon_wires_shared_update_admission(tmp_path, monkeypatch, tracked):
    daemon = object.__new__(DaemonRuntime)
    daemon.root = tmp_path
    daemon.settings = SimpleNamespace(track_branch='dev' if tracked else '', track_repo='', auto_update=True, setup_done=False)
    daemon.runner = SimpleNamespace(status=lambda: {'busy': False})
    daemon.downloads = SimpleNamespace(active=lambda: [])
    daemon.interface = None
    daemon.web = True
    daemon.initial_setup = True
    daemon.bench_host = SimpleNamespace(measuring=lambda: False)
    daemon.serving = SimpleNamespace(live=lambda: [])
    monkeypatch.setattr(updates, 'checkout_here', lambda: tmp_path)
    captured = {}
    monkeypatch.setattr(updates, 'track', lambda source, **kwargs: captured.update(kwargs))
    monkeypatch.setattr(updates, 'watch', lambda **kwargs: captured.update(kwargs))
    daemon.updates()
    assert not captured['idle']()
    daemon.settings.setup_done = True
    assert captured['idle']()
    admission = captured['runtime'].admission if tracked else captured['admission']
    with admission(), pytest.raises(ValueError, match='runtime update'):
        Jobs(tmp_path).start('libraries', {}, lambda progress: {})
    daemon.initial_setup = False
    daemon.settings.setup_done = False
    assert captured['idle']()


@pytest.mark.parametrize('kind', ['release', 'branch'])
def test_update_defers_when_admission_is_held(tmp_path, monkeypatch, kind):
    calls = []
    monkeypatch.setattr(updates, 'apply_if_newer', lambda: calls.append('release'))
    monkeypatch.setattr(updates, 'track_once', lambda *a, **k: calls.append('branch'))

    def admission():
        return only_one(tmp_path / 'runtime-install.lock', wait=False)

    schedule = updates.UpdateSchedule(interval=0, first_after_s=0, rounds=1)
    with admission():
        if kind == 'release':
            thread = updates.watch(wanted=lambda: True, idle=lambda: True, admission=admission, schedule=schedule)
        else:
            thread = updates.track(updates.TrackedBranch('https://example.test/repo', 'dev', tmp_path),
                                   runtime=updates.UpdateRuntime(admission=admission), schedule=schedule)
        thread.join(5)
        assert not thread.is_alive()
    assert not calls


@pytest.mark.parametrize("announce", [False, True])
def test_leaving_last_cluster_stops_discovery_and_rotates_auth(tmp_path, monkeypatch, announce):
    from ml_stack import macauth
    from ml_stack.fleet import daemon as module
    from ml_stack.fleet.discovery import derive_token

    runtime = object.__new__(DaemonRuntime)
    old_key = b'k' * 32
    runtime.root = tmp_path
    runtime.cluster_key_path = tmp_path / 'cluster.key'
    runtime.key = old_key
    runtime.token = module.load_or_create_token(tmp_path, old_key, profile=runtime.cluster_key_path)
    stopped = []
    runtime.advertisers = {'old': SimpleNamespace(stop=lambda: stopped.append(True))}
    runtime.advertiser = runtime.advertisers['old']
    runtime.announcement_lock = threading.RLock()
    runtime.announce = announce
    runtime.fetcher = SimpleNamespace(key=old_key)
    runtime.settings = SimpleNamespace(cluster_mode='dev')
    runtime.daemon = SimpleNamespace(cluster_mode='dev')
    monkeypatch.setattr(module, 'memberships', lambda path: [])
    runtime.joined_a_cluster()
    assert stopped == [True] and runtime.advertisers == {}
    assert runtime.key is None and runtime.fetcher.key is None and runtime.advertiser is None
    assert runtime.token != derive_token(old_key)
    assert runtime.token.startswith(macauth.PREFIX)
    assert module.load_or_create_token(tmp_path, profile=runtime.cluster_key_path) == runtime.token
    assert runtime.every_token() == set()
    auth = macauth.Authenticator(lambda: {runtime.token, *runtime.every_token()})
    url = "https://device.test/status"
    headers = {"Host": "device.test", **macauth.sign(derive_token(old_key), "GET", url, None)}
    assert not auth.check("GET", "/status", headers, None).ok
    headers = {"Host": "device.test", **macauth.sign(runtime.token, "GET", url, None)}
    assert auth.check("GET", "/status", headers, None).ok



def test_unannounced_cluster_change_updates_machine_auth(tmp_path, monkeypatch):
    from ml_stack.fleet import daemon as module
    from ml_stack.fleet.discovery import derive_token

    runtime = object.__new__(DaemonRuntime)
    runtime.root = tmp_path
    runtime.cluster_key_path = tmp_path / 'cluster.key'
    runtime.key = b'a' * 32
    runtime.token = derive_token(runtime.key)
    runtime.fetcher = SimpleNamespace(key=runtime.key)
    runtime.advertiser = None
    runtime.advertisers = {}
    runtime.announcement_lock = threading.RLock()
    runtime.announce = False
    member = SimpleNamespace(group='new', key=b'b' * 32)
    monkeypatch.setattr(module, 'memberships', lambda path: [member])
    runtime.start_announcing()
    assert runtime.key == member.key and runtime.fetcher.key == member.key
    assert runtime.token == derive_token(member.key)
    assert runtime.advertiser is None and not runtime.advertisers
