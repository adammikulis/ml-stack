"""Background setup jobs persist state and release HTTP callers."""
import threading
import time
from types import SimpleNamespace

from poolhouse.fleet import setup_jobs
from poolhouse.fleet.routes import ModelRoutes, SettingsRoutes


def wait_job(jobs, state):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        rows = jobs.all()
        if rows and rows[-1]['state'] == state:
            return rows[-1]
        time.sleep(.02)
    raise AssertionError(jobs.all())


def test_install_returns_before_operation_finishes_and_deduplicates(tmp_path):
    jobs = setup_jobs.Jobs(tmp_path)
    entered, release = threading.Event(), threading.Event()
    def install(progress):
        entered.set()
        assert release.wait(5)
        progress('Extracting runtime')
        return {'server': 'managed/server'}
    queued = jobs.start('server', {}, install)
    assert queued['state'] == 'queued'
    assert entered.wait(5)
    assert jobs.start('server', {}, install)['id'] == queued['id']
    assert jobs.all()[0]['state'] == 'installing'
    assert jobs.active()
    release.set()
    assert wait_job(jobs, 'done')['result']['server'] == 'managed/server'
    assert not jobs.active()
    assert setup_jobs.Jobs(tmp_path).all()[0]['state'] == 'done'


def test_failed_library_is_durable_and_retry_succeeds(tmp_path):
    jobs = setup_jobs.Jobs(tmp_path)
    first = jobs.start('libraries', {'install': ['core']}, lambda progress: {'changed': {'core': {'ok': False, 'error': 'offline'}}})
    assert wait_job(jobs, 'failed')['error'] == 'offline'
    second = jobs.start('libraries', {'install': ['core']}, lambda progress: {'changed': {'core': {'ok': True}}})
    assert second['id'] != first['id']
    assert wait_job(jobs, 'done')['result']['changed']['core']['ok']


def test_crashed_installer_is_reported_for_retry(tmp_path, monkeypatch):
    jobs = setup_jobs.Jobs(tmp_path)
    row = {'id': 'interrupted', 'kind': 'server', 'state': 'installing', 'pid': 987654, 'born': 0, 'created': 1}
    with jobs._store() as graph:
        jobs._save(graph, row)
    monkeypatch.setattr(setup_jobs, 'pid_exists', lambda pid: False)
    recovered = setup_jobs.Jobs(tmp_path).all()[0]
    assert recovered['state'] == 'failed'
    assert 'interrupted' in recovered['error']


def test_library_route_queues_without_reporting_installed(tmp_path):
    entered, release = threading.Event(), threading.Event()
    def install(names, on_progress):
        entered.set()
        assert release.wait(5)
        return {'core': {'ok': True}}
    ui = SimpleNamespace(root=tmp_path, settings=None, setup_jobs=None, _setup_jobs_lock=threading.Lock(), environment=SimpleNamespace(install=install, state=lambda vendor: {'libraries': [{'name': 'core', 'installed': True}]}))
    route = SettingsRoutes()
    route.ui = ui
    route.body = lambda: {'install': ['core']}
    replies = []
    route.send = lambda status, body: replies.append((status, body))
    assert route._change_libraries('cpu')
    assert replies[0][0] == 202
    assert 'libraries' not in replies[0][1]
    assert entered.wait(5)
    release.set()
    assert wait_job(ui.setup_jobs, 'done')['result']['libraries'][0]['installed']


def test_server_route_queues_progress_and_failure(tmp_path, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    def ensure(root, on_progress, sources):
        on_progress('Downloading runtime')
        entered.set()
        assert release.wait(5)
        raise RuntimeError('download interrupted')
    from poolhouse.fleet import llama, routes
    monkeypatch.setattr(llama, 'ensure_server', ensure)
    monkeypatch.setattr(routes, '_can_serve', lambda: True)
    route = ModelRoutes()
    route.ui = SimpleNamespace(root=tmp_path, settings=None, setup_jobs=None, _setup_jobs_lock=threading.Lock())
    replies = []
    route.send = lambda status, body: replies.append((status, body))
    assert route._install_server()
    assert replies[0][0] == 202
    assert entered.wait(5)
    assert route.ui.setup_jobs.all()[0]['note'] == 'Downloading runtime'
    release.set()
    assert wait_job(route.ui.setup_jobs, 'failed')['error'] == 'download interrupted'


def test_direct_environment_callers_share_the_install_lock(tmp_path, monkeypatch):
    from poolhouse.fleet.environment import Environment
    first, release, second = threading.Event(), threading.Event(), threading.Event()
    seen = []
    def install(self, names, on_progress=None):
        seen.append(names)
        if len(seen) == 1:
            first.set()
            assert release.wait(5)
        else:
            second.set()
        return {}
    monkeypatch.setattr(Environment, '_install', install)
    one = threading.Thread(target=lambda: Environment(tmp_path).install(['core']))
    two = threading.Thread(target=lambda: Environment(tmp_path).install(['core']))
    one.start()
    assert first.wait(5)
    two.start()
    assert not second.wait(.1)
    release.set()
    one.join(5)
    two.join(5)
    assert second.is_set()


def test_queue_and_history_are_bounded(tmp_path, monkeypatch):
    monkeypatch.setattr(setup_jobs, 'LIMIT', 2)
    jobs = setup_jobs.Jobs(tmp_path)
    entered, release = threading.Event(), threading.Event()
    def blocking(progress):
        entered.set()
        assert release.wait(5)
        return {}
    first = jobs.start('server', {'number': 1}, blocking, provenance={'channel': 'local-ui'})
    assert entered.wait(5)
    jobs.start('server', {'number': 2}, lambda progress: {})
    import pytest
    with pytest.raises(ValueError, match='queue is full'):
        jobs.start('server', {'number': 3}, lambda progress: {})
    release.set()
    wait_job(jobs, 'done')
    jobs.start('server', {'number': 3}, lambda progress: {})
    rows = jobs.all()
    assert len(rows) == 2
    assert first['id'] not in {row['id'] for row in rows}
    wait_job(jobs, 'done')


def test_unexpected_install_failure_is_terminal_and_next_job_runs(tmp_path):
    jobs = setup_jobs.Jobs(tmp_path)
    def broken(progress):
        raise TypeError('invalid installer response')
    first = jobs.start('server', {'number': 1}, broken)
    second = jobs.start('server', {'number': 2}, lambda progress: {'ok': True})
    assert wait_job(jobs, 'done')['id'] == second['id']
    failed = next(row for row in jobs.all() if row['id'] == first['id'])
    assert failed['state'] == 'failed'
    assert failed['error'] == 'invalid installer response'


def test_queued_libraries_respect_changed_download_policy():
    import pytest
    ui = SimpleNamespace(settings=SimpleNamespace(download_sources='lan'), environment=SimpleNamespace(install=lambda *a, **k: (_ for _ in ()).throw(AssertionError('Internet install must not start'))))
    with pytest.raises(RuntimeError, match='Internet only or Both'):
        setup_jobs.libraries(ui, 'cpu', ['core'], [], lambda note: None)
