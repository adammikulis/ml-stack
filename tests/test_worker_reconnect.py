"""Worker transport recovery preserves authorization and mutation boundaries."""

import errno
from types import SimpleNamespace

import pytest

from poolhouse.http import ServerError
from poolhouse.workspace import worker_reconnect as recovery
from poolhouse.workspace.identity import Denied


def outage(status):
    try:
        raise ServerError('fixture outage', status=status)
    except ServerError as cause:
        error = Denied('project board unavailable')
        error.__cause__ = cause
        return error


def recovering(entries, monkeypatch, *, stopped=lambda: False, window_s=1):
    calls, states = [], []
    def call(operation, token, *args, **kwargs):
        calls.append(operation)
        result = entries.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result
    monkeypatch.setattr(recovery, 'POLL_S', 0)
    remote = recovery.Recovering(SimpleNamespace(call=call), stopped,
                                SimpleNamespace(update=lambda **fields: states.append(fields)),
                                window_s=window_s)
    return remote, calls, states


@pytest.mark.parametrize('failure', [outage(503), outage(502), outage(504),
                                     OSError(errno.ECONNRESET, 'connection reset')])
def test_worker_reads_resume_after_a_transport_outage(monkeypatch, failure):
    remote, calls, states = recovering([failure, {'identity': 'worker'}], monkeypatch)
    assert remote.call('whoami', 'private-fixture-token') == {'identity': 'worker'}
    assert calls == ['whoami', 'whoami'] and states[0]['state'] == 'reconnecting'


@pytest.mark.parametrize('operation', ['whoami', 'ack', 'send', 'claim_model', 'task_command'])
def test_authorization_denial_never_retries(monkeypatch, operation):
    remote, calls, _ = recovering([outage(403), 'unexpected retry'], monkeypatch)
    with pytest.raises(Denied):
        remote.call(operation, 'revoked-fixture-token')
    assert calls == [operation]


def test_arbitrary_mutation_is_not_replayed_after_lost_response(monkeypatch):
    remote, calls, _ = recovering([outage(503), 'unexpected duplicate'], monkeypatch)
    with pytest.raises(Denied):
        remote.call('send', 'private-fixture-token', 'lead', 'answer', 'done')
    assert calls == ['send']


def test_reconnect_exhaustion_is_distinct_from_bad_credentials(monkeypatch):
    remote, calls, _ = recovering([outage(503)], monkeypatch, window_s=0)
    with pytest.raises(recovery.Unavailable, match='bounded reconnect window'):
        remote.call('wait', 'private-fixture-token')
    assert calls == []


def test_cancellation_interrupts_reconnection(monkeypatch):
    cancelled = [False]
    failure = outage(503)
    def stopped():
        if cancelled[0]:
            return True
        cancelled[0] = True
        return False
    remote, calls, _ = recovering([failure], monkeypatch, stopped=stopped)
    with pytest.raises(recovery.Cancelled):
        remote.call('wait', 'private-fixture-token')
    assert calls == ['wait']



def test_shared_absolute_deadline_does_not_start_a_fresh_reconnect_window(monkeypatch):
    remote, calls, _ = recovering(['unexpected call'], monkeypatch)
    monkeypatch.setattr(recovery.time, 'monotonic', lambda: 100.0)
    with pytest.raises(recovery.Unavailable):
        remote.call('outbox', 'private-fixture-token', _reconnect_deadline=99.0)
    assert calls == []


def test_internal_deadline_is_not_forwarded_to_the_authenticated_protocol(monkeypatch):
    observed = []
    transport = SimpleNamespace(call=lambda operation, token, **kwargs: observed.append(kwargs) or [])
    remote = recovery.Recovering(transport, lambda: False, SimpleNamespace(update=lambda **fields: None))
    remote.call('outbox', 'private-fixture-token', limit=100, _reconnect_deadline=recovery.time.monotonic() + 1)
    assert observed == [{'limit': 100}]
