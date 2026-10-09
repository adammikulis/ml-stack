"""Registry persistence grants only the current OS account access."""

import json
import sys

import pytest

from poolhouse import private_path
from poolhouse.workspace.identity import AGENT, Denied, Registry


def test_registry_creation_and_replacement_are_private_and_hash_only(tmp_path):
    registry = Registry(tmp_path / 'workspace')
    credential = registry._add('local-account', 'agent', AGENT, 0)
    assert private_path.problem(registry.path.parent) == ''
    assert private_path.problem(registry.path) == ''
    assert credential.split('.', 2)[-1] not in registry.path.read_text()
    registry.record_model('agent', 'claimed-model', 'codex', 'claimed')
    assert private_path.problem(registry.path) == ''
    assert registry.authenticate(credential).id == 'agent'


def test_failed_registry_serialization_preserves_private_file(tmp_path):
    registry = Registry(tmp_path / 'workspace')
    registry._save({})
    before = registry.path.read_bytes()
    with pytest.raises(TypeError):
        registry._save({'invalid': object()})
    assert registry.path.read_bytes() == before
    assert private_path.problem(registry.path) == ''
    assert sorted(path.name for path in registry.path.parent.iterdir()) == ['agents.json']


@pytest.mark.skipif(sys.platform != 'win32', reason='native Windows DACL')
def test_owned_registry_with_inherited_windows_access_is_restricted_on_save(tmp_path):
    from poolhouse import windows_private
    registry = Registry(tmp_path / 'workspace')
    registry.path.parent.mkdir()
    registry.path.write_text(json.dumps({'version': 2, 'agents': {}}))
    assert windows_private.problem(registry.path)
    registry._save({})
    assert windows_private.problem(registry.path.parent) == ''
    assert windows_private.problem(registry.path) == ''


@pytest.mark.skipif(sys.platform != 'win32', reason='native Windows junction')
@pytest.mark.parametrize('nested', [False, True])
def test_registry_junction_is_refused_before_target_changes(tmp_path, nested):
    import _winapi
    target, link = tmp_path / 'target', tmp_path / 'link'
    target.mkdir()
    path = target / 'agents.json'
    path.write_text('unchanged')
    _winapi.CreateJunction(str(target), str(link))
    try:
        with pytest.raises(Denied, match=r'redirected|ownership could not be verified'):
            Registry(link / 'new-workspace' if nested else link)._save({})
        assert path.read_text() == 'unchanged'
        assert not (target / 'new-workspace').exists()
    finally:
        link.rmdir()


@pytest.mark.skipif(sys.platform != 'win32', reason='native Windows ownership')
def test_foreign_owned_registry_is_refused_before_replacement(tmp_path):
    import pywintypes
    import win32security

    from poolhouse import windows_private
    registry = Registry(tmp_path / 'workspace')
    registry._save({})
    before = registry.path.read_bytes()
    original = windows_private._user()
    foreign = win32security.CreateWellKnownSid(win32security.WinBuiltinAdministratorsSid, None)
    try:
        win32security.SetNamedSecurityInfo(str(registry.path), win32security.SE_FILE_OBJECT,
                                          win32security.OWNER_SECURITY_INFORMATION, foreign, None, None, None)
    except pywintypes.error as error:
        if error.winerror in (1307, 1314):
            pytest.skip('this test account cannot assign a different Windows file owner')
        raise
    try:
        assert 'belongs to another user' in windows_private.problem(registry.path)
        with pytest.raises(Denied, match='ownership'):
            registry._save({})
        assert registry.path.read_bytes() == before
    finally:
        win32security.SetNamedSecurityInfo(str(registry.path), win32security.SE_FILE_OBJECT,
                                          win32security.OWNER_SECURITY_INFORMATION, original, None, None, None)
