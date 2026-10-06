"""Automatic coordinator hosting for authenticated local device agents."""

from ml_stack.workspace import coordinator_client, coordinator_config, device_agent
from ml_stack.workspace.chain import held
from ml_stack.workspace.coordination import workspace_id
from ml_stack.workspace.identity import Denied


def ensure_host(ws, token):
    """Host an existing protected local workspace under its own agent session."""
    with held(ws.base / 'coordinator-selection.lock'):
        return _ensure_host(ws, token)


def _ensure_host(ws, token):
    device_agent.owned_local(ws, token)
    config = coordinator_config.load(ws.base)
    if config.get('mode') == 'remote':
        raise Denied('this device follows an existing coordinator authority')
    if config:
        return config
    if coordinator_client._client(ws.base) is not None:
        raise Denied('this device has an existing shared coordinator authority')
    return coordinator_config.save(ws.base, {'mode': 'host', 'workspace': workspace_id(ws)})


