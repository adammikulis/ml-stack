"""Readable agent presentation, independent of capability and routing identity."""
from ml_stack.workspace.identity import AGENT, CAPS


def metadata(registry, name, label=''):
    """Derive parentage from the registry, never from arbitrary activity text."""
    registry.ensure_presentation(name)
    info = registry.info(name)
    if info['role'] != AGENT:
        return {'display_name': name, 'session_kind': 'person' if info['role'] == 'human' else 'unknown',
                'coordinator_eligible': False}
    parent = info['parent']
    if parent:
        parent_name = metadata(registry, parent)['display_name']
        display = f"Subagent · {name.rpartition('/')[2]} (parent {parent_name})"
        kind = 'subagent'
    else:
        presentation = info.get('presentation', {})
        family = 'Codex' if name == 'codex' or name.startswith('codex-') else name
        device = info.get('device', {})
        system = {'macOS': 'Mac', 'Darwin': 'Mac'}.get(device.get('os'), device.get('os'))
        ordinal = presentation.get('ordinal')
        suffix = f"session {ordinal}" if ordinal else 'unregistered session'
        display = f"{family} · {system or device.get('hostname') or 'device unknown'} · {suffix}"
        kind = presentation.get('kind', 'unknown')
    eligible = bool(not parent and kind == 'main' and registry.role_of(name) == AGENT
                    and set(CAPS) <= set(info['can']))
    if label:
        if label in info.get('label_models', {}):
            display = f"Subagent · {label} (parent {display})"
            kind = 'helper'
        else:
            display += f" · activity {label}"
        eligible = False
    return {'display_name': display, 'session_kind': kind, 'coordinator_eligible': eligible}
