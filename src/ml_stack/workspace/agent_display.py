"""Readable agent presentation, independent of capability and routing identity."""
from ml_stack.workspace.identity import AGENT, CAPS
from ml_stack.workspace.model_tiers import LOWEST, tier_of

SPAWN_NOTICE = ("no main session is eligible to coordinate and only lowest-tier sessions are present: "
                "spawn a subagent at a suitable level (Sonnet 5.5 is acceptable) to coordinate")


def family(model: str) -> str:
    """Return a readable family from a recorded exact model identifier."""
    model = model.lower().rsplit("/", 1)[-1]
    if model.startswith("claude-"):
        return "Claude"
    if model.startswith(("gpt-", "chatgpt-", "o1", "o3", "o4")):
        return "ChatGPT"
    if model == "thinkingcap-qwen3.8-27b" or model.startswith("qwen"):
        return "Qwen"
    return "Model unknown"


def metadata(registry, name, label=''):
    """Derive parentage from the registry, never from arbitrary activity text."""
    info = registry.info(name)
    if info['role'] != AGENT:
        return {'display_name': name, 'session_kind': 'person' if info['role'] == 'human' else 'unknown',
                'coordinator_eligible': False, 'coordinator_reason': 'not an agent session'}
    if not info['parent'] and not info.get('presentation', {}).get('ordinal'):
        registry.ensure_presentation(name)
        info = registry.info(name)
    parent = info['parent']
    if parent:
        parent_name = metadata(registry, parent)['display_name']
        display = f"Subagent · {name.rpartition('/')[2]} (parent {parent_name})"
        kind = 'subagent'
    else:
        presentation = info.get('presentation', {})
        model_family = family(info.get('model', ''))
        device = info.get('device', {})
        system = {'macOS': 'Mac', 'Darwin': 'Mac'}.get(device.get('os'), device.get('os'))
        ordinal = presentation.get('ordinal')
        suffix = f"session {ordinal}" if ordinal else 'unregistered session'
        display = f"{model_family} · {system or device.get('hostname') or 'device unknown'} · {suffix}"
        kind = presentation.get('kind', 'unknown')
    if parent or kind != 'main':
        reason = 'not a main session'
    elif registry.role_of(name) != AGENT or not set(CAPS) <= set(info['can']):
        reason = 'lacks the agent capabilities'
    else:
        reason = tier_of(*registry.model_of(name)).reason
    eligible = not reason
    if label:
        display = f"{display} ({label})"
        kind = 'helper' if label in info.get('label_models', {}) else kind
        eligible, reason = False, 'a helper label is not a main session'
    return {'display_name': display, 'session_kind': kind, 'coordinator_eligible': eligible,
            'coordinator_reason': reason}


def vacancy_notice(rows) -> str:
    """What to do when main sessions exist and none may coordinate because all are lowest tier; empty otherwise."""
    mains = [row for row in rows if row['session_kind'] == 'main' and not row['parent']]
    if mains and not any(row['coordinator_eligible'] for row in mains) \
            and any(row['coordinator_reason'] == LOWEST for row in mains):
        return SPAWN_NOTICE
    return ''
