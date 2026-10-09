"""Readable agent presentation, independent of capability and routing identity."""
import hashlib
import re

from ml_stack.workspace import session_name
from ml_stack.workspace.identity import AGENT, CAPS
from ml_stack.workspace.model_tiers import LOWEST, tier_of

SPAWN_NOTICE = ("no main session is eligible to coordinate and only lowest-tier sessions are present: "
                "spawn a subagent at a suitable level (Sonnet 5.5 is acceptable) to coordinate")


SESSION = re.compile(r"(claude|chatgpt|qwen|agent)-[0-9a-f]{6,64}")


def readable_name(registry, name: str, model: str) -> str:
    """The one name a human reads for ``name``: model family plus a suffix of its whole identity.

    A session id from `session_name` is already that name. Any other id gets the same shape, its
    suffix cut from the hash of the id and lengthened while another id shares it."""
    if SESSION.fullmatch(name):
        return name
    full = hashlib.sha256(name.encode()).hexdigest()
    others = [hashlib.sha256(other.encode()).hexdigest() for other in registry.ids() if other != name]
    width = session_name.SHORT
    while width < len(full) and any(other[:width] == full[:width] for other in others):
        width += 2
    return f"{session_name.family_word(model)}-{full[:width]}"


def metadata(registry, name):
    """Derive parentage from the registry, never from arbitrary activity text."""
    info = registry.info(name)
    if info['role'] != AGENT:
        return {'display_name': name, 'session_kind': 'person' if info['role'] == 'human' else 'unknown',
                'coordinator_eligible': False, 'coordinator_reason': 'not an agent session'}
    if not info['parent'] and not info.get('presentation', {}).get('ordinal'):
        registry.ensure_presentation(name)
        info = registry.info(name)
    parent = info['parent']
    display = readable_name(registry, name, info.get('model', ''))
    kind = 'subagent' if parent else info.get('presentation', {}).get('kind', 'unknown')
    if parent or kind != 'main':
        reason = 'not a main session'
    elif registry.role_of(name) != AGENT or not set(CAPS) <= set(info['can']):
        reason = 'lacks the agent capabilities'
    else:
        reason = tier_of(*registry.model_of(name)).reason
    eligible = not reason
    return {'display_name': display, 'session_kind': kind, 'coordinator_eligible': eligible,
            'coordinator_reason': reason, 'spawned_by': parent}


def spoken(shown: dict) -> str:
    """How a person reads an agent: its unique name, and who spawned it when a subagent."""
    return f"{shown['display_name']} (spawned by {shown['spawned_by']})" if shown.get('spawned_by') else shown['display_name']


def vacancy_notice(rows) -> str:
    """What to do when main sessions exist and none may coordinate because all are lowest tier; empty otherwise."""
    mains = [row for row in rows if row['session_kind'] == 'main' and not row['parent']]
    if mains and not any(row['coordinator_eligible'] for row in mains) \
            and any(row['coordinator_reason'] == LOWEST for row in mains):
        return SPAWN_NOTICE
    return ''
