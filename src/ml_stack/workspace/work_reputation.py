"""Read-only device standings from authenticated canonical reviews and historical evidence."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any

from ml_stack.files import read_json
from ml_stack.reputation import economy
from ml_stack.reputation.work import WorkLedger
from ml_stack.workspace import coordination, device_accounts, tokens
from ml_stack.workspace.identity import HUMAN, Denied, valid_id, valid_name
from ml_stack.workspace.service import Workspace


@contextmanager
def _store(ledger):
    held = ledger or WorkLedger()
    try:
        yield held
    finally:
        if ledger is None:
            held.sealed.close()


def scope(ws) -> str:
    """The workspace identity used by verified-work evidence."""
    return coordination.workspace_id(ws)


def standings(ws, token: str, *, agent: str = "", offset: int = 0,
              ledger: WorkLedger | None = None) -> dict[str, Any]:
    """The authenticated caller's completion standing and its workspace team's evidence."""
    who = ws.auth(token)
    ws._may(who, 'read')
    if type(agent) is not str or (agent and not valid_id(agent)):
        raise ValueError('agent is a registered workspace identity')
    if type(offset) is not int or not 0 <= offset <= 5000:
        raise ValueError('evidence offset must be between zero and 5000')
    with _store(ledger) as held:
        team = held.standings(scope(ws))
    aliases: dict[str, list[str]] = {}
    for path in (ws.base / 'local-agents').glob('*.json'):
        name = path.stem
        if not valid_name(name) or name.endswith('.status'):
            continue
        saved = read_json(path, None)
        if isinstance(saved, dict) and saved.get('version') == 1 and saved.get('name') == name:
            identity = saved.get('identity') or name
            if isinstance(identity, str) and valid_id(identity):
                aliases.setdefault(identity, []).append(name)
    for item in team:
        item['aliases'] = aliases.get(item['agent'], [])
    known = {item['agent'] for item in team}
    for name in ws.registry.ids():
        if name not in known and ws.registry.info(name)['role'] != HUMAN:
            team.append({'agent': name, 'verified_tasks': 0, 'evidence': [],
                         'aliases': aliases.get(name, []), **economy.summary([])})
    team = _accounts(ws, team)
    account = device_accounts.account_for(ws, who.id)
    own_id = account['base_id'] if account else who.id
    own = next((item for item in team if item['agent'] == own_id),
               {'agent': own_id, 'verified_tasks': 0, 'evidence': [], **economy.summary([])})
    for item in team:
        item['evidence_held'] = max(0, len(item['evidence']) - offset - 20)
        item['evidence'] = item['evidence'][offset:offset + 20]
    return {'own': own, 'team': [item for item in team if not agent or item['agent'] == agent or agent in item['members']],
            'offset': offset, 'authority': 'none',
            'economy_mode': 'free',
            'metric': 'Completion credits and reviewed quality bonuses; runs are free.'}


def _accounts(ws, team):
    groups = {}
    for row in team:
        account = device_accounts.account_for(ws, row['agent'])
        base = account['base_id'] if account else row['agent']
        group = groups.setdefault(base, {'agent': base, 'members': set(), 'aliases': set(),
                                        'evidence': [], 'contributions': [], 'enrolled': bool(account)})
        group['members'].update(account['members'] if account else [row['agent']])
        group['aliases'].update(row.get('aliases', []))
        group['evidence'].extend(row['evidence'])
        group['contributions'].extend(row.get('contributions', []))
        if account:
            group['device_id'] = account['device_id']
    result = []
    for group in groups.values():
        evidence = sorted(group['evidence'], key=lambda item: -item['verified_at'])
        result.append({**group, 'members': sorted(group['members']), 'aliases': sorted(group['aliases']),
                       'evidence': evidence, 'verified_tasks': len(evidence),
                       **economy.summary(evidence, group['contributions'])})
    return sorted(result, key=lambda item: item['agent'])


def brief(ws, token: str) -> str:
    """A bounded own/team reputation summary for an agent's task context."""
    try:
        result = standings(ws, token)
    except (OSError, RuntimeError, ValueError):
        return 'Verified-work reputation is unavailable; no credit balance or work rating can be inferred.'
    rows = [f"{item['agent']}: {item['verified_tasks']} verified tasks; "
            f"{item['economy']['balance']} credits; reliability "
            f"{item['work_reputation']['reliability']} ({item['work_reputation']['reliability_samples']} reliability reviews)"
            for item in result['team'][:20]]
    return ('Verified-work reputation (recorded evidence, no authority):\n'
            f"Your verified completions: {result['own']['verified_tasks']}\n"
            f"Your credits: {result['own']['economy']['balance']}; runs are free, no charging.\n" + '\n'.join(rows))


def fleet_route(request) -> bool:
    """The person's read-only verified-work standings behind Fleet authorization."""
    if request.path != '/ui/work-reputation/standings':
        return False
    if request.method != 'GET':
        request.send(405, {'error': 'reputation is read-only'})
        return True
    try:
        ws = Workspace()
        token = tokens.read_file(tokens.directory(ws.base) / tokens.OWNER_FILE)
        if ws.auth(token).role != HUMAN:
            raise Denied('the page needs its person owner')
        request.send(200, standings(ws, token, agent=request.asked('agent', ''),
                                     offset=int(request.asked('offset', '0'))))
    except Denied:
        request.send(403, {'error': 'the workspace person owner is unavailable'})
    except ValueError:
        request.send(400, {'error': 'Invalid reputation agent or evidence offset.'})
    except (OSError, RuntimeError):
        request.send(503, {'error': 'Verified-work reputation is unavailable or locked.'})
    return True
