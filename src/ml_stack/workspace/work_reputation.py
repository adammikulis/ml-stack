"""Parent-authenticated work verification and read-only team reputation."""

from __future__ import annotations

import hashlib
import re
from contextlib import contextmanager
from typing import Any

from ml_stack import activity
from ml_stack.files import read_json
from ml_stack.reputation import economy
from ml_stack.reputation.work import WorkLedger
from ml_stack.workspace import device_accounts, tokens
from ml_stack.workspace.identity import HUMAN, Denied, valid_id, valid_name
from ml_stack.workspace.service import Workspace

_HEX = re.compile(r'[0-9a-f]{64}')


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
    return hashlib.sha256(str(ws.base.resolve()).encode()).hexdigest()


def _evidence(raw: dict[str, Any]) -> dict[str, Any]:
    if type(raw) is not dict or not {'task', 'completion', 'checks', 'artifacts'} <= set(raw) \
            or set(raw) - {'task', 'completion', 'checks', 'artifacts', 'quality', 'review', 'usage', 'provenance'}:
        raise ValueError('verification needs task, completion, checks and artifact hashes')
    if any(type(raw[key]) is not int or raw[key] < 1 for key in ('task', 'completion')):
        raise ValueError('task and completion are positive message numbers')
    checks, artifacts = raw['checks'], raw['artifacts']
    if type(checks) is not list or not 1 <= len(checks) <= 16:
        raise ValueError('verification needs one to sixteen passed checks')
    for check in checks:
        if type(check) is not dict or set(check) != {'name', 'passed'} \
                or check['passed'] is not True or not _text(check['name']):
            raise ValueError('every named verification check must have passed')
    if type(artifacts) is not dict or not 1 <= len(artifacts) <= 16 \
            or any(not _text(name) or type(value) is not str or not _HEX.fullmatch(value)
                   for name, value in artifacts.items()):
        raise ValueError('verification needs bounded artifact names and SHA256 hashes')
    return {**raw, 'award': economy.assessment(raw)}


def _text(value: Any) -> bool:
    return type(value) is str and 1 <= len(value) <= 200 and value.isprintable()


def verify(ws, token: str, agent: str, evidence: dict[str, Any], *,
           ledger: WorkLedger | None = None) -> dict[str, Any]:
    """Credit a completed assigned task after its person or registered parent verifies it."""
    who = ws.auth(token)
    if type(agent) is not str or not valid_id(agent) or agent == who.id:
        raise Denied('an agent cannot award itself completion credit')
    info = ws.registry.info(agent)
    if info['role'] not in ('agent', 'lead') or (who.role != HUMAN and info['parent'] != who.id):
        raise Denied('only the person or the agent parent can verify its work')
    raw = _evidence(evidence)
    task, completion = ws.bus.get(raw['task']), ws.bus.get(raw['completion'])
    if not task or task['type'] != 'task' or task['to'] != agent \
            or (who.role != HUMAN and task['from'] != who.id):
        raise Denied('verification must name an authenticated task assigned to this agent')
    if not completion or completion['from'] != agent or completion['to'] != task['from'] \
            or completion['type'] not in ('answer', 'status', 'handoff', 'done') \
            or completion['seq'] <= task['seq'] \
            or int(completion.get('thread') or completion['seq']) != int(task.get('thread') or task['seq']):
        raise Denied('verification must name the agent completion reply in the assigned task thread')
    with _store(ledger) as held:
        result = held.record({**raw, 'agent': agent, 'verifier': who.id,
                              'workspace': scope(ws), 'verified_at': ws.clock(),
                              'task_hash': task['hash'], 'completion_hash': completion['hash']})
    if result['credited']:
        ws.audit('work.verified', who.id, agent=agent, task=raw['task'], evidence=result['id'])
        activity.record('agent.work_verified', actor=agent, subject=f"Workspace task {raw['task']}",
                        outcome='verified', refs={'workspace_message': str(raw['task']),
                                                   'evidence': result['id'], 'verifier': who.id},
                        meta={'completion_credit': 1, 'checks': len(raw['checks'])})
    return result


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
            team.append({'agent': name, 'score': 0, 'verified_tasks': 0, 'evidence': [],
                         'aliases': aliases.get(name, []), **economy.summary([])})
    team = _accounts(ws, team)
    account = device_accounts.account_for(ws, who.id)
    own_id = account['base_id'] if account else who.id
    own = next((item for item in team if item['agent'] == own_id),
               {'agent': own_id, 'score': 0, 'verified_tasks': 0, 'evidence': [], **economy.summary([])})
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
                                        'evidence': [], 'enrolled': bool(account)})
        group['members'].update(account['members'] if account else [row['agent']])
        group['aliases'].update(row.get('aliases', []))
        group['evidence'].extend(row['evidence'])
        if account:
            group['device_id'] = account['device_id']
    result = []
    for group in groups.values():
        evidence = sorted(group['evidence'], key=lambda item: -item['verified_at'])
        result.append({**group, 'members': sorted(group['members']), 'aliases': sorted(group['aliases']),
                       'evidence': evidence, 'score': len(evidence), 'verified_tasks': len(evidence),
                       **economy.summary(evidence)})
    return sorted(result, key=lambda item: item['agent'])


def brief(ws, token: str) -> str:
    """A bounded own/team reputation summary for an agent's task context."""
    try:
        result = standings(ws, token)
    except (OSError, RuntimeError, ValueError):
        return 'Verified-work reputation is unavailable; no completion score can be inferred.'
    rows = [f"{item['agent']}: {item['verified_tasks']} verified tasks; "
            f"{item['economy']['balance']} credits; reliability "
            f"{item['work_reputation']['reliability']} ({item['work_reputation']['samples']} reviews)"
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
