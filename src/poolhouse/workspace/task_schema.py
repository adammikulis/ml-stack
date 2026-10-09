"""Bounded task specifications, proposals and independent review decisions."""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any

from poolhouse.reputation.economy import assessment

DIGEST = re.compile(r'^[0-9a-f]{64}$')
TASK_ID = re.compile(r'^task:[0-9a-f]{32}$')
SPEC_FIELDS = frozenset({'title', 'description', 'acceptance', 'deps', 'capabilities', 'limits', 'source_key', 'assignees', 'reviewers', 'project'})


def fingerprint(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def text(value: Any, name: str, maximum: int, *, optional: bool = False) -> str:
    if type(value) is not str or len(value) > maximum or (not optional and not value.strip()):
        raise ValueError(f'{name} must be nonempty text of at most {maximum} characters')
    return value


def words(value: Any, name: str, maximum: int = 32) -> list[str]:
    if type(value) is not list or len(value) > maximum:
        raise ValueError(f'{name} must be a list with at most {maximum} entries')
    result = [text(item, name, 500) for item in value]
    if len(set(result)) != len(result):
        raise ValueError(f'{name} cannot contain duplicates')
    return result


def task_spec(value: Any) -> dict[str, Any]:
    if type(value) is not dict or set(value) - SPEC_FIELDS:
        raise ValueError('task specification has unsupported fields')
    project = value.get('project', {})
    if type(project) is not dict or len(project) > 8:
        raise ValueError('project must be bounded permission metadata')
    project = {text(key, 'project key', 100): text(item, 'project value', 2000)
               for key, item in project.items()}
    acceptance = words(value.get('acceptance'), 'acceptance')
    if not acceptance:
        raise ValueError('at least one acceptance criterion is required')
    deps = words(value.get('deps', []), 'deps')
    if any(not TASK_ID.fullmatch(dep) for dep in deps):
        raise ValueError('dependencies must be task IDs')
    limits = value.get('limits', {})
    if type(limits) is not dict or set(limits) - {'model', 'max_wall_s', 'max_retries'}:
        raise ValueError('unsupported task limits')
    wall = limits.get('max_wall_s')
    if wall is not None and (type(wall) not in (int, float) or not math.isfinite(wall) or wall <= 0):
        raise ValueError('max_wall_s must be None or a positive finite number')
    retries = limits.get('max_retries')
    if 'max_retries' in limits and (type(retries) is not int or not 0 <= retries <= 10):
        raise ValueError('max_retries must be between 0 and 10')
    if 'model' in limits:
        text(limits['model'], 'model', 1024)
    return {'title': text(value.get('title'), 'title', 200),
            'description': text(value.get('description', ''), 'description', 20000, optional=True),
            'acceptance': acceptance, 'deps': deps, 'project': project,
            'assignees': words(value.get('assignees', []), 'assignees'),
            'reviewers': words(value.get('reviewers', []), 'reviewers'),
            'capabilities': words(value.get('capabilities', []), 'capabilities'), 'limits': limits,
            'source_key': text(value.get('source_key', ''), 'source_key', 500, optional=True)}


def checks(value: Any) -> list[dict[str, Any]]:
    if type(value) is not list or not 1 <= len(value) <= 32:
        raise ValueError('checks must contain one to 32 recorded results')
    result = []
    for item in value:
        if type(item) is not dict or set(item) != {'name', 'passed'} or type(item['passed']) is not bool:
            raise ValueError('a check requires a name and a boolean passed result')
        result.append({'name': text(item['name'], 'check name', 500), 'passed': item['passed']})
    if len({item['name'] for item in result}) != len(result):
        raise ValueError('check names cannot repeat')
    return result


def usage(value: Any) -> dict[str, Any]:
    if type(value) is not dict or set(value) - {'source', 'tokens_in', 'tokens_out', 'wall_seconds'}:
        raise ValueError('unsupported usage fields')
    result = {}
    for name, item in value.items():
        if name == 'source':
            result[name] = text(item, name, 200)
        elif type(item) not in (int, float) or not math.isfinite(item) or not 0 <= item <= 1e12:
            raise ValueError('usage counters must be finite and nonnegative')
        elif name != 'wall_seconds' and type(item) is not int:
            raise ValueError('token counts must be integers')
        else:
            result[name] = item
    return result


def submission(value: Any) -> dict[str, Any]:
    allowed = {'artifacts', 'checks', 'usage', 'summary', 'provenance'}
    if type(value) is not dict or set(value) - allowed:
        raise ValueError('unsupported submission fields')
    artifacts = value.get('artifacts')
    if type(artifacts) is not dict or not 1 <= len(artifacts) <= 32:
        raise ValueError('one to 32 artifact hashes are required')
    for name, digest in artifacts.items():
        text(name, 'artifact name', 500)
        if type(digest) is not str or not DIGEST.fullmatch(digest):
            raise ValueError('artifacts require SHA256 hashes')
    provenance = value.get('provenance', {})
    if type(provenance) is not dict or set(provenance) - {'commit', 'environment', 'model', 'runtime'}:
        raise ValueError('unsupported execution provenance')
    return {'artifacts': dict(artifacts), 'checks': checks(value.get('checks')),
            'usage': usage(value.get('usage', {})),
            'summary': text(value.get('summary', ''), 'summary', 2000, optional=True),
            'provenance': {key: text(item, key, 1024) for key, item in provenance.items()}}


def review_decision(value: Any, acceptance: list[str], artifacts: dict[str, str]) -> dict[str, Any]:
    allowed = {'accepted', 'reason', 'checks', 'outcome', 'verified_usage', 'quality', 'review'}
    if type(value) is not dict or set(value) - allowed or type(value.get('accepted')) is not bool:
        raise ValueError('review requires a boolean accepted decision')
    outcome = value.get('outcome', 'accepted' if value['accepted'] else 'rejected')
    if outcome not in ('accepted', 'rejected', 'blocked_infrastructure') or (outcome == 'accepted') != value['accepted']:
        raise ValueError('review outcome must agree with accepted')
    verified = checks(value.get('checks'))
    passed = {item['name'] for item in verified if item['passed']}
    if value['accepted'] and (not all(item['passed'] for item in verified) or not set(acceptance) <= passed):
        raise ValueError('acceptance requires independent passing checks for every criterion')
    result = {'accepted': value['accepted'], 'outcome': outcome,
              'reason': text(value.get('reason'), 'review reason', 2000), 'checks': verified}
    if 'verified_usage' in value:
        result['verified_usage'] = value['verified_usage']
    assessment({'checks': verified, 'artifacts': artifacts,
                **{key: value[key] for key in ('quality', 'review') if key in value},
                **({'usage': value['verified_usage']} if 'verified_usage' in value else {})})
    for name in ('quality', 'review'):
        if name in value:
            result[name] = value[name]
    return result
