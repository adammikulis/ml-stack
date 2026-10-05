"""Reviewed work awards, independent ratings and free-mode resource accounting."""

from __future__ import annotations

import math
from typing import Any

BASE_CREDITS = 10
QUALITY_KINDS = frozenset({'validated', 'regression', 'impact'})


def assessment(raw: dict[str, Any]) -> dict[str, Any]:
    """Validate reviewer-supplied quality, ratings and optional measured usage."""
    quality = raw.get('quality', [])
    if type(quality) is not list or len(quality) > 3:
        raise ValueError('quality needs at most three independently reviewed tiers')
    seen = set()
    for item in quality:
        if type(item) is not dict or set(item) != {'kind', 'reason', 'checks', 'artifacts'}:
            raise ValueError('quality needs kind, reason and linked checks and artifacts')
        kind = item['kind']
        if type(kind) is not str or kind not in QUALITY_KINDS or kind in seen:
            raise ValueError('each reviewed quality tier can be awarded only once')
        seen.add(kind)
        _reason(item['reason'])
        for field, available in (('checks', {c['name'] for c in raw['checks']}),
                                 ('artifacts', set(raw['artifacts']))):
            links = item[field]
            if type(links) is not list or not 1 <= len(links) <= 16 \
                    or any(type(link) is not str or link not in available for link in links) \
                    or len(set(links)) != len(links):
                raise ValueError('quality must link passed independent checks and hashed artifacts')
    review = raw.get('review')
    if review is not None:
        if type(review) is not dict or set(review) != {'quality', 'reliability', 'reason'}:
            raise ValueError('work review needs quality, reliability and reason')
        _reason(review['reason'])
        if any(type(review[key]) is not int or not 0 <= review[key] <= 100
               for key in ('quality', 'reliability')):
            raise ValueError('independent quality and reliability ratings range from zero to 100')
    usage = raw.get('usage')
    if usage is not None:
        if type(usage) is not dict or set(usage) != {'source', 'tokens_in', 'tokens_out', 'wall_seconds'}:
            raise ValueError('measured usage needs source, input/output tokens and wall seconds')
        _reason(usage['source'])
        for key in ('tokens_in', 'tokens_out'):
            value = usage[key]
            if value is not None and (type(value) is not int or not 0 <= value <= 1_000_000_000):
                raise ValueError('measured tokens must be bounded nonnegative integers or unknown')
        wall = usage['wall_seconds']
        if wall is not None and (type(wall) not in (int, float) or not math.isfinite(wall)
                                 or not 0 <= wall <= 86400):
            raise ValueError('measured wall time must be bounded nonnegative seconds or unknown')
    provenance = raw.get('provenance')
    if provenance is not None:
        if type(provenance) is not dict or set(provenance) != {'model', 'harness'}:
            raise ValueError('task provenance needs model and harness')
        for value in provenance.values():
            _reason(value)
    return {'base': BASE_CREDITS, 'quality_bonus': 5 * len(quality),
            'total': BASE_CREDITS + 5 * len(quality), 'quality': quality}


def _reason(value: Any) -> None:
    if type(value) is not str or not 1 <= len(value) <= 500 or not value.isprintable():
        raise ValueError('review reasons must be bounded printable text')


def summary(evidence: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate credits and explicitly reviewed ratings, with unmeasured usage left unknown."""
    awards = [item.get('award') or {'base': BASE_CREDITS, 'quality_bonus': 0,
                                  'total': BASE_CREDITS, 'quality': []} for item in evidence]
    reviews = [item['review'] for item in evidence if item.get('review') is not None]
    usage = [item['usage'] for item in evidence if item.get('usage') is not None]
    earned = sum(item['total'] for item in awards)
    ratings = {key: round((100 + sum(item[key] for item in reviews)) / (2 + len(reviews)), 2)
               for key in ('quality', 'reliability')}
    confidence = len(reviews) / (len(reviews) + 2)
    modifier = round(1 + (50 - ratings['reliability']) / 250 * confidence, 3)
    return {'economy': {'mode': 'free', 'earned': earned, 'spent': 0, 'balance': earned,
                        'completion_credits': sum(item['base'] for item in awards),
                        'quality_credits': sum(item['quality_bonus'] for item in awards),
                        'usage': {'recorded': bool(usage), 'tasks': len(usage),
                                  'scope': 'verified_task_evidence',
                                  'measurements': {key: sum(item[key] is not None for item in usage)
                                                   for key in ('tokens_in', 'tokens_out', 'wall_seconds')},
                                  **{key: sum(item[key] for item in usage if item[key] is not None)
                                     if any(item[key] is not None for item in usage) else None
                                     for key in ('tokens_in', 'tokens_out', 'wall_seconds')}}},
            'work_reputation': {**ratings, 'samples': len(reviews),
                                'confidence': round(confidence, 3),
                                'state': 'reviewed' if reviews else 'unrated'},
            'pricing': {'mode': 'free', 'price': 0, 'future_modifier': modifier,
                        'modifier_bounds': [0.8, 1.2], 'baseline_guaranteed': True,
                        'authority': 'none', 'active_charging': False}}
