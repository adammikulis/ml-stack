"""Removing secrets from text and from nested values before they are logged or stored."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

__all__ = ["MASK", "redact", "redact_value"]

MASK = "<redacted>"

_NAMED = (r"(?:password|passwd|passphrase|secret|token|api[_-]?key|access[_-]?key"
          r"|private[_-]?key|authorization)")
_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?(?:-----END [A-Z ]*PRIVATE KEY-----|\Z)",
               re.S),
    re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]{8,}"),
    re.compile(r"ML-Stack-MAC\s+[^\n]*"),
    re.compile(r"\bmlsk1\.[A-Za-z0-9_-]{8,}"),
    re.compile(r"\bhf_[A-Za-z0-9]{16,}"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{16,}|\bgithub_pat_[A-Za-z0-9_]{16,}"),
    re.compile(r"\bsk-(?:ant-)?[A-Za-z0-9_-]{16,}"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(rf"(?i)\b{_NAMED}\b[\"']?\s*[:=]\s*[\"']?[^\s\"',;&]{{4,}}"),
    re.compile(r"(?i)([?&](?:token|key|secret|password|sig|signature)=)[^&\s]+"),
    re.compile(r"(?<![A-Za-z0-9+/=_-])(?=[A-Za-z0-9+/_-]*[a-z])(?=[A-Za-z0-9+/_-]*[A-Z])"
               r"(?=[A-Za-z0-9+/_-]*\d)[A-Za-z0-9+/_-]{40,}={0,2}(?![A-Za-z0-9+/=_-])"),
)
_KEYLIKE = re.compile(rf"(?i){_NAMED}")


def redact(text: str) -> str:
    """``text`` with tokens, keys, bearer values and key=value secrets replaced by a mask."""
    for pattern in _PATTERNS:
        if pattern.groups == 1:
            text = pattern.sub(lambda m: m.group(1) + MASK, text)
        else:
            text = pattern.sub(MASK, text)
    return text


def redact_value(value: Any, *, name: str = "") -> Any:
    """``value`` with every string redacted and every value under a key-like name masked."""
    if name and _KEYLIKE.search(name) and not isinstance(value, bool | int | float | type(None)):
        return MASK
    if isinstance(value, str):
        return redact(value)
    if isinstance(value, Mapping):
        return {str(k): redact_value(v, name=str(k)) for k, v in value.items()}
    if isinstance(value, list | tuple | set | frozenset):
        return [redact_value(v) for v in value]
    return value
