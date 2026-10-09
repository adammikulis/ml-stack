"""Patterns for recognized credentials in text."""

from __future__ import annotations

import re
from collections.abc import Mapping

PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("workspace-token", re.compile(r"\bmlws1\.[A-Za-z0-9._/-]+\.[A-Za-z0-9_-]{8,}")),
    ("cluster-token", re.compile(r"\bmlsk1\.[A-Za-z0-9_-]{8,}")),
    ("private-key", re.compile(
        r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?(?:-----END [A-Z ]*PRIVATE KEY-----|\Z)", re.S)),
    ("aws-key", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("github-token", re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})\b")),
    ("hub-token", re.compile(r"\bhf_[A-Za-z0-9]{30,}\b")),
    ("slack-token", re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}\b")),
    ("api-key", re.compile(r"\bsk-(?:ant-|proj-)?[A-Za-z0-9_-]{20,}\b")),
    ("google-key", re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b")),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b")),
    ("bearer", re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{20,}")),
    ("url-password", re.compile(r"(?i)\b[a-z][a-z0-9+.-]*://[^\s/:@]+:[^\s/@]{3,}@")),
    ("assigned-secret", re.compile(
        r"(?i)\b[\w.-]*(?:api[_-]?key|secret|token|passw(?:or)?d|passphrase)[\w.-]*"
        r"\s*[:=]\s*[\"']?[^\s\"',;]{8,}")),
)


MARK = "[REDACTED:{}]"

ENV_NAME = re.compile(r"(?i)(?:key|secret|token|passw|credential|auth)")
MIN_ENV_VALUE = 8


def env_secrets(env: Mapping[str, str]) -> tuple[str, ...]:
    """The values of the variables whose names say they are credentials, longest first."""
    found = {v for k, v in env.items() if ENV_NAME.search(k) and len(v) >= MIN_ENV_VALUE}
    return tuple(sorted(found, key=len, reverse=True))


def redact(text: str, known: tuple[str, ...] = ()) -> tuple[str, list[str]]:
    """``text`` with every credential replaced by a marker, and the kinds that were found."""
    kinds: list[str] = []
    for value in known:
        if value in text:
            text = text.replace(value, MARK.format("env"))
            kinds.append("env")
    for kind, pattern in PATTERNS:
        text, count = pattern.subn(MARK.format(kind), text)
        if count:
            kinds.append(kind)
    return text, kinds
