"""Patterns for recognized credentials in text."""

from __future__ import annotations

import re

PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
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
