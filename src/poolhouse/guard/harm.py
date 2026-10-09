"""The labels, the verdict and the findings the destructive-action classifier is built from."""

from __future__ import annotations

import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from poolhouse import home

__all__ = ["ASKS", "LABELS", "RANK", "Finding", "Verdict", "outside", "protected", "resolve", "worst"]

LABELS = ("safe", "reversible", "unsure", "destructive")
"""In order of severity."""
ASKS = frozenset({"unsure", "destructive"})
"""The labels that need the person's approval whatever the role."""
RANK = {label: at for at, label in enumerate(LABELS)}

PROTECTED = ("/etc", "/usr", "/bin", "/sbin", "/lib", "/lib64", "/boot", "/dev", "/proc", "/sys",
             "/var", "/opt", "/system", "/library", "/applications", "/private/etc", "/volumes",
             "/root", "/srv")
PROTECTED_HOME = (".ssh", ".aws", ".gnupg", ".kube", ".config", ".docker", "library", ".netrc",
                  ".git-credentials", ".password-store")
SECRET_NAMES = frozenset({".env", ".netrc", ".npmrc", ".pypirc", "id_rsa", "id_ed25519",
                          "authorized_keys", "known_hosts", "credentials", ".git-credentials"})


@dataclass(frozen=True, slots=True)
class Finding:
    """One thing the classifier noticed: a ``label`` and the plain-words ``reason``."""

    label: str
    reason: str


@dataclass(frozen=True, slots=True)
class Verdict:
    """What the classifier concluded about one call: the ``label``, the ``reasons`` in plain
    words, the ``layer`` that decided (``deterministic`` or ``model``) and a ``confidence``
    between 0 and 1 (1 for a structural rule, 0 for ``unsure``)."""

    label: str
    reasons: list[str] = field(default_factory=list)
    layer: str = "deterministic"
    confidence: float = 1.0

    @property
    def asks(self) -> bool:
        return self.label in ASKS

    def public(self) -> dict[str, object]:
        """A JSON-ready dict."""
        return {"label": self.label, "reasons": list(self.reasons), "layer": self.layer,
                "confidence": round(self.confidence, 3)}


def worst(findings: Iterable[Finding]) -> str:
    """The most severe label among ``findings``; ``safe`` for none."""
    return max((f.label for f in findings), key=RANK.__getitem__, default="safe")


def _full(path: str, base: str) -> Path:
    text = home.expand(path.strip())
    return Path(os.path.normpath(text if text.is_absolute() else Path(base) / text))


def resolve(path: str, roots: Sequence[str]) -> Path:
    """``path`` made absolute (relative ones against the first root, else the current
    directory) with ``..`` folded away; the filesystem is not consulted."""
    return _full(path, roots[0] if roots else str(Path.cwd()))


def outside(path: str, roots: Sequence[str]) -> bool:
    """Whether ``path`` (relative ones are read against the first root) leaves every root."""
    if not roots:
        return False
    full = resolve(path, roots)
    return not any(full == (base := _full(root, "/")) or base in full.parents for root in roots)


def protected(path: str) -> bool:
    """Whether ``path`` is the filesystem root, the home folder or inside a system or
    credentials directory."""
    text = home.expand(path.strip())
    text = Path(os.path.normpath(text))
    low = str(text).lower()
    base = home.user_home()
    if text in (Path("/"), base):
        return True
    if any(low == p or low.startswith(p + "/") for p in PROTECTED):
        return True
    if base in text.parents:
        return text.relative_to(base).parts[0].lower() in PROTECTED_HOME
    return ".git" in text.parts[:-1] or text.name.lower() in SECRET_NAMES
