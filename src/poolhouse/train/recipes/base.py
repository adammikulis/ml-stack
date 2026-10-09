"""Base model selection for pretrained recipes."""

from typing import Any


def resolve_base(spec: dict[str, Any], config: dict[str, Any],
                 manifest: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Return the configured base and its matching size estimates."""
    sizes = spec.get("sizes", {})
    size = config.get("size") or (sorted(sizes)[0] if sizes else "")
    entry = dict(sizes.get(size, {}))
    base = str(config.get("base") or manifest.get("base") or entry.get("base") or "")
    return base, entry if base == entry.get("base") else {}
