"""Cluster admission profiles and startup notices."""

DEVELOPMENT = "dev"
PRODUCTION = "prod"
MODES = (DEVELOPMENT, PRODUCTION)


def validate(mode: str) -> str:
    """Return a supported cluster mode."""
    if mode not in MODES:
        raise ValueError("cluster mode is dev or prod")
    return mode


def notice(mode: str) -> str:
    """Return the one-line description of the effective cluster mode."""
    validate(mode)
    if mode == DEVELOPMENT:
        return "Mode: DEV — automatically joins nearby development clusters and shares project coordination with cluster members."
    return "Mode: PROD — devices and agents need explicit approval to join and access shared projects."
