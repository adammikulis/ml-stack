"""``python -m poolhouse.ingest`` -- what `detach` re-runs."""

from poolhouse.ingest.cli import main

if __name__ == "__main__":  # pragma: no cover - what `detach` re-runs
    raise SystemExit(main())
