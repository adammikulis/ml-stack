"""``python -m ml_stack.ingest`` -- what `detach` re-runs."""

from ml_stack.ingest.cli import main

if __name__ == "__main__":  # pragma: no cover - what `detach` re-runs
    raise SystemExit(main())
