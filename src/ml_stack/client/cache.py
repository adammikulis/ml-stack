"""An extraction already done, not done again: one JSON file per (version, schema, text).

Reading a corpus into structure is the expensive half of building a graph — measured here,
ten seconds a message on a large model — and a pipeline is run again and again while
everything downstream of it is worked on. What makes a re-run cheap is knowing which
extractions are still true.

**The key is deliberately not the instructions.** Wording them is an iterative business, and
a pipeline that re-reads its whole corpus because a sentence was rephrased makes rephrasing
it expensive, which means it stops happening. What changes an extraction's *meaning* is the
text, the schema it has to match, and the extractor's own version — so those are the key, and
``version`` is the knob for a change nobody should be allowed to skip. Record what the
instructions were beside the answer if a stale one needs finding later.

``extra`` is the rest of the prompt that varies per record and is not in ``text``: the thread
a message is a reply to, the page a paragraph came from, the vocabulary offered. A caller
that puts nothing there when its prompt carries something is asking two different questions
under one key, and will be handed the wrong answer.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

__all__ = ["extraction_key", "read_cached", "write_cached"]


def extraction_key(text: str, schema: dict[str, Any], *, version: str = "",
                   extra: str = "") -> str:
    """The cache key for one extraction: what would change what it should say.

    Short enough to be a filename and to read in a log, long enough that two of a corpus's
    messages do not collide.
    """
    stamp = version + json.dumps(schema, sort_keys=True) + text + extra
    return hashlib.sha256(stamp.encode()).hexdigest()[:16]


def read_cached(cache_dir: str | Path, key: str) -> Any | None:
    """The answer stored under ``key``, or None when there is none to be had.

    A file that is missing, unreadable or no longer JSON is not an error: it is a cache, and
    the only cost of a miss is asking the model again.
    """
    try:
        return json.loads((Path(cache_dir) / f"{key}.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def write_cached(cache_dir: str | Path, key: str, answer: Any) -> None:
    """Store ``answer`` under ``key``, atomically, so a reader never sees half of it."""
    from ml_stack.files import write_json

    write_json(Path(cache_dir) / f"{key}.json", answer)
