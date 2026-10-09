"""The page's vocabulary switch: plain words or friendly, themed ones.

The page resolves it the same way every time: ``?vocab=`` in the address, then the browser's own
storage, then the environment variable read here, then ``professional``. This module is the last
two: the server's default, and the catalogue the page embeds.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping

from .vocabulary_strings import CATALOGUE

VOCABULARIES = ("professional", "friendly")
DEFAULT_VOCABULARY = VOCABULARIES[0]
VOCABULARY_ENV = "POOLHOUSE_UI_VOCAB"


def default_vocabulary(env: Mapping[str, str] | None = None) -> str:
    """The vocabulary the server renders first: the environment's when valid, else professional."""
    value = (os.environ if env is None else env).get(VOCABULARY_ENV, "").strip().lower()
    return value if value in VOCABULARIES else DEFAULT_VOCABULARY


def wording(key: str, vocabulary: str) -> str:
    """The text for ``key`` in one vocabulary."""
    professional, friendly = CATALOGUE[key]
    return professional if vocabulary == "professional" else friendly


def payload(env: Mapping[str, str] | None = None) -> str:
    """The JSON the page embeds: the server default, the allowed values and the catalogue.

    Escaped for a ``<script>`` element so no string in it can close the tag.
    """
    body = json.dumps({"vocabularies": VOCABULARIES, "vocab": default_vocabulary(env),
                       "catalogue": CATALOGUE}, separators=(",", ":"))
    return body.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
