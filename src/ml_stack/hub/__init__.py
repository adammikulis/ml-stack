"""Finding a model that did not exist when you last learned anything.

`naming` reads what a file's name says, `listing` asks the Hub what a repository holds,
`local` finds what is already on this machine, `drafts` picks the head to serve with a
model, `cards` reads the sampler settings a publisher asks for, `memory` says how much
this machine will give a model, and `cli` is ``ml-stack-models``.

This module is the namespace: everything is imported here, and every call between the
modules to something a test or `bench.selfcheck` patches -- `files`, `located`, `room`,
`on_disk`, `default_roots`, `hub_cache`, `forks`, `choose_head` -- goes through
``hub.<name>`` at call time, so patching it here patches it everywhere.
"""

from __future__ import annotations

from ml_stack.hub.cards import advice, card, in_gguf
from ml_stack.hub.cli import main
from ml_stack.hub.drafts import (
    DRAFT_DEPTH,
    NO_HEAD,
    Chosen,
    Head,
    _DRAFT_NOTES,  # noqa: F401 - the per-process cache a test clears
    choose_head,
    draft_for,
    draft_note,
    drafting,
    drafts_for,
    forks,
    head_choice,
    heads_for,
)
from ml_stack.hub.listing import (
    PREFER,
    Found,
    beside,
    build_files,
    builds,
    fetch,
    files,
    find,
    mmproj_for,
    ref,
)
from ml_stack.hub.local import (
    default_roots,
    hub_cache,
    located,
    on_disk,
    repo_of,
    shards_beside,
    weight_paths,
)
from ml_stack.hub.memory import free_memory, machine_room, room, total_memory
from ml_stack.hub.naming import (
    DRAFT_KINDS,
    DRAFT_MARK,
    QUANT,
    SHARD,
    WEIGHT_SUFFIXES,
    _precision,  # noqa: F401 - `serve.ops` ranks its own candidates by the same rule
    aside,
    base_words,
    borrowed_head,
    is_head,
    iq_on_metal,
    pretty_name,
    spec_for,
)

__all__ = [
    "DRAFT_DEPTH",
    "DRAFT_KINDS",
    "DRAFT_MARK",
    "NO_HEAD",
    "PREFER",
    "QUANT",
    "SHARD",
    "WEIGHT_SUFFIXES",
    "Chosen",
    "Found",
    "Head",
    "advice",
    "aside",
    "base_words",
    "beside",
    "borrowed_head",
    "build_files",
    "builds",
    "card",
    "choose_head",
    "default_roots",
    "draft_for",
    "draft_note",
    "drafting",
    "drafts_for",
    "fetch",
    "files",
    "find",
    "forks",
    "free_memory",
    "head_choice",
    "heads_for",
    "hub_cache",
    "in_gguf",
    "iq_on_metal",
    "is_head",
    "located",
    "machine_room",
    "main",
    "mmproj_for",
    "on_disk",
    "pretty_name",
    "ref",
    "repo_of",
    "room",
    "shards_beside",
    "spec_for",
    "total_memory",
    "weight_paths",
]
