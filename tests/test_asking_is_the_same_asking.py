"""The prompt bytes every way of asking produces, pinned to a hash.

`graph.cache.fingerprint` hashes the system prompt and the tool descriptions into the
answer-cache key, and `bench.keep` stores runs measured against those exact bytes. Each
digest below is the SHA-256 of one way's system prompt, its tool schemas and the
response_format of its first turn.
"""

from __future__ import annotations

import hashlib
import json

import pytest

from ml_stack.graph.ask import converse, tools_for
from ml_stack.graph.asking import Asking

GRAPH = {
    "nodes": [
        {"id": "person:ada", "kind": "person", "label": "Ada Lovelace", "mentions": 4,
         "attrs": {"role": "analyst", "location": "Turin"}, "messages": ["m1"]},
        {"id": "person:bea", "kind": "person", "label": "Bea Marlow", "mentions": 2,
         "attrs": {}, "messages": ["m2"]},
        {"id": "topic:compilers", "kind": "topic", "label": "compilers", "mentions": 3,
         "attrs": {}, "messages": ["m1", "m2"]},
        {"id": "org:pellard", "kind": "org", "label": "Pellard Foundry", "mentions": 1,
         "attrs": {"type": "company"}, "messages": []},
    ],
    "edges": [
        {"source": "person:ada", "target": "topic:compilers", "rel": "interested_in", "weight": 3},
        {"source": "person:bea", "target": "topic:compilers", "rel": "interested_in", "weight": 2},
        {"source": "person:ada", "target": "org:pellard", "rel": "works_at", "weight": 1},
    ],
    "messages": {
        "m1": {"text": "I am Ada and I have spent years on compilers."},
        "m2": {"text": "compilers are what I do too, mostly."},
    },
}

QUESTION = "who works on compilers?"

# Every way `ml-stack-bench --also` measures, and the riders `_ways` puts on each of them.
WAYS = {
    "plain": Asking(),
    "terse": Asking(terse=True),
    "rich": Asking(rich=True),
    "reach": Asking(reach=8000),
    "loose": Asking(tight=False),
    "batch": Asking(batch=True),
    "kinds": Asking(kinds=True),
    "summary": Asking(summary=True),
    "single": Asking(single=True),
    "few": Asking(few=True),
    "constrain_ids": Asking(constrain_ids=True),
    "rounds": Asking(rounds=20),
    "terse-loose": Asking(terse=True, tight=False),
    "terse-reach": Asking(terse=True, reach=8000),
    "batch-kinds-summary": Asking(batch=True, kinds=True, summary=True),
    "terse-batch-summary": Asking(terse=True, batch=True, summary=True),
    "single-few-rounds": Asking(single=True, few=True, rounds=20),
    "few-constrain_ids": Asking(few=True, constrain_ids=True),
    "rich-loose-reach": Asking(rich=True, tight=False, reach=8000),
}

DIGESTS = {
    "plain": "2f4cb3200856035407c25b8e8afa2965185f604c073e3d58734a73c001ef79c5",
    "terse": "d5e9fd7d919675a80ea6542b5b670e1effb00a4376eb61e5615c5ced94a6f460",
    "rich": "063e915f97e99e55a0106ad95c167920439cf0fa59505e1dcb4f9fc25d4548ca",
    "reach": "2f4cb3200856035407c25b8e8afa2965185f604c073e3d58734a73c001ef79c5",
    "loose": "dadec366cf977fb686033f1fd517a8ca3421234fb743767c1a9cd103d1e8e3cb",
    "batch": "c655711c58269a1fa17150295b534da16630c1746d39d76549bef7825cdfd63e",
    "kinds": "2f4cb3200856035407c25b8e8afa2965185f604c073e3d58734a73c001ef79c5",
    "summary": "7f408eec33a3108b0dc3daa65c6339d5b43ea01db84d28432f2b7702e241ae20",
    "single": "10aeb939533278546668efaa81e7e7ab634558500563e58d8e4e5ca03ab6ddf0",
    "few": "3cf7899cac087918d1d3c0bb83c82a725822b7f6e9951eee88d2f320d7caecb3",
    "constrain_ids": "c9533c90a6afe518ee8f8ecbc509f92430e50dfc25d42e0154f2418d09ef9a2b",
    "rounds": "2f4cb3200856035407c25b8e8afa2965185f604c073e3d58734a73c001ef79c5",
    "terse-loose": "56003392d8ec816661b393c18e58c352048a0e30e7e1ab5c728afb4be0cfa58d",
    "terse-reach": "d5e9fd7d919675a80ea6542b5b670e1effb00a4376eb61e5615c5ced94a6f460",
    "batch-kinds-summary": "a6011cf9bb228319be4d3c5d418a3fc02ff13211d93b727baeac724989e63447",
    "terse-batch-summary": "2498c99ace4c28644885eb1344b4d6202f3e4930bb773cb3bd6a1ae50ae2407b",
    "single-few-rounds": "88a59854c307ccedbf18ff0a6a094e3b1621fdc4263d233a1744769dc097d576",
    "few-constrain_ids": "6fc3876c9cfa3436d3d05ae7b0db33eb29c8eaf52f8961714ddaa305fe042e49",
    "rich-loose-reach": "20221bcc2371e34f0db24832e972036be65dbd1eaa9749e5d045c164216cb67f",
}


class _First:
    """Keeps the first turn's system prompt, tool schemas and response_format."""

    def __init__(self) -> None:
        self.system = ""
        self.schemas: list = []
        self.format: dict | None = None
        self.seen = 0

    def chat(self, messages, *, tools=None, **extra):
        from ml_stack.client import Reply

        if not self.seen:
            self.system = str(messages[0]["content"])
            self.schemas = list(tools or [])
            self.format = extra.get("response_format")
        self.seen += 1
        return Reply(content="Ada and Bea both work on compilers.")


def _digest(way: Asking) -> str:
    watching = _First()
    tools = tools_for(GRAPH, terse=True, **way.tools()) if way.terse else None
    converse(QUESTION, GRAPH, watching, tools=tools, asking=way)
    blob = json.dumps([watching.system, watching.schemas, watching.format],
                      sort_keys=False, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


@pytest.mark.parametrize("name", sorted(WAYS))
def test_the_prompt_bytes_of_one_way_are_what_they_were(name):
    assert _digest(WAYS[name]) == DIGESTS[name]


def test_every_way_the_bench_measures_is_pinned():
    assert sorted(WAYS) == sorted(DIGESTS)


# The ways that change what the model reads. `kinds` filters what `show` returns, `rounds`
# caps the loop, and `reach` changes how a tool result is packed; none touches the prompt.
MOVES = ("terse", "rich", "loose", "batch", "summary", "single", "few", "constrain_ids")


@pytest.mark.parametrize("name", MOVES)
def test_a_way_that_changes_the_asking_changes_the_bytes(name):
    assert DIGESTS[name] != DIGESTS["plain"]


@pytest.mark.parametrize("name", ("kinds", "rounds", "reach"))
def test_a_way_outside_the_prompt_leaves_the_bytes_alone(name):
    assert DIGESTS[name] == DIGESTS["plain"]
