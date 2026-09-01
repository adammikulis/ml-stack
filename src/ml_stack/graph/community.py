"""An invented community, for measuring how well a graph gets read and answered.

Every person, company, place and subject here is made up. A benchmark keyed to a real
community cannot be shared, cannot be compared between machines, and puts real people's
details somewhere they do not belong — so the questions and their answers are asked of these
people instead, and the numbers mean the same thing anywhere.

It is small on purpose: enough shape for a question to have a right answer and several
plausible wrong ones, few enough entries to read in one sitting.
"""

from __future__ import annotations

from typing import Any

__all__ = ["QUESTIONS", "graph"]

# id -> (label, kind, what they said). The subjects are attached by the edges below, the way
# a reader would have to work them out.
_SAID: dict[str, tuple[str, str, str]] = {
    "person:ada": ("Ada Lovelace", "person",
                   "I am a robotics technician and I fix machines on the line all day. "
                   "Mostly servo repair and field service for manufacturing plants."),
    "person:grace": ("Grace Hopper", "person",
                     "I run growth and go-to-market. Twenty years selling enterprise "
                     "software, mostly campaigns and channel partnerships."),
    "person:alan": ("Alan Turing", "person",
                    "I build machine learning models, lately for medical imaging at a "
                    "hospital group. Interested in anything clinical."),
    "person:katherine": ("Katherine Johnson", "person",
                         "Finance background — I automate month-end close and reporting. "
                         "Twenty-five years of process work in accounting teams."),
    "person:mary": ("Mary Somerville", "person",
                    "I welcome new people and keep the channels tidy. Community management "
                    "and onboarding is what I do."),
    "person:charles": ("Charles Babbage", "person",
                       "Hardware. I design and repair mechanical assemblies, and I am "
                       "looking for a marketing person to help me sell a machine."),
    "org:quenlow": ("Quenlow Robotics", "org", ""),
    "org:pellard": ("Pellard Foundry", "org", ""),
    "org:harnley": ("Harnley Health", "org", ""),
    "place:turin": ("Turin", "place", ""),
    "place:dunmore": ("Dunmore", "place", ""),
    "topic:robotics": ("robotics", "topic", ""),
    "topic:repair": ("repair", "topic", ""),
    "topic:marketing": ("marketing", "topic", ""),
    "topic:healthcare": ("healthcare", "topic", ""),
    "topic:automation": ("automation", "topic", ""),
    "topic:onboarding": ("onboarding", "topic", ""),
}

_JOINED: list[tuple[str, str, str]] = [
    ("person:ada", "experienced_in", "topic:robotics"),
    ("person:ada", "experienced_in", "topic:repair"),
    ("person:ada", "works_at", "org:quenlow"),
    ("person:ada", "based_in", "place:turin"),
    ("person:grace", "experienced_in", "topic:marketing"),
    ("person:grace", "based_in", "place:dunmore"),
    ("person:alan", "experienced_in", "topic:healthcare"),
    ("person:alan", "works_at", "org:harnley"),
    ("person:katherine", "experienced_in", "topic:automation"),
    ("person:katherine", "based_in", "place:dunmore"),
    ("person:mary", "experienced_in", "topic:onboarding"),
    ("person:charles", "experienced_in", "topic:repair"),
    ("person:charles", "works_at", "org:pellard"),
    ("person:charles", "seeks", "topic:marketing"),
]


def graph() -> dict[str, Any]:
    """The invented community, in the shape every reader of a graph here expects."""
    messages = {f"m{n}": {"text": said, "ts": str(1_700_000_000 + n * 3600),
                          "channel": "#general", "sender": label}
                for n, (node, (label, kind, said)) in enumerate(_SAID.items()) if said}
    behind = {node: [f"m{n}"] for n, (node, (_l, _k, said)) in enumerate(_SAID.items()) if said}
    nodes = [{"id": node, "label": label, "kind": kind, "mentions": 1,
              "attrs": {"member": kind == "person"}, "messages": behind.get(node, [])}
             for node, (label, kind, _s) in _SAID.items()]
    edges = [{"source": a, "rel": rel, "target": b, "weight": 2, "messages": []}
             for a, rel, b in _JOINED]
    return {"nodes": nodes, "edges": edges, "messages": messages,
            "stats": {"messages": len(messages)}, "meta": {"community": "invented"}}


# What a good answer names. An empty list means the right answer is to name nobody — a
# question the graph cannot answer is as much a test as one it can.
QUESTIONS: list[dict[str, Any]] = [
    {"q": "Who fixes machines?", "expect": ["person:ada", "person:charles"]},
    {"q": "Who could help me with a broken conveyor belt?",
     "expect": ["person:ada", "person:charles"]},
    {"q": "Who could work together on a robotics marketing project?",
     "expect": ["person:ada", "person:grace"]},
    {"q": "I need two people to build a healthcare AI prototype. Who?",
     "expect": ["person:alan"]},
    {"q": "Who knows the most about automation?", "expect": ["person:katherine"]},
    {"q": "Someone who can sell things", "expect": ["person:grace"]},
    {"q": "Who is based in Dunmore?", "expect": ["person:grace", "person:katherine"]},
    {"q": "Which companies are represented here?",
     "expect": ["org:quenlow", "org:pellard", "org:harnley"]},
    {"q": "Who should welcome a newcomer?", "expect": ["person:mary"]},
    # The only question here that cannot be answered by finding one person: Ada and Grace
    # share nothing directly, and the way between them runs Ada - repair - Charles -
    # marketing - Grace. It is what `path_between` is for, and nothing else was testing it.
    {"q": "Who could introduce Ada Lovelace to someone who does marketing?",
     "expect": ["person:charles", "person:grace"]},
    {"q": "Nobody here does underwater welding. Who could?", "expect": []},
    {"q": "hi", "expect": []},
]
