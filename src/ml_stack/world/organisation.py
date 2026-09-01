"""An organised group that does not exist, as a graph, at any size.

`make(kind, size, seed)` invents one from a seed and returns a `World`: the graph in the
shape `ml_stack.graph.community` already uses, the people in it, and a persona for each --
so the store, the bench, the page and the ask loop take it unchanged, and the simulation
has somebody to be.

A company is one kind of organised group and not the only one that talks. Five kinds share
one schema and differ in structure and vocabulary:

- ``company``: departments under a CEO, reporting lines with sane spans, customers,
  partners, products, projects.
- ``community``: a Slack community of professionals -- members with day jobs at *different*
  invented organisations, interest groups with moderators, no reporting lines at all.
- ``university``: departments of labs, each led by a principal investigator who advises
  postdocs and students; grants and seminars.
- ``open-source``: one project of many repositories, maintainers and contributors, a
  release cadence, sponsors, employers.
- ``nonprofit``: programmes under an executive director, a board, volunteers, funders.

Everything downstream is kind-agnostic: a unit is whatever a person is ``part_of``, a
hierarchy is whatever ``reports_to`` or ``advises`` there is, and the questions module
reads the relations that exist rather than assuming a company's.

Nothing here is a real person or organisation; the only real things are cities.

This file is a catalogue of invented labels -- titles, events, programmes -- so the
name hook's shape rule is off here (no-real-names: shapes off); the exact list and the
recogniser still run.
"""

from __future__ import annotations

import datetime as _dt
import json
import random
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from ml_stack.world import World
from ml_stack.world.names import company_name, person_name, product_name, slug

__all__ = ["KINDS", "SIZES", "UNIT_KIND", "load", "make", "role_catalogue", "summary"]

SIZES = {"small": 50, "medium": 500, "large": 5000}
"""How many people each size has."""

KINDS = ("company", "community", "university", "open-source", "nonprofit")
"""The organised groups this module can invent."""

UNIT_KIND = {"company": "department", "community": "group", "university": "lab",
             "open-source": "repo", "nonprofit": "programme"}
"""The node kind a person of each organisation is ``part_of``."""

TODAY = _dt.date(2026, 8, 31)
"""The day the world is made on; tenure and calendars count from here."""

# real cities, the one real thing here
CITIES: tuple[tuple[str, str], ...] = (
    ("Lisbon", "Portugal"), ("Turin", "Italy"), ("Leeds", "United Kingdom"),
    ("Tallinn", "Estonia"), ("Porto", "Portugal"), ("Kraków", "Poland"),
    ("Toronto", "Canada"), ("Denver", "United States"), ("Austin", "United States"),
    ("Montréal", "Canada"), ("Raleigh", "United States"), ("Portland", "United States"),
    ("Nairobi", "Kenya"), ("Cape Town", "South Africa"), ("Accra", "Ghana"),
    ("Bengaluru", "India"), ("Pune", "India"), ("Singapore", "Singapore"),
    ("Melbourne", "Australia"), ("Wellington", "New Zealand"), ("Osaka", "Japan"),
    ("Seoul", "South Korea"), ("Taipei", "Taiwan"), ("São Paulo", "Brazil"),
    ("Medellín", "Colombia"), ("Montevideo", "Uruguay"), ("Valencia", "Spain"),
    ("Lyon", "France"), ("Ghent", "Belgium"), ("Gothenburg", "Sweden"),
)

INDUSTRIES: tuple[tuple[str, str], ...] = (
    # industry, what the company's products are called in a sentence
    ("industrial robotics", "robot arms and the controllers that drive them"),
    ("medical imaging software", "imaging analysis software for hospital groups"),
    ("logistics software", "warehouse and fleet software for freight operators"),
    ("fintech", "payment and reconciliation services for mid-sized merchants"),
    ("renewable energy", "battery storage systems and the software that schedules them"),
    ("agricultural technology", "soil sensors and irrigation controllers"),
    ("developer tools", "build systems and observability tools for engineering teams"),
    ("education technology", "course platforms for vocational colleges"),
    ("cybersecurity", "identity and access products for regulated industries"),
    ("construction technology", "site survey drones and planning software"),
    ("consumer hardware", "e-ink readers and the publishing platform behind them"),
    ("telecommunications", "private 5G networks for ports and mines"),
)

# --- the role catalogue ---------------------------------------------------------------
#
# A department: what it does, its individual-contributor tracks with what each is
# responsible for, the skills its people have, and its share of a company's headcount.
# The community and the open-source project borrow the tracks as day jobs.
DEPARTMENTS: dict[str, dict[str, Any]] = {
    "engineering": {
        "weight": 30, "does": "builds and runs the product: services, firmware, the release train.",
        "tracks": [("Software Engineer", "write and review the services behind the product"),
                   ("Site Reliability Engineer", "keep production up and the on-call rota sane"),
                   ("Firmware Engineer", "write the code that runs on the hardware itself"),
                   ("Quality Engineer", "build the test rigs and decide what ships"),
                   ("Platform Engineer", "own the build system, CI and the internal tooling")],
        "skills": ["distributed systems", "embedded firmware", "kubernetes", "python", "rust",
                   "testing", "observability", "databases", "networking", "performance tuning"]},
    "product": {
        "weight": 6, "does": "decides what gets built next and why, and writes it down.",
        "tracks": [("Product Manager", "own a roadmap and the customer conversations behind it"),
                   ("Technical Writer", "write the documentation customers actually read"),
                   ("Product Analyst", "measure whether a release did what it promised")],
        "skills": ["roadmapping", "user research", "specifications", "pricing", "analytics",
                   "documentation"]},
    "design": {
        "weight": 4, "does": "shapes how the product looks and how it is used.",
        "tracks": [("Product Designer", "design the flows and screens people use"),
                   ("UX Researcher", "watch people use the product and report what broke"),
                   ("Brand Designer", "keep everything the company shows looking like one company")],
        "skills": ["interaction design", "prototyping", "typography", "accessibility",
                   "design systems", "usability testing"]},
    "sales": {
        "weight": 12, "does": "finds customers, closes them, and keeps them renewing.",
        "tracks": [("Account Executive", "run deals from first call to signature"),
                   ("Sales Engineer", "prove the product works in the customer's environment"),
                   ("Sales Development Representative", "book the first meeting"),
                   ("Account Manager", "keep existing customers happy and growing")],
        "skills": ["negotiation", "demos", "forecasting", "enterprise procurement",
                   "channel partnerships", "renewals"]},
    "marketing": {
        "weight": 5, "does": "makes the market aware the product exists and why it matters.",
        "tracks": [("Content Marketer", "write what the company publishes"),
                   ("Growth Marketer", "run the campaigns and read the numbers"),
                   ("Product Marketer", "turn a release into a story a customer follows"),
                   ("Events Marketer", "put the company at trade fairs and run its own")],
        "skills": ["campaigns", "copywriting", "SEO", "positioning", "events", "analytics"]},
    "support": {
        "weight": 10, "does": "answers customers when something goes wrong, and says why.",
        "tracks": [("Support Engineer", "reproduce what a customer hit and get it fixed"),
                   ("Customer Success Manager", "get a new customer live and keep them there"),
                   ("Technical Account Manager", "are one named customer's way into engineering")],
        "skills": ["troubleshooting", "onboarding", "escalations", "customer training",
                   "ticket triage", "SQL"]},
    "finance": {
        "weight": 4, "does": "counts the money, closes the books, and plans the year.",
        "tracks": [("Accountant", "close the month and keep the ledgers right"),
                   ("Financial Analyst", "model the plan and explain the variance"),
                   ("Payroll Specialist", "pay everyone correctly, everywhere, on time")],
        "skills": ["month-end close", "forecasting", "payroll", "audit", "spreadsheets",
                   "revenue recognition"]},
    "people": {
        "weight": 4, "does": "hires, onboards, pays fairly and keeps the place humane.",
        "tracks": [("Recruiter", "find the people the company needs next"),
                   ("People Partner", "advise managers and handle the hard conversations"),
                   ("Learning Lead", "run onboarding and management training")],
        "skills": ["recruiting", "onboarding", "compensation", "employment law",
                   "coaching", "performance reviews"]},
    "legal": {
        "weight": 2, "does": "writes the contracts and keeps the company inside the law.",
        "tracks": [("Commercial Counsel", "negotiate customer and partner contracts"),
                   ("Privacy Counsel", "keep the handling of personal data lawful"),
                   ("Paralegal", "keep the contracts filed, signed and findable")],
        "skills": ["contracts", "data protection", "licensing", "compliance",
                   "intellectual property"]},
    "operations": {
        "weight": 6, "does": "runs the offices, the suppliers and the logistics.",
        "tracks": [("Operations Manager", "run an office and everything that arrives at it"),
                   ("Supply Chain Analyst", "keep components arriving before they are needed"),
                   ("Facilities Coordinator", "keep the buildings working"),
                   ("Procurement Specialist", "buy what the company needs at a sane price")],
        "skills": ["procurement", "logistics", "vendor management", "facilities",
                   "inventory", "scheduling"]},
    "data": {
        "weight": 5, "does": "turns what the product records into what the company knows.",
        "tracks": [("Data Engineer", "build the pipelines everything else reads from"),
                   ("Data Scientist", "build the models and say how sure to be of them"),
                   ("Analytics Engineer", "keep the metrics meaning the same thing twice")],
        "skills": ["data pipelines", "machine learning", "statistics", "SQL", "dbt",
                   "experimentation", "data governance"]},
    "security": {
        "weight": 3, "does": "keeps attackers out and proves it to auditors.",
        "tracks": [("Security Engineer", "find the holes before somebody else does"),
                   ("Compliance Analyst", "keep the certifications current"),
                   ("Incident Responder", "run the response when something gets in")],
        "skills": ["threat modelling", "penetration testing", "incident response",
                   "compliance", "identity management", "cryptography"]},
    "research": {
        "weight": 3, "does": "works on what the product will need in three years.",
        "tracks": [("Research Scientist", "publish, prototype and hand over what works"),
                   ("Research Engineer", "turn a paper into something that runs")],
        "skills": ["machine learning", "signal processing", "optimisation", "simulation",
                   "computer vision", "control theory"]},
}

# IC1 to IC5, and the words on the business card
IC_LEVELS = (("IC1", "Associate "), ("IC2", ""), ("IC3", "Senior "), ("IC4", "Staff "),
             ("IC5", "Principal "))
IC_WEIGHTS = (18, 34, 30, 13, 5)

# a C-level and the departments under it; companies of a few hundred and up have these
C_LEVELS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("Chief Technology Officer", "CTO", ("engineering", "data", "security", "research")),
    ("Chief Product Officer", "CPO", ("product", "design")),
    ("Chief Revenue Officer", "CRO", ("sales", "marketing", "support")),
    ("Chief Financial Officer", "CFO", ("finance", "legal", "operations")),
    ("Chief People Officer", "CPeO", ("people",)),
)

ACADEMIC: dict[str, dict[str, Any]] = {
    "computer science": {"does": "studies computation, from compilers to learning systems.",
                         "skills": ["compilers", "machine learning", "distributed systems",
                                    "formal verification", "computer vision", "robotics"]},
    "physics": {"does": "studies matter and energy at every scale that will hold still.",
                "skills": ["condensed matter", "optics", "particle physics", "simulation",
                           "quantum computing", "instrumentation"]},
    "biology": {"does": "studies living systems, from proteins to ecosystems.",
                "skills": ["genomics", "microscopy", "ecology", "protein folding",
                           "bioinformatics", "cell culture"]},
    "chemistry": {"does": "studies what molecules do and how to make new ones.",
                  "skills": ["synthesis", "spectroscopy", "catalysis", "electrochemistry",
                             "materials", "computational chemistry"]},
    "mathematics": {"does": "proves things, some of which turn out to be useful.",
                    "skills": ["topology", "number theory", "statistics", "optimisation",
                               "combinatorics", "numerical analysis"]},
    "economics": {"does": "studies how people and markets decide.",
                  "skills": ["econometrics", "game theory", "labour economics",
                             "development economics", "causal inference"]},
    "linguistics": {"does": "studies language: its sounds, structure and use.",
                    "skills": ["phonology", "syntax", "corpus linguistics",
                               "language acquisition", "computational linguistics"]},
    "engineering": {"does": "designs things that have to work outside the lab.",
                    "skills": ["control theory", "structural analysis", "signal processing",
                               "fluid dynamics", "power electronics", "additive manufacturing"]},
    "psychology": {"does": "studies minds by watching what people do.",
                   "skills": ["cognitive science", "perception", "experiment design",
                              "neuroimaging", "developmental psychology"]},
    "history": {"does": "reads what was written down and argues about what it meant.",
                "skills": ["archival research", "economic history", "palaeography",
                           "oral history", "digital humanities"]},
}

# an open-source project: its repositories, and the skills that go with them
REPOS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("core", "the runtime everything else depends on", ("rust", "performance tuning", "memory safety")),
    ("parser", "read the input format and report what is wrong with it", ("parsing", "error messages", "grammars")),
    ("cli", "the command-line front end", ("argument parsing", "shell integration", "UX writing")),
    ("docs", "the manual, the tutorials and the site", ("documentation", "technical writing", "static sites")),
    ("web", "the browser front end", ("typescript", "accessibility", "web performance")),
    ("bindings-python", "the Python bindings", ("python", "C ABI", "packaging")),
    ("bindings-js", "the JavaScript bindings", ("javascript", "wasm", "packaging")),
    ("std", "the standard library", ("API design", "testing", "compatibility")),
    ("build", "the build system and CI", ("CI", "cross-compilation", "release engineering")),
    ("lsp", "the language server", ("language servers", "incremental computation", "editor integration")),
    ("fmt", "the formatter", ("pretty printing", "style guides", "idempotence")),
    ("pkg", "the package manager", ("dependency resolution", "registries", "lockfiles")),
    ("net", "the networking stack", ("async IO", "TLS", "protocols")),
    ("db", "the embedded database", ("storage engines", "B-trees", "durability")),
    ("gpu", "the GPU backend", ("shaders", "vulkan", "kernels")),
    ("mobile", "the mobile ports", ("android", "iOS", "cross-platform")),
    ("test", "the test framework and fuzzers", ("fuzzing", "property testing", "coverage")),
    ("bench", "the benchmark suite", ("benchmarking", "statistics", "regression tracking")),
    ("infra", "the project's own servers", ("hosting", "monitoring", "security")),
    ("i18n", "translations and locale support", ("localisation", "unicode", "text rendering")),
)

PROGRAMMES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("literacy", "run reading groups and tutoring in schools and libraries", ("tutoring", "curriculum", "child safeguarding")),
    ("food bank", "collects, stores and distributes food to people who need it", ("logistics", "food safety", "warehouse operations")),
    ("housing advice", "helps people keep or find somewhere to live", ("tenancy law", "casework", "benefits advice")),
    ("river restoration", "clears, replants and monitors the local waterways", ("ecology", "volunteer coordination", "water quality")),
    ("youth mentoring", "pairs young people with mentors for a year", ("mentoring", "safeguarding", "matching")),
    ("digital skills", "teaches older people to use the devices they were given", ("teaching", "patience", "accessibility")),
    ("legal clinic", "gives free first advice to people who cannot pay for it", ("family law", "immigration law", "triage")),
    ("community kitchen", "cooks and serves meals five nights a week", ("cooking", "food safety", "rota planning")),
    ("refugee welcome", "meets new arrivals and helps them through the first year", ("interpreting", "casework", "housing")),
    ("repair café", "fixes what people bring rather than letting it be thrown away", ("electronics repair", "sewing", "bike maintenance")),
    ("fundraising", "raises the money the other programmes spend", ("grant writing", "donor relations", "events")),
    ("communications", "tells the story so people give, volunteer and come", ("copywriting", "social media", "press")),
)

INTEREST_GROUPS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("robotics", "people who build and fix machines that move", ("robotics", "controls", "embedded firmware")),
    ("hiring", "who is hiring, who is looking, and how to do either well", ("recruiting", "interviewing", "negotiation")),
    ("founders", "people running something of their own, and the ones about to", ("fundraising", "pricing", "go-to-market")),
    ("data", "pipelines, model and the arguments about them", ("data pipelines", "machine learning", "statistics")),
    ("hardware", "circuit boards, enclosures and supply chains", ("PCB design", "manufacturing", "procurement")),
    ("healthcare", "software and devices used by clinicians", ("clinical data", "regulation", "medical devices")),
    ("climate", "energy, carbon and what to build about it", ("battery storage", "grid software", "carbon accounting")),
    ("design", "interfaces, brands and the research behind them", ("interaction design", "user research", "typography")),
    ("security", "keeping things out and finding out when they got in", ("penetration testing", "incident response", "identity management")),
    ("remote-work", "how to work well from wherever you are", ("async communication", "documentation", "time zones")),
    ("local-meetups", "who is in which city and when the next drinks are", ("events", "hosting", "logistics")),
    ("open-source", "maintaining things in public", ("maintenance", "release engineering", "community management")),
    ("careers", "changing jobs, changing fields, and what happened next", ("coaching", "CV writing", "career changes")),
    ("agriculture", "sensors, soil and the software between them", ("soil sensors", "irrigation", "agronomy")),
    ("logistics", "getting things from one place to another on time", ("warehousing", "fleet routing", "customs")),
    ("education", "teaching, platforms and what actually helps people learn", ("curriculum", "course platforms", "assessment")),
    ("finance", "money, its software and its regulators", ("payments", "reconciliation", "compliance")),
    ("manufacturing", "factories, lines and the machines on them", ("maintenance", "quality control", "CNC")),
    ("legal", "contracts, licences and privacy, for people who are not lawyers", ("contracts", "licensing", "data protection")),
    ("writing", "documentation, essays and getting either finished", ("technical writing", "editing", "publishing")),
)

VOICES = (
    "Terse. Short sentences, no greetings, often just the answer.",
    "Warm and chatty; opens with a greeting and thanks people by name.",
    "Formal, full sentences, no contractions, signs off every message.",
    "Uses emoji liberally and exclamation marks more than most.",
    "Asks a clarifying question before answering almost anything.",
    "Writes long paragraphs that get to the point in the last sentence.",
    "Bullet points for everything, even a two-item answer.",
    "Dry humour; understatement; never uses an exclamation mark.",
    "Precise about numbers and dates, vague about feelings.",
    "Lower-case, no punctuation to speak of, quick replies.",
    "Apologises for interrupting, then says something worth hearing.",
    "Quotes what somebody else said before replying to it.",
    "Direct to the point of bluntness, but always says why.",
    "Hedges everything with 'I think' and 'maybe', and is usually right.",
    "Writes as if dictating: run-on sentences, commas everywhere.",
    "Cheerful, encouraging, ends with a question to keep things going.",
)

DEFAULT_QUESTIONS = 40


# --- the accumulator -----------------------------------------------------------------------

class _Build:
    """Nodes, edges and quotes as they are made, in insertion order, so a seed reproduces them."""

    def __init__(self, rng: random.Random) -> None:
        self.rng = rng
        self.nodes: dict[str, dict[str, Any]] = {}
        self.edges: list[dict[str, Any]] = []
        self.joined: set[tuple[str, str, str]] = set()
        self.said: dict[str, list[str]] = {}
        self.people: list[str] = []
        self.names: set[tuple[str, str]] = set()

    def node(self, node_id: str, kind: str, label: str, **attrs: Any) -> str:
        if node_id in self.nodes:
            return node_id
        self.nodes[node_id] = {"id": node_id, "label": label, "kind": kind, "mentions": 0,
                               "attrs": {"member": kind == "person", **attrs}, "messages": []}
        return node_id

    def edge(self, source: str, rel: str, target: str, weight: int = 1) -> None:
        key = (source, rel, target)
        if key in self.joined or source == target:
            return
        self.joined.add(key)
        self.edges.append({"source": source, "rel": rel, "target": target, "weight": weight,
                           "messages": []})

    def person(self, **attrs: Any) -> str:
        """A new person with a name nobody in this world has yet."""
        while True:
            given, family = person_name(self.rng)
            if (given, family) not in self.names:
                break
        self.names.add((given, family))
        node_id = f"person:{slug(given)}-{slug(family)}"
        n = 2
        while node_id in self.nodes:
            node_id = f"person:{slug(given)}-{slug(family)}-{n}"
            n += 1
        self.node(node_id, "person", f"{given} {family}", **attrs)
        self.people.append(node_id)
        return node_id

    def place(self, city: str, country: str) -> str:
        return self.node(f"place:{slug(city)}", "place", city, country=country)

    def topic(self, name: str) -> str:
        return self.node(f"topic:{slug(name)}", "topic", name)

    def org(self, label: str, **attrs: Any) -> str:
        node_id = f"org:{slug(label)}"
        n = 2
        while node_id in self.nodes:
            node_id = f"org:{slug(label)}-{n}"
            n += 1
        return self.node(node_id, "org", label, **attrs)

    def label(self, node_id: str) -> str:
        return str(self.nodes[node_id]["label"])

    def attrs(self, node_id: str) -> dict[str, Any]:
        return self.nodes[node_id]["attrs"]


# --- helpers shared by every kind -----------------------------------------------------------

def _split(rng: random.Random, total: int, weights: Sequence[float], minimum: int = 1) -> list[int]:
    """``total`` shared out in proportion to ``weights``, nobody below ``minimum``."""
    if not weights:
        return []
    floor = min(minimum, total // len(weights))
    spare = total - floor * len(weights)
    whole = sum(weights)
    raw = [spare * w / whole for w in weights]
    out = [floor + int(r) for r in raw]
    left = total - sum(out)
    for i in sorted(range(len(raw)), key=lambda i: -(raw[i] - int(raw[i])))[:left]:
        out[i] += 1
    return out


def _tree(rng: random.Random, members: list[str], head: str,
          low: int = 5, high: int = 9) -> dict[str, str]:
    """Reporting lines from ``members`` up to ``head``, every span between ``low`` and ``high``.

    Level by level: parents at one level take a span each from what is left; when what is
    left fits in one level, only as many parents as keep the spans sane are used and the rest
    of that level stay leaves. Returns ``child -> parent``.
    """
    parents = {}
    level = [head]
    left = list(members)
    while left:
        if len(left) <= high * len(level):
            k = max(1, min(len(level), -(-len(left) // ((low + high) // 2))))
            takers = level[:k]
            shares = _split(rng, len(left), [1] * k)
        else:
            takers = level
            shares = [rng.randint(low, high) for _ in level]
        nxt = []
        for parent, share in zip(takers, shares):
            for _ in range(share):
                if not left:
                    break
                child = left.pop(0)
                parents[child] = parent
                nxt.append(child)
        level = nxt
    return parents


def _heights(parents: Mapping[str, str]) -> dict[str, int]:
    """How far above the leaves each node stands: 0 for a leaf."""
    kids: dict[str, list[str]] = {}
    for child, parent in parents.items():
        kids.setdefault(parent, []).append(child)
    heights: dict[str, int] = {}

    def height(node: str) -> int:
        if node not in heights:
            heights[node] = 1 + max((height(k) for k in kids.get(node, ())), default=-1)
        return heights[node]

    for node in list(parents) + list(kids):
        height(node)
    return heights


def _ic_level(rng: random.Random) -> tuple[str, str]:
    return rng.choices(IC_LEVELS, IC_WEIGHTS)[0]


def _started(rng: random.Random, founded: int, level_bias: float = 0.0) -> tuple[str, float]:
    """A start date between the founding and today, and the tenure in years it implies."""
    first = _dt.date(max(founded, 1900), 1, 1)
    span = (TODAY - first).days
    # senior people have usually been around longer; the bias pulls the date back
    frac = rng.random() ** (1 + level_bias)
    day = TODAY - _dt.timedelta(days=int(span * frac))
    return day.isoformat(), round((TODAY - day).days / 365.25, 1)


def _offices(b: _Build, n: int) -> list[str]:
    cities = b.rng.sample(CITIES, n)
    return [b.place(city, country) for city, country in cities]


def _skills(b: _Build, who: str, pool: Sequence[str], others: Sequence[str],
            n: tuple[int, int] = (2, 4)) -> list[str]:
    """Two to four skills from the pool, sometimes one from elsewhere. Returns topic ids."""
    chosen = b.rng.sample(list(pool), min(len(pool), b.rng.randint(*n)))
    if others and b.rng.random() < 0.25:
        chosen.append(b.rng.choice(list(others)))
    ids = [b.topic(s) for s in chosen]
    for t in ids:
        b.edge(who, "experienced_in", t, weight=b.rng.randint(1, 3))
    return ids


def _projects(b: _Build, people: Sequence[str], units: Sequence[str], n: int,
              *, prefix: str, naming: Sequence[str], skills_of: Mapping[str, Sequence[str]],
              size: tuple[int, int] = (4, 8), rel: str = "offers") -> list[str]:
    """``n`` pieces of work, each ``offers``-ed by a unit and worked on by a handful of people
    drawn from more than one unit. Each wants one or two skills. Returns the ids."""
    rng = b.rng
    out = []
    by_unit: dict[str, list[str]] = {}
    for p in people:
        by_unit.setdefault(b.attrs(p).get("unit_id", ""), []).append(p)
    for i in range(n):
        name = f"{rng.choice(naming)} {product_name(rng).split()[0]}"
        pid = b.node(f"opportunity:{slug(name)}-{i}", "opportunity", name)
        owner = rng.choice(list(units)) if units else ""
        if owner:
            b.edge(owner, rel, pid)
        want = set(skills_of.get(owner, ()) or ())
        team = []
        home = by_unit.get(owner, [])
        if home:
            team += rng.sample(home, min(len(home), max(2, rng.randint(*size) * 2 // 3)))
        while len(team) < rng.randint(*size) and len(team) < len(people):
            other = rng.choice(list(people))
            if other not in team:
                team.append(other)
        for who in team:
            b.edge(who, "works_on", pid, weight=rng.randint(1, 4))
            b.attrs(who).setdefault("projects", []).append(name)
        for s in rng.sample(sorted(want), min(len(want), rng.randint(1, 2))) if want else []:
            b.edge(pid, "wants", b.topic(s))
        # pairs on the same piece of work know each other
        for x in team:
            for y in team:
                if x < y and rng.random() < 0.6:
                    b.edge(x, "works_with", y, weight=rng.randint(1, 3))
        out.append(pid)
    return out


def _events(b: _Build, people: Sequence[str], names: Sequence[str], n: int,
            crowd: tuple[int, int], *, prefix: str = "event") -> list[str]:
    rng = b.rng
    out = []
    for i in range(n):
        name = names[i % len(names)]
        if i >= len(names):
            name = f"{name} {2020 + i // len(names)}"
        eid = b.node(f"{prefix}:{slug(name)}", "event", name,
                     day=rng.randint(-365, 60))
        for who in rng.sample(list(people), min(len(people), rng.randint(*crowd))):
            b.edge(who, "attended", eid)
        out.append(eid)
    return out


def _teams(b: _Build, groups: Iterable[Sequence[str]], p: float = 0.5) -> None:
    """People in the same small group work with a few of the others."""
    rng = b.rng
    for group in groups:
        group = list(group)
        for x in group:
            for y in rng.sample(group, min(len(group), 3)):
                if x != y and rng.random() < p:
                    b.edge(min(x, y), "works_with", max(x, y), weight=rng.randint(1, 3))


def _mentors(b: _Build, seniors: Sequence[str], juniors: Sequence[str], share: float = 0.08) -> None:
    rng = b.rng
    if not seniors or not juniors:
        return
    for junior in juniors:
        if rng.random() < share:
            b.edge(rng.choice(list(seniors)), "mentors", junior)


def _employers(b: _Build, people: Sequence[str], n_orgs: int, offices: Sequence[str]) -> list[str]:
    """Day jobs at many invented organisations, for the kinds that have no single employer."""
    rng = b.rng
    orgs = [b.org(company_name(rng), industry=rng.choice(INDUSTRIES)[0]) for _ in range(n_orgs)]
    for o in orgs:
        b.edge(o, "based_in", rng.choice(list(offices)))
    at: dict[str, list[str]] = {}
    for who in people:
        # a few big employers and a long tail, the way a community actually is
        o = orgs[min(int(rng.expovariate(1.0) * n_orgs / 3), n_orgs - 1)]
        b.edge(who, "works_at", o)
        b.attrs(who)["employer"] = b.label(o)
        at.setdefault(o, []).append(who)
    _teams(b, at.values(), p=0.35)
    return orgs


def _day_job(b: _Build, who: str) -> tuple[str, str, str]:
    """A title, level and responsibility borrowed from the company catalogue."""
    dept = b.rng.choice(list(DEPARTMENTS))
    track, does = b.rng.choice(DEPARTMENTS[dept]["tracks"])
    level, word = _ic_level(b.rng)
    return f"{word}{track}", level, does


# --- company -------------------------------------------------------------------------------

def _company(b: _Build, n: int, offices: list[str]) -> dict[str, Any]:
    rng = b.rng
    industry, makes = rng.choice(INDUSTRIES)
    founded = rng.randint(1998, 2018)
    name = company_name(rng)
    org = b.org(name, industry=industry, founded=founded, headquarters=b.label(offices[0]),
                type="company")
    products = [product_name(rng) for _ in range(rng.randint(2, 5))]
    for p in products:
        b.edge(org, "makes", b.node(f"product:{slug(p)}", "product", p, of=makes))

    how_many = {"small": 6, "medium": 10}.get(_size_of(n), len(DEPARTMENTS))
    names = sorted(DEPARTMENTS, key=lambda d: -DEPARTMENTS[d]["weight"])[:how_many]
    units = {d: b.node(f"department:{slug(d)}", "department", d.title(),
                       does=DEPARTMENTS[d]["does"]) for d in names}
    for u in units.values():
        b.edge(u, "part_of", org)

    def hire(**attrs: Any) -> str:
        started, tenure = _started(rng, founded, attrs.pop("bias", 0.0))
        who = b.person(started=started, tenure_years=tenure, **attrs)
        b.edge(who, "works_at", org)
        return who

    executive = b.node("department:executive", "department", "Executive",
                       does="runs the company: the CEO and the officers who report to them.")
    b.edge(executive, "part_of", org)
    ceo = hire(title="Chief Executive Officer", level="c-level", bias=3.0, unit="Executive",
               does="run the company and answer to the board for it")
    b.edge(ceo, "part_of", executive)
    parents: dict[str, str] = {}
    heads: dict[str, str] = {}
    executives = [ceo]
    if n >= 200:
        for title, short, depts in C_LEVELS:
            if any(d in units for d in depts):
                c = hire(title=title, level="c-level", bias=2.5, unit="Executive",
                         does=f"run {', '.join(d for d in depts if d in units)} as {short}")
                parents[c] = ceo
                b.edge(c, "part_of", executive)
                executives.append(c)
                for d in depts:
                    heads[d] = c
    counts = _split(rng, n - len(executives), [DEPARTMENTS[d]["weight"] for d in names], 3)
    seniors: list[str] = []
    juniors: list[str] = []
    for dept, count in zip(names, counts):
        unit = units[dept]
        tracks = DEPARTMENTS[dept]["tracks"]
        pool = DEPARTMENTS[dept]["skills"]
        elsewhere = [s for d in names if d != dept for s in DEPARTMENTS[d]["skills"]]
        members = [hire(unit=b.label(unit), unit_id=unit) for _ in range(count)]
        head, rest = members[0], members[1:]
        lines = _tree(rng, rest, head)
        parents.update(lines)
        parents[head] = heads.get(dept, ceo)
        heights = _heights(lines)
        for who in members:
            b.edge(who, "part_of", unit)
            h = heights.get(who, 0)
            a = b.attrs(who)
            if who == head:
                a["title"] = (f"VP {dept.title()}" if h >= 2 else f"Head of {dept.title()}")
                a["level"], a["does"] = ("vp" if h >= 2 else "director"), \
                    f"lead {dept} and answer for it"
                a["bias"] = 2.0
            elif h >= 2:
                a["title"], a["level"] = f"Director of {dept.title()}", "director"
                a["does"] = f"run several {dept} teams and their managers"
            elif h == 1:
                a["title"], a["level"] = f"{dept.title()} Manager", "manager"
                a["does"] = f"manage one {dept} team day to day"
            else:
                track, does = rng.choice(tracks)
                level, word = _ic_level(rng)
                a["title"], a["level"], a["does"] = f"{word}{track}", level, does
            (seniors if a["level"] in ("IC4", "IC5", "manager") else juniors).append(who)
            _skills(b, who, pool, elsewhere)
        # a team is a manager's reports
        teams: dict[str, list[str]] = {}
        for child, parent in lines.items():
            teams.setdefault(parent, []).append(child)
        _teams(b, teams.values())
    for child, parent in parents.items():
        b.edge(child, "reports_to", parent)
        b.attrs(child)["manager"] = b.label(parent)
    _mentors(b, seniors, [j for j in juniors if b.attrs(j).get("level") in ("IC1", "IC2")])

    # where people sit: most at the head office, some remote
    for who in b.people:
        if rng.random() < 0.12:
            b.attrs(who)["remote"] = True
        else:
            b.edge(who, "based_in", rng.choices(offices, [3] + [1] * (len(offices) - 1))[0])

    # customers and partners, invented, in cities that may or may not be offices
    everywhere = [b.place(c, k) for c, k in CITIES]
    for _ in range(max(3, n // 40)):
        c = b.org(company_name(rng), industry=rng.choice(INDUSTRIES)[0], type="customer")
        b.edge(c, "customer_of", org)
        b.edge(c, "based_in", rng.choice(everywhere))
        for who in rng.sample([p for p in b.people if b.attrs(p).get("unit") in ("Sales", "Support")]
                              or b.people[:1], 1):
            b.edge(who, "serves", c)
    for _ in range(max(2, n // 150)):
        p = b.org(company_name(rng), industry=rng.choice(INDUSTRIES)[0], type="partner")
        b.edge(p, "partner_of", org)
        b.edge(p, "based_in", rng.choice(everywhere))

    skills_of = {units[d]: DEPARTMENTS[d]["skills"] for d in names}
    _projects(b, b.people, list(units.values()), max(4, n // 12), prefix="project",
              naming=("Project", "Initiative", "Migration", "Launch", "Rewrite", "Pilot"),
              skills_of=skills_of)
    _events(b, b.people, ("All Hands", "Engineering Offsite", "Sales Kickoff", "Hack Week",
                          "Customer Summit", "Product Launch", "Winter Party", "Trade Fair"),
            max(3, min(20, n // 25)), (6, 12) if n < 200 else (10, 40))
    return {"organisation": org, "root": ceo}


def _size_of(n: int) -> str:
    return next((s for s, count in SIZES.items() if count == n), "custom")


# --- community -------------------------------------------------------------------------------

def _community(b: _Build, n: int, offices: list[str]) -> dict[str, Any]:
    rng = b.rng
    name = f"{company_name(rng, kind='')} Network"
    org = b.org(name, founded=rng.randint(2015, 2022), type="community",
                headquarters="Slack")
    cities = [b.place(c, k) for c, k in CITIES]
    how_many = {"small": 6, "medium": 14}.get(_size_of(n), len(INTEREST_GROUPS))
    groups = rng.sample(INTEREST_GROUPS, how_many)
    units = {g: b.node(f"group:{slug(g)}", "group", f"#{g}", does=does)
             for g, does, _ in groups}
    for u in units.values():
        b.edge(u, "part_of", org)
    skills_of = {units[g]: list(s) for g, _, s in groups}

    for _ in range(n):
        title, level, does = _day_job(b, "")
        started, tenure = _started(rng, b.attrs(org)["founded"])
        who = b.person(title=title, level=level, does=does, started=started,
                       tenure_years=tenure)
        b.edge(who, "based_in", rng.choice(cities))
    orgs = _employers(b, b.people, max(4, n // 3), cities)

    members: dict[str, list[str]] = {u: [] for u in units.values()}
    for who in b.people:
        mine = rng.sample(list(units.values()), rng.randint(1, min(3, len(units))))
        b.attrs(who)["unit"] = b.label(mine[0])
        b.attrs(who)["unit_id"] = mine[0]
        for u in mine:
            b.edge(who, "part_of", u)
            members[u].append(who)
        pool = [s for u in mine for s in skills_of[u]]
        others = [s for u in units.values() if u not in mine for s in skills_of[u]]
        _skills(b, who, pool, others)
        if rng.random() < 0.15:
            b.edge(who, "seeks", b.topic(rng.choice(others or pool)))
    for u, crowd in members.items():
        for mod in rng.sample(crowd, min(len(crowd), rng.randint(1, 3))):
            b.edge(mod, "moderates", u)
            b.attrs(mod)["moderates"] = b.label(u)
    # employers offer work, the way orgs in the bench community do
    for o in rng.sample(orgs, max(2, len(orgs) // 4)):
        want = rng.choice(list(skills_of.values()))
        name = f"{rng.choice(('hiring for', 'contract:', 'looking for help with'))} {rng.choice(want)}"
        opp = b.node(f"opportunity:{slug(name)}-{slug(b.label(o))}", "opportunity", name)
        b.edge(o, "offers", opp)
        b.edge(opp, "wants", b.topic(rng.choice(want)))
    _mentors(b, [p for p in b.people if b.attrs(p)["level"] in ("IC4", "IC5")],
             [p for p in b.people if b.attrs(p)["level"] in ("IC1", "IC2")], 0.1)
    _projects(b, b.people, list(units.values()), max(3, n // 20), prefix="project",
              naming=("Community", "Working group:", "Meetup series:", "Reading group:"),
              skills_of=skills_of, size=(3, 6))
    _events(b, b.people, ("Makers Night", "Spring Meetup", "Demo Day", "Careers Evening",
                          "Founders Breakfast", "Winter Social", "Lightning Talks",
                          "Hardware Hack Day"),
            max(3, min(20, n // 20)), (6, 12) if n < 200 else (10, 40))
    return {"organisation": org, "root": ""}


# --- university ------------------------------------------------------------------------------

def _university(b: _Build, n: int, offices: list[str]) -> dict[str, Any]:
    rng = b.rng
    founded = rng.randint(1850, 1975)
    name = f"{company_name(rng, kind='')} University"
    org = b.org(name, founded=founded, headquarters=b.label(offices[0]), type="university")
    how_many = {"small": 3, "medium": 6}.get(_size_of(n), len(ACADEMIC))
    fields = rng.sample(list(ACADEMIC), how_many)
    departments = {f: b.node(f"department:{slug(f)}", "department", f"Department of {f.title()}",
                             does=ACADEMIC[f]["does"]) for f in fields}
    for d in departments.values():
        b.edge(d, "part_of", org)

    def enrol(**attrs: Any) -> str:
        started, tenure = _started(rng, max(founded, 1990), attrs.pop("bias", 0.0))
        who = b.person(started=started, tenure_years=tenure, **attrs)
        b.edge(who, "works_at", org)
        b.edge(who, "based_in", rng.choices(offices, [4] + [1] * (len(offices) - 1))[0])
        return who

    admin = b.node("department:administration", "department", "Administration",
                   does="runs the university: the dean, the registry and the finance office.")
    b.edge(admin, "part_of", org)
    dean = enrol(title="Dean of Research", level="dean", unit="Administration", bias=3.0,
                 does="answer for every department's research and its money")
    b.edge(dean, "part_of", admin)
    labs: dict[str, list[str]] = {}
    skills_of: dict[str, list[str]] = {}
    counts = _split(rng, n - 1, [1] * len(fields), 4)
    faculty: list[str] = []
    students: list[str] = []
    for field, count in zip(fields, counts):
        dept = departments[field]
        pool = ACADEMIC[field]["skills"]
        elsewhere = [s for f in fields if f != field for s in ACADEMIC[f]["skills"]]
        chair = ""
        left = count
        i = 0
        while left > 0:
            size = min(left, rng.randint(5, 12))
            topic = rng.choice(pool)
            lab = b.node(f"lab:{slug(topic)}-{slug(field)}-{i}", "lab",
                         f"{topic.title()} Lab", does=f"studies {topic} within {field}")
            b.edge(lab, "part_of", dept)
            skills_of[lab] = pool
            rank = rng.choice(("Professor", "Associate Professor", "Assistant Professor"))
            pi = enrol(title=rank, level="faculty", unit=b.label(lab), unit_id=lab, bias=2.0,
                       does="lead the lab, win its grants and advise its students")
            b.edge(pi, "leads", lab)
            faculty.append(pi)
            crowd = [pi]
            if not chair:
                chair = pi
                b.edge(pi, "chairs", dept)
                b.attrs(pi)["title"] = f"{rank}, Chair"
            b.attrs(pi)["manager"] = b.label(dean)
            b.edge(pi, "reports_to", dean)
            for j in range(size - 1):
                role = rng.choices((("Postdoctoral Researcher", "postdoc", "run the experiments and write the papers up"),
                                    ("PhD Student", "student", "do the research a thesis is made of"),
                                    ("Master's Student", "student", "do a year of research and a dissertation"),
                                    ("Research Engineer", "staff", "build the software and rigs the lab runs on"),
                                    ("Lab Manager", "staff", "keep the lab ordered, safe and funded")),
                                   (2, 5, 2, 1, 1))[0]
                who = enrol(title=role[0], level=role[1], unit=b.label(lab), unit_id=lab,
                            does=role[2])
                b.edge(pi, "advises", who)
                b.attrs(who)["manager"] = b.label(pi)
                crowd.append(who)
                (students if role[1] == "student" else faculty).append(who)
            for who in crowd:
                b.edge(who, "part_of", lab)
                _skills(b, who, pool, elsewhere)
            labs[lab] = crowd
            left -= size
            i += 1
    _teams(b, labs.values())
    _mentors(b, [p for p in faculty if b.attrs(p)["level"] == "postdoc"], students, 0.15)
    everywhere = [b.place(c, k) for c, k in CITIES]
    for _ in range(max(2, n // 60)):
        f = b.org(company_name(rng, kind=rng.choice(("Foundation", "Trust", "Council"))),
                  type="funder")
        b.edge(f, "based_in", rng.choice(everywhere))
        for lab in rng.sample(list(labs), min(len(labs), rng.randint(1, 3))):
            b.edge(f, "funds", lab)
    _projects(b, b.people, list(labs), max(3, n // 10), prefix="grant",
              naming=("Grant:", "Study:", "Consortium:", "Fellowship:"), skills_of=skills_of,
              size=(3, 6))
    _events(b, b.people, ("Departmental Seminar", "Graduate Symposium", "Open Day",
                          "Summer School", "Thesis Defences", "Faculty Retreat",
                          "Annual Conference", "Poster Session"),
            max(3, min(20, n // 25)), (6, 12) if n < 200 else (10, 40))
    return {"organisation": org, "root": dean}


# --- open-source -------------------------------------------------------------------------------

def _open_source(b: _Build, n: int, offices: list[str]) -> dict[str, Any]:
    rng = b.rng
    name = product_name(rng).split()[0]
    org = b.org(f"{name} Project", founded=rng.randint(2009, 2021), type="open-source",
                headquarters="GitHub")
    b.edge(org, "makes", b.node(f"product:{slug(name)}", "product", name,
                                of="a programming toolchain"))
    how_many = {"small": 4, "medium": 12}.get(_size_of(n), len(REPOS))
    repos = [REPOS[0], *rng.sample(REPOS[1:], how_many - 1)]
    units = {r: b.node(f"repo:{slug(r)}", "repo", f"{name}/{r}", does=does)
             for r, does, _ in repos}
    for u in units.values():
        b.edge(u, "part_of", org)
    skills_of = {units[r]: list(s) for r, _, s in repos}
    cities = [b.place(c, k) for c, k in CITIES]

    leads = []
    for _ in range(1 if n < 200 else 3):
        started, tenure = _started(rng, b.attrs(org)["founded"], 3.0)
        who = b.person(title="Lead Maintainer", level="lead", started=started,
                       tenure_years=tenure, unit=b.label(units[repos[0][0]]),
                       unit_id=units[repos[0][0]],
                       does="decide what the project is and settle what nobody else can")
        b.edge(who, "based_in", rng.choice(cities))
        leads.append(who)
    roles = (("Core Maintainer", "core", "merge across repositories and cut releases", 6),
             ("Maintainer", "maintainer", "own one repository's issues and review", 12),
             ("Release Manager", "core", "run the release train and its checklist", 2),
             ("Regular Contributor", "contributor", "land a change most months", 30),
             ("Occasional Contributor", "contributor", "fix what got in the way of their own work", 40),
             ("Documentation Writer", "contributor", "keep the manual honest", 5),
             ("Triager", "contributor", "label, reproduce and close what is not a bug", 5))
    for _ in range(n - len(leads)):
        title, level, does, _w = rng.choices(roles, [r[3] for r in roles])[0]
        started, tenure = _started(rng, b.attrs(org)["founded"], 1.0 if level != "contributor" else 0)
        home = rng.choice(list(units.values()))
        who = b.person(title=title, level=level, does=does, started=started, tenure_years=tenure,
                       unit=b.label(home), unit_id=home)
        b.edge(who, "based_in", rng.choice(cities))
    orgs = _employers(b, b.people, max(4, n // 3), cities)
    for o in rng.sample(orgs, max(1, len(orgs) // 8)):
        b.edge(o, "sponsors", org)

    for who in b.people:
        a = b.attrs(who)
        home = a["unit_id"]
        b.edge(who, "part_of", home)
        mine = [home]
        if a["level"] in ("lead", "core"):
            mine += rng.sample(list(units.values()), min(len(units), rng.randint(2, 4)))
            for u in dict.fromkeys(mine):
                b.edge(who, "maintains", u)
        elif a["level"] == "maintainer":
            b.edge(who, "maintains", home)
        else:
            mine += rng.sample(list(units.values()), min(len(units), rng.randint(0, 2)))
            for u in dict.fromkeys(mine):
                b.edge(who, "contributes_to", u)
        pool = [s for u in dict.fromkeys(mine) for s in skills_of[u]]
        others = [s for u in units.values() if u not in mine for s in skills_of[u]]
        _skills(b, who, pool, others)
    by_repo: dict[str, list[str]] = {}
    for e in b.edges:
        if e["rel"] in ("maintains", "contributes_to"):
            by_repo.setdefault(e["target"], []).append(e["source"])
    _teams(b, [crowd for crowd in by_repo.values()], p=0.3)
    _mentors(b, [p for p in b.people if b.attrs(p)["level"] in ("lead", "core", "maintainer")],
             [p for p in b.people if b.attrs(p)["level"] == "contributor"], 0.1)
    _projects(b, b.people, list(units.values()), max(4, n // 10), prefix="rfc",
              naming=("RFC:", "Tracking issue:", "Milestone:", "Epic:"), skills_of=skills_of,
              size=(3, 6))
    # a release cadence: one event per release, most recent first
    major, minor = rng.randint(1, 4), rng.randint(0, 9)
    for i in range(max(4, min(20, n // 25))):
        version = f"v{major}.{minor}.{rng.randint(0, 3)}"
        rel = b.node(f"event:release-{slug(version)}-{i}", "event", f"Release {version}",
                     day=-42 * i, version=version)
        for who in rng.sample(b.people, min(len(b.people), rng.randint(4, 16))):
            b.edge(who, "attended", rel)
        minor -= 1
        if minor < 0:
            major, minor = max(0, major - 1), 9
    return {"organisation": org, "root": leads[0]}


# --- nonprofit ----------------------------------------------------------------------------------

def _nonprofit(b: _Build, n: int, offices: list[str]) -> dict[str, Any]:
    rng = b.rng
    founded = rng.randint(1980, 2015)
    name = company_name(rng, kind=rng.choice(("Trust", "Foundation", "Society", "Alliance")))
    org = b.org(name, founded=founded, headquarters=b.label(offices[0]), type="nonprofit")
    how_many = {"small": 3, "medium": 6}.get(_size_of(n), len(PROGRAMMES) - 2)
    chosen = rng.sample(PROGRAMMES[:-2], how_many - (1 if n >= 200 else 0))
    if n >= 200:
        chosen = [*chosen, rng.choice(PROGRAMMES[-2:])]
    units = {p: b.node(f"programme:{slug(p)}", "programme", p.title(), does=does)
             for p, does, _ in chosen}
    for u in units.values():
        b.edge(u, "part_of", org)
    skills_of = {units[p]: list(s) for p, _, s in chosen}

    def join(**attrs: Any) -> str:
        started, tenure = _started(rng, founded, attrs.pop("bias", 0.0))
        who = b.person(started=started, tenure_years=tenure, **attrs)
        b.edge(who, "works_at", org)
        b.edge(who, "based_in", rng.choices(offices, [3] + [1] * (len(offices) - 1))[0])
        return who

    office = b.node("programme:central-office", "programme", "Central Office",
                    does="runs the organisation: the director, finance and administration.")
    b.edge(office, "part_of", org)
    director = join(title="Executive Director", level="c-level", unit="Central Office",
                    bias=3.0, does="run the organisation and answer to the board for it")
    b.edge(director, "part_of", office)
    board = b.node("body:board", "body", "Board of Trustees",
                   does="governs the organisation and appoints its executive director")
    b.edge(board, "part_of", org)
    trustees = []
    for _ in range(rng.randint(5, 8)):
        title, level, does = _day_job(b, "")
        who = join(title=f"Trustee ({title})", level="board", unit="Board of Trustees",
                   does="sit on the board; by day, " + does)
        b.edge(who, "part_of", board)
        b.edge(who, "sits_on", board)
        b.edge(who, "advises", director)
        trustees.append(who)
    staff_share = 0.55
    left = n - 1 - len(trustees)
    counts = _split(rng, left, [1] * len(units), 3)
    staff: list[str] = []
    juniors: list[str] = []
    parents: dict[str, str] = {}
    for (prog, _d, _s), count in zip(chosen, counts):
        unit = units[prog]
        pool = skills_of[unit]
        elsewhere = [s for u, ss in skills_of.items() if u != unit for s in ss]
        paid = max(1, int(count * staff_share))
        people = [join(unit=b.label(unit), unit_id=unit) for _ in range(paid)]
        head, rest = people[0], people[1:]
        lines = _tree(rng, rest, head)
        parents.update(lines)
        parents[head] = director
        heights = _heights(lines)
        for who in people:
            a = b.attrs(who)
            h = heights.get(who, 0)
            if who == head:
                a["title"], a["level"], a["bias"] = f"Programme Director, {prog.title()}", "director", 2.0
                a["does"] = f"lead the {prog} programme and its budget"
            elif h >= 1:
                a["title"], a["level"] = f"{prog.title()} Coordinator", "manager"
                a["does"] = f"run the {prog} rota and the volunteers on it"
            else:
                a["title"] = rng.choice(("Caseworker", "Programme Officer", "Outreach Worker",
                                         "Administrator"))
                a["level"], a["does"] = "staff", f"do the day-to-day work of the {prog} programme"
            staff.append(who)
        volunteers = []
        for _ in range(count - paid):
            title, _level, does = _day_job(b, "")
            who = join(title=f"Volunteer ({title})", level="volunteer", unit=b.label(unit),
                       unit_id=unit, does=f"volunteer in {prog}; by day, {does}")
            volunteers.append(who)
            juniors.append(who)
        for who in people + volunteers:
            b.edge(who, "part_of", unit)
            _skills(b, who, pool, elsewhere)
        teams: dict[str, list[str]] = {}
        for child, parent in lines.items():
            teams.setdefault(parent, []).append(child)
        _teams(b, [*teams.values(), volunteers[:12]])
    for child, parent in parents.items():
        b.edge(child, "reports_to", parent)
        b.attrs(child)["manager"] = b.label(parent)
    _mentors(b, staff, juniors, 0.06)
    everywhere = [b.place(c, k) for c, k in CITIES]
    for _ in range(max(2, n // 50)):
        f = b.org(company_name(rng, kind=rng.choice(("Foundation", "Trust", "Council"))),
                  type="funder")
        b.edge(f, "based_in", rng.choice(everywhere))
        b.edge(f, "funds", org)
    for _ in range(max(2, n // 40)):
        p = b.org(company_name(rng), type="partner", industry=rng.choice(INDUSTRIES)[0])
        b.edge(p, "partner_of", org)
        b.edge(p, "based_in", rng.choice(everywhere))
    _projects(b, b.people, list(units.values()), max(3, n // 12), prefix="campaign",
              naming=("Campaign:", "Appeal:", "Drive:", "Pilot:"), skills_of=skills_of,
              size=(3, 7))
    _events(b, b.people, ("Annual Gala", "Volunteer Induction", "Spring Fundraiser",
                          "Trustees' Away Day", "Winter Appeal Launch", "Open Evening",
                          "Community Fair", "Sponsored Walk"),
            max(3, min(20, n // 25)), (6, 12) if n < 200 else (10, 40))
    return {"organisation": org, "root": director}


BUILDERS = {"company": _company, "community": _community, "university": _university,
            "open-source": _open_source, "nonprofit": _nonprofit}


# --- what every kind gets ---------------------------------------------------------------------

def _quotes(b: _Build, org: str) -> None:
    """One to three things each person would say about their work, from role and projects,
    so `look_up`'s "said" voter has something to find."""
    skills: dict[str, list[str]] = {}
    for e in b.edges:
        if e["rel"] == "experienced_in":
            skills.setdefault(e["source"], []).append(b.label(e["target"]))
    rng = b.rng
    for who in b.people:
        a = b.attrs(who)
        unit, where = a.get("unit", ""), a.get("employer") or b.label(org)
        title, does = a.get("title", "here"), a.get("does", "keep busy")
        lines = [rng.choice((
            f"I'm {title} in {unit} at {where}. Mostly I {does}.",
            f"{title}, {unit}. What I actually do: I {does}.",
            f"My job is {title}. Day to day that means I {does}.",
        ))]
        projects = a.get("projects") or []
        if projects and rng.random() < 0.8:
            lines.append(f"Lately most of my time goes to {projects[0]}."
                         if len(projects) == 1 or rng.random() < 0.5 else
                         f"Ask me about {projects[0]} -- I'm on it, along with {projects[-1]}.")
        mine = skills.get(who, [])
        if mine and rng.random() < 0.6:
            lines.append(f"Happy to help with {' or '.join(mine[:2])}.")
        b.said[who] = lines[: rng.randint(1, 3)]


def _personas(b: _Build, org: str, kind: str, public: Sequence[str]) -> dict[str, dict[str, Any]]:
    """A voice, a system prompt and what each person would know: the graph two hops out,
    stepping through people and pieces of work but not through hubs, plus every public node."""
    rng = b.rng
    near: dict[str, list[str]] = {}
    for e in b.edges:
        near.setdefault(e["source"], []).append(e["target"])
        near.setdefault(e["target"], []).append(e["source"])
    walkable = {"person", "opportunity"}
    out = {}
    for who in b.people:
        a = b.attrs(who)
        voice = rng.choice(VOICES)
        first = near.get(who, [])
        knows = {who, *public, *first}
        for step in first:
            if b.nodes[step]["kind"] in walkable:
                knows.update(near.get(step, ()))
        bits = [f"You are {b.label(who)}, {a.get('title', 'a member')}"
                + (f" in {a['unit']}" if a.get("unit") else "")
                + f" at {a.get('employer') or b.label(org)}"
                + (f", which is a {kind.replace('-', ' ')}" if not a.get("employer") else
                   f", and a member of {b.label(org)}") + "."]
        if a.get("does"):
            bits.append(f"You {a['does']}.")
        if a.get("manager"):
            bits.append(f"You report to {a['manager']}.")
        if a.get("projects"):
            bits.append("You are working on " + ", ".join(a["projects"][:3]) + ".")
        skills = [b.label(t) for t in first if b.nodes[t]["kind"] == "topic"]
        if skills:
            bits.append("You know about " + ", ".join(skills) + ".")
        place = next((b.label(t) for t in first if b.nodes[t]["kind"] == "place"), "")
        bits.append(f"You are based in {place}." if place else "You work remotely.")
        bits.append(f"How you write: {voice}")
        bits.append("Speak as this person, in the first person, from what you know and "
                    "nothing else; when you do not know, say who would.")
        out[who] = {"voice": voice, "system": " ".join(bits), "knows": sorted(knows)}
    return out


def make(kind: str = "company", size: str = "small", seed: int = 0) -> World:
    """An organised group of that kind and size, invented from the seed: the same every time.

    ``kind`` is one of `KINDS`; ``size`` one of `SIZES` (50, 500 or 5000 people).
    """
    if kind not in BUILDERS:
        raise ValueError(f"kind must be one of {', '.join(KINDS)}, not {kind!r}")
    if size not in SIZES:
        raise ValueError(f"size must be one of {', '.join(SIZES)}, not {size!r}")
    n = SIZES[size]
    b = _Build(random.Random(f"{kind}/{size}/{seed}"))
    offices = _offices(b, {"small": 3, "medium": 6}.get(size, 12))
    made = BUILDERS[kind](b, n, offices)
    org = made["organisation"]
    for who in b.people:
        b.attrs(who).pop("bias", None)
        b.attrs(who).pop("unit_id", None)
    _quotes(b, org)

    # what a reader of this graph sees: quotes as messages, mentions as degree
    messages: dict[str, dict[str, Any]] = {}
    n_msg = 0
    for who, lines in b.said.items():
        for line in lines:
            mid = f"m{n_msg}"
            messages[mid] = {"text": line, "ts": str(1_756_600_000 + n_msg * 900),
                             "channel": "#introductions", "sender": b.label(who)}
            b.nodes[who]["messages"].append(mid)
            n_msg += 1
    degree: dict[str, int] = {}
    for e in b.edges:
        degree[e["source"]] = degree.get(e["source"], 0) + 1
        degree[e["target"]] = degree.get(e["target"], 0) + 1
    for node_id, node in b.nodes.items():
        node["mentions"] = max(1, degree.get(node_id, 0))

    unit_kind = UNIT_KIND[kind]
    public = [i for i, node in b.nodes.items()
              if node["kind"] in ("place", unit_kind, "product", "body") or i == org]
    personas = _personas(b, org, kind, public)
    graph = {"nodes": list(b.nodes.values()), "edges": b.edges, "messages": messages,
             "stats": {"messages": len(messages), "people": len(b.people)},
             "meta": {"community": "invented",
                      "world": {"kind": kind, "size": size, "seed": seed, "organisation": org,
                                "root": made.get("root", ""), "unit_kind": unit_kind}}}
    return World(graph=graph, people=list(b.people), personas=personas, calendar=[],
                 seed=seed, size=size, kind=kind)


def summary(world: World) -> dict[str, Any]:
    """People, units and edges by relation: what `make` made, in one mapping."""
    meta = (world.graph.get("meta") or {}).get("world") or {}
    by_rel: dict[str, int] = {}
    for e in world.graph["edges"]:
        by_rel[e["rel"]] = by_rel.get(e["rel"], 0) + 1
    by_kind: dict[str, int] = {}
    for n in world.graph["nodes"]:
        by_kind[n["kind"]] = by_kind.get(n["kind"], 0) + 1
    return {"kind": meta.get("kind", ""), "size": world.size, "seed": world.seed,
            "organisation": meta.get("organisation", ""), "people": len(world.people),
            "units": by_kind.get(meta.get("unit_kind", ""), 0), "nodes": len(world.graph["nodes"]),
            "edges": len(world.graph["edges"]), "nodes_by_kind": dict(sorted(by_kind.items())),
            "edges_by_relation": dict(sorted(by_rel.items()))}


def role_catalogue() -> dict[str, list[str]]:
    """Every title the company builder can hand out, by department: what the org chart may say."""
    out = {}
    for dept, held in DEPARTMENTS.items():
        titles = [f"{word}{track}" for track, _ in held["tracks"] for _, word in IC_LEVELS]
        titles += [f"{dept.title()} Manager", f"Director of {dept.title()}",
                   f"VP {dept.title()}", f"Head of {dept.title()}"]
        out[dept] = titles
    out["executive"] = ["Chief Executive Officer", *(t for t, _, _ in C_LEVELS)]
    return out


def load(where: str | Path) -> World:
    """The world `ml-stack-world make --out DIR` wrote, read back.

    ``world.json`` (kind, size, seed, people) is read when it is there, the way
    `world.simulate.run` reads it; without it the graph's own ``meta`` says the same.
    """
    where = Path(where).expanduser()
    graph = json.loads((where / "graph.json").read_text(encoding="utf-8"))
    personas = json.loads((where / "personas.json").read_text(encoding="utf-8"))
    calendar_file = where / "calendar.json"
    calendar = json.loads(calendar_file.read_text(encoding="utf-8")) if calendar_file.exists() else []
    meta = (graph.get("meta") or {}).get("world") or {}
    about_file = where / "world.json"
    about = json.loads(about_file.read_text(encoding="utf-8")) if about_file.exists() else {}
    people = [str(p) for p in about.get("people") or ()] or \
        [str(n["id"]) for n in graph.get("nodes") or () if n.get("kind") == "person"]
    return World(graph=graph, people=people, personas=personas, calendar=calendar,
                 seed=int(about.get("seed", meta.get("seed", 0))),
                 size=str(about.get("size") or meta.get("size") or "small"),
                 kind=str(about.get("kind") or meta.get("kind") or "company"))
