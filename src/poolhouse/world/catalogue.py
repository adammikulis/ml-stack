"""What each kind of organised group is made of.

Sizes and kinds, the cities people live in, and the vocabulary each kind draws on: a
company's departments and their tracks, a university's fields, an open-source project's
repositories, a nonprofit's programmes, a community's interest groups, the levels on a
business card and the voices people write in.

Nothing here is a real person or organisation; the only real things are cities.

This is a catalogue of invented labels, so the name hook's shape rule is off
(no-real-names: shapes off); the exact list and the recogniser still run.
"""

from __future__ import annotations

import datetime as _dt
from typing import Any

__all__ = ["ACADEMIC", "CITIES", "C_LEVELS", "DEPARTMENTS", "IC_LEVELS", "IC_WEIGHTS",
           "INDUSTRIES", "INTEREST_GROUPS", "KINDS", "PROGRAMMES", "REPOS", "SIZES", "TODAY",
           "UNIT_KIND", "VOICES"]

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
