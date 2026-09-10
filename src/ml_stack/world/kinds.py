"""How each kind of organised group is put together, one builder each.

- ``company``: departments under a CEO, reporting lines, customers, partners, products.
- ``community``: members with day jobs at different organisations, groups and moderators.
- ``university``: departments of labs led by principal investigators; grants and seminars.
- ``open-source``: one project of many repositories, maintainers, releases and sponsors.
- ``nonprofit``: programmes under an executive director, a board, volunteers and funders.

Each returns the organisation node and the person at the top. `BUILDERS` maps the name
to the builder. (no-real-names: shapes off)
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from ml_stack.world.building import (
    _Build,
    _cities,
    _day_job,
    _employers,
    _events,
    _funders,
    _heights,
    _hiring,
    _ic_level,
    _mentors,
    _partners,
    _projects,
    _reports,
    _skills,
    _split,
    _started,
    _teams,
    _tree,
)
from ml_stack.world.catalogue import (
    ACADEMIC,
    C_LEVELS,
    DEPARTMENTS,
    INDUSTRIES,
    INTEREST_GROUPS,
    PROGRAMMES,
    REPOS,
    SIZES,
)
from ml_stack.world.names import company_name, product_name, slug

__all__ = ["BUILDERS"]


# --- company -------------------------------------------------------------------------------

def _department(b: _Build, dept: str, unit: str, count: int, hire: Callable[..., str],
                boss: str, elsewhere: Sequence[str]) -> tuple[dict[str, str], list[str], list[str]]:
    """One department's people, their titles, its teams and its reporting lines.

    Returns ``child -> parent`` for everyone in it, its head included, and the senior and
    junior halves of it.
    """
    rng = b.rng
    tracks = DEPARTMENTS[dept]["tracks"]
    pool = DEPARTMENTS[dept]["skills"]
    members = [hire(unit=b.label(unit), unit_id=unit) for _ in range(count)]
    head, rest = members[0], members[1:]
    lines = _tree(rng, rest, head)
    heights = _heights(lines)
    seniors: list[str] = []
    juniors: list[str] = []
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
    _teams(b, _reports(lines).values())
    return {**lines, head: boss}, seniors, juniors


def _customers(b: _Build, org: str, n: int, cities: Sequence[str]) -> None:
    """``n`` invented customers of ``org``, each served by somebody in sales or support."""
    rng = b.rng
    for _ in range(n):
        c = b.org(company_name(rng), industry=rng.choice(INDUSTRIES)[0], type="customer")
        b.edge(c, "customer_of", org)
        b.edge(c, "based_in", rng.choice(list(cities)))
        for who in rng.sample([p for p in b.people if b.attrs(p).get("unit") in ("Sales", "Support")]
                              or b.people[:1], 1):
            b.edge(who, "serves", c)


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

    hire = _hiring(b, org, founded)
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
        elsewhere = [s for d in names if d != dept for s in DEPARTMENTS[d]["skills"]]
        lines, senior, junior = _department(b, dept, units[dept], count, hire,
                                            heads.get(dept, ceo), elsewhere)
        parents.update(lines)
        seniors += senior
        juniors += junior
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
    everywhere = _cities(b)
    _customers(b, org, max(3, n // 40), everywhere)
    _partners(b, org, max(2, n // 150), everywhere)

    skills_of = {units[d]: DEPARTMENTS[d]["skills"] for d in names}
    _projects(b, max(4, n // 12),
              naming=("Project", "Initiative", "Migration", "Launch", "Rewrite", "Pilot"),
              skills_of=skills_of)
    _events(b, b.people, ("All Hands", "Engineering Offsite", "Sales Kickoff", "Hack Week",
                          "Customer Summit", "Product Launch", "Winter Party", "Trade Fair"),
            max(3, min(20, n // 25)))
    return {"organisation": org, "root": ceo}


def _size_of(n: int) -> str:
    return next((s for s, count in SIZES.items() if count == n), "custom")


# --- community -------------------------------------------------------------------------------

def _community(b: _Build, n: int, offices: list[str]) -> dict[str, Any]:
    rng = b.rng
    name = f"{company_name(rng, kind='')} Network"
    org = b.org(name, founded=rng.randint(2015, 2022), type="community",
                headquarters="Slack")
    cities = _cities(b)
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
    _projects(b, max(3, n // 20),
              naming=("Community", "Working group:", "Meetup series:", "Reading group:"),
              skills_of=skills_of, size=(3, 6))
    _events(b, b.people, ("Makers Night", "Spring Meetup", "Demo Day", "Careers Evening",
                          "Founders Breakfast", "Winter Social", "Lightning Talks",
                          "Hardware Hack Day"),
            max(3, min(20, n // 20)))
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

    enrol = _hiring(b, org, max(founded, 1990), offices, 4)
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
    _funders(b, max(2, n // 60), _cities(b),
             lambda: rng.sample(list(labs), min(len(labs), rng.randint(1, 3))))
    _projects(b, max(3, n // 10),
              naming=("Grant:", "Study:", "Consortium:", "Fellowship:"), skills_of=skills_of,
              size=(3, 6))
    _events(b, b.people, ("Departmental Seminar", "Graduate Symposium", "Open Day",
                          "Summer School", "Thesis Defences", "Faculty Retreat",
                          "Annual Conference", "Poster Session"),
            max(3, min(20, n // 25)))
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
    cities = _cities(b)

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
    _projects(b, max(4, n // 10),
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

def _programme(b: _Build, prog: str, unit: str, count: int, join: Callable[..., str],
               director: str, skills_of: Mapping[str, Sequence[str]],
               ) -> tuple[dict[str, str], list[str], list[str]]:
    """One programme's paid staff and its volunteers, with titles and reporting lines.

    ``count`` people, a little over half of them paid. Returns ``child -> parent`` for the
    paid staff, the staff, and the volunteers.
    """
    rng = b.rng
    pool = skills_of[unit]
    elsewhere = [s for u, ss in skills_of.items() if u != unit for s in ss]
    paid = max(1, int(count * 0.55))
    people = [join(unit=b.label(unit), unit_id=unit) for _ in range(paid)]
    head, rest = people[0], people[1:]
    lines = _tree(rng, rest, head)
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
    volunteers = []
    for _ in range(count - paid):
        title, _level, does = _day_job(b, "")
        volunteers.append(join(title=f"Volunteer ({title})", level="volunteer",
                               unit=b.label(unit), unit_id=unit,
                               does=f"volunteer in {prog}; by day, {does}"))
    for who in people + volunteers:
        b.edge(who, "part_of", unit)
        _skills(b, who, pool, elsewhere)
    _teams(b, [*_reports(lines).values(), volunteers[:12]])
    return {**lines, head: director}, people, volunteers


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

    join = _hiring(b, org, founded, offices, 3)
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
    counts = _split(rng, n - 1 - len(trustees), [1] * len(units), 3)
    staff: list[str] = []
    juniors: list[str] = []
    parents: dict[str, str] = {}
    for (prog, _d, _s), count in zip(chosen, counts):
        lines, paid, volunteers = _programme(b, prog, units[prog], count, join, director,
                                             skills_of)
        parents.update(lines)
        staff += paid
        juniors += volunteers
    for child, parent in parents.items():
        b.edge(child, "reports_to", parent)
        b.attrs(child)["manager"] = b.label(parent)
    _mentors(b, staff, juniors, 0.06)
    everywhere = _cities(b)
    _funders(b, max(2, n // 50), everywhere, lambda: [org])
    _partners(b, org, max(2, n // 40), everywhere)
    _projects(b, max(3, n // 12),
              naming=("Campaign:", "Appeal:", "Drive:", "Pilot:"), skills_of=skills_of,
              size=(3, 7))
    _events(b, b.people, ("Annual Gala", "Volunteer Induction", "Spring Fundraiser",
                          "Trustees' Away Day", "Winter Appeal Launch", "Open Evening",
                          "Community Fair", "Sponsored Walk"),
            max(3, min(20, n // 25)))
    return {"organisation": org, "root": director}


BUILDERS = {"company": _company, "community": _community, "university": _university,
            "open-source": _open_source, "nonprofit": _nonprofit}
