"""An invented community, for measuring how well a graph gets read and answered.

Every person, company, place and subject here is made up. A benchmark keyed to a real
community cannot be shared, cannot be compared between machines, and puts real people's
details somewhere they do not belong — so the questions and their answers are asked of these
people instead, and the numbers mean the same thing anywhere.

It is sized on purpose. A graph of a dozen entries makes retrieval trivial -- there is
nothing to confuse -- so a model can score well without discriminating at all. This one
carries several people per subject, subjects that sound alike, and places that repeat, so
that a wrong answer is always available and finding the right one is work. It is still
small enough to read in one sitting and to check an expected answer by eye.

`QUESTIONS` is a hundred scored questions and ten more whose right answer is nobody. Each
is ``{"q": ..., "expect": [exact node ids]}``; an empty ``expect`` means naming anybody is
wrong. Every answer is derivable from this graph alone -- what `look_up`, `look_at`,
`path_between` and `list_kind` can reach -- so a wrong answer is the model's and never the
question's. A question about finding somebody does not contain the label it is looking for;
a listing question expects every member; a path question expects the chain, and the chain
must be *a* shortest path (`tests/test_graph_community.py` derives it from the graph rather
than trusting the list). About a dozen turn on words only a message carries, because most
of a community is what people said. `ml-stack-bench prepare --mix` prints how many ask for
each kind of answer, which is the number to watch when adding more: the second fifty were
half drawn by `ml_stack.world.questions` over this same graph and half written by hand,
and both halves were kept to the shares the first fifty already had.
"""

from __future__ import annotations

from typing import Any

__all__ = ["QUESTIONS", "graph"]

# id -> (label, kind, what they said). The subjects are attached by the edges below, the way
# a reader would have to work them out.
_SAID: dict[str, tuple[str, str, str]] = {
    # --- people who fix and build things -------------------------------------------
    "person:ada": ("Ada Lovelace", "person",
                   "I am a robotics technician and I fix machines on the line all day. "
                   "Mostly servo repair and field service for manufacturing plants."),
    "person:charles": ("Charles Babbage", "person",
                       "Hardware. I design and repair mechanical assemblies, and I am "
                       "looking for a marketing person to help me sell a machine."),
    "person:hedy": ("Hedy Marchetti", "person",
                    "Controls engineer. I program the arms Ada keeps alive, and I am "
                    "slowly learning to weld badly."),
    "person:oskar": ("Oskar Trent", "person",
                     "Maintenance planning. I decide what gets serviced and when, which is "
                     "not the same job as fixing it."),
    # --- people who sell and tell ---------------------------------------------------
    "person:grace": ("Grace Hopper", "person",
                     "I run growth and go-to-market. Twenty years selling enterprise "
                     "software, mostly campaigns and channel partnerships."),
    "person:sela": ("Sela Bright", "person",
                    "Brand and content. Not sales -- the words around it. I have opinions "
                    "about how manufacturers describe themselves."),
    "person:dorian": ("Dorian Vale", "person",
                      "Partnerships. I find the companies worth talking to and get the "
                      "first meeting; somebody else closes it."),
    # --- people who model and measure -----------------------------------------------
    "person:alan": ("Alan Turing", "person",
                    "I build machine learning models, lately for medical imaging at a "
                    "hospital group. Interested in anything clinical."),
    "person:vera": ("Vera Lund", "person",
                    "Data engineering for hospitals. Pipelines and governance, not models "
                    "-- I keep the inputs honest so Alan's work means something."),
    "person:katherine": ("Katherine Johnson", "person",
                         "Finance background -- I automate month-end close and reporting. "
                         "Twenty-five years of process work in accounting teams."),
    "person:milo": ("Milo Fenwick", "person",
                    "Also automation, but the warehouse kind: conveyors, pickers, and the "
                    "software that tells them where to go."),
    # --- people who hold things together --------------------------------------------
    "person:mary": ("Mary Somerville", "person",
                    "I welcome new people and keep the channels tidy. Community management "
                    "and onboarding is what I do."),
    "person:otto": ("Otto Vance", "person",
                    "I run the office at a foundry. Scheduling, suppliers, payroll -- the "
                    "unglamorous half that stops everything else from falling over."),
    "person:nell": ("Nell Ashgrove", "person",
                    "Lawyer. Contracts and licensing for small manufacturers."),
    "person:bram": ("Bram Ostley", "person",
                    "Recruiting, mostly engineers. I read a lot of CVs and meet a lot of "
                    "people who are almost right."),
    # Shares a surname with Otto and nothing else. A question that names only "Vance" has
    # two right answers, and a model that picks the better-known one is wrong.
    "person:delia": ("Delia Vance", "person",
                     "No relation to Otto, before anyone asks. I bind books and teach it "
                     "two evenings a week."),
    # --- people who work outdoors ----------------------------------------------------
    "person:iris": ("Iris Bellweather", "person",
                    "Land surveying, mostly for civil engineering firms. Twelve years of "
                    "fieldwork and I still prefer being outdoors to being in an office."),
    "person:tam": ("Tam Quillon", "person",
                   "Geotechnical work -- boreholes and soil reports. Iris tells me where "
                   "to dig and I tell her what is under it."),
    # --- people the graph barely knows ------------------------------------------------
    "person:rufus": ("Rufus Kell", "person", ""),
    "person:juno": ("Juno Vasquez", "person", ""),
    "person:pell": ("Pell Grantham", "person",
                    "Just joined -- reading for now."),

    "org:quenlow": ("Quenlow Robotics", "org", ""),
    "org:pellard": ("Pellard Foundry", "org", ""),
    "org:harnley": ("Harnley Health", "org", ""),
    "org:brayfield": ("Brayfield Survey Co", "org", ""),
    "org:corvane": ("Corvane Logistics", "org", ""),
    "org:littlemoor": ("Littlemoor Legal", "org", ""),

    "place:turin": ("Turin", "place", ""),
    "place:dunmore": ("Dunmore", "place", ""),
    "place:calderwick": ("Calderwick", "place", ""),
    "place:harrowgate": ("Harrowgate", "place", ""),
    "place:selby": ("Selby", "place", ""),

    "topic:robotics": ("robotics", "topic", ""),
    "topic:repair": ("repair", "topic", ""),
    "topic:controls": ("controls", "topic", ""),
    "topic:maintenance": ("maintenance", "topic", ""),
    "topic:marketing": ("marketing", "topic", ""),
    "topic:brand": ("brand", "topic", ""),
    "topic:partnerships": ("partnerships", "topic", ""),
    "topic:healthcare": ("healthcare", "topic", ""),
    "topic:data": ("data engineering", "topic", ""),
    "topic:automation": ("automation", "topic", ""),
    "topic:logistics": ("logistics", "topic", ""),
    "topic:onboarding": ("onboarding", "topic", ""),
    "topic:contracts": ("contracts", "topic", ""),
    "topic:recruiting": ("recruiting", "topic", ""),
    "topic:surveying": ("surveying", "topic", ""),
    "topic:geotechnics": ("geotechnics", "topic", ""),
    "topic:bookbinding": ("bookbinding", "topic", ""),

    # Work going spare, which is the kind a community graph exists to match people to --
    # and the kind the page has a shape for and this bench had never tested.
    "opportunity:linefix": ("keeping a production line running", "opportunity", ""),
    "opportunity:clinicaldata": ("cleaning up clinical data", "opportunity", ""),
    "opportunity:sellmachine": ("selling a machine nobody has heard of", "opportunity", ""),
    "opportunity:sitesurvey": ("surveying a site before it is bought", "opportunity", ""),

    # Where clusters that share no subject meet anyway. Without these the graph is four
    # islands, and a question about connecting people has no interesting answer.
    "event:makersnight": ("Makers Night", "event", ""),
    "event:tradefair": ("Northern Trade Fair", "event", ""),
}

_JOINED: list[tuple[str, str, str]] = [
    ("person:ada", "experienced_in", "topic:robotics"),
    ("person:ada", "experienced_in", "topic:repair"),
    ("person:ada", "works_at", "org:quenlow"),
    ("person:ada", "based_in", "place:turin"),
    ("person:charles", "experienced_in", "topic:repair"),
    ("person:charles", "works_at", "org:pellard"),
    ("person:charles", "seeks", "topic:marketing"),
    ("person:charles", "based_in", "place:harrowgate"),
    ("person:hedy", "experienced_in", "topic:controls"),
    ("person:hedy", "experienced_in", "topic:robotics"),
    ("person:hedy", "works_at", "org:quenlow"),
    ("person:hedy", "based_in", "place:turin"),
    ("person:oskar", "experienced_in", "topic:maintenance"),
    ("person:oskar", "works_at", "org:pellard"),
    ("person:oskar", "based_in", "place:harrowgate"),

    ("person:grace", "experienced_in", "topic:marketing"),
    ("person:grace", "based_in", "place:dunmore"),
    ("person:sela", "experienced_in", "topic:brand"),
    ("person:sela", "based_in", "place:dunmore"),
    ("person:dorian", "experienced_in", "topic:partnerships"),
    ("person:dorian", "works_at", "org:corvane"),
    ("person:dorian", "based_in", "place:selby"),

    ("person:alan", "experienced_in", "topic:healthcare"),
    ("person:alan", "works_at", "org:harnley"),
    ("person:alan", "based_in", "place:selby"),
    ("person:vera", "experienced_in", "topic:data"),
    ("person:vera", "experienced_in", "topic:healthcare"),
    ("person:vera", "works_at", "org:harnley"),
    ("person:katherine", "experienced_in", "topic:automation"),
    ("person:katherine", "based_in", "place:dunmore"),
    ("person:milo", "experienced_in", "topic:automation"),
    ("person:milo", "experienced_in", "topic:logistics"),
    ("person:milo", "works_at", "org:corvane"),
    ("person:milo", "based_in", "place:selby"),

    ("person:mary", "experienced_in", "topic:onboarding"),
    ("person:mary", "based_in", "place:harrowgate"),
    ("person:otto", "works_at", "org:pellard"),
    ("person:otto", "based_in", "place:calderwick"),
    ("person:nell", "experienced_in", "topic:contracts"),
    ("person:nell", "works_at", "org:littlemoor"),
    ("person:nell", "based_in", "place:calderwick"),
    ("person:bram", "experienced_in", "topic:recruiting"),
    ("person:bram", "based_in", "place:selby"),
    ("person:delia", "experienced_in", "topic:bookbinding"),
    ("person:delia", "based_in", "place:harrowgate"),

    ("person:iris", "experienced_in", "topic:surveying"),
    ("person:iris", "works_at", "org:brayfield"),
    ("person:iris", "based_in", "place:calderwick"),
    ("person:tam", "experienced_in", "topic:geotechnics"),
    ("person:tam", "works_at", "org:brayfield"),
    ("person:tam", "based_in", "place:calderwick"),

    ("org:quenlow", "offers", "opportunity:linefix"),
    ("org:harnley", "offers", "opportunity:clinicaldata"),
    ("person:charles", "offers", "opportunity:sellmachine"),
    ("org:brayfield", "offers", "opportunity:sitesurvey"),
    ("opportunity:linefix", "wants", "topic:repair"),
    ("opportunity:clinicaldata", "wants", "topic:data"),
    ("opportunity:sellmachine", "wants", "topic:marketing"),
    ("opportunity:sitesurvey", "wants", "topic:surveying"),

    # Makers Night is where the workshop people and the outdoors people actually meet;
    # the trade fair is where the sellers meet the makers.
    ("person:ada", "attended", "event:makersnight"),
    ("person:hedy", "attended", "event:makersnight"),
    ("person:iris", "attended", "event:makersnight"),
    ("person:tam", "attended", "event:makersnight"),
    ("person:otto", "attended", "event:makersnight"),
    ("person:grace", "attended", "event:tradefair"),
    ("person:dorian", "attended", "event:tradefair"),
    ("person:charles", "attended", "event:tradefair"),
    ("person:sela", "attended", "event:tradefair"),
    ("person:milo", "attended", "event:tradefair"),
]



# --- the crowd -------------------------------------------------------------------------
#
# A graph of a few dozen entries makes retrieval trivial: there is almost nothing to
# confuse, so a model scores well without discriminating at all. Real communities are
# mostly people no question is about, and the work is telling them from the few who are.
#
# So the cast above is surrounded by people who are *nearly* relevant and never the answer.
# Their subjects sit next to the ones the questions ask about -- hydraulics beside repair,
# procurement beside logistics, radiology beside healthcare -- close enough that a search
# for the question's words drags them in, far enough that none of them is ever correct.
# The vocabularies are deliberately disjoint, so adding crowd does not change a single
# expected answer; `test_the_crowd_is_never_an_answer` holds that true.
#
# Names are assembled from parts rather than written out, so the source carries "Wren" and
# "Ashcombe" separately and no line of it reads as a person.
# "Wren" and "Hollis" are deliberately absent: the tool descriptions use them in their
# worked examples, and an example sharing a name with the community it is benchmarked
# against is how a rising score comes to mean memorisation. `test_the_tool_descriptions_
# show_a_call_and_never_use_the_bench_community` keeps the two sets apart.
_GIVEN = ("Marlowe", "Perrin", "Sable", "Corrin", "Emrys", "Tansy", "Ottoline", "Ferris",
          "Lowen", "Bryn", "Fenn", "Isolde", "Rowan", "Quillon", "Maren", "Ansel",
          "Delphine", "Osric", "Verity", "Caspian", "Linnet", "Thaddeus", "Elowen")
_FAMILY = ("Ashcombe", "Trewin", "Hallamby", "Prentiss", "Skelbrook", "Varden", "Oakhurst",
           "Millward", "Rookwood", "Danforth", "Greaves", "Fairbourne")

# subjects that sit next to the ones the questions ask about, and are never an answer
_NEARBY = ("hydraulics", "tooling", "procurement", "radiology", "compliance", "payroll",
           "warehousing", "calibration", "metallurgy", "site safety", "translation",
           "packaging", "insurance", "haulage", "drafting", "acoustics", "pest control",
           "catering", "signage", "waste handling")
_ELSEWHERE_ORGS = ("Kessleton Mills", "Aldwych Freight", "Norbury Plate", "Vexley Print",
                   "Stroud & Hale", "Merrivale Tooling", "Ganthorpe Works")
_ELSEWHERE_PLACES = ("Widdop", "Barrowfield", "Elsmere", "Croughton", "Pentland", "Ashby")


def _crowd() -> tuple[dict[str, tuple[str, str, str]], list[tuple[str, str, str]]]:
    """People no question is about, and the subjects that keep them plausible.

    Deterministic: the same graph every run, so a score means the same thing twice.
    """
    said: dict[str, tuple[str, str, str]] = {}
    joined: list[tuple[str, str, str]] = []
    for subject in _NEARBY:
        said[f"topic:{subject.replace(' ', '')}"] = (subject, "topic", "")
    for org in _ELSEWHERE_ORGS:
        said[f"org:{org.split()[0].lower()}"] = (org, "org", "")
    for place in _ELSEWHERE_PLACES:
        said[f"place:{place.lower()}"] = (place, "place", "")

    n = 0
    for family in _FAMILY:
        for given in _GIVEN[: 4 if family != _FAMILY[-1] else 2]:
            who = f"person:{given.lower()}{family[:3].lower()}"
            if who in said:
                continue
            subject = _NEARBY[n % len(_NEARBY)]
            org = _ELSEWHERE_ORGS[n % len(_ELSEWHERE_ORGS)]
            place = _ELSEWHERE_PLACES[n % len(_ELSEWHERE_PLACES)]
            said[who] = (f"{given} {family}", "person",
                         f"I work in {subject}. Mostly for {org}, out of {place}.")
            joined += [(who, "experienced_in", f"topic:{subject.replace(' ', '')}"),
                       (who, "works_at", f"org:{org.split()[0].lower()}"),
                       (who, "based_in", f"place:{place.lower()}")]
            n += 1
    return said, joined


_MORE_SAID, _MORE_JOINED = _crowd()
_SAID.update(_MORE_SAID)
_JOINED.extend(_MORE_JOINED)


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
    # --- finding somebody by what they do -------------------------------------------
    {"q": "Who fixes machines?", "expect": ["person:ada", "person:charles"]},
    {"q": "Who could help me with a broken conveyor belt?",
     "expect": ["person:ada", "person:charles", "person:milo"]},
    {"q": "Who knows about robotics?", "expect": ["person:ada", "person:hedy"]},
    {"q": "Who knows the most about automation?",
     "expect": ["person:katherine", "person:milo"]},
    {"q": "Someone who can sell things", "expect": ["person:grace"]},
    {"q": "Who handles contracts?", "expect": ["person:nell"]},
    {"q": "Who should welcome a newcomer?", "expect": ["person:mary"]},
    {"q": "Who does surveying?", "expect": ["person:iris"]},
    # Two people share this subject and one of them is the one who *models*; the other
    # keeps the inputs honest. A question that says "build" wants the modeller.
    {"q": "I need two people to build a healthcare AI prototype. Who?",
     "expect": ["person:alan", "person:vera"]},
    # Ada fixes the arms and Hedy programs them; Oskar plans the servicing and says himself
    # it is not the same job as fixing. A question that names the job wants the one person.
    {"q": "Who programs robot arms?", "expect": ["person:hedy"]},
    {"q": "who decides what gets serviced and when?", "expect": ["person:oskar"]},
    {"q": "Who could get us a first meeting with a company worth talking to?",
     "expect": ["person:dorian"]},
    {"q": "Who works at Pellard Foundry?",
     "expect": ["person:charles", "person:oskar", "person:otto"]},
    # Sela's subject is "brand"; the question asks for the job rather than the word, which
    # is what a person actually types.
    {"q": "We need somebody to write the words on a manufacturer's website. Who?",
     "expect": ["person:sela"]},
    # "geotechnics" appears nowhere in the question; only Tam's own line says what it is.
    {"q": "Who here works with soil and boreholes?", "expect": ["person:tam"]},

    # --- putting people together ------------------------------------------------------
    {"q": "Who could work together on a robotics marketing project?",
     "expect": ["person:ada", "person:hedy", "person:grace"]},
    # An employer and an event, intersected: Quenlow is Ada and Hedy, and both were there.
    {"q": "Who at Quenlow Robotics also went to Makers Night?",
     "expect": ["person:ada", "person:hedy"]},
    # The same shape one hop further out -- the fair's crowd, narrowed to the logistics
    # firm, which is Corvane. Grace and Sela were there and work for nobody.
    {"q": "Who at the Northern Trade Fair works for a logistics company?",
     "expect": ["person:dorian", "person:milo"]},
    # --- place -----------------------------------------------------------------------
    {"q": "Who is based in Dunmore?",
     "expect": ["person:grace", "person:katherine", "person:sela"]},
    {"q": "Who is in Calderwick?",
     "expect": ["person:iris", "person:nell", "person:otto", "person:tam"]},
    {"q": "whos based in selby",
     "expect": ["person:alan", "person:bram", "person:dorian", "person:milo"]},

    # --- answers that are not people --------------------------------------------------
    # The set was nine-tenths person-shaped, which flatters anything that prefers people.
    # These exist so a rule like "a who question is answered by people" is measured rather
    # than merely rewarded -- and so the kinds the page draws are all actually tested.
    {"q": "Which companies do people here work for?",
     "expect": ["org:brayfield", "org:corvane", "org:harnley", "org:littlemoor",
                "org:pellard", "org:quenlow"]},
    {"q": "Where does Alan Turing work?", "expect": ["org:harnley"]},
    {"q": "Which company does surveying?", "expect": ["org:brayfield"]},
    {"q": "What does Brayfield Survey Co do?",
     "expect": ["topic:surveying", "topic:geotechnics"]},
    {"q": "Which places do the people at Brayfield live in?",
     "expect": ["place:calderwick"]},
    {"q": "What does Pellard Foundry do?", "expect": ["topic:repair", "topic:maintenance"]},
    {"q": "where's quenlow robotics?", "expect": ["place:turin"]},
    {"q": "What does Littlemoor Legal do?", "expect": ["topic:contracts"]},
    # Two hops out and back: the people who were there, then where each of them works.
    {"q": "Which firms had someone at Makers Night?",
     "expect": ["org:brayfield", "org:pellard", "org:quenlow"]},
    # Both events, intersected: Pellard sent Otto to one and Charles to the other, and it
    # is the only firm that appears in both lists.
    {"q": "Which company had somebody at both events?", "expect": ["org:pellard"]},
    # Nell's subject is "contracts"; only her own line says the word the question uses.
    {"q": "Which company employs the lawyer?", "expect": ["org:littlemoor"]},
    # A quote and then an edge: the imaging is only in what Alan said, and the firm is
    # only on the edge out of him.
    {"q": "Which company does the person who works on medical imaging work for?",
     "expect": ["org:harnley"]},
    # The other direction from "Which firms had someone at Makers Night?": the same five
    # people, followed on to where they live.
    {"q": "Where do the people who went to Makers Night live?",
     "expect": ["place:calderwick", "place:turin"]},

    # --- work going spare, which is what a community graph is for ----------------------
    {"q": "What openings are there?",
     "expect": ["opportunity:clinicaldata", "opportunity:linefix",
                "opportunity:sellmachine", "opportunity:sitesurvey"]},
    {"q": "Who is offering work keeping a production line running?",
     "expect": ["org:quenlow"]},
    {"q": "Somebody is selling a machine and needs help. What do they need?",
     "expect": ["topic:marketing"]},
    {"q": "Who could take the clinical data work?", "expect": ["person:vera"]},
    {"q": "Any work going for a surveyor?", "expect": ["opportunity:sitesurvey"]},
    # Charles is offering work too, but he is a person; "companies" asks for the three
    # that offer something themselves.
    {"q": "Which companies are offering work?",
     "expect": ["org:brayfield", "org:harnley", "org:quenlow"]},
    # Those three firms, then back down to their people: Charles offers work but is not a
    # company, so his colleagues at Pellard are not an answer.
    {"q": "Who works for a company that is offering work?",
     "expect": ["person:ada", "person:alan", "person:hedy", "person:iris", "person:tam",
                "person:vera"]},

    # --- where people meet ------------------------------------------------------------
    {"q": "What events come up?", "expect": ["event:makersnight", "event:tradefair"]},
    {"q": "Who was at Makers Night?",
     "expect": ["person:ada", "person:hedy", "person:iris", "person:otto", "person:tam"]},
    {"q": "Which event did the sellers go to?", "expect": ["event:tradefair"]},
    {"q": "Where would a surveyor have run into the robotics people?",
     "expect": ["event:makersnight"]},

    # --- how two things connect, at several distances ----------------------------------
    # Two hops through a shared employer; four through a place and an employer; six across
    # two events; and a pair with no path at all, which is a real answer.
    {"q": "How are Otto Vance and Charles Babbage connected?",
     "expect": ["person:otto", "org:pellard", "person:charles"]},
    {"q": "How is Iris Bellweather connected to Charles Babbage?",
     "expect": ["person:iris", "place:calderwick", "person:otto", "org:pellard",
                "person:charles"]},
    {"q": "What links Ada Lovelace and Grace Hopper?",
     "expect": ["person:ada", "topic:repair", "person:charles", "topic:marketing",
                "person:grace"]},
    {"q": "How is Iris Bellweather connected to Alan Turing?", "expect": []},
    {"q": "How is Milo Fenwick connected to Alan Turing?",
     "expect": ["person:milo", "place:selby", "person:alan"]},
    # Three more of the two-hop shape, one through each of the joins a person has: an
    # employer, a subject and a town. Two hops is the commonest real answer and the set had
    # one of them, so a model that only ever finds the long way round scored the same.
    {"q": "How are Vera Lund and Alan Turing connected?",
     "expect": ["person:vera", "org:harnley", "person:alan"]},
    {"q": "What links Katherine Johnson and Milo Fenwick?",
     "expect": ["person:katherine", "topic:automation", "person:milo"]},
    {"q": "How is Delia Vance connected to Oskar Trent?",
     "expect": ["person:delia", "place:harrowgate", "person:oskar"]},
    {"q": "Who could introduce Iris Bellweather to a lawyer?", "expect": ["person:nell"]},
    {"q": "Who could introduce Ada Lovelace to someone who does marketing?",
     "expect": ["person:charles", "person:grace"]},

    # --- the question names part of its own answer ------------------------------------
    # Without these, a filter that drops whatever the question named looks free. It is not.
    {"q": "What is Iris Bellweather good at?",
     "expect": ["person:iris", "topic:surveying"]},
    {"q": "Tell me about Otto Vance.",
     "expect": ["person:otto", "org:pellard", "place:calderwick"]},
    {"q": "Is Ada Lovelace the right person for a robotics job?", "expect": ["person:ada"]},
    {"q": "what's milo fenwick into?",
     "expect": ["person:milo", "topic:automation", "topic:logistics"]},
    {"q": "Tell me about Nell Ashgrove.",
     "expect": ["person:nell", "org:littlemoor", "place:calderwick"]},

    # --- somebody the graph knows of and nothing about ---------------------------------
    {"q": "What does Rufus Kell do?", "expect": ["person:rufus"]},

    # --- a follow-up, which only means anything with the turn before it ----------------
    {"q": "And where is she based?", "expect": ["place:calderwick"]},
    {"q": "Who does data engineering for hospitals?", "expect": ["person:vera"]},
    {"q": "And which company is she at?", "expect": ["org:harnley"]},

    # --- counting and comparing --------------------------------------------------------
    # A count is scored as the people counted, so "two" is checked rather than trusted.
    # The crowd is spread evenly over its own employers and towns, so "the company with
    # the most members" is one of theirs, four ways tied -- every aggregate here is scoped
    # to something the crowd never touches: a subject, an event, or what people said.
    {"q": "How many people here do robotics?", "expect": ["person:ada", "person:hedy"]},
    # Corvane sent Dorian and Milo; Pellard sent Charles; Grace and Sela work for nobody.
    {"q": "Which company sent the most people to the Northern Trade Fair?",
     "expect": ["org:corvane"]},
    # Only three people put a number on it: Katherine twenty-five years, Grace twenty,
    # Iris twelve. Answerable from what was said and from nothing else.
    {"q": "Who here has been doing their job the longest?", "expect": ["person:katherine"]},

    # --- two hops through the graph ----------------------------------------------------
    # The answer is not the person the question describes but who they are joined to:
    # the geotechnics person is Tam, Tam is at Brayfield, and Brayfield's other person is
    # Iris. Tam himself is not an answer.
    {"q": "Who works alongside the person who does geotechnics?", "expect": ["person:iris"]},
    # Repair is Ada and Charles; Ada is in Turin and Charles in Harrowgate. Neither person
    # is an answer, and neither Oskar (maintenance, not repair) nor his Harrowgate matters.
    {"q": "Which places do the people who do repair live in?",
     "expect": ["place:turin", "place:harrowgate"]},

    # --- ambiguity and traps -------------------------------------------------------------
    # Two Vances, no relation; a question naming only the surname wants both, and a model
    # that settles on the one it knows more about has answered a different question.
    {"q": "Who here is called Vance?", "expect": ["person:otto", "person:delia"]},
    # A false premise about a real person: Ada is in Turin and has not moved. The graph's
    # Calderwick is unchanged, and an answer that subtracts anyone believed the question.
    {"q": "Since Ada Lovelace moved to Selby, who is left in Calderwick?",
     "expect": ["person:iris", "person:nell", "person:otto", "person:tam"]},
    # Nobody here is a welder; Hedy says she is "slowly learning to weld badly". The
    # question allows the near miss ("even a little"), so she is the answer -- where
    # "Nobody here does underwater welding" below allows nothing and expects nobody.
    {"q": "Who here can weld, even a little?", "expect": ["person:hedy"]},

    # --- what people said, which only their own words carry -------------------------------
    # Pell has no subject, employer or place -- only a line saying he just joined. The
    # graph has no joining date, so "who joined most recently" is not askable here; this
    # is the nearest question that is, and only the quote can answer it.
    {"q": "Who said they had just joined?", "expect": ["person:pell"]},
    # Her own words name data engineering and hospitals; the answer is those two subjects,
    # not Alan, whom she also mentions.
    {"q": "What did Vera Lund say she works on?", "expect": ["topic:data", "topic:healthcare"]},
    # Each of these names something only a message carries -- an evening class, a scanner,
    # a preference, a stack of CVs, a distinction, a half of a job -- and none of them
    # names the subject, employer or town the person is joined to, so the edges cannot
    # answer them. Delia's subject is bookbinding and Alan's is healthcare; neither word
    # appears above.
    {"q": "Who said they teach in the evenings?", "expect": ["person:delia"]},
    {"q": "Who mentioned working on medical imaging?", "expect": ["person:alan"]},
    {"q": "Who said they would rather be outdoors than at a desk?", "expect": ["person:iris"]},
    {"q": "Who said they read a lot of CVs?", "expect": ["person:bram"]},
    {"q": "Who said the planning is not the same job as the fixing?",
     "expect": ["person:oskar"]},
    {"q": "Who said they do the unglamorous half of the work?", "expect": ["person:otto"]},
    # The one whose evidence is in somebody else's message: Tam says it about Iris, and
    # Iris's own line never mentions digging.
    {"q": "Who did Tam Quillon say tells him where to dig?", "expect": ["person:iris"]},

    # --- drawn by the generator, then read before being kept ---------------------------
    #
    # `ml_stack.world.questions` writes questions out of a world's own truth. Handed this
    # graph -- a `World` around it and nothing else -- it asks its templates about these
    # people, so half the set costs a seed rather than an afternoon and is not bounded by
    # whatever the person writing it happened to think of. Its `kind` tags are the kinds
    # this set already files under, so what it draws lands where the hand-written ones
    # land. What it cannot draw here is what the community has no relation for -- units,
    # products, works_with, reports_to -- and its `quote` questions, whose patterns are
    # the simulator's sentences rather than these.
    #
    # Every one below was read before it was kept. The ones naming somebody in the crowd,
    # and the ones that only reworded a question already asked, were dropped.
    {"q": "Who knows about recruiting?", "expect": ["person:bram"]},
    {"q": "Who knows about healthcare?", "expect": ["person:alan", "person:vera"]},
    {"q": "Who knows about geotechnics?", "expect": ["person:tam"]},
    {"q": "Who is based in Turin?", "expect": ["person:ada", "person:hedy"]},
    {"q": "Who is based in Harrowgate?",
     "expect": ["person:charles", "person:delia", "person:mary", "person:oskar"]},
    {"q": "Who was at Northern Trade Fair?",
     "expect": ["person:charles", "person:dorian", "person:grace", "person:milo",
                "person:sela"]},
    {"q": "Who could take on selling a machine nobody has heard of?",
     "expect": ["person:grace"]},
    {"q": "Who could take on keeping a production line running?",
     "expect": ["person:ada", "person:charles"]},
    {"q": "Where does Iris Bellweather work?", "expect": ["org:brayfield"]},
    {"q": "Where does Hedy Marchetti work?", "expect": ["org:quenlow"]},
    {"q": "Where is Nell Ashgrove based?", "expect": ["place:calderwick"]},
    {"q": "Where is Grace Hopper based?", "expect": ["place:dunmore"]},
    {"q": "Which places do the people who know about automation live in?",
     "expect": ["place:dunmore", "place:selby"]},
    {"q": "What is Mary Somerville good at?",
     "expect": ["person:mary", "topic:onboarding"]},
    {"q": "What is Tam Quillon good at?", "expect": ["person:tam", "topic:geotechnics"]},
    {"q": "What does keeping a production line running need?", "expect": ["topic:repair"]},
    {"q": "Which projects or openings want marketing?", "expect": ["opportunity:sellmachine"]},
    {"q": "What is Harnley Health running or offering?",
     "expect": ["opportunity:clinicaldata"]},
    {"q": "Which events did Grace Hopper go to?", "expect": ["event:tradefair"]},
    {"q": "Which events did Ada Lovelace go to?", "expect": ["event:makersnight"]},

    # --- the right answer is nobody ----------------------------------------------------
    {"q": "Nobody here does underwater welding. Who could?", "expect": []},
    {"q": "Who is the chief financial officer?", "expect": []},
    {"q": "Is anyone here a dentist?", "expect": []},
    {"q": "Which of these people has a pilot's licence?", "expect": []},
    {"q": "Who here has worked in film or television?", "expect": []},
    {"q": "Which company here makes shoes?", "expect": []},
    {"q": "hi", "expect": []},
    {"q": "good morning, anyone about?", "expect": []},
    {"q": "thanks, thats all i needed", "expect": []},
]
