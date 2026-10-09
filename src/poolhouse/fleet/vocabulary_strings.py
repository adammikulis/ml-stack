"""Every string the vocabulary switch changes, by id: the plain wording and the friendly one.

Friendly wording is home and poolhouse language, and is limited to labels, headings, button text
and onboarding copy. It never replaces a command, a flag, an error code or an API name, and the
page keeps the plain word beside it (a tooltip, and on headings a visible hint). ``{name}`` slots
are filled by the page.
"""

from __future__ import annotations

#: id -> (professional, friendly)
CATALOGUE: dict[str, tuple[str, str]] = {
    "nav.pool": ("Pool", "The Pool Deck"),
    "nav.devices": ("Devices", "Rooms"),
    "nav.projects": ("Projects", "Workshops"),
    "nav.settings": ("Settings", "Household"),
    "action.leave": ("Leave", "Move out"),
    "action.pause": ("Pause pool", "Quiet hours"),
    "action.resume": ("Resume pool", "Open back up"),
    "action.share": ("Share with pool", "Bring it home"),
    "devices.title": ("Devices", "Rooms"),
    "devices.sub_one": ("One device ready to share compute.", "One room ready to share its power."),
    "devices.sub_many": ("{n} devices sharing compute in this pool.", "{n} rooms sharing the house."),
    "projects.title": ("Projects", "Workshops"),
    "projects.pick": ("Select a project", "Pick a workshop"),
    "settings.title": ("Settings", "Household"),
    "wizard.summary": ("Poolhouse setup · {at} of {total} · {name}", "Moving in · {at} of {total} · {name}"),
    "wizard.title": ("Set up this machine", "Settle into your room"),
    "wizard.pool": ("Clusters", "Pick a house"),
    "wizard.stage.0": ("Device", "Your room"),
    "wizard.stage.1": ("Pool", "The house"),
    "wizard.stage.2": ("Work", "The chores"),
    "wizard.stage.3": ("Startup", "Wake-up"),
    "wizard.stage.4": ("Downloads", "Groceries"),
    "wizard.stage.5": ("Capabilities", "Your skills"),
    "wizard.stage.6": ("Background", "While away"),
    "wizard.stage.7": ("Memory", "Floor space"),
    "wizard.stage.8": ("Ready", "Welcome home"),
    # the Developer section of Settings stays in plain words under both vocabularies
    "dev.heading": ("Developer", "Developer"),
    "dev.hint": ("Switch the interface wording live. The choice is kept in this browser.",) * 2,
    "dev.how": ("Also set with the address or the server environment:",) * 2,
    "dev.vocab": ("Vocabulary",) * 2,
    "dev.vocab_professional": ("Professional",) * 2,
    "dev.vocab_professional_body": ("Plain words everywhere: Devices, Projects, Settings, Join, Revoke, Pool.",) * 2,
    "dev.vocab_friendly": ("Friendly",) * 2,
    "dev.vocab_friendly_body": ("Home and poolhouse labels and headings, with the plain word kept beside them.",) * 2,
}
