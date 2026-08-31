"""What a page looks like, per kind of site.

Reading a conversation out of a web app is the same job every time — find the rows, find who
wrote each one and what it says, find the identifier that makes a row the same row tomorrow —
and different only in which selectors do it. So the differences are data here, and the reading
is written once.

A preset is a starting point, not a promise. Sites move their markup, and the day one does,
the fix is one selector in a copy of a preset rather than a rewrite:

    site = SLACK.but(rows="[data-qa='message_container']")
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace


@dataclass(frozen=True)
class Site:
    """Where the words are on a page, and how to walk to more of them."""

    name: str
    # css for one row, the author, and the words. Several selectors may be given, comma
    # separated, and the first that matches a row wins
    rows: str
    author: str = ""
    body: str = ""
    # where a row's durable identifier lives: an attribute, and the shape to pull out of it
    key_attrs: tuple[str, ...] = ("data-item-key", "id")
    key_pattern: str = r"(\d{10}\.\d{6})"
    # the element that scrolls, and which way older content lies
    pane: str = "[role='main']"
    scroll_up: bool = True
    # when no row matches, the whole pane is taken as one blob so a person can see what moved
    fallback_pane: str = "[role='main']"
    max_chars: int = 2000
    settle_ms: int = 2500
    extra: dict[str, str] = field(default_factory=dict)

    def but(self, **changes) -> Site:
        """This preset with something different about it."""
        return replace(self, **changes)

    def key_of(self, raw: str) -> str:
        """The durable identifier inside an attribute value, or the value itself."""
        found = re.search(self.key_pattern, raw or "") if self.key_pattern else None
        return found.group(1) if found else (raw or "").strip()


# A conversation in Slack. The identifier is the message timestamp, which is the only thing
# about a Slack message that does not change; it is embedded in two different attributes
# depending on how the row was rendered, so both are tried.
SLACK = Site(
    name="slack",
    rows="[data-qa='message_container'], [role='listitem'][id^='message-list_']",
    author="[data-qa='message_sender_name'], .c-message__sender",
    body=".p-rich_text_section, [data-qa='message-text']",
    pane="[data-qa='slack_kit_scrollbar'], .c-scrollbar__hider, [role='main']",
    fallback_pane="[data-qa='message_pane'], .p-message_pane, [role='main']",
    extra={"replies": "[data-qa='reply_bar_count'], [data-qa='reply_bar'], [class*='reply_bar']"},
    settle_ms=4000,
)

# Discord numbers its rows rather than timestamping them, so the identifier is the whole id
# attribute and there is no pattern to pull out of it.
DISCORD = Site(
    name="discord",
    rows="li[id^='chat-messages-'], [class*='messageListItem']",
    author="[class*='username']",
    body="[id^='message-content-'], [class*='messageContent']",
    key_attrs=("id",),
    key_pattern="",
    pane="[class*='scroller'][class*='messagesWrapper'], [role='main']",
    fallback_pane="[class*='messagesWrapper'], [role='main']",
)

# An ordinary page is one row: itself. Reading it is finding the part that is the article and
# not the furniture around it.
WEBSITE = Site(
    name="website",
    rows="article, main, [role='main'], .post, .entry-content",
    author="[rel='author'], .author, [itemprop='author']",
    body="",
    key_attrs=("id",),
    key_pattern="",
    pane="[role='main'], body",
    scroll_up=False,
    fallback_pane="body",
    max_chars=20000,
    settle_ms=1500,
)

PRESETS = {site.name: site for site in (WEBSITE, SLACK, DISCORD)}


def preset(name: str) -> Site:
    """A preset by name. Raises with the list, because a typo should not read as an empty site."""
    try:
        return PRESETS[name.strip().lower()]
    except KeyError:
        raise KeyError(f"no preset called {name!r}. There is: "
                       + ", ".join(sorted(PRESETS))) from None
