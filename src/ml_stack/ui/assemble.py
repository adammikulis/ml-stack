"""One page out of component files.

A component is one ``<name>.html`` file holding, in any order:

* one ``<style>`` block, optional -- the rules for its own markup;
* ``<template>`` blocks, optional -- the markup put inside its ``<name></name>`` tag in the
  shell, or with ``mount="spot"`` the markup put at the shell's ``<!--mount:spot-->`` comment;
* one ``<script>`` block -- the script defining the custom element ``<name>``.

The shell is the page's own markup, with the component tags and mount comments where the
components go, ``__STYLES__`` where their styles go and ``__SCRIPTS__`` where their scripts go.
A component in the list with no tag in the shell contributes nothing; a tag in the shell with
no component behind it is removed.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

_STYLE = re.compile(r"<style>(.*?)</style>", re.S)
_TEMPLATE = re.compile(r'<template(?:\s+mount="([^"]+)")?>(.*?)</template>', re.S)
_SCRIPT = re.compile(r"<script>(.*?)</script>", re.S)
_TAG = re.compile(r"<([a-z][a-z0-9]*-[a-z0-9-]+)></\1>")
_MOUNT = re.compile(r"<!--mount:([a-z0-9-]+)-->")


@dataclass(frozen=True)
class Parts:
    """What one component file holds."""

    style: str
    templates: dict[str, str]
    script: str


@dataclass(frozen=True)
class Component:
    """One component: its tag name and the file it is read from."""

    name: str
    path: Path

    def read(self) -> Parts:
        """The file's style, its templates by mount (``""`` for the tag's own) and its script."""
        text = self.path.read_text(encoding="utf-8")
        style = "".join(m.group(1) for m in _STYLE.finditer(text)).strip("\n")
        templates: dict[str, str] = {}
        for m in _TEMPLATE.finditer(text):
            key = m.group(1) or ""
            templates[key] = templates.get(key, "") + m.group(2).strip("\n")
        scripts = _SCRIPT.findall(text)
        if len(scripts) != 1:
            raise ValueError(f"{self.path}: one <script> block, found {len(scripts)}")
        return Parts(style=style, templates=templates, script=scripts[0].strip("\n"))


def load(directory: Path | str, names: Iterable[str]) -> list[Component]:
    """The components named, read from ``directory/<name>.html``."""
    root = Path(directory)
    return [Component(name, root / f"{name}.html") for name in names]


def assemble(shell: str, components: Sequence[Component]) -> str:
    """The shell with every listed component's markup, style and script in place."""
    parts = {c.name: c.read() for c in components}
    order = [c.name for c in components]
    present: set[str] = set()

    def at_tag(m: re.Match) -> str:
        name = m.group(1)
        if name not in parts:
            return ""
        present.add(name)
        return f"<{name}>{parts[name].templates.get('', '')}</{name}>"

    def at_mount(m: re.Match) -> str:
        spot = m.group(1)
        return "\n".join(parts[n].templates[spot] for n in order
                         if n in present and spot in parts[n].templates)

    # a template may hold another component's tag, so the tags are filled until none is left
    page = shell
    for _ in range(16):
        filled = _TAG.sub(at_tag, page)
        if filled == page:
            break
        page = filled
    page = _MOUNT.sub(at_mount, page)
    # a component with no markup at all is script and style alone, and is always in
    kept = [n for n in order if n in present or not parts[n].templates]
    styles = "\n".join(parts[n].style for n in kept if parts[n].style)
    styles = f"<style>\n{styles}\n</style>" if styles else ""
    scripts = "\n".join(f"<script>\n{parts[n].script}\n</script>" for n in kept)
    return page.replace("__STYLES__", styles, 1).replace("__SCRIPTS__", scripts, 1)
