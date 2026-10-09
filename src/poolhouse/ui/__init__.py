"""Pages assembled out of components: one file per component, a page is the list of them.

``assets_dir`` is the folder of ml-ui, the custom elements and tokens any page can load.
"""

from pathlib import Path

from poolhouse.ui.assemble import Component, assemble, load

__all__ = ["Component", "assemble", "assets", "assets_dir", "load"]


def assets_dir() -> Path:
    """The folder holding ``ml-ui.css``, ``ml-ui.js`` and the modules and gallery beside them."""
    return Path(__file__).parent / "assets"


def assets() -> dict[str, Path]:
    """Every file in `assets_dir`, by name."""
    return {p.name: p for p in assets_dir().iterdir() if p.is_file()}
