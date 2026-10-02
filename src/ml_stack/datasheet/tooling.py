"""The datasheet finder and readers as ``(schema, callable)`` pairs for a model."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ml_stack.datasheet.finder import NotFound, find
from ml_stack.sources import datasheet
from ml_stack.web import Engine, downloads_dir

MOST_PINS = 200
MOST_IMAGES = 2

# Invented parts under invented makers: nothing here is a real component.
SCHEMAS: list[dict[str, Any]] = [
    {"type": "function", "function": {
        "name": "datasheet_find",
        "description": "Find the datasheet PDF for a component by its part number, download "
                       "it, and check the PDF names the part. Returns its path on disk, the "
                       "address it came from, a hash, the page count and a confidence from 0 "
                       "to 1. Examples: \"Get the datasheet for the Quenlow QX2200\" → "
                       "datasheet_find(part=\"QX2200\", manufacturer=\"Quenlow\"); \"I need "
                       "the data on a PX-17A regulator\" → datasheet_find(part=\"PX-17A\"). "
                       "A low confidence means the PDF names only part of the number: check "
                       "the title before relying on it.",
        "parameters": {"type": "object", "properties": {
            "part": {"type": "string", "description": "the part number, e.g. \"QX2200\""},
            "manufacturer": {"type": "string",
                             "description": "the maker, when known, e.g. \"Quenlow\""}},
            "required": ["part"]}}},
    {"type": "function", "function": {
        "name": "datasheet_pins",
        "description": "Read the pin table of a datasheet that datasheet_find or web_download "
                       "saved: one row per pin with its number, name, type and description. "
                       "Example: datasheet_find returned path /cache/ab12_qx2200.pdf → "
                       "datasheet_pins(path=\"/cache/ab12_qx2200.pdf\").",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "the path datasheet_find returned"}},
            "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "datasheet_outline",
        "description": "Show the package outline and recommended land pattern pages of a "
                       "saved datasheet: the page numbers, and the drawings as images "
                       "cropped to the drawing. Use it when a footprint has to match the "
                       "datasheet's dimensions. Example: datasheet_outline(path=\"/cache/"
                       "ab12_qx2200.pdf\").",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "the path datasheet_find returned"}},
            "required": ["path"]}}},
]

PROMPTS: dict[str, tuple[str, ...]] = {
    "datasheet_find": ("get the datasheet for this part", "find the datasheet pdf",
                       "download the datasheet for this chip"),
    "datasheet_pins": ("what are the pins of this chip?", "list the pinout",
                       "which pin is enable?"),
    "datasheet_outline": ("show me the land pattern", "what is the package outline?",
                          "what are the footprint dimensions?"),
}


def _saved(path: str) -> Path:
    """The path, when it is a file under the download directory; ``ValueError`` otherwise."""
    where = Path(path).expanduser().resolve()
    if not where.is_relative_to(downloads_dir().resolve()) or not where.is_file():
        raise ValueError("pass the path datasheet_find or web_download returned")
    return where


def tools(*, engine: Engine | None = None, vision: bool = True
          ) -> list[tuple[dict[str, Any], Callable[[Mapping[str, Any]], Any]]]:
    """The datasheet tools; ``datasheet_outline`` only with ``vision``, since its answer is
    images. Each callable returns ``{"none": reason}`` instead of raising."""
    def finding(args: Mapping[str, Any]) -> Any:
        part = str(args.get("part") or "").strip()
        if not part:
            return {"none": "pass a part number"}
        try:
            return find(part, str(args.get("manufacturer") or ""), engine=engine).as_dict()
        except (NotFound, OSError, ValueError, RuntimeError, ImportError) as exc:
            return {"none": str(exc)}

    def pins(args: Mapping[str, Any]) -> Any:
        try:
            rows = datasheet.pin_tables(_saved(str(args.get("path") or "")))
        except (OSError, ValueError, RuntimeError, ImportError) as exc:
            return {"none": str(exc)}
        return [asdict(p) for p in rows[:MOST_PINS]] or {"none": "no pin table found"}

    def outline(args: Mapping[str, Any]) -> Any:
        try:
            where = _saved(str(args.get("path") or ""))
            found = datasheet.outline_pages(where)
        except (OSError, ValueError, RuntimeError, ImportError) as exc:
            return {"none": str(exc)}
        if not found:
            return {"none": "no package outline or land pattern page found"}
        return {"pages": [asdict(o) for o in found[:6]],
                "_images": [datasheet.render(where, o.page) for o in found[:MOST_IMAGES]]}

    pairs = [(SCHEMAS[0], finding), (SCHEMAS[1], pins)]
    if vision:
        pairs.append((SCHEMAS[2], outline))
    return pairs
