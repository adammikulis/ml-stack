"""Procedural or imported native MetaDrive map definitions."""

import json

from poolhouse.gym.world_files import directory, imported, record

SCHEMA = {"modes": ["procedural", "manual"], "fields": {
    "map": {"type": "text", "default": "SCSCS", "mode": "procedural",
            "format": "Native MetaDrive block sequence or integer block count (1 through 24)"},
    "map_file": {"type": "file", "mode": "manual", "format": "MetaDrive PGMap metadata JSON"}},
    "constraint": "Native PG road generation or exact imported native block configurations"}


def build(world, seed):
    """Return native PG map settings and their definition provenance."""
    seed = int(world.get("seed", seed))
    if not 0 <= seed < 2**31:
        raise ValueError("World seed must be between zero and 2147483647")
    definition = {**world, "mode": world.get("mode", "procedural"), "seed": seed}
    path = directory("metadrive", definition)
    if definition["mode"] == "procedural":
        road = world.get("map", "SCSCS")
        count_valid = type(road) is int and 1 <= road <= 24
        sequence_valid = isinstance(road, str) and 1 <= len(road) <= 24
        if not (count_valid or sequence_valid):
            raise ValueError("Native road must be a block sequence or integer block count from 1 through 24")
        cfg, files = {"map": road}, {}
    elif definition["mode"] == "manual":
        source = imported(world["map_file"])
        saved = json.loads(source.read_text())
        if not isinstance(saved, dict) or not isinstance(saved.get("block_sequence"), list) or not isinstance(saved.get("map_config"), dict):
            raise ValueError("Manual car map requires native block_sequence and map_config JSON")
        native_config = {key: value for key, value in saved["map_config"].items() if key != "seed"}
        cfg = {"map_config": {**native_config, "type": "pg_map_file", "config": saved["block_sequence"]}}
        files = {"map_file": source}
    else:
        raise ValueError("Car world mode must be procedural or manual")
    return {**cfg, "world_seed": seed}, record(path, "metadrive", definition, files)
