"""Procedural or imported native MetaDrive map definitions."""

import json

from ml_stack.gym.world_files import directory, imported, record

SCHEMA = {"modes": ["procedural", "manual"], "fields": {
    "map": {"type": "text", "default": "SCSCS", "mode": "procedural", "format": "Native MetaDrive block sequence"},
    "map_file": {"type": "file", "mode": "manual", "format": "MetaDrive PGMap metadata JSON"}},
    "constraint": "Native PG road generation or exact imported native block configurations"}


def build(world, seed):
    """Return native PG map settings and their definition provenance."""
    seed = int(world.get("seed", seed))
    definition = {**world, "mode": world.get("mode", "procedural"), "seed": seed}
    path = directory("metadrive", definition)
    if definition["mode"] == "procedural":
        sequence = str(world.get("map", "SCSCS"))
        if not sequence or len(sequence) > 24:
            raise ValueError("Native block sequence must contain 1 through 24 blocks")
        cfg, files = {"map": sequence}, {}
    elif definition["mode"] == "manual":
        source = imported(world["map_file"])
        saved = json.loads(source.read_text())
        if not isinstance(saved.get("block_sequence"), list) or not isinstance(saved.get("map_config"), dict):
            raise ValueError("Manual car map requires native block_sequence and map_config JSON")
        cfg = {"map": 3, "map_config": {**saved["map_config"], "type": "pg_map_file", "config": saved["block_sequence"]}}
        files = {"map_file": source}
    else:
        raise ValueError("Car world mode must be procedural or manual")
    return cfg, record(path, "metadrive", definition, files)
