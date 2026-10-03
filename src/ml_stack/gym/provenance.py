"""Native simulator dependency and scenario provenance."""

import hashlib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def native_provenance(environment, env):
    """Describe installed dependencies and the scenario files actually opened."""
    libraries = ["gymnasium", "numpy"]
    if environment in {"car", "traffic-driving"}:
        libraries.append("metadrive-simulator")
    if environment == "warehouse":
        libraries.append("rware")
    if environment in {"traffic", "traffic-driving"}:
        libraries.extend(("sumo-rl", "eclipse-sumo"))
    versions = {}
    for name in libraries:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    files = {}
    native = getattr(env, "unwrapped", env)
    for name, attribute in (("net_file", "_net"), ("route_file", "_route")):
        value = getattr(native, attribute, None)
        if value:
            path = Path(value).expanduser().resolve(strict=True)
            files[name] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    return {"versions": versions, "scenario_files": files}
