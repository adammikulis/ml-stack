"""Write a CycloneDX 1.5 (JSON) software bill of materials for what a release ships.

Components: the installed closure of the base requirements and the shipped extras (the same
list `scripts/notices.py` puts in THIRD_PARTY_NOTICES.md, read from the interpreter that runs
this) and the Rust crates in `app/src-tauri/Cargo.lock` (read from the lock file, with each
crate's registry checksum as its SHA-256). Nothing is fetched: no network, no cargo, no index.
The output is deterministic (same environment, same bytes): the serial number is derived from
the content, and the timestamp is `SOURCE_DATE_EPOCH` when set, else left out.

    python scripts/sbom.py --out sbom.cdx.json
    python scripts/sbom.py --check sbom.cdx.json     # shape check; exit 1 on a problem

Run it in the environment a bundle is built from (release.yml installs the wheel with the
extras into a clean venv first). A Python package missing from that environment is missing
from the SBOM, so the release job also runs `scripts/notices.py --check` there.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import sys
import tomllib
import uuid
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOCK = ROOT / "app" / "src-tauri" / "Cargo.lock"
SPEC_VERSION = "1.5"
PURL = re.compile(r"^pkg:(pypi|cargo)/[A-Za-z0-9._-]+@[A-Za-z0-9._+~-]+$")
NAMESPACE = uuid.UUID("5d1c6a3e-0d7e-4a52-9a0b-6a4f2b1f5f10")


def _notices():
    spec = importlib.util.spec_from_file_location("ml_stack_notices", ROOT / "scripts" / "notices.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _licenses(expression: str) -> list[dict]:
    """CycloneDX ``licenses``: an SPDX expression when the text is one, else a named licence."""
    text = (expression or "").strip()
    if not text or text.upper() == "UNKNOWN":
        return []
    if re.fullmatch(r"[A-Za-z0-9.+-]+( (AND|OR|WITH) [A-Za-z0-9.+-]+)+", text):
        return [{"expression": text}]
    if re.fullmatch(r"[A-Za-z0-9.+-]+", text):
        return [{"license": {"id": text}}] if _spdx_like(text) else [{"license": {"name": text}}]
    return [{"license": {"name": text[:200]}}]


def _spdx_like(text: str) -> bool:
    return bool(re.fullmatch(r"(MIT|Apache-2\.0|BSD-[23]-Clause|ISC|MPL-2\.0|Unlicense|0BSD|Zlib|PSF-2\.0)", text))


def _component(ecosystem: str, name: str, version: str, license_text: str = "", sha256: str = "") -> dict:
    purl = f"pkg:{ecosystem}/{name.lower() if ecosystem == 'pypi' else name}@{version}"
    row: dict = {"type": "library", "bom-ref": purl, "name": name, "version": version, "purl": purl}
    if licenses := _licenses(license_text):
        row["licenses"] = licenses
    if sha256:
        row["hashes"] = [{"alg": "SHA-256", "content": sha256}]
    return row


def python_components() -> list[dict]:
    return [_component("pypi", n, v, lic) for n, v, lic in _notices().python_closure()]


def rust_components(lock: Path = LOCK) -> list[dict]:
    """Every registry or git crate of the lock file (the workspace's own crate has no source)."""
    if not lock.is_file():
        return []
    packages = tomllib.loads(lock.read_text(encoding="utf-8")).get("package", [])
    rows = [_component("cargo", p["name"], p["version"], sha256=p.get("checksum", ""))
            for p in packages if p.get("source")]
    return sorted(rows, key=lambda c: (c["name"].lower(), c["version"]))


def build() -> dict:
    version = (ROOT / "version.txt").read_text(encoding="utf-8").strip()
    components = sorted(python_components(), key=lambda c: c["name"].lower()) + rust_components()
    seen: dict[str, dict] = {}
    for c in components:                       # one component per package url
        seen.setdefault(c["bom-ref"], c)
    components = list(seen.values())
    metadata: dict = {"component": {"type": "application", "bom-ref": f"pkg:pypi/ml-stack@{version}",
                                    "name": "ml-stack", "version": version,
                                    "licenses": [{"license": {"id": "Apache-2.0"}}],
                                    "purl": f"pkg:pypi/ml-stack@{version}"},
                      "tools": {"components": [{"type": "application", "name": "scripts/sbom.py"}]}}
    if epoch := os.environ.get("SOURCE_DATE_EPOCH", "").strip():
        metadata["timestamp"] = datetime.fromtimestamp(int(epoch), UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    digest = json.dumps([c["bom-ref"] for c in components], sort_keys=True)
    return {"bomFormat": "CycloneDX", "specVersion": SPEC_VERSION,
            "serialNumber": f"urn:uuid:{uuid.uuid5(NAMESPACE, version + digest)}", "version": 1,
            "metadata": metadata, "components": components}


def problems(bom: object) -> list[str]:
    """What is wrong with ``bom`` against the part of the CycloneDX 1.5 schema this tool relies on."""
    if not isinstance(bom, dict):
        return ["not an object"]
    out: list[str] = []
    if bom.get("bomFormat") != "CycloneDX":
        out.append("bomFormat is not CycloneDX")
    if bom.get("specVersion") != SPEC_VERSION:
        out.append(f"specVersion is not {SPEC_VERSION}")
    if not re.fullmatch(r"urn:uuid:[0-9a-f]{8}(-[0-9a-f]{4}){3}-[0-9a-f]{12}", str(bom.get("serialNumber", ""))):
        out.append("serialNumber is not a urn:uuid")
    if not isinstance(bom.get("version"), int) or bom["version"] < 1:
        out.append("version is not a positive integer")
    top = (bom.get("metadata") or {}).get("component") or {}
    if not {"type", "name", "version"} <= set(top):
        out.append("metadata.component needs type, name and version")
    refs: set[str] = set()
    for i, c in enumerate(bom.get("components") or []):
        where = f"components[{i}] {c.get('name') if isinstance(c, dict) else c}"
        if not isinstance(c, dict) or not {"type", "name", "version", "purl", "bom-ref"} <= set(c):
            out.append(f"{where}: needs type, name, version, purl and bom-ref")
            continue
        if c["type"] not in ("library", "application", "framework", "file"):
            out.append(f"{where}: unknown type {c['type']!r}")
        if not PURL.match(c["purl"]):
            out.append(f"{where}: malformed purl {c['purl']!r}")
        if c["bom-ref"] in refs:
            out.append(f"{where}: bom-ref used twice")
        refs.add(c["bom-ref"])
        for h in c.get("hashes", []):
            if h.get("alg") != "SHA-256" or not re.fullmatch(r"[0-9a-f]{64}", str(h.get("content", ""))):
                out.append(f"{where}: malformed hash")
        for entry in c.get("licenses", []):
            body = entry.get("license") or {}
            if not ("expression" in entry or "id" in body or "name" in body):
                out.append(f"{where}: a licence needs an id, a name or an expression")
    return out


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, help="write the SBOM here")
    ap.add_argument("--check", type=Path, help="check an existing SBOM file instead of writing one")
    args = ap.parse_args(argv)
    if args.check:
        found = problems(json.loads(args.check.read_text(encoding="utf-8")))
        for p in found:
            sys.stderr.write(f"sbom: {p}\n")
        return 1 if found else 0
    bom = build()
    found = problems(bom)
    if found:
        for p in found:
            sys.stderr.write(f"sbom: {p}\n")
        return 1
    text = json.dumps(bom, indent=1, sort_keys=True) + "\n"
    if args.out:
        args.out.write_text(text, encoding="utf-8")
        py = sum(c["purl"].startswith("pkg:pypi") for c in bom["components"])
        sys.stdout.write(f"wrote {args.out}: {py} Python, {len(bom['components']) - py} Rust components\n")
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
