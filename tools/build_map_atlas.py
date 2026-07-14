#!/usr/bin/env python3
"""Build deterministic Atlas v1 analysis artifacts from a BSP or stock PAK."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.atlas_analyzer import AtlasAnalysisError, analyze_map, canonical_json, sha256_file
from harness.corpus_quarantine import inventory_stock_pak
from harness.generated_claim_probes import (
    generated_bsp_provenance,
    load_generator_claims,
    load_generator_hook_materialization,
    load_generator_safety,
)


DEFAULT_CLIENT = Path("/home/raymondj/multires-worktrees/integration/q2-ml-client")
DEFAULT_LITHIUM = Path("/home/raymondj/multires-worktrees/integration/q2-lithium-3zb2")
DEFAULT_ATTESTATION = Path("/home/raymondj/multires-artifacts/atlas-v1/B1/hook-parity-pullspeed-1700.json")
DEFAULT_PROVENANCE = ROOT / "docs/multires/stock-q2dm1-q2dm8.provenance.json"


def extract_stock(pak: Path, directory: Path) -> dict[str, Path]:
    expected = {f"maps/q2dm{number}.bsp" for number in range(1, 9)}
    output = {}
    with pak.open("rb") as stream:
        header = stream.read(12)
        if len(header) != 12 or header[:4] != b"PACK":
            raise AtlasAnalysisError("invalid stock PAK header")
        offset, length = struct.unpack_from("<ii", header, 4)
        stream.seek(offset)
        directory_bytes = stream.read(length)
        for ordinal in range(length // 64):
            raw_name, member_offset, member_length = struct.unpack_from(
                "<56sii", directory_bytes, ordinal * 64
            )
            name = raw_name.split(b"\0", 1)[0].decode("ascii")
            if name not in expected:
                continue
            stream.seek(member_offset)
            destination = directory / Path(name).name
            destination.write_bytes(stream.read(member_length))
            output[Path(name).stem] = destination
    if set(output) != {f"q2dm{number}" for number in range(1, 9)}:
        raise AtlasAnalysisError("stock PAK extraction set mismatch")
    return output


def ensure_packer(path: Path | None) -> Path:
    if path is not None:
        return path.resolve()
    subprocess.run(
        ["cargo", "build", "--release", "-p", "q2-lattice", "--bin", "q2-atlas-pack"],
        cwd=ROOT, check=True,
    )
    return ROOT / "target/release/q2-atlas-pack"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--bsp", type=Path)
    source.add_argument("--stock-pak", type=Path)
    parser.add_argument(
        "--stock-map", action="append", choices=[f"q2dm{number}" for number in range(1, 9)],
        help="with --stock-pak, build only this map (repeatable)",
    )
    parser.add_argument("--map-id")
    parser.add_argument(
        "--generator-claims", type=Path,
        help="strict q2-generator-claims-v2 proposals for a generated BSP",
    )
    parser.add_argument("--provenance", type=Path, default=DEFAULT_PROVENANCE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--client-root", type=Path, default=DEFAULT_CLIENT)
    parser.add_argument("--lithium-root", type=Path, default=DEFAULT_LITHIUM)
    parser.add_argument("--hook-attestation", type=Path, default=DEFAULT_ATTESTATION)
    parser.add_argument("--packer", type=Path)
    parser.add_argument("--verifier", type=Path)
    args = parser.parse_args()
    if args.stock_map and not args.stock_pak:
        parser.error("--stock-map requires --stock-pak")
    if args.generator_claims and args.stock_pak:
        parser.error("--generator-claims is valid only with --bsp")
    provenance_doc = json.loads(args.provenance.read_bytes())
    records = {record["canonical_id"]: record for record in provenance_doc["records"]}
    packer = ensure_packer(args.packer)
    verifier = (
        args.verifier.resolve() if args.verifier is not None
        else packer.with_name("q2-atlas-verify")
    )
    cm = args.client_root / "release/q2-cm-oracle"
    pmove = args.client_root / "release/q2-pmove-oracle"
    hook = args.lithium_root / "tools/q2-hook-oracle"
    for path in (cm, pmove, hook, packer, verifier, args.hook_attestation):
        if not path.is_file():
            raise AtlasAnalysisError(f"required B1 artifact missing: {path}")
    manifests = []
    if args.stock_pak:
        inventory = inventory_stock_pak(args.stock_pak)
        if inventory.archive_sha256 != provenance_doc["records"][0]["archive_sha256"]:
            raise AtlasAnalysisError("stock PAK/provenance digest mismatch")
        with tempfile.TemporaryDirectory(prefix="q2-atlas-stock-") as temp:
            maps = extract_stock(args.stock_pak, Path(temp))
            selected = set(args.stock_map or maps)
            for map_id, bsp in sorted(maps.items()):
                if map_id not in selected:
                    continue
                started = time.monotonic()
                print(f"atlas: {map_id} started", file=sys.stderr, flush=True)
                manifests.append(analyze_map(
                    bsp, args.output, map_id, records[map_id], cm_oracle=cm,
                    pmove_oracle=pmove, hook_oracle=hook,
                    hook_attestation=args.hook_attestation, packer=packer,
                    verifier=verifier,
                ))
                print(
                    f"atlas: {map_id} completed in {time.monotonic() - started:.3f}s",
                    file=sys.stderr, flush=True,
                )
    else:
        claims = None
        safety = None
        hook_materialization = None
        claims_digest = None
        if args.generator_claims:
            if not args.map_id:
                parser.error("--map-id is required with --generator-claims")
            claims, claims_digest = load_generator_claims(
                args.generator_claims, args.map_id
            )
            safety = load_generator_safety(args.generator_claims, claims)
            hook_materialization = load_generator_hook_materialization(
                args.generator_claims, claims
            )
            record = generated_bsp_provenance(
                args.bsp, claims, claims_digest
            )
        else:
            if not args.map_id or args.map_id not in records:
                parser.error("--bsp requires --map-id present in the provenance document")
            record = records[args.map_id]
        manifests.append(analyze_map(
            args.bsp, args.output, args.map_id, record, cm_oracle=cm,
            pmove_oracle=pmove, hook_oracle=hook,
            hook_attestation=args.hook_attestation, packer=packer,
            verifier=verifier,
            generator_claims_sha256=claims_digest,
            generator_claims=claims,
            generator_safety=safety,
            hook_materialization=hook_materialization,
        ))
    summary = {
        "schema": (
            "q2-atlas-generated-build-v1"
            if args.generator_claims else "q2-atlas-stock-build-v1"
        ),
        "maps": [{
            "canonical_map_id": item["canonical_map_id"],
            "bsp_sha256": item["bsp"]["sha256"],
            "atlas_sha256": item["artifacts"]["atlas"]["uncompressed_sha256"],
            "manifest_sha256": sha256_file(args.output / f"{item['canonical_map_id']}.analysis.manifest.json"),
        } for item in manifests],
    }
    summary_name = "generated-build-summary.json" if args.generator_claims else "stock-build-summary.json"
    (args.output / summary_name).write_bytes(canonical_json(summary) + b"\n")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
