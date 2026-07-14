#!/usr/bin/env python3
"""Private fresh-process worker for full Atlas analyzer determinism checks."""

from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.atlas_analyzer import AnalyzerLimits, analyze_map, canonical_json, sha256_file


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: atlas_cold_worker.py SPECIFICATION.json")
    specification = json.loads(Path(sys.argv[1]).read_bytes())
    if set(specification) != {
        "schema", "bsp", "output_dir", "canonical_map_id", "provenance",
        "cm_oracle", "pmove_oracle", "hook_oracle", "hook_attestation",
        "fall_oracle",
        "packer", "verifier", "limits", "generator_claims_sha256",
        "generator_claims", "generator_safety", "hook_materialization",
    } or specification["schema"] != "q2-atlas-cold-worker-v1":
        raise SystemExit("invalid independent cold analyzer specification")
    optional_path = lambda value: None if value is None else Path(value)
    manifest = analyze_map(
        Path(specification["bsp"]),
        Path(specification["output_dir"]),
        specification["canonical_map_id"],
        specification["provenance"],
        cm_oracle=Path(specification["cm_oracle"]),
        pmove_oracle=optional_path(specification["pmove_oracle"]),
        hook_oracle=optional_path(specification["hook_oracle"]),
        fall_oracle=optional_path(specification["fall_oracle"]),
        hook_attestation=optional_path(specification["hook_attestation"]),
        packer=Path(specification["packer"]),
        verifier=Path(specification["verifier"]),
        limits=AnalyzerLimits(**specification["limits"]),
        generator_claims_sha256=specification["generator_claims_sha256"],
        generator_claims=specification["generator_claims"],
        generator_safety=specification["generator_safety"],
        hook_materialization=specification["hook_materialization"],
        independent_cold=False,
    )
    result = {
        "atlas_sha256": sha256_file(
            Path(specification["output_dir"])
            / f"{manifest['canonical_map_id']}.atlas.bin"
        ),
        "canonical_map_id": manifest["canonical_map_id"],
        "schema": "q2-atlas-cold-worker-result-v1",
    }
    sys.stdout.buffer.write(canonical_json(result) + b"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
