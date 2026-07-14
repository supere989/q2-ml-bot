#!/usr/bin/env python3
"""Prepare or validate a deterministic offline generated-map claim campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.generator_claim_validator import (  # noqa: E402
    ClaimValidationError,
    build_generator_claims,
    canonical_bytes,
    file_sha256,
    generator_claims_sha256,
    validate_generated_map,
)


CAMPAIGN_SCHEMA = "q2-generator-claim-campaign-v1"


def _canonical_error(exc: Exception) -> str:
    return " ".join(str(exc).replace("\n", " ").split())[:4096]


def prepare_claims(map_paths: list[Path], expected_count: int) -> dict:
    rows = []
    for map_path in map_paths:
        try:
            claims = build_generator_claims(map_path)
            output = map_path.with_suffix(".generator-claims.json")
            output.write_bytes(canonical_bytes(claims))
            rows.append(
                {
                    "map": map_path.stem,
                    "passed": True,
                    "generator_claims_sha256": generator_claims_sha256(claims),
                    "error": None,
                }
            )
        except (ClaimValidationError, OSError, ValueError) as exc:
            rows.append(
                {
                    "map": map_path.stem,
                    "passed": False,
                    "generator_claims_sha256": None,
                    "error": _canonical_error(exc),
                }
            )
    failures = [f"{row['map']}: {row['error']}" for row in rows if not row["passed"]]
    if len(rows) != expected_count:
        failures.append(f"map count {len(rows)} != required {expected_count}")
    return {
        "schema": CAMPAIGN_SCHEMA,
        "phase": "prepare_claims",
        "expected_count": expected_count,
        "map_count": len(rows),
        "pass_count": sum(row["passed"] for row in rows),
        "passed": not failures,
        "maps": rows,
        "failures": sorted(failures),
    }


def validate_campaign(
    map_paths: list[Path], expected_count: int, b1_gate: Path
) -> dict:
    rows = []
    for map_path in map_paths:
        analysis_path = map_path.with_suffix(".analysis.manifest.json")
        try:
            report = validate_generated_map(
                map_path, analysis_path, b1_gate_path=b1_gate
            )
            report_bytes = canonical_bytes(report)
            rows.append(
                {
                    "map": map_path.stem,
                    "passed": report["passed"],
                    "report_sha256": hashlib.sha256(report_bytes).hexdigest(),
                    "bsp_sha256": report["identities"]["bsp_sha256"],
                    "atlas_sha256": json.loads(
                        analysis_path.read_text(encoding="utf-8")
                    )["identity"]["atlas_sha256"],
                    "generator_claims_sha256": report["identities"][
                        "generator_claims_sha256"
                    ],
                    "failures": report["failures"],
                }
            )
        except (ClaimValidationError, OSError, ValueError, KeyError) as exc:
            rows.append(
                {
                    "map": map_path.stem,
                    "passed": False,
                    "report_sha256": None,
                    "bsp_sha256": (
                        file_sha256(map_path.with_suffix(".bsp"))
                        if map_path.with_suffix(".bsp").is_file()
                        else None
                    ),
                    "atlas_sha256": None,
                    "generator_claims_sha256": None,
                    "failures": [_canonical_error(exc)],
                }
            )
    failures = [
        f"{row['map']}: {failure}"
        for row in rows
        for failure in row["failures"]
    ]
    if len(rows) != expected_count:
        failures.append(f"map count {len(rows)} != required {expected_count}")
    return {
        "schema": CAMPAIGN_SCHEMA,
        "phase": "compiled_validation",
        "expected_count": expected_count,
        "map_count": len(rows),
        "pass_count": sum(row["passed"] for row in rows),
        "passed": not failures,
        "b1_gate_sha256": file_sha256(b1_gate),
        "maps": rows,
        "failures": sorted(failures),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated-dir", type=Path, required=True)
    parser.add_argument("--glob", default="*.map")
    parser.add_argument("--expected-count", type=int, default=20)
    parser.add_argument(
        "--phase", choices=("prepare", "validate"), default="validate"
    )
    parser.add_argument(
        "--b1-gate",
        type=Path,
        default=ROOT / "docs" / "multires" / "B1-GATE.json",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    map_paths = sorted(args.generated_dir.glob(args.glob))
    if args.phase == "prepare":
        report = prepare_claims(map_paths, args.expected_count)
    else:
        report = validate_campaign(map_paths, args.expected_count, args.b1_gate)
    payload = canonical_bytes(report)
    if args.output:
        args.output.write_bytes(payload)
    sys.stdout.buffer.write(payload)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
