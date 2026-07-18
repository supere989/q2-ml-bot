#!/usr/bin/env python3
"""Evaluate concurrent real-rollout records against a capacity gate."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from harness.runtime_attestation import load_runtime_manifest, verify_runtime_manifest
from harness.throughput_gate import ThroughputGateError, evaluate_throughput_gate


def _load_records(path: Path) -> list[dict]:
    if path.suffix == ".q2rb":
        from harness.rollout_protocol import RolloutBatch

        return [dict(RolloutBatch.decode(path.read_bytes()).metadata)]
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, list):
        records = value
    elif isinstance(value, dict) and isinstance(value.get("records"), list):
        records = value["records"]
    elif isinstance(value, dict):
        records = [value.get("metadata", value)]
    else:
        raise ThroughputGateError(f"{path}: expected a JSON object/list or q2rb batch")
    if not all(isinstance(record, dict) for record in records):
        raise ThroughputGateError(f"{path}: every collection record must be an object")
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("records", nargs="+", type=Path)
    parser.add_argument("--min-workers", type=int, default=2)
    parser.add_argument("--min-aggregate-sps", type=float, default=0.0)
    parser.add_argument("--baseline-sps", type=float, default=0.0)
    parser.add_argument("--min-speedup", type=float, default=1.0)
    parser.add_argument("--min-overlap-ratio", type=float, default=0.5)
    parser.add_argument("--max-timeouts", type=int, default=0)
    parser.add_argument("--expected-manifest", type=Path)
    parser.add_argument("--attestation-key-env", default="")
    parser.add_argument("--require-attestation-signature", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        records = [record for path in args.records for record in _load_records(path)]
        expected_digest = None
        if args.expected_manifest:
            manifest = load_runtime_manifest(args.expected_manifest)
            if (
                args.attestation_key_env
                and args.attestation_key_env not in os.environ
            ):
                raise ThroughputGateError(
                    "attestation key environment variable is unset: "
                    + args.attestation_key_env
                )
            key = (
                os.environ[args.attestation_key_env].encode("utf-8")
                if args.attestation_key_env else None
            )
            verification = verify_runtime_manifest(
                manifest,
                hmac_key=key,
                require_signature=args.require_attestation_signature,
            )
            if not verification.valid:
                raise ThroughputGateError(
                    "expected runtime manifest is invalid: "
                    + "; ".join(verification.errors)
                )
            expected_digest = verification.digest
        result = evaluate_throughput_gate(
            records,
            min_workers=args.min_workers,
            min_aggregate_sps=args.min_aggregate_sps,
            baseline_sps=args.baseline_sps,
            min_speedup=args.min_speedup,
            min_overlap_ratio=args.min_overlap_ratio,
            max_timeouts=args.max_timeouts,
            expected_runtime_manifest_sha256=expected_digest,
        )
        encoded = json.dumps(result.as_dict(), indent=2, sort_keys=True) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            temporary = args.output.with_name(args.output.name + ".tmp")
            temporary.write_text(encoded, encoding="utf-8")
            temporary.replace(args.output)
        print(encoded, end="")
        return 0 if result.passed else 2
    except (OSError, json.JSONDecodeError, ThroughputGateError, ValueError) as error:
        parser.error(str(error))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
