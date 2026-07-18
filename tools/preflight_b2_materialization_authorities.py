#!/usr/bin/env python3
"""Read-only preflight for the exact final B2 materialization authorities."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.materialize_generated_cohort import (  # noqa: E402
    MaterializeCohortError,
    preflight_authority_records,
)


SCHEMA = "q2-b2-materialization-authority-preflight-v1"


def preflight(
    *,
    cm_oracle: Path,
    pmove_oracle: Path,
    hook_oracle: Path,
    fall_oracle: Path,
    hook_parity_attestation: Path,
) -> dict[str, object]:
    """Validate the same byte authorities as the final producer without writes."""

    records = preflight_authority_records(
        cm_oracle=cm_oracle.expanduser().absolute(),
        pmove_oracle=pmove_oracle.expanduser().absolute(),
        hook_oracle=hook_oracle.expanduser().absolute(),
        fall_oracle=fall_oracle.expanduser().absolute(),
        hook_attestation=hook_parity_attestation.expanduser().absolute(),
    )
    return {
        "schema": SCHEMA,
        "passed": True,
        "authorities": records,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the exact B1 gate, oracle, and hook-attestation bytes used "
            "by final B2 materialization without creating cohort evidence."
        )
    )
    parser.add_argument("--cm-oracle", type=Path, required=True)
    parser.add_argument("--pmove-oracle", type=Path, required=True)
    parser.add_argument("--hook-oracle", type=Path, required=True)
    parser.add_argument("--fall-oracle", type=Path, required=True)
    parser.add_argument("--hook-parity-attestation", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = preflight(
            cm_oracle=args.cm_oracle,
            pmove_oracle=args.pmove_oracle,
            hook_oracle=args.hook_oracle,
            fall_oracle=args.fall_oracle,
            hook_parity_attestation=args.hook_parity_attestation,
        )
    except (MaterializeCohortError, OSError) as error:
        print(f"materialization authority preflight failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
