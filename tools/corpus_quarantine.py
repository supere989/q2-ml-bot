#!/usr/bin/env python3
"""Inspect authored-map archives without extracting or installing them."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from harness.corpus_quarantine import (
    admit_corpus,
    classify_duplicates,
    duplicate_signature,
    inventory_stock_pak,
    load_provenance,
    quarantine_archive,
    write_json,
)
from harness.ibsp38 import stock_inventory_record


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Fail-closed ZIP/PAK quarantine and metadata-only IBSP-38 admission; "
            "never extracts, installs, or executes archive content"
        )
    )
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--include-bsp", action="append", default=[], metavar="GLOB",
        help="select BSP members for parsing (repeatable; default is every BSP)",
    )
    parser.add_argument(
        "--stock-q2dm", action="store_true",
        help="select exactly maps/q2dm1.bsp through maps/q2dm8.bsp",
    )
    parser.add_argument(
        "--provenance", type=Path,
        help="q2-corpus-provenance-v1 JSON required for corpus admission",
    )
    parser.add_argument("--near-threshold", type=float, default=0.94)
    args = parser.parse_args()

    patterns = list(args.include_bsp)
    if args.stock_q2dm:
        if patterns:
            parser.error("--stock-q2dm cannot be combined with --include-bsp")
        report = inventory_stock_pak(args.archive)
    else:
        report = quarantine_archive(args.archive, include_bsp=tuple(patterns))
    output = {"quarantine": report.to_dict()}

    if args.provenance is not None:
        entries = admit_corpus(report, load_provenance(args.provenance))
        output["admission"] = {
            "schema": "q2-corpus-admission-v1",
            "entries": [
                {
                    "canonical_id": entry.canonical_id,
                    "aliases": list(entry.aliases),
                    "bsp_member": entry.bsp_member,
                    "provenance": entry.provenance.to_dict(),
                    "duplicate_signature": duplicate_signature(entry.metadata),
                }
                for entry in entries
            ],
            "duplicates": [
                asdict(classification)
                for classification in classify_duplicates(
                    entries, near_threshold=args.near_threshold
                )
            ],
        }
        if args.stock_q2dm:
            output["stock_inventory"] = {
                "schema": "q2-stock-map-fixtures-v1",
                "archive_sha256": report.archive_sha256,
                "maps": [
                    stock_inventory_record(entry.canonical_id, entry.metadata)
                    for entry in entries
                ],
            }

    write_json(args.output, output)
    print(
        f"admitted quarantine report: {len(report.members)} members, "
        f"{len(report.bsp_metadata)} selected IBSP-38 maps -> {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
