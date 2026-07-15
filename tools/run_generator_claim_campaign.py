#!/usr/bin/env python3
"""Prepare and validate one exact, declaration-bound generator cohort.

The source/materialization and Atlas analysis roots are deliberately separate:
both contain a ``<map>.routes.json`` with different authority.  Directory
globs, count-only admission, adjacent analysis lookup, replacement maps, and
passing-subset publication are not supported.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.generator_claim_validator import (  # noqa: E402
    ClaimValidationError,
    build_generator_claims,
    canonical_bytes,
    file_sha256,
    validate_generated_map,
)
from tools.run_generator_cohort import (  # noqa: E402
    GeneratorCohortError,
    STAGE_SUFFIXES,
    load_declaration,
    verify_stage_membership,
)
from tools.retired_cohort_registry import require_unretired_declaration  # noqa: E402


CAMPAIGN_SCHEMA = "q2-generator-claim-campaign-v2"


class ClaimCampaignError(ValueError):
    """Raised when a campaign cannot be published without weakening scope."""


def _canonical_error(exc: Exception) -> str:
    return " ".join(str(exc).replace("\n", " ").split())[:4096]


def _sha256_canonical(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _path_is_within(path: Path, directory: Path) -> bool:
    try:
        path.resolve().relative_to(directory.resolve())
    except ValueError:
        return False
    return True


def _require_distinct_roots(first: Path, second: Path, labels: str) -> None:
    if (
        first.resolve() == second.resolve()
        or _path_is_within(first, second)
        or _path_is_within(second, first)
    ):
        raise ClaimCampaignError(f"{labels} must be separate non-nested roots")


def _require_unpublished(path: Path, label: str) -> None:
    if path.exists() or path.is_symlink():
        raise ClaimCampaignError(f"{label} already exists; refusing overwrite")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _exclusive_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        _fsync_directory(path.parent)
    except Exception:
        path.unlink(missing_ok=True)
        raise


def _membership_projection(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "stage": report["stage"],
        "passed": report["passed"],
        "expected_map_count": report["expected_map_count"],
        "expected_file_count": report["expected_file_count"],
        "actual_file_count": report["actual_file_count"],
        "report_sha256": _sha256_canonical(report),
    }


def _base_report(
    declaration: Mapping[str, Any], declaration_sha256: str, phase: str
) -> dict[str, Any]:
    return {
        "schema": CAMPAIGN_SCHEMA,
        "cohort_id": declaration["cohort_id"],
        "declaration_sha256": declaration_sha256,
        "phase": phase,
        "expected_count": len(declaration["maps"]),
        "map_count": len(declaration["maps"]),
    }


def _failed_input_rows(
    declaration: Mapping[str, Any], message: str, *, validation: bool
) -> list[dict[str, Any]]:
    rows = []
    for declared in declaration["maps"]:
        if validation:
            rows.append({
                "ordinal": declared["ordinal"],
                "map": declared["map"],
                "passed": False,
                "report_sha256": None,
                "bsp_sha256": None,
                "atlas_sha256": None,
                "generator_claims_sha256": None,
                "failures": [message],
            })
        else:
            rows.append({
                "ordinal": declared["ordinal"],
                "map": declared["map"],
                "passed": False,
                "generator_claims_sha256": None,
                "error": message,
            })
    return rows


def _prepare_report(
    declaration: Mapping[str, Any],
    declaration_sha256: str,
    membership: Mapping[str, Any],
    rows: list[dict[str, Any]],
    failures: list[str],
    claims_membership: Mapping[str, Any] | None,
) -> dict[str, Any]:
    report = _base_report(declaration, declaration_sha256, "prepare_claims")
    report.update({
        "input_stages": {
            "materialized": _membership_projection(membership),
            "claims": (
                None if claims_membership is None
                else _membership_projection(claims_membership)
            ),
        },
        "pass_count": sum(row["passed"] for row in rows),
        "passed": not failures,
        "maps": rows,
        "failures": sorted(set(failures)),
    })
    return report


def prepare_claims(
    declaration_path: Path,
    materialized_dir: Path,
    claims_dir: Path,
) -> dict[str, Any]:
    """Build every claim first, then atomically publish one exact claims root."""

    declaration, declaration_sha256 = load_declaration(declaration_path)
    require_unretired_declaration(
        declaration_path, declaration, declaration_sha256
    )
    _require_distinct_roots(
        materialized_dir, claims_dir, "materialized and claims stages"
    )
    _require_unpublished(claims_dir, "claims stage")
    membership = verify_stage_membership(
        declaration, materialized_dir, "materialized"
    )
    if not membership["passed"]:
        failures = [
            f"materialized-stage: {failure}"
            for failure in membership["failures"]
        ]
        return _prepare_report(
            declaration,
            declaration_sha256,
            membership,
            _failed_input_rows(
                declaration, "materialized stage membership failed", validation=False
            ),
            failures,
            None,
        )

    built: list[tuple[Mapping[str, Any], bytes, str]] = []
    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    for declared in declaration["maps"]:
        map_path = materialized_dir / f"{declared['map']}.map"
        try:
            claims = build_generator_claims(map_path)
            payload = canonical_bytes(claims)
            digest = hashlib.sha256(payload).hexdigest()
            built.append((declared, payload, digest))
            rows.append({
                "ordinal": declared["ordinal"],
                "map": declared["map"],
                "passed": True,
                "generator_claims_sha256": digest,
                "error": None,
            })
        except (ClaimValidationError, OSError, ValueError, KeyError) as exc:
            error = _canonical_error(exc)
            failures.append(f"{declared['map']}: {error}")
            rows.append({
                "ordinal": declared["ordinal"],
                "map": declared["map"],
                "passed": False,
                "generator_claims_sha256": None,
                "error": error,
            })
    if failures:
        # No output directory or per-map claim is published from a partial set.
        return _prepare_report(
            declaration, declaration_sha256, membership, rows, failures, None
        )
    if len(built) != len(declaration["maps"]):
        raise ClaimCampaignError("complete claim build count contradicts declaration")

    claims_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(
        prefix=f".{claims_dir.name}.prepare-", dir=claims_dir.parent
    ))
    published = False
    try:
        for declared, payload, _digest in built:
            name = declared["map"]
            for suffix in STAGE_SUFFIXES["materialized"]:
                shutil.copyfile(
                    materialized_dir / f"{name}{suffix}",
                    temporary / f"{name}{suffix}",
                )
            _exclusive_write(
                temporary / f"{name}.generator-claims.json", payload
            )
        claims_membership = verify_stage_membership(
            declaration, temporary, "claims"
        )
        if not claims_membership["passed"]:
            raise ClaimCampaignError(
                "prepared claims stage failed exact membership: "
                + "; ".join(claims_membership["failures"])
            )
        _require_unpublished(claims_dir, "claims stage")
        os.rename(temporary, claims_dir)
        _fsync_directory(claims_dir.parent)
        published = True
    finally:
        if not published:
            shutil.rmtree(temporary, ignore_errors=True)

    final_membership = verify_stage_membership(
        declaration, claims_dir, "claims"
    )
    if not final_membership["passed"]:
        raise ClaimCampaignError(
            "published claims stage failed exact membership: "
            + "; ".join(final_membership["failures"])
        )
    return _prepare_report(
        declaration, declaration_sha256, membership, rows, [], final_membership
    )


def _validation_report(
    declaration: Mapping[str, Any],
    declaration_sha256: str,
    claims_membership: Mapping[str, Any],
    analysis_membership: Mapping[str, Any],
    b1_gate: Path,
    rows: list[dict[str, Any]],
    failures: list[str],
) -> dict[str, Any]:
    report = _base_report(
        declaration, declaration_sha256, "compiled_validation"
    )
    report.update({
        "input_stages": {
            "claims": _membership_projection(claims_membership),
            "analysis": _membership_projection(analysis_membership),
        },
        "pass_count": sum(row["passed"] for row in rows),
        "passed": not failures,
        "b1_gate_sha256": (
            file_sha256(b1_gate) if b1_gate.is_file() else None
        ),
        "maps": rows,
        "failures": sorted(set(failures)),
    })
    return report


def validate_campaign(
    declaration_path: Path,
    claims_dir: Path,
    analysis_dir: Path,
    b1_gate: Path,
) -> dict[str, Any]:
    """Validate exact claims and analysis roots in declared ordinal order."""

    declaration, declaration_sha256 = load_declaration(declaration_path)
    _require_distinct_roots(claims_dir, analysis_dir, "claims and analysis stages")
    claims_membership = verify_stage_membership(
        declaration, claims_dir, "claims"
    )
    analysis_membership = verify_stage_membership(
        declaration, analysis_dir, "analysis"
    )
    membership_failures = [
        *(
            f"claims-stage: {failure}"
            for failure in claims_membership["failures"]
        ),
        *(
            f"analysis-stage: {failure}"
            for failure in analysis_membership["failures"]
        ),
    ]
    if not b1_gate.is_file():
        membership_failures.append("B1 gate is missing")
    if membership_failures:
        return _validation_report(
            declaration,
            declaration_sha256,
            claims_membership,
            analysis_membership,
            b1_gate,
            _failed_input_rows(
                declaration, "exact input stage membership failed", validation=True
            ),
            membership_failures,
        )

    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    for declared in declaration["maps"]:
        name = declared["map"]
        map_path = claims_dir / f"{name}.map"
        analysis_path = analysis_dir / f"{name}.analysis.manifest.json"
        try:
            result = validate_generated_map(
                map_path, analysis_path, b1_gate_path=b1_gate
            )
            result_bytes = canonical_bytes(result)
            analysis = json.loads(
                analysis_path.read_text(encoding="utf-8")
            )
            row = {
                "ordinal": declared["ordinal"],
                "map": name,
                "passed": result["passed"],
                "report_sha256": hashlib.sha256(result_bytes).hexdigest(),
                "bsp_sha256": result["identities"]["bsp_sha256"],
                "atlas_sha256": analysis["identity"]["atlas_sha256"],
                "generator_claims_sha256": result["identities"][
                    "generator_claims_sha256"
                ],
                "failures": list(result["failures"]),
            }
        except (
            ClaimValidationError,
            OSError,
            ValueError,
            KeyError,
            TypeError,
        ) as exc:
            row = {
                "ordinal": declared["ordinal"],
                "map": name,
                "passed": False,
                "report_sha256": None,
                "bsp_sha256": (
                    file_sha256(claims_dir / f"{name}.bsp")
                    if (claims_dir / f"{name}.bsp").is_file()
                    else None
                ),
                "atlas_sha256": None,
                "generator_claims_sha256": None,
                "failures": [_canonical_error(exc)],
            }
        rows.append(row)
        failures.extend(
            f"{name}: {failure}" for failure in row["failures"]
        )
        if row["passed"] is not True and not row["failures"]:
            failures.append(f"{name}: validator returned false without a failure")
    return _validation_report(
        declaration,
        declaration_sha256,
        claims_membership,
        analysis_membership,
        b1_gate,
        rows,
        failures,
    )


def _require_report_location(
    report: Path, *stage_roots: Path
) -> None:
    _require_unpublished(report, "campaign report")
    if any(_path_is_within(report, root) for root in stage_roots):
        raise ClaimCampaignError("campaign report must be outside exact stage roots")


def _main_prepare(args: argparse.Namespace) -> int:
    _require_report_location(
        args.output, args.materialized_dir, args.claims_dir
    )
    report = prepare_claims(
        args.declaration, args.materialized_dir, args.claims_dir
    )
    _exclusive_write(args.output, canonical_bytes(report))
    sys.stdout.buffer.write(canonical_bytes(report))
    return 0 if report["passed"] else 1


def _main_validate(args: argparse.Namespace) -> int:
    _require_report_location(args.output, args.claims_dir, args.analysis_dir)
    report = validate_campaign(
        args.declaration,
        args.claims_dir,
        args.analysis_dir,
        args.b1_gate,
    )
    _exclusive_write(args.output, canonical_bytes(report))
    sys.stdout.buffer.write(canonical_bytes(report))
    return 0 if report["passed"] else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="phase", required=True)

    prepare = subparsers.add_parser(
        "prepare", help="publish an exact claims stage from materialized input"
    )
    prepare.add_argument("--declaration", type=Path, required=True)
    prepare.add_argument("--materialized-dir", type=Path, required=True)
    prepare.add_argument("--claims-dir", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.set_defaults(handler=_main_prepare)

    validate = subparsers.add_parser(
        "validate", help="validate separate exact claims and Atlas roots"
    )
    validate.add_argument("--declaration", type=Path, required=True)
    validate.add_argument("--claims-dir", type=Path, required=True)
    validate.add_argument("--analysis-dir", type=Path, required=True)
    validate.add_argument(
        "--b1-gate",
        type=Path,
        default=ROOT / "docs/multires/B1-GATE.json",
    )
    validate.add_argument("--output", type=Path, required=True)
    validate.set_defaults(handler=_main_validate)

    args = parser.parse_args(argv)
    try:
        return args.handler(args)
    except (ClaimCampaignError, GeneratorCohortError, OSError) as exc:
        print(f"generator claim campaign failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
