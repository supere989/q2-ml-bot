#!/usr/bin/env python3
"""Produce qualification-native materialization and immutable claims stages."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Callable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.atlas_b1_authority import (  # noqa: E402
    B1AuthorityError,
    admit_hook_parity_attestation,
    load_b1_authority_gate,
)
from tools.assemble_b2_qualification import (  # noqa: E402
    REQUIRED_STAGE_CHECKS,
    STAGE_SCHEMA,
)
from tools.generator_claim_validator import (  # noqa: E402
    ClaimValidationError,
    build_generator_claims,
)
from tools.materialize_generated_cohort import (  # noqa: E402
    MaterializeCohortError,
    _run_process_group,
    _validate_result,
)
from tools.run_b2_qualification_compile import (  # noqa: E402
    COMPILED_SUFFIXES,
    QualificationCompileError,
    _current_implementation,
    _file_record,
    _flat_records,
    _load_json,
    _overlap,
    _qualification_path,
    _rename_noreplace,
    _validate_declaration,
    _validate_implementation,
)
from tools.run_generator_cohort import canonical_bytes  # noqa: E402


MATERIALIZED_SUFFIXES = (*COMPILED_SUFFIXES, ".hook-materialization.json")
CLAIMS_SUFFIXES = (*MATERIALIZED_SUFFIXES, ".generator-claims.json")
MATERIALIZED_FILE_COUNT = 196
CLAIMS_FILE_COUNT = 224
DEFAULT_TIMEOUT_SECONDS = 900


class QualificationPostcompileError(RuntimeError):
    """A qualification postcompile stage failed without partial publication."""


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def stage_evidence_sha256(
    map_id: str,
    prior_stage_passed: bool,
    files: Mapping[str, Mapping[str, Any]],
) -> str:
    return _sha256(canonical_bytes({
        "map": map_id,
        "prior_stage_passed": prior_stage_passed,
        "files": dict(files),
    }))


def _exclusive_write(path: Path, payload: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except Exception:
        path.unlink(missing_ok=True)
        raise


def _fresh_outputs(inputs: Sequence[Path], outputs: Sequence[Path]) -> None:
    for path in outputs:
        if path.exists() or path.is_symlink():
            raise QualificationPostcompileError(f"output must be fresh: {path}")
        if not path.parent.is_dir():
            raise QualificationPostcompileError(
                f"output parent must exist: {path.parent}"
            )
    all_paths = [*inputs, *outputs]
    if any(
        _overlap(left, right)
        for index, left in enumerate(all_paths)
        for right in all_paths[index + 1:]
    ):
        raise QualificationPostcompileError("input/output paths must be disjoint")


def _validate_prior_stage(
    path: Path,
    expected_stage: str,
    declaration: Mapping[str, Any],
    declaration_sha256: str,
    implementation: Mapping[str, Any],
) -> tuple[dict[str, Any], bytes, str, set[str]]:
    report, raw = _load_json(path)
    expected_keys = {
        "schema", "qualification_id", "mode", "stage", "non_admissible",
        "retryable", "final_cohort_authorized", "declaration_sha256",
        "implementation", "input_report_sha256", "infrastructure_checks",
        "map_count", "pass_count", "maps", "failures",
    }
    if set(report) != expected_keys:
        raise QualificationPostcompileError("prior stage report keys differ")
    if (
        report["schema"] != STAGE_SCHEMA
        or report["qualification_id"] != declaration["qualification_id"]
        or report["mode"] != "qualification"
        or report["stage"] != expected_stage
        or report["non_admissible"] is not True
        or report["retryable"] is not True
        or report["final_cohort_authorized"] is not False
        or report["declaration_sha256"] != declaration_sha256
        or report["implementation"] != implementation
        or report["failures"] != []
        or not isinstance(report["infrastructure_checks"], Mapping)
        or not report["infrastructure_checks"]
        or not REQUIRED_STAGE_CHECKS[expected_stage].issubset(
            report["infrastructure_checks"]
        )
        or not all(
            value is True for value in report["infrastructure_checks"].values()
        )
    ):
        raise QualificationPostcompileError(
            f"{expected_stage} report identity/infrastructure differs"
        )
    rows = report["maps"]
    if (
        not isinstance(rows, list)
        or report["map_count"] != 28
        or len(rows) != 28
    ):
        raise QualificationPostcompileError("prior stage map count differs")
    passed: set[str] = set()
    for declared, row in zip(declaration["maps"], rows):
        if not isinstance(row, Mapping) or set(row) != {
            "ordinal", "map", "criteria", "evidence_sha256", "failures",
            "passed",
        }:
            raise QualificationPostcompileError("prior stage map row differs")
        recomputed = (
            isinstance(row["criteria"], Mapping)
            and bool(row["criteria"])
            and all(value is True for value in row["criteria"].values())
            and row["failures"] == []
        )
        if (
            row["ordinal"] != declared["ordinal"]
            or row["map"] != declared["map"]
            or row["passed"] is not recomputed
            or not isinstance(row["evidence_sha256"], str)
            or len(row["evidence_sha256"]) != 64
            or any(character not in "0123456789abcdef" for character in row["evidence_sha256"])
        ):
            raise QualificationPostcompileError("prior stage row binding differs")
        if recomputed:
            passed.add(str(row["map"]))
    if report["pass_count"] != len(passed):
        raise QualificationPostcompileError("prior stage pass count differs")
    return report, raw, _sha256(raw), passed


def _authority_records(
    *, repo_root: Path, cm_oracle: Path, pmove_oracle: Path, hook_oracle: Path,
    fall_oracle: Path, hook_attestation: Path,
) -> dict[str, Any]:
    try:
        gate = load_b1_authority_gate(repo_root)
        parity = admit_hook_parity_attestation(
            hook_attestation, repo_root=repo_root
        )
    except B1AuthorityError as error:
        raise QualificationPostcompileError(
            f"fresh B1 authority rejected: {error}"
        ) from error
    paths = {
        "cm": cm_oracle,
        "pmove": pmove_oracle,
        "hook": hook_oracle,
        "fall": fall_oracle,
        "hook_attestation": hook_attestation,
        "b1_gate": repo_root / "docs/multires/B1-GATE.json",
    }
    records = {name: _file_record(path) for name, path in paths.items()}
    expected = {
        "cm": gate.cm_executable_sha256,
        "pmove": gate.pmove_executable_sha256,
        "hook": parity.hook_executable_sha256,
        "fall": gate.fall_executable_sha256,
        "hook_attestation": parity.attestation_sha256,
    }
    if any(records[name]["sha256"] != digest for name, digest in expected.items()):
        raise QualificationPostcompileError("supplied authority bytes differ from fresh B1")
    return records


def _real_materialize_map(
    *, map_id: str, stage_root: Path, log_root: Path,
    compiled_files: Mapping[str, Mapping[str, Any]], authorities: Mapping[str, Any],
    cm_oracle: Path, pmove_oracle: Path, hook_oracle: Path, fall_oracle: Path,
    hook_attestation: Path, timeout_seconds: int,
) -> None:
    materializer = ROOT / "tools/materialize_hook_claims.py"
    attestation = stage_root / f"{map_id}.hook-materialization.json"
    runtime_sidecar = stage_root / f"{map_id}.json"
    command = [
        sys.executable, str(materializer), "--bsp", str(stage_root / f"{map_id}.bsp"),
        "--meta", str(stage_root / f"{map_id}.meta.json"),
        "--runtime-sidecar", str(runtime_sidecar), "--output-attestation",
        str(attestation), "--cm-oracle", str(cm_oracle), "--pmove-oracle",
        str(pmove_oracle), "--hook-oracle", str(hook_oracle), "--fall-oracle",
        str(fall_oracle), "--hook-parity-attestation", str(hook_attestation),
    ]
    try:
        completed = _run_process_group(
            command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            check=False, timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as error:
        raise QualificationPostcompileError(
            f"{map_id} materializer timed out after {timeout_seconds} seconds"
        ) from error
    stdout_path = log_root / f"{map_id}.stdout.json"
    stderr_path = log_root / f"{map_id}.stderr.log"
    _exclusive_write(stdout_path, completed.stdout)
    _exclusive_write(stderr_path, completed.stderr)
    if completed.returncode != 0:
        raise QualificationPostcompileError(
            f"{map_id} materializer exited {completed.returncode}"
        )
    authority_sha256 = {
        name: authorities[name]["sha256"]
        for name in ("cm", "pmove", "hook", "fall", "hook_attestation")
    }
    try:
        _validate_result(
            completed.stdout, map_id=map_id, attestation=attestation,
            runtime_sidecar=runtime_sidecar, bsp=stage_root / f"{map_id}.bsp",
            compiled_files=compiled_files,
            authority_sha256=authority_sha256,
        )
    except MaterializeCohortError as error:
        raise QualificationPostcompileError(str(error)) from error


def _stage_report(
    *, declaration: Mapping[str, Any], declaration_sha256: str,
    implementation: Mapping[str, Any], input_report_sha256: str, stage: str,
    prior_passed: set[str], records: Mapping[str, Mapping[str, Any]] | None,
    suffixes: Sequence[str], infrastructure: Mapping[str, bool],
    global_failures: Sequence[str], map_failures: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    rows = []
    for declared in declaration["maps"]:
        map_id = str(declared["map"])
        files = {} if records is None else records[map_id]
        prior = map_id in prior_passed
        criteria = {
            "prior-stage-passed": prior,
            (
                "real-authority-materialization"
                if stage == "materialization" else "immutable-claims"
            ): not map_failures.get(map_id) and records is not None,
            (
                "materialized-membership"
                if stage == "materialization" else "claims-membership"
            ): records is not None,
            "input-stability": infrastructure.get("input-stability") is True,
        }
        failures = list(map_failures.get(map_id, ()))
        if not prior:
            failures.append("prior stage did not pass this map")
        if records is None:
            failures.append(f"complete {stage} population was not published")
        evidence = stage_evidence_sha256(map_id, prior, files) if records else _sha256(
            canonical_bytes({"map": map_id, "stage": stage, "failures": failures})
        )
        passed = all(criteria.values()) and not failures
        rows.append({
            "ordinal": declared["ordinal"], "map": map_id,
            "criteria": criteria, "evidence_sha256": evidence,
            "failures": failures, "passed": passed,
        })
    return {
        "schema": STAGE_SCHEMA,
        "qualification_id": declaration["qualification_id"],
        "mode": "qualification",
        "stage": stage,
        "non_admissible": True,
        "retryable": True,
        "final_cohort_authorized": False,
        "declaration_sha256": declaration_sha256,
        "implementation": dict(implementation),
        "input_report_sha256": input_report_sha256,
        "infrastructure_checks": dict(infrastructure),
        "map_count": 28,
        "pass_count": sum(row["passed"] is True for row in rows),
        "maps": rows,
        "failures": list(global_failures),
    }


def _publish_stage_and_report(
    staging: Path, publication: Path, report_path: Path, report: Mapping[str, Any]
) -> None:
    _rename_noreplace(staging, publication)
    try:
        _exclusive_write(report_path, canonical_bytes(report))
    except Exception:
        _rename_noreplace(publication, staging)
        raise


def materialize_qualification(
    *, declaration_path: Path, prior_report_path: Path, compiled_root: Path,
    staging_root: Path, materialized_root: Path, log_root: Path,
    report_path: Path, cm_oracle: Path, pmove_oracle: Path, hook_oracle: Path,
    fall_oracle: Path, hook_attestation: Path,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS, repo_root: Path = ROOT,
    implementation_provider: Callable[[Path], dict[str, Any]] = _current_implementation,
    authority_provider: Callable[..., dict[str, Any]] = _authority_records,
    map_materializer: Callable[..., None] = _real_materialize_map,
) -> dict[str, Any]:
    paths = {
        name: _qualification_path(path, name)
        for name, path in {
            "declaration": declaration_path, "prior_report": prior_report_path,
            "compiled": compiled_root, "staging": staging_root,
            "materialized": materialized_root, "logs": log_root,
            "report": report_path, "cm": cm_oracle, "pmove": pmove_oracle,
            "hook": hook_oracle, "fall": fall_oracle,
            "hook_attestation": hook_attestation,
        }.items()
    }
    if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, int) or not 1 <= timeout_seconds <= 3600:
        raise QualificationPostcompileError("timeout must be an integer in [1, 3600]")
    _fresh_outputs(
        [
            paths["compiled"], paths["declaration"], paths["prior_report"],
            paths["cm"], paths["pmove"], paths["hook"], paths["fall"],
            paths["hook_attestation"],
        ],
        [paths["staging"], paths["materialized"], paths["logs"], paths["report"]],
    )
    if paths["staging"].parent.stat().st_dev != paths["materialized"].parent.stat().st_dev:
        raise QualificationPostcompileError(
            "materialization staging/publication must share a filesystem"
        )
    implementation = _validate_implementation(implementation_provider(repo_root))
    declaration, declaration_raw, declaration_sha256 = _validate_declaration(
        paths["declaration"], implementation
    )
    _, prior_raw, prior_sha256, prior_passed = _validate_prior_stage(
        paths["prior_report"], "compiled-cm-preflight", declaration,
        declaration_sha256, implementation,
    )
    compiled_records = _flat_records(declaration, paths["compiled"], COMPILED_SUFFIXES)
    authorities = authority_provider(
        repo_root=repo_root, cm_oracle=paths["cm"], pmove_oracle=paths["pmove"],
        hook_oracle=paths["hook"], fall_oracle=paths["fall"],
        hook_attestation=paths["hook_attestation"],
    )
    paths["staging"].mkdir()
    paths["logs"].mkdir()
    for declared in declaration["maps"]:
        for suffix in COMPILED_SUFFIXES:
            shutil.copyfile(
                paths["compiled"] / f"{declared['map']}{suffix}",
                paths["staging"] / f"{declared['map']}{suffix}",
            )
    map_failures: dict[str, list[str]] = {}
    for declared in declaration["maps"]:
        map_id = str(declared["map"])
        try:
            map_materializer(
                map_id=map_id, stage_root=paths["staging"],
                log_root=paths["logs"], compiled_files=compiled_records[map_id],
                authorities=authorities, cm_oracle=paths["cm"],
                pmove_oracle=paths["pmove"], hook_oracle=paths["hook"],
                fall_oracle=paths["fall"],
                hook_attestation=paths["hook_attestation"],
                timeout_seconds=timeout_seconds,
            )
        except Exception as error:
            map_failures[map_id] = [f"{type(error).__name__}: {error}"]
    records = None
    global_failures = []
    if map_failures:
        global_failures.extend(
            f"{map_id}: {message}"
            for map_id, failures in map_failures.items() for message in failures
        )
    else:
        try:
            records = _flat_records(
                declaration, paths["staging"], MATERIALIZED_SUFFIXES
            )
            for map_id in compiled_records:
                for suffix in COMPILED_SUFFIXES:
                    if suffix != ".json" and records[map_id][suffix] != compiled_records[map_id][suffix]:
                        raise QualificationPostcompileError(
                            f"materializer changed immutable {map_id}{suffix}"
                        )
        except (QualificationCompileError, QualificationPostcompileError) as error:
            global_failures.append(str(error))
            records = None
    try:
        stable = (
            paths["declaration"].read_bytes() == declaration_raw
            and paths["prior_report"].read_bytes() == prior_raw
            and _flat_records(declaration, paths["compiled"], COMPILED_SUFFIXES)
            == compiled_records
            and authority_provider(
                repo_root=repo_root, cm_oracle=paths["cm"],
                pmove_oracle=paths["pmove"], hook_oracle=paths["hook"],
                fall_oracle=paths["fall"],
                hook_attestation=paths["hook_attestation"],
            ) == authorities
            and _validate_implementation(implementation_provider(repo_root))
            == implementation
        )
    except Exception as error:
        stable = False
        global_failures.append(f"input stability: {error}")
    if not stable:
        records = None
        if not any("input stability" in failure for failure in global_failures):
            global_failures.append("input stability failed")
    published = records is not None and not global_failures
    infrastructure = {
        "authority-bound": not map_failures,
        "materialized-membership": published,
        "real-v4-materializer": not map_failures,
        "exact-membership": published,
        "input-stability": stable,
        "exclusive-publication": published,
    }
    report = _stage_report(
        declaration=declaration, declaration_sha256=declaration_sha256,
        implementation=implementation, input_report_sha256=prior_sha256,
        stage="materialization", prior_passed=prior_passed,
        records=records if published else None, suffixes=MATERIALIZED_SUFFIXES,
        infrastructure=infrastructure, global_failures=global_failures,
        map_failures=map_failures,
    )
    if published:
        _publish_stage_and_report(
            paths["staging"], paths["materialized"], paths["report"], report
        )
        return report
    _exclusive_write(paths["report"], canonical_bytes(report))
    raise QualificationPostcompileError("; ".join(global_failures))


def claims_qualification(
    *, declaration_path: Path, prior_report_path: Path,
    materialized_root: Path, staging_root: Path, claims_root: Path,
    report_path: Path, repo_root: Path = ROOT,
    implementation_provider: Callable[[Path], dict[str, Any]] = _current_implementation,
    claim_builder: Callable[[Path], Mapping[str, Any]] = build_generator_claims,
) -> dict[str, Any]:
    paths = {
        name: _qualification_path(path, name)
        for name, path in {
            "declaration": declaration_path, "prior_report": prior_report_path,
            "materialized": materialized_root, "staging": staging_root,
            "claims": claims_root, "report": report_path,
        }.items()
    }
    _fresh_outputs(
        [paths["materialized"], paths["declaration"], paths["prior_report"]],
        [paths["staging"], paths["claims"], paths["report"]],
    )
    if paths["staging"].parent.stat().st_dev != paths["claims"].parent.stat().st_dev:
        raise QualificationPostcompileError(
            "claims staging/publication must share a filesystem"
        )
    implementation = _validate_implementation(implementation_provider(repo_root))
    declaration, declaration_raw, declaration_sha256 = _validate_declaration(
        paths["declaration"], implementation
    )
    _, prior_raw, prior_sha256, prior_passed = _validate_prior_stage(
        paths["prior_report"], "materialization", declaration,
        declaration_sha256, implementation,
    )
    materialized_records = _flat_records(
        declaration, paths["materialized"], MATERIALIZED_SUFFIXES
    )
    paths["staging"].mkdir()
    for declared in declaration["maps"]:
        map_id = str(declared["map"])
        for suffix in MATERIALIZED_SUFFIXES:
            shutil.copyfile(
                paths["materialized"] / f"{map_id}{suffix}",
                paths["staging"] / f"{map_id}{suffix}",
            )
    map_failures: dict[str, list[str]] = {}
    for declared in declaration["maps"]:
        map_id = str(declared["map"])
        try:
            payload = canonical_bytes(claim_builder(paths["staging"] / f"{map_id}.map"))
            _exclusive_write(
                paths["staging"] / f"{map_id}.generator-claims.json", payload
            )
        except (ClaimValidationError, OSError, ValueError, KeyError, TypeError) as error:
            map_failures[map_id] = [f"{type(error).__name__}: {error}"]
    records = None
    global_failures = []
    if map_failures:
        global_failures.extend(
            f"{map_id}: {message}"
            for map_id, failures in map_failures.items() for message in failures
        )
    else:
        try:
            records = _flat_records(declaration, paths["staging"], CLAIMS_SUFFIXES)
            for map_id in materialized_records:
                for suffix in MATERIALIZED_SUFFIXES:
                    if records[map_id][suffix] != materialized_records[map_id][suffix]:
                        raise QualificationPostcompileError(
                            f"claims changed immutable {map_id}{suffix}"
                        )
        except (QualificationCompileError, QualificationPostcompileError) as error:
            global_failures.append(str(error))
            records = None
    try:
        stable = (
            paths["declaration"].read_bytes() == declaration_raw
            and paths["prior_report"].read_bytes() == prior_raw
            and _flat_records(
                declaration, paths["materialized"], MATERIALIZED_SUFFIXES
            ) == materialized_records
            and _validate_implementation(implementation_provider(repo_root))
            == implementation
        )
    except Exception as error:
        stable = False
        global_failures.append(f"input stability: {error}")
    if not stable:
        records = None
        if not any("input stability" in failure for failure in global_failures):
            global_failures.append("input stability failed")
    published = records is not None and not global_failures
    infrastructure = {
        "immutable-claims": not map_failures,
        "claims-membership": published,
        "exact-membership": published,
        "input-stability": stable,
        "exclusive-publication": published,
    }
    report = _stage_report(
        declaration=declaration, declaration_sha256=declaration_sha256,
        implementation=implementation, input_report_sha256=prior_sha256,
        stage="claims", prior_passed=prior_passed,
        records=records if published else None, suffixes=CLAIMS_SUFFIXES,
        infrastructure=infrastructure, global_failures=global_failures,
        map_failures=map_failures,
    )
    if published:
        _publish_stage_and_report(
            paths["staging"], paths["claims"], paths["report"], report
        )
        return report
    _exclusive_write(paths["report"], canonical_bytes(report))
    raise QualificationPostcompileError("; ".join(global_failures))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    materialize = subparsers.add_parser("materialize")
    materialize.add_argument("--declaration", type=Path, required=True)
    materialize.add_argument("--prior-report", type=Path, required=True)
    materialize.add_argument("--compiled-root", type=Path, required=True)
    materialize.add_argument("--staging-root", type=Path, required=True)
    materialize.add_argument("--materialized-root", type=Path, required=True)
    materialize.add_argument("--log-root", type=Path, required=True)
    materialize.add_argument("--report", type=Path, required=True)
    materialize.add_argument("--cm-oracle", type=Path, required=True)
    materialize.add_argument("--pmove-oracle", type=Path, required=True)
    materialize.add_argument("--hook-oracle", type=Path, required=True)
    materialize.add_argument("--fall-oracle", type=Path, required=True)
    materialize.add_argument("--hook-attestation", type=Path, required=True)
    materialize.add_argument("--timeout-seconds", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    claims = subparsers.add_parser("claims")
    claims.add_argument("--declaration", type=Path, required=True)
    claims.add_argument("--prior-report", type=Path, required=True)
    claims.add_argument("--materialized-root", type=Path, required=True)
    claims.add_argument("--staging-root", type=Path, required=True)
    claims.add_argument("--claims-root", type=Path, required=True)
    claims.add_argument("--report", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "materialize":
            report = materialize_qualification(
                declaration_path=args.declaration,
                prior_report_path=args.prior_report,
                compiled_root=args.compiled_root, staging_root=args.staging_root,
                materialized_root=args.materialized_root, log_root=args.log_root,
                report_path=args.report, cm_oracle=args.cm_oracle,
                pmove_oracle=args.pmove_oracle, hook_oracle=args.hook_oracle,
                fall_oracle=args.fall_oracle,
                hook_attestation=args.hook_attestation,
                timeout_seconds=args.timeout_seconds,
            )
        else:
            report = claims_qualification(
                declaration_path=args.declaration,
                prior_report_path=args.prior_report,
                materialized_root=args.materialized_root,
                staging_root=args.staging_root, claims_root=args.claims_root,
                report_path=args.report,
            )
    except (QualificationPostcompileError, QualificationCompileError) as error:
        print(f"B2 qualification {args.command} failed: {error}", file=sys.stderr)
        return 1
    sys.stdout.buffer.write(canonical_bytes(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
