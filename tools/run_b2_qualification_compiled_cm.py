#!/usr/bin/env python3
"""Run the real compiled-CM preflight as a disposable B2 qualification stage.

This adapter accepts only the qualification-native declaration and compile
report.  It challenges every compiled map with the existing real BSP/CM
preflight implementation, preserves the complete result for every map, and
emits the exact ``q2-b2-qualification-stage-v1`` shape consumed by
``assemble_b2_qualification.py``.  Its evidence is retryable and explicitly
incapable of authorizing a final cohort or admission.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import math
import os
from pathlib import Path
import re
import stat
import sys
from typing import Any, Callable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.atlas_analyzer import AnalyzerLimits  # noqa: E402
from harness.atlas_b1_authority import (  # noqa: E402
    B1AuthorityError,
    canonical_cm_physics_identity,
    load_b1_authority_gate,
)
from tools.assemble_b2_qualification import (  # noqa: E402
    B2QualificationError,
    EXPECTED_MAP_COUNT,
    STAGE_SCHEMA,
    _retired_identities,
    _validate_declaration,
    _validate_stage_report,
)
from tools.run_b2_qualification_compile import (  # noqa: E402
    QualificationCompileError,
    _load_json,
    _sparse_compiled_records,
)
from tools.run_compiled_cm_preflight import (  # noqa: E402
    ESCAPE_DISTANCE_UNITS,
    ESCAPE_STEP_UNITS,
    HAZARD_SAMPLE_FRACTIONS,
    MAX_JOBS,
    MILLIUNITS,
    MIN_COLUMN_UNITS,
    MIN_SPAWN_XY_SEPARATION_UNITS,
    SPAWN_COUNT,
    SPAWN_LINK_LIFT,
    _implementation_identity,
    _validate_map,
)
from tools.run_generator_cohort import (  # noqa: E402
    canonical_bytes,
    file_sha256,
    repository_binding,
)


MAP_EVIDENCE_SCHEMA = "q2-b2-qualification-compiled-cm-map-evidence-v1"
STAGE = "compiled-cm-preflight"
DEFAULT_JOBS = min(4, os.cpu_count() or 1)
DEFAULT_ORACLE_TIMEOUT_SECONDS = 10.0
HEX64 = re.compile(r"^[0-9a-f]{64}$")
EXPECTED_COMPILE_CRITERIA = {
    "source-stage-bound",
    "q2tool-exit-zero",
    "q2tool-not-timed-out",
    "ibsp38-lightdata",
    "compiled-stage-published",
}
COMPILE_SKIP_FAILURE = "compile stage did not publish a BSP for compiled-CM preflight"


class QualificationCompiledCmError(RuntimeError):
    """The qualification CM stage could not publish trustworthy evidence."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise QualificationCompiledCmError(message)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _file_record(path: Path, *, executable: bool = False) -> dict[str, Any]:
    try:
        mode = path.lstat().st_mode
    except OSError as error:
        raise QualificationCompiledCmError(f"cannot stat {path}: {error}") from error
    _require(
        stat.S_ISREG(mode) and not path.is_symlink(),
        f"required regular file is absent or a symlink: {path}",
    )
    if executable:
        _require(mode & 0o111 != 0, f"required executable is not executable: {path}")
    return {"bytes": path.stat().st_size, "sha256": file_sha256(path)}


def _absolute(path: Path) -> Path:
    return Path(os.path.abspath(path.expanduser()))


def _within(path: Path, root: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(root.resolve(strict=False))
    except ValueError:
        return False
    return True


def _qualification_path(path: Path, label: str) -> Path:
    result = _absolute(path)
    lowered = [part.lower() for part in result.parts]
    _require(
        not any("retired" in part or "generated-final" in part for part in lowered),
        f"{label} is under a retired/final artifact path",
    )
    return result


def _exclusive_write(path: Path, payload: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        parent = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(parent)
        finally:
            os.close(parent)
    except Exception:
        path.unlink(missing_ok=True)
        raise


def _compile_report(
    path: Path,
    declaration: Mapping[str, Any],
    declaration_sha256: str,
    implementation: Mapping[str, Any],
    retired: Mapping[str, set[Any]],
) -> tuple[dict[str, Any], bytes, str, set[str]]:
    try:
        report, raw = _load_json(path)
    except QualificationCompileError as error:
        raise QualificationCompiledCmError(
            f"qualification compile report rejected: {error}"
        ) from error
    input_digest = report.get("input_report_sha256")
    _require(
        isinstance(input_digest, str) and HEX64.fullmatch(input_digest) is not None,
        "qualification compile report has no source-stage hash binding",
    )
    try:
        summary, digest, passed = _validate_stage_report(
            path,
            "compile",
            declaration,
            declaration_sha256,
            implementation,
            input_digest,
            retired,
        )
    except B2QualificationError as error:
        raise QualificationCompiledCmError(
            f"qualification compile report rejected: {error}"
        ) from error
    _require(summary["map_count"] == EXPECTED_MAP_COUNT,
             "qualification compile report map count differs")
    for declared, row in zip(declaration["maps"], report["maps"]):
        criteria = row.get("criteria")
        _require(
            isinstance(criteria, Mapping)
            and set(criteria) == EXPECTED_COMPILE_CRITERIA
            and all(isinstance(value, bool) for value in criteria.values())
            and row.get("ordinal") == declared["ordinal"]
            and row.get("map") == declared["map"]
            and row.get("passed")
            is (str(declared["map"]) in passed),
            f"qualification compile map evidence differs for {declared['map']}",
        )
    return report, raw, digest, passed


def _spawn_invariants(value: object) -> bool:
    if not isinstance(value, Mapping):
        return False
    escape = value.get("basic_escape")
    return bool(
        value.get("standing_clear") is True
        and value.get("crouched_clear") is True
        and value.get("supported") is True
        and value.get("engine_link_lift_milliunits")
        == round(SPAWN_LINK_LIFT * MILLIUNITS)
        and isinstance(value.get("column_clearance_milliunits"), int)
        and value["column_clearance_milliunits"]
        >= MIN_COLUMN_UNITS * MILLIUNITS
        and value.get("column_clear_96") is True
        and isinstance(escape, Mapping)
        and escape.get("distance_milliunits")
        == ESCAPE_DISTANCE_UNITS * MILLIUNITS
        and escape.get("support_step_milliunits")
        == ESCAPE_STEP_UNITS * MILLIUNITS
        and isinstance(escape.get("passing_direction_indices"), list)
        and bool(escape["passing_direction_indices"])
        and escape.get("passed") is True
        and value.get("failures") == []
        and value.get("passed") is True
    )


def _hazard_invariants(value: object) -> bool:
    if not isinstance(value, Mapping):
        return False
    hazards = value.get("hazards")
    if not isinstance(hazards, list):
        return False
    return bool(
        isinstance(value.get("declared_hazard_count"), int)
        and value.get("declared_hazard_count") == len(hazards)
        and value.get("checked_hazard_count") == len(hazards)
        and value.get("failures") == []
        and value.get("passed") is True
        and all(
            isinstance(hazard, Mapping)
            and hazard.get("probe_count") == len(HAZARD_SAMPLE_FRACTIONS)
            and hazard.get("failures") == []
            and hazard.get("passed") is True
            for hazard in hazards
        )
    )


def _map_criteria(
    declared: Mapping[str, Any],
    compile_row: Mapping[str, Any],
    compiled_files: Mapping[str, Any],
    result: Mapping[str, Any],
    cm_record: Mapping[str, Any],
    gate: Any,
) -> dict[str, bool]:
    bsp = compiled_files.get(".bsp")
    oracle = result.get("oracle")
    real_cm = bool(
        isinstance(bsp, Mapping)
        and isinstance(oracle, Mapping)
        and result.get("ordinal") == declared["ordinal"]
        and result.get("map") == declared["map"]
        and result.get("bsp") == bsp
        and oracle.get("executable_sha256") == cm_record["sha256"]
        and oracle.get("tool_identity") == gate.oracle_tool_identity
        and oracle.get("map_sha256") == bsp.get("sha256")
        and oracle.get("physics_identity")
        == canonical_cm_physics_identity(
            gate.oracle_tool_identity, bsp.get("sha256", "")
        )
    )
    lightdata = result.get("compiled_lightdata")
    source_spawns = result.get("source_spawn_origins_milliunits")
    compiled_spawns = result.get("compiled_spawn_origins_milliunits")
    spawns = result.get("spawns")
    compiled_invariants = bool(
        isinstance(lightdata, Mapping)
        and isinstance(lightdata.get("bytes"), int)
        and lightdata["bytes"] > 0
        and isinstance(lightdata.get("sha256"), str)
        and HEX64.fullmatch(lightdata["sha256"]) is not None
        and lightdata.get("present") is True
        and isinstance(source_spawns, list)
        and isinstance(compiled_spawns, list)
        and len(source_spawns) == len(compiled_spawns) == SPAWN_COUNT
        and source_spawns == compiled_spawns
        and result.get("spawn_origin_sets_match") is True
        and isinstance(result.get("minimum_spawn_xy_separation_milliunits"), int)
        and result["minimum_spawn_xy_separation_milliunits"]
        >= MIN_SPAWN_XY_SEPARATION_UNITS * MILLIUNITS
        and result.get("spawn_count") == SPAWN_COUNT
        and isinstance(spawns, list)
        and len(spawns) == SPAWN_COUNT
        and all(_spawn_invariants(spawn) for spawn in spawns)
        and _hazard_invariants(result.get("basic_hazard_containment"))
        and result.get("failures") == []
        and result.get("passed") is True
    )
    return {
        "compile-stage-bound": bool(
            compile_row.get("ordinal") == declared["ordinal"]
            and compile_row.get("map") == declared["map"]
            and compile_row.get("passed") is True
            and compile_row.get("failures") == []
        ),
        "real-bsp-cm": real_cm,
        "compiled-invariants": compiled_invariants,
    }


def _run_maps(
    declaration: Mapping[str, Any],
    compiled_root: Path,
    cm_oracle: Path,
    gate: Any,
    jobs: int,
    timeout_seconds: float,
    map_validator: Callable[..., dict[str, Any]],
    eligible_maps: set[str],
) -> dict[str, dict[str, Any]]:
    limits = AnalyzerLimits(oracle_batch_timeout_seconds=timeout_seconds)
    by_map: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=jobs) as executor:
        futures = {
            executor.submit(
                map_validator,
                row,
                compiled_root,
                cm_oracle,
                limits,
                file_sha256(cm_oracle),
                gate.oracle_tool_identity,
                gate.oracle_source_closure_sha256,
            ): row
            for row in declaration["maps"]
            if str(row["map"]) in eligible_maps
        }
        for future in as_completed(futures):
            declared = futures[future]
            map_id = str(declared["map"])
            try:
                value = future.result()
                if not isinstance(value, Mapping):
                    raise TypeError("map validator result is not an object")
                by_map[map_id] = dict(value)
            except Exception as error:  # defensive executor boundary
                by_map[map_id] = {
                    "ordinal": declared["ordinal"],
                    "map": map_id,
                    "failures": [
                        f"executor {type(error).__name__}: {error}"
                    ],
                    "passed": False,
                }
    _require(
        set(by_map) == eligible_maps,
        "compiled-CM executor result membership differs from compile pass set",
    )
    return by_map


def _compile_skipped_result(declared: Mapping[str, Any]) -> dict[str, Any]:
    """Return the one replayable result permitted for a compile-failed map."""

    return {
        "ordinal": declared["ordinal"],
        "map": declared["map"],
        "bsp": None,
        "compiled_lightdata": None,
        "source_spawn_origins_milliunits": None,
        "compiled_spawn_origins_milliunits": None,
        "spawn_origin_sets_match": False,
        "minimum_spawn_xy_separation_milliunits": None,
        "oracle": None,
        "spawn_count": 0,
        "spawns": [],
        "basic_hazard_containment": None,
        "failures": [COMPILE_SKIP_FAILURE],
        "passed": False,
    }


def run_qualification_compiled_cm(
    *,
    declaration_path: Path,
    compile_report_path: Path,
    compiled_root: Path,
    cm_oracle: Path,
    evidence_root: Path,
    report_path: Path,
    jobs: int = DEFAULT_JOBS,
    oracle_batch_timeout_seconds: float = DEFAULT_ORACLE_TIMEOUT_SECONDS,
    repo_root: Path = ROOT,
    implementation_provider: Callable[[Path], dict[str, Any]] = repository_binding,
    gate_loader: Callable[[Path], Any] = load_b1_authority_gate,
    preflight_implementation_provider: Callable[[Path], dict[str, Any]] = _implementation_identity,
    map_validator: Callable[..., dict[str, Any]] = _validate_map,
) -> dict[str, Any]:
    """Run and publish one complete qualification-native compiled-CM stage."""

    paths = {
        name: _qualification_path(path, name)
        for name, path in {
            "declaration": declaration_path,
            "compile_report": compile_report_path,
            "compiled_root": compiled_root,
            "cm_oracle": cm_oracle,
            "evidence_root": evidence_root,
            "report": report_path,
        }.items()
    }
    _require(
        isinstance(jobs, int) and not isinstance(jobs, bool) and 1 <= jobs <= MAX_JOBS,
        f"jobs must be in [1, {MAX_JOBS}]",
    )
    _require(
        isinstance(oracle_batch_timeout_seconds, (int, float))
        and not isinstance(oracle_batch_timeout_seconds, bool)
        and math.isfinite(oracle_batch_timeout_seconds)
        and 0 < oracle_batch_timeout_seconds <= 60,
        "oracle batch timeout must be finite and in (0, 60]",
    )
    timeout_seconds = float(oracle_batch_timeout_seconds)
    _require(paths["compiled_root"].is_dir(), "compiled root is absent")
    _require(not paths["compiled_root"].is_symlink(), "compiled root is a symlink")
    _file_record(paths["declaration"])
    _file_record(paths["compile_report"])
    cm_record = _file_record(paths["cm_oracle"], executable=True)
    for output in (paths["evidence_root"], paths["report"]):
        _require(
            not output.exists() and not output.is_symlink(),
            f"output must be fresh: {output}",
        )
        _require(output.parent.is_dir(), f"output parent is absent: {output.parent}")
    _require(
        not _within(paths["evidence_root"], paths["compiled_root"])
        and not _within(paths["report"], paths["compiled_root"])
        and not _within(paths["report"], paths["evidence_root"]),
        "evidence/report outputs overlap an input or each other",
    )

    try:
        initial_implementation = implementation_provider(repo_root)
        retired = _retired_identities(repo_root)
        declaration, declaration_sha256 = _validate_declaration(
            paths["declaration"], initial_implementation, retired
        )
        compile_report, compile_raw, compile_sha256, compile_passed = _compile_report(
            paths["compile_report"], declaration, declaration_sha256,
            initial_implementation, retired,
        )
        compiled_records = _sparse_compiled_records(
            declaration, paths["compiled_root"], compile_passed
        )
        gate = gate_loader(repo_root)
        preflight_implementation = preflight_implementation_provider(repo_root)
    except (
        B1AuthorityError,
        B2QualificationError,
        QualificationCompileError,
    ) as error:
        raise QualificationCompiledCmError(str(error)) from error
    _require(
        cm_record["sha256"] == gate.cm_executable_sha256,
        "CM oracle executable bytes differ from fresh B1 authority",
    )
    gate_path = repo_root / "docs/multires/B1-GATE.json"
    gate_record = _file_record(gate_path)
    results = _run_maps(
        declaration, paths["compiled_root"], paths["cm_oracle"], gate,
        jobs, timeout_seconds, map_validator, compile_passed,
    )

    try:
        stable = (
            paths["declaration"].read_bytes()
            == canonical_bytes(declaration)
            and paths["compile_report"].read_bytes() == compile_raw
            and _sparse_compiled_records(
                declaration, paths["compiled_root"], compile_passed
            ) == compiled_records
            and _file_record(paths["cm_oracle"], executable=True) == cm_record
            and _file_record(gate_path) == gate_record
            and gate_loader(repo_root) == gate
            and implementation_provider(repo_root) == initial_implementation
            and preflight_implementation_provider(repo_root)
            == preflight_implementation
        )
    except (
        B1AuthorityError,
        B2QualificationError,
        QualificationCompileError,
        OSError,
    ) as error:
        raise QualificationCompiledCmError(
            f"final input stability check failed: {error}"
        ) from error
    _require(stable, "declaration/compile/BSP/B1/implementation input changed during CM preflight")

    evidence_payloads: list[tuple[str, bytes]] = []
    stage_rows = []
    b1_identity = {
        "gate": gate_record,
        "cm_executable_sha256": gate.cm_executable_sha256,
        "cm_tool_identity": gate.oracle_tool_identity,
        "cm_source_closure_sha256": gate.oracle_source_closure_sha256,
    }
    for declared, compile_row in zip(
        declaration["maps"], compile_report["maps"]
    ):
        map_id = str(declared["map"])
        result = (
            results[map_id]
            if map_id in compile_passed
            else _compile_skipped_result(declared)
        )
        criteria = _map_criteria(
            declared, compile_row, compiled_records[map_id], result,
            cm_record, gate,
        )
        failures = list(result.get("failures", [])) if isinstance(
            result.get("failures"), list
        ) else ["real compiled-CM preflight failures are malformed"]
        if not all(criteria.values()) and not failures:
            failures.append(
                "real compiled-CM result is incomplete or internally inconsistent"
            )
        failures = sorted(set(
            str(failure) for failure in failures if str(failure)
        ))
        passed = all(criteria.values()) and not failures
        evidence = {
            "schema": MAP_EVIDENCE_SCHEMA,
            "qualification_id": declaration["qualification_id"],
            "ordinal": declared["ordinal"],
            "map": map_id,
            "compile_report_sha256": compile_sha256,
            "compile_map_evidence_sha256": compile_row["evidence_sha256"],
            "compiled_files": compiled_records[map_id],
            "b1_authority": b1_identity,
            "preflight_implementation": preflight_implementation,
            "real_preflight_result": result,
            "criteria": criteria,
            "failures": failures,
            "passed": passed,
        }
        payload = canonical_bytes(evidence)
        evidence_payloads.append(
            (f"{int(declared['ordinal']):03d}-{map_id}.evidence.json", payload)
        )
        stage_rows.append({
            "ordinal": declared["ordinal"],
            "map": map_id,
            "criteria": criteria,
            "evidence_sha256": _sha256(payload),
            "failures": failures,
            "passed": passed,
        })

    paths["evidence_root"].mkdir(mode=0o755)
    for name, payload in evidence_payloads:
        _exclusive_write(paths["evidence_root"] / name, payload)
    evidence_parent = os.open(
        paths["evidence_root"].parent, os.O_RDONLY | os.O_DIRECTORY
    )
    try:
        os.fsync(evidence_parent)
    finally:
        os.close(evidence_parent)

    report = {
        "schema": STAGE_SCHEMA,
        "qualification_id": declaration["qualification_id"],
        "mode": "qualification",
        "stage": STAGE,
        "non_admissible": True,
        "retryable": True,
        "final_cohort_authorized": False,
        "declaration_sha256": declaration_sha256,
        "implementation": initial_implementation,
        "toolchain_authority_sha256": declaration[
            "toolchain_authority_sha256"
        ],
        "input_report_sha256": compile_sha256,
        "infrastructure_checks": {
            "real-bsp-cm": True,
            "compiled-invariants": True,
            "exact-membership": True,
            "fresh-b1-authority": True,
            "bounded-parallel-workers": True,
            "input-stability": True,
            "exclusive-evidence-publication": True,
        },
        "map_count": EXPECTED_MAP_COUNT,
        "pass_count": sum(row["passed"] is True for row in stage_rows),
        "maps": stage_rows,
        "failures": [],
    }
    _exclusive_write(paths["report"], canonical_bytes(report))
    return report


def validate_published_qualification_compiled_cm(
    *,
    declaration_path: Path,
    compile_report_path: Path,
    compiled_root: Path,
    cm_oracle: Path,
    evidence_root: Path,
    report_path: Path,
    repo_root: Path = ROOT,
    implementation_provider: Callable[[Path], dict[str, Any]] = repository_binding,
    gate_loader: Callable[[Path], Any] = load_b1_authority_gate,
    preflight_implementation_provider: Callable[[Path], dict[str, Any]] = _implementation_identity,
) -> tuple[dict[str, Any], bytes, str, set[str]]:
    """Replay a published compiled-CM stage from its raw evidence.

    This is the consumer-side trust boundary.  It deliberately does not trust
    the stage report's booleans or opaque evidence digests: every evidence
    document is canonical-loaded, rebound to the current compiled population
    and B1 authority, and its criteria/result are independently recomputed.
    """

    paths = {
        name: _qualification_path(path, name)
        for name, path in {
            "declaration": declaration_path,
            "compile_report": compile_report_path,
            "compiled_root": compiled_root,
            "cm_oracle": cm_oracle,
            "evidence_root": evidence_root,
            "report": report_path,
        }.items()
    }
    try:
        implementation = implementation_provider(repo_root)
        retired = _retired_identities(repo_root)
        declaration, declaration_sha256 = _validate_declaration(
            paths["declaration"], implementation, retired
        )
        compile_report, compile_raw, compile_sha256, compile_passed = _compile_report(
            paths["compile_report"], declaration, declaration_sha256,
            implementation, retired,
        )
        compiled_records = _sparse_compiled_records(
            declaration, paths["compiled_root"], compile_passed
        )
        report, report_raw = _load_json(paths["report"])
        _validate_stage_report(
            paths["report"], STAGE, declaration, declaration_sha256,
            implementation, compile_sha256, retired,
        )
        gate = gate_loader(repo_root)
        preflight_implementation = preflight_implementation_provider(repo_root)
        cm_record = _file_record(paths["cm_oracle"], executable=True)
        gate_path = repo_root / "docs/multires/B1-GATE.json"
        gate_record = _file_record(gate_path)
    except (
        B1AuthorityError,
        B2QualificationError,
        QualificationCompileError,
        OSError,
    ) as error:
        raise QualificationCompiledCmError(
            f"published compiled-CM stage rejected: {error}"
        ) from error
    _require(
        cm_record["sha256"] == gate.cm_executable_sha256,
        "published compiled-CM oracle differs from fresh B1 authority",
    )
    expected_names = {
        f"{int(row['ordinal']):03d}-{row['map']}.evidence.json"
        for row in declaration["maps"]
    }
    _require(
        paths["evidence_root"].is_dir()
        and not paths["evidence_root"].is_symlink(),
        "published compiled-CM evidence root is absent or a symlink",
    )
    actual_names: set[str] = set()
    evidence_raw: dict[str, bytes] = {}
    for path in paths["evidence_root"].iterdir():
        _require(
            path.is_file() and not path.is_symlink(),
            f"compiled-CM evidence contains non-regular entry: {path.name}",
        )
        actual_names.add(path.name)
    _require(
        actual_names == expected_names,
        "compiled-CM evidence membership differs; "
        f"missing={sorted(expected_names - actual_names)}, "
        f"extra={sorted(actual_names - expected_names)}",
    )

    b1_identity = {
        "gate": gate_record,
        "cm_executable_sha256": gate.cm_executable_sha256,
        "cm_tool_identity": gate.oracle_tool_identity,
        "cm_source_closure_sha256": gate.oracle_source_closure_sha256,
    }
    expected_evidence_keys = {
        "schema", "qualification_id", "ordinal", "map",
        "compile_report_sha256", "compile_map_evidence_sha256",
        "compiled_files", "b1_authority", "preflight_implementation",
        "real_preflight_result", "criteria", "failures", "passed",
    }
    passed: set[str] = set()
    for declared, compile_row, report_row in zip(
        declaration["maps"], compile_report["maps"], report["maps"]
    ):
        map_id = str(declared["map"])
        name = f"{int(declared['ordinal']):03d}-{map_id}.evidence.json"
        try:
            evidence, raw = _load_json(paths["evidence_root"] / name)
        except QualificationCompileError as error:
            raise QualificationCompiledCmError(
                f"compiled-CM evidence rejected for {map_id}: {error}"
            ) from error
        evidence_raw[name] = raw
        _require(
            set(evidence) == expected_evidence_keys,
            f"compiled-CM evidence keys differ for {map_id}",
        )
        result = evidence["real_preflight_result"]
        _require(
            isinstance(result, Mapping),
            f"compiled-CM real result is not an object for {map_id}",
        )
        if map_id not in compile_passed:
            _require(
                result == _compile_skipped_result(declared),
                f"compile-skipped compiled-CM result differs for {map_id}",
            )
        criteria = _map_criteria(
            declared, compile_row, compiled_records[map_id], result,
            cm_record, gate,
        )
        failures = list(result.get("failures", [])) if isinstance(
            result.get("failures"), list
        ) else ["real compiled-CM preflight failures are malformed"]
        if not all(criteria.values()) and not failures:
            failures.append(
                "real compiled-CM result is incomplete or internally inconsistent"
            )
        failures = sorted(set(
            str(failure) for failure in failures if str(failure)
        ))
        is_passed = all(criteria.values()) and not failures
        expected_evidence = {
            "schema": MAP_EVIDENCE_SCHEMA,
            "qualification_id": declaration["qualification_id"],
            "ordinal": declared["ordinal"],
            "map": map_id,
            "compile_report_sha256": compile_sha256,
            "compile_map_evidence_sha256": compile_row["evidence_sha256"],
            "compiled_files": compiled_records[map_id],
            "b1_authority": b1_identity,
            "preflight_implementation": preflight_implementation,
            "real_preflight_result": dict(result),
            "criteria": criteria,
            "failures": failures,
            "passed": is_passed,
        }
        _require(
            raw == canonical_bytes(expected_evidence),
            f"compiled-CM evidence replay differs for {map_id}",
        )
        expected_row = {
            "ordinal": declared["ordinal"],
            "map": map_id,
            "criteria": criteria,
            "evidence_sha256": _sha256(raw),
            "failures": failures,
            "passed": is_passed,
        }
        _require(
            report_row == expected_row,
            f"compiled-CM report row differs from raw evidence for {map_id}",
        )
        if is_passed:
            passed.add(map_id)
    _require(
        report["pass_count"] == len(passed),
        "compiled-CM report pass count differs from raw evidence",
    )
    try:
        stable = (
            paths["declaration"].read_bytes() == canonical_bytes(declaration)
            and paths["compile_report"].read_bytes() == compile_raw
            and paths["report"].read_bytes() == report_raw
            and {
                name: (paths["evidence_root"] / name).read_bytes()
                for name in expected_names
            } == evidence_raw
            and _sparse_compiled_records(
                declaration, paths["compiled_root"], compile_passed
            ) == compiled_records
            and _file_record(paths["cm_oracle"], executable=True) == cm_record
            and _file_record(gate_path) == gate_record
            and gate_loader(repo_root) == gate
            and implementation_provider(repo_root) == implementation
            and preflight_implementation_provider(repo_root)
            == preflight_implementation
        )
    except Exception as error:
        raise QualificationCompiledCmError(
            f"compiled-CM replay stability check failed: {error}"
        ) from error
    _require(stable, "compiled-CM replay inputs changed during validation")
    return report, report_raw, _sha256(report_raw), passed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--declaration", type=Path, required=True)
    parser.add_argument("--compile-report", type=Path, required=True)
    parser.add_argument("--compiled-root", type=Path, required=True)
    parser.add_argument("--cm-oracle", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=DEFAULT_JOBS)
    parser.add_argument(
        "--oracle-batch-timeout-seconds",
        type=float,
        default=DEFAULT_ORACLE_TIMEOUT_SECONDS,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = run_qualification_compiled_cm(
            declaration_path=args.declaration,
            compile_report_path=args.compile_report,
            compiled_root=args.compiled_root,
            cm_oracle=args.cm_oracle,
            evidence_root=args.evidence_root,
            report_path=args.report,
            jobs=args.jobs,
            oracle_batch_timeout_seconds=args.oracle_batch_timeout_seconds,
        )
    except QualificationCompiledCmError as error:
        print(f"B2 qualification compiled-CM failed: {error}", file=sys.stderr)
        return 1
    sys.stdout.buffer.write(canonical_bytes({
        "schema": STAGE_SCHEMA,
        "qualification_id": report["qualification_id"],
        "stage": STAGE,
        "non_admissible": True,
        "final_cohort_authorized": False,
        "report": str(_absolute(args.report)),
        "report_sha256": file_sha256(args.report),
        "map_count": report["map_count"],
        "pass_count": report["pass_count"],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
