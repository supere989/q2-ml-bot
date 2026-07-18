#!/usr/bin/env python3
"""Assemble disposable B2 toolchain-qualification evidence.

Qualification is deliberately separate from final-cohort admission.  This
assembler accepts only qualification-native declarations and reports, checks
their exact hash chain and current implementation binding, independently
recomputes every report/map disposition, and creates one canonical report
outside the repository with exclusive-create semantics.

The existing final-cohort producers do not yet emit the schemas consumed here.
They must gain an explicit qualification mode; relabelling a final report or
wrapping a self-declared success boolean is intentionally insufficient.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.atlas_b1_authority import (  # noqa: E402
    B1AuthorityError,
    canonical_cm_physics_identity,
    load_b1_authority_gate,
)
from tools.b2_qualification_toolchain import (  # noqa: E402
    ToolchainAuthorityError,
    load_toolchain_authority,
)
from tools.run_b2_compiled_boundary_qualification import (  # noqa: E402
    CASES as BOUNDARY_CASES,
    PROOF_SCHEMA as BOUNDARY_PROOF_SCHEMA,
    QualificationError as BoundaryQualificationError,
    load_compile_report as load_boundary_compile_report,
    qualification_only as boundary_qualification_only,
)
from tools.run_generator_cohort import (  # noqa: E402
    CONCRETE_STYLES,
    GeneratorCohortError,
    canonical_bytes,
    repository_binding,
)
from tools.retired_cohort_registry import (  # noqa: E402
    RetiredCohortRegistryError,
    _load_registry,
)


QUALIFICATION_SCHEMA = "q2-b2-toolchain-qualification-v1"
DECLARATION_SCHEMA = "q2-b2-qualification-declaration-v1"
STAGE_SCHEMA = "q2-b2-qualification-stage-v1"
INFRASTRUCTURE_SCHEMA = "q2-b2-qualification-infrastructure-v2"
EXPECTED_MAP_COUNT = 28
REQUIRED_END_TO_END_PASSES = 20
HEX64 = re.compile(r"^[0-9a-f]{64}$")
HEX40 = re.compile(r"^[0-9a-f]{40}$")
TOKEN = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
QUALIFICATION_ID = re.compile(r"^b2q26_[a-z0-9][a-z0-9_-]*$")
STAGES = (
    "source",
    "compile",
    "compiled-cm-preflight",
    "materialization",
    "claims",
    "atlas-build",
    "generated-promotion",
)
REQUIRED_STAGE_CHECKS = {
    "source": {"source-static", "deterministic-cold-rebuild"},
    "compile": {"real-q2tool", "compiled-membership"},
    "compiled-cm-preflight": {"real-bsp-cm", "compiled-invariants"},
    "materialization": {"authority-bound", "materialized-membership"},
    "claims": {"immutable-claims", "claims-membership"},
    "atlas-build": {"full-atlas", "deterministic-cold-rebuild"},
    "generated-promotion": {"independent-promotion-validation"},
}
REQUIRED_INFRASTRUCTURE_CHECKS = {
    "deterministic-cold-rebuild",
    "exact-stage-membership",
    "exclusive-create",
    "python310-syntax-floor",
    "resource-bounds",
    "timeout-fail-closed",
}


class B2QualificationError(ValueError):
    """Any input which cannot prove qualification fails closed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise B2QualificationError(message)


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise B2QualificationError(f"{label} must be an object")
    return value


def _array(value: object, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise B2QualificationError(f"{label} must be an array")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        raise B2QualificationError(
            f"{label} keys differ; missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _integer(value: object, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise B2QualificationError(f"{label} must be an integer >= {minimum}")
    return value


def _digest(value: object, label: str) -> str:
    if not isinstance(value, str) or HEX64.fullmatch(value) is None:
        raise B2QualificationError(f"{label} must be a lowercase SHA-256")
    return value


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise B2QualificationError(f"cannot hash {path}: {error}") from error
    return digest.hexdigest()


def _file_record(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise B2QualificationError(
            f"required regular file is missing or a symlink: {path}"
        )
    return {"bytes": path.stat().st_size, "sha256": _file_sha256(path)}


def _reject_duplicates(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise B2QualificationError(f"duplicate JSON key {key!r}")
        output[key] = value
    return output


def _load_json(path: Path, *, canonical: bool = True) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
        value = json.loads(
            raw,
            object_pairs_hook=_reject_duplicates,
            parse_constant=lambda token: (_ for _ in ()).throw(
                B2QualificationError(f"non-finite JSON token {token}")
            ),
        )
    except B2QualificationError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise B2QualificationError(f"cannot read JSON {path}: {error}") from error
    document = dict(_mapping(value, str(path)))
    if canonical and raw != canonical_bytes(document):
        raise B2QualificationError(f"JSON is not canonical: {path}")
    return document, raw


def _exclusive_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        parent = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(parent)
        finally:
            os.close(parent)
    except Exception:
        path.unlink(missing_ok=True)
        raise


def _require_qualification_path(path: Path, label: str) -> None:
    lowered = [part.lower() for part in path.resolve().parts]
    if any("retired" in part or "generated-final" in part for part in lowered):
        raise B2QualificationError(f"{label} is under a retired/final artifact path")


def _walk_digests(value: object) -> set[str]:
    output: set[str] = set()
    if isinstance(value, Mapping):
        for item in value.values():
            output.update(_walk_digests(item))
    elif isinstance(value, list):
        for item in value:
            output.update(_walk_digests(item))
    elif isinstance(value, str) and HEX64.fullmatch(value):
        output.add(value)
    return output


def _retired_identities(repo_root: Path) -> dict[str, set[Any]]:
    """Validate the registry, then expose identities qualification cannot reuse."""

    if repo_root.resolve() != ROOT.resolve():
        raise B2QualificationError("repo root must be the repository containing this tool")
    cohorts, declarations, maps, seeds = _load_registry()
    digests: set[str] = set(declarations)
    for path in sorted((repo_root / "docs/multires").glob(
        "B2-GENERATED-COHORT-*-FAILURE.json"
    )):
        value, _ = _load_json(path)
        digests.update(_walk_digests(value))
    return {
        "cohorts": set(cohorts),
        "declarations": set(declarations),
        "maps": set(maps),
        "seeds": set(seeds),
        "digests": digests,
    }


def _validate_declaration(
    path: Path,
    implementation: Mapping[str, Any],
    retired: Mapping[str, set[Any]],
) -> tuple[dict[str, Any], str]:
    _require_qualification_path(path, "qualification declaration")
    declaration, raw = _load_json(path)
    _exact_keys(
        declaration,
        {
            "schema", "qualification_id", "mode", "non_admissible",
            "retryable", "final_cohort_authorized", "generator",
            "selection", "implementation", "toolchain_authority_sha256",
            "maps",
        },
        "qualification declaration",
    )
    _require(declaration["schema"] == DECLARATION_SCHEMA, "qualification declaration schema differs")
    qualification_id = declaration["qualification_id"]
    _require(
        isinstance(qualification_id, str)
        and QUALIFICATION_ID.fullmatch(qualification_id) is not None
        and "final" not in qualification_id,
        "qualification ID is invalid or final-mode",
    )
    _require(qualification_id not in retired["cohorts"], "retired cohort ID reused")
    _require(declaration["mode"] == "qualification", "final-mode declaration rejected")
    _require(declaration["non_admissible"] is True, "qualification must be non-admissible")
    _require(declaration["retryable"] is True, "qualification must be retryable")
    _require(declaration["final_cohort_authorized"] is False, "qualification cannot authorize a final cohort")
    _require(declaration["implementation"] == implementation, "declaration implementation binding differs")
    try:
        authority = load_toolchain_authority(ROOT)
    except ToolchainAuthorityError as error:
        raise B2QualificationError(
            f"canonical toolchain authority rejected: {error}"
        ) from error
    _require(
        declaration["toolchain_authority_sha256"]
        == authority.manifest_sha256,
        "qualification declaration toolchain authority differs",
    )

    generator = _mapping(declaration["generator"], "qualification generator")
    _exact_keys(generator, {"version", "grid", "gym", "observed_heat"}, "qualification generator")
    _require(
        generator == {"version": "v6", "grid": 5, "gym": False, "observed_heat": None},
        "qualification generator contract differs",
    )
    selection = _mapping(declaration["selection"], "qualification selection")
    _exact_keys(
        selection,
        {"required_map_count", "required_concrete_styles", "required_maps_per_style"},
        "qualification selection",
    )
    _require(selection["required_map_count"] == EXPECTED_MAP_COUNT, "qualification must declare 28 maps")
    _require(selection["required_concrete_styles"] == list(CONCRETE_STYLES), "qualification styles differ")
    _require(selection["required_maps_per_style"] == 4, "qualification must declare four maps per style")

    maps = _array(declaration["maps"], "qualification maps")
    _require(len(maps) == EXPECTED_MAP_COUNT, "qualification map count differs")
    names: set[str] = set()
    seeds: set[int] = set()
    styles: Counter[str] = Counter()
    for expected_ordinal, item in enumerate(maps):
        row = _mapping(item, f"qualification map {expected_ordinal}")
        _exact_keys(row, {"ordinal", "map", "seed", "style", "grid", "observed_heat"}, f"qualification map {expected_ordinal}")
        _require(row["ordinal"] == expected_ordinal, "qualification map ordinals differ")
        name = row["map"]
        _require(
            isinstance(name, str) and TOKEN.fullmatch(name) is not None
            and name.startswith("b2q26_") and "final" not in name,
            f"qualification map {expected_ordinal} ID is invalid",
        )
        _require(name not in names, f"duplicate qualification map {name}")
        _require(name not in retired["maps"], f"retired map ID reused: {name}")
        names.add(name)
        seed = _integer(row["seed"], f"qualification map {name} seed")
        _require(seed not in seeds, f"duplicate qualification seed {seed}")
        _require(seed not in retired["seeds"], f"retired map seed reused: {seed}")
        seeds.add(seed)
        style = row["style"]
        _require(style in CONCRETE_STYLES, f"qualification map {name} style differs")
        styles[str(style)] += 1
        _require(row["grid"] == 5 and row["observed_heat"] is None, f"qualification map {name} generator inputs differ")
    _require(styles == Counter({style: 4 for style in CONCRETE_STYLES}), "qualification style balance differs")
    digest = _sha256_bytes(raw)
    _require(digest not in retired["declarations"], "retired declaration digest reused")
    return declaration, digest


def _validate_requalification(
    b1_gate_path: Path,
    repo_root: Path,
    normative: Mapping[str, Any],
) -> tuple[dict[str, Any], Any, dict[str, str]]:
    expected_path = repo_root / "docs/multires/B1-GATE.json"
    _require(
        _file_record(b1_gate_path) == _file_record(expected_path),
        "supplied B1 gate bytes differ from the repository trust root",
    )
    try:
        authority = load_b1_authority_gate(repo_root)
    except B1AuthorityError as error:
        raise B2QualificationError(f"fresh B1 authority rejected: {error}") from error
    gate, _ = _load_json(b1_gate_path, canonical=False)
    requalification = _mapping(gate.get("authority_requalification"), "B1 authority requalification")
    _exact_keys(
        requalification,
        {
            "schema", "status", "recorded_at", "historical_gate_sha256",
            "historical_normative_documents", "current_normative_documents",
            "probe_bsp_sha256", "probe_runtime_authority_seal", "repository",
            "inputs", "live_identities", "checks", "failures",
        },
        "B1 authority requalification",
    )
    _require(requalification["schema"] == "q2-b1-authority-requalification-v1", "B1 requalification schema differs")
    _require(requalification["status"] == "green", "B1 requalification is not green")
    _digest(requalification["historical_gate_sha256"], "historical B1 gate digest")
    _digest(requalification["probe_bsp_sha256"], "B1 probe BSP digest")
    runtime_seal = _mapping(
        requalification["probe_runtime_authority_seal"],
        "B1 probe runtime authority seal",
    )
    _require(
        runtime_seal.get("schema") == "q2-b1-runtime-authority-seal-v1",
        "B1 probe runtime authority seal schema differs",
    )
    current = _mapping(requalification["current_normative_documents"], "B1 current normative documents")
    _require(
        current == {
            "design_sha256": normative["design"]["sha256"],
            "plan_sha256": normative["plan"]["sha256"],
        },
        "B1 requalification does not bind supplied amended documents",
    )
    _require(
        authority.design_sha256 == current["design_sha256"]
        and authority.plan_sha256 == current["plan_sha256"],
        "B1 authority loader and requalification document disagree",
    )
    _require(
        runtime_seal.get("normative_documents") == dict(current),
        "B1 probe runtime authority seal does not bind amended documents",
    )
    repository = _mapping(requalification["repository"], "B1 requalification repository")
    _require(
        set(repository) == {"commit", "tree", "clean"}
        and isinstance(repository.get("commit"), str)
        and HEX40.fullmatch(repository["commit"]) is not None
        and isinstance(repository.get("tree"), str)
        and HEX40.fullmatch(repository["tree"]) is not None
        and repository.get("clean") is True,
        "B1 requalification repository binding is malformed",
    )
    checks = _mapping(requalification["checks"], "B1 requalification checks")
    _require(bool(checks) and all(value is True for value in checks.values()), "not every B1 requalification check is green")
    _require(requalification["failures"] == [], "B1 requalification retains failures")
    live = _mapping(requalification["live_identities"], "B1 live identities")
    collision = _mapping(live.get("collision"), "B1 live collision identity")
    collision_identity = {
        "tool_identity": _digest(
            collision.get("tool_identity"), "B1 live collision tool identity"
        ),
        "physics_identity": _digest(
            collision.get("physics_identity"),
            "B1 live collision physics identity",
        ),
    }
    _require(
        collision_identity["tool_identity"] == authority.oracle_tool_identity,
        "B1 live collision tool identity differs from its sealed authority",
    )
    return {
        "gate": _file_record(b1_gate_path),
        "requalification_sha256": _sha256_bytes(canonical_bytes(requalification)),
        "runtime_authority_seal_sha256": _sha256_bytes(
            canonical_bytes(runtime_seal)
        ),
        "reseal_repository": dict(repository),
        "collision_identity": collision_identity,
    }, authority, collision_identity


def _validate_boundary_proof(
    path: Path,
    authority: Any,
    collision_identity: Mapping[str, str],
    retired: Mapping[str, set[Any]],
) -> dict[str, Any]:
    try:
        toolchain = load_toolchain_authority(ROOT)
    except ToolchainAuthorityError as error:
        raise B2QualificationError(
            f"canonical toolchain authority rejected: {error}"
        ) from error
    _require_qualification_path(path, "compiled-boundary proof")
    proof, raw = _load_json(path)
    _require(proof.get("schema") == BOUNDARY_PROOF_SCHEMA, "compiled-boundary proof report is missing; compile-only evidence rejected")
    _require(proof.get("status") == "passed-non-admissible-qualification", "compiled-boundary proof status differs")
    _require(proof.get("passed") is True, "compiled-boundary proof is not passed")
    _require(proof.get("admission") == boundary_qualification_only(), "compiled-boundary proof is not qualification-only")
    _require(
        proof.get("toolchain_authority") == toolchain.manifest_record(),
        "compiled-boundary toolchain authority differs",
    )
    contract = _mapping(proof.get("contract"), "compiled-boundary contract")
    _require(
        contract.get("engine_link_lift_units") == 9
        and contract.get("column_requirement_units") == 96,
        "compiled-boundary spawn contract differs",
    )
    cm = _mapping(proof.get("cm_oracle"), "compiled-boundary CM oracle")
    _require(cm.get("sha256") == authority.cm_executable_sha256, "compiled-boundary CM bytes differ from fresh B1")
    compile_record = _mapping(proof.get("compile_evidence"), "compiled-boundary compile evidence")
    compile_path_raw = compile_record.get("path")
    _require(isinstance(compile_path_raw, str) and compile_path_raw, "compiled-boundary compile evidence path is missing")
    compile_path = Path(compile_path_raw)
    _require_qualification_path(compile_path, "compiled-boundary compile evidence")
    actual_compile_record = {
        "path": str(compile_path.absolute()),
        "sha256": _file_sha256(compile_path),
        "size_bytes": compile_path.stat().st_size,
    }
    _require(dict(compile_record) == actual_compile_record, "compiled-boundary compile report/hash mismatch")
    try:
        compile_report, loaded_record = load_boundary_compile_report(compile_path)
    except BoundaryQualificationError as error:
        raise B2QualificationError(f"compiled-boundary compile evidence rejected: {error}") from error
    _require(loaded_record == actual_compile_record, "compiled-boundary loader record mismatch")
    _require(proof.get("q2tool") == compile_report.get("q2tool"), "compiled-boundary q2tool binding differs")
    _require(
        compile_report.get("toolchain_authority")
        == toolchain.manifest_record(),
        "compiled-boundary compile toolchain authority differs",
    )
    _require(
        _mapping(compile_report.get("q2tool"), "compiled-boundary q2tool").get(
            "sha256"
        ) == toolchain.q2tool_sha256,
        "compiled-boundary q2tool bytes differ from canonical authority",
    )
    _require(
        compile_report.get("fixed_q2tool_flags")
        == list(toolchain.q2tool_flags),
        "compiled-boundary q2tool flags differ from canonical authority",
    )
    compile_basedir = _mapping(
        compile_report.get("basedir"), "compiled-boundary baseq2 assets"
    )
    _require(
        _mapping(compile_basedir.get("pak0"), "compiled-boundary pak0").get(
            "sha256"
        ) == toolchain.pak0_sha256
        and _mapping(
            compile_basedir.get("required_member"),
            "compiled-boundary colormap",
        ).get("sha256") == toolchain.colormap_sha256,
        "compiled-boundary baseq2 assets differ from canonical authority",
    )
    reported_fixtures = {
        row.get("case_id"): row
        for raw in _array(
            compile_report.get("fixtures"), "compiled-boundary compile fixtures"
        )
        for row in [_mapping(raw, "compiled-boundary compile fixture")]
    }
    _require(
        set(reported_fixtures)
        == {str(item["case_id"]) for item in toolchain.fixtures},
        "compiled-boundary fixture authority membership differs",
    )
    for fixture in toolchain.fixtures:
        case_id = str(fixture["case_id"])
        reported = reported_fixtures[case_id]
        geometry = _mapping(
            reported.get("geometry"), f"{case_id} authored geometry"
        )
        _require(
            _mapping(reported.get("source"), f"{case_id} source").get(
                "sha256"
            ) == fixture["sha256"]
            and geometry.get("floor_top_units")
            == fixture["floor_top_units"]
            and geometry.get("ceiling_bottom_units")
            == fixture["ceiling_bottom_units"]
            and geometry.get("spawn_origin_units")
            == fixture["spawn_origin_units"],
            f"{case_id} fixture bytes/geometry differ from canonical authority",
        )

    expected = {case["case_id"]: case for case in BOUNDARY_CASES}
    rows = _array(proof.get("proofs"), "compiled-boundary proofs")
    _require(len(rows) == len(expected), "compiled-boundary proof count differs")
    seen: set[str] = set()
    for item in rows:
        row = _mapping(item, "compiled-boundary proof row")
        case_id = row.get("case_id")
        _require(case_id in expected and case_id not in seen, "compiled-boundary case membership differs")
        seen.add(str(case_id))
        case = expected[str(case_id)]
        _require(row.get("authored_floor_to_ceiling_units") == case["ceiling_units"], f"{case_id} ceiling differs")
        _require(row.get("expected_pass") is case["expected_pass"], f"{case_id} expected disposition differs")
        _require(row.get("column_clear_96") is case["expected_pass"], f"{case_id} measured disposition differs")
        _require(row.get("column_clearance_milliunits") == case["expected_clearance_units"] * 1000, f"{case_id} measured clearance differs")
        _require(row.get("linked_standing_clear") is True, f"{case_id} linked spawn is not clear")
        identity = _mapping(row.get("cm_identity"), f"{case_id} CM identity")
        _require(identity.get("tool_identity") == collision_identity["tool_identity"], f"{case_id} CM tool identity differs from fresh B1")
        bsp_sha256 = _mapping(row.get("bsp"), f"{case_id} BSP").get("sha256")
        _digest(bsp_sha256, f"{case_id} BSP digest")
        expected_physics_identity = canonical_cm_physics_identity(
            collision_identity["tool_identity"], bsp_sha256
        )
        _require(identity.get("physics_identity") == expected_physics_identity, f"{case_id} CM physics identity is not canonical for its BSP")
        _require(identity.get("map_sha256") == bsp_sha256, f"{case_id} CM/BSP binding differs")
    _require(seen == set(expected), "compiled-boundary proof membership differs")
    digest = _sha256_bytes(raw)
    _require(digest not in retired["digests"], "retired compiled-boundary proof reused")
    return {
        "report": _file_record(path),
        "compile_report": {"bytes": actual_compile_record["size_bytes"], "sha256": actual_compile_record["sha256"]},
        "cases": [
            {"case_id": case["case_id"], "ceiling_units": case["ceiling_units"], "passed_requirement": case["expected_pass"]}
            for case in BOUNDARY_CASES
        ],
    }


def _validate_stage_report(
    path: Path,
    expected_stage: str,
    declaration: Mapping[str, Any],
    declaration_sha256: str,
    implementation: Mapping[str, Any],
    expected_input_sha256: str | None,
    retired: Mapping[str, set[Any]],
) -> tuple[dict[str, Any], str, set[str]]:
    _require_qualification_path(path, f"{expected_stage} report")
    report, raw = _load_json(path)
    _exact_keys(
        report,
        {
            "schema", "qualification_id", "mode", "stage",
            "non_admissible", "retryable", "final_cohort_authorized",
            "declaration_sha256", "implementation",
            "toolchain_authority_sha256", "input_report_sha256",
            "infrastructure_checks", "map_count", "pass_count", "maps",
            "failures",
        },
        f"{expected_stage} report",
    )
    _require(report["schema"] == STAGE_SCHEMA, f"{expected_stage} report schema differs")
    _require(report["qualification_id"] == declaration["qualification_id"], f"{expected_stage} qualification ID differs")
    _require(report["mode"] == "qualification", f"{expected_stage} final-mode report rejected")
    _require(report["stage"] == expected_stage, f"{expected_stage} stage identity differs")
    _require(report["non_admissible"] is True and report["retryable"] is True, f"{expected_stage} qualification disposition differs")
    _require(report["final_cohort_authorized"] is False, f"{expected_stage} report authorizes a final cohort")
    _require(report["declaration_sha256"] == declaration_sha256, f"{expected_stage} declaration binding differs")
    _require(report["implementation"] == implementation, f"{expected_stage} implementation binding differs")
    _require(
        report["toolchain_authority_sha256"]
        == declaration["toolchain_authority_sha256"],
        f"{expected_stage} toolchain authority binding differs",
    )
    _require(report["input_report_sha256"] == expected_input_sha256, f"{expected_stage} input report hash chain differs")
    infrastructure = _mapping(report["infrastructure_checks"], f"{expected_stage} infrastructure checks")
    required = REQUIRED_STAGE_CHECKS[expected_stage]
    _require(required.issubset(infrastructure), f"{expected_stage} infrastructure checks are incomplete")
    _require(all(value is True for value in infrastructure.values()), f"{expected_stage} infrastructure check failed")
    _require(report["failures"] == [], f"{expected_stage} report retains infrastructure failures")

    rows = _array(report["maps"], f"{expected_stage} maps")
    _require(report["map_count"] == len(rows) == EXPECTED_MAP_COUNT, f"{expected_stage} map count differs")
    passed_maps: set[str] = set()
    evidence_digests: set[str] = set()
    for declared, item in zip(declaration["maps"], rows):
        row = _mapping(item, f"{expected_stage} map row")
        _exact_keys(row, {"ordinal", "map", "criteria", "evidence_sha256", "failures", "passed"}, f"{expected_stage} map row")
        _require(row["ordinal"] == declared["ordinal"] and row["map"] == declared["map"], f"{expected_stage} map membership/order differs")
        criteria = _mapping(row["criteria"], f"{expected_stage} {row['map']} criteria")
        _require(bool(criteria) and all(isinstance(key, str) and TOKEN.fullmatch(key) for key in criteria), f"{expected_stage} {row['map']} criteria are malformed")
        failures = _array(row["failures"], f"{expected_stage} {row['map']} failures")
        _require(all(isinstance(item, str) and item for item in failures), f"{expected_stage} {row['map']} failures are malformed")
        recomputed = all(value is True for value in criteria.values()) and not failures
        _require(row["passed"] is recomputed, f"{expected_stage} {row['map']} self-declared result differs from criteria")
        evidence = _digest(row["evidence_sha256"], f"{expected_stage} {row['map']} evidence")
        _require(evidence not in retired["digests"], f"{expected_stage} {row['map']} reuses retired/final artifact evidence")
        _require(evidence not in evidence_digests, f"{expected_stage} reuses one per-map evidence digest")
        evidence_digests.add(evidence)
        if recomputed:
            passed_maps.add(str(row["map"]))
    _require(report["pass_count"] == len(passed_maps), f"{expected_stage} pass count was not recomputed correctly")
    digest = _sha256_bytes(raw)
    _require(digest not in retired["digests"], f"retired {expected_stage} report reused")
    return {
        "report": {"bytes": len(raw), "sha256": digest},
        "map_count": len(rows),
        "pass_count": len(passed_maps),
    }, digest, passed_maps


def _validate_infrastructure(
    path: Path,
    declaration: Mapping[str, Any],
    declaration_sha256: str,
    implementation: Mapping[str, Any],
    stage_sha256s: Mapping[str, str],
    retired: Mapping[str, set[Any]],
) -> dict[str, Any]:
    _require_qualification_path(path, "qualification infrastructure report")
    report, raw = _load_json(path)
    _exact_keys(
        report,
        {
            "schema", "qualification_id", "mode", "non_admissible",
            "retryable", "final_cohort_authorized", "declaration_sha256",
            "implementation", "toolchain_authority_sha256",
            "stage_report_sha256s", "checks",
            "pass_count", "failures",
        },
        "qualification infrastructure report",
    )
    _require(report["schema"] == INFRASTRUCTURE_SCHEMA, "qualification infrastructure schema differs")
    _require(report["qualification_id"] == declaration["qualification_id"], "infrastructure qualification ID differs")
    _require(report["mode"] == "qualification", "final-mode infrastructure report rejected")
    _require(report["non_admissible"] is True and report["retryable"] is True, "infrastructure qualification disposition differs")
    _require(report["final_cohort_authorized"] is False, "infrastructure report authorizes final cohort")
    _require(report["declaration_sha256"] == declaration_sha256, "infrastructure declaration binding differs")
    _require(report["implementation"] == implementation, "infrastructure implementation binding differs")
    _require(
        report["toolchain_authority_sha256"]
        == declaration["toolchain_authority_sha256"],
        "infrastructure toolchain authority binding differs",
    )
    _require(report["stage_report_sha256s"] == dict(stage_sha256s), "infrastructure stage hash binding differs")
    checks = _array(report["checks"], "qualification infrastructure checks")
    by_id: dict[str, Mapping[str, Any]] = {}
    for item in checks:
        check = _mapping(item, "qualification infrastructure check")
        _exact_keys(check, {"id", "criteria", "evidence_sha256", "failures", "passed"}, "qualification infrastructure check")
        check_id = check["id"]
        _require(isinstance(check_id, str) and TOKEN.fullmatch(check_id), "infrastructure check ID is malformed")
        _require(check_id not in by_id, f"duplicate infrastructure check {check_id}")
        criteria = _mapping(check["criteria"], f"infrastructure {check_id} criteria")
        failures = _array(check["failures"], f"infrastructure {check_id} failures")
        _require(bool(criteria) and all(value is True for value in criteria.values()), f"infrastructure {check_id} criterion failed")
        _require(failures == [], f"infrastructure {check_id} retains failures")
        _require(check["passed"] is True, f"infrastructure {check_id} self-declared result differs")
        evidence = _digest(check["evidence_sha256"], f"infrastructure {check_id} evidence")
        _require(evidence not in retired["digests"], f"infrastructure {check_id} reuses retired evidence")
        by_id[str(check_id)] = check
    _require(REQUIRED_INFRASTRUCTURE_CHECKS.issubset(by_id), "required infrastructure gates are missing")
    _require(report["pass_count"] == len(checks), "infrastructure pass count differs")
    _require(report["failures"] == [], "qualification infrastructure retains failures")
    digest = _sha256_bytes(raw)
    _require(digest not in retired["digests"], "retired infrastructure report reused")
    return {"report": {"bytes": len(raw), "sha256": digest}, "checks": sorted(by_id), "pass_count": len(checks)}


def _normative_documents(design: Path, plan: Path) -> dict[str, Any]:
    return {
        "design": _file_record(design),
        "plan": _file_record(plan),
    }


def _raw_evidence_path(path: Path) -> dict[str, Any]:
    absolute = path.absolute()
    return {"path": str(absolute), "record": _file_record(absolute)}


def _stage_evidence_paths(args: argparse.Namespace) -> dict[str, str] | None:
    names = (
        "source_root", "source_cold_root", "compiled_root",
        "compile_evidence_root", "q2tool", "basedir",
        "compiled_cm_evidence_root", "cm_oracle", "pmove_oracle",
        "hook_oracle", "fall_oracle", "hook_attestation", "python_runtime",
        "materialized_root", "materialization_log_root", "claims_root",
        "analysis_root", "atlas_evidence_root",
        "promotion_evidence_root", "infrastructure_evidence_root",
        "syntax_report",
    )
    values = [getattr(args, name, None) for name in names]
    if all(value is None for value in values):
        return None
    _require(all(isinstance(value, Path) for value in values),
             "all qualification stage evidence roots are required")
    result = {name: str(value.absolute()) for name, value in zip(names, values)}
    for name, value in result.items():
        _require_qualification_path(Path(value), name)
    return result


def _validate_retained_stage_evidence(
    declaration: Mapping[str, Any], stage_documents: Mapping[str, Mapping[str, Any]],
    infrastructure_document: Mapping[str, Any], roots: Mapping[str, str],
    args: argparse.Namespace, repo_root: Path,
    replay_implementation: Mapping[str, Any] | None = None,
) -> None:
    try:
        from tools.run_b2_qualification_source import (
            validate_published_qualification_source,
        )
        from tools.run_b2_qualification_compile import (
            validate_published_qualification_compile,
        )
        from tools.run_b2_qualification_compiled_cm import (
            validate_published_qualification_compiled_cm,
        )
        from tools.run_b2_qualification_postcompile import (
            validate_published_qualification_postcompile,
        )
        from tools.run_b2_qualification_promotion import (
            _validate_atlas_artifacts,
            validate_promotion_evidence,
        )
        from tools.run_b2_qualification_infrastructure import (
            validate_infrastructure_evidence,
        )
        implementation_provider = (
            (lambda _root: dict(replay_implementation))
            if replay_implementation is not None
            else repository_binding
        )
        _, _, _, source_passed = validate_published_qualification_source(
            declaration_path=args.declaration,
            source_root=Path(roots["source_root"]),
            cold_root=Path(roots["source_cold_root"]),
            report_path=args.source_report, repo_root=repo_root,
            implementation_provider=implementation_provider,
        )
        _, _, _, compile_passed = validate_published_qualification_compile(
            declaration_path=args.declaration,
            source_report_path=args.source_report,
            source_root=Path(roots["source_root"]),
            compiled_root=Path(roots["compiled_root"]),
            evidence_root=Path(roots["compile_evidence_root"]),
            report_path=args.compile_report, q2tool=Path(roots["q2tool"]),
            basedir=Path(roots["basedir"]), repo_root=repo_root,
            implementation_provider=implementation_provider,
        )
        _, _, _, cm_passed = validate_published_qualification_compiled_cm(
            declaration_path=args.declaration,
            compile_report_path=args.compile_report,
            compiled_root=Path(roots["compiled_root"]),
            cm_oracle=Path(roots["cm_oracle"]),
            evidence_root=Path(roots["compiled_cm_evidence_root"]),
            report_path=args.compiled_cm_preflight_report,
            repo_root=repo_root,
            implementation_provider=implementation_provider,
        )
        postcompile = validate_published_qualification_postcompile(
            declaration_path=args.declaration,
            compile_report_path=args.compile_report,
            compiled_root=Path(roots["compiled_root"]),
            compiled_cm_report_path=args.compiled_cm_preflight_report,
            compiled_cm_evidence_root=Path(roots["compiled_cm_evidence_root"]),
            materialization_report_path=args.materialization_report,
            materialized_root=Path(roots["materialized_root"]),
            materialization_log_root=Path(roots["materialization_log_root"]),
            claims_report_path=args.claims_report,
            claims_root=Path(roots["claims_root"]),
            cm_oracle=Path(roots["cm_oracle"]),
            pmove_oracle=Path(roots["pmove_oracle"]),
            hook_oracle=Path(roots["hook_oracle"]),
            fall_oracle=Path(roots["fall_oracle"]),
            hook_attestation=Path(roots["hook_attestation"]),
            python_runtime=Path(roots["python_runtime"]), repo_root=repo_root,
            implementation_provider=implementation_provider,
            compiled_cm_validator=(
                lambda **kwargs: validate_published_qualification_compiled_cm(
                    **kwargs,
                    implementation_provider=implementation_provider,
                )
            ),
        )
        expected_passes = {
            stage: {
                str(row["map"]) for row in stage_documents[stage]["maps"]
                if row["passed"] is True
            }
            for stage in STAGES
        }
        _require(source_passed == expected_passes["source"],
                 "source replay pass set differs")
        _require(compile_passed == expected_passes["compile"],
                 "compile replay pass set differs")
        _require(cm_passed == expected_passes["compiled-cm-preflight"],
                 "compiled-CM replay pass set differs")
        _require(set(postcompile["materialization_passed"]) == expected_passes["materialization"],
                 "materialization replay pass set differs")
        _require(set(postcompile["claims_passed"]) == expected_passes["claims"],
                 "claims replay pass set differs")
        _validate_atlas_artifacts(
            declaration, stage_documents["atlas-build"],
            Path(roots["analysis_root"]), Path(roots["atlas_evidence_root"]),
        )
        validate_promotion_evidence(
            declaration, stage_documents["generated-promotion"],
            Path(roots["claims_root"]), Path(roots["analysis_root"]),
            Path(roots["promotion_evidence_root"]),
        )
        validate_infrastructure_evidence(
            infrastructure_document, Path(roots["infrastructure_evidence_root"]),
            declaration=declaration, stage_reports=stage_documents,
            roots={
                "source": Path(roots["source_root"]),
                "compile": Path(roots["compiled_root"]),
                "materialization": Path(roots["materialized_root"]),
                "claims": Path(roots["claims_root"]),
                "atlas-build": Path(roots["analysis_root"]),
                "generated-promotion": Path(roots["promotion_evidence_root"]),
            },
            syntax_report=Path(roots["syntax_report"]),
        )
    except (ValueError, RuntimeError, OSError, KeyError, TypeError) as error:
        raise B2QualificationError(
            f"retained qualification stage evidence rejected: {error}"
        ) from error


def assemble_qualification(
    args: argparse.Namespace,
    *,
    replay_implementation: Mapping[str, Any] | None = None,
    replay_raw_evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    repo_root = args.repo_root.resolve()
    _require(repo_root == ROOT.resolve(), "repo root must be the repository containing this tool")
    normative = _normative_documents(args.design, args.plan)
    implementation = (
        dict(replay_implementation)
        if replay_implementation is not None
        else repository_binding(repo_root)
    )
    try:
        toolchain_authority = load_toolchain_authority(repo_root)
    except ToolchainAuthorityError as error:
        raise B2QualificationError(
            f"canonical toolchain authority rejected: {error}"
        ) from error
    retired = _retired_identities(repo_root)
    declaration, declaration_sha256 = _validate_declaration(
        args.declaration, implementation, retired
    )
    b1, authority, collision_identity = _validate_requalification(
        args.b1_gate, repo_root, normative
    )
    boundary = _validate_boundary_proof(
        args.boundary_proof_report, authority, collision_identity, retired
    )

    paths = {
        "source": args.source_report,
        "compile": args.compile_report,
        "compiled-cm-preflight": args.compiled_cm_preflight_report,
        "materialization": args.materialization_report,
        "claims": args.claims_report,
        "atlas-build": args.atlas_build_report,
        "generated-promotion": args.generated_promotion_report,
    }
    stage_summaries: dict[str, Any] = {}
    stage_documents: dict[str, Mapping[str, Any]] = {}
    stage_sha256s: dict[str, str] = {}
    stage_passes: dict[str, set[str]] = {}
    previous: str | None = None
    for stage in STAGES:
        summary, digest, passed = _validate_stage_report(
            paths[stage], stage, declaration, declaration_sha256,
            implementation, previous, retired,
        )
        stage_summaries[stage] = summary
        stage_documents[stage] = _load_json(paths[stage])[0]
        stage_sha256s[stage] = digest
        stage_passes[stage] = passed
        previous = digest
    for earlier, later in zip(STAGES, STAGES[1:]):
        _require(
            stage_passes[later].issubset(stage_passes[earlier]),
            f"{later} passes maps which did not pass {earlier}",
        )
    end_to_end = set.intersection(*(stage_passes[stage] for stage in STAGES))
    _require(
        end_to_end == stage_passes["generated-promotion"],
        "promotion pass set is not the exact end-to-end lifecycle pass set",
    )
    _require(
        len(end_to_end) >= REQUIRED_END_TO_END_PASSES,
        f"qualification has {len(end_to_end)}/28 end-to-end passes; 20 required",
    )
    infrastructure = _validate_infrastructure(
        args.infrastructure_report, declaration, declaration_sha256,
        implementation, stage_sha256s, retired,
    )
    infrastructure_document = _load_json(args.infrastructure_report)[0]
    retained_stage_evidence = _stage_evidence_paths(args)
    if retained_stage_evidence is not None:
        _validate_retained_stage_evidence(
            declaration, stage_documents, infrastructure_document,
            retained_stage_evidence,
            args, repo_root, replay_implementation,
        )
    raw_evidence = {
        "normative_documents": {
            "design": _raw_evidence_path(args.design),
            "plan": _raw_evidence_path(args.plan),
        },
        "b1_gate": _raw_evidence_path(args.b1_gate),
        "compiled_boundary_report": _raw_evidence_path(
            args.boundary_proof_report
        ),
        "declaration": _raw_evidence_path(args.declaration),
        "stage_reports": {
            stage: _raw_evidence_path(paths[stage]) for stage in STAGES
        },
        "infrastructure_report": _raw_evidence_path(
            args.infrastructure_report
        ),
        "stage_evidence": retained_stage_evidence,
    }
    if replay_raw_evidence is not None:
        _require(
            replay_implementation is not None,
            "raw-evidence replay requires a reported implementation",
        )
        raw_evidence = dict(replay_raw_evidence)
    return {
        "schema": QUALIFICATION_SCHEMA,
        "status": "green",
        "qualification_id": declaration["qualification_id"],
        "non_admissible": True,
        "retryable": True,
        "final_cohort_authorized": False,
        "normative_documents": normative,
        "implementation": implementation,
        "toolchain_authority": toolchain_authority.manifest_record(),
        "b1_authority": b1,
        "compiled_boundary": boundary,
        "declaration": {
            "bytes": args.declaration.stat().st_size,
            "sha256": declaration_sha256,
            "map_count": EXPECTED_MAP_COUNT,
            "mode": "qualification",
        },
        "stages": stage_summaries,
        "infrastructure": infrastructure,
        "end_to_end": {
            "required_pass_count": REQUIRED_END_TO_END_PASSES,
            "pass_count": len(end_to_end),
            "failed_count": EXPECTED_MAP_COUNT - len(end_to_end),
            "passed_maps": sorted(end_to_end),
        },
        "authorization": {
            "final_declaration_allowed_by_this_report": False,
            "qualification_artifact_reuse_as_final_evidence": False,
            "passing_subset_admissible": False,
        },
        "raw_evidence": raw_evidence,
        "failures": [],
    }


def validate_qualification(value: object) -> dict[str, Any]:
    report = _mapping(value, "B2 qualification")
    _exact_keys(
        report,
        {
            "schema", "status", "qualification_id", "non_admissible",
            "retryable", "final_cohort_authorized", "normative_documents",
            "implementation", "b1_authority", "compiled_boundary",
            "toolchain_authority", "declaration", "stages",
            "infrastructure", "end_to_end",
            "authorization", "raw_evidence", "failures",
        },
        "B2 qualification",
    )
    _require(report.get("schema") == QUALIFICATION_SCHEMA, "B2 qualification schema differs")
    _require(report.get("status") == "green", "B2 qualification is not green")
    _require(
        isinstance(report.get("qualification_id"), str)
        and QUALIFICATION_ID.fullmatch(report["qualification_id"]) is not None,
        "B2 qualification ID is malformed",
    )
    _require(report.get("non_admissible") is True, "B2 qualification is not explicitly non-admissible")
    _require(report.get("retryable") is True, "B2 qualification is not explicitly retryable")
    _require(report.get("final_cohort_authorized") is False, "B2 qualification improperly authorizes a final cohort")
    normative = _mapping(report["normative_documents"], "B2 qualification normative documents")
    _exact_keys(normative, {"design", "plan"}, "B2 qualification normative documents")
    for name, raw in normative.items():
        record = _mapping(raw, f"B2 qualification {name} record")
        _exact_keys(record, {"bytes", "sha256"}, f"B2 qualification {name} record")
        _integer(record["bytes"], f"B2 qualification {name} bytes", 1)
        _digest(record["sha256"], f"B2 qualification {name} digest")
    implementation = _mapping(report["implementation"], "B2 qualification implementation")
    _require(implementation.get("git_clean") is True, "B2 qualification implementation was not clean")
    _require(
        isinstance(implementation.get("repository_commit"), str)
        and HEX40.fullmatch(implementation["repository_commit"]) is not None
        and isinstance(implementation.get("repository_tree"), str)
        and HEX40.fullmatch(implementation["repository_tree"]) is not None,
        "B2 qualification implementation identity is malformed",
    )
    toolchain = _mapping(
        report["toolchain_authority"], "B2 qualification toolchain authority"
    )
    _exact_keys(
        toolchain, {"bytes", "sha256"},
        "B2 qualification toolchain authority",
    )
    _integer(toolchain["bytes"], "B2 qualification toolchain bytes", 1)
    _digest(toolchain["sha256"], "B2 qualification toolchain digest")
    try:
        current_toolchain = load_toolchain_authority(ROOT)
    except ToolchainAuthorityError as error:
        raise B2QualificationError(
            f"canonical toolchain authority rejected: {error}"
        ) from error
    _require(
        dict(toolchain) == current_toolchain.manifest_record(),
        "B2 qualification toolchain authority differs",
    )
    b1 = _mapping(report["b1_authority"], "B2 qualification B1 authority")
    _exact_keys(
        b1,
        {
            "gate", "requalification_sha256",
            "runtime_authority_seal_sha256", "reseal_repository",
            "collision_identity",
        },
        "B2 qualification B1 authority",
    )
    for name in ("requalification_sha256", "runtime_authority_seal_sha256"):
        _digest(b1[name], f"B2 qualification {name}")
    collision = _mapping(b1["collision_identity"], "B2 qualification collision identity")
    _exact_keys(collision, {"tool_identity", "physics_identity"}, "B2 qualification collision identity")
    _digest(collision["tool_identity"], "B2 qualification collision tool identity")
    _digest(collision["physics_identity"], "B2 qualification collision physics identity")
    boundary = _mapping(report["compiled_boundary"], "B2 compiled-boundary proof")
    cases = _array(boundary.get("cases"), "B2 compiled-boundary cases")
    _require(
        cases == [
            {
                "case_id": case["case_id"],
                "ceiling_units": case["ceiling_units"],
                "passed_requirement": case["expected_pass"],
            }
            for case in BOUNDARY_CASES
        ],
        "B2 compiled-boundary dispositions differ",
    )
    declaration = _mapping(report["declaration"], "B2 qualification declaration")
    _require(
        declaration.get("mode") == "qualification"
        and declaration.get("map_count") == EXPECTED_MAP_COUNT,
        "B2 qualification declaration disposition differs",
    )
    _digest(declaration.get("sha256"), "B2 qualification declaration digest")
    stages = _mapping(report["stages"], "B2 qualification stages")
    _require(set(stages) == set(STAGES), "B2 qualification stages are missing")
    for stage, raw in stages.items():
        summary = _mapping(raw, f"B2 qualification {stage} summary")
        _exact_keys(summary, {"report", "map_count", "pass_count"}, f"B2 qualification {stage} summary")
        _require(summary["map_count"] == EXPECTED_MAP_COUNT, f"B2 qualification {stage} map count differs")
        count = _integer(summary["pass_count"], f"B2 qualification {stage} pass count")
        _require(count <= EXPECTED_MAP_COUNT, f"B2 qualification {stage} pass count exceeds 28")
    end_to_end = _mapping(report.get("end_to_end"), "B2 qualification end-to-end result")
    _exact_keys(
        end_to_end,
        {"required_pass_count", "pass_count", "failed_count", "passed_maps"},
        "B2 qualification end-to-end result",
    )
    pass_count = _integer(end_to_end.get("pass_count"), "end-to-end pass count")
    failed_count = _integer(end_to_end.get("failed_count"), "end-to-end failed count")
    passed_maps = _array(end_to_end.get("passed_maps"), "end-to-end passed maps")
    _require(end_to_end["required_pass_count"] == REQUIRED_END_TO_END_PASSES, "B2 qualification threshold differs")
    _require(pass_count >= REQUIRED_END_TO_END_PASSES, "B2 qualification has fewer than 20 end-to-end passes")
    _require(pass_count + failed_count == EXPECTED_MAP_COUNT, "B2 qualification end-to-end counts differ")
    _require(len(passed_maps) == pass_count and len(set(passed_maps)) == pass_count, "B2 qualification passed-map membership differs")
    authorization = _mapping(report["authorization"], "B2 qualification authorization")
    _require(
        authorization == {
            "final_declaration_allowed_by_this_report": False,
            "qualification_artifact_reuse_as_final_evidence": False,
            "passing_subset_admissible": False,
        },
        "B2 qualification authorization differs",
    )
    raw_evidence = _mapping(report["raw_evidence"], "B2 qualification raw evidence")
    _exact_keys(
        raw_evidence,
        {
            "normative_documents", "b1_gate", "compiled_boundary_report",
            "declaration", "stage_reports", "infrastructure_report",
            "stage_evidence",
        },
        "B2 qualification raw evidence",
    )
    raw_normative = _mapping(
        raw_evidence["normative_documents"],
        "B2 qualification raw normative evidence",
    )
    _exact_keys(raw_normative, {"design", "plan"}, "raw normative evidence")
    raw_stages = _mapping(raw_evidence["stage_reports"], "raw stage reports")
    _require(set(raw_stages) == set(STAGES), "raw qualification stage reports differ")
    for label, item in (
        *((f"normative {name}", value) for name, value in raw_normative.items()),
        ("B1 gate", raw_evidence["b1_gate"]),
        ("compiled boundary", raw_evidence["compiled_boundary_report"]),
        ("declaration", raw_evidence["declaration"]),
        *((f"stage {name}", value) for name, value in raw_stages.items()),
        ("infrastructure", raw_evidence["infrastructure_report"]),
    ):
        entry = _mapping(item, f"raw {label} evidence")
        _exact_keys(entry, {"path", "record"}, f"raw {label} evidence")
        path = entry["path"]
        _require(
            isinstance(path, str) and Path(path).is_absolute(),
            f"raw {label} evidence path is not absolute",
        )
        record = _mapping(entry["record"], f"raw {label} evidence record")
        _exact_keys(record, {"bytes", "sha256"}, f"raw {label} evidence record")
        _integer(record["bytes"], f"raw {label} evidence bytes", 1)
        _digest(record["sha256"], f"raw {label} evidence digest")
    stage_evidence = raw_evidence["stage_evidence"]
    if stage_evidence is not None:
        roots = _mapping(stage_evidence, "raw retained stage evidence")
        _exact_keys(roots, {
            "source_root", "source_cold_root", "compiled_root",
            "compile_evidence_root", "q2tool", "basedir",
            "compiled_cm_evidence_root", "cm_oracle", "pmove_oracle",
            "hook_oracle", "fall_oracle", "hook_attestation", "python_runtime",
            "materialized_root", "materialization_log_root", "claims_root",
            "analysis_root", "atlas_evidence_root",
            "promotion_evidence_root", "infrastructure_evidence_root",
            "syntax_report",
        }, "raw retained stage evidence")
        for name, value in roots.items():
            _require(isinstance(value, str) and Path(value).is_absolute(),
                     f"retained {name} path is not absolute")
    _require(report.get("failures") == [], "B2 qualification retains failures")
    return dict(report)


def replay_qualification(
    value: object,
    *,
    repo_root: Path = ROOT,
    use_reported_implementation: bool = False,
) -> dict[str, Any]:
    """Reopen every retained raw input and reproduce the exact summary bytes."""

    report = validate_qualification(value)
    raw = _mapping(report["raw_evidence"], "B2 qualification raw evidence")
    stage_evidence = _mapping(
        raw.get("stage_evidence"), "retained qualification stage evidence"
    )

    retained: list[tuple[Path, Mapping[str, Any]]] = []

    def reopen(item: object, label: str) -> Path:
        entry = _mapping(item, f"raw {label}")
        path = Path(str(entry["path"]))
        _require(path.is_absolute(), f"raw {label} path is not absolute")
        expected = _mapping(entry["record"], f"raw {label} record")
        _require(_file_record(path) == dict(expected), f"raw {label} bytes changed")
        retained.append((path, expected))
        return path

    normative = _mapping(raw["normative_documents"], "raw normative evidence")
    stages = _mapping(raw["stage_reports"], "raw stage evidence")
    raw_b1_gate = reopen(raw["b1_gate"], "B1 gate")
    current_b1_gate = repo_root.resolve() / "docs/multires/B1-GATE.json"
    _require(
        _file_record(current_b1_gate) == _file_record(raw_b1_gate),
        "current B1 trust-root bytes differ from retained qualification evidence",
    )
    arguments = argparse.Namespace(
        design=reopen(normative["design"], "design"),
        plan=reopen(normative["plan"], "plan"),
        repo_root=repo_root.resolve(),
        b1_gate=current_b1_gate,
        boundary_proof_report=reopen(
            raw["compiled_boundary_report"], "compiled boundary report"
        ),
        declaration=reopen(raw["declaration"], "declaration"),
        source_report=reopen(stages["source"], "source report"),
        compile_report=reopen(stages["compile"], "compile report"),
        compiled_cm_preflight_report=reopen(
            stages["compiled-cm-preflight"], "compiled-CM report"
        ),
        materialization_report=reopen(
            stages["materialization"], "materialization report"
        ),
        claims_report=reopen(stages["claims"], "claims report"),
        atlas_build_report=reopen(stages["atlas-build"], "Atlas report"),
        generated_promotion_report=reopen(
            stages["generated-promotion"], "promotion report"
        ),
        infrastructure_report=reopen(
            raw["infrastructure_report"], "infrastructure report"
        ),
        **{name: Path(str(value)) for name, value in stage_evidence.items()},
    )
    recomputed = assemble_qualification(
        arguments,
        replay_implementation=(
            _mapping(report["implementation"], "B2 qualification implementation")
            if use_reported_implementation
            else None
        ),
        replay_raw_evidence=(raw if use_reported_implementation else None),
    )
    _require(
        canonical_bytes(recomputed) == canonical_bytes(report),
        "qualification canonical summary differs from raw-evidence replay",
    )
    for path, expected in retained:
        _require(_file_record(path) == dict(expected), f"raw evidence changed during replay: {path}")
    return recomputed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--b1-gate", type=Path, required=True)
    parser.add_argument("--boundary-proof-report", type=Path, required=True)
    parser.add_argument("--declaration", type=Path, required=True)
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--compile-report", type=Path, required=True)
    parser.add_argument("--compiled-cm-preflight-report", type=Path, required=True)
    parser.add_argument("--materialization-report", type=Path, required=True)
    parser.add_argument("--claims-report", type=Path, required=True)
    parser.add_argument("--atlas-build-report", type=Path, required=True)
    parser.add_argument("--generated-promotion-report", type=Path, required=True)
    parser.add_argument("--infrastructure-report", type=Path, required=True)
    parser.add_argument("--claims-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-cold-root", type=Path, required=True)
    parser.add_argument("--compiled-root", type=Path, required=True)
    parser.add_argument("--compile-evidence-root", type=Path, required=True)
    parser.add_argument("--q2tool", type=Path, required=True)
    parser.add_argument("--basedir", type=Path, required=True)
    parser.add_argument("--compiled-cm-evidence-root", type=Path, required=True)
    parser.add_argument("--cm-oracle", type=Path, required=True)
    parser.add_argument("--pmove-oracle", type=Path, required=True)
    parser.add_argument("--hook-oracle", type=Path, required=True)
    parser.add_argument("--fall-oracle", type=Path, required=True)
    parser.add_argument("--hook-attestation", type=Path, required=True)
    parser.add_argument("--python-runtime", type=Path, required=True)
    parser.add_argument("--materialized-root", type=Path, required=True)
    parser.add_argument("--materialization-log-root", type=Path, required=True)
    parser.add_argument("--analysis-root", type=Path, required=True)
    parser.add_argument("--atlas-evidence-root", type=Path, required=True)
    parser.add_argument("--promotion-evidence-root", type=Path, required=True)
    parser.add_argument("--infrastructure-evidence-root", type=Path, required=True)
    parser.add_argument("--syntax-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.output.exists() or args.output.is_symlink():
            raise B2QualificationError("qualification output already exists; refusing overwrite")
        try:
            args.output.resolve().relative_to(args.repo_root.resolve())
        except ValueError:
            pass
        else:
            raise B2QualificationError("qualification output must be outside the repository")
        _require_qualification_path(args.output, "qualification output")
        report = assemble_qualification(args)
        _require(
            repository_binding(args.repo_root.resolve()) == report["implementation"],
            "repository changed immediately before qualification publication",
        )
        payload = canonical_bytes(report)
        _exclusive_write(args.output, payload)
        sys.stdout.buffer.write(payload)
        return 0
    except (
        B2QualificationError, B1AuthorityError, BoundaryQualificationError,
        GeneratorCohortError, RetiredCohortRegistryError, OSError,
    ) as error:
        print(f"B2 qualification refused: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
