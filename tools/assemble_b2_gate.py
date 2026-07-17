#!/usr/bin/env python3
"""Assemble the canonical B2 gate from exact, independently checked evidence.

This command has no discovery, repair, fallback, or partial-publication mode.
Every path is supplied explicitly, all stage memberships and content hashes are
recomputed, and the output is created exclusively only after every predicate
is green.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
import re
import struct
import subprocess
import sys
from typing import Any, Mapping, Sequence

try:
    import zstandard
except ImportError:  # pragma: no cover - the gate reports this explicitly.
    zstandard = None


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.atlas_source_closure import (  # noqa: E402
    atlas_analyzer_authority_inputs,
    atlas_analyzer_authority_sha256,
)
from harness.hook_claims_v4 import (  # noqa: E402
    HookClaimsV4Error,
    validate_materialization as validate_hook_materialization_v4,
    validate_runtime_sidecar,
)
from harness.ibsp38 import BspValidationError, parse_ibsp38  # noqa: E402
from harness.atlas_b1_authority import (  # noqa: E402
    B1AuthorityError,
    load_b1_authority_gate,
)
from tools.generator_claim_validator import (  # noqa: E402
    ClaimValidationError,
    _hurt_bounds,
    build_generator_claims,
    validate_report as validate_claim_report,
    validate_stock_analysis,
)
from tools.assemble_b2_qualification import (  # noqa: E402
    B2QualificationError,
    QUALIFICATION_SCHEMA,
    replay_qualification,
)
from tools.run_generator_claim_campaign import (  # noqa: E402
    CAMPAIGN_SCHEMA,
    validate_campaign,
)
from tools.run_generator_cohort import (  # noqa: E402
    CONCRETE_STYLES,
    SOURCE_SUFFIXES,
    STAGE_SUFFIXES,
    GeneratorCohortError,
    canonical_bytes,
    load_declaration,
    repository_binding,
    verify_stage_membership,
)
from tools.retired_cohort_registry import (  # noqa: E402
    RetiredCohortRegistryError,
    require_unretired_declaration,
)
from tools.source_route_contract import ROUTE_CONTRACT_SCHEMA  # noqa: E402
from tools.validate_maps import deathmatch_spawn_origins  # noqa: E402


GATE_SCHEMA = "q2-multires-b2-gate-v1"
EXPECTED_COHORT = "b2g26_final_71444"
EXPECTED_DESIGN_SHA256 = (
    "c55fc7ffc32bd0e88410b8493b46c179f3333f3806632ff8e6530f1c717508e6"
)
EXPECTED_PLAN_SHA256 = (
    "371577feb8c40f542c90eec4b4aa91ef84c4a8e2019bf1614e59c46aedfec410"
)
COMPILED_CM_PREFLIGHT_SCHEMA = "q2-b2-compiled-cm-preflight-v1"
COMPILED_CM_PREFLIGHT_STAGE = "post-q2tool-compiled-cm-preflight"
COMPILED_CM_PREFLIGHT_STATUS = "non-admissible-preflight-only"
PREFLIGHT_IMPLEMENTATION_PATHS = (
    "tools/run_compiled_cm_preflight.py",
    "harness/atlas_analyzer.py",
    "harness/atlas_b1_authority.py",
    "harness/ibsp38.py",
    "tools/run_generator_cohort.py",
    "tools/validate_maps.py",
)
HEX64 = re.compile(r"^[0-9a-f]{64}$")
HEX40 = re.compile(r"^[0-9a-f]{40}$")
STOCK_IDS = tuple(f"q2dm{number}" for number in range(1, 9))
STOCK_ANALYSIS_SUFFIXES = STAGE_SUFFIXES["analysis"]
MAX_L0_CHUNKS = 1200
MAX_L0_BYTES = 16 * 1024 * 1024
MAX_ATLAS_BYTES = 32 * 1024 * 1024
MAX_BUILD_RSS_BYTES = 512 * 1024 * 1024
MAX_DYN_BATCH_BYTES = 8 * 1024 * 1024
MAX_FEATURE_ASSEMBLY_P99_NS = 500_000
QUALIFICATION_STABLE_IMPLEMENTATION_KEYS = frozenset({
    "atlas_analyzer_authority_file_count",
    "atlas_analyzer_authority_sha256",
    "generator_sha256",
    "git_clean",
    "routes_sha256",
})
QUALIFICATION_SUCCESSOR_PATHS = frozenset({
    "docs/multires/B2-C-GENERATOR-CLAIM-CONTRACT.md",
    "docs/multires/B2-GATE-ASSEMBLY.md",
    "docs/multires/B2-GENERATED-COHORT-71444-DECLARATION.json",
    "docs/multires/B2-GENERATED-COHORT-DECLARATION.json",
    "schemas/q2-multires-b2-gate-v1.schema.json",
    "tests/test_b2_gate.py",
    "tests/test_b2_operational_docs.py",
    "tests/test_generator_claim_campaign.py",
    "tests/test_generator_cohort.py",
    "tests/test_retired_cohort_registry.py",
    "tools/assemble_b2_gate.py",
})


class B2GateError(ValueError):
    """Raised before publication when any B2 evidence is inadmissible."""


@dataclass(frozen=True)
class B2GatePaths:
    design: Path
    plan: Path
    repo_root: Path
    b1_gate: Path
    cm_oracle: Path
    pmove_oracle: Path
    hook_oracle: Path
    fall_oracle: Path
    hook_attestation: Path
    atlas_verifier: Path
    declaration: Path
    source_dir: Path
    source_cold_dir: Path
    source_freeze_report: Path
    compiled_dir: Path
    compiled_membership_report: Path
    compiled_static_report: Path
    compiled_cm_preflight_report: Path
    materialized_dir: Path
    materialized_membership_report: Path
    claims_dir: Path
    claims_prepare_report: Path
    analysis_dir: Path
    generated_build_report: Path
    generated_validation_report: Path
    stock_provenance: Path
    stock_inventory: Path
    stock_bsp_dir: Path
    stock_analysis_dir: Path
    stock_validation_dir: Path
    dyn_evidence_executable: Path
    dyn_evidence_report: Path
    test_report: Path
    qualification_report: Path


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise B2GateError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def _file_record(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise B2GateError(f"required regular file is missing or a symlink: {path}")
    return {"bytes": path.stat().st_size, "sha256": _file_sha256(path)}


def _reject_duplicates(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise B2GateError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _load_json_and_raw(
    path: Path, *, canonical: bool = True,
) -> tuple[Any, bytes]:
    try:
        raw = path.read_bytes()
        value = json.loads(
            raw,
            object_pairs_hook=_reject_duplicates,
            parse_constant=lambda token: (_ for _ in ()).throw(
                B2GateError(f"non-finite JSON token {token}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise B2GateError(f"cannot read JSON {path}: {exc}") from exc
    if canonical and raw != canonical_bytes(value):
        raise B2GateError(f"JSON is not canonical compact/sorted JSON plus LF: {path}")
    return value, raw


def _load_json(path: Path, *, canonical: bool = True) -> Any:
    return _load_json_and_raw(path, canonical=canonical)[0]


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise B2GateError(f"{label} must be an object")
    return value


def _list(value: object, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise B2GateError(f"{label} must be an array")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        raise B2GateError(
            f"{label} keys differ; missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _integer(value: object, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise B2GateError(f"{label} must be an integer >= {minimum}")
    return value


def _digest(value: object, label: str) -> str:
    if not isinstance(value, str) or not HEX64.fullmatch(value):
        raise B2GateError(f"{label} must be a lowercase SHA-256")
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise B2GateError(message)


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


def _exact_directory_files(directory: Path, expected: set[str], label: str) -> None:
    if not directory.is_dir() or directory.is_symlink():
        raise B2GateError(f"{label} is missing or is a symlink")
    actual: set[str] = set()
    symlinks: list[str] = []
    for path in sorted(directory.rglob("*")):
        relative = path.relative_to(directory).as_posix()
        if path.is_symlink():
            symlinks.append(relative)
        elif path.is_file():
            actual.add(relative)
    if symlinks or actual != expected:
        raise B2GateError(
            f"{label} membership differs; symlinks={symlinks}, "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )


def _require_same_files(
    first: Path, second: Path, names: Sequence[str], label: str
) -> None:
    for name in names:
        left = first / name
        right = second / name
        if _file_record(left) != _file_record(right):
            raise B2GateError(f"{label} changed across stages: {name}")


def _membership_with_declaration(
    declaration: Mapping[str, Any], declaration_sha256: str, directory: Path, stage: str
) -> dict[str, Any]:
    report = verify_stage_membership(declaration, directory, stage)
    report["declaration_sha256"] = declaration_sha256
    if report["passed"] is not True or report["failures"]:
        raise B2GateError(
            f"{stage} exact membership failed: " + "; ".join(report["failures"])
        )
    return report


def _require_report_equals(path: Path, expected: object, label: str) -> None:
    actual = _load_json(path)
    if actual != expected:
        raise B2GateError(f"{label} differs from independently recomputed evidence")


def _validate_normative_documents(paths: B2GatePaths) -> dict[str, Any]:
    design = _file_record(paths.design)
    plan = _file_record(paths.plan)
    _require(
        design["sha256"] == EXPECTED_DESIGN_SHA256,
        "normative design digest differs from accepted specification",
    )
    _require(
        plan["sha256"] == EXPECTED_PLAN_SHA256,
        "normative plan digest differs from accepted execution plan",
    )
    return {"design": design, "plan": plan}


def _validate_implementation(
    paths: B2GatePaths, supplied: Mapping[str, Any] | None
) -> dict[str, Any]:
    binding = dict(supplied) if supplied is not None else repository_binding(paths.repo_root)
    expected_keys = {
        "repository_commit", "repository_tree", "git_clean",
        "atlas_analyzer_authority_sha256", "atlas_analyzer_authority_file_count",
        "generator_sha256", "routes_sha256",
    }
    _exact_keys(binding, expected_keys, "implementation binding")
    _require(binding["git_clean"] is True, "implementation binding is not clean")
    for name in ("repository_commit", "repository_tree"):
        _require(
            isinstance(binding[name], str) and HEX40.fullmatch(binding[name]) is not None,
            f"implementation {name} is malformed",
        )
    for name in (
        "atlas_analyzer_authority_sha256", "generator_sha256", "routes_sha256"
    ):
        _digest(binding[name], f"implementation {name}")
    _integer(
        binding["atlas_analyzer_authority_file_count"],
        "implementation analyzer closure file count", minimum=1,
    )
    # A supplied binding is only a test seam; it still has to match live bytes.
    closure_inputs = atlas_analyzer_authority_inputs(paths.repo_root)
    _require(
        binding["atlas_analyzer_authority_sha256"]
        == atlas_analyzer_authority_sha256(paths.repo_root),
        "supplied analyzer closure is stale",
    )
    _require(
        binding["atlas_analyzer_authority_file_count"] == len(closure_inputs),
        "supplied analyzer closure file count differs",
    )
    _require(
        binding["generator_sha256"] == _file_sha256(paths.repo_root / "maps/generator.py"),
        "supplied generator digest is stale",
    )
    _require(
        binding["routes_sha256"] == _file_sha256(paths.repo_root / "maps/routes.py"),
        "supplied routes digest is stale",
    )
    if supplied is not None:
        # Unit fixtures may not be Git worktrees. Production never supplies this seam.
        return binding
    return binding


def _validate_b1_and_oracles(
    paths: B2GatePaths, normative: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, str]]:
    expected_gate = paths.repo_root / "docs/multires/B1-GATE.json"
    _require(
        paths.b1_gate.resolve() == expected_gate.resolve(),
        "supplied B1 gate is not the repository trust root",
    )
    try:
        authority = load_b1_authority_gate(paths.repo_root)
    except B1AuthorityError as exc:
        raise B2GateError(f"canonical B1 authority rejected: {exc}") from exc
    gate = _mapping(_load_json(paths.b1_gate, canonical=False), "B1 gate")
    _require(gate.get("schema") == "q2-multires-batch-gate-v1", "B1 schema differs")
    _require(gate.get("batch") == "B1", "B1 batch identity differs")
    _require(gate.get("status") == "green", "B1 status is not green")
    b1_predicate = _mapping(gate.get("gate"), "B1 predicate")
    _require(b1_predicate.get("green") is True, "B1 predicate is not green")
    _require(b1_predicate.get("failures") == [], "B1 retains failures")
    b1_docs = _mapping(gate.get("normative_documents"), "B1 normative documents")
    _require(
        b1_docs.get("design_sha256") == normative["design"]["sha256"]
        and b1_docs.get("plan_sha256") == normative["plan"]["sha256"],
        "B1 normative bindings differ",
    )

    binaries = {
        "cm": _file_sha256(paths.cm_oracle),
        "pmove": _file_sha256(paths.pmove_oracle),
        "hook": _file_sha256(paths.hook_oracle),
        "fall": _file_sha256(paths.fall_oracle),
        "hook_attestation": _file_sha256(paths.hook_attestation),
        "atlas_verifier": _file_sha256(paths.atlas_verifier),
    }
    artifacts = _mapping(gate.get("artifacts"), "B1 artifacts")
    transformed = _mapping(
        artifacts.get("transformed_inline_collision"), "B1 collision artifacts"
    )
    fall = _mapping(artifacts.get("fall_damage_oracle"), "B1 fall artifact")
    parity = _mapping(artifacts.get("hook_parity_attestation"), "B1 parity artifact")
    _require(transformed.get("cm_oracle_sha256") == binaries["cm"], "CM oracle differs from B1")
    _require(transformed.get("pmove_oracle_sha256") == binaries["pmove"], "Pmove oracle differs from B1")
    _require(fall.get("executable_sha256") == binaries["fall"], "fall oracle differs from B1")
    _require(parity.get("sha256") == binaries["hook_attestation"], "hook attestation differs from B1")
    _require(parity.get("passed") is True, "B1 hook parity is not passed")
    _require(
        authority.cm_executable_sha256 == binaries["cm"]
        and authority.pmove_executable_sha256 == binaries["pmove"]
        and authority.fall_executable_sha256 == binaries["fall"]
        and authority.hook_attestation_sha256 == binaries["hook_attestation"],
        "canonical B1 loader authority differs from supplied oracle bytes",
    )
    for name in ("oracle_tool_identity", "oracle_source_closure_sha256"):
        _digest(transformed.get(name), f"B1 collision {name}")
    binaries["cm_tool_identity"] = transformed["oracle_tool_identity"]
    binaries["cm_source_closure_sha256"] = transformed[
        "oracle_source_closure_sha256"
    ]
    hook_attestation = _mapping(
        _load_json(paths.hook_attestation, canonical=False),
        "B1 hook parity attestation",
    )
    _require(
        hook_attestation.get("schema") == "q2-hook-parity-attestation-v1"
        and hook_attestation.get("passed") is True,
        "B1 hook parity attestation bytes are not passed",
    )
    hook_attested_binaries = _mapping(
        hook_attestation.get("binaries"),
        "B1 hook parity attested binaries",
    )
    _require(
        hook_attested_binaries.get("cm_oracle_sha256") == binaries["cm"]
        and hook_attested_binaries.get("pmove_oracle_sha256") == binaries["pmove"]
        and hook_attested_binaries.get("hook_oracle_sha256") == binaries["hook"],
        "oracle bytes differ from B1 hook parity attestation",
    )
    return {
        "gate": _file_record(paths.b1_gate),
        "oracles": {name: _file_record(path) for name, path in {
            "cm": paths.cm_oracle, "pmove": paths.pmove_oracle,
            "hook": paths.hook_oracle, "fall": paths.fall_oracle,
            "hook_attestation": paths.hook_attestation,
            "atlas_verifier": paths.atlas_verifier,
        }.items()},
    }, binaries


def _git_capture(repo_root: Path, arguments: Sequence[str]) -> subprocess.CompletedProcess[bytes]:
    environment = os.environ.copy()
    environment["GIT_OPTIONAL_LOCKS"] = "0"
    try:
        return subprocess.run(
            ["git", "-C", str(repo_root), *arguments],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=30,
            env=environment,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise B2GateError(f"qualification successor git check failed: {error}") from error


def _validate_qualification_successor(
    repo_root: Path,
    qualified: Mapping[str, Any],
    current: Mapping[str, Any],
) -> dict[str, Any]:
    expected_keys = QUALIFICATION_STABLE_IMPLEMENTATION_KEYS | {
        "repository_commit", "repository_tree"
    }
    _require(
        set(qualified) == expected_keys and set(current) == expected_keys,
        "B2 qualification implementation keys differ",
    )
    _require(
        all(qualified[key] == current[key] for key in QUALIFICATION_STABLE_IMPLEMENTATION_KEYS),
        "B2 qualification producer/analyzer authority differs",
    )
    qualified_commit = str(qualified["repository_commit"])
    current_commit = str(current["repository_commit"])
    _require(
        HEX40.fullmatch(qualified_commit) is not None
        and HEX40.fullmatch(current_commit) is not None,
        "B2 qualification successor commit is malformed",
    )
    ancestor = _git_capture(
        repo_root, ["merge-base", "--is-ancestor", qualified_commit, current_commit]
    )
    _require(
        ancestor.returncode == 0,
        "B2 qualification commit is not an ancestor of the final implementation",
    )
    diff = _git_capture(
        repo_root,
        ["diff", "--name-status", "--no-renames", qualified_commit, current_commit],
    )
    _require(diff.returncode == 0, "B2 qualification successor diff failed")
    changed: dict[str, str] = {}
    for raw_line in diff.stdout.decode("utf-8", errors="strict").splitlines():
        parts = raw_line.split("\t")
        _require(
            len(parts) == 2 and parts[0] in {"A", "M"} and parts[1] not in changed,
            "B2 qualification successor diff is malformed",
        )
        changed[parts[1]] = parts[0]
    _require(
        set(changed) == QUALIFICATION_SUCCESSOR_PATHS,
        "B2 qualification successor changed files differ from the declaration-only authority",
    )
    _require(
        changed["docs/multires/B2-GENERATED-COHORT-71444-DECLARATION.json"] == "A",
        "B2 qualification successor did not add the immutable 71444 declaration",
    )
    return {
        "qualified_repository_commit": qualified_commit,
        "qualified_repository_tree": qualified["repository_tree"],
        "final_repository_commit": current_commit,
        "final_repository_tree": current["repository_tree"],
        "stable_authority_equal": True,
        "changed_paths": sorted(changed),
    }


def _validate_qualification_report(
    paths: B2GatePaths,
    normative: Mapping[str, Any],
    implementation: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        report = replay_qualification(
            _load_json(paths.qualification_report), repo_root=paths.repo_root,
            use_reported_implementation=True,
        )
    except B2QualificationError as exc:
        raise B2GateError(f"B2 toolchain qualification rejected: {exc}") from exc
    _require(
        report["normative_documents"] == normative,
        "B2 qualification normative document binding differs",
    )
    relation = _validate_qualification_successor(
        paths.repo_root, report["implementation"], implementation
    )
    b1_gate = _mapping(
        _load_json(paths.b1_gate, canonical=False), "B1 gate"
    )
    requalification = _mapping(
        b1_gate.get("authority_requalification"),
        "B1 authority requalification",
    )
    runtime_seal = _mapping(
        requalification.get("probe_runtime_authority_seal"),
        "B1 probe runtime authority seal",
    )
    repository = _mapping(
        requalification.get("repository"), "B1 requalification repository"
    )
    live = _mapping(
        requalification.get("live_identities"), "B1 requalification identities"
    )
    collision = _mapping(
        live.get("collision"), "B1 requalification collision identity"
    )
    expected_b1 = {
        "gate": _file_record(paths.b1_gate),
        "requalification_sha256": _sha256_bytes(canonical_bytes(requalification)),
        "runtime_authority_seal_sha256": _sha256_bytes(
            canonical_bytes(runtime_seal)
        ),
        "reseal_repository": dict(repository),
        "collision_identity": {
            "tool_identity": collision.get("tool_identity"),
            "physics_identity": collision.get("physics_identity"),
        },
    }
    _require(
        report["b1_authority"] == expected_b1,
        "B2 qualification B1 reseal binding differs",
    )
    return {
        "report": _file_record(paths.qualification_report),
        "schema": QUALIFICATION_SCHEMA,
        "qualification_id": report["qualification_id"],
        "status": "green",
        "non_admissible": True,
        "retryable": True,
        "final_cohort_authorized": False,
        "end_to_end_pass_count": report["end_to_end"]["pass_count"],
        "required_end_to_end_pass_count": report["end_to_end"][
            "required_pass_count"
        ],
        "implementation_successor": relation,
    }


def _expected_71444_rows() -> list[dict[str, Any]]:
    rows = []
    ordinal = 0
    for style_index, style in enumerate(CONCRETE_STYLES):
        for member in range(4):
            seed = 71_444_000 + style_index * 100 + member
            rows.append({
                "ordinal": ordinal,
                "map": f"b2g26_{style}_{seed}",
                "seed": seed,
                "style": style,
                "grid": 5,
                "observed_heat": None,
            })
            ordinal += 1
    return rows


def _validate_declaration(path: Path) -> tuple[dict[str, Any], str]:
    declaration, digest = load_declaration(path)
    try:
        require_unretired_declaration(path, declaration, digest)
    except RetiredCohortRegistryError as exc:
        raise B2GateError(f"B2 declaration admission refused: {exc}") from exc
    _require(
        declaration["cohort_id"] == EXPECTED_COHORT,
        "B2 gate accepts only cohort 71444",
    )
    _require(
        declaration["maps"] == _expected_71444_rows(),
        "71444 map/seed selection differs",
    )
    return declaration, digest


def _validate_source_freeze(
    paths: B2GatePaths,
    declaration: Mapping[str, Any],
    declaration_sha256: str,
    implementation: Mapping[str, Any],
) -> dict[str, Any]:
    primary = _membership_with_declaration(
        declaration, declaration_sha256, paths.source_dir, "source"
    )
    cold = _membership_with_declaration(
        declaration, declaration_sha256, paths.source_cold_dir, "source"
    )
    names = [
        f"{row['map']}{suffix}"
        for row in declaration["maps"] for suffix in SOURCE_SUFFIXES
    ]
    _require_same_files(paths.source_dir, paths.source_cold_dir, names, "source cold rebuild")
    report = _mapping(_load_json(paths.source_freeze_report), "source-freeze report")
    _exact_keys(
        report,
        {
            "schema", "cohort_id", "status", "bundle_admissible",
            "atlas_admissible", "selection_timing", "selection_policy",
            "replacement_allowed", "salvage_allowed", "declaration_sha256",
            "implementation", "map_count", "style_counts",
            "unique_layout_count", "route_contract_pass_count",
            "spawn_origin_binding_pass_count",
            "source_suffixes", "cold_rebuild", "maps", "failures", "passed",
        },
        "source-freeze report",
    )
    _require(report.get("schema") == "q2-b2-generated-source-freeze-v1", "source-freeze schema differs")
    _require(report.get("cohort_id") == EXPECTED_COHORT, "source-freeze cohort differs")
    _require(report.get("declaration_sha256") == declaration_sha256, "source-freeze declaration differs")
    _require(report.get("implementation") == implementation, "source-freeze implementation is stale")
    _require(report.get("map_count") == 28, "source-freeze map count differs")
    _require(report.get("unique_layout_count") == 28, "source-freeze layout uniqueness differs")
    _require(report.get("route_contract_pass_count") == 28, "source-freeze routes are incomplete")
    _validate_source_spawn_origin_binding_pass_count(
        report.get("spawn_origin_binding_pass_count")
    )
    _require(report.get("passed") is True and report.get("failures") == [], "source-freeze is not green")
    _require(report.get("bundle_admissible") is False, "source freeze cannot claim bundle admission")
    _require(report.get("atlas_admissible") is False, "source freeze cannot claim Atlas admission")
    _require(report.get("status") == "source-frozen-pre-compile", "source-freeze status differs")
    _require(report.get("selection_timing") == "declared-before-generation", "source-freeze selection timing differs")
    _require(report.get("selection_policy") == "all-or-nothing", "source-freeze selection policy differs")
    _require(report.get("replacement_allowed") is False, "source-freeze permits replacement")
    _require(report.get("salvage_allowed") is False, "source-freeze permits salvage")
    _require(
        report.get("style_counts") == {
            style: 4 for style in sorted(CONCRETE_STYLES)
        },
        "source-freeze style counts differ",
    )
    _require(report.get("source_suffixes") == list(SOURCE_SUFFIXES), "source-freeze suffixes differ")
    _require(
        report.get("cold_rebuild") == {
            "fresh_process_required": False,
            "independent_directory": True,
            "file_count": 28 * len(SOURCE_SUFFIXES),
            "all_file_bytes_match": True,
        },
        "source-freeze cold-rebuild contract differs",
    )
    rows = _list(report.get("maps"), "source-freeze maps")
    _require(len(rows) == 28, "source-freeze rows differ")
    for declared, row in zip(declaration["maps"], rows):
        record = _mapping(row, f"source-freeze {declared['map']}")
        _exact_keys(
            record,
            {
                "ordinal", "map", "seed", "style", "grid", "metadata",
                "source_static", "route_contract", "spawn_origin_binding",
                "source_files",
            },
            f"source-freeze {declared['map']}",
        )
        _require(record.get("ordinal") == declared["ordinal"], "source-freeze ordinal differs")
        _require(record.get("map") == declared["map"], "source-freeze map differs")
        for field in ("seed", "style", "grid"):
            _require(
                record.get(field) == declared[field],
                f"source-freeze {field} differs for {declared['map']}",
            )
        source_static = _mapping(
            record.get("source_static"), f"source-freeze static {declared['map']}"
        )
        _require(
            source_static.get("map") == declared["map"]
            and source_static.get("static_ok") is True,
            f"source-freeze static validation differs for {declared['map']}",
        )
        route_contract = _validate_source_route_contract(
            record.get("route_contract"), declared["map"]
        )
        _validate_source_spawn_origin_binding(
            record.get("spawn_origin_binding"),
            route_contract,
            paths.source_dir / f"{declared['map']}.map",
            declared["map"],
        )
        files = _mapping(record.get("source_files"), "source-freeze file identities")
        _exact_keys(files, set(SOURCE_SUFFIXES), "source-freeze file identities")
        for suffix in SOURCE_SUFFIXES:
            _exact_keys(
                _mapping(files[suffix], f"source-freeze {declared['map']}{suffix}"),
                {"bytes", "sha256"},
                f"source-freeze {declared['map']}{suffix}",
            )
            _require(
                files.get(suffix) == _file_record(paths.source_dir / f"{declared['map']}{suffix}"),
                f"source-freeze digest is stale for {declared['map']}{suffix}",
            )
    return {
        "report": _file_record(paths.source_freeze_report),
        "primary_membership_sha256": _sha256_bytes(canonical_bytes(primary)),
        "cold_membership_sha256": _sha256_bytes(canonical_bytes(cold)),
        "map_count": 28,
    }


def _validate_source_route_contract(
    value: object, map_id: str
) -> Mapping[str, Any]:
    """Admit every source-route predicate required by the frozen B2 contract."""
    route_contract = _mapping(
        value, f"source-freeze route contract {map_id}"
    )
    _require(
        route_contract.get("schema") == ROUTE_CONTRACT_SCHEMA,
        f"source-freeze route contract schema differs for {map_id}",
    )
    _require(
        _integer(
            route_contract.get("spawn_count"),
            f"source-freeze route contract spawn count {map_id}",
        ) == 8,
        f"source-freeze route contract spawn count differs for {map_id}",
    )
    _require(
        route_contract.get("all_spawns_share_source_standing_component") is True,
        f"source-freeze route contract spawn component failed for {map_id}",
    )
    _require(
        route_contract.get("published_dist_covers_endpoint_loop_geometry") is True,
        f"source-freeze route contract endpoint-loop geometry failed for {map_id}",
    )
    _require(
        route_contract.get(
            "all_selected_endpoints_share_source_standing_component"
        ) is True
        and route_contract.get("exact_start_nodes_declared") is True
        and route_contract.get("room_edges_used_as_reachability") is False,
        f"source-freeze route contract failed for {map_id}",
    )
    return route_contract


def _validate_source_spawn_origin_binding_pass_count(value: object) -> None:
    _require(
        _integer(value, "source-freeze spawn-origin binding pass count") == 28,
        "source-freeze spawn-origin bindings are incomplete",
    )


def _spawn_origin_rows(value: object, label: str) -> list[list[float]]:
    rows = _list(value, label)
    _require(len(rows) == 8, f"{label} must contain exactly eight origins")
    normalized: list[list[float]] = []
    for index, value_row in enumerate(rows):
        row = _list(value_row, f"{label} {index}")
        _require(len(row) == 3, f"{label} {index} must contain three coordinates")
        coordinates: list[float] = []
        for axis, coordinate in zip("xyz", row):
            try:
                normalized_coordinate = float(coordinate)
            except (TypeError, ValueError, OverflowError):
                normalized_coordinate = math.nan
            _require(
                not isinstance(coordinate, bool)
                and isinstance(coordinate, (int, float))
                and math.isfinite(normalized_coordinate),
                f"{label} {index} {axis} must be finite numeric",
            )
            coordinates.append(normalized_coordinate)
        normalized.append(coordinates)
    _require(
        len({tuple(row) for row in normalized}) == 8,
        f"{label} must contain eight unique origins",
    )
    return normalized


def _validate_source_spawn_origin_binding(
    value: object,
    route_contract: Mapping[str, Any],
    source_map: Path,
    map_id: str,
) -> None:
    binding = _mapping(value, f"source-freeze spawn-origin binding {map_id}")
    _exact_keys(
        binding,
        {
            "schema", "source_artifact", "source_parser",
            "deathmatch_spawn_count", "spawn_origins",
            "source_spawn_origins_sha256", "route_spawn_origins_sha256",
            "route_contract_exact_match", "all_spawn_origins_unique",
            "source_component", "all_spawns_share_source_standing_component",
        },
        f"source-freeze spawn-origin binding {map_id}",
    )
    _require(
        binding.get("schema") == "q2-generator-source-spawn-origin-binding-v1",
        f"source-freeze spawn-origin binding schema differs for {map_id}",
    )
    _require(
        binding.get("source_artifact") == ".map"
        and binding.get("source_parser")
        == "tools.validate_maps.deathmatch_spawn_origins-v1",
        f"source-freeze spawn-origin authority differs for {map_id}",
    )
    try:
        parsed_source_origins = deathmatch_spawn_origins(source_map)
    except ValueError as exc:
        raise B2GateError(str(exc)) from exc
    source_origins = [list(origin) for origin in parsed_source_origins]
    binding_origins = _spawn_origin_rows(
        binding.get("spawn_origins"),
        f"source-freeze bound source origins {map_id}",
    )
    route_origins = _spawn_origin_rows(
        route_contract.get("spawn_origins"),
        f"source-freeze route origins {map_id}",
    )
    _require(
        _integer(
            binding.get("deathmatch_spawn_count"),
            f"source-freeze deathmatch spawn count {map_id}",
        ) == 8
        and len(source_origins) == 8,
        f"source-freeze deathmatch spawn count differs for {map_id}",
    )
    _require(
        binding_origins == source_origins,
        f"source-freeze bound origins differ from source map for {map_id}",
    )
    _require(
        route_origins == source_origins,
        f"source-freeze route origins differ from source map for {map_id}",
    )
    source_sha256 = _sha256_bytes(canonical_bytes(source_origins))
    route_sha256 = _sha256_bytes(canonical_bytes(route_origins))
    _require(
        binding.get("source_spawn_origins_sha256") == source_sha256
        and binding.get("route_spawn_origins_sha256") == route_sha256
        and route_contract.get("spawn_origins_sha256") == route_sha256,
        f"source-freeze spawn-origin digest binding differs for {map_id}",
    )
    route_component = route_contract.get("spawn_source_component")
    _require(
        isinstance(route_component, int)
        and not isinstance(route_component, bool)
        and route_component >= 0
        and binding.get("source_component") == route_component,
        f"source-freeze spawn-origin component binding differs for {map_id}",
    )
    _require(
        binding.get("route_contract_exact_match") is True
        and binding.get("all_spawn_origins_unique") is True
        and binding.get("all_spawns_share_source_standing_component") is True
        and route_contract.get("all_spawn_origins_unique") is True
        and route_contract.get("all_spawns_share_source_standing_component") is True,
        f"source-freeze spawn-origin binding predicates failed for {map_id}",
    )


def _validate_compiled_and_static(
    paths: B2GatePaths,
    declaration: Mapping[str, Any],
    declaration_sha256: str,
) -> dict[str, Any]:
    membership = _membership_with_declaration(
        declaration, declaration_sha256, paths.compiled_dir, "compiled"
    )
    _require_report_equals(
        paths.compiled_membership_report, membership, "compiled membership report"
    )
    shared = [
        f"{row['map']}{suffix}"
        for row in declaration["maps"] for suffix in SOURCE_SUFFIXES
    ]
    _require_same_files(paths.source_dir, paths.compiled_dir, shared, "compiled source")

    from tools.run_generator_cohort import _default_static_validator

    rows = []
    for declared in declaration["maps"]:
        result = dict(_default_static_validator(paths.compiled_dir / f"{declared['map']}.map"))
        _require(result.get("static_ok") is True, f"compiled static validation failed for {declared['map']}")
        rows.append(result)
    expected_static = {
        "schema": "q2-generator-v6-compiled-static-campaign-v1",
        "map_count": 28,
        "pass_count": 28,
        "passed": True,
        "maps": rows,
    }
    _require_report_equals(
        paths.compiled_static_report, expected_static, "compiled static report"
    )
    return {
        "membership": _file_record(paths.compiled_membership_report),
        "static": _file_record(paths.compiled_static_report),
        "map_count": 28,
    }


def _preflight_implementation_identity(repo_root: Path) -> dict[str, Any]:
    files = []
    for relative in PREFLIGHT_IMPLEMENTATION_PATHS:
        path = repo_root / relative
        record = _file_record(path)
        files.append({
            "path": relative,
            "bytes": record["bytes"],
            "sha256": record["sha256"],
        })
    return {
        "schema": "q2-b2-compiled-cm-preflight-implementation-v1",
        "files": files,
        "source_closure_sha256": _sha256_bytes(canonical_bytes(files)),
    }


def _origin_milliunits(raw: str, label: str) -> list[int]:
    words = raw.split()
    if len(words) != 3:
        raise B2GateError(f"{label} must have a three-axis origin")
    result = []
    for word in words:
        try:
            value = float(word)
        except ValueError as exc:
            raise B2GateError(f"{label} origin is not numeric") from exc
        _require(math.isfinite(value), f"{label} origin is not finite")
        milliunits = round(value * 1000)
        _require(
            abs(value * 1000 - milliunits) <= 1e-6,
            f"{label} origin is not representable in milliunits",
        )
        result.append(milliunits)
    return result


def _validate_preflight_spawn(
    value: object,
    compiled_origins: Mapping[int, list[int]],
    map_id: str,
) -> None:
    spawn = _mapping(value, f"compiled-CM spawn {map_id}")
    _exact_keys(
        spawn,
        {
            "entity_ordinal", "authored_origin_milliunits",
            "engine_link_lift_milliunits", "standing_clear",
            "crouched_clear", "supported", "support_drop_milliunits",
            "column_clearance_milliunits", "column_clear_96", "basic_escape",
            "failures", "passed",
        },
        f"compiled-CM spawn {map_id}",
    )
    entity_ordinal = _integer(
        spawn["entity_ordinal"], f"compiled-CM spawn entity ordinal {map_id}"
    )
    _require(
        entity_ordinal in compiled_origins,
        f"compiled-CM spawn entity is absent from BSP for {map_id}",
    )
    _require(
        spawn["authored_origin_milliunits"] == compiled_origins[entity_ordinal],
        f"compiled-CM spawn origin differs from BSP for {map_id}",
    )
    _require(
        spawn["engine_link_lift_milliunits"] == 9000,
        f"compiled-CM spawn link lift differs for {map_id}",
    )
    _require(
        spawn["standing_clear"] is True
        and spawn["crouched_clear"] is True
        and spawn["supported"] is True,
        f"compiled-CM stance/support failed for {map_id}",
    )
    support_drop = spawn["support_drop_milliunits"]
    _require(
        isinstance(support_drop, int) and not isinstance(support_drop, bool)
        and 0 <= support_drop <= 96_000,
        f"compiled-CM support drop differs for {map_id}",
    )
    _require(
        _integer(
            spawn["column_clearance_milliunits"],
            f"compiled-CM column clearance {map_id}",
        ) >= 96_000
        and spawn["column_clear_96"] is True,
        f"compiled-CM 96-unit column failed for {map_id}",
    )
    escape = _mapping(spawn["basic_escape"], f"compiled-CM escape {map_id}")
    _exact_keys(
        escape,
        {
            "distance_milliunits", "support_step_milliunits",
            "passing_direction_indices", "passed",
        },
        f"compiled-CM escape {map_id}",
    )
    directions = _list(
        escape["passing_direction_indices"],
        f"compiled-CM escape directions {map_id}",
    )
    _require(
        escape["distance_milliunits"] == 96_000
        and escape["support_step_milliunits"] == 16_000
        and escape["passed"] is True
        and directions
        and len(directions) == len(set(directions))
        and all(
            isinstance(item, int) and not isinstance(item, bool) and 0 <= item < 8
            for item in directions
        ),
        f"compiled-CM basic escape failed for {map_id}",
    )
    _require(
        spawn["failures"] == [] and spawn["passed"] is True,
        f"compiled-CM spawn retained failures for {map_id}",
    )


def _compiled_preflight_hazard_claims(compiled_dir: Path, map_id: str) -> list[dict[str, Any]]:
    lattice = _mapping(
        _load_json(compiled_dir / f"{map_id}.lattice.json", canonical=False),
        f"compiled-CM lattice hazards {map_id}",
    )
    danger = _list(lattice.get("danger"), f"compiled-CM danger claims {map_id}")
    claims = []
    danger_bounds = []
    for index, value in enumerate(danger):
        bounds = _list(value, f"compiled-CM danger bounds {map_id}:{index}")
        _require(
            len(bounds) == 6,
            f"compiled-CM danger bounds width differs for {map_id}",
        )
        scaled = []
        for axis, item in enumerate(bounds):
            _require(
                isinstance(item, (int, float)) and not isinstance(item, bool)
                and math.isfinite(float(item)),
                f"compiled-CM danger bound is invalid for {map_id}:{index}:{axis}",
            )
            milliunits = round(float(item) * 1000)
            _require(
                abs(float(item) * 1000 - milliunits) <= 1e-6,
                f"compiled-CM danger bound is not milliunit-exact for {map_id}",
            )
            scaled.append(milliunits)
        _require(
            all(scaled[axis] < scaled[axis + 3] for axis in range(3)),
            f"compiled-CM danger bounds are unordered for {map_id}",
        )
        danger_bounds.append(scaled)
    for index, scaled in enumerate(sorted(danger_bounds)):
        claims.append({
            "claim_id": f"hazard:lava:{index:04d}",
            "type": "lava",
            "bounds_milliunits": scaled,
        })
    try:
        hurt = _hurt_bounds(
            (compiled_dir / f"{map_id}.map").read_text(encoding="utf-8")
        )
    except (OSError, UnicodeError, ClaimValidationError) as exc:
        raise B2GateError(
            f"cannot derive compiled-CM hurt claims for {map_id}: {exc}"
        ) from exc
    for index, bounds in enumerate(hurt):
        claims.append({
            "claim_id": f"hazard:hurt:{index:04d}",
            "type": "hurt",
            "bounds_milliunits": bounds,
        })
    return sorted(claims, key=lambda claim: claim["claim_id"])


def _validate_compiled_cm_preflight(
    paths: B2GatePaths,
    declaration: Mapping[str, Any],
    declaration_sha256: str,
    binaries: Mapping[str, str],
) -> dict[str, Any]:
    report = _mapping(
        _load_json(paths.compiled_cm_preflight_report),
        "compiled-CM preflight report",
    )
    _exact_keys(
        report,
        {
            "schema", "stage", "admission_status", "promotion_authority",
            "cohort_id", "declaration", "compiled_root",
            "compiled_membership", "b1_authority", "implementation",
            "execution", "checks", "input_stability", "maps", "map_count",
            "pass_count", "failure_count", "failures", "passed",
            "canonical_record_sha256",
        },
        "compiled-CM preflight report",
    )
    _require(
        report["schema"] == COMPILED_CM_PREFLIGHT_SCHEMA
        and report["stage"] == COMPILED_CM_PREFLIGHT_STAGE,
        "compiled-CM preflight identity differs",
    )
    _require(
        report["admission_status"] == COMPILED_CM_PREFLIGHT_STATUS
        and report["promotion_authority"] is False,
        "compiled-CM preflight must remain non-admission evidence",
    )
    _require(
        report["cohort_id"] == declaration["cohort_id"],
        "compiled-CM preflight cohort differs",
    )
    declared = _mapping(report["declaration"], "compiled-CM declaration")
    _exact_keys(
        declared, {"path", "sha256", "map_count"},
        "compiled-CM declaration",
    )
    _require(
        declared == {
            "path": str(paths.declaration.expanduser().absolute()),
            "sha256": declaration_sha256,
            "map_count": 28,
        },
        "compiled-CM declaration binding differs",
    )
    _require(
        report["compiled_root"] == str(paths.compiled_dir.expanduser().absolute()),
        "compiled-CM root binding differs",
    )
    membership = verify_stage_membership(
        declaration, paths.compiled_dir, "compiled"
    )
    membership_identity = _mapping(
        report["compiled_membership"], "compiled-CM membership identity"
    )
    _exact_keys(
        membership_identity, {"report", "report_sha256"},
        "compiled-CM membership identity",
    )
    _require(
        membership_identity["report"] == membership
        and membership_identity["report_sha256"]
        == _sha256_bytes(canonical_bytes(membership)),
        "compiled-CM membership binding differs",
    )
    b1 = _mapping(report["b1_authority"], "compiled-CM B1 authority")
    _exact_keys(
        b1,
        {
            "gate_sha256", "cm_executable_sha256", "cm_tool_identity",
            "cm_source_closure_sha256",
        },
        "compiled-CM B1 authority",
    )
    _require(
        b1 == {
            "gate_sha256": _file_sha256(paths.b1_gate),
            "cm_executable_sha256": binaries["cm"],
            "cm_tool_identity": binaries["cm_tool_identity"],
            "cm_source_closure_sha256": binaries["cm_source_closure_sha256"],
        },
        "compiled-CM B1/CM binding differs",
    )
    expected_implementation = _preflight_implementation_identity(paths.repo_root)
    _require(
        report["implementation"] == expected_implementation,
        "compiled-CM implementation closure differs",
    )
    execution = _mapping(report["execution"], "compiled-CM execution")
    _exact_keys(
        execution,
        {
            "parallel_jobs", "oracle_batch_timeout_milliseconds", "map_order"
        },
        "compiled-CM execution",
    )
    _require(
        1 <= _integer(execution["parallel_jobs"], "compiled-CM parallel jobs") <= 32
        and 1 <= _integer(
            execution["oracle_batch_timeout_milliseconds"],
            "compiled-CM timeout",
        ) <= 60_000
        and execution["map_order"] == "canonical-declaration-order",
        "compiled-CM execution contract differs",
    )
    checks = _mapping(report["checks"], "compiled-CM checks")
    _exact_keys(
        checks,
        {
            "compiled_spawn_origins_exact", "engine_spawn_link_lift_milliunits",
            "standing_and_crouched_stationary_hulls",
            "support_depth_milliunits", "oracle_swept_column_minimum_milliunits",
            "minimum_spawn_xy_separation_milliunits", "bounded_basic_escape",
            "basic_hazard_containment", "compiled_lightdata_presence",
            "all_to_all_reachability",
        },
        "compiled-CM checks",
    )
    _require(
        checks == {
            "compiled_spawn_origins_exact": True,
            "engine_spawn_link_lift_milliunits": 9000,
            "standing_and_crouched_stationary_hulls": True,
            "support_depth_milliunits": 96_000,
            "oracle_swept_column_minimum_milliunits": 96_000,
            "minimum_spawn_xy_separation_milliunits": 384_000,
            "bounded_basic_escape": True,
            "basic_hazard_containment": True,
            "compiled_lightdata_presence": True,
            "all_to_all_reachability": "deferred-to-full-Atlas-admission",
        },
        "compiled-CM declared check contract differs",
    )
    _require(
        report["input_stability"] == {
            "declaration": True,
            "compiled_membership": True,
            "implementation": True,
            "cm_oracle": True,
        },
        "compiled-CM input stability failed",
    )
    rows = _list(report["maps"], "compiled-CM map rows")
    _require(len(rows) == 28, "compiled-CM preflight does not contain 28 maps")
    for declared_row, raw_row in zip(declaration["maps"], rows):
        row = _mapping(raw_row, "compiled-CM map row")
        _exact_keys(
            row,
            {
                "ordinal", "map", "bsp", "source_spawn_origins_milliunits",
                "compiled_spawn_origins_milliunits", "spawn_origin_sets_match",
                "minimum_spawn_xy_separation_milliunits", "oracle",
                "spawn_count", "spawns", "compiled_lightdata",
                "basic_hazard_containment", "basic_escape_scope",
                "all_to_all_reachability", "failures", "passed",
            },
            "compiled-CM map row",
        )
        map_id = declared_row["map"]
        _require(
            row["ordinal"] == declared_row["ordinal"] and row["map"] == map_id,
            "compiled-CM map order or identity differs",
        )
        bsp_path = paths.compiled_dir / f"{map_id}.bsp"
        _require(row["bsp"] == _file_record(bsp_path), f"compiled-CM BSP differs for {map_id}")
        try:
            metadata = parse_ibsp38(bsp_path)
        except BspValidationError as exc:
            raise B2GateError(f"cannot parse compiled-CM BSP {map_id}: {exc}") from exc
        compiled_origins = {
            entity.index: _origin_milliunits(
                entity.value("origin"), f"compiled-CM entity {entity.index} {map_id}"
            )
            for entity in metadata.entities
            if entity.classname == "info_player_deathmatch"
        }
        source_origins = sorted([
            [round(float(axis) * 1000) for axis in origin]
            for origin in deathmatch_spawn_origins(
                paths.compiled_dir / f"{map_id}.map"
            )
        ])
        _require(
            len(compiled_origins) == 8
            and row["source_spawn_origins_milliunits"] == source_origins
            and row["compiled_spawn_origins_milliunits"]
            == sorted(compiled_origins.values())
            and row["spawn_origin_sets_match"] is True,
            f"compiled-CM spawn identity differs for {map_id}",
        )
        _require(
            _integer(
                row["minimum_spawn_xy_separation_milliunits"],
                f"compiled-CM spawn separation {map_id}",
            ) >= 384_000,
            f"compiled-CM spawn separation failed for {map_id}",
        )
        oracle = _mapping(row["oracle"], f"compiled-CM oracle {map_id}")
        _exact_keys(
            oracle,
            {"executable_sha256", "tool_identity", "physics_identity", "map_sha256"},
            f"compiled-CM oracle {map_id}",
        )
        _require(
            oracle["executable_sha256"] == binaries["cm"]
            and oracle["tool_identity"] == binaries["cm_tool_identity"]
            and oracle["map_sha256"] == row["bsp"]["sha256"],
            f"compiled-CM oracle binding differs for {map_id}",
        )
        _digest(oracle["physics_identity"], f"compiled-CM physics identity {map_id}")
        spawns = _list(row["spawns"], f"compiled-CM spawns {map_id}")
        _require(
            row["spawn_count"] == 8 and len(spawns) == 8,
            f"compiled-CM spawn count differs for {map_id}",
        )
        for spawn in spawns:
            _validate_preflight_spawn(spawn, compiled_origins, map_id)
        _require(
            len({spawn["entity_ordinal"] for spawn in spawns}) == 8,
            f"compiled-CM spawn rows are duplicated for {map_id}",
        )
        lighting_lump = next(
            (lump for lump in metadata.lumps if lump.name == "lighting"), None
        )
        _require(lighting_lump is not None, f"compiled-CM lighting lump is absent for {map_id}")
        bsp_payload = bsp_path.read_bytes()
        lightdata_payload = bsp_payload[
            lighting_lump.offset:lighting_lump.offset + lighting_lump.length
        ]
        lightdata = _mapping(
            row["compiled_lightdata"], f"compiled-CM lightdata {map_id}"
        )
        _exact_keys(
            lightdata, {"bytes", "sha256", "present"},
            f"compiled-CM lightdata {map_id}",
        )
        _require(
            lightdata == {
                "bytes": len(lightdata_payload),
                "sha256": _sha256_bytes(lightdata_payload),
                "present": True,
            }
            and len(lightdata_payload) == metadata.lightmaps.byte_count
            and len(lightdata_payload) > 0,
            f"compiled-CM lightdata check failed for {map_id}",
        )
        hazard = _mapping(
            row["basic_hazard_containment"],
            f"compiled-CM hazard containment {map_id}",
        )
        _exact_keys(
            hazard,
            {
                "declared_hazard_count", "checked_hazard_count", "hazards",
                "failures", "passed",
            },
            f"compiled-CM hazard containment {map_id}",
        )
        expected_hazards = _compiled_preflight_hazard_claims(
            paths.compiled_dir, map_id
        )
        evidence_rows = _list(
            hazard["hazards"], f"compiled-CM hazard evidence {map_id}"
        )
        _require(
            hazard["declared_hazard_count"] == len(expected_hazards)
            and hazard["checked_hazard_count"] == len(expected_hazards)
            and len(evidence_rows) == len(expected_hazards)
            and hazard["failures"] == [] and hazard["passed"] is True,
            f"compiled-CM hazard containment failed for {map_id}",
        )
        for expected_hazard, raw_hazard in zip(expected_hazards, evidence_rows):
            evidence = _mapping(
                raw_hazard, f"compiled-CM hazard evidence row {map_id}"
            )
            _exact_keys(
                evidence,
                {
                    "claim_id", "type", "bounds_milliunits", "probe_count",
                    "failures", "passed",
                },
                f"compiled-CM hazard evidence row {map_id}",
            )
            _require(
                {name: evidence[name] for name in (
                    "claim_id", "type", "bounds_milliunits"
                )} == expected_hazard
                and _integer(
                    evidence["probe_count"],
                    f"compiled-CM hazard probe count {map_id}",
                    minimum=1,
                ) >= 1
                and evidence["failures"] == []
                and evidence["passed"] is True,
                f"compiled-CM hazard evidence differs for {map_id}",
            )
        _require(
            row["basic_escape_scope"]
            == (
                "bounded straight standing-hull CM sweeps with 16-unit support "
                "samples; not an all-to-all Atlas reachability claim"
            )
            and row["all_to_all_reachability"] == "not-evaluated-by-preflight"
            and row["failures"] == []
            and row["passed"] is True,
            f"compiled-CM map retained failures for {map_id}",
        )
    _require(
        report["map_count"] == 28
        and report["pass_count"] == 28
        and report["failure_count"] == 0
        and report["failures"] == []
        and report["passed"] is True,
        "compiled-CM preflight is not 28/28 green",
    )
    canonical_record = dict(report)
    recorded_sha256 = canonical_record.pop("canonical_record_sha256")
    _require(
        recorded_sha256 == _sha256_bytes(canonical_bytes(canonical_record)),
        "compiled-CM canonical record digest differs",
    )
    return {
        "report": _file_record(paths.compiled_cm_preflight_report),
        "schema": COMPILED_CM_PREFLIGHT_SCHEMA,
        "stage": COMPILED_CM_PREFLIGHT_STAGE,
        "admission_status": COMPILED_CM_PREFLIGHT_STATUS,
        "compiled_membership_sha256": membership_identity["report_sha256"],
        "implementation_source_closure_sha256": expected_implementation[
            "source_closure_sha256"
        ],
        "map_count": 28,
        "pass_count": 28,
    }


def _validate_materialized(
    paths: B2GatePaths,
    declaration: Mapping[str, Any],
    declaration_sha256: str,
    binaries: Mapping[str, str],
    normative: Mapping[str, Any],
) -> dict[str, Any]:
    membership = _membership_with_declaration(
        declaration, declaration_sha256, paths.materialized_dir, "materialized"
    )
    _require_report_equals(
        paths.materialized_membership_report, membership, "materialized membership report"
    )
    immutable = [
        f"{row['map']}{suffix}"
        for row in declaration["maps"]
        for suffix in STAGE_SUFFIXES["compiled"]
        if suffix != ".json"
    ]
    _require_same_files(
        paths.compiled_dir, paths.materialized_dir, immutable,
        "materialized immutable input",
    )
    attestations = []
    for row in declaration["maps"]:
        map_id = row["map"]
        path = paths.materialized_dir / f"{map_id}.hook-materialization.json"
        try:
            raw_value, attestation_payload = _load_json_and_raw(path)
            value = validate_hook_materialization_v4(
                dict(_mapping(raw_value, "hook materialization"))
            )
        except HookClaimsV4Error as exc:
            raise B2GateError(
                f"invalid V4 materialization for {map_id}: {exc}"
            ) from exc
        _require(value["map"] == map_id, "materialization map differs")
        _require(
            len(value["selected_records"]) == 6,
            "materialization does not seal exactly six hooks",
        )
        oracles = _mapping(
            value["oracles"], f"materialization oracles {map_id}",
        )
        admitted_oracles = {
            "collision": binaries["cm"],
            "pmove": binaries["pmove"],
            "hook": binaries["hook"],
            "fall": binaries["fall"],
        }
        _require(
            all(
                _mapping(
                    oracles[name], f"materialization {name} oracle {map_id}",
                ).get("executable_sha256") == admitted_sha256
                for name, admitted_sha256 in admitted_oracles.items()
            ),
            f"materialization oracle differs from admitted B1 bytes for {map_id}",
        )
        _require(
            oracles.get("hook_parity_attestation_sha256")
            == binaries["hook_attestation"],
            f"materialization hook parity differs from admitted B1 bytes for {map_id}",
        )
        seal = _mapping(
            oracles["b1_runtime_authority_seal"],
            f"materialization retained B1 seal {map_id}",
        )
        _require(
            dict(_mapping(
                seal.get("executables"),
                f"materialization retained B1 executables {map_id}",
            )) == {
                "cm_sha256": binaries["cm"],
                "pmove_sha256": binaries["pmove"],
                "hook_sha256": binaries["hook"],
                "fall_sha256": binaries["fall"],
            },
            f"materialization retained B1 executables differ for {map_id}",
        )
        _require(
            seal.get("hook_parity_attestation_sha256")
            == binaries["hook_attestation"],
            f"materialization retained B1 hook parity differs for {map_id}",
        )
        _require(
            dict(_mapping(
                seal.get("normative_documents"),
                f"materialization retained B1 normative documents {map_id}",
            )) == {
                "design_sha256": normative["design"]["sha256"],
                "plan_sha256": normative["plan"]["sha256"],
            },
            f"materialization retained B1 normative documents differ for {map_id}",
        )

        compiled_runtime = paths.compiled_dir / f"{map_id}.json"
        materialized_runtime = paths.materialized_dir / f"{map_id}.json"
        compiled_runtime_payload = compiled_runtime.read_bytes()
        materialized_runtime_payload = materialized_runtime.read_bytes()
        compiled_runtime_sha256 = _sha256_bytes(compiled_runtime_payload)
        _require(
            value["source_projection_sha256"] == compiled_runtime_sha256,
            f"materialization source projection differs for {map_id}",
        )
        _require(
            materialized_runtime_payload != compiled_runtime_payload,
            f"materialization runtime sidecar was not upgraded for {map_id}",
        )

        materialized_bsp = paths.materialized_dir / f"{map_id}.bsp"
        materialized_bsp_payload = materialized_bsp.read_bytes()
        materialized_bsp_sha256 = _sha256_bytes(materialized_bsp_payload)
        _require(
            value["bsp"] == {
                "sha256": materialized_bsp_sha256,
                "size_bytes": len(materialized_bsp_payload),
            },
            f"materialization BSP binding differs for {map_id}",
        )
        materialization_sha256 = _sha256_bytes(attestation_payload)
        try:
            validate_runtime_sidecar(
                materialized_runtime_payload,
                map_id=map_id,
                bsp_sha256=materialized_bsp_sha256,
                materialization_sha256=materialization_sha256,
                records=value["selected_records"],
            )
        except HookClaimsV4Error as exc:
            raise B2GateError(
                f"invalid V4 runtime sidecar for {map_id}: {exc}"
            ) from exc
        attestations.append(materialization_sha256)
    return {
        "membership": _file_record(paths.materialized_membership_report),
        "attestation_set_sha256": _sha256_bytes(canonical_bytes(attestations)),
        "map_count": 28,
    }


def _validate_prepare_report(
    report: Mapping[str, Any], declaration: Mapping[str, Any], declaration_sha256: str,
    materialized_membership: Mapping[str, Any], claims_membership: Mapping[str, Any],
) -> None:
    _exact_keys(
        report,
        {
            "schema", "cohort_id", "declaration_sha256", "phase",
            "expected_count", "map_count", "input_stages", "pass_count",
            "passed", "maps", "failures",
        },
        "claims prepare report",
    )
    _require(report.get("schema") == CAMPAIGN_SCHEMA, "claims prepare schema differs")
    _require(report.get("phase") == "prepare_claims", "claims report phase differs")
    _require(report.get("cohort_id") == EXPECTED_COHORT, "claims report cohort differs")
    _require(report.get("declaration_sha256") == declaration_sha256, "claims report declaration differs")
    _require(report.get("expected_count") == 28 and report.get("map_count") == 28, "claims report count differs")
    _require(report.get("pass_count") == 28 and report.get("passed") is True, "claims report is not green")
    _require(report.get("failures") == [], "claims report retains failures")
    stages = _mapping(report.get("input_stages"), "claims report stages")
    _exact_keys(stages, {"materialized", "claims"}, "claims report stages")
    for stage, full in (("materialized", materialized_membership), ("claims", claims_membership)):
        projection = _mapping(stages.get(stage), f"claims report {stage} stage")
        expected = {
            "stage": full["stage"], "passed": full["passed"],
            "expected_map_count": full["expected_map_count"],
            "expected_file_count": full["expected_file_count"],
            "actual_file_count": full["actual_file_count"],
            "report_sha256": _sha256_bytes(canonical_bytes({
                key: value for key, value in full.items() if key != "declaration_sha256"
            })),
        }
        _require(dict(projection) == expected, f"claims report {stage} membership is stale")
    rows = _list(report.get("maps"), "claims report maps")
    _require(len(rows) == 28, "claims report rows differ")
    for declared, row in zip(declaration["maps"], rows):
        row = _mapping(row, f"claims row {declared['map']}")
        _exact_keys(
            row,
            {"ordinal", "map", "passed", "generator_claims_sha256", "error"},
            f"claims row {declared['map']}",
        )
        _require(
            row.get("ordinal") == declared["ordinal"] and row.get("map") == declared["map"]
            and row.get("passed") is True and row.get("error") is None,
            f"claims row is not green for {declared['map']}",
        )


def _validate_claims(
    paths: B2GatePaths,
    declaration: Mapping[str, Any],
    declaration_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    materialized = _membership_with_declaration(
        declaration, declaration_sha256, paths.materialized_dir, "materialized"
    )
    claims = _membership_with_declaration(
        declaration, declaration_sha256, paths.claims_dir, "claims"
    )
    common = [
        f"{row['map']}{suffix}"
        for row in declaration["maps"] for suffix in STAGE_SUFFIXES["materialized"]
    ]
    _require_same_files(paths.materialized_dir, paths.claims_dir, common, "claims input")
    report = _mapping(_load_json(paths.claims_prepare_report), "claims prepare report")
    _validate_prepare_report(report, declaration, declaration_sha256, materialized, claims)
    digests = []
    for declared, reported in zip(declaration["maps"], report["maps"]):
        claim_path = paths.claims_dir / f"{declared['map']}.generator-claims.json"
        expected = build_generator_claims(paths.claims_dir / f"{declared['map']}.map")
        _require(_load_json(claim_path) == expected, f"canonical claims differ for {declared['map']}")
        digest = _file_sha256(claim_path)
        _require(reported["generator_claims_sha256"] == digest, "claims report digest differs")
        digests.append(digest)
    return {
        "prepare_report": _file_record(paths.claims_prepare_report),
        "claims_membership_sha256": _sha256_bytes(canonical_bytes(claims)),
        "claims_set_sha256": _sha256_bytes(canonical_bytes(digests)),
        "map_count": 28,
    }, claims


def _validate_generated_build_report(
    path: Path,
    declaration: Mapping[str, Any],
    declaration_sha256: str,
    implementation: Mapping[str, Any],
    claims_membership: Mapping[str, Any],
    analysis_membership: Mapping[str, Any],
    claims_dir: Path,
    analysis_dir: Path,
) -> Mapping[str, Any]:
    """Validate the generated Atlas campaign producer's frozen public contract."""
    from tools.run_generated_atlas_campaign import (
        GeneratedAtlasCampaignError,
        _load_analysis_binding,
    )

    report = _mapping(_load_json(path), "generated Atlas build report")
    _exact_keys(
        report,
        {
            "schema", "cohort_id", "declaration_sha256", "implementation",
            "claims_snapshot_sha256", "input_claims", "output_analysis",
            "expected_map_count", "pass_count", "published", "passed", "maps",
            "failures",
        },
        "generated Atlas build report",
    )
    _require(
        report.get("schema") == "q2-generated-atlas-build-campaign-v1",
        "generated Atlas build report schema differs",
    )
    _require(report.get("cohort_id") == EXPECTED_COHORT, "generated build cohort differs")
    _require(report.get("declaration_sha256") == declaration_sha256, "generated build declaration differs")
    _require(report.get("implementation") == implementation, "generated build implementation is stale")
    claims_snapshot = {
        key: value for key, value in claims_membership.items()
        if key != "declaration_sha256"
    }
    expected_claims_snapshot_sha256 = _sha256_bytes(
        canonical_bytes(claims_snapshot)
    )
    _require(
        report.get("claims_snapshot_sha256") == expected_claims_snapshot_sha256,
        "generated build immutable claims snapshot differs",
    )
    _require(report.get("expected_map_count") == 28 and report.get("pass_count") == 28, "generated build count differs")
    _require(report.get("published") is True, "generated analysis stage was not atomically published")
    _require(report.get("passed") is True and report.get("failures") == [], "generated build is not green")
    for field, full in (
        ("input_claims", claims_membership),
        ("output_analysis", analysis_membership),
    ):
        expected = {
            "stage": full["stage"],
            "passed": full["passed"],
            "expected_map_count": full["expected_map_count"],
            "expected_file_count": full["expected_file_count"],
            "actual_file_count": full["actual_file_count"],
            "report_sha256": _sha256_bytes(canonical_bytes({
                key: value for key, value in full.items()
                if key != "declaration_sha256"
            })),
        }
        _require(report.get(field) == expected, f"generated build {field} is stale")
    rows = _list(report.get("maps"), "generated build maps")
    _require(len(rows) == 28, "generated build rows differ")
    for declared, row in zip(declaration["maps"], rows):
        row = _mapping(row, f"generated build {declared['map']}")
        _exact_keys(
            row,
            {
                "ordinal", "map", "passed", "bsp_sha256",
                "generator_claims_sha256", "artifacts", "analysis_manifest",
                "build_summary", "build_summary_sha256", "stdout", "stderr",
                "error",
            },
            f"generated build {declared['map']}",
        )
        _require(row.get("ordinal") == declared["ordinal"] and row.get("map") == declared["map"], "generated build order differs")
        _require(row.get("passed") is True and row.get("error") is None, f"generated build failed for {declared['map']}")
        _require(
            row.get("bsp_sha256")
            == _file_sha256(claims_dir / f"{declared['map']}.bsp"),
            f"generated build BSP differs for {declared['map']}",
        )
        _require(
            row.get("generator_claims_sha256")
            == _file_sha256(
                claims_dir / f"{declared['map']}.generator-claims.json"
            ),
            f"generated build claims differ for {declared['map']}",
        )
        artifacts = _mapping(row.get("artifacts"), "generated build artifacts")
        _require(set(artifacts) == set(STOCK_ANALYSIS_SUFFIXES), "generated build artifact set differs")
        for suffix in STOCK_ANALYSIS_SUFFIXES:
            _require(
                artifacts.get(suffix) == _file_record(analysis_dir / f"{declared['map']}{suffix}"),
                f"generated build artifact differs for {declared['map']}{suffix}",
            )
        try:
            expected_analysis_binding = _load_analysis_binding(
                analysis_dir, declared, claims_dir, implementation
            )
        except GeneratedAtlasCampaignError as exc:
            raise B2GateError(
                f"generated analysis binding failed for {declared['map']}: {exc}"
            ) from exc
        _require(
            row.get("analysis_manifest") == expected_analysis_binding,
            f"generated build analysis manifest binding differs for {declared['map']}",
        )
        summary = _mapping(row.get("build_summary"), "generated build summary")
        _exact_keys(summary, {"schema", "maps"}, "generated build summary")
        _require(summary["schema"] == "q2-atlas-generated-build-v1", "generated summary schema differs")
        summary_rows = _list(summary["maps"], "generated summary maps")
        _require(len(summary_rows) == 1, "generated summary does not name one map")
        expected_summary = {
            "canonical_map_id": declared["map"],
            "bsp_sha256": row["bsp_sha256"],
            "atlas_sha256": _load_json(
                analysis_dir / f"{declared['map']}.analysis.manifest.json"
            )["identity"]["atlas_sha256"],
            "manifest_sha256": _file_sha256(
                analysis_dir / f"{declared['map']}.analysis.manifest.json"
            ),
        }
        _require(summary_rows[0] == expected_summary, "generated build summary is stale")
        _require(
            row.get("build_summary_sha256")
            == _sha256_bytes(canonical_bytes(summary)),
            "generated build summary digest differs",
        )
        for stream in ("stdout", "stderr"):
            stream_record = _mapping(row.get(stream), f"generated build {stream}")
            _exact_keys(stream_record, {"bytes", "sha256"}, f"generated build {stream}")
            _integer(stream_record["bytes"], f"generated build {stream} bytes")
            _digest(stream_record["sha256"], f"generated build {stream} digest")
    return report


def _validate_generated(
    paths: B2GatePaths,
    declaration: Mapping[str, Any],
    declaration_sha256: str,
    implementation: Mapping[str, Any],
    claims_membership: Mapping[str, Any],
    binaries: Mapping[str, str],
) -> dict[str, Any]:
    analysis = _membership_with_declaration(
        declaration, declaration_sha256, paths.analysis_dir, "analysis"
    )
    _validate_generated_build_report(
        paths.generated_build_report, declaration, declaration_sha256,
        implementation, claims_membership, analysis, paths.claims_dir,
        paths.analysis_dir,
    )
    recomputed = validate_campaign(
        paths.declaration, paths.claims_dir, paths.analysis_dir, paths.b1_gate
    )
    _require(recomputed["passed"] is True and recomputed["pass_count"] == 28, "generated validation is not green")
    _require_report_equals(
        paths.generated_validation_report, recomputed, "generated validation report"
    )
    atlas_hashes = []
    for declared in declaration["maps"]:
        manifest_path = paths.analysis_dir / f"{declared['map']}.analysis.manifest.json"
        manifest = _mapping(_load_json(manifest_path), "generated analysis manifest")
        _validate_analysis_runtime_bindings(manifest, implementation, binaries)
        atlas_hashes.append(_digest(
            _mapping(manifest.get("identity"), "analysis identity").get("atlas_sha256"),
            "generated Atlas digest",
        ))
    _require(len(set(atlas_hashes)) == 28, "generated Atlas hashes are not unique")
    return {
        "build_report": _file_record(paths.generated_build_report),
        "validation_report": _file_record(paths.generated_validation_report),
        "analysis_membership_sha256": _sha256_bytes(canonical_bytes(analysis)),
        "atlas_set_sha256": _sha256_bytes(canonical_bytes(atlas_hashes)),
        "map_count": 28,
    }


def _validate_analysis_runtime_bindings(
    manifest: Mapping[str, Any], implementation: Mapping[str, Any], binaries: Mapping[str, str]
) -> None:
    identity = _mapping(manifest.get("identity"), "analysis identity")
    _require(
        identity.get("analyzer_sha256") == implementation["atlas_analyzer_authority_sha256"],
        "analysis uses a stale analyzer closure",
    )
    oracles = _mapping(manifest.get("oracles"), "analysis oracles")
    _require(_mapping(oracles.get("collision"), "collision oracle").get("executable_sha256") == binaries["cm"], "analysis CM binary differs")
    _require(_mapping(oracles.get("pmove"), "Pmove oracle").get("executable_sha256") == binaries["pmove"], "analysis Pmove binary differs")
    _require(_mapping(oracles.get("fall"), "fall oracle").get("executable_sha256") == binaries["fall"], "analysis fall binary differs")
    hook = _mapping(oracles.get("hook"), "hook oracle")
    _require(hook.get("binary_sha256") == binaries["hook"], "analysis hook binary differs")
    _require(hook.get("attestation_sha256") == binaries["hook_attestation"], "analysis hook attestation differs")
    artifacts = _mapping(manifest.get("artifacts"), "analysis artifacts")
    atlas_manifest = _mapping(artifacts.get("atlas_manifest"), "analysis Atlas manifest")
    _require(atlas_manifest.get("verifier_sha256") == binaries["atlas_verifier"], "analysis verifier binary differs")


def _validate_stock(
    paths: B2GatePaths,
    implementation: Mapping[str, Any],
    binaries: Mapping[str, str],
) -> dict[str, Any]:
    provenance = _mapping(_load_json(paths.stock_provenance), "stock provenance")
    inventory = _mapping(_load_json(paths.stock_inventory), "stock inventory")
    _require(provenance.get("schema") == "q2-corpus-provenance-v1", "stock provenance schema differs")
    _require(inventory.get("schema") == "q2-stock-map-fixtures-v1", "stock inventory schema differs")
    _require(
        [row.get("canonical_id") for row in provenance.get("records", [])] == list(STOCK_IDS),
        "stock provenance is not exactly q2dm1-q2dm8",
    )
    _require(
        [row.get("canonical_id") for row in inventory.get("maps", [])] == list(STOCK_IDS),
        "stock inventory is not exactly q2dm1-q2dm8",
    )
    _exact_directory_files(
        paths.stock_bsp_dir, {f"{map_id}.bsp" for map_id in STOCK_IDS}, "stock BSP root"
    )
    _exact_directory_files(
        paths.stock_analysis_dir,
        {f"{map_id}{suffix}" for map_id in STOCK_IDS for suffix in STOCK_ANALYSIS_SUFFIXES},
        "stock analysis root",
    )
    _exact_directory_files(
        paths.stock_validation_dir,
        {f"{map_id}.stock-validation.json" for map_id in STOCK_IDS},
        "stock validation root",
    )
    atlas_hashes = []
    validation_hashes = []
    cold_elapsed = []
    for map_id in STOCK_IDS:
        bsp = paths.stock_bsp_dir / f"{map_id}.bsp"
        manifest_path = paths.stock_analysis_dir / f"{map_id}.analysis.manifest.json"
        manifest = _mapping(_load_json(manifest_path), f"{map_id} analysis")
        _validate_analysis_runtime_bindings(manifest, implementation, binaries)
        recomputed = validate_stock_analysis(
            bsp, manifest_path, b1_gate_path=paths.b1_gate,
            stock_provenance_path=paths.stock_provenance,
            stock_inventory_path=paths.stock_inventory,
        )
        validate_claim_report(recomputed)
        _require(recomputed["passed"] is True, f"stock validation failed for {map_id}")
        report_path = paths.stock_validation_dir / f"{map_id}.stock-validation.json"
        _require_report_equals(report_path, recomputed, f"{map_id} stock validation")
        validation_hashes.append(_file_sha256(report_path))
        identity = _mapping(manifest.get("identity"), f"{map_id} identity")
        atlas_hashes.append(_digest(identity.get("atlas_sha256"), f"{map_id} Atlas digest"))
        proof = _mapping(
            _mapping(manifest.get("performance"), f"{map_id} performance").get("full_cold_rebuild"),
            f"{map_id} cold evidence",
        )
        _require(proof.get("artifact_sha256") == proof.get("cold_artifact_sha256"), f"{map_id} cold byte digests differ")
        _require(proof.get("artifact_semantic_sha256") == proof.get("cold_artifact_semantic_sha256"), f"{map_id} cold semantic digests differ")
        cold_elapsed.append(_integer(proof.get("elapsed_milliseconds"), f"{map_id} cold elapsed", minimum=1))
    return {
        "provenance": _file_record(paths.stock_provenance),
        "inventory": _file_record(paths.stock_inventory),
        "atlas_set_sha256": _sha256_bytes(canonical_bytes(atlas_hashes)),
        "validation_set_sha256": _sha256_bytes(canonical_bytes(validation_hashes)),
        "cold_elapsed_milliseconds_max": max(cold_elapsed),
        "map_count": 8,
    }


def _file_evidence_matches(value: object, expected: Path, label: str) -> None:
    record = _mapping(value, label)
    _exact_keys(record, {"path", "sha256", "size_bytes"}, label)
    _require(isinstance(record["path"], str) and record["path"], f"{label} path is missing")
    actual = _file_record(expected)
    _require(
        record["sha256"] == actual["sha256"]
        and record["size_bytes"] == actual["bytes"],
        f"{label} bytes differ from admitted campaign",
    )


def _atlas_l0_bytes(path: Path) -> int:
    with path.open("rb") as stream:
        header = stream.read(136)
    if len(header) != 136 or header[:8] != b"Q2ATL001":
        raise B2GateError("representative Atlas header differs")
    schema, byte_order, header_bytes = struct.unpack_from("<HHI", header, 8)
    if (schema, byte_order, header_bytes) != (1, 0x454C, 136):
        raise B2GateError("representative Atlas schema/byte order differs")
    lengths = struct.unpack_from("<5Q", header, 96)
    if 136 + sum(lengths) != path.stat().st_size:
        raise B2GateError("representative Atlas section lengths differ")
    return int(lengths[0])


def _compact_json_bytes(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def _rust_source_paths(directory: Path) -> list[Path]:
    if not directory.is_dir() or directory.is_symlink():
        raise B2GateError(f"Dyn source directory is missing or a symlink: {directory}")
    paths = sorted(directory.rglob("*.rs"))
    if any(path.is_symlink() or not path.is_file() for path in paths):
        raise B2GateError(f"Dyn source closure contains a symlink: {directory}")
    return paths


def _dyn_source_closure(
    repo_root: Path, relative_paths: Sequence[str], repo_commit: str, label: str
) -> dict[str, Any]:
    inputs = []
    for relative in sorted(relative_paths):
        path = repo_root / relative
        record = _file_record(path)
        inputs.append({
            "path": relative,
            "sha256": record["sha256"],
            "size_bytes": record["bytes"],
        })
    _require(inputs, f"{label} source closure is empty")
    hash_records = [
        {"path": item["path"], "sha256": item["sha256"]} for item in inputs
    ]
    closure_sha256 = _sha256_bytes(_compact_json_bytes(hash_records))
    commit_bound_sha256 = _sha256_bytes(_compact_json_bytes({
        "repo_commit": repo_commit,
        "source_closure_sha256": closure_sha256,
    }))
    return {
        "algorithm": "sha256(canonical-json([{path,sha256},...]))-v1",
        "sha256": closure_sha256,
        "embedded_sha256": closure_sha256,
        "repo_commit": repo_commit,
        "commit_binding_algorithm": (
            "sha256(canonical-json({repo_commit,source_closure_sha256}))-v1"
        ),
        "commit_bound_sha256": commit_bound_sha256,
        "inputs": inputs,
    }


def _dyn_source_authority(repo_root: Path, repo_commit: str) -> dict[str, Any]:
    helper_base = repo_root / "tools/q2-dyn-evidence"
    lattice_base = repo_root / "crates/q2-lattice"
    helper_paths = [
        helper_base / "Cargo.lock",
        helper_base / "Cargo.toml",
        helper_base / "README.md",
        helper_base / "build.rs",
        *_rust_source_paths(helper_base / "src"),
    ]
    lattice_paths = [
        lattice_base / "Cargo.toml",
        *_rust_source_paths(lattice_base / "src"),
    ]
    return {
        "helper_source_closure": _dyn_source_closure(
            repo_root,
            [path.relative_to(repo_root).as_posix() for path in helper_paths],
            repo_commit,
            "Dyn helper",
        ),
        "q2_lattice_source_closure": _dyn_source_closure(
            repo_root,
            [path.relative_to(repo_root).as_posix() for path in lattice_paths],
            repo_commit,
            "q2-lattice",
        ),
    }


def _f32_add(left: float, right: float) -> float:
    return struct.unpack("<f", struct.pack("<f", left + right))[0]


def _decode_dyn_cells(
    payload: bytes, start: int, count: int, label: str
) -> tuple[list[tuple[tuple[int, int, int], tuple[float, ...]]], int]:
    cells = []
    prior = None
    offset = start
    for _ordinal in range(count):
        if offset + 40 > len(payload):
            raise B2GateError(f"{label} Dyn cell payload is truncated")
        x, y, z, *values = struct.unpack_from("<iii7f", payload, offset)
        index = (x, y, z)
        order = (z, y, x)
        if prior is not None and order <= prior:
            raise B2GateError(f"{label} Dyn cells are not strictly ordered (iz,iy,ix)")
        prior = order
        for value_index, value in enumerate(values):
            if not float(value) >= 0.0 or not float(value) < float("inf"):
                raise B2GateError(f"{label} Dyn cell value {value_index} is invalid")
            if value == 0.0 and struct.pack("<f", value) != b"\0\0\0\0":
                raise B2GateError(f"{label} Dyn cell contains noncanonical negative zero")
        if values[6] > 1.0:
            raise B2GateError(f"{label} Dyn confidence is above one")
        cells.append((index, tuple(values)))
        offset += 40
    return cells, offset


def _validate_derived_l3(
    l2: Sequence[tuple[tuple[int, int, int], tuple[float, ...]]],
    l3: Sequence[tuple[tuple[int, int, int], tuple[float, ...]]],
) -> None:
    aggregate: dict[tuple[int, int, int], list[float]] = {}
    for (x, y, z), values in l2:
        parent = (x // 4, y // 4, z // 4)
        current = aggregate.setdefault(parent, [0.0] * 7)
        for index in range(6):
            current[index] = _f32_add(current[index], values[index])
            if not current[index] < float("inf"):
                raise B2GateError("Dyn derived L3 aggregate is non-finite")
        current[6] = max(current[6], values[6])
    expected = sorted(aggregate.items(), key=lambda item: (
        item[0][2], item[0][1], item[0][0]
    ))
    if len(expected) != len(l3):
        raise B2GateError("Dyn L3 count is not the canonical derived mip")
    for (expected_index, expected_values), (actual_index, actual_values) in zip(expected, l3):
        if expected_index != actual_index or any(
            struct.pack("<f", expected_value) != struct.pack("<f", actual_value)
            for expected_value, actual_value in zip(expected_values, actual_values)
        ):
            raise B2GateError("Dyn L3 is not the canonical derived mip of L2")


def _decode_dyn_snapshot(
    payload: bytes,
    *,
    atlas_sha256: str,
    map_sha256: str,
    origin: Sequence[int],
    map_epoch: int,
    environment_steps: int,
    client_count: int,
) -> dict[str, int]:
    if zstandard is None:
        raise B2GateError("zstandard Python support is required for independent Dyn decoding")
    if len(payload) < 208 or len(payload) > 2 * 1024 * 1024:
        raise B2GateError("Dyn snapshot length is outside the complete Q2LAT002 envelope")
    if payload[:8] != b"Q2LAT002":
        raise B2GateError("Dyn snapshot magic differs")
    schema, byte_order, header_bytes = struct.unpack_from("<HHI", payload, 8)
    compression, level, reserved16, reserved32 = struct.unpack_from("<BbHI", payload, 16)
    uncompressed_len, compressed_len = struct.unpack_from("<QQ", payload, 24)
    if (schema, byte_order, header_bytes) != (2, 0x454C, 208):
        raise B2GateError("Dyn snapshot schema/byte order/header differs")
    if (compression, level, reserved16, reserved32) != (1, 3, 0, 0):
        raise B2GateError("Dyn snapshot compression/reserved fields differ")
    if compressed_len != len(payload) - 208 or uncompressed_len > 2 * 1024 * 1024:
        raise B2GateError("Dyn compressed/uncompressed length fence differs")
    expected_atlas = bytes.fromhex(atlas_sha256)
    expected_map = bytes.fromhex(map_sha256)
    if payload[72:104] != expected_atlas or payload[104:136] != expected_map:
        raise B2GateError("Dyn snapshot Atlas/BSP digest fence differs")
    snapshot_origin = list(struct.unpack_from("<qqq", payload, 136))
    if snapshot_origin != list(origin) or any(value % 256 for value in snapshot_origin):
        raise B2GateError("Dyn snapshot origin fence differs")
    l2_size, l3_size = struct.unpack_from("<II", payload, 160)
    found_epoch, found_steps = struct.unpack_from("<QQ", payload, 168)
    client_id, found_clients = struct.unpack_from("<II", payload, 184)
    l2_count, l3_count = struct.unpack_from("<QQ", payload, 192)
    if (l2_size, l3_size) != (64, 256):
        raise B2GateError("Dyn snapshot cell-size fence differs")
    if found_epoch != map_epoch or found_steps != environment_steps:
        raise B2GateError("Dyn snapshot epoch/environment-step fence differs")
    if found_clients != client_count or client_id >= client_count:
        raise B2GateError("Dyn snapshot client identity fence differs")
    if l2_count > 20_000 or l3_count > 20_000 or l2_count + l3_count > 20_000:
        raise B2GateError("Dyn snapshot materialized-cell limit differs")
    if uncompressed_len != (l2_count + l3_count) * 40:
        raise B2GateError("Dyn snapshot cell counts do not cover its payload")
    try:
        decoded = zstandard.ZstdDecompressor().decompress(
            payload[208:], max_output_size=2 * 1024 * 1024 + 1
        )
    except zstandard.ZstdError as exc:
        raise B2GateError(f"Dyn zstd payload is invalid: {exc}") from exc
    if len(decoded) != uncompressed_len:
        raise B2GateError("Dyn uncompressed payload length differs")
    if hashlib.sha256(decoded).digest() != payload[40:72]:
        raise B2GateError("Dyn uncompressed payload digest differs")
    l2, offset = _decode_dyn_cells(decoded, 0, int(l2_count), "L2")
    l3, offset = _decode_dyn_cells(decoded, offset, int(l3_count), "L3")
    if offset != len(decoded):
        raise B2GateError("Dyn cell decoder did not consume the exact payload")
    _validate_derived_l3(l2, l3)
    reencoded_payload = b"".join(
        struct.pack("<iii7f", *index, *values)
        for index, values in (*l2, *l3)
    )
    if reencoded_payload != decoded:
        raise B2GateError("Dyn semantic decode/re-encode is not byte-identical")
    recompressed = zstandard.ZstdCompressor(level=3).compress(reencoded_payload)
    if recompressed != payload[208:]:
        raise B2GateError("Dyn canonical zstd re-encode is not byte-identical")
    # x86_64 Rust layout: 96-byte DynFence, identity/step fields, and two
    # 24-byte BTreeMap handles. This is separately fenced by WSL x86_64.
    resident_bytes = 160 + int(l2_count + l3_count) * 96
    if resident_bytes > 2 * 1024 * 1024 - 1:
        raise B2GateError("Dyn per-client resident limit differs")
    return {
        "client_id": client_id,
        "l2_cells": int(l2_count),
        "l3_cells": int(l3_count),
        "resident_bytes": resident_bytes,
    }


def _latency_distribution(value: object, label: str) -> Mapping[str, Any]:
    result = _mapping(value, label)
    _exact_keys(result, {"p50_ns", "p95_ns", "p99_ns", "max_ns"}, label)
    values = [_integer(result[name], f"{label} {name}") for name in (
        "p50_ns", "p95_ns", "p99_ns", "max_ns"
    )]
    _require(values == sorted(values), f"{label} percentiles are not monotonic")
    return result


def _validate_dyn_evidence(
    path: Path,
    implementation: Mapping[str, Any],
    declaration: Mapping[str, Any],
    paths: B2GatePaths,
) -> tuple[dict[str, Any], dict[str, Any]]:
    report = _mapping(_load_json(path), "Dyn evidence report")
    _exact_keys(
        report,
        {
            "schema", "passed", "authority", "provenance", "host", "atlas",
            "dyn_state", "negative_fences_and_limits", "performance",
        },
        "Dyn evidence report",
    )
    _require(report["schema"] == "q2-b2-dyn-evidence-v1", "Dyn evidence schema differs")
    _require(report["passed"] is True, "Dyn evidence is not green")
    authority = _mapping(report["authority"], "Dyn authority")
    _exact_keys(
        authority,
        {
            "specification_sha256", "analyzer_name", "analyzer_version",
            "analyzer_authority_sha256", "crate_commit", "executable_sha256",
            "canonical_map_id", "map_epoch", "environment_steps",
        },
        "Dyn authority",
    )
    _require(authority["specification_sha256"] == EXPECTED_DESIGN_SHA256, "Dyn design authority differs")
    _require(authority["analyzer_name"] == "q2-atlas-analyzer", "Dyn analyzer name differs")
    _require(authority["analyzer_authority_sha256"] == implementation["atlas_analyzer_authority_sha256"], "Dyn analyzer authority is stale")
    _require(authority["crate_commit"] == implementation["repository_commit"], "Dyn crate commit is stale")
    _digest(authority["executable_sha256"], "Dyn evidence executable")
    executable = _file_record(paths.dyn_evidence_executable)
    _require(
        authority["executable_sha256"] == executable["sha256"],
        "Dyn evidence executable bytes differ from the pinned producer",
    )
    source_authority = _dyn_source_authority(
        paths.repo_root, implementation["repository_commit"]
    )
    provenance = _mapping(report["provenance"], "Dyn provenance")
    _exact_keys(
        provenance,
        {
            "embedded_repo_commit", "executable", "helper_source_closure",
            "q2_lattice_source_closure",
        },
        "Dyn provenance",
    )
    _require(
        provenance["embedded_repo_commit"] == implementation["repository_commit"],
        "Dyn embedded repository commit is stale",
    )
    _file_evidence_matches(
        provenance["executable"], paths.dyn_evidence_executable,
        "Dyn provenance executable",
    )
    _require(
        provenance["executable"]["sha256"] == authority["executable_sha256"],
        "Dyn authority and provenance executable digests differ",
    )
    for field in ("helper_source_closure", "q2_lattice_source_closure"):
        reported_closure = _mapping(provenance[field], f"Dyn provenance {field}")
        expected_closure = source_authority[field]
        _exact_keys(
            reported_closure,
            {
                "algorithm", "sha256", "embedded_sha256", "repo_commit",
                "commit_binding_algorithm", "commit_bound_sha256", "inputs",
            },
            f"Dyn provenance {field}",
        )
        _require(
            dict(reported_closure) == expected_closure,
            f"Dyn provenance {field} differs from the current source closure",
        )
    _integer(authority["map_epoch"], "Dyn map epoch", minimum=1)
    _integer(authority["environment_steps"], "Dyn environment steps")
    map_id = authority["canonical_map_id"]
    _require(
        map_id in {row["map"] for row in declaration["maps"]},
        "Dyn map is outside cohort 71444",
    )

    host = _mapping(report["host"], "Dyn host")
    _exact_keys(host, {"hostname", "kernel_release", "architecture"}, "Dyn host")
    _require(host["hostname"] == "DESKTOP-RTX2080", "Dyn evidence host is not DESKTOP-RTX2080")
    _require(
        isinstance(host["kernel_release"], str)
        and "microsoft-standard-WSL2" in host["kernel_release"],
        "Dyn evidence kernel is not WSL2",
    )
    _require(host["architecture"] == "x86_64", "Dyn evidence architecture differs")

    analysis_path = paths.analysis_dir / f"{map_id}.analysis.manifest.json"
    atlas_manifest_path = paths.analysis_dir / f"{map_id}.atlas.manifest.json"
    atlas_path = paths.analysis_dir / f"{map_id}.atlas.bin"
    bsp_path = paths.claims_dir / f"{map_id}.bsp"
    analysis = _mapping(_load_json(analysis_path), "representative analysis")
    atlas_manifest = _mapping(
        _load_json(atlas_manifest_path), "representative Atlas manifest"
    )
    manifest_bsp = _mapping(atlas_manifest.get("bsp"), "representative manifest BSP")
    manifest_analyzer = _mapping(
        atlas_manifest.get("analyzer"), "representative manifest analyzer"
    )
    _require(
        atlas_manifest.get("specification_sha256") == EXPECTED_DESIGN_SHA256,
        "representative Atlas manifest design authority differs",
    )
    _require(
        manifest_bsp.get("canonical_map_id") == map_id
        and manifest_bsp.get("sha256") == _file_sha256(bsp_path),
        "representative Atlas manifest BSP authority differs",
    )
    _require(
        manifest_analyzer.get("name") == authority["analyzer_name"]
        and manifest_analyzer.get("version") == authority["analyzer_version"]
        and manifest_analyzer.get("sha256")
        == authority["analyzer_authority_sha256"],
        "Dyn analyzer fields differ from the admitted Atlas manifest",
    )
    atlas = _mapping(report["atlas"], "Dyn Atlas evidence")
    _exact_keys(
        atlas,
        {
            "manifest", "artifact", "bsp", "origin", "counts",
            "resident_bytes", "representative_l2_cells", "lookup",
        },
        "Dyn Atlas evidence",
    )
    _file_evidence_matches(atlas["manifest"], atlas_manifest_path, "Dyn Atlas manifest")
    _file_evidence_matches(atlas["artifact"], atlas_path, "Dyn Atlas artifact")
    _file_evidence_matches(atlas["bsp"], bsp_path, "Dyn BSP")
    grid = _mapping(analysis["grid"], "representative grid")
    _require(atlas["origin"] == grid["origin"], "Dyn Atlas origin differs")
    counts = _mapping(atlas["counts"], "Dyn Atlas counts")
    expected_counts = {
        name: analysis["counts"][name]
        for name in ("l0_chunks", "l1_nodes", "l1_edges", "l2_cells", "l3_cells")
    }
    _require(dict(counts) == expected_counts, "Dyn Atlas counts differ")
    atlas_artifact = _mapping(analysis["artifacts"]["atlas"], "representative Atlas artifact")
    l0_bytes = _atlas_l0_bytes(atlas_path)
    resident_bytes = _integer(atlas["resident_bytes"], "Dyn Atlas resident bytes")
    _require(counts["l0_chunks"] <= MAX_L0_CHUNKS, "representative L0 chunks exceed 1200")
    _require(l0_bytes <= MAX_L0_BYTES, "representative L0 bytes exceed 16 MiB")
    _require(atlas_path.stat().st_size <= MAX_ATLAS_BYTES, "representative Atlas exceeds 32 MiB")
    _require(resident_bytes <= MAX_ATLAS_BYTES, "representative Atlas resident bytes exceed 32 MiB")
    _require(resident_bytes == atlas_artifact["resident_bytes_estimate"], "Dyn Atlas resident estimate differs")
    _require(atlas_artifact["build_peak_rss_bytes"] <= MAX_BUILD_RSS_BYTES, "representative Atlas build RSS exceeds 512 MiB")
    representative_l2_cells = _integer(
        atlas["representative_l2_cells"], "Dyn representative L2 cells", minimum=1
    )
    _require(
        atlas["lookup"]
        == "origin-indexed exact L2 aggregate binary search in the admitted resident Atlas",
        "Dyn Atlas lookup scope differs",
    )

    dyn_state = _mapping(report["dyn_state"], "Dyn state evidence")
    _exact_keys(
        dyn_state,
        {
            "snapshot_magic", "schema_version", "client_ids", "client_count",
            "common_environment_steps", "population", "snapshots",
            "combined_compressed_bytes", "combined_resident_bytes",
            "combined_limit_bytes", "batch_ids_and_step_admitted",
        },
        "Dyn state evidence",
    )
    _require(dyn_state["snapshot_magic"] == "Q2LAT002", "Dyn snapshot magic differs")
    _require(dyn_state["schema_version"] == 2, "Dyn snapshot schema differs")
    _require(dyn_state["client_ids"] == [0, 1, 2, 3] and dyn_state["client_count"] == 4, "Dyn client population differs")
    _require(dyn_state["common_environment_steps"] == authority["environment_steps"], "Dyn environment step differs")
    _require(
        dyn_state["population"]
        == "deterministic per-client representative channels over admitted Atlas L2 cells; authority identities are never synthetic",
        "Dyn representative population differs",
    )
    _require(dyn_state["combined_limit_bytes"] == MAX_DYN_BATCH_BYTES, "Dyn combined limit differs")
    combined_compressed_bytes = _integer(
        dyn_state["combined_compressed_bytes"], "Dyn combined compressed bytes"
    )
    combined_resident_bytes = _integer(
        dyn_state["combined_resident_bytes"], "Dyn combined resident bytes"
    )
    _require(combined_compressed_bytes < MAX_DYN_BATCH_BYTES, "Dyn snapshots exceed 8 MiB")
    _require(combined_resident_bytes < MAX_DYN_BATCH_BYTES, "Dyn resident state exceeds 8 MiB")
    _require(dyn_state["batch_ids_and_step_admitted"] is True, "Dyn batch IDs/step were not admitted")
    expected_files = {"b2-dyn-evidence.json", *(f"client{client}.q2lat002" for client in range(4))}
    _exact_directory_files(path.parent, expected_files, "Dyn evidence root")
    snapshots = _list(dyn_state["snapshots"], "Dyn snapshots")
    _require(len(snapshots) == 4, "Dyn snapshot count differs")
    compressed_sum = 0
    resident_sum = 0
    decoded_ids = []
    for client, snapshot_value in enumerate(snapshots):
        snapshot = _mapping(snapshot_value, f"Dyn client {client}")
        _exact_keys(snapshot, {"client_id", "file", "magic", "schema_version", "l2_cells", "l3_cells", "resident_bytes", "byte_identical_roundtrip"}, f"Dyn client {client}")
        _require(snapshot["client_id"] == client, "Dyn client order differs")
        _require(snapshot["magic"] == "Q2LAT002" and snapshot["schema_version"] == 2, "Dyn client schema differs")
        _require(snapshot["byte_identical_roundtrip"] is True, "Dyn snapshot round trip differs")
        snapshot_path = path.parent / f"client{client}.q2lat002"
        _file_evidence_matches(snapshot["file"], snapshot_path, f"Dyn client {client} file")
        _require(
            Path(snapshot["file"]["path"]).name == snapshot_path.name,
            "Dyn snapshot filename differs",
        )
        decoded = _decode_dyn_snapshot(
            snapshot_path.read_bytes(),
            atlas_sha256=_file_sha256(atlas_path),
            map_sha256=_file_sha256(bsp_path),
            origin=atlas["origin"],
            map_epoch=authority["map_epoch"],
            environment_steps=authority["environment_steps"],
            client_count=4,
        )
        _require(decoded["client_id"] == client, "decoded Dyn client order differs")
        _require(decoded["l2_cells"] == snapshot["l2_cells"], "decoded Dyn L2 count differs")
        _require(decoded["l3_cells"] == snapshot["l3_cells"], "decoded Dyn L3 count differs")
        _require(
            decoded["l2_cells"] == representative_l2_cells,
            "Dyn snapshot does not cover the representative Atlas L2 population",
        )
        _require(decoded["resident_bytes"] == snapshot["resident_bytes"], "decoded Dyn resident bytes differ")
        decoded_ids.append(decoded["client_id"])
        compressed_sum += snapshot_path.stat().st_size
        resident_sum += _integer(snapshot["resident_bytes"], "Dyn client resident bytes")
    _require(decoded_ids == [0, 1, 2, 3], "decoded Dyn batch IDs differ")
    _require(compressed_sum == dyn_state["combined_compressed_bytes"], "Dyn compressed total differs")
    _require(resident_sum == dyn_state["combined_resident_bytes"], "Dyn resident total differs")
    _require(
        len({(item["l2_cells"], item["l3_cells"]) for item in snapshots}) == 1,
        "Dyn client snapshot populations differ",
    )

    negatives = _mapping(report["negative_fences_and_limits"], "Dyn negative fences")
    _exact_keys(
        negatives,
        {
            "stale_atlas_sha256_rejected", "stale_map_sha256_rejected",
            "stale_origin_rejected", "stale_map_epoch_rejected",
            "stale_environment_step_rejected", "wrong_client_count_rejected",
            "duplicate_client_rejected", "retired_schema_rejected",
            "mixed_schema_rejected", "payload_digest_corruption_rejected",
            "cell_size_mismatch_rejected",
            "soft_compressed_limit_reported", "hard_compressed_limit_rejected",
            "hard_resident_limit_rejected", "materialized_cell_limit_rejected",
        },
        "Dyn negative fences",
    )
    _require(all(value is True for value in negatives.values()), "a Dyn negative fence did not reject")
    performance = _mapping(report["performance"], "Dyn performance")
    _exact_keys(
        performance,
        {
            "scope", "resident_samples", "warmup_samples", "clients_per_sample",
            "atlas_lookup", "dyn_feature_assembly", "total",
            "total_p99_limit_ns", "total_p99_passed", "feature_width",
        },
        "Dyn performance",
    )
    _require(_integer(performance["resident_samples"], "Dyn resident samples") >= 2000, "Dyn performance sample count is insufficient")
    _require(performance["warmup_samples"] == 256, "Dyn performance warmup differs")
    _require(performance["clients_per_sample"] == 4, "Dyn performance does not cover four clients")
    _require(performance["feature_width"] == 24, "Dyn feature width differs")
    _require(
        performance["scope"]
        == "one accepted resident transition: exact admitted Atlas L2 lookup plus 24-float Dyn feature assembly for clients 0..3",
        "Dyn performance scope differs",
    )
    _latency_distribution(performance["atlas_lookup"], "Dyn Atlas lookup")
    _latency_distribution(performance["dyn_feature_assembly"], "Dyn feature assembly")
    total = _latency_distribution(performance["total"], "Dyn total")
    _require(performance["total_p99_limit_ns"] == MAX_FEATURE_ASSEMBLY_P99_NS, "Dyn p99 limit differs")
    _require(total["p99_ns"] < MAX_FEATURE_ASSEMBLY_P99_NS and performance["total_p99_passed"] is True, "Dyn p99 is not below 0.5 ms")
    budget = {
        "representative_map": map_id,
        "atlas_sha256": _file_sha256(atlas_path),
        "l0_chunks": counts["l0_chunks"],
        "l0_decompressed_bytes": l0_bytes,
        "atlas_decompressed_bytes": atlas_path.stat().st_size,
        "atlas_resident_bytes": resident_bytes,
        "atlas_build_peak_rss_bytes": atlas_artifact["build_peak_rss_bytes"],
        "four_dyn_compressed_bytes": dyn_state["combined_compressed_bytes"],
        "four_dyn_resident_bytes": dyn_state["combined_resident_bytes"],
        "four_client_feature_assembly_p99_ns": total["p99_ns"],
    }
    evidence = {
        "report": _file_record(path),
        "executable": executable,
        "source_authority": source_authority,
        "magic": "Q2LAT002",
        "client_count": 4,
        "host": host["hostname"],
        "kernel_release": host["kernel_release"],
    }
    return evidence, budget


def _validate_test_report(path: Path, implementation: Mapping[str, Any]) -> dict[str, Any]:
    from tools.run_b2_test_suite import B2TestSuiteError, _parse_counts

    _require(path.name == "b2-test-report.json", "B2 test report filename differs")
    report = _mapping(_load_json(path), "B2 test report")
    _exact_keys(report, {"schema", "implementation", "runs", "failures", "passed"}, "B2 test report")
    _require(report["schema"] == "q2-b2-test-report-v1", "B2 test report schema differs")
    _require(report["implementation"] == implementation, "B2 test report implementation is stale")
    runs = _list(report["runs"], "B2 test runs")
    expected_commands: list[tuple[str, list[str] | None]] = [
        ("python-syntax-floor", None),
        ("python", None),
        ("rust-fmt", ["cargo", "fmt", "--all", "--", "--check"]),
        (
            "rust-clippy",
            ["cargo", "clippy", "--locked", "--all-targets", "--", "-D", "warnings"],
        ),
        ("rust-tests", ["cargo", "test", "--locked", "--all-targets"]),
        (
            "dyn-fmt",
            [
                "cargo", "fmt", "--manifest-path",
                "tools/q2-dyn-evidence/Cargo.toml", "--", "--check",
            ],
        ),
        (
            "dyn-clippy",
            [
                "cargo", "clippy", "--locked", "--manifest-path",
                "tools/q2-dyn-evidence/Cargo.toml", "--all-targets", "--", "-D",
                "warnings",
            ],
        ),
        (
            "dyn-tests",
            [
                "cargo", "test", "--locked", "--manifest-path",
                "tools/q2-dyn-evidence/Cargo.toml",
            ],
        ),
    ]
    _require(len(runs) == len(expected_commands), "B2 test report suite count differs")
    names = []
    expected_files = {path.name}
    for item, (expected_name, expected_command) in zip(runs, expected_commands):
        run = _mapping(item, "B2 test run")
        _exact_keys(run, {"name", "command", "exit_code", "passed_count", "skipped_count", "ignored_count", "log"}, "B2 test run")
        _require(isinstance(run["name"], str) and run["name"], "test run name is missing")
        _require(isinstance(run["command"], list) and run["command"] and all(isinstance(v, str) and v for v in run["command"]), "test command differs")
        _require(run["name"] == expected_name, "B2 test suite order/name differs")
        if expected_name == "python-syntax-floor":
            _require(
                len(run["command"]) == 3
                and Path(run["command"][0]).name.startswith("python")
                and run["command"][1:]
                == ["-B", "tools/check_python_syntax_floor.py"],
                "B2 Python syntax-floor command differs",
            )
        elif expected_command is None:
            _require(
                len(run["command"]) == 4
                and Path(run["command"][0]).name.startswith("python")
                and run["command"][1:] == ["-m", "pytest", "-q"],
                "B2 pytest command differs",
            )
        else:
            _require(run["command"] == expected_command, f"B2 {expected_name} command differs")
        _require(run["exit_code"] == 0, f"test run failed: {run['name']}")
        _integer(run["passed_count"], "test passed count", minimum=1)
        _integer(run["skipped_count"], "test skipped count")
        _integer(run["ignored_count"], "test ignored count")
        log = _mapping(run["log"], "test raw log")
        _exact_keys(log, {"path", "bytes", "sha256"}, "test raw log")
        log_path = Path(log["path"])
        _require(log_path.is_absolute(), "test raw-log path must be absolute")
        _require(log_path.parent == path.resolve().parent, "test raw log is outside the exact evidence root")
        _require(dict(log) == {"path": str(log_path), **_file_record(log_path)}, "test raw-log identity differs")
        _require(log_path.name == f"{expected_name}.log", "test raw-log filename differs")
        try:
            recomputed_counts = _parse_counts(
                expected_name, log_path.read_bytes(), run["exit_code"]
            )
        except (B2TestSuiteError, OSError) as exc:
            raise B2GateError(
                f"cannot independently parse {expected_name} test counts: {exc}"
            ) from exc
        _require(
            recomputed_counts
            == (
                run["passed_count"], run["skipped_count"], run["ignored_count"]
            ),
            f"B2 {expected_name} test counts differ from the raw log",
        )
        expected_files.add(log_path.name)
        names.append(run["name"])
    _require(len(names) == len(set(names)), "test run names are duplicated")
    _exact_directory_files(path.parent, expected_files, "B2 test evidence root")
    _require(report["passed"] is True and report["failures"] == [], "B2 test report is not green")
    return {"report": _file_record(path), "run_count": len(runs), "passed_count": sum(run["passed_count"] for run in runs)}


def assemble_gate(
    paths: B2GatePaths, *, implementation_binding: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    normative = _validate_normative_documents(paths)
    implementation = _validate_implementation(paths, implementation_binding)
    b1, binaries = _validate_b1_and_oracles(paths, normative)
    declaration, declaration_sha256 = _validate_declaration(paths.declaration)
    qualification = _validate_qualification_report(
        paths, normative, implementation
    )
    source = _validate_source_freeze(
        paths, declaration, declaration_sha256, implementation
    )
    compiled = _validate_compiled_and_static(
        paths, declaration, declaration_sha256
    )
    compiled_cm_preflight = _validate_compiled_cm_preflight(
        paths, declaration, declaration_sha256, binaries
    )
    materialized = _validate_materialized(
        paths, declaration, declaration_sha256, binaries, normative
    )
    claims, claims_membership = _validate_claims(
        paths, declaration, declaration_sha256
    )
    generated = _validate_generated(
        paths, declaration, declaration_sha256, implementation,
        claims_membership, binaries,
    )
    stock = _validate_stock(paths, implementation, binaries)
    dyn, budgets = _validate_dyn_evidence(
        paths.dyn_evidence_report, implementation, declaration, paths
    )
    tests = _validate_test_report(paths.test_report, implementation)
    gate = {
        "schema": GATE_SCHEMA,
        "batch": "B2",
        "status": "green",
        "owner_directive": {
            "replacement": "one-way",
            "legacy_model_lineages": "retired",
            "operational_fallback": "forbidden",
            "legacy_runtime_or_model_used": False,
        },
        "normative_documents": normative,
        "implementation": implementation,
        "b1_authority": b1,
        "toolchain_qualification": qualification,
        "generated_cohort": {
            "cohort_id": EXPECTED_COHORT,
            "declaration": _file_record(paths.declaration),
            "declaration_sha256": declaration_sha256,
            "source": source,
            "compiled": compiled,
            "compiled_cm_preflight": compiled_cm_preflight,
            "materialized": materialized,
            "claims": claims,
            "analysis": generated,
        },
        "stock_corpus": stock,
        "representative_budgets": budgets,
        "dyn_evidence": dyn,
        "tests": tests,
        "deployment": {
            "public_or_teacher_service_changed": False,
            "cross_host_runtime_copy_performed": False,
            "trainer_or_tensorboard_started": False,
        },
        "gate": {
            "stock_q2dm1_q2dm8_cold_stable": True,
            "stock_pins_match": True,
            "generated_maps_passed": 28,
            "generated_static_pass_rate_preserved": True,
            "toolchain_qualification_non_admissible": True,
            "compiled_cm_preflight_maps_passed": 28,
            "oracle_failures_overridable": False,
            "atlas_budgets_passed": True,
            "feature_assembly_budget_passed": True,
            "dyn_snapshot_budget_passed": True,
            "failures": [],
            "green": True,
        },
    }
    validate_gate(gate)
    if implementation_binding is None:
        _require(
            repository_binding(paths.repo_root) == implementation,
            "repository changed while B2 gate evidence was being assembled",
        )
    return gate


def validate_gate(value: object) -> dict[str, Any]:
    gate = _mapping(value, "B2 gate")
    _exact_keys(
        gate,
        {"schema", "batch", "status", "owner_directive", "normative_documents", "implementation", "b1_authority", "toolchain_qualification", "generated_cohort", "stock_corpus", "representative_budgets", "dyn_evidence", "tests", "deployment", "gate"},
        "B2 gate",
    )
    _require(gate["schema"] == GATE_SCHEMA and gate["batch"] == "B2", "B2 gate identity differs")
    _require(gate["status"] == "green", "B2 gate status is not green")
    predicate = _mapping(gate["gate"], "B2 predicate")
    _require(predicate.get("green") is True and predicate.get("failures") == [], "B2 predicate is not green")
    _require(
        predicate.get("toolchain_qualification_non_admissible") is True
        and predicate.get("compiled_cm_preflight_maps_passed") == 28,
        "B2 qualification or compiled-CM predicate differs",
    )
    qualification = _mapping(
        gate["toolchain_qualification"], "B2 toolchain qualification"
    )
    _require(
        qualification.get("schema") == QUALIFICATION_SCHEMA
        and qualification.get("non_admissible") is True
        and qualification.get("retryable") is True
        and qualification.get("final_cohort_authorized") is False,
        "B2 toolchain qualification disposition differs",
    )
    generated = _mapping(gate["generated_cohort"], "generated cohort")
    _require(generated.get("cohort_id") == EXPECTED_COHORT, "B2 gate cohort differs")
    preflight = _mapping(
        generated.get("compiled_cm_preflight"), "B2 compiled-CM preflight"
    )
    _require(
        preflight.get("schema") == COMPILED_CM_PREFLIGHT_SCHEMA
        and preflight.get("admission_status") == COMPILED_CM_PREFLIGHT_STATUS
        and preflight.get("pass_count") == 28,
        "B2 compiled-CM evidence differs",
    )
    _require(_mapping(gate["stock_corpus"], "stock corpus").get("map_count") == 8, "B2 stock count differs")
    return dict(gate)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--b1-gate", type=Path, required=True)
    parser.add_argument("--cm-oracle", type=Path, required=True)
    parser.add_argument("--pmove-oracle", type=Path, required=True)
    parser.add_argument("--hook-oracle", type=Path, required=True)
    parser.add_argument("--fall-oracle", type=Path, required=True)
    parser.add_argument("--hook-attestation", type=Path, required=True)
    parser.add_argument("--atlas-verifier", type=Path, required=True)
    parser.add_argument("--declaration", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--source-cold-dir", type=Path, required=True)
    parser.add_argument("--source-freeze-report", type=Path, required=True)
    parser.add_argument("--compiled-dir", type=Path, required=True)
    parser.add_argument("--compiled-membership-report", type=Path, required=True)
    parser.add_argument("--compiled-static-report", type=Path, required=True)
    parser.add_argument("--compiled-cm-preflight-report", type=Path, required=True)
    parser.add_argument("--materialized-dir", type=Path, required=True)
    parser.add_argument("--materialized-membership-report", type=Path, required=True)
    parser.add_argument("--claims-dir", type=Path, required=True)
    parser.add_argument("--claims-prepare-report", type=Path, required=True)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--generated-build-report", type=Path, required=True)
    parser.add_argument("--generated-validation-report", type=Path, required=True)
    parser.add_argument("--stock-provenance", type=Path, required=True)
    parser.add_argument("--stock-inventory", type=Path, required=True)
    parser.add_argument("--stock-bsp-dir", type=Path, required=True)
    parser.add_argument("--stock-analysis-dir", type=Path, required=True)
    parser.add_argument("--stock-validation-dir", type=Path, required=True)
    parser.add_argument("--dyn-evidence-executable", type=Path, required=True)
    parser.add_argument("--dyn-evidence-report", type=Path, required=True)
    parser.add_argument("--test-report", type=Path, required=True)
    parser.add_argument("--qualification-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    output = args.output
    values = vars(args).copy()
    del values["output"]
    try:
        if output.exists() or output.is_symlink():
            raise B2GateError("B2 gate output already exists; refusing overwrite")
        try:
            output.resolve().relative_to(args.repo_root.resolve())
        except ValueError:
            pass
        else:
            raise B2GateError(
                "B2 gate output must be outside the implementation repository"
            )
        gate = assemble_gate(B2GatePaths(**values))
        _require(
            repository_binding(args.repo_root) == gate["implementation"],
            "repository changed immediately before B2 gate publication",
        )
        payload = canonical_bytes(gate)
        _exclusive_write(output, payload)
        sys.stdout.buffer.write(payload)
        return 0
    except (
        B2GateError, B2QualificationError, GeneratorCohortError, ClaimValidationError,
        HookClaimsV4Error, OSError, subprocess.CalledProcessError,
    ) as exc:
        print(f"B2 gate refused: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
