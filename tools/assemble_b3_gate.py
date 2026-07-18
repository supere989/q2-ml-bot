#!/usr/bin/env python3
"""Assemble the fail-closed offline B3 milestone gate.

The assembler consumes a green B2 gate, the measured design-prior campaign,
and separately produced recovery/guide and bundle-v3 evidence.  It recomputes
source closures and repository identity and refuses placeholder, synthetic,
or cross-identity claims.  It performs no build, deployment, staging, or
runtime launch.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.run_b3_design_prior_campaign import (  # noqa: E402
    B3PriorError,
    CAMPAIGN_SCHEMA,
    canonical_bytes,
    file_sha256,
    load_json,
    validate_campaign_report,
    validate_implementation,
)
from tools import assemble_b2_gate as b2_gate  # noqa: E402


GATE_SCHEMA = "q2-multires-b3-gate-v1"
RECOVERY_GUIDE_SCHEMA = "q2-b3-recovery-guide-evidence-v1"
BUNDLE_SCHEMA = "q2-b3-bundle-evidence-v1"
B2_GATE_SCHEMA = "q2-multires-b2-gate-v1"
HEX64 = re.compile(r"^[0-9a-f]{64}$")
HEX40 = re.compile(r"^[0-9a-f]{40}$")
PLACEHOLDER_MARKERS = ("placeholder", "synthetic", "dummy", "tbd", "todo", "fixme", "changeme")
HAZARD_CLASSES = ["lava", "slime", "hurt", "void_or_lethal_drop", "crush_or_current"]
OBJECTIVE_CLASSES = [
    "weapon", "ammunition", "health", "armor", "powerup", "rune",
    "control", "spawn_egress",
]
RECOVERY_TEST_COMMANDS = (
    ("cargo", "test", "-q", "-p", "q2-lattice", "recovery"),
    ("cargo", "test", "-q", "-p", "q2-lattice", "guide"),
    ("cargo", "test", "-q", "-p", "q2-lattice", "hook_necessity"),
    (
        sys.executable, "-m", "pytest", "-q",
        "tests/test_rust_provider_extension_qualification.py",
        "tests/test_multires_contract.py",
    ),
)
BUNDLE_CLAIM_NODE_IDS = {
    "bundle_v2.analysis_present_install_passed": (
        "tests/test_map_farm.py::test_v2_install_accepts_optional_analysis_artifacts",
    ),
    "bundle_v2.analysis_absent_install_passed": (
        "tests/test_map_farm.py::test_farm_bundle_is_verified_and_installed_atomically",
    ),
    "bundle_v2.consumer_behavior_unchanged": (
        "tests/test_map_farm.py::test_farm_bundle_is_verified_and_installed_atomically",
        "tests/test_map_farm.py::test_v2_install_accepts_optional_analysis_artifacts",
        "tests/test_map_farm.py::test_v2_bundle_keeps_v1_digest_shape_for_worker_first_rollout",
    ),
    "bundle_v3.isolated_runtime_only": (
        "tests/test_map_farm.py::test_bundle_v3_is_isolated_by_default_and_explicitly_admitted",
    ),
    "bundle_v3.public_enabled": (
        "tests/test_map_farm.py::test_bundle_v3_is_isolated_by_default_and_explicitly_admitted",
    ),
    "bundle_v3.mandatory_atlas_missing_rejected": (
        "tests/test_map_farm.py::test_bundle_v3_rejects_missing_analysis",
    ),
    "bundle_v3.mandatory_atlas_mismatch_rejected": (
        "tests/test_map_farm.py::test_bundle_v3_rejects_mismatched_analysis",
    ),
    "bundle_v3.atomic_failure_restored_prior_generation": (
        "tests/test_map_farm.py::test_bundle_v3_mid_publication_failure_restores_prior_generation",
    ),
    "farm.analysis_failure_reported": (
        "tests/test_map_farm.py::test_farm_health_reports_analysis_failure_without_vps_fallback",
    ),
    "farm.vps_compilation_fallback_enabled": (
        "tests/test_map_farm.py::test_farm_health_reports_analysis_failure_without_vps_fallback",
    ),
}
BUNDLE_REQUIRED_NODE_IDS = tuple(dict.fromkeys(
    node_id
    for node_ids in BUNDLE_CLAIM_NODE_IDS.values()
    for node_id in node_ids
))
BUNDLE_CLAIM_EXPECTED_VALUES = {
    claim: False if claim in {
        "bundle_v3.public_enabled", "farm.vps_compilation_fallback_enabled",
    } else True
    for claim in BUNDLE_CLAIM_NODE_IDS
}
BUNDLE_TEST_COMMANDS = tuple(
    (sys.executable, "-m", "pytest", "-q", node_id)
    for node_id in BUNDLE_REQUIRED_NODE_IDS
)
RECOVERY_GUIDE_SOURCE_PATHS = (
    "Cargo.toml",
    "Cargo.lock",
    "tools/run_b3_component_evidence.py",
    "harness/atlas_analyzer.py",
    "crates/q2-lattice/Cargo.toml",
    "crates/q2-lattice/src/atlas/admission.rs",
    "crates/q2-lattice/src/atlas/aggregate.rs",
    "crates/q2-lattice/src/atlas/coord.rs",
    "crates/q2-lattice/src/atlas/error.rs",
    "crates/q2-lattice/src/atlas/graph.rs",
    "crates/q2-lattice/src/atlas/guide.rs",
    "crates/q2-lattice/src/atlas/l0.rs",
    "crates/q2-lattice/src/atlas/manifest.rs",
    "crates/q2-lattice/src/atlas/mod.rs",
    "crates/q2-lattice/src/atlas/objective.rs",
    "crates/q2-lattice/src/atlas/recovery.rs",
    "crates/q2-lattice/src/atlas/runtime.rs",
    "crates/q2-lattice/src/atlas/storage.rs",
    "crates/q2-lattice/src/atlas/SCHEMA.md",
    "crates/q2-lattice/src/dynstate.rs",
    "crates/q2-lattice/src/lib.rs",
    "crates/q2-lattice/tests/atlas_schema.rs",
    "crates/q2-lattice/tests/dyn_snapshot.rs",
    "tests/test_rust_provider_extension_qualification.py",
    "tests/test_multires_contract.py",
)
BUNDLE_SOURCE_PATHS = (
    "tools/run_b3_component_evidence.py",
    "tools/map_bundle.py",
    "tools/map_farm_client.py",
    "tools/map_farm_worker.py",
    "tests/test_map_farm.py",
)


class B3GateError(ValueError):
    """Raised before publication when B3 evidence is inadmissible."""


def gate_sha256(value: Mapping[str, Any]) -> str:
    """Return the domain-separated seal over the exact unsealed B3 gate."""

    body = dict(value)
    body.pop("gate_sha256", None)
    return hashlib.sha256(canonical_bytes({
        "domain": GATE_SCHEMA,
        "gate": body,
    })).hexdigest()


@dataclass(frozen=True)
class B3GatePaths:
    repo_root: Path
    b2_gate: Path
    prior_campaign: Path
    recovery_guide_evidence: Path
    bundle_evidence: Path
    output: Path


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise B3GateError(f"{label} must be an object")
    return value


def _list(value: object, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise B3GateError(f"{label} must be an array")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        raise B3GateError(
            f"{label} keys differ; missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _integer(value: object, label: str, minimum: int = 0, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise B3GateError(f"{label} must be an integer >= {minimum}")
    if maximum is not None and value > maximum:
        raise B3GateError(f"{label} must be <= {maximum}")
    return value


def _digest(value: object, label: str, *, git: bool = False) -> str:
    pattern = HEX40 if git else HEX64
    if not isinstance(value, str) or not pattern.fullmatch(value) or set(value) == {"0"}:
        raise B3GateError(f"{label} is not a nonzero lowercase digest")
    return value


def _reject_placeholders(value: object, label: str) -> None:
    if isinstance(value, str):
        if any(marker in value.lower() for marker in PLACEHOLDER_MARKERS):
            raise B3GateError(f"{label} contains placeholder/synthetic text")
    elif isinstance(value, Mapping):
        for key, child in value.items():
            _reject_placeholders(child, f"{label}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_placeholders(child, f"{label}[{index}]")


def _file_record(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise B3GateError(f"required regular evidence file is missing or a symlink: {path}")
    return {"bytes": path.stat().st_size, "sha256": file_sha256(path)}


def _validate_file_record(value: object, label: str) -> dict[str, Any]:
    record = _mapping(value, label)
    _exact_keys(record, {"bytes", "sha256"}, label)
    _integer(record["bytes"], f"{label} bytes", 1)
    _digest(record["sha256"], f"{label} SHA-256")
    return dict(record)


def _embedded_file_record(value: object) -> dict[str, Any]:
    payload = canonical_bytes(value)
    return {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def _validate_closure_record(value: object, label: str) -> dict[str, Any]:
    closure = _mapping(value, label)
    _exact_keys(closure, {"file_count", "sha256"}, label)
    _integer(closure["file_count"], f"{label} file count", 1)
    _digest(closure["sha256"], f"{label} SHA-256")
    return dict(closure)


def _source_closure(repo_root: Path, paths: Sequence[str]) -> dict[str, Any]:
    records = []
    for relative in paths:
        path = repo_root / relative
        if not path.is_file() or path.is_symlink():
            raise B3GateError(f"source closure member is missing or a symlink: {relative}")
        records.append({"path": relative, "bytes": path.stat().st_size, "sha256": file_sha256(path)})
    return {
        "file_count": len(records),
        "sha256": hashlib.sha256(canonical_bytes(records)).hexdigest(),
    }


def repository_identity(repo_root: Path) -> dict[str, Any]:
    try:
        status = subprocess.run(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"],
            cwd=repo_root, check=True, capture_output=True, text=True,
        ).stdout.strip()
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
        tree = subprocess.run(
            ["git", "rev-parse", "HEAD^{tree}"], cwd=repo_root, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise B3GateError(f"cannot bind repository identity: {exc}") from exc
    if status:
        raise B3GateError("repository is not clean; refusing B3 gate assembly")
    _digest(commit, "repository commit", git=True)
    _digest(tree, "repository tree", git=True)
    return {"repository_commit": commit, "repository_tree": tree, "git_clean": True}


def _load(path: Path) -> Any:
    try:
        return load_json(path)
    except B3PriorError as exc:
        raise B3GateError(str(exc)) from exc


def validate_b2_gate(value: object, normative: Mapping[str, str]) -> dict[str, Any]:
    gate = _mapping(value, "B2 gate")
    try:
        gate = b2_gate.validate_gate(gate)
        authority = b2_gate._require_active_final_authority()
    except b2_gate.B2GateError as exc:
        raise B3GateError(f"B2 predecessor authority rejected: {exc}") from exc
    if gate.get("schema") != B2_GATE_SCHEMA or gate.get("batch") != "B2" or gate.get("status") != "green":
        raise B3GateError("B2 gate is not the canonical green predecessor")
    generated = _mapping(gate.get("generated_cohort"), "B2 generated cohort")
    if (
        generated.get("cohort_id") != authority.cohort_id
        or generated.get("cohort_id") == b2_gate.RETIRED_COHORT_71446
        or generated.get("declaration_sha256") != authority.declaration_sha256
    ):
        raise B3GateError("B2 predecessor differs from the active final authority")
    decision = _mapping(gate.get("gate"), "B2 decision")
    if decision.get("green") is not True or decision.get("failures") != []:
        raise B3GateError("B2 predecessor gate is not green")
    deployment = _mapping(gate.get("deployment"), "B2 deployment")
    if any(deployment.get(key) is not False for key in (
        "public_or_teacher_service_changed", "cross_host_runtime_copy_performed",
        "trainer_or_tensorboard_started",
    )):
        raise B3GateError("B2 evidence includes a forbidden deployment mutation")
    documents = _mapping(gate.get("normative_documents"), "B2 normative documents")
    for key, expected in (("design", normative["design_sha256"]), ("plan", normative["plan_sha256"])):
        record = _mapping(documents.get(key), f"B2 normative {key}")
        if record.get("sha256") != expected:
            raise B3GateError(f"B2 gate {key} identity differs from B3 authority")
    _reject_placeholders(gate, "B2 gate")
    return dict(gate)


def _commands_sha256(commands: Sequence[Sequence[str]]) -> str:
    return hashlib.sha256(canonical_bytes([list(command) for command in commands])).hexdigest()


def _validate_test_summary(
    value: object,
    label: str,
    expected_commands: Sequence[Sequence[str]],
) -> dict[str, Any]:
    tests = _mapping(value, label)
    _exact_keys(
        tests,
        {"report_sha256", "commands_sha256", "passed_count", "failed_count", "runs"},
        label,
    )
    if tests["commands_sha256"] != _commands_sha256(expected_commands):
        raise B3GateError(f"{label} command-list identity differs")
    runs = _list(tests["runs"], f"{label} runs")
    if len(runs) != len(expected_commands):
        raise B3GateError(f"{label} run count differs")
    total_passed = 0
    normalized_runs = []
    for ordinal, (raw, expected) in enumerate(zip(runs, expected_commands, strict=True)):
        run = _mapping(raw, f"{label} run {ordinal}")
        _exact_keys(
            run,
            {"command", "exit_code", "passed_count", "stdout", "stderr"},
            f"{label} run {ordinal}",
        )
        if run["command"] != list(expected) or run["exit_code"] != 0:
            raise B3GateError(f"{label} run {ordinal} command or exit status differs")
        passed = _integer(run["passed_count"], f"{label} run {ordinal} passed count", 1)
        total_passed += passed
        for stream_name in ("stdout", "stderr"):
            stream = _mapping(run[stream_name], f"{label} run {ordinal} {stream_name}")
            _exact_keys(stream, {"bytes", "sha256"}, f"{label} run {ordinal} {stream_name}")
            _integer(stream["bytes"], f"{label} run {ordinal} {stream_name} bytes", 0)
            _digest(stream["sha256"], f"{label} run {ordinal} {stream_name} SHA-256")
        normalized_runs.append(dict(run))
    if tests["report_sha256"] != hashlib.sha256(canonical_bytes({
        "schema": "q2-b3-component-test-runs-v1",
        "runs": normalized_runs,
    })).hexdigest():
        raise B3GateError(f"{label} report digest does not recompute")
    if _integer(tests["passed_count"], f"{label} passed count", 1) != total_passed:
        raise B3GateError(f"{label} aggregate passed count differs")
    _integer(tests["failed_count"], f"{label} failed count", 0, 0)
    return dict(tests)


def _validate_rust_extension(
    value: object,
    repository: Mapping[str, Any],
    closure: Mapping[str, Any],
    tests: Mapping[str, Any],
) -> dict[str, Any]:
    record = _mapping(value, "recovery Rust extension")
    _exact_keys(
        record,
        {
            "path", "bytes", "sha256", "repository_tree",
            "source_closure_sha256", "qualification_commands_sha256",
        },
        "recovery Rust extension",
    )
    raw_path = record["path"]
    if not isinstance(raw_path, str) or not raw_path or not Path(raw_path).is_absolute():
        raise B3GateError("recovery Rust extension path must be absolute")
    path = Path(raw_path)
    if path.is_symlink() or not path.is_file() or path.name not in {
        "libq2_lattice_rs.so", "libq2_lattice_rs.dylib", "q2_lattice_rs.dll",
    }:
        raise B3GateError("recovery Rust extension is not an exact regular extension file")
    if _integer(record["bytes"], "recovery Rust extension bytes", 1) != path.stat().st_size:
        raise B3GateError("recovery Rust extension size drifted")
    if _digest(record["sha256"], "recovery Rust extension SHA-256") != file_sha256(path):
        raise B3GateError("recovery Rust extension digest drifted")
    if record["repository_tree"] != repository["repository_tree"]:
        raise B3GateError("recovery Rust extension repository tree drifted")
    if record["source_closure_sha256"] != closure["sha256"]:
        raise B3GateError("recovery Rust extension source closure drifted")
    if record["qualification_commands_sha256"] != tests["commands_sha256"]:
        raise B3GateError("recovery Rust extension qualification command identity drifted")
    return dict(record)


def _validate_component_implementation(
    value: object,
    label: str,
    repository: Mapping[str, Any],
    expected_closure: Mapping[str, Any],
) -> dict[str, Any]:
    implementation = _mapping(value, f"{label} implementation")
    _exact_keys(
        implementation,
        {"repository_commit", "repository_tree", "git_clean", "source_closure"},
        f"{label} implementation",
    )
    for key in ("repository_commit", "repository_tree", "git_clean"):
        if implementation[key] != repository[key]:
            raise B3GateError(f"{label} repository identity drifted at {key}")
    closure = _mapping(implementation["source_closure"], f"{label} source closure")
    if closure != expected_closure:
        raise B3GateError(f"{label} source closure identity drifted")
    return dict(implementation)


def validate_recovery_guide_evidence(
    value: object,
    repository: Mapping[str, Any],
    expected_closure: Mapping[str, Any],
) -> dict[str, Any]:
    evidence = _mapping(value, "recovery/guide evidence")
    _exact_keys(
        evidence,
        {
            "schema", "evidence_kind", "synthetic_claims", "implementation",
            "rust_extension", "atlas_set_sha256", "map_count", "recovery", "guide", "tests",
            "failures", "passed",
        },
        "recovery/guide evidence",
    )
    if evidence["schema"] != RECOVERY_GUIDE_SCHEMA:
        raise B3GateError("recovery/guide evidence schema differs")
    if evidence["evidence_kind"] != "measured-offline-atlas" or evidence["synthetic_claims"] is not False:
        raise B3GateError("recovery/guide evidence is placeholder or synthetic")
    _validate_component_implementation(evidence["implementation"], "recovery/guide", repository, expected_closure)
    _digest(evidence["atlas_set_sha256"], "recovery/guide Atlas-set SHA-256")
    _integer(evidence["map_count"], "recovery/guide map count", 8)
    recovery = _mapping(evidence["recovery"], "recovery measurements")
    _exact_keys(
        recovery,
        {
            "finite_non_safe_cells", "strict_descending_cells",
            "mover_plateau_cells", "unresolved_cells", "max_local_repair_nodes",
            "hazard_classes", "hook_necessity_walking_budget_ticks",
            "hook_necessity_game_tick_hz", "hook_necessity_walk_speed_q8_per_second",
            "hook_necessity_evaluated_edges", "hook_necessity_query_cells",
            "hook_necessity_path_to_safety_cases", "hook_necessity_positive_cases",
            "recovery_width",
        },
        "recovery measurements",
    )
    finite = _integer(recovery["finite_non_safe_cells"], "finite non-safe cells", 1)
    descending = _integer(recovery["strict_descending_cells"], "strict descending cells", 0)
    plateaus = _integer(recovery["mover_plateau_cells"], "mover plateau cells", 0)
    unresolved = _integer(recovery["unresolved_cells"], "unresolved recovery cells", 0, 0)
    if unresolved != 0 or descending + plateaus != finite:
        raise B3GateError("finite recovery cells do not all descend or name a mover plateau")
    _integer(recovery["max_local_repair_nodes"], "maximum local repair nodes", 1, 4096)
    if recovery["hazard_classes"] != HAZARD_CLASSES:
        raise B3GateError("recovery hazard classes differ from the frozen five classes")
    if (
        recovery["hook_necessity_walking_budget_ticks"] != 15
        or recovery["hook_necessity_game_tick_hz"] != 10
        or recovery["hook_necessity_walk_speed_q8_per_second"] != 76_800
        or recovery["recovery_width"] != 16
    ):
        raise B3GateError("recovery packing or hook-necessity budget differs")
    evaluated = _integer(
        recovery["hook_necessity_evaluated_edges"], "hook-necessity evaluated edges", 1,
    )
    queries = _integer(recovery["hook_necessity_query_cells"], "hook-necessity query cells", 1)
    paths = _integer(
        recovery["hook_necessity_path_to_safety_cases"],
        "hook-necessity path-to-safety cases", 0, queries,
    )
    _integer(
        recovery["hook_necessity_positive_cases"],
        "hook-necessity positive cases", 0, paths,
    )
    if evaluated < queries:
        raise B3GateError("hook-necessity evaluated-edge count is smaller than query count")
    guide = _mapping(evidence["guide"], "guide measurements")
    _exact_keys(
        guide,
        {
            "guide_width", "candidate_count", "candidate_width",
            "objective_classes", "objective_bearing_fixtures",
            "universal_zero_objective_fixtures",
        },
        "guide measurements",
    )
    if guide["guide_width"] != 60 or guide["candidate_count"] != 4 or guide["candidate_width"] != 15:
        raise B3GateError("guide packing differs from frozen 4x15/60")
    if guide["objective_classes"] != OBJECTIVE_CLASSES:
        raise B3GateError("guide objective classes differ from the frozen eight classes")
    _integer(guide["objective_bearing_fixtures"], "objective-bearing fixtures", 1)
    _integer(
        guide["universal_zero_objective_fixtures"],
        "universal-zero objective fixtures", 0, 0,
    )
    tests = _validate_test_summary(
        evidence["tests"], "recovery/guide tests", RECOVERY_TEST_COMMANDS,
    )
    _validate_rust_extension(evidence["rust_extension"], repository, expected_closure, tests)
    if evidence["failures"] != [] or evidence["passed"] is not True:
        raise B3GateError("recovery/guide evidence is not passing")
    _reject_placeholders(evidence, "recovery/guide evidence")
    return dict(evidence)


def derive_bundle_claim_evidence(value: object) -> dict[str, Any]:
    """Derive every B3 bundle claim from its frozen pytest node outcomes."""

    tests = _mapping(value, "bundle claim test summary")
    runs = _list(tests.get("runs"), "bundle claim test runs")
    outcomes: dict[str, str] = {}
    for ordinal, run_value in enumerate(runs):
        run = _mapping(run_value, f"bundle claim test run {ordinal}")
        command = run.get("command")
        if not isinstance(command, list) or len(command) != 5:
            raise B3GateError("bundle claim test command shape differs")
        node_id = command[-1]
        expected = (sys.executable, "-m", "pytest", "-q", node_id)
        if tuple(command) != expected or node_id not in BUNDLE_REQUIRED_NODE_IDS:
            raise B3GateError("bundle claim test node ID is undeclared")
        if node_id in outcomes:
            raise B3GateError("bundle claim test node ID is duplicated")
        if run.get("exit_code") != 0 or run.get("passed_count") != 1:
            raise B3GateError(f"bundle claim test node did not pass exactly once: {node_id}")
        outcomes[node_id] = "passed"
    if set(outcomes) != set(BUNDLE_REQUIRED_NODE_IDS):
        raise B3GateError("bundle claim test node membership differs")

    derived: dict[str, Any] = {}
    for claim, required in BUNDLE_CLAIM_NODE_IDS.items():
        node_outcomes = [
            {"node_id": node_id, "outcome": outcomes[node_id]}
            for node_id in required
        ]
        passed = all(item["outcome"] == "passed" for item in node_outcomes)
        if not passed:
            raise B3GateError(f"bundle claim has a non-passing required node: {claim}")
        derived[claim] = {
            "required_node_ids": list(required),
            "node_outcomes": node_outcomes,
            "all_required_passed": True,
            "derived_value": BUNDLE_CLAIM_EXPECTED_VALUES[claim],
        }
    return derived


def _bundle_sections_from_claims(claims: Mapping[str, Any]) -> dict[str, dict[str, bool]]:
    def derived(section: str, field: str) -> bool:
        claim = _mapping(claims[f"{section}.{field}"], f"bundle claim {section}.{field}")
        if claim.get("all_required_passed") is not True or not isinstance(
            claim.get("derived_value"), bool,
        ):
            raise B3GateError(f"bundle claim is not derived and passing: {section}.{field}")
        return bool(claim["derived_value"])

    return {
        "bundle_v2": {
            field: derived("bundle_v2", field)
            for field in (
                "analysis_present_install_passed", "analysis_absent_install_passed",
                "consumer_behavior_unchanged",
            )
        },
        "bundle_v3": {
            field: derived("bundle_v3", field)
            for field in (
                "isolated_runtime_only", "public_enabled",
                "mandatory_atlas_missing_rejected", "mandatory_atlas_mismatch_rejected",
                "atomic_failure_restored_prior_generation",
            )
        },
        "farm": {
            field: derived("farm", field)
            for field in (
                "analysis_failure_reported", "vps_compilation_fallback_enabled",
            )
        },
    }


def validate_bundle_evidence(
    value: object,
    repository: Mapping[str, Any],
    expected_closure: Mapping[str, Any],
) -> dict[str, Any]:
    evidence = _mapping(value, "bundle evidence")
    _exact_keys(
        evidence,
        {
            "schema", "evidence_kind", "synthetic_claims", "implementation",
            "claim_tests", "bundle_v2", "bundle_v3", "farm", "tests", "failures", "passed",
        },
        "bundle evidence",
    )
    if evidence["schema"] != BUNDLE_SCHEMA:
        raise B3GateError("bundle evidence schema differs")
    if evidence["evidence_kind"] != "measured-offline-installer" or evidence["synthetic_claims"] is not False:
        raise B3GateError("bundle evidence is placeholder or synthetic")
    _validate_component_implementation(evidence["implementation"], "bundle", repository, expected_closure)
    tests = _validate_test_summary(evidence["tests"], "bundle tests", BUNDLE_TEST_COMMANDS)
    derived_claims = derive_bundle_claim_evidence(tests)
    if evidence["claim_tests"] != derived_claims:
        raise B3GateError("bundle claim evidence does not recompute from named test nodes")
    expected_sections = _bundle_sections_from_claims(derived_claims)
    for section in ("bundle_v2", "bundle_v3", "farm"):
        if evidence[section] != expected_sections[section]:
            raise B3GateError(f"{section} booleans do not derive from named test nodes")

    v2 = _mapping(evidence["bundle_v2"], "bundle v2 evidence")
    _exact_keys(v2, {"analysis_present_install_passed", "analysis_absent_install_passed", "consumer_behavior_unchanged"}, "bundle v2 evidence")
    if any(value is not True for value in v2.values()):
        raise B3GateError("bundle v2 compatibility proof is incomplete")
    v3 = _mapping(evidence["bundle_v3"], "bundle v3 evidence")
    _exact_keys(
        v3,
        {
            "isolated_runtime_only", "public_enabled", "mandatory_atlas_missing_rejected",
            "mandatory_atlas_mismatch_rejected", "atomic_failure_restored_prior_generation",
        },
        "bundle v3 evidence",
    )
    if (
        v3["isolated_runtime_only"] is not True
        or v3["public_enabled"] is not False
        or any(v3[key] is not True for key in (
            "mandatory_atlas_missing_rejected", "mandatory_atlas_mismatch_rejected",
            "atomic_failure_restored_prior_generation",
        ))
    ):
        raise B3GateError("bundle v3 is not isolated and fail-closed")
    farm = _mapping(evidence["farm"], "farm evidence")
    _exact_keys(farm, {"analysis_failure_reported", "vps_compilation_fallback_enabled"}, "farm evidence")
    if farm["analysis_failure_reported"] is not True or farm["vps_compilation_fallback_enabled"] is not False:
        raise B3GateError("farm evidence hides analysis failure or restores VPS compilation")
    if evidence["failures"] != [] or evidence["passed"] is not True:
        raise B3GateError("bundle evidence is not passing")
    _reject_placeholders(evidence, "bundle evidence")
    return dict(evidence)


def validate_b3_gate(value: object) -> dict[str, Any]:
    """Validate the exact self-contained B3 predecessor contract and seal."""

    try:
        authority = b2_gate._require_active_final_authority()
    except b2_gate.B2GateError as exc:
        raise B3GateError(f"B2 predecessor authority rejected: {exc}") from exc

    report = _mapping(value, "B3 gate")
    _exact_keys(
        report,
        {
            "schema", "batch", "status", "evidence_kind", "synthetic_claims",
            "normative_documents", "repository", "predecessor", "design_prior",
            "recovery_guide", "bundle", "component_evidence", "deployment",
            "gate", "gate_sha256",
        },
        "B3 gate",
    )
    if (
        report["schema"] != GATE_SCHEMA
        or report["batch"] != "B3"
        or report["status"] != "green"
        or report["evidence_kind"] != "measured-offline"
        or report["synthetic_claims"] is not False
    ):
        raise B3GateError("B3 gate identity or evidence kind differs")
    _digest(report["gate_sha256"], "B3 gate seal")
    if report["gate_sha256"] != gate_sha256(report):
        raise B3GateError("B3 gate seal differs")

    normative = _mapping(report["normative_documents"], "B3 normative documents")
    _exact_keys(normative, {"design_sha256", "plan_sha256"}, "B3 normative documents")
    for name, digest in normative.items():
        _digest(digest, f"B3 normative {name}")

    repository = _mapping(report["repository"], "B3 repository")
    _exact_keys(
        repository, {"repository_commit", "repository_tree", "git_clean"}, "B3 repository",
    )
    _digest(repository["repository_commit"], "B3 repository commit", git=True)
    _digest(repository["repository_tree"], "B3 repository tree", git=True)
    if repository["git_clean"] is not True:
        raise B3GateError("B3 repository is not clean")

    components = _mapping(report["component_evidence"], "B3 component evidence")
    _exact_keys(
        components,
        {"b2_gate", "design_prior", "recovery_guide", "bundle"},
        "B3 component evidence",
    )

    predecessor = _mapping(report["predecessor"], "B3 predecessor")
    _exact_keys(
        predecessor,
        {"b2_gate", "status", "cohort_id", "declaration_sha256"},
        "B3 predecessor",
    )
    _validate_file_record(predecessor["b2_gate"], "B3 predecessor B2 gate")
    if (
        predecessor["status"] != "green"
        or predecessor["cohort_id"] != authority.cohort_id
        or predecessor["cohort_id"] == b2_gate.RETIRED_COHORT_71446
        or predecessor["declaration_sha256"] != authority.declaration_sha256
    ):
        raise B3GateError("B3 predecessor authority differs")
    if _embedded_file_record(components["b2_gate"]) != predecessor["b2_gate"]:
        raise B3GateError("embedded B2 predecessor bytes differ")
    embedded_b2 = validate_b2_gate(components["b2_gate"], normative)
    if (
        embedded_b2["generated_cohort"]["cohort_id"] != predecessor["cohort_id"]
        or embedded_b2["generated_cohort"]["declaration_sha256"]
        != predecessor["declaration_sha256"]
    ):
        raise B3GateError("embedded B2 predecessor summary differs")

    prior = _mapping(report["design_prior"], "B3 design prior")
    _exact_keys(
        prior,
        {
            "report", "campaign_id", "pair_count", "metrics_improved",
            "static_pass_rate_preserved", "diversity_passed",
        },
        "B3 design prior",
    )
    _validate_file_record(prior["report"], "B3 design-prior report")
    if (
        not isinstance(prior["campaign_id"], str)
        or re.fullmatch(r"[a-z0-9][a-z0-9_.-]{0,63}", prior["campaign_id"]) is None
    ):
        raise B3GateError("B3 campaign ID is not canonical")
    if prior["pair_count"] != 28 or any(
        prior[name] is not True for name in (
            "metrics_improved", "static_pass_rate_preserved", "diversity_passed",
        )
    ):
        raise B3GateError("B3 design-prior predicates differ")
    if _embedded_file_record(components["design_prior"]) != prior["report"]:
        raise B3GateError("embedded design-prior bytes differ")
    try:
        embedded_prior = validate_campaign_report(components["design_prior"])
        prior_impl = validate_implementation(embedded_prior["implementation"])
    except B3PriorError as exc:
        raise B3GateError(f"embedded design-prior evidence rejected: {exc}") from exc
    if (
        embedded_prior.get("campaign_id") != prior["campaign_id"]
        or embedded_prior.get("status") != "green"
        or embedded_prior.get("passed") is not True
        or embedded_prior.get("normative_documents") != normative
        or any(prior_impl[key] != repository[key] for key in repository)
        or embedded_prior["lanes"]["baseline"]["map_count"]
        != prior["pair_count"]
        or embedded_prior["decision"]["all_metrics_improved"]
        is not prior["metrics_improved"]
        or embedded_prior["decision"]["static_pass_rate_preserved"]
        is not prior["static_pass_rate_preserved"]
        or all(
            embedded_prior["decision"][key]
            for key in (
                "layout_diversity_passed", "style_diversity_passed",
                "descriptor_diversity_passed",
            )
        ) is not prior["diversity_passed"]
    ):
        raise B3GateError("embedded design-prior summary/source differs")

    recovery = _mapping(report["recovery_guide"], "B3 recovery/guide")
    _exact_keys(
        recovery,
        {
            "report", "source_closure", "atlas_set_sha256", "finite_non_safe_cells",
            "unresolved_cells", "spatial_width", "hook_walk_budget_ticks",
            "game_tick_hz", "walk_speed_q8_per_second", "rust_extension",
        },
        "B3 recovery/guide",
    )
    _validate_file_record(recovery["report"], "B3 recovery/guide report")
    recovery_closure = _validate_closure_record(
        recovery["source_closure"], "B3 recovery/guide source closure",
    )
    _digest(recovery["atlas_set_sha256"], "B3 recovery Atlas-set SHA-256")
    _integer(recovery["finite_non_safe_cells"], "B3 finite non-safe cells", 1)
    _integer(recovery["unresolved_cells"], "B3 unresolved recovery cells", 0, 0)
    if (
        recovery["spatial_width"] != 76
        or recovery["hook_walk_budget_ticks"] != 15
        or recovery["game_tick_hz"] != 10
        or recovery["walk_speed_q8_per_second"] != 76_800
    ):
        raise B3GateError("B3 recovery packing or physics identity differs")
    if _embedded_file_record(components["recovery_guide"]) != recovery["report"]:
        raise B3GateError("embedded recovery/guide bytes differ")
    embedded_recovery = validate_recovery_guide_evidence(
        components["recovery_guide"], repository, recovery_closure,
    )
    if (
        embedded_recovery["atlas_set_sha256"] != recovery["atlas_set_sha256"]
        or embedded_recovery["recovery"]["finite_non_safe_cells"]
        != recovery["finite_non_safe_cells"]
        or embedded_recovery["recovery"]["unresolved_cells"]
        != recovery["unresolved_cells"]
    ):
        raise B3GateError("embedded recovery/guide summary differs")
    extension = _mapping(recovery["rust_extension"], "B3 Rust extension")
    _exact_keys(
        extension,
        {
            "path", "bytes", "sha256", "repository_tree",
            "source_closure_sha256", "qualification_commands_sha256",
        },
        "B3 Rust extension",
    )
    if (
        not isinstance(extension["path"], str)
        or not extension["path"]
        or not Path(extension["path"]).is_absolute()
    ):
        raise B3GateError("B3 Rust extension path is not absolute")
    _integer(extension["bytes"], "B3 Rust extension bytes", 1)
    for name in ("sha256", "source_closure_sha256", "qualification_commands_sha256"):
        _digest(extension[name], f"B3 Rust extension {name}")
    _digest(extension["repository_tree"], "B3 Rust extension repository tree", git=True)
    if (
        extension["repository_tree"] != repository["repository_tree"]
        or extension["source_closure_sha256"] != recovery_closure["sha256"]
        or embedded_recovery["rust_extension"] != extension
        or embedded_recovery["recovery"][
            "hook_necessity_walking_budget_ticks"
        ] != recovery["hook_walk_budget_ticks"]
        or embedded_recovery["recovery"]["hook_necessity_game_tick_hz"]
        != recovery["game_tick_hz"]
        or embedded_recovery["recovery"][
            "hook_necessity_walk_speed_q8_per_second"
        ] != recovery["walk_speed_q8_per_second"]
        or embedded_recovery["recovery"]["recovery_width"]
        + embedded_recovery["guide"]["guide_width"]
        != recovery["spatial_width"]
    ):
        raise B3GateError("B3 Rust extension source/repository identity differs")

    bundle = _mapping(report["bundle"], "B3 bundle")
    _exact_keys(
        bundle,
        {"report", "source_closure", "v2_compatible", "v3_isolated", "public_enabled"},
        "B3 bundle",
    )
    _validate_file_record(bundle["report"], "B3 bundle report")
    _validate_closure_record(bundle["source_closure"], "B3 bundle source closure")
    if (
        bundle["v2_compatible"] is not True
        or bundle["v3_isolated"] is not True
        or bundle["public_enabled"] is not False
    ):
        raise B3GateError("B3 bundle predicates differ")
    if _embedded_file_record(components["bundle"]) != bundle["report"]:
        raise B3GateError("embedded bundle bytes differ")
    embedded_bundle = validate_bundle_evidence(
        components["bundle"], repository,
        _validate_closure_record(
            bundle["source_closure"], "B3 bundle source closure"
        ),
    )
    if (
        all(embedded_bundle["bundle_v2"].values()) is not True
        or embedded_bundle["bundle_v3"]["isolated_runtime_only"] is not True
        or embedded_bundle["bundle_v3"]["public_enabled"] is not False
    ):
        raise B3GateError("embedded bundle summary differs")

    deployment = _mapping(report["deployment"], "B3 deployment")
    deployment_keys = {
        "public_or_teacher_service_changed", "cross_host_runtime_copy_performed",
        "trainer_or_tensorboard_started",
    }
    _exact_keys(deployment, deployment_keys, "B3 deployment")
    if any(deployment[name] is not False for name in deployment_keys):
        raise B3GateError("B3 gate records a forbidden deployment")

    decision = _mapping(report["gate"], "B3 decision")
    decision_keys = {
        "finite_cells_descend_or_plateau", "spatial_addition_width",
        "matched_treatment_improved", "static_pass_rate_preserved",
        "diversity_preserved", "v2_compatibility_passed", "bundle_v3_isolated",
        "failures", "green",
    }
    _exact_keys(decision, decision_keys, "B3 decision")
    if decision["spatial_addition_width"] != 76 or decision["failures"] != []:
        raise B3GateError("B3 decision width or failures differ")
    if any(
        decision[name] is not True
        for name in decision_keys - {"spatial_addition_width", "failures"}
    ):
        raise B3GateError("B3 decision is not completely green")

    _reject_placeholders(report, "B3 gate")
    return dict(report)


def _exclusive_write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except OSError as exc:
        raise B3GateError(f"gate output already exists or cannot be created: {path}: {exc}") from exc
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(canonical_bytes(value))
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise


def _build_b3_gate_report(
    paths: B3GatePaths,
    *,
    repository: Mapping[str, Any],
) -> dict[str, Any]:
    """Build and validate an in-memory gate without publishing evidence."""

    repo_root = paths.repo_root.resolve()
    repository = dict(repository)
    _exact_keys(repository, {"repository_commit", "repository_tree", "git_clean"}, "repository identity")
    _digest(repository["repository_commit"], "repository commit", git=True)
    _digest(repository["repository_tree"], "repository tree", git=True)
    if repository["git_clean"] is not True:
        raise B3GateError("repository identity is not clean")
    normative = {
        "design_sha256": file_sha256(repo_root / "docs/MULTIRES-LATTICE-MAP-ATLAS-DESIGN-2026-07-14.md"),
        "plan_sha256": file_sha256(repo_root / "docs/MULTIRES-LATTICE-MAP-ATLAS-PLAN-2026-07-14.md"),
    }
    b2 = validate_b2_gate(_load(paths.b2_gate), normative)
    try:
        prior = validate_campaign_report(_load(paths.prior_campaign))
    except B3PriorError as exc:
        raise B3GateError(str(exc)) from exc
    if prior["passed"] is not True or prior["status"] != "green":
        raise B3GateError("design-prior campaign is not green")
    if prior["normative_documents"] != normative:
        raise B3GateError("design-prior campaign normative identity drifted")
    prior_impl = validate_implementation(prior["implementation"])
    for key in ("repository_commit", "repository_tree", "git_clean"):
        if prior_impl[key] != repository[key]:
            raise B3GateError(f"design-prior repository identity drifted at {key}")
    recovery_closure = _source_closure(repo_root, RECOVERY_GUIDE_SOURCE_PATHS)
    bundle_closure = _source_closure(repo_root, BUNDLE_SOURCE_PATHS)
    recovery = validate_recovery_guide_evidence(
        _load(paths.recovery_guide_evidence), repository, recovery_closure,
    )
    bundle = validate_bundle_evidence(
        _load(paths.bundle_evidence), repository, bundle_closure,
    )
    bundle_v2_compatible = all(bundle["bundle_v2"].values())
    bundle_v3_isolated = bundle["bundle_v3"]["isolated_runtime_only"]
    bundle_v3_public_enabled = bundle["bundle_v3"]["public_enabled"]
    report = {
        "schema": GATE_SCHEMA,
        "batch": "B3",
        "status": "green",
        "evidence_kind": "measured-offline",
        "synthetic_claims": False,
        "normative_documents": normative,
        "repository": repository,
        "predecessor": {
            "b2_gate": _file_record(paths.b2_gate),
            "status": b2["status"],
            "cohort_id": b2["generated_cohort"]["cohort_id"],
            "declaration_sha256": b2["generated_cohort"]["declaration_sha256"],
        },
        "design_prior": {
            "report": _file_record(paths.prior_campaign),
            "campaign_id": prior["campaign_id"],
            "pair_count": prior["lanes"]["baseline"]["map_count"],
            "metrics_improved": prior["decision"]["all_metrics_improved"],
            "static_pass_rate_preserved": prior["decision"]["static_pass_rate_preserved"],
            "diversity_passed": all(prior["decision"][key] for key in (
                "layout_diversity_passed", "style_diversity_passed",
                "descriptor_diversity_passed",
            )),
        },
        "recovery_guide": {
            "report": _file_record(paths.recovery_guide_evidence),
            "source_closure": recovery_closure,
            "atlas_set_sha256": recovery["atlas_set_sha256"],
            "finite_non_safe_cells": recovery["recovery"]["finite_non_safe_cells"],
            "unresolved_cells": recovery["recovery"]["unresolved_cells"],
            "spatial_width": recovery["recovery"]["recovery_width"] + recovery["guide"]["guide_width"],
            "hook_walk_budget_ticks": recovery["recovery"]["hook_necessity_walking_budget_ticks"],
            "game_tick_hz": recovery["recovery"]["hook_necessity_game_tick_hz"],
            "walk_speed_q8_per_second": recovery["recovery"]["hook_necessity_walk_speed_q8_per_second"],
            "rust_extension": recovery["rust_extension"],
        },
        "bundle": {
            "report": _file_record(paths.bundle_evidence),
            "source_closure": bundle_closure,
            "v2_compatible": bundle_v2_compatible,
            "v3_isolated": bundle_v3_isolated,
            "public_enabled": bundle_v3_public_enabled,
        },
        "component_evidence": {
            "b2_gate": b2,
            "design_prior": prior,
            "recovery_guide": recovery,
            "bundle": bundle,
        },
        "deployment": {
            "public_or_teacher_service_changed": False,
            "cross_host_runtime_copy_performed": False,
            "trainer_or_tensorboard_started": False,
        },
        "gate": {
            "finite_cells_descend_or_plateau": True,
            "spatial_addition_width": 76,
            "matched_treatment_improved": True,
            "static_pass_rate_preserved": True,
            "diversity_preserved": True,
            "v2_compatibility_passed": bundle_v2_compatible,
            "bundle_v3_isolated": bundle_v3_isolated,
            "failures": [],
            "green": True,
        },
    }
    report["gate_sha256"] = gate_sha256(report)
    validate_b3_gate(report)
    return report


def assemble_b3_gate(paths: B3GatePaths) -> dict[str, Any]:
    """Read the real clean repository identity and publish one B3 gate."""

    if paths.output.exists() or paths.output.is_symlink():
        raise B3GateError("B3 gate output already exists")
    repository = repository_identity(paths.repo_root.resolve())
    report = _build_b3_gate_report(paths, repository=repository)
    _exclusive_write(paths.output, report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--b2-gate", type=Path, required=True)
    parser.add_argument("--prior-campaign", type=Path, required=True)
    parser.add_argument("--recovery-guide-evidence", type=Path, required=True)
    parser.add_argument("--bundle-evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = assemble_b3_gate(B3GatePaths(
            repo_root=args.repo_root,
            b2_gate=args.b2_gate,
            prior_campaign=args.prior_campaign,
            recovery_guide_evidence=args.recovery_guide_evidence,
            bundle_evidence=args.bundle_evidence,
            output=args.output,
        ))
    except B3GateError as exc:
        print(f"B3 gate refused: {exc}", file=sys.stderr)
        return 2
    sys.stdout.buffer.write(canonical_bytes({
        "schema": GATE_SCHEMA, "status": report["status"],
        "output_sha256": file_sha256(args.output),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
