from __future__ import annotations

import copy
import hashlib
import inspect
import json
from pathlib import Path

import pytest

import tools.assemble_b2_gate as b2_gate
from tools.assemble_b3_gate import (
    B3GateError,
    B3GatePaths,
    BUNDLE_SCHEMA,
    BUNDLE_TEST_COMMANDS,
    GATE_SCHEMA,
    HAZARD_CLASSES,
    OBJECTIVE_CLASSES,
    RECOVERY_GUIDE_SCHEMA,
    RECOVERY_TEST_COMMANDS,
    BUNDLE_SOURCE_PATHS,
    RECOVERY_GUIDE_SOURCE_PATHS,
    _build_b3_gate_report,
    _source_closure,
    _bundle_sections_from_claims,
    assemble_b3_gate,
    derive_bundle_claim_evidence,
    gate_sha256,
    validate_b3_gate,
)
from tools.assemble_b2_gate import ActiveFinalAuthority, RETIRED_COHORT_71446
from tools.assemble_b2_qualification import activation_successor_policy
from tools.run_b3_design_prior_campaign import canonical_bytes
from tests.test_b3_design_prior_campaign import build_green_campaign


ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = {"repository_commit": "1" * 40, "repository_tree": "2" * 40, "git_clean": True}
AUTHORITY_PATH = activation_successor_policy()["immutable_declaration_path"]
AUTHORITY = ActiveFinalAuthority(
    cohort_id=activation_successor_policy()["cohort_id"],
    declaration_sha256="3" * 64,
    immutable_declaration_path=AUTHORITY_PATH,
    qualification_successor_paths=frozenset(
        activation_successor_policy()["allowed_changed_paths"]
    ),
)


@pytest.fixture(autouse=True)
def _active_b2_authority(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise B3 under an explicit, non-retired successor authority."""

    monkeypatch.setattr(b2_gate, "ACTIVE_FINAL_AUTHORITY", AUTHORITY)


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(value))


def _component_implementation(closure: dict) -> dict:
    return {**REPOSITORY, "source_closure": closure}


def _tests(commands: tuple[tuple[str, ...], ...], marker: bytes) -> dict:
    runs = []
    for command in commands:
        runs.append({
            "command": list(command),
            "exit_code": 0,
            "passed_count": 1,
            "stdout": {"bytes": len(marker), "sha256": hashlib.sha256(marker).hexdigest()},
            "stderr": {"bytes": 0, "sha256": hashlib.sha256(b"").hexdigest()},
        })
    body = {"schema": "q2-b3-component-test-runs-v1", "runs": runs}
    return {
        "report_sha256": hashlib.sha256(canonical_bytes(body)).hexdigest(),
        "commands_sha256": hashlib.sha256(
            canonical_bytes([list(command) for command in commands])
        ).hexdigest(),
        "passed_count": len(runs), "failed_count": 0, "runs": runs,
    }


def _recovery(closure: dict, extension: Path) -> dict:
    tests = _tests(RECOVERY_TEST_COMMANDS, b"recovery tests\n")
    return {
        "schema": RECOVERY_GUIDE_SCHEMA,
        "evidence_kind": "measured-offline-atlas",
        "synthetic_claims": False,
        "implementation": _component_implementation(closure),
        "rust_extension": {
            "path": str(extension.resolve()),
            "bytes": extension.stat().st_size,
            "sha256": hashlib.sha256(extension.read_bytes()).hexdigest(),
            "repository_tree": REPOSITORY["repository_tree"],
            "source_closure_sha256": closure["sha256"],
            "qualification_commands_sha256": tests["commands_sha256"],
        },
        "atlas_set_sha256": "5" * 64,
        "map_count": 36,
        "recovery": {
            "finite_non_safe_cells": 100,
            "strict_descending_cells": 97,
            "mover_plateau_cells": 3,
            "unresolved_cells": 0,
            "max_local_repair_nodes": 4096,
            "hazard_classes": HAZARD_CLASSES,
            "hook_necessity_walking_budget_ticks": 15,
            "hook_necessity_game_tick_hz": 10,
            "hook_necessity_walk_speed_q8_per_second": 76800,
            "hook_necessity_evaluated_edges": 12,
            "hook_necessity_query_cells": 8,
            "hook_necessity_path_to_safety_cases": 5,
            "hook_necessity_positive_cases": 3,
            "recovery_width": 16,
        },
        "guide": {
            "guide_width": 60,
            "candidate_count": 4,
            "candidate_width": 15,
            "objective_classes": OBJECTIVE_CLASSES,
            "objective_bearing_fixtures": 8,
            "universal_zero_objective_fixtures": 0,
        },
        "tests": tests,
        "failures": [],
        "passed": True,
    }


def _bundle(closure: dict) -> dict:
    tests = _tests(BUNDLE_TEST_COMMANDS, b"bundle tests\n")
    claim_tests = derive_bundle_claim_evidence(tests)
    sections = _bundle_sections_from_claims(claim_tests)
    return {
        "schema": BUNDLE_SCHEMA,
        "evidence_kind": "measured-offline-installer",
        "synthetic_claims": False,
        "implementation": _component_implementation(closure),
        "claim_tests": claim_tests,
        "bundle_v2": sections["bundle_v2"],
        "bundle_v3": sections["bundle_v3"],
        "farm": sections["farm"],
        "tests": tests,
        "failures": [],
        "passed": True,
    }


def _b2_final_lifecycle(authority: ActiveFinalAuthority) -> dict:
    """Minimal but exact-shape B2 lifecycle summary for B3 predecessor tests."""

    return {
        "evidence": {"bytes": 1, "sha256": "7" * 64},
        "schema": b2_gate.FINAL_LIFECYCLE_EVIDENCE_SCHEMA,
        "status": "ready-for-assembly",
        "cohort_id": authority.cohort_id,
        "declaration": {
            "path": str((ROOT / authority.immutable_declaration_path).resolve()),
            "sha256": authority.declaration_sha256,
        },
        "plan_sha256": "8" * 64,
        "implementation": {},
        "execution_binding": {
            "schema": "q2-b2-final-execution-binding-v2",
            "host": {
                "hostname": "DESKTOP-RTX2080",
                "kernel_release": "5.15.153.1-microsoft-standard-WSL2",
                "machine_identity": {
                    "path": "/etc/machine-id",
                    "sha256": "9" * 64,
                },
                "euid": 1000,
            },
            "state_root": {
                "path": "/tmp/q2-b2-final-journal",
                "owner_uid": 1000,
                "mode": "0700",
                "device": 1,
                "inode": 1,
            },
        },
        "source_authorization_marker": {
            "path": "/tmp/q2-b2-final-journal/marker.json",
            "bytes": 1,
            "sha256": "a" * 64,
        },
        "completed_stages": list(b2_gate.FINAL_LIFECYCLE_PREASSEMBLY_STAGES),
        "stage_executions": [
            {
                "stage": stage,
                "command_sha256": "b" * 64,
                "returncode": 0,
                "stdout": {"bytes": 0, "sha256": "c" * 64},
                "stderr": {"bytes": 0, "sha256": "d" * 64},
            }
            for stage in b2_gate.FINAL_LIFECYCLE_PREASSEMBLY_STAGES
        ],
        "assembly_command_sha256": "e" * 64,
    }


def _b2(authority: ActiveFinalAuthority = AUTHORITY) -> dict:
    design = ROOT / "docs/MULTIRES-LATTICE-MAP-ATLAS-DESIGN-2026-07-14.md"
    plan = ROOT / "docs/MULTIRES-LATTICE-MAP-ATLAS-PLAN-2026-07-14.md"
    return {
        "schema": "q2-multires-b2-gate-v1",
        "batch": "B2",
        "status": "green",
        "owner_directive": {
            "replacement": "one-way",
            "legacy_model_lineages": "retired",
            "operational_fallback": "forbidden",
            "legacy_runtime_or_model_used": False,
        },
        "normative_documents": {
            "design": {"bytes": design.stat().st_size, "sha256": hashlib.sha256(design.read_bytes()).hexdigest()},
            "plan": {"bytes": plan.stat().st_size, "sha256": hashlib.sha256(plan.read_bytes()).hexdigest()},
        },
        "implementation": {},
        "b1_authority": {},
        "toolchain_qualification": {
            "schema": b2_gate.QUALIFICATION_SCHEMA,
            "non_admissible": True,
            "retryable": True,
            "final_cohort_authorized": False,
            "end_to_end_pass_count": b2_gate.FINAL_QUALIFICATION_END_TO_END_PASSES,
            "activation_successor_policy": activation_successor_policy(),
            "implementation_successor": {
                "changed_paths": sorted(
                    activation_successor_policy()["allowed_changed_paths"]
                ),
            },
            "preactivation_tests": {
                "report": {"bytes": 1, "sha256": "0" * 64},
                "run_count": 8,
                "passed_count": 8,
            },
        },
        "final_lifecycle": _b2_final_lifecycle(authority),
        "generated_cohort": {
            "cohort_id": authority.cohort_id,
            "declaration_sha256": authority.declaration_sha256,
            "compiled_cm_preflight": {
                "schema": b2_gate.COMPILED_CM_PREFLIGHT_SCHEMA,
                "admission_status": b2_gate.COMPILED_CM_PREFLIGHT_STATUS,
                "pass_count": 28,
            },
        },
        "stock_corpus": {"map_count": 8},
        "representative_budgets": {},
        "dyn_evidence": {},
        "tests": {},
        "deployment": {
            "public_or_teacher_service_changed": False,
            "cross_host_runtime_copy_performed": False,
            "trainer_or_tensorboard_started": False,
        },
        "gate": {
            "toolchain_qualification_non_admissible": True,
            "compiled_cm_preflight_maps_passed": 28,
            "failures": [],
            "green": True,
        },
    }


def _fixture(tmp_path: Path) -> tuple[B3GatePaths, dict, dict]:
    _campaign, campaign_path = build_green_campaign(tmp_path / "prior")
    recovery_closure = _source_closure(ROOT, RECOVERY_GUIDE_SOURCE_PATHS)
    bundle_closure = _source_closure(ROOT, BUNDLE_SOURCE_PATHS)
    extension = tmp_path / "libq2_lattice_rs.so"
    extension.write_bytes(b"exact extension fixture")
    recovery = _recovery(recovery_closure, extension)
    bundle = _bundle(bundle_closure)
    b2_path = tmp_path / "b2.json"
    recovery_path = tmp_path / "recovery.json"
    bundle_path = tmp_path / "bundle.json"
    _write(b2_path, _b2())
    _write(recovery_path, recovery)
    _write(bundle_path, bundle)
    return B3GatePaths(
        repo_root=ROOT,
        b2_gate=b2_path,
        prior_campaign=campaign_path,
        recovery_guide_evidence=recovery_path,
        bundle_evidence=bundle_path,
        output=tmp_path / "B3-GATE.json",
    ), recovery, bundle


def test_gate_assembles_exact_green_evidence(tmp_path: Path, monkeypatch):
    paths, _recovery_doc, _bundle_doc = _fixture(tmp_path)
    monkeypatch.setattr(
        "tools.assemble_b3_gate.repository_identity",
        lambda _repo_root: dict(REPOSITORY),
    )
    report = assemble_b3_gate(paths)
    assert report["schema"] == GATE_SCHEMA
    assert report["gate"]["green"] is True
    assert report["gate_sha256"] == gate_sha256(report)
    assert validate_b3_gate(report) == report
    assert report["recovery_guide"]["spatial_width"] == 76
    assert report["bundle"] == {
        "report": report["bundle"]["report"],
        "source_closure": report["bundle"]["source_closure"],
        "v2_compatible": True,
        "v3_isolated": True,
        "public_enabled": False,
    }
    assert paths.output.read_bytes() == canonical_bytes(report)


def test_production_gate_api_has_no_repository_bypass_and_refuses_existing_output(
    tmp_path: Path,
):
    signature = inspect.signature(assemble_b3_gate)
    assert tuple(signature.parameters) == ("paths",)
    assert all(
        parameter.kind is not inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )

    paths, _recovery_doc, _bundle_doc = _fixture(tmp_path)
    paths.output.write_text("owned")
    with pytest.raises(B3GateError, match="output already exists"):
        assemble_b3_gate(paths)


def test_no_active_b2_authority_blocks_assembly_and_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, _recovery_doc, _bundle_doc = _fixture(tmp_path)
    report = _build_b3_gate_report(paths, repository=REPOSITORY)

    monkeypatch.setattr(b2_gate, "ACTIVE_FINAL_AUTHORITY", None)
    with pytest.raises(B3GateError, match="no active final cohort"):
        _build_b3_gate_report(paths, repository=REPOSITORY)
    with pytest.raises(B3GateError, match="no active final cohort"):
        validate_b3_gate(report)


def test_gate_rejects_forged_minimal_b2_before_other_evidence(tmp_path: Path) -> None:
    paths, _recovery_doc, _bundle_doc = _fixture(tmp_path)
    _write(paths.b2_gate, {
        "schema": "q2-multires-b2-gate-v1",
        "batch": "B2",
        "status": "green",
        "gate": {"failures": [], "green": True},
    })

    with pytest.raises(
        B3GateError, match=r"B2 predecessor authority rejected: .*keys differ",
    ):
        _build_b3_gate_report(paths, repository=REPOSITORY)


def test_gate_rejects_retired_71446_even_as_process_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    retired_path = "docs/multires/B2-GENERATED-COHORT-71446-DECLARATION.json"
    retired = ActiveFinalAuthority(
        cohort_id=RETIRED_COHORT_71446,
        declaration_sha256="4" * 64,
        immutable_declaration_path=retired_path,
        qualification_successor_paths=frozenset({retired_path}),
    )
    monkeypatch.setattr(b2_gate, "ACTIVE_FINAL_AUTHORITY", retired)
    paths, _recovery_doc, _bundle_doc = _fixture(tmp_path)
    _write(paths.b2_gate, _b2(retired))

    with pytest.raises(B3GateError, match="active authority is retired"):
        _build_b3_gate_report(paths, repository=REPOSITORY)


def test_gate_rejects_b2_cohort_and_declaration_authority_mismatch(tmp_path: Path) -> None:
    paths, _recovery_doc, _bundle_doc = _fixture(tmp_path)
    wrong_cohort = copy.deepcopy(_b2())
    wrong_cohort["generated_cohort"]["cohort_id"] = "b2g98_final_99998"
    _write(paths.b2_gate, wrong_cohort)
    with pytest.raises(B3GateError, match="B2 gate cohort differs"):
        _build_b3_gate_report(paths, repository=REPOSITORY)

    wrong_declaration = copy.deepcopy(_b2())
    wrong_declaration["generated_cohort"]["declaration_sha256"] = "5" * 64
    _write(paths.b2_gate, wrong_declaration)
    with pytest.raises(B3GateError, match="declaration differs from active authority"):
        _build_b3_gate_report(paths, repository=REPOSITORY)


def test_gate_rejects_b2_successor_paths_rebound_from_the_frozen_policy(
    tmp_path: Path,
) -> None:
    paths, _recovery_doc, _bundle_doc = _fixture(tmp_path)
    rebound = copy.deepcopy(_b2())
    rebound["toolchain_qualification"]["implementation_successor"][
        "changed_paths"
    ].append("tools/unauthorized_activation_tool.py")
    _write(paths.b2_gate, rebound)

    with pytest.raises(B3GateError, match="successor paths differ"):
        _build_b3_gate_report(paths, repository=REPOSITORY)


def test_sealed_b3_predecessor_is_rejected_after_authority_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, _recovery_doc, _bundle_doc = _fixture(tmp_path)
    report = _build_b3_gate_report(paths, repository=REPOSITORY)
    successor_path = "docs/multires/B2-GENERATED-COHORT-100000-DECLARATION.json"
    monkeypatch.setattr(
        b2_gate,
        "ACTIVE_FINAL_AUTHORITY",
        ActiveFinalAuthority(
            cohort_id="b2g100_final_100000",
            declaration_sha256="6" * 64,
            immutable_declaration_path=successor_path,
            qualification_successor_paths=frozenset({successor_path}),
        ),
    )

    with pytest.raises(B3GateError, match="B3 predecessor authority differs"):
        validate_b3_gate(report)


def test_gate_seal_and_exact_structure_reject_minimal_or_tampered_predecessor(tmp_path: Path):
    paths, _recovery_doc, _bundle_doc = _fixture(tmp_path)
    report = _build_b3_gate_report(paths, repository=REPOSITORY)

    tampered = copy.deepcopy(report)
    tampered["design_prior"]["campaign_id"] = "different_campaign"
    with pytest.raises(B3GateError, match="gate seal differs"):
        validate_b3_gate(tampered)

    extra = copy.deepcopy(report)
    extra["unsealed_note"] = "not admitted"
    extra["gate_sha256"] = gate_sha256(extra)
    with pytest.raises(B3GateError, match="keys differ"):
        validate_b3_gate(extra)

    minimal = {"schema": GATE_SCHEMA, "status": "green", "green": True}
    minimal["gate_sha256"] = gate_sha256(minimal)
    with pytest.raises(B3GateError, match="keys differ"):
        validate_b3_gate(minimal)


def test_resealed_gate_cannot_replace_measured_component_with_green_summary(
    tmp_path: Path,
) -> None:
    paths, _recovery_doc, _bundle_doc = _fixture(tmp_path)
    report = _build_b3_gate_report(paths, repository=REPOSITORY)
    forged = copy.deepcopy(report)
    component = forged["component_evidence"]["recovery_guide"]
    component["recovery"]["unresolved_cells"] = 1
    payload = canonical_bytes(component)
    forged["recovery_guide"]["report"] = {
        "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest(),
    }
    forged["gate_sha256"] = gate_sha256(forged)
    with pytest.raises(B3GateError, match="unresolved recovery cells"):
        validate_b3_gate(forged)


def test_gate_rejects_unresolved_recovery_and_width_forgery(tmp_path: Path):
    paths, recovery, _bundle_doc = _fixture(tmp_path)
    recovery["recovery"]["unresolved_cells"] = 1
    _write(paths.recovery_guide_evidence, recovery)
    with pytest.raises(B3GateError, match="unresolved recovery cells"):
        _build_b3_gate_report(paths, repository=REPOSITORY)
    recovery["recovery"]["unresolved_cells"] = False
    _write(paths.recovery_guide_evidence, recovery)
    with pytest.raises(B3GateError, match="must be an integer"):
        _build_b3_gate_report(paths, repository=REPOSITORY)
    recovery["recovery"]["unresolved_cells"] = 0
    recovery["guide"]["guide_width"] = 59
    _write(paths.recovery_guide_evidence, recovery)
    with pytest.raises(B3GateError, match="guide packing"):
        _build_b3_gate_report(paths, repository=REPOSITORY)


def test_gate_rejects_source_identity_drift_and_synthetic_claims(tmp_path: Path):
    paths, recovery, _bundle_doc = _fixture(tmp_path)
    recovery["implementation"]["source_closure"]["sha256"] = "a" * 64
    _write(paths.recovery_guide_evidence, recovery)
    with pytest.raises(B3GateError, match="source closure identity drifted"):
        _build_b3_gate_report(paths, repository=REPOSITORY)
    recovery = _recovery(
        _source_closure(ROOT, RECOVERY_GUIDE_SOURCE_PATHS),
        Path(json.loads(paths.recovery_guide_evidence.read_text())["rust_extension"]["path"]),
    )
    recovery["synthetic_claims"] = True
    _write(paths.recovery_guide_evidence, recovery)
    with pytest.raises(B3GateError, match="synthetic"):
        _build_b3_gate_report(paths, repository=REPOSITORY)


def test_gate_rejects_public_bundle_v3_and_vps_fallback(tmp_path: Path):
    paths, _recovery_doc, bundle = _fixture(tmp_path)
    bundle["bundle_v3"]["public_enabled"] = True
    _write(paths.bundle_evidence, bundle)
    with pytest.raises(B3GateError, match="do not derive from named test nodes"):
        _build_b3_gate_report(paths, repository=REPOSITORY)
    bundle["bundle_v3"]["public_enabled"] = False
    bundle["farm"]["vps_compilation_fallback_enabled"] = True
    _write(paths.bundle_evidence, bundle)
    with pytest.raises(B3GateError, match="do not derive from named test nodes"):
        _build_b3_gate_report(paths, repository=REPOSITORY)


def test_gate_rejects_forged_bundle_claim_mapping(tmp_path: Path):
    paths, _recovery_doc, bundle = _fixture(tmp_path)
    claim = "bundle_v2.analysis_present_install_passed"
    bundle["claim_tests"][claim]["required_node_ids"] = [
        "tests/test_map_farm.py::test_stock_rotation_shuffles_without_repeats_in_a_cycle"
    ]
    _write(paths.bundle_evidence, bundle)
    with pytest.raises(B3GateError, match="claim evidence does not recompute"):
        _build_b3_gate_report(paths, repository=REPOSITORY)


def test_gate_recomputes_exact_test_commands_runs_and_extension(tmp_path: Path):
    paths, recovery, bundle = _fixture(tmp_path)
    bundle["tests"]["commands_sha256"] = "a" * 64
    _write(paths.bundle_evidence, bundle)
    with pytest.raises(B3GateError, match="command-list identity differs"):
        _build_b3_gate_report(paths, repository=REPOSITORY)

    _write(paths.bundle_evidence, _bundle(_source_closure(ROOT, BUNDLE_SOURCE_PATHS)))
    recovery["tests"]["runs"][0]["stdout"]["bytes"] += 1
    _write(paths.recovery_guide_evidence, recovery)
    with pytest.raises(B3GateError, match="report digest does not recompute"):
        _build_b3_gate_report(paths, repository=REPOSITORY)

    extension = Path(recovery["rust_extension"]["path"])
    extension.write_bytes(extension.read_bytes() + b"drift")
    recovery = _recovery(_source_closure(ROOT, RECOVERY_GUIDE_SOURCE_PATHS), extension)
    extension.write_bytes(extension.read_bytes() + b"post-evidence drift")
    _write(paths.recovery_guide_evidence, recovery)
    with pytest.raises(B3GateError, match="size drifted"):
        _build_b3_gate_report(paths, repository=REPOSITORY)


def test_gate_rejects_normative_and_repository_identity_drift(tmp_path: Path):
    paths, recovery, _bundle_doc = _fixture(tmp_path)
    b2 = json.loads(paths.b2_gate.read_text())
    b2["normative_documents"]["plan"]["sha256"] = "a" * 64
    _write(paths.b2_gate, b2)
    with pytest.raises(B3GateError, match="plan identity differs"):
        _build_b3_gate_report(paths, repository=REPOSITORY)
    _write(paths.b2_gate, _b2())
    recovery["implementation"]["repository_commit"] = "f" * 40
    _write(paths.recovery_guide_evidence, recovery)
    with pytest.raises(B3GateError, match="repository identity drifted"):
        _build_b3_gate_report(paths, repository=REPOSITORY)


def test_gate_schema_accepts_assembled_report(tmp_path: Path):
    jsonschema = pytest.importorskip("jsonschema")
    paths, _recovery_doc, _bundle_doc = _fixture(tmp_path)
    report = _build_b3_gate_report(paths, repository=REPOSITORY)
    schema = json.loads((ROOT / "schemas/q2-multires-b3-gate-v1.schema.json").read_text())
    jsonschema.Draft202012Validator(schema).validate(report)


def test_gate_schema_rejects_retired_71446_predecessor(tmp_path: Path):
    jsonschema = pytest.importorskip("jsonschema")
    paths, _recovery_doc, _bundle_doc = _fixture(tmp_path)
    report = _build_b3_gate_report(paths, repository=REPOSITORY)
    report["predecessor"]["cohort_id"] = RETIRED_COHORT_71446
    report["gate_sha256"] = gate_sha256(report)
    schema = json.loads((ROOT / "schemas/q2-multires-b3-gate-v1.schema.json").read_text())

    with pytest.raises(jsonschema.ValidationError):
        jsonschema.Draft202012Validator(schema).validate(report)
