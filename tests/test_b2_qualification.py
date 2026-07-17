from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import tools.assemble_b2_qualification as gate
import tools.assemble_b2_gate as final_gate
from tools.b2_qualification_toolchain import load_toolchain_authority
from tools.run_generator_cohort import CONCRETE_STYLES, canonical_bytes


SHA = "1" * 64
COMMIT = "2" * 40
TREE = "3" * 40


def _digest(label: str) -> str:
    return sha256(label.encode("ascii")).hexdigest()


def _write(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(value))
    return path


def _implementation() -> dict[str, object]:
    return {
        "repository_commit": COMMIT,
        "repository_tree": TREE,
        "git_clean": True,
        "atlas_analyzer_authority_sha256": _digest("analyzer"),
        "atlas_analyzer_authority_file_count": 7,
        "generator_sha256": _digest("generator"),
        "routes_sha256": _digest("routes"),
    }


def _declaration(implementation: dict[str, object]) -> dict[str, object]:
    rows = []
    ordinal = 0
    for style_index, style in enumerate(CONCRETE_STYLES):
        for member in range(4):
            seed = 880000 + style_index * 100 + member
            rows.append({
                "ordinal": ordinal,
                "map": f"b2q26_{style}_{seed}",
                "seed": seed,
                "style": style,
                "grid": 5,
                "observed_heat": None,
            })
            ordinal += 1
    return {
        "schema": gate.DECLARATION_SCHEMA,
        "qualification_id": "b2q26_test",
        "mode": "qualification",
        "non_admissible": True,
        "retryable": True,
        "final_cohort_authorized": False,
        "toolchain_authority_sha256": (
            load_toolchain_authority().manifest_sha256
        ),
        "generator": {
            "version": "v6", "grid": 5, "gym": False,
            "observed_heat": None,
        },
        "selection": {
            "required_map_count": 28,
            "required_concrete_styles": list(CONCRETE_STYLES),
            "required_maps_per_style": 4,
        },
        "implementation": implementation,
        "maps": rows,
    }


def _retired() -> dict[str, set[object]]:
    return {
        "cohorts": set(), "declarations": set(), "maps": set(),
        "seeds": set(), "digests": set(),
    }


def _stage(
    stage: str,
    declaration: dict[str, object],
    declaration_sha256: str,
    implementation: dict[str, object],
    input_sha256: str | None,
    pass_count: int,
) -> dict[str, object]:
    rows = []
    for item in declaration["maps"]:
        row = dict(item)
        passed = row["ordinal"] < pass_count
        rows.append({
            "ordinal": row["ordinal"],
            "map": row["map"],
            "criteria": {"producer-evidence-bound": passed},
            "evidence_sha256": _digest(f"{stage}:{row['map']}"),
            "failures": [] if passed else ["disposable map did not qualify"],
            "passed": passed,
        })
    return {
        "schema": gate.STAGE_SCHEMA,
        "qualification_id": declaration["qualification_id"],
        "mode": "qualification",
        "stage": stage,
        "non_admissible": True,
        "retryable": True,
        "final_cohort_authorized": False,
        "declaration_sha256": declaration_sha256,
        "implementation": implementation,
        "toolchain_authority_sha256": declaration[
            "toolchain_authority_sha256"
        ],
        "input_report_sha256": input_sha256,
        "infrastructure_checks": {
            name: True for name in gate.REQUIRED_STAGE_CHECKS[stage]
        },
        "map_count": 28,
        "pass_count": pass_count,
        "maps": rows,
        "failures": [],
    }


def _boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *, physics_identity: str | None = None,
) -> Path:
    toolchain = load_toolchain_authority()
    compile_report = {
        "toolchain_authority": toolchain.manifest_record(),
        "q2tool": {"sha256": toolchain.q2tool_sha256},
        "fixed_q2tool_flags": list(toolchain.q2tool_flags),
        "basedir": {
            "pak0": {"sha256": toolchain.pak0_sha256},
            "required_member": {"sha256": toolchain.colormap_sha256},
        },
        "fixtures": [
            {
                "case_id": fixture["case_id"],
                "source": {"sha256": fixture["sha256"]},
                "geometry": {
                    "floor_top_units": fixture["floor_top_units"],
                    "ceiling_bottom_units": fixture["ceiling_bottom_units"],
                    "spawn_origin_units": fixture["spawn_origin_units"],
                },
            }
            for fixture in toolchain.fixtures
        ],
    }
    compile_path = _write(tmp_path / "boundary-compile.json", compile_report)
    compile_record = {
        "path": str(compile_path.absolute()),
        "sha256": gate._file_sha256(compile_path),
        "size_bytes": compile_path.stat().st_size,
    }
    monkeypatch.setattr(
        gate,
        "load_boundary_compile_report",
        lambda path: (compile_report, compile_record),
    )
    proofs = []
    for case in gate.BOUNDARY_CASES:
        bsp_sha = _digest(case["case_id"])
        canonical_physics_identity = gate._sha256_bytes(
            (
                "schema=q2-physics-oracle-v1;kind=cm;tool_identity="
                f"{_digest('cm-tool')};map={bsp_sha}"
            ).encode("ascii")
        )
        proofs.append({
            "case_id": case["case_id"],
            "authored_floor_to_ceiling_units": case["ceiling_units"],
            "expected_pass": case["expected_pass"],
            "column_clear_96": case["expected_pass"],
            "column_clearance_milliunits": case["expected_clearance_units"] * 1000,
            "linked_standing_clear": True,
            "bsp": {"sha256": bsp_sha},
            "cm_identity": {
                "tool_identity": _digest("cm-tool"),
                "physics_identity": (
                    canonical_physics_identity
                    if physics_identity is None else physics_identity
                ),
                "map_sha256": bsp_sha,
            },
        })
    return _write(tmp_path / "boundary-proof.json", {
        "schema": gate.BOUNDARY_PROOF_SCHEMA,
        "status": "passed-non-admissible-qualification",
        "passed": True,
        "admission": gate.boundary_qualification_only(),
        "toolchain_authority": toolchain.manifest_record(),
        "contract": {
            "engine_link_lift_units": 9,
            "column_requirement_units": 96,
        },
        "cm_oracle": {"sha256": _digest("cm-executable")},
        "compile_evidence": compile_record,
        "q2tool": compile_report["q2tool"],
        "proofs": proofs,
    })


def _infrastructure(
    declaration: dict[str, object],
    declaration_sha256: str,
    implementation: dict[str, object],
    stage_sha256s: dict[str, str],
) -> dict[str, object]:
    checks = []
    for check_id in sorted(gate.REQUIRED_INFRASTRUCTURE_CHECKS):
        checks.append({
            "id": check_id,
            "criteria": {"recomputed-evidence-green": True},
            "evidence_sha256": _digest(f"infrastructure:{check_id}"),
            "failures": [],
            "passed": True,
        })
    return {
        "schema": gate.INFRASTRUCTURE_SCHEMA,
        "qualification_id": declaration["qualification_id"],
        "mode": "qualification",
        "non_admissible": True,
        "retryable": True,
        "final_cohort_authorized": False,
        "declaration_sha256": declaration_sha256,
        "implementation": implementation,
        "toolchain_authority_sha256": declaration[
            "toolchain_authority_sha256"
        ],
        "stage_report_sha256s": stage_sha256s,
        "checks": checks,
        "pass_count": len(checks),
        "failures": [],
    }


def _inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *, pass_count: int = 20,
) -> argparse.Namespace:
    implementation = _implementation()
    declaration = _declaration(implementation)
    declaration_path = _write(tmp_path / "qualification-declaration.json", declaration)
    declaration_sha256 = gate._file_sha256(declaration_path)
    stage_paths: dict[str, Path] = {}
    stage_sha256s: dict[str, str] = {}
    previous = None
    for stage in gate.STAGES:
        path = _write(
            tmp_path / f"{stage}.json",
            _stage(
                stage, declaration, declaration_sha256, implementation,
                previous, pass_count,
            ),
        )
        previous = gate._file_sha256(path)
        stage_paths[stage] = path
        stage_sha256s[stage] = previous
    infrastructure_path = _write(
        tmp_path / "infrastructure.json",
        _infrastructure(
            declaration, declaration_sha256, implementation, stage_sha256s
        ),
    )
    design = tmp_path / "design.md"
    plan = tmp_path / "plan.md"
    design.write_text("amended design\n")
    plan.write_text("amended plan\n")
    boundary = _boundary(tmp_path, monkeypatch)
    authority = SimpleNamespace(
        cm_executable_sha256=_digest("cm-executable"),
        oracle_tool_identity=_digest("cm-tool"),
    )
    monkeypatch.setattr(gate, "repository_binding", lambda root: implementation)
    monkeypatch.setattr(gate, "_retired_identities", lambda root: _retired())
    monkeypatch.setattr(
        gate,
        "_validate_requalification",
        lambda *args: (
            {"gate": {"bytes": 1, "sha256": SHA}, "requalification_sha256": SHA,
             "runtime_authority_seal_sha256": SHA,
             "reseal_repository": {
                 "commit": COMMIT, "tree": TREE, "clean": True,
             },
             "collision_identity": {
                 "tool_identity": _digest("cm-tool"),
                 "physics_identity": _digest("cm-physics"),
             }},
            authority,
            {
                "tool_identity": _digest("cm-tool"),
                "physics_identity": _digest("cm-physics"),
            },
        ),
    )
    return argparse.Namespace(
        design=design,
        plan=plan,
        repo_root=gate.ROOT,
        b1_gate=gate.ROOT / "docs/multires/B1-GATE.json",
        boundary_proof_report=boundary,
        declaration=declaration_path,
        source_report=stage_paths["source"],
        compile_report=stage_paths["compile"],
        compiled_cm_preflight_report=stage_paths["compiled-cm-preflight"],
        materialization_report=stage_paths["materialization"],
        claims_report=stage_paths["claims"],
        atlas_build_report=stage_paths["atlas-build"],
        generated_promotion_report=stage_paths["generated-promotion"],
        infrastructure_report=infrastructure_path,
        output=tmp_path / "qualification-output.json",
    )


def test_assembles_20_of_28_non_admissible_qualification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _inputs(tmp_path, monkeypatch)
    report = gate.assemble_qualification(args)
    assert report["status"] == "green"
    assert report["non_admissible"] is True
    assert report["retryable"] is True
    assert report["final_cohort_authorized"] is False
    assert report["end_to_end"]["pass_count"] == 20
    assert report["authorization"] == {
        "final_declaration_allowed_by_this_report": False,
        "qualification_artifact_reuse_as_final_evidence": False,
        "passing_subset_admissible": False,
    }
    assert gate.validate_qualification(report) == report


def test_hand_authored_summary_forgery_validates_but_replay_and_final_gate_reject(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _inputs(tmp_path, monkeypatch)
    for name in (
        "source_root", "source_cold_root", "compiled_root",
        "compile_evidence_root", "q2tool", "basedir",
        "compiled_cm_evidence_root", "cm_oracle", "pmove_oracle",
        "hook_oracle", "fall_oracle", "hook_attestation", "python_runtime",
        "materialized_root", "materialization_log_root", "claims_root",
        "analysis_root", "atlas_evidence_root",
        "promotion_evidence_root", "infrastructure_evidence_root",
        "syntax_report",
    ):
        path = tmp_path / name
        path.mkdir()
        setattr(args, name, path)
    monkeypatch.setattr(gate, "_validate_retained_stage_evidence", lambda *unused: None)
    legitimate = gate.assemble_qualification(args)
    forged = json.loads(canonical_bytes(legitimate))
    forged["end_to_end"]["passed_maps"][0] = "hand_authored_forgery"
    # This documents the old summary-only weakness: its internal shape is green.
    assert gate.validate_qualification(forged) == forged
    with pytest.raises(gate.B2QualificationError, match="canonical summary differs"):
        gate.replay_qualification(forged, repo_root=gate.ROOT)

    forged_path = tmp_path / "forged-qualification.json"
    forged_path.write_bytes(canonical_bytes(forged))
    paths = SimpleNamespace(
        qualification_report=forged_path, repo_root=gate.ROOT,
    )
    with pytest.raises(final_gate.B2GateError, match="qualification rejected"):
        final_gate._validate_qualification_report(paths, {}, {})


def test_retained_replay_can_bind_the_qualified_predecessor_implementation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _inputs(tmp_path, monkeypatch)
    for name in (
        "source_root", "source_cold_root", "compiled_root",
        "compile_evidence_root", "q2tool", "basedir",
        "compiled_cm_evidence_root", "cm_oracle", "pmove_oracle",
        "hook_oracle", "fall_oracle", "hook_attestation", "python_runtime",
        "materialized_root", "materialization_log_root", "claims_root",
        "analysis_root", "atlas_evidence_root", "promotion_evidence_root",
        "infrastructure_evidence_root", "syntax_report",
    ):
        path = tmp_path / name
        path.mkdir()
        setattr(args, name, path)
    monkeypatch.setattr(gate, "_validate_retained_stage_evidence", lambda *unused: None)
    legitimate = gate.assemble_qualification(args)
    qualified = dict(legitimate["implementation"])
    successor = dict(qualified)
    successor["repository_commit"] = "ab" * 20
    successor["repository_tree"] = "cd" * 20
    monkeypatch.setattr(gate, "repository_binding", lambda _root: successor)

    with pytest.raises(gate.B2QualificationError, match="implementation binding differs"):
        gate.replay_qualification(legitimate, repo_root=gate.ROOT)
    assert gate.replay_qualification(
        legitimate,
        repo_root=gate.ROOT,
        use_reported_implementation=True,
    ) == legitimate


def test_retained_replay_wires_all_strict_pre_atlas_consumers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tools import run_b2_qualification_compile as compile_stage
    from tools import run_b2_qualification_compiled_cm as cm_stage
    from tools import run_b2_qualification_infrastructure as infrastructure_stage
    from tools import run_b2_qualification_postcompile as postcompile_stage
    from tools import run_b2_qualification_promotion as promotion_stage
    from tools import run_b2_qualification_source as source_stage

    args = _inputs(tmp_path, monkeypatch)
    declaration = json.loads(args.declaration.read_bytes())
    report_paths = {
        "source": args.source_report,
        "compile": args.compile_report,
        "compiled-cm-preflight": args.compiled_cm_preflight_report,
        "materialization": args.materialization_report,
        "claims": args.claims_report,
        "atlas-build": args.atlas_build_report,
        "generated-promotion": args.generated_promotion_report,
    }
    stage_documents = {
        stage: json.loads(path.read_bytes())
        for stage, path in report_paths.items()
    }
    passed = {
        stage: {
            row["map"] for row in report["maps"] if row["passed"] is True
        }
        for stage, report in stage_documents.items()
    }
    roots = {
        name: str((tmp_path / name).absolute())
        for name in (
            "source_root", "source_cold_root", "compiled_root",
            "compile_evidence_root", "q2tool", "basedir",
            "compiled_cm_evidence_root", "cm_oracle", "pmove_oracle",
            "hook_oracle", "fall_oracle", "hook_attestation",
            "python_runtime", "materialized_root",
            "materialization_log_root", "claims_root", "analysis_root",
            "atlas_evidence_root", "promotion_evidence_root",
            "infrastructure_evidence_root", "syntax_report",
        )
    }
    successes = {
        (source_stage, "validate_published_qualification_source"):
            lambda **unused: ({}, b"", "0" * 64, passed["source"]),
        (compile_stage, "validate_published_qualification_compile"):
            lambda **unused: ({}, b"", "0" * 64, passed["compile"]),
        (cm_stage, "validate_published_qualification_compiled_cm"):
            lambda **unused: (
                {}, b"", "0" * 64, passed["compiled-cm-preflight"]
            ),
        (postcompile_stage, "validate_published_qualification_postcompile"):
            lambda **unused: {
                "materialization_passed": sorted(passed["materialization"]),
                "claims_passed": sorted(passed["claims"]),
            },
    }
    for (module, name), success in successes.items():
        monkeypatch.setattr(module, name, success)
    monkeypatch.setattr(promotion_stage, "_validate_atlas_artifacts", lambda *unused: None)
    monkeypatch.setattr(promotion_stage, "validate_promotion_evidence", lambda *unused: None)
    monkeypatch.setattr(
        infrastructure_stage, "validate_infrastructure_evidence",
        lambda *unused, **kwargs: None,
    )
    infrastructure = json.loads(args.infrastructure_report.read_bytes())
    gate._validate_retained_stage_evidence(
        declaration, stage_documents, infrastructure, roots, args, gate.ROOT,
    )

    for (module, name), success in successes.items():
        def reject(**unused):
            raise RuntimeError(f"handcrafted {name} report")

        monkeypatch.setattr(module, name, reject)
        with pytest.raises(
            gate.B2QualificationError, match=f"handcrafted {name} report"
        ):
            gate._validate_retained_stage_evidence(
                declaration, stage_documents, infrastructure, roots, args,
                gate.ROOT,
            )
        monkeypatch.setattr(module, name, success)


def test_rejects_final_mode_declaration_and_retired_seed(tmp_path: Path) -> None:
    implementation = _implementation()
    declaration = _declaration(implementation)
    declaration["mode"] = "final"
    path = _write(tmp_path / "qualification-declaration.json", declaration)
    with pytest.raises(gate.B2QualificationError, match="final-mode"):
        gate._validate_declaration(path, implementation, _retired())

    declaration["mode"] = "qualification"
    retired = _retired()
    retired["seeds"].add(declaration["maps"][0]["seed"])
    _write(path, declaration) if not path.exists() else path.write_bytes(canonical_bytes(declaration))
    with pytest.raises(gate.B2QualificationError, match="retired map seed"):
        gate._validate_declaration(path, implementation, retired)


def test_rejects_fewer_than_20_end_to_end_maps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _inputs(tmp_path, monkeypatch, pass_count=19)
    with pytest.raises(gate.B2QualificationError, match="19/28"):
        gate.assemble_qualification(args)


def test_rejects_report_hash_chain_and_self_declared_map_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _inputs(tmp_path, monkeypatch)
    compile_report = json.loads(args.compile_report.read_bytes())
    compile_report["input_report_sha256"] = _digest("wrong-input")
    args.compile_report.write_bytes(canonical_bytes(compile_report))
    with pytest.raises(gate.B2QualificationError, match="input report hash chain"):
        gate.assemble_qualification(args)

    args = _inputs(tmp_path / "second", monkeypatch)
    source = json.loads(args.source_report.read_bytes())
    source["maps"][0]["passed"] = False
    args.source_report.write_bytes(canonical_bytes(source))
    with pytest.raises(gate.B2QualificationError, match="self-declared result"):
        gate.assemble_qualification(args)


def test_rejects_promotion_of_map_that_failed_an_earlier_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _inputs(tmp_path, monkeypatch)
    materialization = json.loads(args.materialization_report.read_bytes())
    materialization["maps"][0]["criteria"] = {"producer-evidence-bound": False}
    materialization["maps"][0]["failures"] = ["failed before Atlas"]
    materialization["maps"][0]["passed"] = False
    materialization["pass_count"] = 19
    args.materialization_report.write_bytes(canonical_bytes(materialization))
    # Repair the downstream hash chain so this reaches the semantic subset check.
    previous = gate._file_sha256(args.materialization_report)
    for path in (args.claims_report, args.atlas_build_report,
                 args.generated_promotion_report):
        report = json.loads(path.read_bytes())
        report["input_report_sha256"] = previous
        path.write_bytes(canonical_bytes(report))
        previous = gate._file_sha256(path)
    infrastructure = json.loads(args.infrastructure_report.read_bytes())
    infrastructure["stage_report_sha256s"] = {
        stage: gate._file_sha256(getattr(
            args,
            {
                "source": "source_report", "compile": "compile_report",
                "compiled-cm-preflight": "compiled_cm_preflight_report",
                "materialization": "materialization_report",
                "claims": "claims_report", "atlas-build": "atlas_build_report",
                "generated-promotion": "generated_promotion_report",
            }[stage],
        )) for stage in gate.STAGES
    }
    args.infrastructure_report.write_bytes(canonical_bytes(infrastructure))
    with pytest.raises(gate.B2QualificationError, match="did not pass materialization"):
        gate.assemble_qualification(args)


def test_boundary_requires_proof_and_fresh_b1_physics_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    proof_path = _boundary(
        tmp_path, monkeypatch, physics_identity=_digest("wrong-physics")
    )
    authority = SimpleNamespace(
        cm_executable_sha256=_digest("cm-executable"),
        oracle_tool_identity=_digest("cm-tool"),
    )
    with pytest.raises(gate.B2QualificationError, match="physics identity"):
        gate._validate_boundary_proof(
            proof_path,
            authority,
            {"tool_identity": _digest("cm-tool"),
             "physics_identity": _digest("cm-physics")},
            _retired(),
        )

    compile_only = json.loads((tmp_path / "boundary-compile.json").read_bytes())
    compile_only_path = _write(tmp_path / "compile-only-proof.json", compile_only)
    with pytest.raises(gate.B2QualificationError, match="compile-only"):
        gate._validate_boundary_proof(
            compile_only_path,
            authority,
            {"tool_identity": _digest("cm-tool"),
             "physics_identity": _digest("cm-physics")},
            _retired(),
        )


def test_requalification_binds_amended_docs_repo_and_live_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    b1_path = repo / "docs/multires/B1-GATE.json"
    implementation = _implementation()
    normative = {
        "design": {"sha256": _digest("design")},
        "plan": {"sha256": _digest("plan")},
    }
    requalification = {
        "schema": "q2-b1-authority-requalification-v1",
        "status": "green",
        "recorded_at": "2026-07-16T12:00:00-07:00",
        "historical_gate_sha256": _digest("historical"),
        "historical_normative_documents": {
            "design_sha256": _digest("old-design"),
            "plan_sha256": _digest("old-plan"),
        },
        "current_normative_documents": {
            "design_sha256": normative["design"]["sha256"],
            "plan_sha256": normative["plan"]["sha256"],
        },
        "probe_bsp_sha256": _digest("probe"),
        "probe_runtime_authority_seal": {
            "schema": "q2-b1-runtime-authority-seal-v1",
            "normative_documents": {
                "design_sha256": normative["design"]["sha256"],
                "plan_sha256": normative["plan"]["sha256"],
            },
        },
        "repository": {"commit": COMMIT, "tree": TREE, "clean": True},
        "inputs": {},
        "live_identities": {
            "collision": {
                "tool_identity": _digest("cm-tool"),
                "physics_identity": _digest("cm-physics"),
            },
        },
        "checks": {"live-identities-recomputed": True},
        "failures": [],
    }
    _write(b1_path, {"authority_requalification": requalification})
    monkeypatch.setattr(
        gate,
        "load_b1_authority_gate",
        lambda root: SimpleNamespace(
            design_sha256=normative["design"]["sha256"],
            plan_sha256=normative["plan"]["sha256"],
            oracle_tool_identity=_digest("cm-tool"),
        ),
    )
    summary, _, collision = gate._validate_requalification(
        b1_path, repo, normative
    )
    assert summary["collision_identity"] == collision
    assert collision["physics_identity"] == _digest("cm-physics")

    requalification["current_normative_documents"]["plan_sha256"] = _digest("substituted")
    _write(b1_path, {"authority_requalification": requalification}) if not b1_path.exists() else b1_path.write_bytes(canonical_bytes({"authority_requalification": requalification}))
    with pytest.raises(gate.B2QualificationError, match="amended documents"):
        gate._validate_requalification(b1_path, repo, normative)


def test_main_exclusive_creates_canonical_output_outside_repo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "qualification.json"
    implementation = _implementation()
    report = {
        "schema": gate.QUALIFICATION_SCHEMA,
        "status": "green",
        "non_admissible": True,
        "retryable": True,
        "final_cohort_authorized": False,
        "implementation": implementation,
        "end_to_end": {"pass_count": 20},
        "failures": [],
    }
    monkeypatch.setattr(gate, "assemble_qualification", lambda args: report)
    monkeypatch.setattr(gate, "repository_binding", lambda root: implementation)
    argv = []
    for name in (
        "design", "plan", "b1-gate", "boundary-proof-report", "declaration",
        "source-report", "compile-report", "compiled-cm-preflight-report",
            "materialization-report", "claims-report", "atlas-build-report",
            "generated-promotion-report", "infrastructure-report",
            "claims-root", "analysis-root", "atlas-evidence-root",
            "promotion-evidence-root", "infrastructure-evidence-root",
            "source-root", "compiled-root", "materialized-root", "syntax-report",
            "source-cold-root", "compile-evidence-root", "q2tool", "basedir",
            "compiled-cm-evidence-root", "cm-oracle", "pmove-oracle",
            "hook-oracle", "fall-oracle", "hook-attestation", "python-runtime",
            "materialization-log-root",
        ):
        argv.extend((f"--{name}", str(tmp_path / f"{name}.json")))
    argv.extend(("--repo-root", str(gate.ROOT), "--output", str(output)))
    assert gate.main(argv) == 0
    assert output.read_bytes() == canonical_bytes(report)
    assert gate.main(argv) == 1
