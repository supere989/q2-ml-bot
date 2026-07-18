"""Deterministic tests for the final-cohort pre-authorization plan validator."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import validate_b2_final_cohort_plan as plan_tool
from tools.run_generator_cohort import canonical_bytes
from tools.validate_b2_final_cohort_plan import (
    DEFECT_COHORT_ARTIFACT,
    DEFECT_RUNNER_CONFIGURATION,
    DRIVER_STAGES,
    EVIDENCE_SCHEMA,
    MUTATING_EXECUTION_ACK,
    PLAN_SCHEMA,
    FinalCohortPlanError,
    build_evidence,
    build_plan,
    parse_stage_command,
    run_plan,
    validate_plan,
)


ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DECLARATION = (
    ROOT / "tests/fixtures/multires/B2-GENERATED-COHORT-FRESH-DECLARATION.json"
)
RETIRED_DECLARATION = (
    ROOT / "docs/multires/B2-GENERATED-COHORT-71445-DECLARATION.json"
)


@pytest.fixture(autouse=True)
def _isolated_source_authorization_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> object:
    previous_authority = plan_tool.b2_gate.ACTIVE_FINAL_AUTHORITY
    monkeypatch.setattr(
        plan_tool,
        "SOURCE_AUTHORIZATION_STATE_ROOT",
        tmp_path / "source-authorization-state",
    )
    yield
    plan_tool.b2_gate.ACTIVE_FINAL_AUTHORITY = previous_authority


def _write(path: Path, payload: bytes = b"fixture\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    path.chmod(0o755)
    return path


def _activate_fixture_declaration(path: Path) -> None:
    declaration = json.loads(path.read_bytes())
    absolute = str(path.resolve())
    plan_tool.b2_gate.ACTIVE_FINAL_AUTHORITY = (
        plan_tool.b2_gate.ActiveFinalAuthority(
            cohort_id=declaration["cohort_id"],
            declaration_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            immutable_declaration_path=absolute,
            qualification_successor_paths=frozenset({absolute}),
        )
    )


def _fixture_args(
    tmp_path: Path,
    *,
    tag: str = "primary",
    declaration_src: Path = FIXTURE_DECLARATION,
) -> argparse.Namespace:
    root = tmp_path / tag
    root.mkdir(parents=True, exist_ok=True)
    workspace = root / "final-workspace"
    basedir = root / "baseq2"
    basedir.mkdir(exist_ok=True)
    (basedir / "pak0.pak").write_bytes(b"fixture:pak0\n")
    declaration = root / "declaration.json"
    declaration.write_bytes(declaration_src.read_bytes())
    if declaration_src == FIXTURE_DECLARATION:
        fixture_value = json.loads(declaration.read_bytes())
        fixture_value["cohort_id"] = "b2g26_final_99000"
        declaration.write_bytes(canonical_bytes(fixture_value))
    python = _write(root / "python-bin", b"#!/bin/sh\n")
    q2tool = _write(root / "q2tool", b"#!/bin/sh\n")
    cm = _write(root / "cm-oracle", b"oracle-cm\n")
    pmove = _write(root / "pmove-oracle", b"oracle-pmove\n")
    hook = _write(root / "hook-oracle", b"oracle-hook\n")
    fall = _write(root / "fall-oracle", b"oracle-fall\n")
    attestation = root / "hook-attestation.json"
    attestation.write_bytes(b'{"schema":"fixture-hook-attestation"}\n')
    b1_gate = root / "b1-gate.json"
    b1_gate.write_bytes((ROOT / "docs/multires/B1-GATE.json").read_bytes())
    result = argparse.Namespace(
        workspace=workspace,
        repo_root=ROOT,
        declaration=declaration,
        python=python,
        q2tool=q2tool,
        basedir=basedir,
        cm_oracle=cm,
        pmove_oracle=pmove,
        hook_oracle=hook,
        fall_oracle=fall,
        hook_attestation=attestation,
        b1_gate=b1_gate,
        client_root=None,
        lithium_root=None,
        packer=None,
        verifier=None,
        source_root=None,
        source_cold=None,
        compile_staging=None,
        compiled_root=None,
        compile_logs=None,
        materialize_staging=None,
        materialized_root=None,
        materialize_logs=None,
        claims_root=None,
        analysis_root=None,
        atlas_diagnostics=None,
        compile_timeout_seconds=3600.0,
        oracle_batch_timeout_seconds=10.0,
        materialize_timeout_seconds=900,
        cm_jobs=4,
        dry_run=True,
        execute=False,
        acknowledge_mutating_execution=None,
    )
    _activate_fixture_declaration(declaration)
    return result


def _cli_base(args: argparse.Namespace) -> list[str]:
    return [
        "--workspace",
        str(args.workspace),
        "--repo-root",
        str(args.repo_root),
        "--declaration",
        str(args.declaration),
        "--python",
        str(args.python),
        "--q2tool",
        str(args.q2tool),
        "--basedir",
        str(args.basedir),
        "--cm-oracle",
        str(args.cm_oracle),
        "--pmove-oracle",
        str(args.pmove_oracle),
        "--hook-oracle",
        str(args.hook_oracle),
        "--fall-oracle",
        str(args.fall_oracle),
        "--hook-attestation",
        str(args.hook_attestation),
        "--b1-gate",
        str(args.b1_gate),
    ]


def test_out_of_range_oracle_batch_timeout_is_runner_configuration_defect(
    tmp_path: Path,
) -> None:
    """Reproduce the 71446 defect: 3600 is legal for compile, illegal for CM."""

    args = _fixture_args(tmp_path)
    args.oracle_batch_timeout_seconds = 3600.0
    with pytest.raises(FinalCohortPlanError) as raised:
        build_plan(args)
    assert raised.value.defect_class == DEFECT_RUNNER_CONFIGURATION
    assert raised.value.check_id == "timeout-domain"
    assert "(0, 60]" in str(raised.value)

    evidence = plan_tool.main(
        [
            *_cli_base(args),
            "--oracle-batch-timeout-seconds",
            "3600",
            "--dry-run",
        ]
    )
    # main returns 1 and prints evidence; re-run through library path for body.
    try:
        build_plan(args)
    except FinalCohortPlanError as error:
        body = build_evidence(
            {
                "schema": PLAN_SCHEMA,
                "cohort_id": None,
                "commands": [],
                "argument_domains": {},
                "declaration": None,
            },
            [
                {
                    "check_id": error.check_id,
                    "passed": False,
                    "detail": str(error),
                    "defect_class": error.defect_class,
                }
            ],
            dry_run=True,
            mutating_stages_executed=[],
            failure={
                "defect_class": error.defect_class,
                "check_id": error.check_id,
                "message": str(error),
            },
        )
    assert body["passed"] is False
    assert body["defect_class"] == DEFECT_RUNNER_CONFIGURATION
    assert body["authorization"]["source_authorization_consumed"] is False
    assert body["authorization"]["cohort_retirement_triggered"] is False
    assert body["authorization"]["retry_under_same_declaration_allowed"] is True
    assert body["authorization"]["validation_does_not_create_or_authorize_cohort"] is True
    assert body["mutating_stages_executed"] == []
    assert evidence == 1


def test_configuration_defect_after_source_consumption_is_terminal() -> None:
    evidence = build_evidence(
        {
            "schema": PLAN_SCHEMA,
            "cohort_id": "b2g26_historical_consumed",
            "commands": [{"stage": "source"}, {"stage": "compiled-cm-preflight"}],
            "argument_domains": {},
            "declaration": {"sha256": "ab" * 32},
        },
        [],
        dry_run=False,
        mutating_stages_executed=["source", "compile", "compiled-static"],
        failure={
            "defect_class": DEFECT_RUNNER_CONFIGURATION,
            "check_id": "timeout-domain",
            "message": "oracle batch timeout must be finite and in (0, 60]",
        },
    )

    assert evidence["passed"] is False
    assert evidence["defect_class"] == DEFECT_RUNNER_CONFIGURATION
    assert evidence["authorization"]["source_authorization_consumed"] is True
    assert evidence["authorization"]["cohort_retirement_triggered"] is True
    assert evidence["authorization"]["retry_under_same_declaration_allowed"] is False


def test_missing_tools_and_paths_are_configuration_defects(tmp_path: Path) -> None:
    args = _fixture_args(tmp_path, tag="missing-q2tool")
    args.q2tool = tmp_path / "missing-q2tool"
    with pytest.raises(FinalCohortPlanError) as raised:
        build_plan(args)
    assert raised.value.defect_class == DEFECT_RUNNER_CONFIGURATION
    assert raised.value.check_id == "path-identity"
    assert "absent" in str(raised.value)

    args = _fixture_args(tmp_path, tag="missing-cm")
    args.cm_oracle = tmp_path / "missing-cm"
    with pytest.raises(FinalCohortPlanError) as raised:
        build_plan(args)
    assert "CM oracle" in str(raised.value)

    args = _fixture_args(tmp_path, tag="missing-declaration")
    args.declaration = tmp_path / "missing-declaration.json"
    with pytest.raises(FinalCohortPlanError) as raised:
        build_plan(args)
    assert "declaration" in str(raised.value).lower()


def test_mismatched_stage_inputs_fail_validation(tmp_path: Path) -> None:
    args = _fixture_args(tmp_path)
    plan = build_plan(args)
    # Break compile's binding to the source stage output.
    compile_step = next(
        step for step in plan["commands"] if step["stage"] == "compile"
    )
    source_index = compile_step["command"].index("--source-root")
    compile_step["command"][source_index + 1] = str(tmp_path / "wrong-source")
    with pytest.raises(FinalCohortPlanError) as raised:
        validate_plan(plan)
    assert raised.value.check_id == "stage-input-binding:compile-source"
    assert raised.value.defect_class == DEFECT_RUNNER_CONFIGURATION

    # Break declaration binding on the CM stage.
    plan = build_plan(args)
    cm_step = next(
        step
        for step in plan["commands"]
        if step["stage"] == "compiled-cm-preflight"
    )
    decl_index = cm_step["command"].index("--declaration")
    cm_step["command"][decl_index + 1] = str(tmp_path / "other-declaration.json")
    with pytest.raises(FinalCohortPlanError) as raised:
        validate_plan(plan)
    assert raised.value.check_id.startswith("stage-input-binding:")


def test_fresh_fixture_declaration_is_accepted(tmp_path: Path) -> None:
    """A non-retired fresh declaration must plan successfully under dry-run."""

    args = _fixture_args(tmp_path)
    plan = build_plan(args)
    assert plan["schema"] == PLAN_SCHEMA
    assert plan["cohort_id"] == "b2g26_final_99000"
    assert plan["authorization"]["source_authorization_consumed"] is False
    assert plan["authorization"]["retired_registry_admission_required"] is True
    assert plan["authorization"]["validation_does_not_create_or_authorize_cohort"] is True
    # Hard-coded permanent refusal of arbitrary future IDs is forbidden tool
    # semantics: only the retired registry may deny admission.
    assert "cohort_71447_authorization_forbidden" not in plan["authorization"]

    checks = validate_plan(plan)
    assert all(check["passed"] for check in checks)
    assert any(check["check_id"] == "retired-registry" for check in checks)

    evidence = run_plan(plan, dry_run=True)
    assert evidence["passed"] is True
    assert evidence["dry_run"] is True
    assert evidence["cohort_id"] == "b2g26_final_99000"
    assert evidence["mutating_stages_executed"] == []
    assert evidence["authorization"]["source_authorization_consumed"] is False


def test_plan_requires_exact_active_declaration_path_and_identity(
    tmp_path: Path,
) -> None:
    args = _fixture_args(tmp_path)
    active = plan_tool.b2_gate.ACTIVE_FINAL_AUTHORITY
    plan_tool.b2_gate.ACTIVE_FINAL_AUTHORITY = None
    try:
        with pytest.raises(
            FinalCohortPlanError, match="no active final cohort"
        ) as raised:
            build_plan(args)
    finally:
        plan_tool.b2_gate.ACTIVE_FINAL_AUTHORITY = active
    assert raised.value.check_id == "active-authority"

    _activate_fixture_declaration(args.declaration)
    copied = tmp_path / "byte-identical-copy.json"
    copied.write_bytes(args.declaration.read_bytes())
    args.declaration = copied
    with pytest.raises(FinalCohortPlanError, match="path/cohort/digest") as raised:
        build_plan(args)
    assert raised.value.check_id == "active-authority"


def test_non_retired_final_cohort_id_is_not_hard_refused(tmp_path: Path) -> None:
    """Milestone 'do not create 71447' is not permanent tool semantics."""

    args = _fixture_args(tmp_path, tag="id-71447")
    payload = json.loads(args.declaration.read_text())
    # Keep fresh maps/seeds; only rename the cohort identity to the milestone ID.
    payload["cohort_id"] = "b2g26_final_71447"
    args.declaration.write_bytes(canonical_bytes(payload))
    _activate_fixture_declaration(args.declaration)
    plan = build_plan(args)
    assert plan["cohort_id"] == "b2g26_final_71447"
    evidence = run_plan(plan, dry_run=True)
    assert evidence["passed"] is True
    assert evidence["authorization"]["source_authorization_consumed"] is False


def test_retired_declaration_is_rejected(tmp_path: Path) -> None:
    args = _fixture_args(
        tmp_path, tag="retired-71445", declaration_src=RETIRED_DECLARATION
    )
    with pytest.raises(FinalCohortPlanError) as raised:
        build_plan(args)
    assert raised.value.check_id == "retired-registry"
    assert raised.value.defect_class == DEFECT_RUNNER_CONFIGURATION
    assert "permanently retired" in str(raised.value)
    assert "71445" in str(raised.value)
    # Validation refusal must not create workspace stage roots.
    assert not args.workspace.exists() or not any(args.workspace.iterdir())


def test_validation_does_not_create_or_authorize_cohort(
    tmp_path: Path, capsys
) -> None:
    args = _fixture_args(tmp_path, tag="no-create")
    plan = build_plan(args)
    evidence = run_plan(plan, dry_run=True)
    assert evidence["passed"] is True
    assert evidence["dry_run"] is True
    assert evidence["mutating_stages_executed"] == []
    assert evidence["authorization"]["source_authorization_consumed"] is False
    assert evidence["authorization"]["cohort_retirement_triggered"] is False
    assert evidence["authorization"]["validation_does_not_create_or_authorize_cohort"] is True

    code = plan_tool.main([*_cli_base(args), "--dry-run"])
    assert code == 0
    loaded = json.loads(capsys.readouterr().out)
    assert loaded["passed"] is True
    assert loaded["mutating_stages_executed"] == []
    assert loaded["authorization"]["source_authorization_consumed"] is False

    # No stage outputs under the workspace after plan + dry-run validation.
    assert not (args.workspace / "source").exists()
    assert not (args.workspace / "compiled").exists()
    assert not (args.workspace / "materialized").exists()
    assert not (args.workspace / "claims").exists()
    assert not (args.workspace / "reports").exists()


def test_dry_run_success_emits_canonical_evidence(tmp_path: Path, capsys) -> None:
    args = _fixture_args(tmp_path)
    plan = build_plan(args)
    assert plan["schema"] == PLAN_SCHEMA
    assert [step["stage"] for step in plan["commands"]] == list(DRIVER_STAGES)
    assert plan["argument_domains"][
        "compiled_cm_preflight.oracle_batch_timeout_seconds"
    ]["value"] == 10.0
    assert plan["argument_domains"][
        "compiled_cm_preflight.oracle_batch_timeout_seconds"
    ]["domain"] == "(0, 60]"
    assert plan["authorization"]["source_authorization_consumed"] is False

    evidence = run_plan(plan, dry_run=True)
    assert evidence["schema"] == EVIDENCE_SCHEMA
    assert evidence["passed"] is True
    assert evidence["dry_run"] is True
    assert evidence["mutating_stages_executed"] == []
    assert evidence["defect_class"] is None
    assert evidence["authorization"]["source_authorization_consumed"] is False
    assert evidence["authorization"]["cohort_retirement_triggered"] is False
    assert evidence["stage_order"] == list(DRIVER_STAGES)
    # Canonical encoding is byte-stable.
    payload = canonical_bytes(evidence)
    assert payload == canonical_bytes(json.loads(payload))
    assert evidence["plan_sha256"] == hashlib.sha256(
        canonical_bytes(plan)
    ).hexdigest()

    code = plan_tool.main(
        [
            *_cli_base(args),
            "--oracle-batch-timeout-seconds",
            "10",
            "--compile-timeout-seconds",
            "3600",
            "--dry-run",
        ]
    )
    assert code == 0
    stdout = capsys.readouterr().out
    loaded = json.loads(stdout)
    assert loaded["passed"] is True
    assert stdout.encode("ascii") == canonical_bytes(loaded)


def test_no_mutating_stage_runs_before_full_plan_validation(tmp_path: Path) -> None:
    args = _fixture_args(tmp_path)
    plan = build_plan(args)
    calls: list[str] = []

    def runner(command, *, stage, plan):
        del command, plan
        calls.append(stage)
        return SimpleNamespace(returncode=0)

    # Dry-run never touches the runner.
    evidence = run_plan(plan, dry_run=True, runner=runner)
    assert evidence["passed"] is True
    assert calls == []
    assert evidence["mutating_stages_executed"] == []

    # Force a validation failure after plan construction; execute must not run.
    broken = json.loads(canonical_bytes(plan))
    broken["commands"] = list(reversed(broken["commands"]))
    evidence = run_plan(
        broken,
        dry_run=False,
        runner=runner,
        acknowledge_mutating_execution=True,
    )
    assert evidence["passed"] is False
    assert evidence["defect_class"] == DEFECT_RUNNER_CONFIGURATION
    assert calls == []
    assert evidence["mutating_stages_executed"] == []
    assert evidence["authorization"]["source_authorization_consumed"] is False
    assert evidence["authorization"]["retry_under_same_declaration_allowed"] is True

    # Execute without acknowledgement must not mutate even with a runner.
    calls.clear()
    evidence = run_plan(plan, dry_run=False, runner=runner)
    assert evidence["passed"] is False
    assert evidence["failure"]["check_id"] == "execute-acknowledgement"
    assert calls == []
    assert evidence["mutating_stages_executed"] == []

    # Only after full validation + acknowledgement may stages execute.
    calls.clear()
    evidence = run_plan(
        plan,
        dry_run=False,
        runner=runner,
        acknowledge_mutating_execution=MUTATING_EXECUTION_ACK,
    )
    assert evidence["passed"] is True
    assert calls == list(DRIVER_STAGES)
    assert evidence["mutating_stages_executed"] == list(DRIVER_STAGES)
    assert evidence["authorization"]["source_authorization_consumed"] is True


def test_stage_failure_after_source_is_cohort_artifact_no_retry(
    tmp_path: Path,
) -> None:
    args = _fixture_args(tmp_path)
    plan = build_plan(args)

    def runner(command, *, stage, plan):
        del command, plan
        if stage == "compile":
            return SimpleNamespace(returncode=7)
        return SimpleNamespace(returncode=0)

    evidence = run_plan(
        plan,
        dry_run=False,
        runner=runner,
        acknowledge_mutating_execution=True,
    )
    assert evidence["passed"] is False
    assert evidence["defect_class"] == DEFECT_COHORT_ARTIFACT
    assert evidence["mutating_stages_executed"] == ["source", "compile"]
    assert evidence["authorization"]["source_authorization_consumed"] is True
    assert evidence["authorization"]["cohort_retirement_triggered"] is True
    assert evidence["authorization"]["retry_under_same_declaration_allowed"] is False
    marker = evidence["source_authorization_marker"]
    assert marker["path"].endswith(f"/{plan['declaration']['sha256']}.json")
    assert Path(marker["path"]).is_file()


def test_source_authorization_marker_prevents_second_execution(
    tmp_path: Path,
) -> None:
    args = _fixture_args(tmp_path)
    plan = build_plan(args)
    calls: list[str] = []

    def runner(_command, *, stage, plan):
        del plan
        calls.append(stage)
        return SimpleNamespace(returncode=0)

    first = run_plan(
        plan,
        dry_run=False,
        runner=runner,
        acknowledge_mutating_execution=True,
    )
    assert first["passed"] is True
    assert calls == list(DRIVER_STAGES)

    calls.clear()
    second = run_plan(
        plan,
        dry_run=False,
        runner=runner,
        acknowledge_mutating_execution=True,
    )
    assert second["passed"] is False
    assert calls == []
    assert second["failure"]["check_id"] == (
        "source-authorization-already-consumed"
    )
    assert second["authorization"]["source_authorization_consumed"] is True
    assert second["authorization"]["cohort_retirement_triggered"] is True
    assert second["authorization"]["retry_under_same_declaration_allowed"] is False
    assert second["authorization"]["immutable_no_retry"] is True


def test_declaration_tombstone_blocks_a_different_workspace(
    tmp_path: Path,
) -> None:
    args = _fixture_args(tmp_path)
    plan = build_plan(args)
    first = run_plan(
        plan,
        dry_run=False,
        runner=lambda *_args, **_kwargs: SimpleNamespace(returncode=0),
        acknowledge_mutating_execution=MUTATING_EXECUTION_ACK,
    )
    assert first["passed"] is True

    other_args = _fixture_args(tmp_path, tag="other-workspace")
    other_args.declaration.write_bytes(args.declaration.read_bytes())
    _activate_fixture_declaration(other_args.declaration)
    other_plan = build_plan(other_args)

    evidence = run_plan(
        other_plan,
        dry_run=False,
        runner=lambda *_args, **_kwargs: pytest.fail("source must not run"),
        acknowledge_mutating_execution=MUTATING_EXECUTION_ACK,
    )
    assert evidence["passed"] is False
    assert evidence["failure"]["check_id"] == "source-authorization-already-consumed"
    assert evidence["mutating_stages_executed"] == ["source"]
    assert evidence["authorization"]["source_authorization_consumed"] is True
    assert evidence["authorization"]["cohort_retirement_triggered"] is True


def test_incomplete_source_authorization_journal_is_terminal_tombstone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _fixture_args(tmp_path)
    plan = build_plan(args)
    calls: list[str] = []
    real_fsync = plan_tool.os.fsync
    failed = False

    def interrupted_fsync(descriptor: int) -> None:
        nonlocal failed
        if not failed:
            failed = True
            raise OSError("simulated journal fsync failure")
        real_fsync(descriptor)

    def runner(_command, *, stage, plan):
        del plan
        calls.append(stage)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(plan_tool.os, "fsync", interrupted_fsync)
    first = run_plan(
        plan,
        dry_run=False,
        runner=runner,
        acknowledge_mutating_execution=True,
    )
    assert first["passed"] is False
    assert calls == []
    assert first["failure"]["check_id"] == (
        "source-authorization-journal-incomplete"
    )
    assert first["authorization"]["source_authorization_consumed"] is True
    marker = (
        plan_tool.SOURCE_AUTHORIZATION_STATE_ROOT
        / f"{plan['declaration']['sha256']}.json"
    )
    assert marker.is_file()

    calls.clear()
    second = run_plan(
        plan,
        dry_run=False,
        runner=runner,
        acknowledge_mutating_execution=True,
    )
    assert second["passed"] is False
    assert calls == []
    assert second["failure"]["check_id"] == (
        "source-authorization-already-consumed"
    )
    assert second["authorization"]["cohort_retirement_triggered"] is True


def test_source_stage_exception_is_recorded_write_ahead_and_retires(
    tmp_path: Path,
) -> None:
    args = _fixture_args(tmp_path)
    plan = build_plan(args)

    def runner(_command, *, stage, plan):
        del plan
        assert stage == "source"
        raise RuntimeError("crash after first source byte")

    evidence = run_plan(
        plan,
        dry_run=False,
        runner=runner,
        acknowledge_mutating_execution=MUTATING_EXECUTION_ACK,
    )

    assert evidence["passed"] is False
    assert evidence["failure"]["check_id"] == "stage-exception:source"
    assert evidence["failure"]["exception_type"] == "RuntimeError"
    assert evidence["mutating_stages_executed"] == ["source"]
    assert evidence["authorization"]["source_authorization_consumed"] is True
    assert evidence["authorization"]["cohort_retirement_triggered"] is True
    assert evidence["authorization"]["retry_under_same_declaration_allowed"] is False


def test_pre_marker_journal_failure_does_not_claim_durable_consumption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _fixture_args(tmp_path)
    plan = build_plan(args)

    def fail_before_create(_plan):
        raise FinalCohortPlanError(
            "journal parent unavailable",
            check_id="source-authorization-journal",
        )

    monkeypatch.setattr(plan_tool, "_source_authorization_marker", fail_before_create)
    evidence = run_plan(
        plan,
        dry_run=False,
        runner=lambda *_args, **_kwargs: pytest.fail("source must not run"),
        acknowledge_mutating_execution=MUTATING_EXECUTION_ACK,
    )

    assert evidence["passed"] is False
    assert evidence["failure"]["check_id"] == "source-authorization-journal"
    assert evidence["mutating_stages_executed"] == []
    assert evidence["source_authorization_marker"] is None
    assert evidence["authorization"]["source_authorization_consumed"] is False
    assert evidence["authorization"]["cohort_retirement_triggered"] is False


def test_cli_execute_without_ack_or_runner_does_not_mutate(
    tmp_path: Path, capsys
) -> None:
    args = _fixture_args(tmp_path)
    code = plan_tool.main([*_cli_base(args), "--execute"])
    assert code == 1
    loaded = json.loads(capsys.readouterr().out)
    assert loaded["passed"] is False
    assert loaded["mutating_stages_executed"] == []
    assert loaded["authorization"]["source_authorization_consumed"] is False
    assert loaded["defect_class"] == DEFECT_RUNNER_CONFIGURATION
    assert loaded["failure"]["check_id"] == "execute-acknowledgement"
    assert not (args.workspace / "source").exists()
    assert not (args.workspace / "compiled").exists()

    code = plan_tool.main(
        [
            *_cli_base(args),
            "--execute",
            "--acknowledge-mutating-execution",
            MUTATING_EXECUTION_ACK,
        ]
    )
    assert code == 1
    loaded = json.loads(capsys.readouterr().out)
    assert loaded["passed"] is False
    assert loaded["mutating_stages_executed"] == []
    assert loaded["authorization"]["source_authorization_consumed"] is False
    assert loaded["failure"]["check_id"] == "execute-runner"
    assert not (args.workspace / "source").exists()


def test_workspace_must_be_outside_repository(tmp_path: Path) -> None:
    args = _fixture_args(tmp_path)
    args.workspace = ROOT / "docs" / "multires" / "would-be-final-workspace"
    with pytest.raises(FinalCohortPlanError) as raised:
        build_plan(args)
    assert "outside the repository" in str(raised.value)


def test_planned_commands_conform_to_stage_tool_parsers(tmp_path: Path) -> None:
    """Option names and timeout domains cannot silently drift from stage tools."""

    args = _fixture_args(tmp_path)
    plan = build_plan(args)
    # Domain constants are bound to live stage-tool exports.
    assert plan["argument_domains"]["compiled_cm_preflight.jobs"]["domain"] == (
        f"[{plan_tool.CM_JOBS_MIN}, {plan_tool.CM_JOBS_MAX}]"
    )
    assert plan["argument_domains"]["materialization.timeout_seconds"]["domain"] == (
        f"[{plan_tool.MATERIALIZE_TIMEOUT_MIN}, {plan_tool.MATERIALIZE_TIMEOUT_MAX}]"
    )
    from tools.materialize_generated_cohort import MAX_MATERIALIZER_TIMEOUT_SECONDS
    from tools.compile_generated_cohort import MAX_MAP_TIMEOUT_SECONDS
    from tools.run_compiled_cm_preflight import (
        MAX_JOBS,
        MAX_ORACLE_BATCH_TIMEOUT_SECONDS,
    )

    assert plan_tool.MATERIALIZE_TIMEOUT_MAX == MAX_MATERIALIZER_TIMEOUT_SECONDS
    assert plan_tool.CM_JOBS_MAX == MAX_JOBS
    assert plan_tool.COMPILE_TIMEOUT_MAX == MAX_MAP_TIMEOUT_SECONDS
    assert (
        plan_tool.ORACLE_BATCH_TIMEOUT_MAX
        == MAX_ORACLE_BATCH_TIMEOUT_SECONDS
    )

    for step in plan["commands"]:
        stage = step["stage"]
        parsed = parse_stage_command(stage, step["command"])
        assert parsed is not None
        # Tool digests are versioned in the plan for silent-drift detection.
        tool_record = plan["tools"][stage]
        live = Path(tool_record["path"])
        assert live.is_file()
        digest = hashlib.sha256(live.read_bytes()).hexdigest()
        assert digest == tool_record["sha256"]

    # Renaming a required option must fail parser conformance without mutation.
    compile_step = next(
        step for step in plan["commands"] if step["stage"] == "compile"
    )
    command = list(compile_step["command"])
    idx = command.index("--timeout-seconds")
    command[idx] = "--timeout-seconds-renamed"
    with pytest.raises(FinalCohortPlanError) as raised:
        parse_stage_command("compile", command)
    assert raised.value.check_id == "parser-conformance"
    assert not (args.workspace / "compiled").exists()
