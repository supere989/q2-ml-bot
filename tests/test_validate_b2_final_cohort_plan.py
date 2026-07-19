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
    implementation = {
        "repository_commit": "1" * 40,
        "repository_tree": "2" * 40,
        "git_clean": True,
        "atlas_analyzer_authority_sha256": "3" * 64,
        "atlas_analyzer_authority_file_count": 1,
        "generator_sha256": "4" * 64,
        "routes_sha256": "5" * 64,
    }
    monkeypatch.setattr(
        plan_tool,
        "repository_binding",
        lambda _repo_root: dict(implementation),
    )

    def fixture_execution_binding(*, state_root, repo_root, workspace):
        del repo_root, workspace
        root = Path(state_root)
        return {
            "schema": "fixture-final-execution-binding-v1",
            "host": {"hostname": "fixture", "euid": 1000},
            "state_root": {"path": str(root)},
        }

    def validate_fixture_execution_binding(binding, *, repo_root, workspace):
        del repo_root, workspace
        return dict(binding)

    class FixtureJournal:
        def __init__(self, binding, *, repo_root, workspace):
            del repo_root, workspace
            self.root = Path(binding["state_root"]["path"])

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            del exc_type, exc, traceback

        def marker_path(self, declaration_sha256):
            return self.root / f"{declaration_sha256}.json"

        def exists(self, declaration_sha256):
            return self.marker_path(declaration_sha256).exists()

        def read(self, declaration_sha256):
            path = self.marker_path(declaration_sha256)
            return path.read_bytes() if path.exists() else None

        def create(self, declaration_sha256, payload):
            path = self.marker_path(declaration_sha256)
            try:
                with path.open("xb") as stream:
                    stream.write(payload)
                    stream.flush()
                    plan_tool.os.fsync(stream.fileno())
            except FileExistsError:
                return path.read_bytes(), False
            except BaseException as error:
                raise plan_tool.SourceAuthorizationJournalError(
                    "fixture journal creation incomplete", consumed=True
                ) from error
            return payload, True

        def revalidate_path_binding(self):
            return None

    def fixture_marker(
        *, binding, cohort_id, declaration_sha256, plan_sha256, workspace,
        repo_root, declaration_path, implementation, source_output, source_cold,
        source_report,
    ):
        del repo_root, declaration_path, implementation, source_output, source_cold, source_report
        return canonical_bytes({
            "schema": "fixture-source-authorization-v3",
            "cohort_id": cohort_id,
            "declaration_sha256": declaration_sha256,
            "plan_sha256": plan_sha256,
            "workspace": str(workspace),
            "execution_binding": binding,
        })

    def verify_fixture_marker(
        payload, *, binding, cohort_id, declaration_sha256, repo_root, workspace,
    ):
        del repo_root, workspace
        loaded = json.loads(payload)
        if (
            loaded.get("cohort_id") != cohort_id
            or loaded.get("declaration_sha256") != declaration_sha256
            or loaded.get("execution_binding") != binding
        ):
            raise plan_tool.FinalExecutionBindingError("fixture marker differs")
        return loaded

    def fixture_preactivation_binding(**kwargs):
        report = Path(kwargs["preactivation_test_report"])
        payload = report.read_bytes()
        return {
            "report": {
                "path": str(report),
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            },
            "implementation": dict(kwargs["implementation"]),
        }

    monkeypatch.setattr(plan_tool, "build_execution_binding", fixture_execution_binding)
    monkeypatch.setattr(
        plan_tool, "validate_execution_binding", validate_fixture_execution_binding
    )
    monkeypatch.setattr(plan_tool, "open_secure_marker_journal", FixtureJournal)
    monkeypatch.setattr(plan_tool, "build_source_authorization_marker", fixture_marker)
    monkeypatch.setattr(
        plan_tool, "verify_source_authorization_marker", verify_fixture_marker
    )
    monkeypatch.setattr(
        plan_tool.b2_gate,
        "validate_preactivation_test_binding",
        fixture_preactivation_binding,
    )
    yield
    plan_tool.b2_gate.ACTIVE_FINAL_AUTHORITY = previous_authority


def _write(path: Path, payload: bytes = b"fixture\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    path.chmod(0o755)
    return path


def _reviewed_plan_sha256(plan: object) -> str:
    return hashlib.sha256(canonical_bytes(plan)).hexdigest()


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
    repo = root / "repo"
    (repo / "tools").mkdir(parents=True)
    for tool_name in set(plan_tool.TOOL_FILES.values()):
        (repo / "tools" / tool_name).write_bytes(
            (ROOT / "tools" / tool_name).read_bytes()
        )
    workspace = root / "final-workspace"
    basedir = root / "baseq2"
    basedir.mkdir(exist_ok=True)
    (basedir / "pak0.pak").write_bytes(b"fixture:pak0\n")
    declaration_value = json.loads(declaration_src.read_bytes())
    if declaration_src == FIXTURE_DECLARATION:
        declaration_value["cohort_id"] = "b2g26_final_99000"
    declaration_number = str(declaration_value["cohort_id"]).rsplit("_", 1)[-1]
    declaration = (
        repo
        / "docs/multires"
        / f"B2-GENERATED-COHORT-{declaration_number}-DECLARATION.json"
    )
    declaration.parent.mkdir(parents=True)
    declaration.write_bytes(canonical_bytes(declaration_value))
    python = _write(root / "python-bin", b"#!/bin/sh\n")
    q2tool = _write(root / "q2tool", b"#!/bin/sh\n")
    cm = _write(root / "cm-oracle", b"oracle-cm\n")
    pmove = _write(root / "pmove-oracle", b"oracle-pmove\n")
    hook = _write(root / "hook-oracle", b"oracle-hook\n")
    fall = _write(root / "fall-oracle", b"oracle-fall\n")
    client_root = root / "q2-ml-client"
    _write(client_root / "release/q2-cm-oracle", cm.read_bytes())
    _write(client_root / "release/q2-pmove-oracle", pmove.read_bytes())
    lithium_root = root / "q2-lithium-3zb2"
    _write(lithium_root / "tools/q2-hook-oracle", hook.read_bytes())
    packer = _write(root / "q2-atlas-pack", b"#!/bin/sh\n")
    verifier = _write(root / "q2-atlas-verify", b"#!/bin/sh\n")
    attestation = root / "hook-attestation.json"
    attestation.write_bytes(b'{"schema":"fixture-hook-attestation"}\n')
    b1_gate = root / "b1-gate.json"
    b1_gate.write_bytes((ROOT / "docs/multires/B1-GATE.json").read_bytes())
    design = _write(root / "B2-design.md")
    execution_plan = _write(root / "B2-plan.md")
    qualification_report = _write(root / "qualification-report.json")
    preactivation_test_report = _write(root / "preactivation-test-report.json")
    authorization_state_root = root / "source-authorization-state"
    authorization_state_root.mkdir(mode=0o700)
    authorization_state_root.chmod(0o700)
    stock_pak = _write(root / "stock-pak0.pak")
    stock_provenance = _write(root / "stock-provenance.json")
    stock_inventory = _write(root / "stock-inventory.json")
    dyn_evidence_executable = _write(
        root / "q2-lattice-evidence",
        b"#!/bin/sh\nexit 0\n",
    )
    result = argparse.Namespace(
        workspace=workspace,
        repo_root=repo,
        declaration=declaration,
        design=design,
        plan=execution_plan,
        qualification_report=qualification_report,
        preactivation_test_report=preactivation_test_report,
        authorization_state_root=authorization_state_root,
        python=python,
        q2tool=q2tool,
        basedir=basedir,
        stock_pak=stock_pak,
        stock_provenance=stock_provenance,
        stock_inventory=stock_inventory,
        cm_oracle=cm,
        pmove_oracle=pmove,
        hook_oracle=hook,
        fall_oracle=fall,
        hook_attestation=attestation,
        b1_gate=b1_gate,
        client_root=client_root,
        lithium_root=lithium_root,
        packer=packer,
        verifier=verifier,
        dyn_evidence_executable=dyn_evidence_executable,
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
        stock_root=None,
        dyn_evidence_root=None,
        test_evidence_root=None,
        gate_output=None,
        dyn_map_epoch=1,
        dyn_environment_steps=4000,
        dyn_samples=4000,
        compile_timeout_seconds=3600.0,
        oracle_batch_timeout_seconds=10.0,
        materialize_timeout_seconds=900,
        cm_jobs=4,
        dry_run=True,
        execute=False,
        acknowledge_mutating_execution=None,
        expected_plan_sha256=None,
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
        "--design",
        str(args.design),
        "--plan",
        str(args.plan),
        "--qualification-report",
        str(args.qualification_report),
        "--preactivation-test-report",
        str(args.preactivation_test_report),
        "--authorization-state-root",
        str(args.authorization_state_root),
        "--python",
        str(args.python),
        "--q2tool",
        str(args.q2tool),
        "--basedir",
        str(args.basedir),
        "--stock-pak",
        str(args.stock_pak),
        "--stock-provenance",
        str(args.stock_provenance),
        "--stock-inventory",
        str(args.stock_inventory),
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
        "--client-root",
        str(args.client_root),
        "--lithium-root",
        str(args.lithium_root),
        "--packer",
        str(args.packer),
        "--verifier",
        str(args.verifier),
        "--dyn-evidence-executable",
        str(args.dyn_evidence_executable),
    ]


def test_missing_canonical_atlas_authority_fails_before_generation(
    tmp_path: Path,
) -> None:
    args = _fixture_args(tmp_path, tag="missing-atlas-cm")
    (args.client_root / "release/q2-cm-oracle").unlink()

    with pytest.raises(
        FinalCohortPlanError, match="canonical Atlas CM oracle.*absent"
    ) as raised:
        build_plan(args)

    assert raised.value.defect_class == DEFECT_RUNNER_CONFIGURATION
    assert raised.value.check_id == "path-identity"
    assert not args.workspace.exists()


def test_canonical_atlas_authority_must_match_supplied_bytes(
    tmp_path: Path,
) -> None:
    args = _fixture_args(tmp_path, tag="mismatched-atlas-hook")
    (args.lithium_root / "tools/q2-hook-oracle").write_bytes(b"wrong hook\n")

    with pytest.raises(
        FinalCohortPlanError,
        match="canonical Atlas hook_oracle bytes differ",
    ) as raised:
        build_plan(args)

    assert raised.value.defect_class == DEFECT_RUNNER_CONFIGURATION
    assert raised.value.check_id == "atlas-release-closure"
    assert not args.workspace.exists()


@pytest.mark.parametrize("tool_name", ("packer", "verifier"))
def test_exact_atlas_tool_closure_is_required_before_generation(
    tmp_path: Path, tool_name: str,
) -> None:
    args = _fixture_args(tmp_path, tag=f"missing-atlas-{tool_name}")
    getattr(args, tool_name).unlink()

    with pytest.raises(FinalCohortPlanError, match=f"Atlas {tool_name}.*absent") as raised:
        build_plan(args)

    assert raised.value.defect_class == DEFECT_RUNNER_CONFIGURATION
    assert raised.value.check_id == "path-identity"
    assert not args.workspace.exists()


def test_final_plan_pins_complete_atlas_release_closure(tmp_path: Path) -> None:
    args = _fixture_args(tmp_path, tag="complete-atlas-closure")
    plan = build_plan(args)
    pinned = plan["pinned_inputs"]

    assert {
        "packer",
        "verifier",
        "atlas_cm_oracle",
        "atlas_pmove_oracle",
        "atlas_hook_oracle",
    }.issubset(pinned)
    for name in (
        "packer",
        "verifier",
        "atlas_cm_oracle",
        "atlas_pmove_oracle",
        "atlas_hook_oracle",
    ):
        assert len(pinned[name]["sha256"]) == 64

    atlas_command = next(
        step["command"] for step in plan["commands"] if step["stage"] == "atlas-build"
    )
    assert atlas_command[atlas_command.index("--packer") + 1] == str(args.packer.resolve())
    assert atlas_command[atlas_command.index("--verifier") + 1] == str(
        args.verifier.resolve()
    )


@pytest.mark.parametrize(
    ("pinned_name", "mutated_path"),
    (
        ("atlas_cm_oracle", "client"),
        ("packer", "packer"),
    ),
)
def test_atlas_release_closure_drift_is_rejected_before_source(
    tmp_path: Path, pinned_name: str, mutated_path: str,
) -> None:
    args = _fixture_args(tmp_path, tag=f"drift-{pinned_name}")
    plan = build_plan(args)
    path = (
        args.client_root / "release/q2-cm-oracle"
        if mutated_path == "client"
        else args.packer
    )
    path.write_bytes(b"post-plan drift\n")

    with pytest.raises(FinalCohortPlanError) as raised:
        validate_plan(plan)

    assert raised.value.check_id == f"pinned-digest:{pinned_name}"
    assert raised.value.defect_class == DEFECT_RUNNER_CONFIGURATION
    assert not args.workspace.exists()


@pytest.mark.parametrize("flag", ("--client-root", "--packer"))
def test_atlas_command_rebinding_is_rejected_without_consuming_authorization(
    tmp_path: Path, flag: str,
) -> None:
    args = _fixture_args(tmp_path, tag=f"rebind-{flag.removeprefix('--')}")
    plan = build_plan(args)
    atlas_command = next(
        step["command"] for step in plan["commands"] if step["stage"] == "atlas-build"
    )
    atlas_command[atlas_command.index(flag) + 1] = str(tmp_path / "wrong-input")

    check_id = f"stage-input-binding:atlas-{flag.removeprefix('--')}"
    with pytest.raises(FinalCohortPlanError) as raised:
        validate_plan(plan)
    assert raised.value.check_id == check_id

    evidence = run_plan(
        plan,
        dry_run=False,
        runner=lambda *_args, **_kwargs: pytest.fail("source must not run"),
        acknowledge_mutating_execution=MUTATING_EXECUTION_ACK,
    )
    assert evidence["passed"] is False
    assert evidence["failure"]["check_id"] == check_id
    assert evidence["mutating_stages_executed"] == []
    assert evidence["source_authorization_marker"] is None
    assert evidence["authorization"]["source_authorization_consumed"] is False
    assert evidence["authorization"]["cohort_retirement_triggered"] is False


def test_duplicate_atlas_flag_cannot_override_pinned_binding(tmp_path: Path) -> None:
    args = _fixture_args(tmp_path, tag="duplicate-client-root")
    plan = build_plan(args)
    atlas_command = next(
        step["command"] for step in plan["commands"] if step["stage"] == "atlas-build"
    )
    atlas_command.extend(("--client-root", str(tmp_path / "wrong-client")))

    evidence = run_plan(
        plan,
        dry_run=False,
        runner=lambda *_args, **_kwargs: pytest.fail("source must not run"),
        acknowledge_mutating_execution=MUTATING_EXECUTION_ACK,
    )
    assert evidence["passed"] is False
    assert evidence["failure"]["check_id"] == "parser-conformance"
    assert "exactly once; found 2" in evidence["failure"]["message"]
    assert evidence["mutating_stages_executed"] == []
    assert evidence["source_authorization_marker"] is None
    assert evidence["authorization"]["source_authorization_consumed"] is False


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
    with pytest.raises(
        FinalCohortPlanError, match="versioned immutable repository path"
    ) as raised:
        build_plan(args)
    assert raised.value.check_id == "declaration-binding"


@pytest.mark.parametrize("kind", ("current-alias", "wrong-number", "symlink"))
def test_plan_rejects_nonimmutable_named_declaration_paths(
    tmp_path: Path,
    kind: str,
) -> None:
    args = _fixture_args(tmp_path, tag=f"declaration-{kind}")
    immutable = args.declaration
    if kind == "current-alias":
        candidate = immutable.parent / "B2-GENERATED-COHORT-DECLARATION.json"
        candidate.write_bytes(immutable.read_bytes())
    elif kind == "wrong-number":
        candidate = (
            immutable.parent / "B2-GENERATED-COHORT-99001-DECLARATION.json"
        )
        candidate.write_bytes(immutable.read_bytes())
    else:
        candidate = (
            immutable.parent / "B2-GENERATED-COHORT-99000-DECLARATION-LINK.json"
        )
        candidate.symlink_to(immutable)
    args.declaration = candidate
    _activate_fixture_declaration(candidate)

    with pytest.raises(FinalCohortPlanError) as raised:
        build_plan(args)
    assert raised.value.check_id in {"declaration-binding", "path-identity"}


def test_plan_rejects_output_roots_outside_the_workspace(tmp_path: Path) -> None:
    args = _fixture_args(tmp_path, tag="escaped-output")
    args.source_root = ROOT / ".forbidden-final-source-output"

    with pytest.raises(
        FinalCohortPlanError, match="escapes the final-cohort workspace"
    ) as raised:
        build_plan(args)
    assert raised.value.check_id == "path-identity"


def test_plan_rejects_output_roots_through_workspace_symlinks(
    tmp_path: Path,
) -> None:
    args = _fixture_args(tmp_path, tag="symlinked-output")
    args.workspace.mkdir()
    external = tmp_path / "external-output"
    external.mkdir()
    redirect = args.workspace / "redirect"
    redirect.symlink_to(external, target_is_directory=True)
    args.source_root = redirect / "source"

    with pytest.raises(
        FinalCohortPlanError, match="symlinked ancestor"
    ) as raised:
        build_plan(args)
    assert raised.value.check_id == "path-identity"


def test_plan_rejects_a_symlinked_repository_root_before_source(
    tmp_path: Path,
) -> None:
    args = _fixture_args(tmp_path, tag="symlinked-repository")
    linked_repo = tmp_path / "repo-link"
    linked_repo.symlink_to(args.repo_root, target_is_directory=True)
    args.repo_root = linked_repo

    with pytest.raises(
        FinalCohortPlanError, match="direct canonical directory"
    ) as raised:
        build_plan(args)
    assert raised.value.check_id == "path-identity"
    assert not args.workspace.exists()


def test_rebound_assembly_command_is_rejected_before_source(
    tmp_path: Path,
) -> None:
    args = _fixture_args(tmp_path, tag="rebound-assembly")
    plan = build_plan(args)
    assembly = next(
        step for step in plan["commands"] if step["stage"] == "assembly"
    )
    plan_index = assembly["command"].index("--plan") + 1
    assembly["command"][plan_index] = str(tmp_path / "rebound-plan.md")

    with pytest.raises(FinalCohortPlanError) as raised:
        validate_plan(plan)
    assert raised.value.check_id == "assembly-command-binding"

    evidence = run_plan(
        plan,
        dry_run=False,
        runner=lambda *_args, **_kwargs: pytest.fail("source must not run"),
        acknowledge_mutating_execution=MUTATING_EXECUTION_ACK,
    )
    assert evidence["passed"] is False
    assert evidence["failure"]["check_id"] == "assembly-command-binding"
    assert evidence["mutating_stages_executed"] == []
    assert evidence["authorization"]["source_authorization_consumed"] is False


def test_retired_final_cohort_id_is_hard_refused(tmp_path: Path) -> None:
    """A terminal cohort ID remains retired even with fresh maps and seeds."""

    args = _fixture_args(tmp_path, tag="id-71451")
    payload = json.loads(args.declaration.read_text())
    # Keep fresh maps/seeds; only reuse the permanently retired cohort identity.
    payload["cohort_id"] = "b2g26_final_71451"
    retired_identity_path = (
        args.declaration.parent / "B2-GENERATED-COHORT-71451-DECLARATION.json"
    )
    args.declaration.rename(retired_identity_path)
    args.declaration = retired_identity_path
    args.declaration.write_bytes(canonical_bytes(payload))
    _activate_fixture_declaration(args.declaration)
    with pytest.raises(FinalCohortPlanError) as raised:
        build_plan(args)
    assert raised.value.check_id == "retired-registry"


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
        del command
        marker = (
            args.authorization_state_root
            / f"{plan['declaration']['sha256']}.json"
        )
        if stage == "dyn-shape-preflight":
            assert (args.workspace / "reports").is_dir()
            assert not marker.exists()
        if stage == "source":
            assert marker.is_file()
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
    evidence = run_plan(
        plan,
        dry_run=False,
        runner=runner,
        expected_plan_sha256=_reviewed_plan_sha256(plan),
    )
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
        expected_plan_sha256=_reviewed_plan_sha256(plan),
    )
    assert evidence["passed"] is True
    assert calls == list(DRIVER_STAGES)
    assert evidence["mutating_stages_executed"] == list(DRIVER_STAGES)
    assert evidence["authorization"]["source_authorization_consumed"] is True


def test_driver_publishes_exact_preassembly_lifecycle_before_assembly(
    tmp_path: Path,
) -> None:
    """Assembly can consume only the one driver-produced completed prefix."""

    args = _fixture_args(tmp_path)
    plan = build_plan(args)
    seen: dict[str, object] = {}

    def runner(command, *, stage, plan):
        if stage == "assembly":
            lifecycle_path = Path(plan["stage_roots"]["lifecycle_evidence"])
            payload = lifecycle_path.read_bytes()
            lifecycle = json.loads(payload)
            assert payload == canonical_bytes(lifecycle)
            assert lifecycle["schema"] == plan_tool.PREASSEMBLY_LIFECYCLE_SCHEMA
            assert lifecycle["status"] == "ready-for-assembly"
            assert lifecycle["cohort_id"] == plan["cohort_id"]
            assert lifecycle["declaration"] == {
                "path": plan["declaration"]["path"],
                "sha256": plan["declaration"]["sha256"],
            }
            assert lifecycle["plan_sha256"] == _reviewed_plan_sha256(plan)
            assert lifecycle["implementation"] == plan["repository"]
            assert lifecycle["execution_binding"] == plan["authorization"][
                "execution_binding"
            ]
            assert lifecycle["completed_stages"] == list(DRIVER_STAGES[:-1])
            assert [entry["stage"] for entry in lifecycle["stage_executions"]] == list(
                DRIVER_STAGES[:-1]
            )
            assert all(entry["returncode"] == 0 for entry in lifecycle["stage_executions"])
            assert lifecycle["assembly_command_sha256"] == hashlib.sha256(
                canonical_bytes(command)
            ).hexdigest()
            source_marker = lifecycle["source_authorization_marker"]
            assert source_marker["path"] == str(
                args.authorization_state_root
                / f"{plan['declaration']['sha256']}.json"
            )
            seen["lifecycle"] = lifecycle
        return SimpleNamespace(returncode=0, stdout=b"stage-out", stderr=b"stage-err")

    evidence = run_plan(
        plan,
        dry_run=False,
        runner=runner,
        acknowledge_mutating_execution=MUTATING_EXECUTION_ACK,
        expected_plan_sha256=_reviewed_plan_sha256(plan),
    )
    assert evidence["passed"] is True
    assert "lifecycle" in seen


def test_source_command_receives_the_exact_bound_journal_leaf(tmp_path: Path) -> None:
    args = _fixture_args(tmp_path)
    plan = build_plan(args)
    source_command = next(
        step["command"] for step in plan["commands"] if step["stage"] == "source"
    )
    marker_index = source_command.index("--final-source-authorization")
    assert source_command[marker_index + 1] == str(
        args.authorization_state_root / f"{plan['declaration']['sha256']}.json"
    )


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
        expected_plan_sha256=_reviewed_plan_sha256(plan),
    )
    assert evidence["passed"] is False
    assert evidence["defect_class"] == DEFECT_COHORT_ARTIFACT
    assert evidence["mutating_stages_executed"] == [
        "dyn-shape-preflight",
        "source",
        "compile",
    ]
    assert evidence["authorization"]["source_authorization_consumed"] is True
    assert evidence["authorization"]["cohort_retirement_triggered"] is True
    assert evidence["authorization"]["retry_under_same_declaration_allowed"] is False
    marker = evidence["source_authorization_marker"]
    assert marker["path"].endswith(f"/{plan['declaration']['sha256']}.json")
    assert Path(marker["path"]).is_file()


def test_library_execute_requires_the_reviewed_plan_hash(tmp_path: Path) -> None:
    args = _fixture_args(tmp_path)
    plan = build_plan(args)
    calls: list[str] = []

    def runner(_command, *, stage, plan):
        del plan
        calls.append(stage)
        return SimpleNamespace(returncode=0)

    for expected in (None, "0" * 64):
        evidence = run_plan(
            plan,
            dry_run=False,
            runner=runner,
            acknowledge_mutating_execution=MUTATING_EXECUTION_ACK,
            expected_plan_sha256=expected,
        )
        assert evidence["passed"] is False
        assert evidence["failure"]["check_id"] == "expected-plan-sha256"
        assert evidence["mutating_stages_executed"] == []
        assert evidence["authorization"]["source_authorization_consumed"] is False
        assert calls == []


def test_phase_a_input_mutation_is_rejected_before_source_journal(
    tmp_path: Path,
) -> None:
    args = _fixture_args(tmp_path)
    plan = build_plan(args)
    calls: list[str] = []

    def runner(_command, *, stage, plan):
        del plan
        calls.append(stage)
        assert stage == "dyn-shape-preflight"
        args.stock_inventory.write_bytes(b"mutated after Phase A\n")
        return SimpleNamespace(returncode=0)

    evidence = run_plan(
        plan,
        dry_run=False,
        runner=runner,
        acknowledge_mutating_execution=MUTATING_EXECUTION_ACK,
        expected_plan_sha256=_reviewed_plan_sha256(plan),
    )
    assert evidence["passed"] is False
    assert evidence["failure"]["check_id"].startswith(
        "stage-revalidation:source:pinned-digest:stock_inventory"
    )
    assert calls == ["dyn-shape-preflight"]
    assert evidence["mutating_stages_executed"] == ["dyn-shape-preflight"]
    assert evidence["authorization"]["source_authorization_consumed"] is False
    marker = args.authorization_state_root / f"{plan['declaration']['sha256']}.json"
    assert not marker.exists()


def test_preactivation_evidence_drift_is_rejected_before_source_journal(
    tmp_path: Path,
) -> None:
    args = _fixture_args(tmp_path)
    plan = build_plan(args)
    calls: list[str] = []

    def runner(_command, *, stage, plan):
        del plan
        calls.append(stage)
        assert stage == "dyn-shape-preflight"
        args.preactivation_test_report.write_bytes(b"mutated preactivation evidence\n")
        return SimpleNamespace(returncode=0)

    evidence = run_plan(
        plan,
        dry_run=False,
        runner=runner,
        acknowledge_mutating_execution=MUTATING_EXECUTION_ACK,
        expected_plan_sha256=_reviewed_plan_sha256(plan),
    )

    assert evidence["passed"] is False
    assert evidence["failure"]["check_id"].startswith(
        "stage-revalidation:source:pinned-digest:preactivation_test_report"
    )
    assert calls == ["dyn-shape-preflight"]
    assert evidence["authorization"]["source_authorization_consumed"] is False
    marker = args.authorization_state_root / f"{plan['declaration']['sha256']}.json"
    assert not marker.exists()


def test_execution_binding_drift_rejects_before_phase_a(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _fixture_args(tmp_path)
    plan = build_plan(args)

    def rejected_binding(*_args, **_kwargs):
        raise plan_tool.FinalExecutionBindingError("fixture host changed")

    monkeypatch.setattr(plan_tool, "validate_execution_binding", rejected_binding)
    evidence = run_plan(
        plan,
        dry_run=False,
        runner=lambda *_args, **_kwargs: pytest.fail("Phase A must not run"),
        acknowledge_mutating_execution=MUTATING_EXECUTION_ACK,
        expected_plan_sha256=_reviewed_plan_sha256(plan),
    )

    assert evidence["passed"] is False
    assert evidence["failure"]["check_id"] == "execution-authorization-binding"
    assert evidence["mutating_stages_executed"] == []
    assert evidence["authorization"]["source_authorization_consumed"] is False


def test_post_source_input_mutation_is_terminal_before_next_stage(
    tmp_path: Path,
) -> None:
    args = _fixture_args(tmp_path)
    plan = build_plan(args)
    calls: list[str] = []

    def runner(_command, *, stage, plan):
        del plan
        calls.append(stage)
        if stage == "source":
            args.stock_inventory.write_bytes(b"mutated after source\n")
        return SimpleNamespace(returncode=0)

    evidence = run_plan(
        plan,
        dry_run=False,
        runner=runner,
        acknowledge_mutating_execution=MUTATING_EXECUTION_ACK,
        expected_plan_sha256=_reviewed_plan_sha256(plan),
    )
    assert evidence["passed"] is False
    assert evidence["failure"]["check_id"].startswith(
        "stage-revalidation:compile:pinned-digest:stock_inventory"
    )
    assert calls == ["dyn-shape-preflight", "source"]
    assert evidence["authorization"]["source_authorization_consumed"] is True
    assert evidence["authorization"]["cohort_retirement_triggered"] is True


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
        expected_plan_sha256=_reviewed_plan_sha256(plan),
    )
    assert first["passed"] is True
    assert calls == list(DRIVER_STAGES)

    calls.clear()
    second = run_plan(
        plan,
        dry_run=False,
        runner=runner,
        acknowledge_mutating_execution=True,
        expected_plan_sha256=_reviewed_plan_sha256(plan),
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
        expected_plan_sha256=_reviewed_plan_sha256(plan),
    )
    assert first["passed"] is True

    other_args = _fixture_args(tmp_path, tag="other-workspace")
    other_args.authorization_state_root = args.authorization_state_root
    other_args.declaration.write_bytes(args.declaration.read_bytes())
    _activate_fixture_declaration(other_args.declaration)
    other_plan = build_plan(other_args)

    evidence = run_plan(
        other_plan,
        dry_run=False,
        runner=lambda *_args, **_kwargs: pytest.fail("source must not run"),
        acknowledge_mutating_execution=MUTATING_EXECUTION_ACK,
        expected_plan_sha256=_reviewed_plan_sha256(other_plan),
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
    fsync_calls = 0

    def interrupted_fsync(descriptor: int) -> None:
        nonlocal fsync_calls
        fsync_calls += 1
        # The driver durably creates workspace/reports before Phase A. Fail the
        # next fsync, which is the source-journal inode after O_EXCL.
        if fsync_calls == 3:
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
        expected_plan_sha256=_reviewed_plan_sha256(plan),
    )
    assert first["passed"] is False
    assert calls == ["dyn-shape-preflight"]
    assert first["failure"]["check_id"] == (
        "source-authorization-journal-incomplete"
    )
    assert first["authorization"]["source_authorization_consumed"] is True
    marker = args.authorization_state_root / f"{plan['declaration']['sha256']}.json"
    assert marker.is_file()

    calls.clear()
    second = run_plan(
        plan,
        dry_run=False,
        runner=runner,
        acknowledge_mutating_execution=True,
        expected_plan_sha256=_reviewed_plan_sha256(plan),
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
        if stage == "dyn-shape-preflight":
            return SimpleNamespace(returncode=0)
        assert stage == "source"
        raise RuntimeError("crash after first source byte")

    evidence = run_plan(
        plan,
        dry_run=False,
        runner=runner,
        acknowledge_mutating_execution=MUTATING_EXECUTION_ACK,
        expected_plan_sha256=_reviewed_plan_sha256(plan),
    )

    assert evidence["passed"] is False
    assert evidence["failure"]["check_id"] == "stage-exception:source"
    assert evidence["failure"]["exception_type"] == "RuntimeError"
    assert evidence["mutating_stages_executed"] == [
        "dyn-shape-preflight",
        "source",
    ]
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
    def runner(_command, *, stage, plan):
        del plan
        if stage == "dyn-shape-preflight":
            return SimpleNamespace(returncode=0)
        pytest.fail("source must not run")

    evidence = run_plan(
        plan,
        dry_run=False,
        runner=runner,
        acknowledge_mutating_execution=MUTATING_EXECUTION_ACK,
        expected_plan_sha256=_reviewed_plan_sha256(plan),
    )

    assert evidence["passed"] is False
    assert evidence["failure"]["check_id"] == "source-authorization-journal"
    assert evidence["mutating_stages_executed"] == ["dyn-shape-preflight"]
    assert evidence["source_authorization_marker"] is None
    assert evidence["authorization"]["source_authorization_consumed"] is False
    assert evidence["authorization"]["cohort_retirement_triggered"] is False


def test_cli_execute_requires_reviewed_hash_and_ack_then_runs_complete_plan(
    tmp_path: Path, capsys, monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _fixture_args(tmp_path)
    plan_sha256 = hashlib.sha256(canonical_bytes(build_plan(args))).hexdigest()

    code = plan_tool.main([*_cli_base(args), "--execute"])
    assert code == 1
    loaded = json.loads(capsys.readouterr().out)
    assert loaded["passed"] is False
    assert loaded["mutating_stages_executed"] == []
    assert loaded["authorization"]["source_authorization_consumed"] is False
    assert loaded["defect_class"] == DEFECT_RUNNER_CONFIGURATION
    assert loaded["failure"]["check_id"] == "expected-plan-sha256"
    assert not (args.workspace / "source").exists()
    assert not (args.workspace / "compiled").exists()

    code = plan_tool.main(
        [
            *_cli_base(args),
            "--execute",
            "--expected-plan-sha256",
            plan_sha256,
        ]
    )
    assert code == 1
    loaded = json.loads(capsys.readouterr().out)
    assert loaded["passed"] is False
    assert loaded["mutating_stages_executed"] == []
    assert loaded["authorization"]["source_authorization_consumed"] is False
    assert loaded["failure"]["check_id"] == "execute-acknowledgement"

    calls: list[str] = []

    def runner(_command, *, stage, plan):
        del plan
        calls.append(stage)
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setattr(plan_tool, "_subprocess_stage_runner", runner)
    code = plan_tool.main(
        [
            *_cli_base(args),
            "--execute",
            "--expected-plan-sha256",
            plan_sha256,
            "--acknowledge-mutating-execution",
            MUTATING_EXECUTION_ACK,
        ]
    )
    assert code == 0
    loaded = json.loads(capsys.readouterr().out)
    assert loaded["passed"] is True
    assert loaded["mutating_stages_executed"] == list(DRIVER_STAGES)
    assert loaded["authorization"]["source_authorization_consumed"] is True
    assert calls == list(DRIVER_STAGES)
    assert [row["stage"] for row in loaded["stage_executions"]] == list(DRIVER_STAGES)


def test_workspace_must_be_outside_repository(tmp_path: Path) -> None:
    args = _fixture_args(tmp_path)
    args.workspace = args.repo_root / "would-be-final-workspace"
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
