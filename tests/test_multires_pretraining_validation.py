from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest

import tools.run_multires_pretraining_validation as validation_tool
import tools.assemble_b4_evidence as b4_tool
from tests.b5_evidence_fixtures import (
    AUTHORITY,
    RUNTIME_IDENTITY,
    SOURCE,
    campaign,
    input_bindings,
    proof,
    write_inputs,
)
from tools.run_multires_pretraining_validation import (
    REQUIRED_CAMPAIGN_MODES,
    PretrainingValidationError,
    ValidationConfig,
    canonical_bytes,
    run_validation,
    validate_campaign_evidence,
    validate_campaign_set,
    validate_proof,
)


def _inputs(tmp_path: Path) -> dict[str, Path]:
    return write_inputs(tmp_path)


def _runtime_arguments(tmp_path: Path, paths: dict[str, Path]) -> dict:
    return {
        "b3_gate": paths["b3_gate"],
        "b4_evidence": paths["b4_evidence"],
        "runtime_manifest": paths["runtime_manifest"],
        "checkpoint": paths["checkpoint"],
        "training_manifest": paths["training_manifest"],
        "bundle_manifest": paths["bundle_manifest"],
        "objectives": paths["objectives"],
        "atlas": paths["atlas"],
        "q2ded": Path("/usr/bin/true").resolve(),
        "client_binary": Path("/usr/bin/true").resolve(),
        "runtime_root": tmp_path,
    }


@pytest.fixture(autouse=True)
def _active_predecessor(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(b4_tool.b2_gate, "ACTIVE_FINAL_AUTHORITY", AUTHORITY)
    monkeypatch.setattr(validation_tool, "validate_b3_gate", lambda value: value)


def test_campaign_contract_and_deterministic_schedule_are_exact(tmp_path: Path) -> None:
    bindings = input_bindings(_inputs(tmp_path))
    rows = []
    for mode in REQUIRED_CAMPAIGN_MODES:
        for replicate in (0, 1):
            row = campaign(mode, replicate, bindings)
            rows.append(validate_campaign_evidence(
                row, mode=mode, replicate=replicate, seed=17, game_seed=23,
                transition_count=64, source=SOURCE, bindings=bindings,
                runtime_manifest_sha256=RUNTIME_IDENTITY,
            ))
    summary = validate_campaign_set(rows)
    assert summary["campaign_count"] == 10
    assert summary["deterministic_pairs"] == 5


@pytest.mark.parametrize("field", ["result_sha256", "trajectory_sha256"])
def test_campaign_rejects_forged_digest(tmp_path: Path, field: str) -> None:
    bindings = input_bindings(_inputs(tmp_path))
    row = campaign("hazard_hook", 0, bindings)
    row[field] = "ab" * 32
    with pytest.raises(PretrainingValidationError, match="forged|stale"):
        validate_campaign_evidence(
            row, mode="hazard_hook", replicate=0, seed=17, game_seed=23,
            transition_count=64, source=SOURCE, bindings=bindings,
            runtime_manifest_sha256=RUNTIME_IDENTITY,
        )


def test_campaign_rejects_update_and_rate_reward(tmp_path: Path) -> None:
    bindings = input_bindings(_inputs(tmp_path))
    row = campaign("hazard_hook", 0, bindings)
    row["counters"]["optimizer_steps"] = 1
    row["results"]["rate_reward_violations"] = 1
    # Re-signing cannot make semantically forbidden evidence admissible.
    from tools.run_multires_pretraining_validation import campaign_result_sha256
    row["result_sha256"] = campaign_result_sha256(row)
    with pytest.raises(PretrainingValidationError, match="optimizer_steps"):
        validate_campaign_evidence(
            row, mode="hazard_hook", replicate=0, seed=17, game_seed=23,
            transition_count=64, source=SOURCE, bindings=bindings,
            runtime_manifest_sha256=RUNTIME_IDENTITY,
        )


def test_campaign_rejects_vacuous_offline_privilege_zero(tmp_path: Path) -> None:
    bindings = input_bindings(_inputs(tmp_path))
    row = campaign("guide_on", 0, bindings)
    row["season_report"]["privilege"] = {"teacher_field_violations": 0}
    row["season_report_sha256"] = validation_tool.canonical_sha256(
        row["season_report"]
    )
    row["result_sha256"] = validation_tool.campaign_result_sha256(row)
    with pytest.raises(PretrainingValidationError, match="privilege scope"):
        validate_campaign_evidence(
            row, mode="guide_on", replicate=0, seed=17, game_seed=23,
            transition_count=64, source=SOURCE, bindings=bindings,
            runtime_manifest_sha256=RUNTIME_IDENTITY,
        )


def test_proof_requires_exact_production_500_and_runtime_binding(tmp_path: Path) -> None:
    bindings = input_bindings(_inputs(tmp_path))
    good = proof(bindings)
    assert validate_proof(
        good, seed=17, game_seed=23, bindings=bindings,
        runtime_manifest_sha256=RUNTIME_IDENTITY,
        objective_identity_sha256=bindings["objectives"]["sha256"],
    ) == good
    bad = copy.deepcopy(good)
    bad["runtime_manifest_sha256"] = "ef" * 32
    with pytest.raises(PretrainingValidationError, match="runtime manifest"):
        validate_proof(
            bad, seed=17, game_seed=23, bindings=bindings,
            runtime_manifest_sha256=RUNTIME_IDENTITY,
            objective_identity_sha256=bindings["objectives"]["sha256"],
        )


def test_suite_runs_ten_campaigns_and_exact_proof_then_publishes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _inputs(tmp_path)
    bindings = input_bindings(paths)
    commands: list[list[str]] = []

    def runner(argv, _timeout):
        command = list(argv)
        commands.append(command)
        if "--campaign" in command:
            mode = command[command.index("--campaign") + 1]
            replicate = int(command[command.index("--replicate") + 1])
            output = Path(command[command.index("--output") + 1])
            output.write_bytes(canonical_bytes(campaign(mode, replicate, bindings)))
        else:
            output = Path(command[command.index("--out") + 1])
            output.write_bytes(canonical_bytes(proof(bindings))[:-1])
        return subprocess.CompletedProcess(command, 0, b"", b"")

    output = tmp_path / "suite.json"
    work = tmp_path / "work"
    monkeypatch.setattr(validation_tool, "_run_command", runner)
    monkeypatch.setattr(validation_tool, "git_identity", lambda _root: dict(SOURCE))
    report = run_validation(ValidationConfig(
        repo_root=Path(__file__).resolve().parents[1],
        **_runtime_arguments(tmp_path, paths), seed=17, game_seed=23,
        campaign_transitions=64, jobs=4, timeout_seconds=10,
        work_dir=work, output=output,
    ))
    assert report["passed"] is True
    assert output.read_bytes() == canonical_bytes(report)
    assert len(commands) == 11
    proof_command = next(command for command in commands if "--out" in command)
    assert proof_command[proof_command.index("--mode") + 1] == "production"
    assert proof_command[proof_command.index("--transition_count") + 1] == "500"
    assert proof_command[proof_command.index("--game_seed") + 1] == "23"
    assert proof_command[proof_command.index("--objectives") + 1] == str(
        paths["objectives"]
    )
    assert proof_command[proof_command.index("--trainer_executable") + 1] == str(
        Path(sys.executable).resolve()
    )
    assert proof_command[proof_command.index("--trainer_arg") + 1] == str(
        (Path(__file__).resolve().parents[1] / "train/multires_one_run.py").resolve()
    )
    assert report["b4_privilege_admission"]["datagrams_seen"] == 32
    assert report["b4_privilege_admission"]["teacher_packets_detected"] == 0


def test_suite_rejects_resealed_b4_teacher_violation_before_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _inputs(tmp_path)
    called = False

    def runner(_argv, _timeout):
        nonlocal called
        called = True
        raise AssertionError("must not execute")

    monkeypatch.setattr(validation_tool, "_run_command", runner)
    monkeypatch.setattr(validation_tool, "git_identity", lambda _root: dict(SOURCE))
    b4_path = paths["b4_evidence"]
    b4 = json.loads(b4_path.read_bytes())
    b4["public_privilege_proof"]["teacher_packets_detected"] = 1
    body = dict(b4)
    body.pop("evidence_sha256")
    import hashlib
    b4["evidence_sha256"] = hashlib.sha256(canonical_bytes(body)[:-1]).hexdigest()
    b4_path.write_bytes(canonical_bytes(b4))
    with pytest.raises(PretrainingValidationError, match="B4 privilege evidence"):
        run_validation(ValidationConfig(
            repo_root=Path(__file__).resolve().parents[1],
            **_runtime_arguments(tmp_path, paths), seed=17, game_seed=23,
            campaign_transitions=64, jobs=1, timeout_seconds=10,
            work_dir=tmp_path / "work", output=tmp_path / "out.json",
        ))
    assert called is False


def test_suite_refuses_out_of_order_when_b2_b3_authority_is_inactive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _inputs(tmp_path)
    called = False

    def runner(_argv, _timeout):
        nonlocal called
        called = True
        raise AssertionError("must not execute")

    monkeypatch.setattr(validation_tool, "_run_command", runner)
    monkeypatch.setattr(validation_tool, "git_identity", lambda _root: dict(SOURCE))
    monkeypatch.setattr(b4_tool.b2_gate, "ACTIVE_FINAL_AUTHORITY", None)
    with pytest.raises(PretrainingValidationError, match="no active final cohort"):
        run_validation(ValidationConfig(
            repo_root=Path(__file__).resolve().parents[1],
            **_runtime_arguments(tmp_path, paths), seed=17, game_seed=23,
            campaign_transitions=64, jobs=1, timeout_seconds=10,
            work_dir=tmp_path / "work", output=tmp_path / "out.json",
        ))
    assert called is False


def test_suite_rejects_cross_artifact_b3_substitution_before_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _inputs(tmp_path)
    replacement = json.loads(paths["b3_gate"].read_bytes())
    replacement["gate_sha256"] = "ab" * 32
    paths["b3_gate"].write_bytes(canonical_bytes(replacement))
    called = False

    def runner(_argv, _timeout):
        nonlocal called
        called = True
        raise AssertionError("must not execute")

    monkeypatch.setattr(validation_tool, "_run_command", runner)
    monkeypatch.setattr(validation_tool, "git_identity", lambda _root: dict(SOURCE))
    with pytest.raises(
        PretrainingValidationError, match="exact supplied B3 gate"
    ):
        run_validation(ValidationConfig(
            repo_root=Path(__file__).resolve().parents[1],
            **_runtime_arguments(tmp_path, paths), seed=17, game_seed=23,
            campaign_transitions=64, jobs=1, timeout_seconds=10,
            work_dir=tmp_path / "work", output=tmp_path / "out.json",
        ))
    assert called is False


def test_suite_rejects_runtime_atlas_mismatch_before_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _inputs(tmp_path)
    runtime = json.loads(paths["runtime_manifest"].read_bytes())
    runtime["atlas_sha256"] = "ab" * 32
    paths["runtime_manifest"].write_bytes(canonical_bytes(runtime))
    called = False

    def runner(_argv, _timeout):
        nonlocal called
        called = True
        raise AssertionError("must not execute")

    monkeypatch.setattr(validation_tool, "_run_command", runner)
    monkeypatch.setattr(validation_tool, "git_identity", lambda _root: dict(SOURCE))
    with pytest.raises(PretrainingValidationError, match="runtime evidence"):
        run_validation(ValidationConfig(
            repo_root=Path(__file__).resolve().parents[1],
            **_runtime_arguments(tmp_path, paths), seed=17, game_seed=23,
            campaign_transitions=64,
            jobs=1, timeout_seconds=10, work_dir=tmp_path / "work",
            output=tmp_path / "out.json",
        ))
    assert called is False


def test_suite_publishes_nothing_if_checkpoint_changes_during_proof(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _inputs(tmp_path)
    bindings = input_bindings(paths)

    def runner(argv, _timeout):
        command = list(argv)
        if "--campaign" in command:
            mode = command[command.index("--campaign") + 1]
            replicate = int(command[command.index("--replicate") + 1])
            output = Path(command[command.index("--output") + 1])
            output.write_bytes(canonical_bytes(campaign(mode, replicate, bindings)))
        else:
            output = Path(command[command.index("--out") + 1])
            output.write_bytes(canonical_bytes(proof(bindings))[:-1])
            paths["checkpoint"].write_bytes(b"forbidden update\n")
        return subprocess.CompletedProcess(command, 0, b"", b"")

    monkeypatch.setattr(validation_tool, "_run_command", runner)
    monkeypatch.setattr(validation_tool, "git_identity", lambda _root: dict(SOURCE))
    output = tmp_path / "validation.json"
    with pytest.raises(PretrainingValidationError, match="immutable.*changed"):
        run_validation(ValidationConfig(
            repo_root=Path(__file__).resolve().parents[1],
            **_runtime_arguments(tmp_path, paths), seed=17, game_seed=23,
            campaign_transitions=64,
            jobs=4, timeout_seconds=10, work_dir=tmp_path / "work",
            output=output,
        ))
    assert not output.exists()
