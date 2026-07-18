from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from tests.b5_evidence_fixtures import (
    AUTHORITY,
    SOURCE,
    campaign,
    input_bindings,
    proof,
    write_inputs,
)
import tools.assemble_b5_gate as gate_tool
import tools.assemble_b4_evidence as b4_tool
import tools.assemble_b6_wsl_g1_campaign as b6_tool
import tools.run_multires_pretraining_validation as validation_tool
from tools.assemble_b5_gate import B5GateError, assemble_gate, gate_sha256
from tools.run_multires_pretraining_validation import (
    ValidationConfig,
    campaign_result_sha256,
    canonical_bytes,
    report_sha256,
    run_validation,
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


def _qualified(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
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
        return subprocess.CompletedProcess(command, 0, b"", b"")

    validation_path = tmp_path / "validation.json"
    work = tmp_path / "work"
    monkeypatch.setattr(validation_tool, "_run_command", runner)
    monkeypatch.setattr(validation_tool, "git_identity", lambda _root: dict(SOURCE))
    report = run_validation(ValidationConfig(
        repo_root=Path(__file__).resolve().parents[1],
        **_runtime_arguments(tmp_path, paths), seed=17, game_seed=23,
        campaign_transitions=64,
        jobs=3, timeout_seconds=10, work_dir=work, output=validation_path,
    ))
    return paths, validation_path, work / "proof-500.json", report


def _assemble(tmp_path: Path, paths, validation: Path, proof_path: Path):
    return assemble_gate(
        repo_root=Path(__file__).resolve().parents[1],
        validation_report=validation, proof_report=proof_path,
        b3_gate=paths["b3_gate"], b4_evidence=paths["b4_evidence"],
        runtime_manifest=paths["runtime_manifest"], checkpoint=paths["checkpoint"],
        training_manifest=paths["training_manifest"],
        bundle_manifest=paths["bundle_manifest"], objectives=paths["objectives"],
        atlas=paths["atlas"],
        output=tmp_path / "B5-GATE.json",
    )


def test_assembler_revalidates_and_publishes_green_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, validation, proof_path, report = _qualified(tmp_path, monkeypatch)
    monkeypatch.setattr(gate_tool, "git_identity", lambda _root: dict(SOURCE))
    gate = _assemble(tmp_path, paths, validation, proof_path)
    assert gate["schema"] == "q2-multires-b5-gate-v1"
    assert gate["green"] is True
    assert gate["campaigns"]["campaign_count"] == 10
    assert gate["proof"]["transition_count"] == 500
    assert gate["privilege"]["basis"] == "sealed-b4-real-public-datagram-audit-v1"
    assert gate["privilege"]["public_packets_decoded"] == 32
    assert gate["gate_sha256"] == gate_sha256(gate)
    assert b6_tool._validate_b5(
        gate,
        objective_identity_sha256=hashlib.sha256(
            paths["objectives"].read_bytes()
        ).hexdigest(),
    ) == gate
    assert (tmp_path / "B5-GATE.json").read_bytes() == canonical_bytes(gate)
    jsonschema = pytest.importorskip("jsonschema")
    root = Path(__file__).resolve().parents[1]
    validation_schema = json.loads(
        (root / "schemas/q2-multires-pretraining-validation-v1.schema.json").read_text()
    )
    gate_schema = json.loads(
        (root / "schemas/q2-multires-b5-gate-v1.schema.json").read_text()
    )
    jsonschema.Draft202012Validator.check_schema(validation_schema)
    jsonschema.Draft202012Validator.check_schema(gate_schema)
    jsonschema.Draft202012Validator(validation_schema).validate(report)
    jsonschema.Draft202012Validator(gate_schema).validate(gate)


def test_assembler_rejects_missing_or_changed_proof_without_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, validation, proof_path, _report = _qualified(tmp_path, monkeypatch)
    monkeypatch.setattr(gate_tool, "git_identity", lambda _root: dict(SOURCE))
    body = json.loads(proof_path.read_bytes())
    body["same_seed_match"] = False
    proof_path.write_bytes(canonical_bytes(body)[:-1])
    with pytest.raises(B5GateError, match="proof report differs"):
        _assemble(tmp_path, paths, validation, proof_path)
    assert not (tmp_path / "B5-GATE.json").exists()


def test_resigned_validation_cannot_forge_runtime_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, validation, proof_path, report = _qualified(tmp_path, monkeypatch)
    monkeypatch.setattr(gate_tool, "git_identity", lambda _root: dict(SOURCE))
    report["bindings"]["checkpoint"]["sha256"] = "ab" * 32
    report["report_sha256"] = report_sha256(report)
    validation.write_bytes(canonical_bytes(report))
    with pytest.raises(B5GateError, match="checkpoint.*differs"):
        _assemble(tmp_path, paths, validation, proof_path)
    assert not (tmp_path / "B5-GATE.json").exists()


def test_assembler_rejects_b3_substitution_after_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, validation, proof_path, _report = _qualified(tmp_path, monkeypatch)
    monkeypatch.setattr(gate_tool, "git_identity", lambda _root: dict(SOURCE))
    replacement = json.loads(paths["b3_gate"].read_bytes())
    replacement["gate_sha256"] = "ab" * 32
    paths["b3_gate"].write_bytes(canonical_bytes(replacement))
    with pytest.raises(B5GateError, match="validation b3_gate differs"):
        _assemble(tmp_path, paths, validation, proof_path)
    assert not (tmp_path / "B5-GATE.json").exists()


def test_resigned_validation_cannot_forge_semantic_campaign_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, validation, proof_path, report = _qualified(tmp_path, monkeypatch)
    monkeypatch.setattr(gate_tool, "git_identity", lambda _root: dict(SOURCE))
    holdout = next(
        row for row in report["campaigns"]
        if row["campaign"] == "aim_combat_holdout" and row["replicate"] == 0
    )
    holdout["results"]["hidden_fire"] = 1
    holdout["result_sha256"] = campaign_result_sha256(holdout)
    report["report_sha256"] = report_sha256(report)
    validation.write_bytes(canonical_bytes(report))
    with pytest.raises(B5GateError, match="hidden fire"):
        _assemble(tmp_path, paths, validation, proof_path)
    assert not (tmp_path / "B5-GATE.json").exists()


def test_gate_output_is_exclusive_create(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, validation, proof_path, _report = _qualified(tmp_path, monkeypatch)
    monkeypatch.setattr(gate_tool, "git_identity", lambda _root: dict(SOURCE))
    output = tmp_path / "B5-GATE.json"
    output.write_text("do not replace", encoding="ascii")
    with pytest.raises(B5GateError, match="already exists"):
        _assemble(tmp_path, paths, validation, proof_path)
    assert output.read_text(encoding="ascii") == "do not replace"
