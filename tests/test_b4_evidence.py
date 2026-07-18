from __future__ import annotations

import copy
import hashlib
import inspect
import json
from pathlib import Path
import subprocess

import pytest

from harness.runtime_attestation import MANIFEST_SCHEMA, semantic_digest
from tools import assemble_b4_evidence as b4
from tools import verify_multires_integration as integration


ATLAS = "a" * 64
AUTHORITY = b4.b2_gate.ActiveFinalAuthority(
    cohort_id="b2g99_final_99999",
    declaration_sha256="d" * 64,
    immutable_declaration_path="docs/multires/B2-GENERATED-COHORT-99999-DECLARATION.json",
    qualification_successor_paths=frozenset({
        "docs/multires/B2-GENERATED-COHORT-99999-DECLARATION.json"
    }),
)
INTEGRATION_ROOT = Path(__file__).resolve().parents[2]
BOT_REPO = INTEGRATION_ROOT / "q2-ml-bot"
CLIENT_REPO = INTEGRATION_ROOT / "q2-ml-client"
GAME_REPO = INTEGRATION_ROOT / "q2-lithium-3zb2"


def _write_compact(path: Path, value: object) -> None:
    path.write_bytes(b4._canonical_bytes(value) + b"\n")


def _write_manifest(path: Path, value: object) -> None:
    path.write_bytes(b4._manifest_bytes(value))


def _binary(name: str, character: str) -> dict[str, object]:
    return {"name": name, "sha256": character * 64, "size": 123}


def _identity(character: str) -> dict[str, object]:
    return {"commit": character * 40, "tree": character.upper().lower() * 40, "clean": True}


def _raw_record(path: Path, scenario: str, raw: dict[str, object]) -> dict[str, object]:
    data = path.read_bytes()
    return {
        "scenario": scenario,
        "path": path.name,
        "sha256": hashlib.sha256(data).hexdigest(),
        "size": len(data),
        "raw_evidence_sha256": raw["evidence_sha256"],
    }


def _source_contract() -> dict[str, object]:
    return b4._source_contract(BOT_REPO, CLIENT_REPO, GAME_REPO)


def _fixture(tmp_path: Path) -> dict[str, object]:
    source_repositories = {
        "bot": _identity("1"),
        "client": _identity("2"),
        "game": _identity("3"),
    }
    rows = [{
        "ordinal": 1,
        "clients": [
            {
                "client_id": f"qual-{slot:02d}",
                "map_epoch": 7,
                "map_name": "q2dm1",
                "action_tick": 1,
                "causal": {
                    "client_life_epoch": 1,
                    "echo_valid": True,
                    "transition_trainable": True,
                    "action_generation": slot + 1,
                },
                "applied_action": {"vertical_intent": slot % 3},
            }
            for slot in range(4)
        ],
    }]
    baseline = b4._seal({
        "scenario": "baseline-cold-1",
        "trajectory_rows": rows,
        "public_telemetry_audit": [
            {
                "client_id": f"qual-{slot:02d}",
                "datagrams_seen": 33,
                "public_packets_decoded": 33,
                "routed_packets_accepted": 33,
                "malformed_packets_rejected": 0,
                "foreign_client_packets_rejected": 0,
                "stale_packets_rejected": 0,
                "teacher_packets_detected": 0,
            }
            for slot in range(4)
        ],
    })
    epoch = b4._seal({
        "scenario": "epoch-drain",
        "actions_dispatched_during_epoch_drain": 0,
    })
    stale = b4._seal({
        "scenario": "old-telemetry",
        "structured_events": [{
            "event": "old_telemetry",
            "result": "discarded",
            "clock_regressed": "0",
        }],
    })
    raw_values = {
        "baseline-cold-1": baseline,
        "epoch-drain": epoch,
        "old-telemetry": stale,
    }
    records = []
    for name, raw in raw_values.items():
        path = tmp_path / f"{name}.json"
        _write_compact(path, raw)
        records.append(_raw_record(path, name, raw))

    binaries = {
        "q2ded": _binary("q2ded", "4"),
        "game_module": _binary("game.so", "5"),
        "client_binary": _binary("yquake2", "6"),
    }
    execution_digest = "7" * 64
    execution = {
        "test_mode": False,
        "full_network_executed": True,
        "execution_evidence_sha256": execution_digest,
        "runtime_binaries": binaries,
        "source_repositories": source_repositories,
        "source_closure_sha256": hashlib.sha256(
            b4._canonical_bytes(source_repositories)
        ).hexdigest(),
        "scenario_evidence": records,
    }
    semantic = {
        "artifacts": {
            "q2ded": dict(binaries["q2ded"]),
            "game_module": dict(binaries["game_module"]),
        },
        "runtime_config": {
            "network_barrier_execution_evidence_sha256": execution_digest,
            "expected_atlas_sha256": ATLAS,
            "client_binary_sha256": binaries["client_binary"]["sha256"],
            "client_binary_size": binaries["client_binary"]["size"],
        },
    }
    manifest_digest = semantic_digest(semantic)
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "semantic": semantic,
        "manifest_sha256": manifest_digest,
        "diagnostics": {},
    }
    manifest_path = tmp_path / "runtime-manifest.json"
    _write_manifest(manifest_path, manifest)
    closure = {
        "runtime_manifest_sha256": manifest_digest,
        "execution_evidence_sha256": execution_digest,
    }
    qualification = b4._seal({
        "schema": b4.barrier.SCHEMA,
        "passed": True,
        "mode": b4.barrier.MODE,
        "protocol_version": b4.barrier.PROTOCOL_VERSION,
        "test_mode": False,
        "non_admissible_for_training": True,
        "runtime_manifest_sha256": manifest_digest,
        "execution_evidence_sha256": execution_digest,
        "runtime_closure_sha256": hashlib.sha256(
            b4._canonical_bytes(closure)
        ).hexdigest(),
        "execution_evidence": execution,
    })
    qualification_path = tmp_path / "qualification.json"
    _write_compact(qualification_path, qualification)
    b3_gate = {
        "schema": "q2-multires-b3-gate-v1",
        "batch": "B3",
        "status": "green",
        "normative_documents": {
            "design_sha256": "8" * 64,
            "plan_sha256": "9" * 64,
        },
        "repository": {
            "repository_commit": source_repositories["bot"]["commit"],
            "repository_tree": source_repositories["bot"]["tree"],
            "git_clean": True,
        },
        "predecessor": {
            "status": "green",
            "cohort_id": AUTHORITY.cohort_id,
            "declaration_sha256": AUTHORITY.declaration_sha256,
        },
        "recovery_guide": {"atlas_set_sha256": "e" * 64},
        "gate_sha256": "f" * 64,
    }
    b3_gate_path = tmp_path / "B3-GATE.json"
    _write_compact(b3_gate_path, b3_gate)
    return {
        "source_repositories": source_repositories,
        "execution": execution,
        "manifest": manifest,
        "manifest_path": manifest_path,
        "qualification": qualification,
        "qualification_path": qualification_path,
        "b3_gate": b3_gate,
        "b3_gate_path": b3_gate_path,
        "raw_values": raw_values,
    }


def _patch_sources(monkeypatch: pytest.MonkeyPatch, fixture: dict[str, object]) -> None:
    identities = fixture["source_repositories"]
    monkeypatch.setattr(
        b4,
        "_git_identity",
        lambda path: copy.deepcopy(identities[{  # type: ignore[index]
            "q2-ml-bot": "bot",
            "q2-ml-client": "client",
            "q2-lithium-3zb2": "game",
        }[Path(path).name]]),
    )
    source = _source_contract()
    monkeypatch.setattr(b4, "_source_contract", lambda *unused: copy.deepcopy(source))
    monkeypatch.setattr(
        b4,
        "_normative_records",
        lambda unused: [
            {"name": name, "sha256": character * 64, "size": 10}
            for name, character in zip(b4.NORMATIVE_PATHS, ("8", "9"))
        ],
    )
    monkeypatch.setattr(
        b4.barrier,
        "_validate_execution_evidence",
        lambda execution, unused: dict(execution),
    )
    monkeypatch.setattr(b4.b2_gate, "ACTIVE_FINAL_AUTHORITY", AUTHORITY)
    monkeypatch.setattr(
        b4, "validate_b3_gate", lambda value: copy.deepcopy(value)
    )


def _assemble(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    fixture = _fixture(tmp_path)
    _patch_sources(monkeypatch, fixture)
    aggregate, documents = b4.assemble_b4_evidence(
        b3_gate_path=fixture["b3_gate_path"],  # type: ignore[arg-type]
        qualification_path=fixture["qualification_path"],  # type: ignore[arg-type]
        runtime_manifest_path=fixture["manifest_path"],  # type: ignore[arg-type]
        bot_repo=BOT_REPO,
        client_repo=CLIENT_REPO,
        game_repo=GAME_REPO,
        atlas_sha256=ATLAS,
    )
    return fixture, aggregate, documents


def _rebind_raw_scenario(
    fixture: dict[str, object], tmp_path: Path, name: str, raw: dict[str, object]
) -> None:
    path = tmp_path / f"{name}.json"
    _write_compact(path, raw)
    execution = fixture["execution"]
    for index, record in enumerate(execution["scenario_evidence"]):  # type: ignore[index]
        if record["scenario"] == name:
            execution["scenario_evidence"][index] = _raw_record(path, name, raw)  # type: ignore[index]
            break
    qualification = dict(fixture["qualification"])
    qualification.pop("evidence_sha256")
    qualification["execution_evidence"] = execution
    qualification = b4._seal(qualification)
    fixture["qualification"] = qualification
    _write_compact(fixture["qualification_path"], qualification)  # type: ignore[arg-type]


def test_green_assembly_publishes_integration_admissible_documents(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    unused, aggregate, documents = _assemble(tmp_path, monkeypatch)
    b4.validate_b4_evidence(aggregate, documents)

    context = {"atlas_sha256": ATLAS}
    integration._check_feature_action_contract(
        documents["feature_action_contract"], context
    )
    integration._check_runtime_epoch_fencing(
        documents["runtime_epoch_fencing"], context
    )
    integration._check_b4_wire_generation(documents["b4_wire_generation"], context)
    integration._check_causal_reward_admission(
        documents["causal_reward_admission"], context
    )

    destination = tmp_path / "published"
    b4.publish_b4_evidence(destination, aggregate, documents)
    assert sorted(path.name for path in destination.iterdir()) == sorted(
        ["B4-EVIDENCE.json", *b4.DOCUMENT_NAMES.values()]
    )
    with pytest.raises(b4.B4EvidenceError, match="new path"):
        b4.publish_b4_evidence(destination, aggregate, documents)


def test_b4_refuses_without_active_b2_b3_predecessor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)
    _patch_sources(monkeypatch, fixture)
    monkeypatch.setattr(b4.b2_gate, "ACTIVE_FINAL_AUTHORITY", None)
    with pytest.raises(b4.B4EvidenceError, match="no active final cohort"):
        b4.assemble_b4_evidence(
            b3_gate_path=fixture["b3_gate_path"],  # type: ignore[arg-type]
            qualification_path=fixture["qualification_path"],  # type: ignore[arg-type]
            runtime_manifest_path=fixture["manifest_path"],  # type: ignore[arg-type]
            bot_repo=BOT_REPO, client_repo=CLIENT_REPO, game_repo=GAME_REPO,
            atlas_sha256=ATLAS,
        )


def test_b4_rejects_b3_from_different_bot_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)
    _patch_sources(monkeypatch, fixture)
    b3_gate = copy.deepcopy(fixture["b3_gate"])
    b3_gate["repository"]["repository_tree"] = "c" * 40
    _write_compact(fixture["b3_gate_path"], b3_gate)  # type: ignore[arg-type]
    with pytest.raises(b4.B4EvidenceError, match="bot source differs"):
        b4.assemble_b4_evidence(
            b3_gate_path=fixture["b3_gate_path"],  # type: ignore[arg-type]
            qualification_path=fixture["qualification_path"],  # type: ignore[arg-type]
            runtime_manifest_path=fixture["manifest_path"],  # type: ignore[arg-type]
            bot_repo=BOT_REPO, client_repo=CLIENT_REPO, game_repo=GAME_REPO,
            atlas_sha256=ATLAS,
        )


def test_b4_validator_rejects_resealed_predecessor_authority_forgery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    unused, aggregate, documents = _assemble(tmp_path, monkeypatch)
    forged = copy.deepcopy(aggregate)
    forged.pop("evidence_sha256")
    forged["predecessor"]["cohort_id"] = "b2g98_final_99998"
    forged = b4._seal(forged)
    with pytest.raises(b4.B4EvidenceError, match="predecessor authority"):
        b4.validate_b4_evidence(forged, documents)


def test_b4_validator_rejects_resealed_extra_gate_predicate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    unused, aggregate, documents = _assemble(tmp_path, monkeypatch)
    forged = copy.deepcopy(aggregate)
    forged.pop("evidence_sha256")
    forged["gate"]["self_declared_extra"] = True
    forged = b4._seal(forged)
    with pytest.raises(b4.B4EvidenceError, match="non-green gate"):
        b4.validate_b4_evidence(forged, documents)


def test_b4_validator_rejects_resealed_fixture_shaped_component(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    unused, aggregate, documents = _assemble(tmp_path, monkeypatch)
    forged_documents = copy.deepcopy(documents)
    feature = forged_documents["feature_action_contract"]
    feature.pop("evidence_sha256")
    feature["schema"] = "fixture-feature"
    forged_documents["feature_action_contract"] = b4._seal(feature)
    forged = copy.deepcopy(aggregate)
    forged.pop("evidence_sha256")
    forged["component_evidence"] = forged_documents
    payload = b4._canonical_bytes(forged_documents["feature_action_contract"]) + b"\n"
    forged["documents"]["feature_action_contract"] = {
        "name": b4.DOCUMENT_NAMES["feature_action_contract"],
        "sha256": b4._sha256_bytes(payload),
        "size": len(payload),
        "evidence_sha256": forged_documents["feature_action_contract"][
            "evidence_sha256"
        ],
    }
    forged = b4._seal(forged)
    with pytest.raises(b4.B4EvidenceError, match="feature/action component"):
        b4.validate_b4_evidence(forged, forged_documents)


def test_aggregate_matches_published_json_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    unused, aggregate, unused_documents = _assemble(tmp_path, monkeypatch)
    schema = json.loads(
        (BOT_REPO / "schemas/q2-multires-b4-evidence-v1.schema.json").read_text()
    )
    assert schema["properties"]["schema"]["const"] == b4.SCHEMA
    assert schema["properties"]["status"]["const"] == "green"
    assert set(schema["required"]) == set(aggregate)
    assert set(schema["properties"]["gate"]["required"]) == set(aggregate["gate"])
    try:
        import jsonschema
    except ImportError:
        return
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.Draft202012Validator(schema).validate(aggregate)


@pytest.mark.parametrize("target", ["qualification", "manifest"])
def test_noncanonical_input_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, target: str
):
    fixture = _fixture(tmp_path)
    _patch_sources(monkeypatch, fixture)
    path = fixture[f"{target}_path"]
    path.write_bytes(b" " + path.read_bytes())  # type: ignore[union-attr]
    with pytest.raises(b4.B4EvidenceError, match="canonical"):
        b4.assemble_b4_evidence(
            b3_gate_path=fixture["b3_gate_path"],  # type: ignore[arg-type]
            qualification_path=fixture["qualification_path"],  # type: ignore[arg-type]
            runtime_manifest_path=fixture["manifest_path"],  # type: ignore[arg-type]
            bot_repo=BOT_REPO,
            client_repo=CLIENT_REPO,
            game_repo=GAME_REPO,
            atlas_sha256=ATLAS,
        )


def test_synthetic_qualification_rejected_before_execution_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    fixture = _fixture(tmp_path)
    _patch_sources(monkeypatch, fixture)
    qualification = dict(fixture["qualification"])
    qualification.pop("evidence_sha256")
    qualification.update({"passed": False, "test_mode": True})
    qualification = b4._seal(qualification)
    _write_compact(fixture["qualification_path"], qualification)  # type: ignore[arg-type]

    def should_not_run(*unused):
        raise AssertionError("execution replay reached for synthetic evidence")

    monkeypatch.setattr(b4.barrier, "_validate_execution_evidence", should_not_run)
    with pytest.raises(b4.B4EvidenceError, match="real-network"):
        b4.assemble_b4_evidence(
            b3_gate_path=fixture["b3_gate_path"],  # type: ignore[arg-type]
            qualification_path=fixture["qualification_path"],  # type: ignore[arg-type]
            runtime_manifest_path=fixture["manifest_path"],  # type: ignore[arg-type]
            bot_repo=BOT_REPO,
            client_repo=CLIENT_REPO,
            game_repo=GAME_REPO,
            atlas_sha256=ATLAS,
        )


def test_runtime_manifest_atlas_mismatch_rejected(tmp_path: Path):
    fixture = _fixture(tmp_path)
    with pytest.raises(b4.B4EvidenceError, match="Atlas binding"):
        b4._validate_runtime_manifest(
            fixture["manifest"],  # type: ignore[arg-type]
            fixture["qualification"],  # type: ignore[arg-type]
            fixture["execution"],  # type: ignore[arg-type]
            "b" * 64,
        )


def test_old_telemetry_requires_exact_discard_proof(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    fixture = _fixture(tmp_path)
    _patch_sources(monkeypatch, fixture)
    stale_path = tmp_path / "old-telemetry.json"
    stale = b4._seal({
        "scenario": "old-telemetry",
        "structured_events": [{
            "event": "old_telemetry", "result": "accepted", "clock_regressed": "1"
        }],
    })
    _write_compact(stale_path, stale)
    execution = fixture["execution"]
    for index, record in enumerate(execution["scenario_evidence"]):  # type: ignore[index]
        if record["scenario"] == "old-telemetry":
            execution["scenario_evidence"][index] = _raw_record(  # type: ignore[index]
                stale_path, "old-telemetry", stale
            )
    qualification = dict(fixture["qualification"])
    qualification.pop("evidence_sha256")
    qualification["execution_evidence"] = execution
    qualification = b4._seal(qualification)
    _write_compact(fixture["qualification_path"], qualification)  # type: ignore[arg-type]
    with pytest.raises(b4.B4EvidenceError, match="old telemetry"):
        b4.assemble_b4_evidence(
            b3_gate_path=fixture["b3_gate_path"],  # type: ignore[arg-type]
            qualification_path=fixture["qualification_path"],  # type: ignore[arg-type]
            runtime_manifest_path=fixture["manifest_path"],  # type: ignore[arg-type]
            bot_repo=BOT_REPO,
            client_repo=CLIENT_REPO,
            game_repo=GAME_REPO,
            atlas_sha256=ATLAS,
        )


def test_dirty_source_repository_rejected(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "B4 Test"], cwd=repo, check=True)
    (repo / "tracked").write_text("clean\n")
    subprocess.run(["git", "add", "tracked"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "fixture"], cwd=repo, check=True)
    (repo / "untracked").write_text("dirty\n")
    with pytest.raises(b4.B4EvidenceError, match="dirty"):
        b4._git_identity(repo)


def test_exact_integration_sources_are_generation_two_and_abi_aligned():
    source = _source_contract()
    assert source["descriptor"]["total_dim"] == 298
    assert source["descriptor"]["client_wire_version"] == 8
    assert source["private_causal_facts"] == list(b4.PRIVATE_CAUSAL_FACTS)
    assert source["privilege_abi"]["public"]["packet_bytes"] == 1248
    assert source["privilege_abi"]["teacher"]["packet_bytes"] == 1224
    assert source["privilege_abi"]["yamagi_teacher_identity_present"] is False
    assert source["privilege_abi"]["qualification_teacher_enabled"] is False


def test_public_assembler_has_no_execution_validator_bypass():
    assert "execution_validator" not in inspect.signature(
        b4.assemble_b4_evidence
    ).parameters


def test_sealed_normal_run_teacher_violation_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    fixture = _fixture(tmp_path)
    _patch_sources(monkeypatch, fixture)
    baseline = copy.deepcopy(fixture["raw_values"]["baseline-cold-1"])
    baseline.pop("evidence_sha256")
    first = baseline["public_telemetry_audit"][0]
    first["datagrams_seen"] += 1
    first["teacher_packets_detected"] = 1
    _rebind_raw_scenario(fixture, tmp_path, "baseline-cold-1", b4._seal(baseline))
    with pytest.raises(b4.B4EvidenceError, match="teacher packet was detected"):
        b4.assemble_b4_evidence(
            b3_gate_path=fixture["b3_gate_path"],  # type: ignore[arg-type]
            qualification_path=fixture["qualification_path"],  # type: ignore[arg-type]
            runtime_manifest_path=fixture["manifest_path"],  # type: ignore[arg-type]
            bot_repo=BOT_REPO,
            client_repo=CLIENT_REPO,
            game_repo=GAME_REPO,
            atlas_sha256=ATLAS,
        )


def test_missing_normal_run_public_audit_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    fixture = _fixture(tmp_path)
    _patch_sources(monkeypatch, fixture)
    baseline = copy.deepcopy(fixture["raw_values"]["baseline-cold-1"])
    baseline.pop("evidence_sha256")
    baseline.pop("public_telemetry_audit")
    _rebind_raw_scenario(fixture, tmp_path, "baseline-cold-1", b4._seal(baseline))
    with pytest.raises(b4.B4EvidenceError, match="audit is absent"):
        b4.assemble_b4_evidence(
            b3_gate_path=fixture["b3_gate_path"],  # type: ignore[arg-type]
            qualification_path=fixture["qualification_path"],  # type: ignore[arg-type]
            runtime_manifest_path=fixture["manifest_path"],  # type: ignore[arg-type]
            bot_repo=BOT_REPO,
            client_repo=CLIENT_REPO,
            game_repo=GAME_REPO,
            atlas_sha256=ATLAS,
        )


def test_integration_rejects_forged_teacher_negative_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    unused, unused_aggregate, documents = _assemble(tmp_path, monkeypatch)
    wire = copy.deepcopy(documents["b4_wire_generation"])
    wire["public_privilege_separation"]["negative_probe"][
        "public_parser_result"
    ] = "ignored-as-malformed"
    with pytest.raises(integration.EvidenceError, match="negative injection proof"):
        integration._check_b4_wire_generation(wire, {"atlas_sha256": ATLAS})


def test_legacy_client_wire_source_rejected(tmp_path: Path):
    client = tmp_path / "client"
    source = CLIENT_REPO / "src/client/cl_ml_harness.c"
    destination = client / "src/client/cl_ml_harness.c"
    destination.parent.mkdir(parents=True)
    destination.write_text(
        source.read_text().replace(
            "#define ML_CLIENT_WIRE_VERSION 8u",
            "#define ML_CLIENT_WIRE_VERSION 4u",
            1,
        )
    )
    with pytest.raises(b4.B4EvidenceError, match="client wire ABI"):
        b4._source_contract(BOT_REPO, client, GAME_REPO)
