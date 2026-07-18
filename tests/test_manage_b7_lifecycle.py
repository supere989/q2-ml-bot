import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import tools.manage_b7_lifecycle as lifecycle_module
from tools.manage_b7_lifecycle import (
    B7LifecycleError,
    GUIDE_REFERENCE_SCHEMA,
    advance_stage,
    author_guide_on_reference,
    author_stage_one,
    evaluate_completed_stage,
    finalize_stage_seven,
    resume_stage,
)
from train.multires_train import CurriculumStage, create_curriculum_gate_evidence


RUNTIME = "a" * 64
ATLAS = "b" * 64
ATLAS_CATALOG = "f" * 64
LINEAGE = "c" * 64


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def seal(value):
    payload = dict(value)
    payload["evidence_sha256"] = hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return payload


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical(value))
    return path


def record(path):
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def stage_configuration(minimum=8):
    return {
        "minimum_accepted_transitions": minimum,
        "gate_predicates": [{
            "name": "echo-green", "path": "causal_metrics_window.transport.command_echo_match_rate",
            "operator": "ge", "threshold": 0.97,
        }],
        "maps": ["fixture"],
    }


def completed_season(path, stage, checkpoint):
    value = seal({
        "schema": "q2-multires-current-season-v1",
        "training_core_schema": "q2-multires-continuous-training-v1",
        "health": "training-active", "promotion_claim": False,
        "stage_id": stage.stage_id, "stage_name": stage.stage_name,
        "stage_configuration_sha256": stage.configuration_sha256,
        "runtime_manifest_sha256": RUNTIME, "atlas_sha256": ATLAS,
        "atlas_catalog_sha256": ATLAS_CATALOG,
        "lineage_root_sha256": LINEAGE,
        "counters": {"accepted_transitions": 8, "policy_updates": 2,
                     "optimizer_steps": 4, "next_policy_version": 2},
        "last_checkpoint": f"checkpoints/{checkpoint.name}",
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "checkpoint_manifest": {"training_step": 8, "runtime_manifest_sha256": RUNTIME,
                                "atlas_sha256": ATLAS,
                                "atlas_catalog_sha256": ATLAS_CATALOG,
                                "lineage_root_sha256": LINEAGE},
        "last_rollout_evidence": "evidence/rollouts/x.json",
        "last_update_evidence": "evidence/updates/x.json",
        "causal_metrics_window": {"accepted_transitions": 8, "transport": {
            "command_echo_match_rate": 1.0}},
        "network_metrics_window": {"network_client/transitions_accepted": 8},
    })
    return write_json(path, value)


def guide_on_fixture(tmp_path):
    runtime = (tmp_path / "guide-runtime").resolve()
    run = runtime / "public_network_multires_atlas_fresh_v1"
    for child in ("checkpoints", "tensorboard", "evidence/rollouts", "evidence/updates", "season"):
        (run / child).mkdir(parents=True)
    (runtime / "b7-admission").mkdir()
    configuration = {
        **stage_configuration(), "guide_mode": "on", "matched_seed": 77,
    }
    stage = CurriculumStage.create(
        6, configuration, predecessor_stage_sha256="e" * 64,
    )
    checkpoint = run / "checkpoints/guide-on.pt"
    checkpoint.write_bytes(b"guide-on-policy")
    season_path = completed_season(run / "season/current.json", stage, checkpoint)
    season = json.loads(season_path.read_text())
    unsigned = {key: value for key, value in season.items() if key != "evidence_sha256"}
    unsigned["causal_metrics_window"]["guide"] = {"task_success_rate": 0.8}
    write_json(season_path, seal(unsigned))
    primary = {
        "schema": "q2-multires-primary-training-runtime-v1",
        "runtime_manifest_sha256": RUNTIME,
        "atlas_catalog_sha256": ATLAS_CATALOG,
        "curriculum": {
            "stage_id": 6, "configuration": configuration,
            "configuration_sha256": stage.configuration_sha256,
            "predecessor_stage_sha256": "e" * 64,
            "predecessor_gate": record(season_path),
        },
        "checkpoint": {
            "mode": "same-lineage-stage-advance", "path": str(checkpoint),
            "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "lineage_root_sha256": LINEAGE, "current_season_report": None,
        },
    }
    primary_path = write_json(runtime / "primary-stage-06.json", primary)
    roots = {
        "current_run_root": run, "checkpoint_root": run / "checkpoints",
        "tensorboard_root": run / "tensorboard", "rollout_root": run / "evidence/rollouts",
        "update_root": run / "evidence/updates", "season_report_root": run / "season",
    }
    cold_path = write_json(runtime / "cold-stage-06.json", {
        "schema": "q2-multires-cold-start-v2",
        "lineage": {key: str(value) for key, value in roots.items()},
        "inputs": {},
    })
    write_json(runtime / "multires-service.json", {
        "schema": "q2-multires-service-v2",
        "retirement_cold_start": str(cold_path),
        "training_runtime": record(primary_path),
    })
    result = author_guide_on_reference(SimpleNamespace(
        runtime_root=runtime,
        task_success_metric_path="causal_metrics_window.guide.task_success_rate",
    ))
    return {
        "runtime": runtime, "run": run, "roots": roots, "stage": stage,
        "configuration": configuration, "checkpoint": checkpoint,
        "season": season_path, "primary": primary_path, "cold": cold_path,
        "reference": Path(result["guide_on_reference"]["path"]),
    }


def test_evaluator_derives_decision_from_frozen_stage_predicates(tmp_path):
    stage = CurriculumStage.create(1, stage_configuration())
    checkpoint = tmp_path / "run/checkpoints/checkpoint.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    season = completed_season(tmp_path / "run/season/current.json", stage, checkpoint)
    evaluation = evaluate_completed_stage(season, stage)
    assert evaluation["decision"] == "passed"
    assert evaluation["predicates"][0]["observed"] == 1.0
    assert evaluation["automatic_promotion"] is False

    bad = json.loads(season.read_text())
    unsigned = {key: value for key, value in bad.items() if key != "evidence_sha256"}
    unsigned["causal_metrics_window"]["transport"]["command_echo_match_rate"] = 0.5
    write_json(season, seal(unsigned))
    assert evaluate_completed_stage(season, stage)["decision"] == "failed"


def test_advance_archives_and_switches_one_selector_without_stage_report_mismatch(
    tmp_path, monkeypatch
):
    runtime = (tmp_path / "runtime").resolve()
    run = runtime / "public_network_multires_atlas_fresh_v1"
    for child in ("checkpoints", "tensorboard", "evidence/rollouts", "evidence/updates", "season"):
        (run / child).mkdir(parents=True)
    (runtime / "b7-admission").mkdir()
    checkpoint = run / "checkpoints/checkpoint.pt"
    checkpoint.write_bytes(b"trained")
    stage = CurriculumStage.create(1, stage_configuration())
    season = completed_season(run / "season/current.json", stage, checkpoint)
    primary = {
        "schema": "q2-multires-primary-training-runtime-v1",
        "curriculum": {
            "stage_id": 1, "configuration": stage_configuration(),
            "configuration_sha256": stage.configuration_sha256,
            "predecessor_stage_sha256": None, "predecessor_gate": None,
        },
        "checkpoint": {"mode": "fresh-step-zero", "path": str(checkpoint),
                       "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                       "lineage_root_sha256": LINEAGE, "current_season_report": None},
    }
    primary_path = write_json(runtime / "primary-stage-01.json", primary)
    roots = {
        "current_run_root": run, "checkpoint_root": run / "checkpoints",
        "tensorboard_root": run / "tensorboard", "rollout_root": run / "evidence/rollouts",
        "update_root": run / "evidence/updates", "season_report_root": run / "season",
    }
    cold = {"schema": "q2-multires-cold-start-v2", "lineage": {
        "run_tag": run.name, "initialization": "random", "training_step": 0,
        "optimizer_state": "fresh-empty", "normalization_state": "fresh-empty",
        **{key: str(value) for key, value in roots.items()},
    }, "inputs": {"trainer_checkpoint": record(checkpoint),
                   "trainer_current_season": None, "training_runtime": record(primary_path)}}
    cold_path = write_json(runtime / "cold-stage-01.json", cold)
    service = {"schema": "q2-multires-service-v2", "retirement_cold_start": str(cold_path),
               "training_runtime": record(primary_path)}
    service_path = write_json(runtime / "multires-service.json", service)
    next_config = write_json(tmp_path / "stage2.json", stage_configuration())
    import train.multires_service as service_module
    monkeypatch.setattr(service_module, "service_preflight", lambda _root: object())

    result = advance_stage(SimpleNamespace(
        runtime_root=runtime, next_stage_configuration=next_config,
    ))
    selected = json.loads(service_path.read_text())
    successor = json.loads(Path(selected["training_runtime"]["path"]).read_text())
    assert successor["curriculum"]["stage_id"] == 2
    assert successor["checkpoint"]["mode"] == "same-lineage-stage-advance"
    assert successor["checkpoint"]["current_season_report"] is None
    assert season.is_file()
    archives = list((run / "season").glob("stage-01-*.json"))
    assert len(archives) == 1 and archives[0].read_bytes() == season.read_bytes()
    assert archives[0].stat().st_ino != season.stat().st_ino
    gate = json.loads(Path(result["gate"]["path"]).read_text())
    assert gate["automatic_promotion"] is False
    assert gate["artifacts"]["completed_stage"] == record(archives[0])
    evaluation = json.loads(Path(result["evaluation"]["path"]).read_text())
    assert evaluation["completed_season"] == record(archives[0])


def test_stage_seven_enforces_matched_guide_off_contract(tmp_path):
    guide = guide_on_fixture(tmp_path)
    reference = guide["reference"]
    config = {
        "guide_mode": "off", "matched_seed": 77,
        "matched_seed_guide_on_reference": record(reference),
        "maximum_task_success_degradation_fraction": 0.15,
        "minimum_accepted_transitions": 8,
        "task_success_metric_path": "causal_metrics_window.guide.task_success_rate",
        "global_dropout_metric_path": "causal_metrics_window.guide.global_dropout_rate",
        "neutral_baseline_task_success": 0.1,
        "gate_predicates": [{"name": "guide-off-success", "path":
            "causal_metrics_window.guide.task_success_rate", "operator": "ge",
            "threshold": 0.68}],
    }
    predecessor = "d" * 64
    stage = CurriculumStage.create(7, config, predecessor_stage_sha256=predecessor)
    checkpoint = tmp_path / "run/checkpoints/checkpoint.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    season_path = completed_season(tmp_path / "run/season/current.json", stage, checkpoint)
    season = json.loads(season_path.read_text())
    unsigned = {key: value for key, value in season.items() if key != "evidence_sha256"}
    unsigned["causal_metrics_window"]["guide"] = {
        "task_success_rate": 0.7, "global_dropout_rate": 1.0,
    }
    write_json(season_path, seal(unsigned))
    result = evaluate_completed_stage(season_path, stage)
    assert result["decision"] == "passed"
    assert result["stage_specific"]["degradation_fraction"] == pytest.approx(0.125)

    reference_document = json.loads(reference.read_text())
    source_season = Path(reference_document["completed_season"]["path"])
    source_season.write_text("{}\n", encoding="utf-8")
    with pytest.raises(B7LifecycleError, match="digest differs"):
        evaluate_completed_stage(season_path, stage)


def test_finalize_stage_seven_publishes_terminal_gate_without_rebinding_selector(tmp_path):
    guide = guide_on_fixture(tmp_path)
    runtime = guide["runtime"]
    run = guide["run"]
    reference = guide["reference"]
    configuration = {
        "guide_mode": "off", "matched_seed": 77,
        "matched_seed_guide_on_reference": record(reference),
        "maximum_task_success_degradation_fraction": 0.15,
        "minimum_accepted_transitions": 8,
        "task_success_metric_path": "causal_metrics_window.guide.task_success_rate",
        "global_dropout_metric_path": "causal_metrics_window.guide.global_dropout_rate",
        "neutral_baseline_task_success": 0.1,
        "gate_predicates": [{
            "name": "guide-off-success",
            "path": "causal_metrics_window.guide.task_success_rate",
            "operator": "ge", "threshold": 0.68,
        }],
    }
    stage = CurriculumStage.create(
        7, configuration,
        predecessor_stage_sha256=guide["stage"].configuration_sha256,
    )
    checkpoint = guide["checkpoint"]
    source_season_sha256 = hashlib.sha256(guide["season"].read_bytes()).hexdigest()
    predecessor_evaluator = write_json(
        runtime / "b7-admission/stage-06-evaluator.json",
        evaluate_completed_stage(guide["season"], guide["stage"]),
    )
    source_season = json.loads(guide["season"].read_text())
    predecessor_gate = write_json(
        runtime / "b7-admission/stage-06-gate.json",
        create_curriculum_gate_evidence(
            guide["stage"], decision="passed",
            runtime_manifest_sha256=RUNTIME,
            atlas_catalog_sha256=ATLAS_CATALOG,
            lineage_root_sha256=LINEAGE,
            accepted_transitions=source_season["counters"]["accepted_transitions"],
            policy_updates=source_season["counters"]["policy_updates"],
            optimizer_steps=source_season["counters"]["optimizer_steps"],
            completed_stage={
                "path": str(guide["season"]), "sha256": source_season_sha256,
            },
            evaluator=record(predecessor_evaluator),
        ),
    )
    season_path = completed_season(run / "season/current.json", stage, checkpoint)
    season = json.loads(season_path.read_text())
    unsigned = {key: value for key, value in season.items() if key != "evidence_sha256"}
    unsigned["causal_metrics_window"]["guide"] = {
        "task_success_rate": 0.7, "global_dropout_rate": 1.0,
    }
    write_json(season_path, seal(unsigned))
    primary = {
        "schema": "q2-multires-primary-training-runtime-v1",
        "atlas_catalog_sha256": ATLAS_CATALOG,
        "curriculum": {
            "stage_id": 7, "configuration": configuration,
            "configuration_sha256": stage.configuration_sha256,
            "predecessor_stage_sha256": guide["stage"].configuration_sha256,
            "predecessor_gate": record(predecessor_gate),
        },
        "checkpoint": {
            "mode": "same-lineage-stage-advance", "path": str(checkpoint),
            "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "lineage_root_sha256": LINEAGE, "current_season_report": None,
        },
    }
    primary_path = write_json(runtime / "primary-stage-07.json", primary)
    roots = guide["roots"]
    cold_path = write_json(runtime / "cold-stage-07.json", {
        "schema": "q2-multires-cold-start-v2",
        "lineage": {key: str(value) for key, value in roots.items()},
    })
    service_path = write_json(runtime / "multires-service.json", {
        "schema": "q2-multires-service-v2",
        "retirement_cold_start": str(cold_path),
        "training_runtime": record(primary_path),
    })
    service_before = service_path.read_bytes()

    result = finalize_stage_seven(SimpleNamespace(runtime_root=runtime))

    assert service_path.read_bytes() == service_before
    gate = json.loads(Path(result["gate"]["path"]).read_text())
    assert gate["decision"] == "passed"
    assert gate["stage_id"] == 7
    assert gate["automatic_promotion"] is False
    lifecycle = json.loads(Path(result["lifecycle"]["path"]).read_text())
    assert lifecycle["state"] == "stage-complete"
    assert lifecycle["terminal_gate"] == record(Path(result["gate"]["path"]))


def test_author_stage_one_copies_only_b6_bound_step_zero(tmp_path, monkeypatch):
    runtime = (tmp_path / "runtime").resolve()
    runtime.mkdir()
    checkpoint = write_json(tmp_path / "evidence/fresh.pt", {"fresh": True})
    b5 = write_json(tmp_path / "evidence/B5.json", {
        "schema": "q2-multires-b5-gate-v1", "status": "green"})
    b6 = write_json(tmp_path / "evidence/B6.json", {
        "schema": "q2-multires-b6-wsl-g1-v1", "status": "green",
        "bindings": {
            "b5_gate": {"bytes": b5.stat().st_size, "sha256": record(b5)["sha256"]},
            "checkpoint": {
                "bytes": checkpoint.stat().st_size,
                "sha256": record(checkpoint)["sha256"],
            },
        }})
    stage_config = write_json(tmp_path / "stage1.json", stage_configuration())
    primary = write_json(tmp_path / "primary-template.json", {
        "schema": "q2-multires-primary-training-runtime-v1",
        "checkpoint": {"lineage_root_sha256": LINEAGE},
    })
    cold = write_json(tmp_path / "cold-template.json", {
        "schema": "q2-multires-cold-start-v2", "inputs": {}})
    service = write_json(tmp_path / "service-template.json", {
        "schema": "q2-multires-service-v2",
        "proof": {"checkpoint": str(checkpoint)},
    })
    monkeypatch.setattr(
        lifecycle_module, "_prevalidate_stage_one_authorities",
        lambda **_kwargs: None,
    )
    import train.multires_service as service_module
    monkeypatch.setattr(service_module, "service_preflight", lambda _root: object())
    result = author_stage_one(SimpleNamespace(
        b5_gate=b5, b6_gate=b6, stage_configuration=stage_config,
        primary_template=primary, cold_start_template=cold,
        service_template=service, runtime_root=runtime,
    ))
    selector = json.loads((runtime / "multires-service.json").read_text())
    primary_doc = json.loads(Path(selector["training_runtime"]["path"]).read_text())
    copied = Path(primary_doc["checkpoint"]["path"])
    assert copied.read_bytes() == checkpoint.read_bytes()
    assert primary_doc["curriculum"]["stage_id"] == 1
    assert result["lifecycle"]["sha256"]


def test_invalid_b5_b6_are_rejected_before_run_or_selector_creation(tmp_path):
    runtime = (tmp_path / "runtime").resolve()
    runtime.mkdir()
    checkpoint = write_json(tmp_path / "evidence/fresh.pt", {"fresh": True})
    b5 = write_json(tmp_path / "evidence/B5.json", {
        "schema": "q2-multires-b5-gate-v1", "status": "green",
    })
    b6 = write_json(tmp_path / "evidence/B6.json", {
        "schema": "q2-multires-b6-wsl-g1-v1", "status": "green",
        "bindings": {},
    })
    stage_config = write_json(tmp_path / "stage1.json", stage_configuration())
    primary = write_json(tmp_path / "primary.json", {
        "schema": "q2-multires-primary-training-runtime-v1",
        "checkpoint": {"lineage_root_sha256": LINEAGE},
    })
    cold = write_json(tmp_path / "cold.json", {
        "schema": "q2-multires-cold-start-v2", "inputs": {},
    })
    service = write_json(tmp_path / "service.json", {
        "schema": "q2-multires-service-v2",
        "proof": {"checkpoint": str(checkpoint)},
    })
    with pytest.raises(B7LifecycleError, match="B5 gate"):
        author_stage_one(SimpleNamespace(
            b5_gate=b5, b6_gate=b6, stage_configuration=stage_config,
            primary_template=primary, cold_start_template=cold,
            service_template=service, runtime_root=runtime,
        ))
    assert not (runtime / "multires-service.json").exists()
    assert not (runtime / "public_network_multires_atlas_fresh_v1").exists()


def test_resume_selects_only_current_season_checkpoint_and_preflights(tmp_path, monkeypatch):
    runtime = (tmp_path / "runtime").resolve()
    run = runtime / "public_network_multires_atlas_fresh_v1"
    for child in ("checkpoints", "tensorboard", "evidence/rollouts", "evidence/updates", "season"):
        (run / child).mkdir(parents=True)
    (runtime / "b7-admission").mkdir()
    selected = run / "checkpoints/checkpoint-000000000008.pt"
    selected.write_bytes(b"selected")
    (run / "checkpoints/checkpoint-999999999999.pt").write_bytes(b"later-not-selected")
    stage = CurriculumStage.create(1, stage_configuration())
    season = completed_season(run / "season/current.json", stage, selected)
    runtime_evidence = write_json(runtime / "runtime-evidence.json", {
        "atlas_sha256": ATLAS,
    })
    primary = write_json(runtime / "primary-stage-01.json", {
        "schema": "q2-multires-primary-training-runtime-v1",
        "runtime_manifest_sha256": RUNTIME,
        "atlas_catalog_sha256": ATLAS_CATALOG,
        "curriculum": {
            "stage_id": 1, "configuration": stage_configuration(),
            "configuration_sha256": stage.configuration_sha256,
            "predecessor_stage_sha256": None, "predecessor_gate": None,
        },
        "checkpoint": {
            "mode": "fresh-step-zero", "path": str(selected),
            "sha256": record(selected)["sha256"],
            "lineage_root_sha256": LINEAGE, "current_season_report": None,
        },
    })
    roots = {
        "current_run_root": run, "checkpoint_root": run / "checkpoints",
        "tensorboard_root": run / "tensorboard", "rollout_root": run / "evidence/rollouts",
        "update_root": run / "evidence/updates", "season_report_root": run / "season",
    }
    cold = write_json(runtime / "cold-stage-01.json", {
        "schema": "q2-multires-cold-start-v2",
        "lineage": {key: str(value) for key, value in roots.items()},
        "inputs": {
            "runtime_evidence": record(runtime_evidence),
            "trainer_checkpoint": record(selected),
            "trainer_current_season": None,
            "training_runtime": record(primary),
        },
    })
    service_path = write_json(runtime / "multires-service.json", {
        "schema": "q2-multires-service-v2",
        "retirement_cold_start": str(cold),
        "training_runtime": record(primary),
    })
    import train.multires_service as service_module
    calls = []
    monkeypatch.setattr(
        service_module, "service_preflight", lambda root: calls.append(Path(root)),
    )

    result = resume_stage(SimpleNamespace(runtime_root=runtime))

    assert calls == [runtime]
    selector = json.loads(service_path.read_text())
    resumed = json.loads(Path(selector["training_runtime"]["path"]).read_text())
    assert resumed["checkpoint"]["mode"] == "same-lineage-resume"
    assert Path(resumed["checkpoint"]["path"]) == selected
    assert resumed["checkpoint"]["current_season_report"] == record(season)
    assert result["lifecycle"]["sha256"]


def test_resume_preflight_failure_restores_prior_selector(tmp_path, monkeypatch):
    runtime = (tmp_path / "runtime").resolve()
    run = runtime / "public_network_multires_atlas_fresh_v1"
    for child in ("checkpoints", "tensorboard", "evidence/rollouts", "evidence/updates", "season"):
        (run / child).mkdir(parents=True)
    (runtime / "b7-admission").mkdir()
    checkpoint = run / "checkpoints/checkpoint.pt"
    checkpoint.write_bytes(b"selected")
    stage = CurriculumStage.create(1, stage_configuration())
    completed_season(run / "season/current.json", stage, checkpoint)
    runtime_evidence = write_json(runtime / "runtime-evidence.json", {
        "atlas_sha256": ATLAS,
    })
    primary = write_json(runtime / "primary.json", {
        "schema": "q2-multires-primary-training-runtime-v1",
        "runtime_manifest_sha256": RUNTIME,
        "atlas_catalog_sha256": ATLAS_CATALOG,
        "curriculum": {
            "stage_id": 1, "configuration": stage_configuration(),
            "configuration_sha256": stage.configuration_sha256,
            "predecessor_stage_sha256": None, "predecessor_gate": None,
        },
        "checkpoint": {
            "mode": "fresh-step-zero", "path": str(checkpoint),
            "sha256": record(checkpoint)["sha256"],
            "lineage_root_sha256": LINEAGE, "current_season_report": None,
        },
    })
    roots = {
        "current_run_root": run, "checkpoint_root": run / "checkpoints",
        "tensorboard_root": run / "tensorboard", "rollout_root": run / "evidence/rollouts",
        "update_root": run / "evidence/updates", "season_report_root": run / "season",
    }
    cold = write_json(runtime / "cold.json", {
        "schema": "q2-multires-cold-start-v2",
        "lineage": {key: str(value) for key, value in roots.items()},
        "inputs": {"runtime_evidence": record(runtime_evidence)},
    })
    selector = write_json(runtime / "multires-service.json", {
        "schema": "q2-multires-service-v2",
        "retirement_cold_start": str(cold),
        "training_runtime": record(primary),
    })
    before = selector.read_bytes()
    import train.multires_service as service_module

    def reject(_root):
        raise RuntimeError("synthetic-preflight-rejection")

    monkeypatch.setattr(service_module, "service_preflight", reject)
    with pytest.raises(B7LifecycleError, match="synthetic-preflight-rejection"):
        resume_stage(SimpleNamespace(runtime_root=runtime))
    assert selector.read_bytes() == before
