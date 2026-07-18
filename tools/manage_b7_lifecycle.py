#!/usr/bin/env python3
"""Author and advance the fail-closed B7 fresh-training lifecycle.

The service configuration is the sole mutable selector.  All stage, evaluator,
gate, cold-start, and primary-runtime documents are digest-named immutable
files.  Stage advancement publishes those files first and replaces the service
configuration last, so an interrupted operation cannot select a partial next
stage.  This tool never starts a trainer or promotes a policy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train.multires_train import (  # noqa: E402
    CURRICULUM_GATE_SCHEMA,
    SEASON_REPORT_SCHEMA,
    CurriculumStage,
    create_curriculum_gate_evidence,
    progress_from_season_report,
)


LIFECYCLE_SCHEMA = "q2-multires-b7-lifecycle-v1"
EVALUATION_SCHEMA = "q2-multires-b7-stage-evaluation-v1"
GUIDE_REFERENCE_SCHEMA = "q2-multires-b7-guide-on-reference-v1"
SERVICE_SCHEMA = "q2-multires-service-v2"
PRIMARY_SCHEMA = "q2-multires-primary-training-runtime-v1"
COLD_START_SCHEMA = "q2-multires-cold-start-v2"
B5_SCHEMA = "q2-multires-b5-gate-v1"
B6_SCHEMA = "q2-multires-b6-wsl-g1-v1"
RUN_TAG = "public_network_multires_atlas_fresh_v1"


class B7LifecycleError(RuntimeError):
    """Raised before a partial lifecycle selector can become active."""


def _canonical(value: object) -> bytes:
    try:
        return (
            json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise B7LifecycleError("lifecycle evidence is not canonical JSON") from error


def _seal_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise B7LifecycleError("lifecycle seal payload is not canonical JSON") from error


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _valid_digest(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value != "0" * 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _file(path: Path, label: str) -> Path:
    value = Path(path).expanduser()
    if not value.is_absolute() or value.is_symlink() or not value.is_file():
        raise B7LifecycleError(f"{label} must be an absolute regular non-symlink file")
    return value.resolve()


def _directory(path: Path, label: str, *, create: bool = False) -> Path:
    value = Path(path).expanduser()
    if not value.is_absolute() or value.is_symlink():
        raise B7LifecycleError(f"{label} must be an absolute non-symlink directory")
    if create:
        value.mkdir(parents=True, exist_ok=True)
    if not value.is_dir() or value.is_symlink():
        raise B7LifecycleError(f"{label} is not a directory")
    return value.resolve()


def _json(path: Path, label: str) -> dict[str, Any]:
    source = _file(path, label)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise B7LifecycleError(f"{label} is not valid JSON") from error
    if not isinstance(value, dict):
        raise B7LifecycleError(f"{label} must be a JSON object")
    return value


def _record(path: Path) -> dict[str, str]:
    source = _file(path, "artifact")
    return {"path": str(source), "sha256": _sha256(source)}


def _record_path(value: object, label: str) -> Path:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise B7LifecycleError(f"{label} artifact record is malformed")
    source = _file(Path(str(value["path"])), label)
    if not _valid_digest(value["sha256"]) or _sha256(source) != value["sha256"]:
        raise B7LifecycleError(f"{label} artifact digest differs")
    return source


def _b6_bound_path(value: object, path: Path, label: str) -> Path:
    source = _file(path, label)
    if (
        not isinstance(value, Mapping)
        or set(value) != {"bytes", "sha256"}
        or value.get("bytes") != source.stat().st_size
        or value.get("sha256") != _sha256(source)
    ):
        raise B7LifecycleError(f"{label} differs from the B6 file binding")
    return source


def _seal(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(value)
    if "evidence_sha256" in payload:
        raise B7LifecycleError("unsealed lifecycle evidence contains a seal")
    digest = hashlib.sha256(_seal_bytes(payload)).hexdigest()
    return {**payload, "evidence_sha256": digest}


def _validate_seal(value: Mapping[str, Any], schema: str, label: str) -> None:
    payload = dict(value)
    digest = payload.pop("evidence_sha256", None)
    if (
        payload.get("schema") != schema
        or not _valid_digest(digest)
        or hashlib.sha256(_seal_bytes(payload)).hexdigest() != digest
    ):
        raise B7LifecycleError(f"{label} schema/seal differs")


def _atomic_json(path: Path, value: Mapping[str, Any], *, replace: bool) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        raise B7LifecycleError(f"JSON destination is a symlink: {destination}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(_canonical(value))
            stream.flush()
            os.fsync(stream.fileno())
        if not replace and destination.exists():
            if destination.is_file() and destination.read_bytes() == _canonical(value):
                return
            raise B7LifecycleError(f"immutable JSON already exists: {destination}")
        os.replace(temporary, destination)
        os.chmod(destination, 0o600)
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def _publish(directory: Path, prefix: str, value: Mapping[str, Any]) -> Path:
    digest = hashlib.sha256(_canonical(value)).hexdigest()
    path = directory / f"{prefix}-{digest[:20]}.json"
    _atomic_json(path, value, replace=False)
    return path.resolve()


def _metric(document: Mapping[str, Any], path: str, label: str) -> Any:
    if not isinstance(path, str) or not path or path.startswith(".") or path.endswith("."):
        raise B7LifecycleError(f"{label} metric path is invalid")
    value: Any = document
    for component in path.split("."):
        if not isinstance(value, Mapping) or component not in value:
            raise B7LifecycleError(f"{label} metric {path!r} is absent")
        value = value[component]
    return value


def _validate_guide_reference(path: Path) -> tuple[dict[str, Any], Path]:
    source = _file(path, "stage-7 guide-on reference")
    reference = _json(source, "stage-7 guide-on reference")
    _validate_seal(reference, GUIDE_REFERENCE_SCHEMA, "stage-7 guide-on reference")
    if set(reference) != {
        "schema", "matched_seed", "task_success_metric_path", "task_success_rate",
        "stage_id", "stage_configuration_sha256", "accepted_transitions",
        "policy_version", "runtime_manifest_sha256", "atlas_sha256",
        "atlas_catalog_sha256",
        "lineage_root_sha256", "source_primary_runtime", "completed_season",
        "checkpoint", "evidence_sha256",
    }:
        raise B7LifecycleError("stage-7 guide-on reference fields differ")
    primary_path = _record_path(
        reference["source_primary_runtime"], "guide-on primary runtime"
    )
    primary = _json(primary_path, "guide-on primary runtime")
    curriculum = primary.get("curriculum")
    if primary.get("schema") != PRIMARY_SCHEMA or not isinstance(curriculum, Mapping):
        raise B7LifecycleError("guide-on primary runtime identity differs")
    source_stage = CurriculumStage.create(
        curriculum.get("stage_id"), curriculum.get("configuration", {}),
        predecessor_stage_sha256=curriculum.get("predecessor_stage_sha256"),
    )
    source_configuration = json.loads(source_stage.configuration_json)["configuration"]
    if (
        source_stage.stage_id != 6
        or source_stage.configuration_sha256
        != curriculum.get("configuration_sha256")
        or source_configuration.get("guide_mode") != "on"
        or source_configuration.get("matched_seed") != reference.get("matched_seed")
        or reference.get("stage_id") != 6
        or reference.get("stage_configuration_sha256")
        != source_stage.configuration_sha256
    ):
        raise B7LifecycleError("guide-on source stage/configuration differs")
    season_path = _record_path(
        reference["completed_season"], "guide-on completed season"
    )
    season = _json(season_path, "guide-on completed season")
    _validate_seal(season, SEASON_REPORT_SCHEMA, "guide-on completed season")
    checkpoint = _record_path(reference["checkpoint"], "guide-on checkpoint")
    checkpoint_relative = Path(str(season.get("last_checkpoint")))
    reported_checkpoint = (checkpoint.parent.parent / checkpoint_relative).resolve()
    counters = season.get("counters")
    manifest = season.get("checkpoint_manifest")
    metric_path = reference.get("task_success_metric_path")
    task_success = _metric(season, metric_path, "guide-on task success")
    scalar_values = (task_success, reference.get("task_success_rate"))
    if (
        season.get("stage_id") != 6
        or season.get("stage_configuration_sha256")
        != source_stage.configuration_sha256
        or season.get("promotion_claim") is not False
        or season.get("runtime_manifest_sha256")
        != reference.get("runtime_manifest_sha256")
        or season.get("atlas_sha256") != reference.get("atlas_sha256")
        or season.get("atlas_catalog_sha256")
        != reference.get("atlas_catalog_sha256")
        or season.get("lineage_root_sha256")
        != reference.get("lineage_root_sha256")
        or primary.get("runtime_manifest_sha256")
        != reference.get("runtime_manifest_sha256")
        or checkpoint_relative.is_absolute()
        or ".." in checkpoint_relative.parts
        or len(checkpoint_relative.parts) != 2
        or checkpoint_relative.parts[0] != "checkpoints"
        or reported_checkpoint != checkpoint
        or checkpoint.parent.name != "checkpoints"
        or season.get("checkpoint_sha256") != _sha256(checkpoint)
        or not isinstance(counters, Mapping)
        or counters.get("accepted_transitions")
        != reference.get("accepted_transitions")
        or counters.get("next_policy_version") != reference.get("policy_version")
        or not isinstance(manifest, Mapping)
        or manifest.get("training_step") != reference.get("accepted_transitions")
        or manifest.get("runtime_manifest_sha256")
        != reference.get("runtime_manifest_sha256")
        or manifest.get("atlas_sha256") != reference.get("atlas_sha256")
        or manifest.get("atlas_catalog_sha256")
        != reference.get("atlas_catalog_sha256")
        or manifest.get("lineage_root_sha256")
        != reference.get("lineage_root_sha256")
        or any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in scalar_values
        )
        or float(task_success) <= 0.0
        or float(task_success) != float(reference["task_success_rate"])
        or type(reference.get("matched_seed")) is not int
        or type(reference.get("accepted_transitions")) is not int
        or reference["accepted_transitions"] < 1
        or type(reference.get("policy_version")) is not int
        or reference["policy_version"] < 1
        or not all(
            _valid_digest(reference.get(name))
            for name in (
                "stage_configuration_sha256", "runtime_manifest_sha256",
                "atlas_sha256", "atlas_catalog_sha256", "lineage_root_sha256",
            )
        )
    ):
        raise B7LifecycleError("guide-on source season/checkpoint/policy differs")
    return reference, season_path


def evaluate_completed_stage(
    season_path: Path, stage: CurriculumStage
) -> dict[str, Any]:
    """Derive a sealed evaluator result only from the immutable stage contract."""
    season_source = _file(season_path, "completed season")
    season = _json(season_source, "completed season")
    _validate_seal(season, SEASON_REPORT_SCHEMA, "completed season")
    decoded = json.loads(stage.configuration_json)
    configuration = decoded["configuration"]
    if (
        season.get("stage_id") != stage.stage_id
        or season.get("stage_configuration_sha256") != stage.configuration_sha256
        or season.get("promotion_claim") is not False
    ):
        raise B7LifecycleError("completed season differs from the selected stage")
    counters = season.get("counters")
    if not isinstance(counters, Mapping):
        raise B7LifecycleError("completed season counters are absent")
    minimum = configuration.get("minimum_accepted_transitions")
    if type(minimum) is not int or minimum < 1:
        raise B7LifecycleError("stage minimum_accepted_transitions is invalid")
    results: list[dict[str, Any]] = []
    predicates = configuration.get("gate_predicates")
    if not isinstance(predicates, list) or not predicates:
        raise B7LifecycleError("stage has no frozen gate predicates")
    operators = {
        "eq": lambda found, wanted: found == wanted,
        "ge": lambda found, wanted: found >= wanted,
        "gt": lambda found, wanted: found > wanted,
        "le": lambda found, wanted: found <= wanted,
        "lt": lambda found, wanted: found < wanted,
    }
    for index, predicate in enumerate(predicates):
        if not isinstance(predicate, Mapping) or set(predicate) != {
            "name", "path", "operator", "threshold"
        }:
            raise B7LifecycleError(f"gate predicate[{index}] is malformed")
        operator = predicate["operator"]
        found = _metric(season, str(predicate["path"]), f"predicate[{index}]")
        wanted = predicate["threshold"]
        if operator not in operators:
            raise B7LifecycleError(f"gate predicate[{index}] operator is invalid")
        if isinstance(found, float) and not math.isfinite(found):
            raise B7LifecycleError(f"gate predicate[{index}] is non-finite")
        try:
            passed = bool(operators[operator](found, wanted))
        except TypeError as error:
            raise B7LifecycleError(
                f"gate predicate[{index}] values are not comparable"
            ) from error
        results.append({
            "name": predicate["name"], "path": predicate["path"],
            "operator": operator, "threshold": wanted, "observed": found,
            "passed": passed,
        })

    stage_specific: dict[str, Any] | None = None
    if stage.stage_id == 7:
        reference = configuration["matched_seed_guide_on_reference"]
        reference_path = _record_path(reference, "stage-7 guide-on reference")
        reference_doc, reference_season = _validate_guide_reference(reference_path)
        guide_off_success = _metric(
            season, str(configuration.get("task_success_metric_path")),
            "stage-7 guide-off task success",
        )
        global_dropout = _metric(
            season, str(configuration.get("global_dropout_metric_path")),
            "stage-7 global guide dropout",
        )
        guide_on_success = reference_doc.get("task_success_rate")
        if any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in (guide_off_success, global_dropout, guide_on_success)
        ) or float(guide_on_success) <= 0.0:
            raise B7LifecycleError("stage-7 guide dependence metrics are invalid")
        degradation = max(
            0.0, (float(guide_on_success) - float(guide_off_success))
            / float(guide_on_success),
        )
        maximum = float(configuration["maximum_task_success_degradation_fraction"])
        guide_pass = (
            reference_doc.get("matched_seed") == configuration.get("matched_seed")
            and reference_doc.get("runtime_manifest_sha256")
            == season.get("runtime_manifest_sha256")
            and reference_doc.get("atlas_catalog_sha256")
            == season.get("atlas_catalog_sha256")
            and reference_doc.get("lineage_root_sha256")
            == season.get("lineage_root_sha256")
            and float(global_dropout) > 0.0
            and degradation <= maximum
            and float(guide_off_success)
            > float(configuration.get("neutral_baseline_task_success", 0.0))
        )
        stage_specific = {
            "guide_on_reference": _record(reference_path),
            "guide_on_completed_season": _record(reference_season),
            "guide_on_checkpoint": dict(reference_doc["checkpoint"]),
            "guide_on_policy_version": reference_doc["policy_version"],
            "matched_seed": configuration.get("matched_seed"),
            "guide_on_task_success": float(guide_on_success),
            "guide_off_task_success": float(guide_off_success),
            "degradation_fraction": degradation,
            "maximum_degradation_fraction": maximum,
            "global_dropout_rate": float(global_dropout),
            "passed": guide_pass,
        }

    accepted = counters.get("accepted_transitions")
    decision = "passed" if (
        type(accepted) is int and accepted >= minimum
        and all(result["passed"] for result in results)
        and (stage_specific is None or stage_specific["passed"])
    ) else "failed"
    return _seal({
        "schema": EVALUATION_SCHEMA,
        "decision": decision,
        "stage_id": stage.stage_id,
        "stage_name": stage.stage_name,
        "stage_configuration_sha256": stage.configuration_sha256,
        "runtime_manifest_sha256": season.get("runtime_manifest_sha256"),
        "atlas_sha256": season.get("atlas_sha256"),
        "atlas_catalog_sha256": season.get("atlas_catalog_sha256"),
        "lineage_root_sha256": season.get("lineage_root_sha256"),
        "completed_season": _record(season_source),
        "minimum_accepted_transitions": minimum,
        "predicates": results,
        "stage_specific": stage_specific,
        "automatic_promotion": False,
    })


def _copy_exclusive(source: Path, destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        raise B7LifecycleError(f"refusing to replace fresh artifact: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with source.open("rb") as incoming, os.fdopen(descriptor, "wb") as outgoing:
            shutil.copyfileobj(incoming, outgoing)
            outgoing.flush()
            os.fsync(outgoing.fileno())
    except Exception:
        destination.unlink(missing_ok=True)
        raise


def _stage_mapping(stage: CurriculumStage, gate: dict[str, str] | None) -> dict[str, Any]:
    configuration = json.loads(stage.configuration_json)["configuration"]
    return {
        "stage_id": stage.stage_id,
        "configuration": configuration,
        "configuration_sha256": stage.configuration_sha256,
        "predecessor_stage_sha256": stage.predecessor_stage_sha256,
        "predecessor_gate": gate,
    }


def _require_selector_idle(runtime_root: Path) -> None:
    for name in ("multires-service.lock", "multires-service-state.json"):
        candidate = runtime_root / name
        if candidate.exists() or candidate.is_symlink():
            raise B7LifecycleError(
                f"service must be stopped and reconciled before lifecycle mutation: {name}"
            )


def _prevalidate_stage_one_authorities(
    *, runtime_root: Path, b5_path: Path, b5: Mapping[str, Any],
    b6_path: Path, b6: Mapping[str, Any], service: Mapping[str, Any],
    cold: Mapping[str, Any],
) -> None:
    """Validate the complete B5/B6/integration boundary before creating a run."""
    try:
        from tools.assemble_b5_gate import (
            B5_PREDICATE_KEYS,
            gate_sha256 as b5_gate_sha256,
        )
        from tools.assemble_b6_wsl_g1_campaign import validate_campaign
        from train.multires_service import _verify_integration_admission

        predicates = b5.get("predicates")
        if (
            b5.get("schema") != B5_SCHEMA
            or b5.get("tool") != "assemble_b5_gate"
            or b5.get("status") != "green"
            or b5.get("green") is not True
            or b5.get("failures") != []
            or not _valid_digest(b5.get("gate_sha256"))
            or b5.get("gate_sha256") != b5_gate_sha256(b5)
            or not isinstance(predicates, Mapping)
            or set(predicates) != set(B5_PREDICATE_KEYS)
            or any(value is not True for value in predicates.values())
        ):
            raise B7LifecycleError("B5 gate identity/seal/predicates differ")
        validate_campaign(b6)
        bindings = b6.get("bindings")
        if not isinstance(bindings, Mapping):
            raise B7LifecycleError("B6 immutable bindings are absent")
        _b6_bound_path(bindings.get("b5_gate"), b5_path, "B6-bound B5 gate")

        if set(service) != {
            "schema", "retirement_manifest_sha256", "runtime_manifest_sha256",
            "retirement_manifest", "retirement_cold_start", "operational_roots",
            "service_selectors", "modules", "proof", "training_runtime",
            "integration_admission", "evidence_dir", "log_path", "tensorboard",
        } or service.get("schema") != SERVICE_SCHEMA:
            raise B7LifecycleError("service template fields/schema differ")
        cold_inputs = cold.get("inputs")
        if not isinstance(cold_inputs, Mapping):
            raise B7LifecycleError("cold-start-v2 template inputs are absent")
        retirement_manifest = _file(
            Path(str(service["retirement_manifest"])), "retirement manifest"
        )
        envelope_path, _report_path, _report = _verify_integration_admission(
            runtime_root=runtime_root,
            config=service,
            cold_inputs=cold_inputs,
            retirement_manifest=retirement_manifest,
        )
        envelope = _json(envelope_path, "integration evidence envelope")
        evidence = envelope.get("evidence")
        if not isinstance(evidence, Mapping):
            raise B7LifecycleError("integration evidence inventory is absent")
        integrated_b6 = Path(str(evidence.get("wsl_b6_campaign", "")))
        if not integrated_b6.is_absolute():
            integrated_b6 = envelope_path.parent / integrated_b6
        if _file(integrated_b6, "integrated B6 campaign") != b6_path:
            raise B7LifecycleError(
                "service integration envelope does not bind the supplied B6 campaign"
            )
    except B7LifecycleError:
        raise
    except (KeyError, TypeError, ValueError, OSError, RuntimeError) as error:
        raise B7LifecycleError(
            f"B5/B6/integration prevalidation failed: {error}"
        ) from error


def author_guide_on_reference(args: argparse.Namespace) -> dict[str, Any]:
    """Derive the only admissible stage-7 guide-on reference from stage 6."""
    runtime_root = _directory(args.runtime_root, "runtime root")
    _require_selector_idle(runtime_root)
    service_path = _file(runtime_root / "multires-service.json", "service selector")
    service = _json(service_path, "service selector")
    if service.get("schema") != SERVICE_SCHEMA:
        raise B7LifecycleError("service selector is not v2")
    primary_path = _record_path(service.get("training_runtime"), "primary runtime")
    primary = _json(primary_path, "primary runtime")
    cold_path = _file(Path(str(service.get("retirement_cold_start"))), "cold start")
    cold = _json(cold_path, "cold start")
    curriculum = primary.get("curriculum")
    lineage = cold.get("lineage")
    if (
        primary.get("schema") != PRIMARY_SCHEMA
        or cold.get("schema") != COLD_START_SCHEMA
        or not isinstance(curriculum, Mapping)
        or not isinstance(lineage, Mapping)
    ):
        raise B7LifecycleError("guide-on source primary/cold-start differs")
    stage = CurriculumStage.create(
        curriculum.get("stage_id"), curriculum.get("configuration", {}),
        predecessor_stage_sha256=curriculum.get("predecessor_stage_sha256"),
    )
    configuration = json.loads(stage.configuration_json)["configuration"]
    if (
        stage.stage_id != 6
        or stage.configuration_sha256 != curriculum.get("configuration_sha256")
        or configuration.get("guide_mode") != "on"
        or type(configuration.get("matched_seed")) is not int
    ):
        raise B7LifecycleError(
            "guide-on reference requires the active matched-seed stage-6 guide-on runtime"
        )
    season_root = _directory(
        Path(str(lineage["season_report_root"])), "season report root"
    )
    run_root = _directory(Path(str(lineage["current_run_root"])), "current run root")
    checkpoint_root = _directory(
        Path(str(lineage["checkpoint_root"])), "checkpoint root"
    )
    if (
        season_root != (run_root / "season").resolve()
        or checkpoint_root != (run_root / "checkpoints").resolve()
    ):
        raise B7LifecycleError("guide-on run-root layout differs")
    season_path = _file(season_root / "current.json", "guide-on current season")
    season = _json(season_path, "guide-on current season")
    evaluation = evaluate_completed_stage(season_path, stage)
    if evaluation["decision"] != "passed":
        raise B7LifecycleError("guide-on stage 6 has not passed its frozen predicates")
    checkpoint = _file(
        run_root / str(season.get("last_checkpoint")), "guide-on checkpoint"
    )
    if (
        checkpoint.parent != checkpoint_root
        or season.get("checkpoint_sha256") != _sha256(checkpoint)
    ):
        raise B7LifecycleError("guide-on checkpoint path/digest differs")
    metric_path = args.task_success_metric_path
    task_success = _metric(season, metric_path, "guide-on task success")
    counters = season.get("counters")
    if (
        isinstance(task_success, bool)
        or not isinstance(task_success, (int, float))
        or not math.isfinite(float(task_success))
        or float(task_success) <= 0.0
        or not isinstance(counters, Mapping)
        or type(counters.get("accepted_transitions")) is not int
        or counters["accepted_transitions"] < 1
        or type(counters.get("next_policy_version")) is not int
        or counters["next_policy_version"] < 1
    ):
        raise B7LifecycleError("guide-on task metric/counters are invalid")
    admission_root = _directory(runtime_root / "b7-admission", "B7 admission root")
    immutable_season = _publish(admission_root, "guide-on-stage-06-season", season)
    reference = _seal({
        "schema": GUIDE_REFERENCE_SCHEMA,
        "matched_seed": configuration["matched_seed"],
        "task_success_metric_path": metric_path,
        "task_success_rate": float(task_success),
        "stage_id": 6,
        "stage_configuration_sha256": stage.configuration_sha256,
        "accepted_transitions": counters["accepted_transitions"],
        "policy_version": counters["next_policy_version"],
        "runtime_manifest_sha256": season["runtime_manifest_sha256"],
        "atlas_sha256": season["atlas_sha256"],
        "atlas_catalog_sha256": season["atlas_catalog_sha256"],
        "lineage_root_sha256": season["lineage_root_sha256"],
        "source_primary_runtime": _record(primary_path),
        "completed_season": _record(immutable_season),
        "checkpoint": _record(checkpoint),
    })
    reference_path = _publish(admission_root, "guide-on-reference", reference)
    _validate_guide_reference(reference_path)
    return {"guide_on_reference": _record(reference_path)}


def author_stage_one(args: argparse.Namespace) -> dict[str, Any]:
    """Create a unique fresh run from the exact B5/B6 step-zero authority."""
    b5_path = _file(args.b5_gate, "B5 gate")
    b6_path = _file(args.b6_gate, "B6 gate")
    b5 = _json(b5_path, "B5 gate")
    b6 = _json(b6_path, "B6 gate")
    runtime_root = _directory(args.runtime_root, "runtime root", create=True)
    service_path = runtime_root / "multires-service.json"
    if service_path.exists() or service_path.is_symlink():
        raise B7LifecycleError("fresh lifecycle refuses an occupied service selector")
    run_root = runtime_root / RUN_TAG
    if run_root.exists() or run_root.is_symlink():
        raise B7LifecycleError("fresh run tag is already occupied")
    _require_selector_idle(runtime_root)
    configuration = _json(args.stage_configuration, "stage-1 configuration")
    stage = CurriculumStage.create(1, configuration)
    primary = _json(args.primary_template, "primary runtime template")
    cold = _json(args.cold_start_template, "cold-start-v2 template")
    service = _json(args.service_template, "service-v2 template")
    if primary.get("schema") != PRIMARY_SCHEMA:
        raise B7LifecycleError("primary runtime template schema differs")
    if cold.get("schema") != COLD_START_SCHEMA:
        raise B7LifecycleError("final cold-start template must be v2")
    if service.get("schema") != SERVICE_SCHEMA:
        raise B7LifecycleError("service template must be q2-multires-service-v2")
    _prevalidate_stage_one_authorities(
        runtime_root=runtime_root, b5_path=b5_path, b5=b5,
        b6_path=b6_path, b6=b6, service=service, cold=cold,
    )
    bindings = b6["bindings"]
    proof = service.get("proof")
    if not isinstance(proof, Mapping):
        raise B7LifecycleError("service template proof inputs are absent")
    checkpoint_source = _b6_bound_path(
        bindings.get("checkpoint"), Path(str(proof.get("checkpoint"))),
        "B6-bound step-zero checkpoint",
    )

    admission_root_path = runtime_root / "b7-admission"
    admission_root_preexisted = admission_root_path.exists()
    preexisting_admission = (
        set(admission_root_path.iterdir()) if admission_root_preexisted else set()
    )
    roots = {
        "current_run_root": run_root,
        "checkpoint_root": run_root / "checkpoints",
        "tensorboard_root": run_root / "tensorboard",
        "rollout_root": run_root / "evidence/rollouts",
        "update_root": run_root / "evidence/updates",
        "season_report_root": run_root / "season",
    }
    for path in roots.values():
        path.mkdir(parents=True, exist_ok=False)
    admission_root = _directory(runtime_root / "b7-admission", "B7 admission root", create=True)
    checkpoint = roots["checkpoint_root"] / "fresh-step-zero.pt"
    _copy_exclusive(checkpoint_source, checkpoint)
    if _sha256(checkpoint) != _sha256(checkpoint_source):
        raise B7LifecycleError("copied step-zero checkpoint digest differs")

    primary["curriculum"] = _stage_mapping(stage, None)
    checkpoint_selector = primary.get("checkpoint")
    if not isinstance(checkpoint_selector, Mapping):
        raise B7LifecycleError("primary runtime template checkpoint is absent")
    lineage = checkpoint_selector.get("lineage_root_sha256")
    if not _valid_digest(lineage):
        raise B7LifecycleError("primary template fresh lineage digest is invalid")
    primary["checkpoint"] = {
        "mode": "fresh-step-zero", "path": str(checkpoint),
        "sha256": _sha256(checkpoint), "lineage_root_sha256": lineage,
        "current_season_report": None,
    }
    primary_path = _publish(admission_root, "primary-stage-01", primary)

    cold["lineage"] = {
        "run_tag": RUN_TAG, "initialization": "random", "training_step": 0,
        "optimizer_state": "fresh-empty", "normalization_state": "fresh-empty",
        **{name: str(path) for name, path in roots.items()},
    }
    inputs = cold.get("inputs")
    if not isinstance(inputs, dict):
        raise B7LifecycleError("cold-start-v2 template inputs are absent")
    # ``checkpoint`` remains the exact B6-bound proof source.  The distinct
    # trainer selector is an exact byte copy inside the new run root; service
    # admission independently rebinds the source to B6 and requires equality.
    inputs["checkpoint"] = _record(checkpoint_source)
    inputs["trainer_checkpoint"] = _record(checkpoint)
    inputs["trainer_current_season"] = None
    inputs["training_runtime"] = _record(primary_path)
    cold_path = _publish(admission_root, "cold-start-stage-01", cold)

    service["retirement_cold_start"] = str(cold_path)
    service["training_runtime"] = _record(primary_path)
    _atomic_json(service_path, service, replace=False)
    try:
        from train.multires_service import service_preflight

        service_preflight(runtime_root)
    except Exception as error:
        if service_path.is_file() and service_path.read_bytes() == _canonical(service):
            service_path.unlink()
        shutil.rmtree(run_root, ignore_errors=True)
        if admission_root_path.is_dir() and not admission_root_path.is_symlink():
            for candidate in tuple(admission_root_path.iterdir()):
                if candidate not in preexisting_admission and candidate.is_file():
                    candidate.unlink()
            if not admission_root_preexisted and not any(admission_root_path.iterdir()):
                admission_root_path.rmdir()
        raise B7LifecycleError(
            f"authored stage-1 selector failed complete service preflight: {error}"
        ) from error
    lifecycle = _seal({
        "schema": LIFECYCLE_SCHEMA, "state": "stage-active",
        "stage_id": 1, "stage_configuration_sha256": stage.configuration_sha256,
        "automatic_promotion": False, "b5_gate": _record(b5_path),
        "b6_gate": _record(b6_path), "source_checkpoint": _record(checkpoint_source),
        "selected_checkpoint": _record(checkpoint),
        "primary_runtime": _record(primary_path), "cold_start": _record(cold_path),
        "service_selector": _record(service_path),
    })
    lifecycle_path = _publish(admission_root, "lifecycle-stage-01", lifecycle)
    return {"lifecycle": _record(lifecycle_path), "service_selector": _record(service_path)}


def advance_stage(args: argparse.Namespace) -> dict[str, Any]:
    """Evaluate one completed stage and atomically select its successor."""
    runtime_root = _directory(args.runtime_root, "runtime root")
    _require_selector_idle(runtime_root)
    service_path = _file(runtime_root / "multires-service.json", "service selector")
    service_bytes = service_path.read_bytes()
    service = _json(service_path, "service selector")
    if service.get("schema") != SERVICE_SCHEMA:
        raise B7LifecycleError("service selector is not v2")
    primary_path = _record_path(service.get("training_runtime"), "primary runtime")
    primary = _json(primary_path, "primary runtime")
    cold_path = _file(Path(str(service.get("retirement_cold_start"))), "cold start")
    cold = _json(cold_path, "cold start")
    if primary.get("schema") != PRIMARY_SCHEMA or cold.get("schema") != COLD_START_SCHEMA:
        raise B7LifecycleError("active primary/cold-start schemas differ")
    curriculum = primary.get("curriculum")
    if not isinstance(curriculum, Mapping):
        raise B7LifecycleError("active curriculum is absent")
    current_stage = CurriculumStage.create(
        curriculum.get("stage_id"), curriculum.get("configuration", {}),
        predecessor_stage_sha256=curriculum.get("predecessor_stage_sha256"),
    )
    if current_stage.configuration_sha256 != curriculum.get("configuration_sha256"):
        raise B7LifecycleError("active curriculum configuration digest differs")
    if current_stage.stage_id >= 7:
        raise B7LifecycleError("stage 7 is terminal here; automatic promotion is forbidden")

    lineage = cold.get("lineage")
    if not isinstance(lineage, Mapping):
        raise B7LifecycleError("cold-start lineage roots are absent")
    season = _file(
        Path(str(lineage["season_report_root"])) / "current.json",
        "completed current season",
    )
    admission_root = _directory(runtime_root / "b7-admission", "B7 admission root")
    archive = Path(str(lineage["season_report_root"])) / (
        f"stage-{current_stage.stage_id:02d}-{_sha256(season)[:20]}.json"
    )
    if archive.exists():
        if archive.is_symlink() or _sha256(archive) != _sha256(season):
            raise B7LifecycleError("completed-stage archive collision differs")
    else:
        _copy_exclusive(season, archive)
    evaluation = evaluate_completed_stage(archive, current_stage)
    evaluation_path = _publish(
        admission_root, f"evaluation-stage-{current_stage.stage_id:02d}", evaluation
    )
    if evaluation["decision"] != "passed":
        raise B7LifecycleError(
            f"stage {current_stage.stage_id} evaluation failed; no gate or selector changed"
        )
    season_doc = _json(season, "completed current season")
    counters = season_doc["counters"]
    gate = create_curriculum_gate_evidence(
        current_stage, decision="passed",
        runtime_manifest_sha256=season_doc["runtime_manifest_sha256"],
        atlas_catalog_sha256=season_doc["atlas_catalog_sha256"],
        lineage_root_sha256=season_doc["lineage_root_sha256"],
        accepted_transitions=counters["accepted_transitions"],
        policy_updates=counters["policy_updates"],
        optimizer_steps=counters["optimizer_steps"],
        completed_stage=_record(archive), evaluator=_record(evaluation_path),
    )
    gate_path = _publish(
        admission_root, f"gate-stage-{current_stage.stage_id:02d}", gate
    )

    next_configuration = _json(args.next_stage_configuration, "next-stage configuration")
    next_stage = CurriculumStage.create(
        current_stage.stage_id + 1, next_configuration,
        predecessor_stage_sha256=current_stage.configuration_sha256,
    )
    checkpoint_relative = season_doc.get("last_checkpoint")
    relative = Path(str(checkpoint_relative))
    if (
        not isinstance(checkpoint_relative, str)
        or relative.is_absolute()
        or ".." in relative.parts
        or len(relative.parts) != 2
        or relative.parts[0] != "checkpoints"
    ):
        raise B7LifecycleError("completed-stage checkpoint path is outside its run")
    run_root = Path(str(lineage["current_run_root"])).resolve()
    checkpoint = _file(run_root / relative, "completed-stage checkpoint")
    if (
        checkpoint.parent != (run_root / "checkpoints").resolve()
        or season_doc.get("checkpoint_sha256") != _sha256(checkpoint)
    ):
        raise B7LifecycleError("completed-stage checkpoint digest differs")
    successor = json.loads(json.dumps(primary))
    successor["curriculum"] = _stage_mapping(next_stage, _record(gate_path))
    successor["checkpoint"] = {
        "mode": "same-lineage-stage-advance", "path": str(checkpoint),
        "sha256": _sha256(checkpoint),
        "lineage_root_sha256": season_doc["lineage_root_sha256"],
        "current_season_report": None,
    }
    primary_next = _publish(
        admission_root, f"primary-stage-{next_stage.stage_id:02d}", successor
    )
    cold_next = json.loads(json.dumps(cold))
    cold_next["inputs"]["trainer_checkpoint"] = _record(checkpoint)
    cold_next["inputs"]["trainer_current_season"] = None
    cold_next["inputs"]["training_runtime"] = _record(primary_next)
    cold_next_path = _publish(
        admission_root, f"cold-start-stage-{next_stage.stage_id:02d}", cold_next
    )
    service_next = json.loads(json.dumps(service))
    service_next["retirement_cold_start"] = str(cold_next_path)
    service_next["training_runtime"] = _record(primary_next)
    if service_path.read_bytes() != service_bytes:
        raise B7LifecycleError("service selector changed during stage transaction")
    _require_selector_idle(runtime_root)
    _atomic_json(service_path, service_next, replace=True)
    try:
        from train.multires_service import service_preflight

        service_preflight(runtime_root)
    except Exception as error:
        if service_path.read_bytes() == _canonical(service_next):
            _atomic_json(service_path, service, replace=True)
        raise B7LifecycleError(
            f"successor selector failed complete service preflight: {error}"
        ) from error
    lifecycle = _seal({
        "schema": LIFECYCLE_SCHEMA, "state": "stage-active",
        "stage_id": next_stage.stage_id,
        "stage_configuration_sha256": next_stage.configuration_sha256,
        "predecessor_gate": _record(gate_path),
        "completed_stage_archive": _record(archive),
        "selected_checkpoint": _record(checkpoint),
        "primary_runtime": _record(primary_next),
        "cold_start": _record(cold_next_path),
        "service_selector": _record(service_path),
        "automatic_promotion": False,
    })
    lifecycle_path = _publish(
        admission_root, f"lifecycle-stage-{next_stage.stage_id:02d}", lifecycle
    )
    return {
        "evaluation": _record(evaluation_path), "gate": _record(gate_path),
        "lifecycle": _record(lifecycle_path), "service_selector": _record(service_path),
    }


def resume_stage(args: argparse.Namespace) -> dict[str, Any]:
    """Select only the checkpoint named by the sealed current same-stage report."""
    runtime_root = _directory(args.runtime_root, "runtime root")
    _require_selector_idle(runtime_root)
    service_path = _file(runtime_root / "multires-service.json", "service selector")
    service_bytes = service_path.read_bytes()
    service = _json(service_path, "service selector")
    if service.get("schema") != SERVICE_SCHEMA:
        raise B7LifecycleError("service selector is not v2")
    primary_path = _record_path(service.get("training_runtime"), "primary runtime")
    primary = _json(primary_path, "primary runtime")
    cold_path = _file(Path(str(service.get("retirement_cold_start"))), "cold start")
    cold = _json(cold_path, "cold start")
    if primary.get("schema") != PRIMARY_SCHEMA or cold.get("schema") != COLD_START_SCHEMA:
        raise B7LifecycleError("active primary/cold-start schemas differ")
    curriculum = primary.get("curriculum")
    if not isinstance(curriculum, Mapping):
        raise B7LifecycleError("active curriculum is absent")
    stage = CurriculumStage.create(
        curriculum.get("stage_id"), curriculum.get("configuration", {}),
        predecessor_stage_sha256=curriculum.get("predecessor_stage_sha256"),
    )
    if stage.configuration_sha256 != curriculum.get("configuration_sha256"):
        raise B7LifecycleError("active curriculum configuration digest differs")
    lineage = cold.get("lineage")
    inputs = cold.get("inputs")
    if not isinstance(lineage, Mapping) or not isinstance(inputs, Mapping):
        raise B7LifecycleError("cold-start lineage/input records are absent")
    run_root = _directory(
        Path(str(lineage["current_run_root"])), "current run root"
    )
    checkpoint_root = _directory(
        Path(str(lineage["checkpoint_root"])), "checkpoint root"
    )
    season_root = _directory(
        Path(str(lineage["season_report_root"])), "season report root"
    )
    if (
        checkpoint_root != (run_root / "checkpoints").resolve()
        or season_root != (run_root / "season").resolve()
    ):
        raise B7LifecycleError("resume run-root layout differs")
    season_path = _file(season_root / "current.json", "current season report")
    season = _json(season_path, "current season report")
    runtime_sha256 = primary.get("runtime_manifest_sha256")
    atlas_sha256 = season.get("atlas_sha256")
    atlas_catalog_sha256 = primary.get("atlas_catalog_sha256")
    lineage_sha256 = season.get("lineage_root_sha256")
    if not all(
        _valid_digest(value)
        for value in (
            runtime_sha256, atlas_sha256, atlas_catalog_sha256, lineage_sha256,
        )
    ):
        raise B7LifecycleError(
            "resume runtime/Atlas-catalog/active-Atlas/lineage identity is invalid"
        )
    if season.get("atlas_catalog_sha256") != atlas_catalog_sha256:
        raise B7LifecycleError("resume season Atlas catalog differs from primary")
    try:
        progress_from_season_report(
            season, stage,
            runtime_manifest_sha256=str(runtime_sha256),
            atlas_sha256=str(atlas_sha256),
            atlas_catalog_sha256=str(atlas_catalog_sha256),
            lineage_root_sha256=str(lineage_sha256),
        )
    except (TypeError, ValueError, RuntimeError) as error:
        raise B7LifecycleError(f"current season cannot authorize resume: {error}") from error
    checkpoint = _file(
        run_root / str(season.get("last_checkpoint")), "current-season checkpoint"
    )
    if (
        checkpoint.parent != checkpoint_root
        or season.get("checkpoint_sha256") != _sha256(checkpoint)
    ):
        raise B7LifecycleError("current-season checkpoint path/digest differs")

    admission_root = _directory(runtime_root / "b7-admission", "B7 admission root")
    successor = json.loads(json.dumps(primary))
    successor["checkpoint"] = {
        "mode": "same-lineage-resume", "path": str(checkpoint),
        "sha256": _sha256(checkpoint),
        "lineage_root_sha256": str(lineage_sha256),
        "current_season_report": _record(season_path),
    }
    primary_next = _publish(
        admission_root, f"primary-stage-{stage.stage_id:02d}-resume", successor
    )
    cold_next = json.loads(json.dumps(cold))
    cold_next["inputs"]["trainer_checkpoint"] = _record(checkpoint)
    cold_next["inputs"]["trainer_current_season"] = _record(season_path)
    cold_next["inputs"]["training_runtime"] = _record(primary_next)
    cold_next_path = _publish(
        admission_root, f"cold-start-stage-{stage.stage_id:02d}-resume", cold_next
    )
    service_next = json.loads(json.dumps(service))
    service_next["retirement_cold_start"] = str(cold_next_path)
    service_next["training_runtime"] = _record(primary_next)
    if service_path.read_bytes() != service_bytes:
        raise B7LifecycleError("service selector changed during resume transaction")
    _require_selector_idle(runtime_root)
    _atomic_json(service_path, service_next, replace=True)
    try:
        from train.multires_service import service_preflight

        service_preflight(runtime_root)
    except Exception as error:
        if service_path.read_bytes() == _canonical(service_next):
            _atomic_json(service_path, service, replace=True)
        raise B7LifecycleError(
            f"resume selector failed complete service preflight: {error}"
        ) from error
    lifecycle = _seal({
        "schema": LIFECYCLE_SCHEMA, "state": "stage-active-resume",
        "stage_id": stage.stage_id,
        "stage_configuration_sha256": stage.configuration_sha256,
        "selected_checkpoint": _record(checkpoint),
        "current_season_report": _record(season_path),
        "primary_runtime": _record(primary_next),
        "cold_start": _record(cold_next_path),
        "service_selector": _record(service_path),
        "automatic_promotion": False,
    })
    lifecycle_path = _publish(
        admission_root, f"lifecycle-stage-{stage.stage_id:02d}-resume", lifecycle
    )
    return {
        "lifecycle": _record(lifecycle_path),
        "service_selector": _record(service_path),
    }


def finalize_stage_seven(args: argparse.Namespace) -> dict[str, Any]:
    """Evaluate terminal stage 7 and publish its gate without changing selectors."""
    runtime_root = _directory(args.runtime_root, "runtime root")
    _require_selector_idle(runtime_root)
    service_path = _file(runtime_root / "multires-service.json", "service selector")
    service_bytes = service_path.read_bytes()
    service = _json(service_path, "service selector")
    if service.get("schema") != SERVICE_SCHEMA:
        raise B7LifecycleError("service selector is not v2")
    primary_path = _record_path(service.get("training_runtime"), "primary runtime")
    primary = _json(primary_path, "primary runtime")
    cold_path = _file(Path(str(service.get("retirement_cold_start"))), "cold start")
    cold = _json(cold_path, "cold start")
    if primary.get("schema") != PRIMARY_SCHEMA or cold.get("schema") != COLD_START_SCHEMA:
        raise B7LifecycleError("active primary/cold-start schemas differ")
    curriculum = primary.get("curriculum")
    if not isinstance(curriculum, Mapping):
        raise B7LifecycleError("active curriculum is absent")
    current_stage = CurriculumStage.create(
        curriculum.get("stage_id"), curriculum.get("configuration", {}),
        predecessor_stage_sha256=curriculum.get("predecessor_stage_sha256"),
    )
    if current_stage.configuration_sha256 != curriculum.get("configuration_sha256"):
        raise B7LifecycleError("active curriculum configuration digest differs")
    if current_stage.stage_id != 7:
        raise B7LifecycleError("terminal finalization requires active stage 7")
    current_configuration = json.loads(current_stage.configuration_json)["configuration"]
    reference_path = _record_path(
        current_configuration.get("matched_seed_guide_on_reference"),
        "stage-7 guide-on reference",
    )
    reference, _reference_season = _validate_guide_reference(reference_path)
    predecessor_path = _record_path(
        curriculum.get("predecessor_gate"), "stage-7 predecessor gate"
    )
    predecessor = _json(predecessor_path, "stage-7 predecessor gate")
    _validate_seal(predecessor, CURRICULUM_GATE_SCHEMA, "stage-7 predecessor gate")
    predecessor_artifacts = predecessor.get("artifacts")
    primary_checkpoint = primary.get("checkpoint")
    if (
        predecessor.get("decision") != "passed"
        or predecessor.get("stage_id") != 6
        or predecessor.get("stage_configuration_sha256")
        != current_stage.predecessor_stage_sha256
        or predecessor.get("runtime_manifest_sha256")
        != reference["runtime_manifest_sha256"]
        or predecessor.get("atlas_catalog_sha256")
        != reference["atlas_catalog_sha256"]
        or predecessor.get("lineage_root_sha256")
        != reference["lineage_root_sha256"]
        or predecessor.get("accepted_transitions")
        != reference["accepted_transitions"]
        or predecessor.get("policy_updates") != reference["policy_version"]
        or not isinstance(predecessor_artifacts, Mapping)
        or not isinstance(predecessor_artifacts.get("completed_stage"), Mapping)
        or predecessor_artifacts["completed_stage"].get("sha256")
        != reference["completed_season"]["sha256"]
        or not isinstance(primary_checkpoint, Mapping)
        or primary_checkpoint.get("path") != reference["checkpoint"]["path"]
        or primary_checkpoint.get("sha256") != reference["checkpoint"]["sha256"]
        or primary_checkpoint.get("lineage_root_sha256")
        != reference["lineage_root_sha256"]
    ):
        raise B7LifecycleError(
            "stage-7 guide reference is not the exact predecessor policy/season"
        )

    lineage = cold.get("lineage")
    if not isinstance(lineage, Mapping):
        raise B7LifecycleError("cold-start lineage roots are absent")
    season = _file(
        Path(str(lineage["season_report_root"])) / "current.json",
        "completed current season",
    )
    admission_root = _directory(runtime_root / "b7-admission", "B7 admission root")
    archive = Path(str(lineage["season_report_root"])) / (
        f"stage-07-{_sha256(season)[:20]}.json"
    )
    if archive.exists():
        if archive.is_symlink() or _sha256(archive) != _sha256(season):
            raise B7LifecycleError("completed-stage archive collision differs")
    else:
        _copy_exclusive(season, archive)
    evaluation = evaluate_completed_stage(archive, current_stage)
    evaluation_path = _publish(admission_root, "evaluation-stage-07", evaluation)
    if evaluation["decision"] != "passed":
        raise B7LifecycleError(
            "stage 7 evaluation failed; no gate or selector changed"
        )
    season_doc = _json(season, "completed current season")
    counters = season_doc["counters"]
    gate = create_curriculum_gate_evidence(
        current_stage, decision="passed",
        runtime_manifest_sha256=season_doc["runtime_manifest_sha256"],
        atlas_catalog_sha256=season_doc["atlas_catalog_sha256"],
        lineage_root_sha256=season_doc["lineage_root_sha256"],
        accepted_transitions=counters["accepted_transitions"],
        policy_updates=counters["policy_updates"],
        optimizer_steps=counters["optimizer_steps"],
        completed_stage=_record(archive), evaluator=_record(evaluation_path),
    )
    gate_path = _publish(admission_root, "gate-stage-07", gate)
    run_root = Path(str(lineage["current_run_root"])).resolve()
    checkpoint_relative = season_doc.get("last_checkpoint")
    relative = Path(str(checkpoint_relative))
    if (
        not isinstance(checkpoint_relative, str)
        or relative.is_absolute()
        or ".." in relative.parts
        or len(relative.parts) != 2
        or relative.parts[0] != "checkpoints"
    ):
        raise B7LifecycleError("completed-stage checkpoint path is outside its run")
    checkpoint = _file(run_root / relative, "completed-stage checkpoint")
    if (
        checkpoint.parent != (run_root / "checkpoints").resolve()
        or season_doc.get("checkpoint_sha256") != _sha256(checkpoint)
    ):
        raise B7LifecycleError("completed-stage checkpoint digest differs")
    if service_path.read_bytes() != service_bytes:
        raise B7LifecycleError("service selector changed during stage finalization")
    lifecycle = _seal({
        "schema": LIFECYCLE_SCHEMA, "state": "stage-complete",
        "stage_id": 7,
        "stage_configuration_sha256": current_stage.configuration_sha256,
        "terminal_gate": _record(gate_path),
        "completed_stage_archive": _record(archive),
        "selected_checkpoint": _record(checkpoint),
        "primary_runtime": _record(primary_path),
        "cold_start": _record(cold_path),
        "service_selector": _record(service_path),
        "automatic_promotion": False,
    })
    lifecycle_path = _publish(admission_root, "lifecycle-stage-07-complete", lifecycle)
    return {
        "evaluation": _record(evaluation_path), "gate": _record(gate_path),
        "lifecycle": _record(lifecycle_path), "service_selector": _record(service_path),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    author = subparsers.add_parser("author-stage1")
    author.add_argument("--b5-gate", type=Path, required=True)
    author.add_argument("--b6-gate", type=Path, required=True)
    author.add_argument("--stage-configuration", type=Path, required=True)
    author.add_argument("--primary-template", type=Path, required=True)
    author.add_argument("--cold-start-template", type=Path, required=True)
    author.add_argument("--service-template", type=Path, required=True)
    author.add_argument("--runtime-root", type=Path, required=True)
    guide = subparsers.add_parser("author-guide-reference")
    guide.add_argument("--runtime-root", type=Path, required=True)
    guide.add_argument("--task-success-metric-path", required=True)
    advance = subparsers.add_parser("advance")
    advance.add_argument("--runtime-root", type=Path, required=True)
    advance.add_argument("--next-stage-configuration", type=Path, required=True)
    resume = subparsers.add_parser("resume")
    resume.add_argument("--runtime-root", type=Path, required=True)
    finalize = subparsers.add_parser("finalize-stage7")
    finalize.add_argument("--runtime-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "author-stage1":
            result = author_stage_one(args)
        elif args.command == "author-guide-reference":
            result = author_guide_on_reference(args)
        elif args.command == "advance":
            result = advance_stage(args)
        elif args.command == "resume":
            result = resume_stage(args)
        else:
            result = finalize_stage_seven(args)
    except (B7LifecycleError, KeyError, TypeError, ValueError, OSError) as error:
        print(f"B7 lifecycle failed: {error}", file=sys.stderr)
        return 2
    sys.stdout.buffer.write(_canonical(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
