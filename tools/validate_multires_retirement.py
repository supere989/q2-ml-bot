#!/usr/bin/env python3
"""Read-only B5/G0 retirement and fresh cold-start admission gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


RETIREMENT_SCHEMA = "q2-multires-runtime-retirement-v1"
COLD_START_SCHEMA = "q2-multires-cold-start-v1"
REPORT_SCHEMA = "q2-multires-retirement-validation-v1"
CHECKPOINT_ATTESTATION_SCHEMA = "q2-multires-cold-checkpoint-v1"

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_PLACEHOLDER = re.compile(
    r"(?:<[^>]+>|\$\{[^}]+\}|\bTODO\b|\bTBD\b|CHANGEME|REPLACE[_-]?ME)",
    re.IGNORECASE,
)
_LEGACY_CHECKPOINT = re.compile(
    r"(?:^|[-_.])(?:policy_[0-9]{8}|latest|optimizer(?:_state)?)(?:[-_.]|$)",
    re.IGNORECASE,
)
_TEXT_SUFFIXES = frozenset(
    (".cfg", ".ini", ".json", ".py", ".service", ".sh", ".toml", ".txt", ".yaml", ".yml")
)

_ACTIVE_RUNTIME = {
    "checkpoint_format": "q2-multires-attested-checkpoint-v1",
    "client_builder": "harness.client_batch.build_network_client_batch",
    "client_wire_version": 8,
    "collector": "harness.multires_collector.MultiresSynchronousCollector",
    "dyn_magic": "Q2LAT002",
    "map_bundle_version": 3,
    "policy_class": "models.multires_policy.MultiresQ2BotPolicy",
    "policy_generation": "multires-atlas-policy-v1",
    "provider_class": "harness.rust_multires_provider.RustAtlasSpatialProvider",
    "qm3c_magic": "0x514d3343",
    "qm3c_version": 2,
    "rollout_schema": "ppo-telemetry-multires-v1",
    "run_tag": "public_network_multires_atlas_fresh_v1",
    "rust_dyn_call": "q2_lattice_rs.DynRuntime.commit_frame",
    "service_module": "train.multires_service",
    "teacher_version": 4,
    "proof_module": "train.multires_one_run",
    "trainer_module": "train.multires_primary",
    "wire_generation": "B4",
}

_REQUIRED_SELECTOR_CLASSES = frozenset(
    (
        "runtime.python_module",
        "runtime.service_or_resume",
        "policy.python_symbol",
        "policy.checkpoint",
        "optimizer.serialized_state",
        "dyn.snapshot_or_adapter",
        "protocol.public_wire",
        "protocol.teacher",
        "protocol.rollout",
        "map_bundle.manifest",
    )
)


class RetirementValidationError(RuntimeError):
    """Raised without mutating the inspected operational roots."""


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _valid_digest(value: object) -> bool:
    return (
        isinstance(value, str)
        and bool(_SHA256.fullmatch(value))
        and len(set(value)) > 1
    )


def _reject_placeholder(value: object, label: str) -> None:
    if isinstance(value, str):
        if _PLACEHOLDER.search(value) or value == "0" * 64:
            raise RetirementValidationError(f"{label} contains a placeholder")
    elif isinstance(value, Mapping):
        for key, item in value.items():
            _reject_placeholder(item, f"{label}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_placeholder(item, f"{label}[{index}]")


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RetirementValidationError(f"{label} is not valid JSON") from error
    if not isinstance(value, dict):
        raise RetirementValidationError(f"{label} must be a JSON object")
    _reject_placeholder(value, label)
    return value


def _reject_symlink_components(path: Path, label: str) -> None:
    candidate = path
    while True:
        if candidate.is_symlink():
            raise RetirementValidationError(f"{label} contains a symbolic link")
        if candidate.parent == candidate:
            break
        candidate = candidate.parent


def _absolute_path(path: Path, label: str, *, kind: str) -> Path:
    source = Path(path).expanduser()
    if not source.is_absolute() or ".." in source.parts:
        raise RetirementValidationError(f"{label} must be an absolute path without '..'")
    _reject_symlink_components(source, label)
    if kind == "file" and not source.is_file():
        raise RetirementValidationError(f"{label} must be a regular file")
    if kind == "directory" and not source.is_dir():
        raise RetirementValidationError(f"{label} must be a directory")
    return source.resolve(strict=True)


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _require_within(path: Path, roots: Sequence[Path], label: str) -> None:
    if not any(_is_within(path, root) for root in roots):
        raise RetirementValidationError(f"{label} escapes every operational root")


def _manifest(path: Path, expected_sha256: str) -> tuple[dict[str, Any], str]:
    manifest_path = _absolute_path(path, "retirement manifest", kind="file")
    raw = manifest_path.read_bytes()
    manifest = _load_json(manifest_path, "retirement manifest")
    if raw != _canonical_bytes(manifest):
        raise RetirementValidationError("retirement manifest is not canonical JSON")
    digest = hashlib.sha256(raw).hexdigest()
    if not _valid_digest(expected_sha256) or digest != expected_sha256:
        raise RetirementValidationError("retirement manifest SHA-256 differs")
    required = {
        "schema", "status", "fallback_allowed", "operational_selection",
        "legacy_checkpoint_discovery_allowed", "legacy_runtime_import_allowed",
        "active_runtime", "forbidden_selectors", "retired_run_tags",
        "historical_evidence",
    }
    if set(manifest) != required:
        raise RetirementValidationError("retirement manifest fields differ")
    if (
        manifest["schema"] != RETIREMENT_SCHEMA
        or manifest["status"] != "legacy-runtime-retired"
        or manifest["fallback_allowed"] is not False
        or manifest["legacy_checkpoint_discovery_allowed"] is not False
        or manifest["legacy_runtime_import_allowed"] is not False
        or manifest["operational_selection"] != "fresh-explicit-only"
        or manifest["active_runtime"] != _ACTIVE_RUNTIME
    ):
        raise RetirementValidationError("retirement manifest does not freeze the active runtime")
    selectors = manifest["forbidden_selectors"]
    if not isinstance(selectors, list):
        raise RetirementValidationError("forbidden selectors must be a list")
    classes: set[str] = set()
    for index, record in enumerate(selectors):
        if not isinstance(record, dict) or set(record) != {"selector_class", "values"}:
            raise RetirementValidationError(f"forbidden selector[{index}] is malformed")
        selector_class = record["selector_class"]
        values = record["values"]
        if (
            not isinstance(selector_class, str)
            or selector_class in classes
            or not isinstance(values, list)
            or not values
            or any(not isinstance(value, str) or not value for value in values)
            or len(values) != len(set(values))
        ):
            raise RetirementValidationError(f"forbidden selector[{index}] is invalid")
        classes.add(selector_class)
    if classes != _REQUIRED_SELECTOR_CLASSES:
        raise RetirementValidationError("forbidden selector classes are incomplete")
    run_tags = manifest["retired_run_tags"]
    if (
        not isinstance(run_tags, list)
        or not run_tags
        or any(not isinstance(value, str) or not value for value in run_tags)
        or len(run_tags) != len(set(run_tags))
    ):
        raise RetirementValidationError("retired run tags are invalid")
    return manifest, digest


def _historical_roots(manifest: Mapping[str, Any]) -> tuple[Path, ...]:
    records = manifest["historical_evidence"]
    if not isinstance(records, list) or not records:
        raise RetirementValidationError("historical evidence inventory is empty")
    roots = []
    for index, record in enumerate(records):
        if (
            not isinstance(record, dict)
            or set(record) != {"path", "source_record", "status"}
            or record.get("status") != "evidence-only"
            or record.get("source_record") != "docs/multires/B0-GATE.json"
        ):
            raise RetirementValidationError(f"historical evidence[{index}] is malformed")
        path = Path(str(record["path"]))
        if not path.is_absolute() or ".." in path.parts:
            raise RetirementValidationError("historical evidence path is unsafe")
        roots.append(path.resolve(strict=False))
    if len(roots) != len(set(roots)):
        raise RetirementValidationError("historical evidence paths repeat")
    return tuple(roots)


def _forbidden_tokens(manifest: Mapping[str, Any]) -> tuple[str, ...]:
    values = [
        value
        for record in manifest["forbidden_selectors"]
        for value in record["values"]
    ]
    values.extend(manifest["retired_run_tags"])
    return tuple(sorted(set(values), key=lambda value: (len(value), value), reverse=True))


def _scan_text(path: Path, tokens: Sequence[str], label: str) -> None:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        raise RetirementValidationError(f"{label} is not UTF-8 text") from error
    lowered = text.lower()
    matches = [token for token in tokens if token.lower() in lowered]
    if matches:
        raise RetirementValidationError(
            f"{label} selects retired value {sorted(matches)[0]!r}"
        )
    if path.suffix.lower() == ".json":
        document = _load_json(path, label)
        legacy_fields = {
            "client_wire_version": 4,
            "public_client_wire_version": 4,
            "teacher_version": 2,
            "bundle_version": 2,
            "rust_dyn_magic": "Q2LAT001",
            "dyn_magic": "Q2LAT001",
            "rollout_schema": "ppo-telemetry-v8",
            "rollout_telemetry_schema": "ppo-telemetry-v8",
        }

        def walk(value: object) -> None:
            if isinstance(value, Mapping):
                for key, item in value.items():
                    if key in legacy_fields and item == legacy_fields[key]:
                        raise RetirementValidationError(
                            f"{label} contains legacy {key}={item!r}"
                        )
                    if key == "legacy_fallback_enabled" and item is not False:
                        raise RetirementValidationError(f"{label} enables legacy fallback")
                    walk(item)
            elif isinstance(value, list):
                for item in value:
                    walk(item)

        walk(document)


def _scan_operational_roots(
    roots: Sequence[Path],
    *,
    tokens: Sequence[str],
    admitted_files: set[Path],
) -> int:
    files_checked = 0
    for root in roots:
        for directory, directory_names, file_names in os.walk(root, followlinks=False):
            base = Path(directory)
            for name in sorted(directory_names):
                candidate = base / name
                if candidate.is_symlink():
                    raise RetirementValidationError(
                        f"operational root contains symlink {candidate}"
                    )
            for name in sorted(file_names):
                candidate = base / name
                if candidate.is_symlink():
                    raise RetirementValidationError(
                        f"operational root contains symlink {candidate}"
                    )
                resolved = candidate.resolve(strict=True)
                files_checked += 1
                if any(tag.lower() in name.lower() for tag in tokens):
                    raise RetirementValidationError(
                        f"operational filename selects retired value: {candidate}"
                    )
                if resolved not in admitted_files and (
                    candidate.suffix.lower() in {".pt", ".pth", ".ckpt"}
                    or _LEGACY_CHECKPOINT.search(name)
                ):
                    raise RetirementValidationError(
                        f"unadmitted checkpoint/optimizer artifact: {candidate}"
                    )
                with candidate.open("rb") as handle:
                    magic = handle.read(8)
                if magic == b"Q2LAT001":
                    raise RetirementValidationError(
                        f"operational root contains Q2LAT001: {candidate}"
                    )
                if candidate.suffix.lower() in _TEXT_SUFFIXES:
                    _scan_text(candidate, tokens, f"operational file {candidate}")
    return files_checked


def _artifact_record(
    value: object,
    label: str,
    *,
    operational_roots: Sequence[Path],
) -> tuple[Path, str]:
    if not isinstance(value, dict) or set(value) != {"path", "sha256"}:
        raise RetirementValidationError(f"{label} record is malformed")
    if not _valid_digest(value["sha256"]):
        raise RetirementValidationError(f"{label} SHA-256 is invalid")
    path = _absolute_path(Path(str(value["path"])), label, kind="file")
    _require_within(path, operational_roots, label)
    if _sha256(path) != value["sha256"]:
        raise RetirementValidationError(f"{label} bytes differ from declared SHA-256")
    return path, str(value["sha256"])


def _load_checkpoint(
    path: Path,
    attestation_path: Path,
    active: Mapping[str, Any],
    *,
    runtime_evidence: Mapping[str, Any],
    training_manifest_path: Path,
    optimizer_configuration: Mapping[str, Any],
) -> None:
    """Load the real weights-only envelope; sidecars are corroboration only."""
    attestation = _load_json(attestation_path, "checkpoint attestation")
    expected_fields = {
        "schema", "checkpoint_sha256", "checkpoint_format", "policy_generation",
        "architecture", "initialization", "training_step", "observation_dim",
        "action_dim", "posture_classes", "optimizer_state", "normalization_state",
    }
    if set(attestation) != expected_fields:
        raise RetirementValidationError("checkpoint attestation fields differ")
    if (
        attestation["schema"] != CHECKPOINT_ATTESTATION_SCHEMA
        or attestation["checkpoint_sha256"] != _sha256(path)
        or attestation["checkpoint_format"] != active["checkpoint_format"]
        or attestation["policy_generation"] != active["policy_generation"]
        or attestation["architecture"] != active["policy_class"]
        or attestation["initialization"] != "random"
        or attestation["training_step"] != 0
        or attestation["observation_dim"] != 298
        or attestation["action_dim"] != 8
        or attestation["posture_classes"] != 3
        or attestation["optimizer_state"] != "fresh-empty"
        or attestation["normalization_state"] != "fresh-empty"
    ):
        raise RetirementValidationError("checkpoint attestation is not fresh multires")
    try:
        import torch
        from harness.multires_lineage import load_attested_checkpoint
        from harness.multires_runtime import validate_runtime_evidence
        from harness.multires_training_config import MultiresTrainingConfiguration
        from models.multires_policy import MultiresQ2BotPolicy

        training_configuration = MultiresTrainingConfiguration.from_json(
            training_manifest_path.read_text(encoding="utf-8")
        )
        validated_runtime = validate_runtime_evidence(
            runtime_evidence,
            expected_atlas_sha256=str(runtime_evidence.get("atlas_sha256", "")),
        )
        if (
            not isinstance(optimizer_configuration, Mapping)
            or set(optimizer_configuration)
            != {"class", "learning_rate", "kwargs"}
            or optimizer_configuration.get("class") != "torch.optim.Adam"
            or isinstance(optimizer_configuration.get("learning_rate"), bool)
            or not isinstance(
                optimizer_configuration.get("learning_rate"), (int, float)
            )
            or not math.isfinite(float(optimizer_configuration["learning_rate"]))
            or float(optimizer_configuration["learning_rate"]) <= 0.0
            or not isinstance(optimizer_configuration.get("kwargs"), Mapping)
            or "lr" in optimizer_configuration["kwargs"]
        ):
            raise RetirementValidationError(
                "cold-start optimizer configuration is not exact Adam"
            )
        policy = MultiresQ2BotPolicy().to(torch.device("cpu"))
        optimizer = torch.optim.Adam(
            policy.parameters(),
            lr=float(optimizer_configuration["learning_rate"]),
            **dict(optimizer_configuration["kwargs"]),
        )
        manifest = load_attested_checkpoint(
            path,
            policy,
            expected_atlas_sha256=validated_runtime.atlas_sha256,
            expected_runtime_manifest_sha256=(
                validated_runtime.runtime_manifest_sha256
            ),
            expected_training_config=training_configuration,
            optimizer=optimizer,
            map_location="cpu",
        )
    except RetirementValidationError:
        raise
    except (
        ImportError, OSError, UnicodeDecodeError, ValueError, TypeError, RuntimeError
    ) as error:
        # LineageError is a ValueError and is intentionally collapsed at this
        # external gate so no raw/marker/sidecar path appears admissible.
        raise RetirementValidationError(
            f"checkpoint is not a real attested weights-only step-zero envelope: {error}"
        ) from error
    if (
        manifest.initialization != "random"
        or manifest.training_step != 0
        or optimizer.state_dict().get("state") != {}
        or attestation["checkpoint_format"] != manifest.checkpoint_format
        or attestation["policy_generation"] != manifest.policy_generation
        or attestation["architecture"] != manifest.architecture
        or attestation["initialization"] != manifest.initialization
        or attestation["training_step"] != manifest.training_step
        or attestation["observation_dim"] != manifest.observation_dim
        or attestation["action_dim"] != manifest.action_dim
        or attestation["posture_classes"] != manifest.posture_classes
    ):
        raise RetirementValidationError(
            "checkpoint/sidecar do not corroborate a fresh real step-zero lineage"
        )


def _cold_start(
    path: Path,
    *,
    manifest_sha256: str,
    active: Mapping[str, Any],
    operational_roots: Sequence[Path],
    historical_roots: Sequence[Path],
) -> tuple[dict[str, Any], set[Path]]:
    cold_path = _absolute_path(path, "cold-start declaration", kind="file")
    cold = _load_json(cold_path, "cold-start declaration")
    required = {
        "schema", "retirement_manifest_sha256", "runtime_manifest_sha256",
        "selectors", "lineage", "optimizer", "inputs",
    }
    if set(cold) != required or cold.get("schema") != COLD_START_SCHEMA:
        raise RetirementValidationError("cold-start declaration fields/schema differ")
    if cold["retirement_manifest_sha256"] != manifest_sha256:
        raise RetirementValidationError("cold-start retirement digest differs")
    if not _valid_digest(cold["runtime_manifest_sha256"]):
        raise RetirementValidationError("cold-start runtime digest is invalid")
    selector_keys = {
        "service_module", "proof_module", "trainer_module", "client_builder", "collector",
        "policy_class", "provider_class", "rust_dyn_call",
    }
    expected_selectors = {key: active[key] for key in selector_keys}
    if cold["selectors"] != expected_selectors:
        raise RetirementValidationError("cold-start selectors are not exact multires selectors")
    lineage = cold["lineage"]
    lineage_keys = {
        "run_tag", "initialization", "training_step", "optimizer_state",
        "normalization_state", "current_run_root", "checkpoint_root",
        "tensorboard_root", "rollout_root", "update_root", "season_report_root",
    }
    if not isinstance(lineage, dict) or set(lineage) != lineage_keys:
        raise RetirementValidationError("cold-start lineage fields differ")
    if (
        lineage["run_tag"] != active["run_tag"]
        or lineage["initialization"] != "random"
        or lineage["training_step"] != 0
        or lineage["optimizer_state"] != "fresh-empty"
        or lineage["normalization_state"] != "fresh-empty"
    ):
        raise RetirementValidationError("cold-start lineage is not fresh random step zero")
    run_paths = {
        key: _absolute_path(Path(str(lineage[key])), f"lineage.{key}", kind="directory")
        for key in (
            "current_run_root", "checkpoint_root", "tensorboard_root",
            "rollout_root", "update_root", "season_report_root",
        )
    }
    for key, run_path in run_paths.items():
        _require_within(run_path, operational_roots, f"lineage.{key}")
        if any(_is_within(run_path, historical) for historical in historical_roots):
            raise RetirementValidationError(f"lineage.{key} selects historical evidence")
    current = run_paths["current_run_root"]
    if current.name != active["run_tag"]:
        raise RetirementValidationError("current run directory does not equal the fresh run tag")
    child_paths = [path for key, path in run_paths.items() if key != "current_run_root"]
    if len(child_paths) != len(set(child_paths)) or any(
        not _is_within(child, current) or child == current for child in child_paths
    ):
        raise RetirementValidationError("fresh run subroots are not distinct children")

    inputs = cold["inputs"]
    if not isinstance(inputs, dict) or set(inputs) != {
        "checkpoint", "checkpoint_attestation", "runtime_evidence",
        "training_manifest", "bundle_manifest", "dyn_snapshots",
        "training_runtime", "trainer_checkpoint", "trainer_current_season",
    }:
        raise RetirementValidationError("cold-start input fields differ")
    checkpoint, _ = _artifact_record(
        inputs["checkpoint"], "checkpoint", operational_roots=operational_roots
    )
    checkpoint_attestation, _ = _artifact_record(
        inputs["checkpoint_attestation"],
        "checkpoint attestation",
        operational_roots=operational_roots,
    )
    runtime_evidence, _ = _artifact_record(
        inputs["runtime_evidence"], "runtime evidence", operational_roots=operational_roots
    )
    training_manifest, _ = _artifact_record(
        inputs["training_manifest"],
        "training manifest",
        operational_roots=operational_roots,
    )
    bundle_manifest, _ = _artifact_record(
        inputs["bundle_manifest"], "bundle manifest", operational_roots=operational_roots
    )
    training_runtime, _ = _artifact_record(
        inputs["training_runtime"], "primary training runtime",
        operational_roots=operational_roots,
    )
    trainer_checkpoint, _ = _artifact_record(
        inputs["trainer_checkpoint"], "primary trainer checkpoint",
        operational_roots=operational_roots,
    )
    trainer_current_season = None
    if inputs["trainer_current_season"] is not None:
        trainer_current_season, _ = _artifact_record(
            inputs["trainer_current_season"], "primary trainer current season",
            operational_roots=operational_roots,
        )
    if not _is_within(checkpoint, run_paths["checkpoint_root"]):
        raise RetirementValidationError("checkpoint is outside the fresh checkpoint root")
    if not _is_within(trainer_checkpoint, run_paths["checkpoint_root"]):
        raise RetirementValidationError(
            "primary trainer checkpoint is outside the declared checkpoint root"
        )
    if trainer_current_season is not None and (
        trainer_current_season != run_paths["season_report_root"] / "current.json"
    ):
        raise RetirementValidationError(
            "primary trainer season differs from the declared season root"
        )
    dyn_records = inputs["dyn_snapshots"]
    if not isinstance(dyn_records, list) or not dyn_records:
        raise RetirementValidationError("cold-start Dyn snapshots are empty")
    dyn_paths = [
        _artifact_record(
            record, f"Dyn snapshot[{index}]", operational_roots=operational_roots
        )[0]
        for index, record in enumerate(dyn_records)
    ]
    selected = {
        checkpoint, checkpoint_attestation, runtime_evidence, training_manifest,
        bundle_manifest, training_runtime, trainer_checkpoint,
        *dyn_paths,
    }
    if trainer_current_season is not None:
        selected.add(trainer_current_season)
    if any(
        any(_is_within(candidate, historical) for historical in historical_roots)
        for candidate in selected
    ):
        raise RetirementValidationError("cold-start input selects historical evidence")
    evidence = _load_json(runtime_evidence, "runtime evidence")
    expected_evidence = {
        "policy_generation": active["policy_generation"],
        "protocol_generation": 2,
        "client_wire_version": active["client_wire_version"],
        "teacher_version": active["teacher_version"],
        "rollout_schema": active["rollout_schema"],
        "causal_magic": int(str(active["qm3c_magic"]), 16),
        "causal_version": active["qm3c_version"],
        "public_teacher_packing_separate": True,
        "public_teacher_field_violations": 0,
        "runtime_manifest_sha256": cold["runtime_manifest_sha256"],
    }
    mismatches = {
        key: (evidence.get(key), wanted)
        for key, wanted in expected_evidence.items()
        if evidence.get(key) != wanted
    }
    if mismatches:
        raise RetirementValidationError(f"runtime evidence is not B4/QM3C: {mismatches}")
    bundle = _load_json(bundle_manifest, "bundle manifest")
    if bundle.get("bundle_version") != 3 or bundle.get("artifact_state") != "admitted":
        raise RetirementValidationError("map input is not admitted bundle-v3")
    for index, dyn_path in enumerate(dyn_paths):
        with dyn_path.open("rb") as handle:
            if handle.read(8) != b"Q2LAT002":
                raise RetirementValidationError(f"Dyn snapshot[{index}] is not Q2LAT002")
    _load_checkpoint(
        checkpoint,
        checkpoint_attestation,
        active,
        runtime_evidence=evidence,
        training_manifest_path=training_manifest,
        optimizer_configuration=cold["optimizer"],
    )
    return cold, selected


def validate_retirement(
    *,
    manifest_path: Path,
    expected_manifest_sha256: str,
    cold_start_path: Path,
    operational_roots: Sequence[Path],
    service_selector_files: Sequence[Path],
) -> dict[str, Any]:
    """Validate selection without deleting, moving, writing, or importing services."""
    manifest, manifest_sha256 = _manifest(manifest_path, expected_manifest_sha256)
    historical = _historical_roots(manifest)
    if not operational_roots:
        raise RetirementValidationError("at least one operational root is required")
    roots = tuple(
        _absolute_path(Path(path), f"operational root[{index}]", kind="directory")
        for index, path in enumerate(operational_roots)
    )
    if len(roots) != len(set(roots)):
        raise RetirementValidationError("operational roots repeat")
    for root in roots:
        for historical_root in historical:
            if _is_within(root, historical_root) or _is_within(historical_root, root):
                raise RetirementValidationError(
                    "operational and historical evidence roots are not disjoint"
                )
    if not service_selector_files:
        raise RetirementValidationError("at least one service selector file is required")
    selector_paths = tuple(
        _absolute_path(Path(path), f"service selector[{index}]", kind="file")
        for index, path in enumerate(service_selector_files)
    )
    if len(selector_paths) != len(set(selector_paths)):
        raise RetirementValidationError("service selector files repeat")
    tokens = _forbidden_tokens(manifest)
    for selector in selector_paths:
        if any(_is_within(selector, root) for root in historical):
            raise RetirementValidationError("service selector is inside historical evidence")
        _scan_text(selector, tokens, f"service selector {selector}")
    aggregate = "\n".join(path.read_text(encoding="utf-8") for path in selector_paths)
    if manifest["active_runtime"]["service_module"] not in aggregate:
        raise RetirementValidationError("service selectors do not select train.multires_service")

    cold, selected = _cold_start(
        cold_start_path,
        manifest_sha256=manifest_sha256,
        active=manifest["active_runtime"],
        operational_roots=roots,
        historical_roots=historical,
    )
    files_checked = _scan_operational_roots(
        roots, tokens=tokens, admitted_files=selected
    )
    return {
        "schema": REPORT_SCHEMA,
        "status": "pass",
        "manifest_sha256": manifest_sha256,
        "cold_start_sha256": hashlib.sha256(_canonical_bytes(cold)).hexdigest(),
        "operational_root_count": len(roots),
        "service_selector_count": len(selector_paths),
        "historical_evidence_root_count": len(historical),
        "cold_start_input_count": len(selected),
        "operational_files_checked": files_checked,
        "read_only": True,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--cold-start", type=Path, required=True)
    parser.add_argument("--operational-root", type=Path, action="append", required=True)
    parser.add_argument("--service-selector", type=Path, action="append", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = validate_retirement(
            manifest_path=args.manifest,
            expected_manifest_sha256=args.expected_manifest_sha256,
            cold_start_path=args.cold_start,
            operational_roots=args.operational_root,
            service_selector_files=args.service_selector,
        )
    except RetirementValidationError as error:
        print(f"retirement validation failed: {error}", file=sys.stderr)
        return 2
    sys.stdout.buffer.write(_canonical_bytes(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
