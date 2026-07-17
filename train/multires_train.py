"""Continuous evidence-owning loop for the fresh multires trainer.

This module composes an already-constructed :class:`MultiresLiveTrainer`; it
does not launch clients, select a checkpoint, or integrate with a service.
Every successful policy version produces one attested checkpoint, immutable
rollout/update evidence, a current evidence-only season report, and scalar
telemetry.  There is no stage promotion or legacy fallback here.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Callable, Mapping, Protocol

import numpy as np

from harness.causal_protocol import CausalTelemetry
from harness.multires_admission import (
    PRIVATE_CAUSAL_INFO_KEY,
    PRIVATE_SPATIAL_REWARD_INFO_KEY,
    PrivateSpatialRewardEvidence,
)
from harness.multires_metrics import MultiresSeasonMetrics, SEASON_SCHEMA
from harness.multires_reward import CausalRewardFrame, CausalRewardResult


TRAINING_CORE_SCHEMA = "q2-multires-continuous-training-v1"
CURRICULUM_STAGE_SCHEMA = "q2-multires-b7-curriculum-stage-v1"
CURRICULUM_GATE_SCHEMA = "q2-multires-b7-predecessor-gate-v1"
ROLLOUT_EVIDENCE_SCHEMA = "q2-multires-rollout-evidence-v1"
UPDATE_EVIDENCE_SCHEMA = "q2-multires-update-evidence-v1"
SEASON_REPORT_SCHEMA = "q2-multires-current-season-v1"

CURRICULUM_STAGE_NAMES = {
    1: "transport-posture-water-death-screen-echo",
    2: "standing-crouched-traversal",
    3: "hazard-avoidance-walking-recovery",
    4: "hook-assisted-recovery-controlled-drops",
    5: "pickups-guide-dropout",
    6: "generated-map-combat",
}

_SHA256_CHARS = frozenset("0123456789abcdef")
_REQUIRED_NETWORK_METRIC_TAGS = frozenset({
    "network_client/rounds_dispatched",
    "network_client/rounds_accepted",
    "network_client/failed_rounds",
    "network_client/actions_dispatched",
    "network_client/transitions_accepted",
    "network_client/stale_policy_rounds_rejected",
    "network_client/stale_echoes_rejected",
    "network_client/mismatched_echoes_rejected",
    "network_client/echo_timeouts",
    "network_client/map_epoch_resyncs",
    "network_client/telemetry_gap_resyncs",
    "network_client/realtime_catchup_resyncs",
    "network_client/action_state_resyncs",
    "network_client/preflight_packets_drained",
    "network_client/max_observed_frame_span",
    "network_client/fire_gate_suppressions",
    "network_client/authoritative_echo_accept_rate",
})


class MultiresTrainingCoreError(RuntimeError):
    """Raised when continuous training cannot emit truthful evidence."""


class ScalarWriter(Protocol):
    def add_scalar(self, tag: str, scalar_value: float, global_step: int) -> Any:
        ...


class CausalMetricsSnapshotter(Protocol):
    def snapshot(self, *, policy_end_version: int) -> Mapping[str, Any]:
        ...


class SanitizedCausalMetricsAccumulator:
    """Observer/report seam backed by the frozen B5 season aggregator.

    The collector calls this object only after a complete rollout validates.
    This adapter accepts no observation vector and recursively rejects both
    privileged conduit types, so it cannot become a second teacher channel.
    Missing optional public coverage is reported explicitly rather than
    silently claimed as measured telemetry.
    """

    def __init__(self, metrics: MultiresSeasonMetrics):
        if not isinstance(metrics, MultiresSeasonMetrics):
            raise TypeError("causal accumulator requires MultiresSeasonMetrics")
        self.metrics = metrics
        self._coverage = {
            "movement_command_samples": 0,
            "movement_speed_samples": 0,
            "true_view_pitch_samples": 0,
            "guide_dropout_samples": 0,
        }

    @staticmethod
    def _reject_private(value: Any, label: str) -> None:
        if isinstance(value, (CausalTelemetry, PrivateSpatialRewardEvidence)):
            raise MultiresTrainingCoreError(
                f"causal metrics received private payload at {label}"
            )
        if isinstance(value, Mapping):
            for name, item in value.items():
                SanitizedCausalMetricsAccumulator._reject_private(
                    item, f"{label}.{name}"
                )
        elif isinstance(value, (tuple, list)):
            for index, item in enumerate(value):
                SanitizedCausalMetricsAccumulator._reject_private(
                    item, f"{label}[{index}]"
                )

    @staticmethod
    def _finite(value: Any, label: str) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise MultiresTrainingCoreError(f"{label} must be numeric")
        result = float(value)
        if not math.isfinite(result):
            raise MultiresTrainingCoreError(f"{label} must be finite")
        return result

    def __call__(
        self,
        client_id: str,
        frame: CausalRewardFrame,
        result: CausalRewardResult,
        info: Mapping[str, Any],
    ) -> None:
        if not isinstance(client_id, str) or not client_id:
            raise MultiresTrainingCoreError("causal metrics client_id is invalid")
        if not isinstance(frame, CausalRewardFrame) or not isinstance(
            result, CausalRewardResult
        ):
            raise MultiresTrainingCoreError(
                "causal metrics require detached reward frame/result"
            )
        if not isinstance(info, Mapping):
            raise MultiresTrainingCoreError("causal metrics public info is invalid")
        if (
            PRIVATE_CAUSAL_INFO_KEY in info
            or PRIVATE_SPATIAL_REWARD_INFO_KEY in info
        ):
            raise MultiresTrainingCoreError(
                "causal metrics received private conduit keys"
            )
        self._reject_private(info, "public_info")
        if info.get("client_id") != client_id:
            raise MultiresTrainingCoreError(
                "causal metrics client/public identity differs"
            )

        forward_command = 0.0
        backward_command = 0.0
        movement = info.get("action_debug_movement")
        if movement is not None:
            if not isinstance(movement, (tuple, list)) or len(movement) != 4:
                raise MultiresTrainingCoreError(
                    "public action_debug_movement must contain four values"
                )
            forward_axis = self._finite(
                movement[0], "public action_debug_movement.forward"
            )
            for index, value in enumerate(movement[1:], start=1):
                self._finite(value, f"public action_debug_movement[{index}]")
            forward_command = max(0.0, forward_axis)
            backward_command = max(0.0, -forward_axis)
            self._coverage["movement_command_samples"] += 1

        movement_speed = 0.0
        if "movement_speed" in info:
            movement_speed = self._finite(
                info["movement_speed"], "public movement_speed"
            )
            if movement_speed < 0.0:
                raise MultiresTrainingCoreError(
                    "public movement_speed cannot be negative"
                )
            self._coverage["movement_speed_samples"] += 1

        true_view_pitch = None
        if "true_view_pitch_deg" in info:
            true_view_pitch = self._finite(
                info["true_view_pitch_deg"], "public true_view_pitch_deg"
            )
            self._coverage["true_view_pitch_samples"] += 1

        guide_dropped = info.get(
            "guide_dropped", (False, False, False, False)
        )
        guide_classes = info.get("guide_classes", (None, None, None, None))
        global_guide_drop = info.get("global_guide_drop", False)
        if any(
            name in info
            for name in ("guide_dropped", "guide_classes", "global_guide_drop")
        ):
            if (
                not isinstance(guide_dropped, (tuple, list))
                or len(guide_dropped) != 4
                or any(type(value) is not bool for value in guide_dropped)
                or not isinstance(guide_classes, (tuple, list))
                or len(guide_classes) != 4
                or type(global_guide_drop) is not bool
            ):
                raise MultiresTrainingCoreError(
                    "public guide-dropout telemetry is malformed"
                )
            self._coverage["guide_dropout_samples"] += 1

        self.metrics.observe(
            frame,
            result,
            command_echo_match=bool(frame.authoritative_echo_valid),
            state_resync=bool(frame.state_resync),
            guide_dropped=tuple(guide_dropped),
            guide_classes=tuple(guide_classes),
            global_guide_drop=bool(global_guide_drop),
            teacher_field_violations=int(frame.teacher_field_violations),
            forward_command=forward_command,
            backward_command=backward_command,
            movement_speed=movement_speed,
            true_view_pitch_deg=true_view_pitch,
        )

    def snapshot(self, *, policy_end_version: int) -> dict[str, Any]:
        report = self.metrics.report(policy_end_version=policy_end_version)
        transitions = int(report["accepted_transitions"])
        return _normalize_json({
            **report,
            "observer_coverage": {
                **self._coverage,
                "missing_movement_command_samples": (
                    transitions - self._coverage["movement_command_samples"]
                ),
                "missing_movement_speed_samples": (
                    transitions - self._coverage["movement_speed_samples"]
                ),
                "missing_true_view_pitch_samples": (
                    transitions - self._coverage["true_view_pitch_samples"]
                ),
                "missing_guide_dropout_samples": (
                    transitions - self._coverage["guide_dropout_samples"]
                ),
            },
            "private_causal_payload_serialized": False,
        }, "causal_metrics_snapshot")

    def observe_runtime_snapshot(self, **public_runtime_metrics: Any) -> None:
        """Add the public Atlas/Dyn resident-query snapshot to this window."""
        self._reject_private(public_runtime_metrics, "public_runtime_metrics")
        self.metrics.observe_runtime_snapshot(**public_runtime_metrics)


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in _SHA256_CHARS for character in value)
        and value != "0" * 64
    )


def _normalize_json(value: Any, label: str) -> Any:
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) and key for key in value):
            raise MultiresTrainingCoreError(
                f"{label} mapping keys must be nonempty strings"
            )
        return {
            key: _normalize_json(value[key], f"{label}.{key}")
            for key in sorted(value)
        }
    if isinstance(value, (tuple, list)):
        return [
            _normalize_json(item, f"{label}[{index}]")
            for index, item in enumerate(value)
        ]
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise MultiresTrainingCoreError(f"{label} contains NaN or infinity")
        return value
    raise MultiresTrainingCoreError(
        f"{label} contains unsupported value {type(value).__name__}"
    )


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise MultiresTrainingCoreError("evidence is not canonical JSON") from error


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _seal_evidence(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = _normalize_json(value, "evidence")
    if "evidence_sha256" in payload:
        raise MultiresTrainingCoreError("unsealed evidence contains evidence_sha256")
    return {**payload, "evidence_sha256": _sha256_json(payload)}


def _validate_sealed_evidence(
    value: Mapping[str, Any], *, schema: str, label: str
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise MultiresTrainingCoreError(f"{label} must be a JSON object")
    payload = _normalize_json(value, label)
    if payload.get("schema") != schema:
        raise MultiresTrainingCoreError(f"{label} schema differs")
    digest = payload.pop("evidence_sha256", None)
    if not _valid_sha256(digest) or digest != _sha256_json(payload):
        raise MultiresTrainingCoreError(f"{label} evidence digest differs")
    return {**payload, "evidence_sha256": digest}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ensure_directory(path: Path) -> None:
    if path.is_symlink():
        raise MultiresTrainingCoreError(f"output directory is a symlink: {path}")
    path.mkdir(parents=True, exist_ok=True)
    if not path.is_dir() or path.is_symlink():
        raise MultiresTrainingCoreError(f"output directory is invalid: {path}")


def _atomic_json(path: Path, payload: Mapping[str, Any], *, replace: bool) -> None:
    destination = Path(path)
    _ensure_directory(destination.parent)
    if destination.is_symlink():
        raise MultiresTrainingCoreError(
            f"evidence destination is a symlink: {destination}"
        )
    encoded = _canonical_bytes(payload) + b"\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        if replace:
            os.replace(temporary, destination)
        else:
            try:
                os.link(temporary, destination)
            except FileExistsError as error:
                raise MultiresTrainingCoreError(
                    f"immutable evidence already exists: {destination}"
                ) from error
            temporary.unlink()
        directory_descriptor = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def _publish_content_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Publish immutable digest-named JSON, accepting only identical replay."""
    destination = Path(path)
    if destination.is_symlink():
        raise MultiresTrainingCoreError(
            f"content evidence symlink rejected: {destination}"
        )
    if destination.exists():
        try:
            existing = json.loads(destination.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise MultiresTrainingCoreError(
                f"existing content evidence is unreadable: {destination}"
            ) from error
        if existing != payload:
            raise MultiresTrainingCoreError(
                f"content-address evidence collision differs: {destination}"
            )
        return
    _atomic_json(destination, payload, replace=False)


@dataclass(frozen=True)
class CurriculumStage:
    schema: str
    stage_id: int
    stage_name: str
    configuration_json: str
    configuration_sha256: str
    predecessor_stage_sha256: str | None

    @classmethod
    def create(
        cls,
        stage_id: int,
        configuration: Mapping[str, Any],
        *,
        predecessor_stage_sha256: str | None = None,
    ) -> "CurriculumStage":
        if type(stage_id) is not int or stage_id not in CURRICULUM_STAGE_NAMES:
            raise MultiresTrainingCoreError("curriculum stage must be an integer 1..6")
        if not isinstance(configuration, Mapping) or not configuration:
            raise MultiresTrainingCoreError(
                "curriculum stage requires a nonempty immutable configuration"
            )
        if stage_id == 1:
            if predecessor_stage_sha256 is not None:
                raise MultiresTrainingCoreError(
                    "curriculum stage 1 cannot declare predecessor evidence"
                )
        elif not _valid_sha256(predecessor_stage_sha256):
            raise MultiresTrainingCoreError(
                "curriculum stage >1 requires the predecessor stage digest"
            )
        normalized = _normalize_json(configuration, "stage.configuration")
        envelope = {
            "schema": CURRICULUM_STAGE_SCHEMA,
            "stage_id": stage_id,
            "stage_name": CURRICULUM_STAGE_NAMES[stage_id],
            "configuration": normalized,
            "predecessor_stage_sha256": predecessor_stage_sha256,
        }
        configuration_json = _canonical_bytes(envelope).decode("utf-8")
        return cls(
            schema=CURRICULUM_STAGE_SCHEMA,
            stage_id=stage_id,
            stage_name=CURRICULUM_STAGE_NAMES[stage_id],
            configuration_json=configuration_json,
            configuration_sha256=hashlib.sha256(
                configuration_json.encode("utf-8")
            ).hexdigest(),
            predecessor_stage_sha256=predecessor_stage_sha256,
        )

    def validate(self) -> None:
        try:
            decoded = json.loads(self.configuration_json)
        except (TypeError, json.JSONDecodeError) as error:
            raise MultiresTrainingCoreError(
                "curriculum stage configuration JSON is invalid"
            ) from error
        rebuilt = CurriculumStage.create(
            self.stage_id,
            decoded.get("configuration", {}),
            predecessor_stage_sha256=self.predecessor_stage_sha256,
        )
        if self != rebuilt:
            raise MultiresTrainingCoreError(
                "curriculum stage identity/configuration digest differs"
            )

    def to_mapping(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema": self.schema,
            "stage_id": self.stage_id,
            "stage_name": self.stage_name,
            "configuration_json": self.configuration_json,
            "configuration_sha256": self.configuration_sha256,
            "predecessor_stage_sha256": self.predecessor_stage_sha256,
        }


def create_curriculum_gate_evidence(
    stage: CurriculumStage,
    *,
    decision: str,
    runtime_manifest_sha256: str,
    lineage_root_sha256: str,
    accepted_transitions: int,
    policy_updates: int,
    optimizer_steps: int,
) -> dict[str, Any]:
    """Seal an explicit evaluator decision; this performs no promotion."""
    stage.validate()
    if decision not in ("passed", "failed"):
        raise MultiresTrainingCoreError("curriculum gate decision is invalid")
    if not _valid_sha256(runtime_manifest_sha256):
        raise MultiresTrainingCoreError("curriculum gate runtime digest is invalid")
    if not _valid_sha256(lineage_root_sha256):
        raise MultiresTrainingCoreError("curriculum gate lineage digest is invalid")
    if any(
        type(value) is not int or value < 0
        for value in (accepted_transitions, policy_updates, optimizer_steps)
    ):
        raise MultiresTrainingCoreError("curriculum gate counters are invalid")
    if decision == "passed" and (
        accepted_transitions < 1 or policy_updates < 1 or optimizer_steps < 1
    ):
        raise MultiresTrainingCoreError(
            "a passed curriculum gate requires positive training evidence"
        )
    return _seal_evidence({
        "schema": CURRICULUM_GATE_SCHEMA,
        "decision": decision,
        "stage_id": stage.stage_id,
        "stage_name": stage.stage_name,
        "stage_configuration_sha256": stage.configuration_sha256,
        "runtime_manifest_sha256": runtime_manifest_sha256,
        "lineage_root_sha256": lineage_root_sha256,
        "accepted_transitions": accepted_transitions,
        "policy_updates": policy_updates,
        "optimizer_steps": optimizer_steps,
        "automatic_promotion": False,
    })


def validate_predecessor_gate(
    stage: CurriculumStage,
    evidence: Mapping[str, Any] | None,
    *,
    runtime_manifest_sha256: str,
    lineage_root_sha256: str | None,
) -> dict[str, Any] | None:
    stage.validate()
    if stage.stage_id == 1:
        if evidence is not None:
            raise MultiresTrainingCoreError(
                "curriculum stage 1 rejects predecessor gate evidence"
            )
        return None
    if evidence is None:
        raise MultiresTrainingCoreError(
            "curriculum stage >1 requires a passed predecessor gate"
        )
    if not _valid_sha256(lineage_root_sha256):
        raise MultiresTrainingCoreError(
            "curriculum continuation requires an existing lineage root"
        )
    gate = _validate_sealed_evidence(
        evidence, schema=CURRICULUM_GATE_SCHEMA, label="predecessor gate"
    )
    expected = {
        "decision": "passed",
        "stage_id": stage.stage_id - 1,
        "stage_name": CURRICULUM_STAGE_NAMES[stage.stage_id - 1],
        "stage_configuration_sha256": stage.predecessor_stage_sha256,
        "runtime_manifest_sha256": runtime_manifest_sha256,
        "lineage_root_sha256": lineage_root_sha256,
        "automatic_promotion": False,
    }
    mismatches = {
        name: (gate.get(name), wanted)
        for name, wanted in expected.items()
        if gate.get(name) != wanted
    }
    if (
        type(gate.get("accepted_transitions")) is not int
        or gate["accepted_transitions"] < 1
        or type(gate.get("policy_updates")) is not int
        or gate["policy_updates"] < 1
        or type(gate.get("optimizer_steps")) is not int
        or gate["optimizer_steps"] < gate["policy_updates"]
    ):
        mismatches["training_evidence"] = (
            (
                gate.get("accepted_transitions"), gate.get("policy_updates"),
                gate.get("optimizer_steps"),
            ),
            "positive counters",
        )
    if mismatches:
        raise MultiresTrainingCoreError(
            f"predecessor curriculum gate differs: {mismatches!r}"
        )
    return gate


@dataclass(frozen=True)
class TrainingProgress:
    accepted_transitions: int = 0
    policy_updates: int = 0
    optimizer_steps: int = 0
    next_policy_version: int = 0
    lineage_root_sha256: str | None = None

    def validate(self) -> None:
        for name in (
            "accepted_transitions", "policy_updates", "optimizer_steps",
            "next_policy_version",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise MultiresTrainingCoreError(f"progress {name} is invalid")
        if self.accepted_transitions < self.policy_updates:
            raise MultiresTrainingCoreError(
                "accepted-transition progress cannot trail policy updates"
            )
        if self.optimizer_steps < self.policy_updates:
            raise MultiresTrainingCoreError(
                "optimizer-step progress cannot trail policy updates"
            )
        if self.next_policy_version != self.policy_updates:
            raise MultiresTrainingCoreError(
                "next policy version must equal one-update-per-version progress"
            )
        if self.policy_updates > 0 and not _valid_sha256(
            self.lineage_root_sha256
        ):
            raise MultiresTrainingCoreError(
                "nonzero progress requires an attested lineage root"
            )
        if self.lineage_root_sha256 is not None and not _valid_sha256(
            self.lineage_root_sha256
        ):
            raise MultiresTrainingCoreError("progress lineage root is invalid")


def progress_from_season_report(
    report: Mapping[str, Any],
    stage: CurriculumStage,
    *,
    runtime_manifest_sha256: str,
    atlas_sha256: str,
    lineage_root_sha256: str,
) -> TrainingProgress:
    """Recover counters only from the exact current report/checkpoint identity."""
    current = _validate_sealed_evidence(
        report, schema=SEASON_REPORT_SCHEMA, label="current season report"
    )
    stage.validate()
    expected = {
        "health": "training-active",
        "promotion_claim": False,
        "stage_configuration_sha256": stage.configuration_sha256,
        "runtime_manifest_sha256": runtime_manifest_sha256,
        "atlas_sha256": atlas_sha256,
        "lineage_root_sha256": lineage_root_sha256,
    }
    mismatches = {
        name: (current.get(name), wanted)
        for name, wanted in expected.items()
        if current.get(name) != wanted
    }
    counters = current.get("counters")
    manifest = current.get("checkpoint_manifest")
    if not isinstance(counters, Mapping) or not isinstance(manifest, Mapping):
        raise MultiresTrainingCoreError(
            "current season report lacks counters/checkpoint manifest"
        )
    accepted = counters.get("accepted_transitions")
    updates = counters.get("policy_updates")
    optimizer_steps = counters.get("optimizer_steps")
    next_version = counters.get("next_policy_version")
    if (
        type(accepted) is not int
        or accepted < 1
        or type(updates) is not int
        or updates < 1
        or type(optimizer_steps) is not int
        or optimizer_steps < updates
        or type(next_version) is not int
        or next_version < 1
    ):
        raise MultiresTrainingCoreError(
            "current season report cannot resume zero/invalid progress"
        )
    if manifest.get("training_step") != accepted:
        mismatches["checkpoint_training_step"] = (
            manifest.get("training_step"), accepted,
        )
    for name, wanted in (
        ("runtime_manifest_sha256", runtime_manifest_sha256),
        ("atlas_sha256", atlas_sha256),
        ("lineage_root_sha256", lineage_root_sha256),
    ):
        if manifest.get(name) != wanted:
            mismatches[f"checkpoint_{name}"] = (manifest.get(name), wanted)
    if mismatches:
        raise MultiresTrainingCoreError(
            f"current season resume evidence differs: {mismatches!r}"
        )
    result = TrainingProgress(
        accepted_transitions=accepted,
        policy_updates=updates,
        optimizer_steps=optimizer_steps,
        next_policy_version=next_version,
        lineage_root_sha256=lineage_root_sha256,
    )
    result.validate()
    return result


@dataclass(frozen=True)
class TrainingRunSummary:
    accepted_transitions: int
    policy_updates: int
    optimizer_steps: int
    next_policy_version: int
    lineage_root_sha256: str
    current_season_report: Path


class MultiresContinuousTrainingCore:
    """Finite or continuous update loop around one live multires trainer."""

    def __init__(
        self,
        trainer: Any,
        *,
        output_root: Path,
        stage: CurriculumStage,
        writer: ScalarWriter,
        causal_metrics: CausalMetricsSnapshotter,
        predecessor_gate: Mapping[str, Any] | None = None,
        progress: TrainingProgress = TrainingProgress(),
    ):
        if not callable(getattr(trainer, "train_update", None)):
            raise TypeError("training core requires a MultiresLiveTrainer")
        if not callable(getattr(writer, "add_scalar", None)):
            raise TypeError("training core requires an injected scalar writer")
        if not callable(getattr(causal_metrics, "snapshot", None)):
            raise TypeError(
                "training core requires a sanitized causal-metrics snapshotter"
            )
        runtime_owner = getattr(trainer, "runtime", None)
        runtime = getattr(runtime_owner, "runtime", None)
        if runtime is None or not callable(getattr(runtime_owner, "checkpoint", None)):
            raise TypeError("live trainer lacks its attested runtime/checkpoint seam")
        if not hasattr(trainer, "optimizer"):
            raise TypeError("live trainer does not expose its optimizer")
        runtime_manifest_sha256 = getattr(
            runtime, "runtime_manifest_sha256", None
        )
        atlas_sha256 = getattr(runtime, "atlas_sha256", None)
        if not _valid_sha256(runtime_manifest_sha256) or not _valid_sha256(
            atlas_sha256
        ):
            raise MultiresTrainingCoreError(
                "training core requires attested runtime and Atlas digests"
            )
        root = Path(output_root).expanduser()
        if not root.is_absolute():
            raise MultiresTrainingCoreError("training output root must be absolute")
        _ensure_directory(root)
        for child in ("checkpoints", "evidence/rollouts", "evidence/updates", "season"):
            _ensure_directory(root / child)
        stage.validate()
        progress.validate()
        runtime_lineage = getattr(runtime_owner, "lineage_root_sha256", None)
        if progress.lineage_root_sha256 is not None and (
            progress.lineage_root_sha256 != runtime_lineage
        ):
            raise MultiresTrainingCoreError(
                "resume progress lineage differs from loaded trainer runtime"
            )
        validated_gate = validate_predecessor_gate(
            stage,
            predecessor_gate,
            runtime_manifest_sha256=runtime_manifest_sha256,
            lineage_root_sha256=runtime_lineage,
        )
        if validated_gate is not None:
            expected_progress = TrainingProgress(
                accepted_transitions=validated_gate["accepted_transitions"],
                policy_updates=validated_gate["policy_updates"],
                optimizer_steps=validated_gate["optimizer_steps"],
                next_policy_version=validated_gate["policy_updates"],
                lineage_root_sha256=validated_gate["lineage_root_sha256"],
            )
            if progress != expected_progress:
                raise MultiresTrainingCoreError(
                    "stage continuation progress differs from predecessor gate"
                )
        current_report = root / "season/current.json"
        if current_report.exists():
            if progress.policy_updates < 1:
                raise MultiresTrainingCoreError(
                    "existing season report requires explicit resume progress"
                )
            decoded = json.loads(current_report.read_text(encoding="utf-8"))
            restored = progress_from_season_report(
                decoded,
                stage,
                runtime_manifest_sha256=runtime_manifest_sha256,
                atlas_sha256=atlas_sha256,
                lineage_root_sha256=str(runtime_lineage),
            )
            if restored != progress:
                raise MultiresTrainingCoreError(
                    "explicit progress differs from current season report"
                )

        self.trainer = trainer
        self.output_root = root
        self.stage = stage
        self.writer = writer
        self.causal_metrics = causal_metrics
        self.runtime_manifest_sha256 = str(runtime_manifest_sha256)
        self.atlas_sha256 = str(atlas_sha256)
        self.accepted_transitions = progress.accepted_transitions
        self.policy_updates = progress.policy_updates
        self.optimizer_steps = progress.optimizer_steps
        self.next_policy_version = progress.next_policy_version
        self._lineage_root_sha256 = runtime_lineage
        self._failed = False
        initial_metrics = self._causal_snapshot(
            policy_end_version=self.next_policy_version
        )
        self._causal_metric_transitions = int(
            initial_metrics["accepted_transitions"]
        )
        initial_network_metrics = self._network_metrics_snapshot()
        self._network_metric_transitions = int(
            initial_network_metrics["network_client/transitions_accepted"]
        )

    @property
    def current_season_report(self) -> Path:
        return self.output_root / "season/current.json"

    @staticmethod
    def _ppo_metrics(value: Mapping[str, Any]) -> tuple[dict[str, float], int]:
        if not isinstance(value, Mapping) or not value:
            raise MultiresTrainingCoreError("PPO update metrics are missing")
        raw_steps = value.get("optimizer_steps")
        if type(raw_steps) is not int or raw_steps < 1:
            raise MultiresTrainingCoreError(
                "PPO update must report positive exact optimizer_steps"
            )
        metrics: dict[str, float] = {}
        for name, raw in sorted(value.items()):
            if name == "optimizer_steps":
                continue
            if not isinstance(name, str) or not name:
                raise MultiresTrainingCoreError("PPO metric name is invalid")
            if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                raise MultiresTrainingCoreError(f"PPO metric {name} is not numeric")
            number = float(raw)
            if not math.isfinite(number):
                raise MultiresTrainingCoreError(f"PPO metric {name} is non-finite")
            metrics[name] = number
        return metrics, raw_steps

    @staticmethod
    def _manifest_mapping(manifest: Any) -> dict[str, Any]:
        if callable(getattr(manifest, "to_mapping", None)):
            value = manifest.to_mapping()
        elif isinstance(manifest, Mapping):
            value = dict(manifest)
        else:
            fields = (
                "training_step", "lineage_root_sha256", "atlas_sha256",
                "runtime_manifest_sha256",
            )
            value = {name: getattr(manifest, name, None) for name in fields}
        if not isinstance(value, Mapping):
            raise MultiresTrainingCoreError("checkpoint manifest is malformed")
        return _normalize_json(value, "checkpoint_manifest")

    def _causal_snapshot(self, *, policy_end_version: int) -> dict[str, Any]:
        try:
            value = self.causal_metrics.snapshot(
                policy_end_version=policy_end_version
            )
        except Exception as error:
            raise MultiresTrainingCoreError(
                "sanitized causal-metrics snapshot failed"
            ) from error
        if not isinstance(value, Mapping):
            raise MultiresTrainingCoreError(
                "sanitized causal-metrics snapshot is malformed"
            )
        snapshot = _normalize_json(value, "causal_metrics_snapshot")
        expected = {
            "schema": SEASON_SCHEMA,
            "atlas_sha256": self.atlas_sha256,
            "policy_end_version": policy_end_version,
            "private_causal_payload_serialized": False,
        }
        mismatches = {
            name: (snapshot.get(name), wanted)
            for name, wanted in expected.items()
            if snapshot.get(name) != wanted
        }
        transitions = snapshot.get("accepted_transitions")
        if type(transitions) is not int or transitions < 0:
            mismatches["accepted_transitions"] = (
                transitions, "nonnegative integer",
            )
        privilege = snapshot.get("privilege")
        if (
            not isinstance(privilege, Mapping)
            or privilege.get("teacher_field_violations") != 0
        ):
            mismatches["teacher_field_violations"] = (
                privilege, "exactly zero",
            )
        if mismatches:
            raise MultiresTrainingCoreError(
                f"sanitized causal-metrics snapshot differs: {mismatches!r}"
            )
        return snapshot

    def _network_metrics_snapshot(self) -> dict[str, float | int]:
        collector = getattr(self.trainer, "collector", None)
        batch = getattr(collector, "batch", None)
        metrics = getattr(batch, "metrics", None)
        as_dict = getattr(metrics, "as_dict", None)
        if not callable(as_dict):
            raise MultiresTrainingCoreError(
                "live trainer lacks network-client metrics"
            )
        try:
            value = as_dict()
        except Exception as error:
            raise MultiresTrainingCoreError(
                "network-client metrics snapshot failed"
            ) from error
        if not isinstance(value, Mapping):
            raise MultiresTrainingCoreError(
                "network-client metrics snapshot is malformed"
            )
        missing = sorted(_REQUIRED_NETWORK_METRIC_TAGS - set(value))
        if missing:
            raise MultiresTrainingCoreError(
                f"network-client metrics omit required tags: {missing!r}"
            )
        result: dict[str, float | int] = {}
        for tag in sorted(_REQUIRED_NETWORK_METRIC_TAGS):
            raw = value[tag]
            if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                raise MultiresTrainingCoreError(
                    f"network-client metric {tag} is not numeric"
                )
            number = float(raw)
            if not math.isfinite(number) or number < 0.0:
                raise MultiresTrainingCoreError(
                    f"network-client metric {tag} is invalid"
                )
            result[tag] = int(raw) if type(raw) is int else number
        return result

    @staticmethod
    def _numeric_metric_tags(
        value: Mapping[str, Any], *, prefix: str = "season"
    ) -> dict[str, float]:
        result: dict[str, float] = {}

        def visit(item: Any, path: str) -> None:
            if isinstance(item, Mapping):
                for name in sorted(item):
                    visit(item[name], f"{path}/{name}")
            elif isinstance(item, bool):
                result[path] = float(item)
            elif isinstance(item, (int, float)):
                number = float(item)
                if not math.isfinite(number):
                    raise MultiresTrainingCoreError(
                        f"causal metric {path} is non-finite"
                    )
                result[path] = number

        visit(value, prefix)
        return result

    def _emit_scalars(
        self,
        *,
        policy_version: int,
        accepted_this_update: int,
        reward_mean: float,
        boundary_rounds: int,
        ppo_metrics: Mapping[str, float],
        causal_metrics: Mapping[str, Any],
        network_metrics: Mapping[str, float | int],
        cumulative_transitions: int,
        cumulative_policy_updates: int,
        cumulative_optimizer_steps: int,
    ) -> None:
        values = {
            "train/accepted_transitions": float(cumulative_transitions),
            "train/accepted_transitions_update": float(accepted_this_update),
            "train/policy_updates": float(cumulative_policy_updates),
            "train/optimizer_steps": float(cumulative_optimizer_steps),
            "train/policy_version": float(policy_version),
            "rollout/reward_mean": float(reward_mean),
            "rollout/boundary_rounds": float(boundary_rounds),
            "curriculum/stage_id": float(self.stage.stage_id),
        }
        values.update({f"train/{name}": value for name, value in ppo_metrics.items()})
        values.update(self._numeric_metric_tags(causal_metrics))
        values.update({name: float(value) for name, value in network_metrics.items()})
        for tag, value in values.items():
            self.writer.add_scalar(tag, value, cumulative_transitions)
        flush = getattr(self.writer, "flush", None)
        if callable(flush):
            flush()

    def train_one_update(self) -> dict[str, Any]:
        if self._failed:
            raise MultiresTrainingCoreError(
                "training core is failed; construct a new attested process"
            )
        policy_version = self.next_policy_version
        checkpoint_staging: Path | None = None
        try:
            update = self.trainer.train_update(policy_version=policy_version)
            rollout = getattr(update, "rollout", None)
            if rollout is None or not callable(getattr(rollout, "validate", None)):
                raise MultiresTrainingCoreError("train_update returned no rollout")
            rollout.validate()
            if getattr(rollout, "policy_version", None) != policy_version:
                raise MultiresTrainingCoreError(
                    "train_update rollout policy version differs"
                )
            valid = np.asarray(getattr(rollout, "valid", None))
            if valid.dtype != np.bool_ or valid.size < 1 or not bool(valid.all()):
                raise MultiresTrainingCoreError(
                    "training update contains zero or non-admitted transitions"
                )
            accepted_this_update = int(valid.sum())
            if accepted_this_update < 1:
                raise MultiresTrainingCoreError(
                    "training update contains zero accepted transitions"
                )
            rewards = np.asarray(getattr(rollout, "rewards", None), dtype=np.float64)
            if rewards.shape != valid.shape or not np.isfinite(rewards).all():
                raise MultiresTrainingCoreError("rollout rewards are invalid")
            rollout_digest = rollout.deterministic_sha256()
            if not _valid_sha256(rollout_digest):
                raise MultiresTrainingCoreError("rollout digest is invalid")
            ppo_metrics, optimizer_steps_this_update = self._ppo_metrics(
                getattr(update, "ppo_metrics", None)
            )
            causal_metrics = self._causal_snapshot(
                policy_end_version=policy_version + 1
            )
            causal_metric_transitions = int(
                causal_metrics["accepted_transitions"]
            )
            expected_metric_transitions = (
                self._causal_metric_transitions + accepted_this_update
            )
            if causal_metric_transitions != expected_metric_transitions:
                raise MultiresTrainingCoreError(
                    "causal-metrics transition delta differs from admitted rollout: "
                    f"{causal_metric_transitions} != {expected_metric_transitions}"
                )
            causal_metrics_sha256 = _sha256_json(causal_metrics)
            network_metrics = self._network_metrics_snapshot()
            network_metric_transitions = int(
                network_metrics["network_client/transitions_accepted"]
            )
            expected_network_transitions = (
                self._network_metric_transitions + accepted_this_update
            )
            if network_metric_transitions != expected_network_transitions:
                raise MultiresTrainingCoreError(
                    "network-client transition delta differs from admitted rollout: "
                    f"{network_metric_transitions} != "
                    f"{expected_network_transitions}"
                )
            network_metrics_sha256 = _sha256_json(network_metrics)
            cumulative_transitions = (
                self.accepted_transitions + accepted_this_update
            )
            cumulative_policy_updates = self.policy_updates + 1
            cumulative_optimizer_steps = (
                self.optimizer_steps + optimizer_steps_this_update
            )
            checkpoint_staging = self.output_root / "checkpoints" / (
                f".checkpoint-{cumulative_transitions:012d}-{os.getpid()}.pending"
            )
            if checkpoint_staging.exists() or checkpoint_staging.is_symlink():
                raise MultiresTrainingCoreError(
                    f"checkpoint staging destination exists: {checkpoint_staging}"
                )
            previous_lineage = self._lineage_root_sha256
            manifest = self.trainer.runtime.checkpoint(
                checkpoint_staging,
                training_step=cumulative_transitions,
                optimizer=self.trainer.optimizer,
            )
            if not checkpoint_staging.is_file() or checkpoint_staging.is_symlink():
                raise MultiresTrainingCoreError(
                    "runtime checkpoint did not atomically publish a regular file"
                )
            manifest_mapping = self._manifest_mapping(manifest)
            lineage_root = manifest_mapping.get("lineage_root_sha256")
            expected_manifest = {
                "training_step": cumulative_transitions,
                "atlas_sha256": self.atlas_sha256,
                "runtime_manifest_sha256": self.runtime_manifest_sha256,
            }
            mismatches = {
                name: (manifest_mapping.get(name), wanted)
                for name, wanted in expected_manifest.items()
                if manifest_mapping.get(name) != wanted
            }
            if not _valid_sha256(lineage_root):
                mismatches["lineage_root_sha256"] = (
                    lineage_root, "lowercase SHA-256",
                )
            if previous_lineage is not None and lineage_root != previous_lineage:
                mismatches["preserved_lineage_root_sha256"] = (
                    lineage_root, previous_lineage,
                )
            if getattr(
                self.trainer.runtime, "lineage_root_sha256", None
            ) != lineage_root:
                mismatches["runtime_lineage_root_sha256"] = (
                    getattr(self.trainer.runtime, "lineage_root_sha256", None),
                    lineage_root,
                )
            if mismatches:
                raise MultiresTrainingCoreError(
                    f"checkpoint manifest differs: {mismatches!r}"
                )
            checkpoint_sha256 = _file_sha256(checkpoint_staging)
            checkpoint = self.output_root / "checkpoints" / (
                f"checkpoint-{cumulative_transitions:012d}-"
                f"{checkpoint_sha256[:16]}.pt"
            )
            if checkpoint.is_symlink():
                raise MultiresTrainingCoreError(
                    "content-addressed checkpoint destination is a symlink"
                )
            if checkpoint.exists():
                if not checkpoint.is_file() or _file_sha256(checkpoint) != checkpoint_sha256:
                    raise MultiresTrainingCoreError(
                        "content-addressed checkpoint collision differs"
                    )
                checkpoint_staging.unlink()
            else:
                os.replace(checkpoint_staging, checkpoint)
            reward_mean = float(rewards.mean())
            boundary_rounds = int(getattr(rollout, "boundary_rounds", 0))
            if boundary_rounds < 0:
                raise MultiresTrainingCoreError("rollout boundary count is invalid")

            rollout_evidence = _seal_evidence({
                "schema": ROLLOUT_EVIDENCE_SCHEMA,
                "training_core_schema": TRAINING_CORE_SCHEMA,
                "stage_id": self.stage.stage_id,
                "stage_configuration_sha256": self.stage.configuration_sha256,
                "policy_version": policy_version,
                "accepted_transitions": accepted_this_update,
                "cumulative_accepted_transitions": cumulative_transitions,
                "rollout_sha256": rollout_digest,
                "reward_mean": reward_mean,
                "boundary_rounds": boundary_rounds,
                "runtime_manifest_sha256": self.runtime_manifest_sha256,
                "atlas_sha256": self.atlas_sha256,
                "private_causal_payload_serialized": False,
                "causal_metrics_window_sha256": causal_metrics_sha256,
                "causal_metrics_window": causal_metrics,
                "network_metrics_window_sha256": network_metrics_sha256,
                "network_metrics_window": network_metrics,
            })
            rollout_path = self.output_root / "evidence/rollouts" / (
                f"rollout-{cumulative_policy_updates:08d}-"
                f"{rollout_evidence['evidence_sha256'][:16]}.json"
            )
            _publish_content_json(rollout_path, rollout_evidence)

            update_evidence = _seal_evidence({
                "schema": UPDATE_EVIDENCE_SCHEMA,
                "training_core_schema": TRAINING_CORE_SCHEMA,
                "stage_id": self.stage.stage_id,
                "stage_configuration_sha256": self.stage.configuration_sha256,
                "policy_version": policy_version,
                "policy_update": cumulative_policy_updates,
                "optimizer_steps_this_update": optimizer_steps_this_update,
                "cumulative_optimizer_steps": cumulative_optimizer_steps,
                "cumulative_accepted_transitions": cumulative_transitions,
                "rollout_evidence_sha256": rollout_evidence["evidence_sha256"],
                "ppo_metrics": ppo_metrics,
                "checkpoint": str(checkpoint.relative_to(self.output_root)),
                "checkpoint_sha256": checkpoint_sha256,
                "checkpoint_manifest": manifest_mapping,
                "lineage_root_sha256": lineage_root,
                "runtime_manifest_sha256": self.runtime_manifest_sha256,
                "atlas_sha256": self.atlas_sha256,
                "promotion_claim": False,
                "causal_metrics_window_sha256": causal_metrics_sha256,
                "network_metrics_window_sha256": network_metrics_sha256,
            })
            update_path = self.output_root / "evidence/updates" / (
                f"update-{cumulative_policy_updates:08d}-"
                f"{update_evidence['evidence_sha256'][:16]}.json"
            )
            _publish_content_json(update_path, update_evidence)

            season = _seal_evidence({
                "schema": SEASON_REPORT_SCHEMA,
                "training_core_schema": TRAINING_CORE_SCHEMA,
                "health": "training-active",
                "promotion_claim": False,
                "stage_id": self.stage.stage_id,
                "stage_name": self.stage.stage_name,
                "stage_configuration_sha256": self.stage.configuration_sha256,
                "runtime_manifest_sha256": self.runtime_manifest_sha256,
                "atlas_sha256": self.atlas_sha256,
                "lineage_root_sha256": lineage_root,
                "counters": {
                    "accepted_transitions": cumulative_transitions,
                    "policy_updates": cumulative_policy_updates,
                    "optimizer_steps": cumulative_optimizer_steps,
                    "next_policy_version": policy_version + 1,
                },
                "last_policy_version": policy_version,
                "last_rollout_evidence_sha256": (
                    rollout_evidence["evidence_sha256"]
                ),
                "last_update_evidence_sha256": update_evidence["evidence_sha256"],
                "last_checkpoint": str(checkpoint.relative_to(self.output_root)),
                "checkpoint_sha256": checkpoint_sha256,
                "checkpoint_manifest": manifest_mapping,
                "last_reward_mean": reward_mean,
                "last_ppo_metrics": ppo_metrics,
                "causal_metrics_window_sha256": causal_metrics_sha256,
                "causal_metrics_window": causal_metrics,
                "network_metrics_window_sha256": network_metrics_sha256,
                "network_metrics_window": network_metrics,
            })
            _atomic_json(self.current_season_report, season, replace=True)

            # TensorBoard is observer output.  Publish it only after the
            # sealed current-season pointer commits the same transaction.
            self._emit_scalars(
                policy_version=policy_version,
                accepted_this_update=accepted_this_update,
                reward_mean=reward_mean,
                boundary_rounds=boundary_rounds,
                ppo_metrics=ppo_metrics,
                causal_metrics=causal_metrics,
                network_metrics=network_metrics,
                cumulative_transitions=cumulative_transitions,
                cumulative_policy_updates=cumulative_policy_updates,
                cumulative_optimizer_steps=cumulative_optimizer_steps,
            )

            self.accepted_transitions = cumulative_transitions
            self.policy_updates = cumulative_policy_updates
            self.optimizer_steps = cumulative_optimizer_steps
            self.next_policy_version = policy_version + 1
            self._lineage_root_sha256 = str(lineage_root)
            self._causal_metric_transitions = causal_metric_transitions
            self._network_metric_transitions = network_metric_transitions
            return season
        except Exception:
            if checkpoint_staging is not None:
                try:
                    if checkpoint_staging.is_file() and not checkpoint_staging.is_symlink():
                        checkpoint_staging.unlink()
                except OSError:
                    pass
            self._failed = True
            raise

    def run(
        self,
        *,
        maximum_updates: int | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> TrainingRunSummary:
        if maximum_updates is not None and (
            type(maximum_updates) is not int or maximum_updates < 1
        ):
            raise MultiresTrainingCoreError(
                "finite training requires at least one optimizer update"
            )
        if should_stop is not None and not callable(should_stop):
            raise TypeError("should_stop must be callable")
        updates_before = self.policy_updates
        while maximum_updates is None or (
            self.policy_updates - updates_before < maximum_updates
        ):
            if should_stop is not None and should_stop():
                break
            self.train_one_update()
        if self.policy_updates == updates_before:
            raise MultiresTrainingCoreError(
                "training stopped with zero policy updates; no health claim emitted"
            )
        if not _valid_sha256(self._lineage_root_sha256):
            raise MultiresTrainingCoreError("training has no attested lineage root")
        return TrainingRunSummary(
            accepted_transitions=self.accepted_transitions,
            policy_updates=self.policy_updates,
            optimizer_steps=self.optimizer_steps,
            next_policy_version=self.next_policy_version,
            lineage_root_sha256=str(self._lineage_root_sha256),
            current_season_report=self.current_season_report,
        )
