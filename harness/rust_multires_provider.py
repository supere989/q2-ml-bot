"""Production B3 adapter for exact Rust Dyn24 + Atlas recovery/guide fields.

The provider consumes only identity-fenced query facts and objective
availability beliefs.  It never accepts prepacked guides, normalized reward
facts, zero-filled Dyn state, or a Python lattice fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import numpy as np

from .client_protocol import ClientTelemetry
from .multires_admission import (
    SPATIAL_PROVIDER_SCHEMA,
    SPATIAL_REWARD_EVIDENCE_SCHEMA,
    PrivateSpatialRewardEvidence,
    SpatialProviderFrame,
    validate_spatial_provider_frame,
)
from .multires_contract import FEATURE_SCHEMA_SHA256
from .protocol import SpatialPolicyFeatures


RUST_RECOVERY_EVIDENCE_SCHEMA = "q2-recovery-evidence-v1"
RUST_DYN_EVENT_SCHEMA = "q2-dyn-named-event-v1"
_SHA256_CHARS = frozenset("0123456789abcdef")
_BUNDLE_FILES = (".bsp", ".json", ".lattice.json", ".routes.json")
_BUNDLE_ANALYSIS_FILES = (
    ".analysis.manifest.json",
    ".atlas.manifest.json",
    ".atlas.bin.zst",
    ".navigation.bin.zst",
    ".visibility.bin.zst",
    ".design-signature.json",
    ".objectives.json",
)


class RustSpatialProviderError(RuntimeError):
    """Raised when the Rust extension or one of its fences is incomplete."""


def _valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in _SHA256_CHARS for character in value)
        and value != "0" * 64
    )


@dataclass(frozen=True)
class RustMapArtifacts:
    """Pinned inputs needed to construct one map/epoch Rust provider.

    The raw Atlas digest is not trusted by itself: construction also proves it
    against the digest and size embedded in the bundle-attested Atlas manifest.
    """

    bundle_manifest_path: Path
    uncompressed_atlas_path: Path
    dyn_snapshot_path: Path
    expected_atlas_sha256: str
    environment_steps_base: int = 0
    dyn_checkpoint_path: Path | None = None
    dyn_checkpoint_sha256: str = ""
    checkpoint_client_life_epoch: int = 0
    checkpoint_server_frame: int = 0


@dataclass(frozen=True)
class SpatialQueryInputs:
    client_id: str
    client_epoch: int
    map_name: str
    map_epoch: int
    server_frame: int
    expected_environment_steps: int
    environment_steps: int
    survivability: tuple[float, float, float]
    objective_beliefs: tuple[tuple[int, float], ...]
    dyn_events: tuple[
        tuple[int, str, tuple[float, float, float]], ...
    ] = ()
    thermal: tuple[int, tuple[float, float, float], float, int] | None = None
    blocked_nodes: tuple[tuple[int, int, int], ...] = ()
    dynamic_penalties: tuple[tuple[tuple[int, int, int], int], ...] = ()
    enabled_mover_blockers: tuple[int, ...] = ()
    time_to_impact_seconds: float | None = None

    def validate(self, telemetry: ClientTelemetry, *, expected_map_epoch: int) -> None:
        expected = (
            telemetry.client_id,
            telemetry.causal.client_life_epoch,
            telemetry.map_name,
            expected_map_epoch,
            telemetry.server_frame,
        )
        actual = (
            self.client_id, self.client_epoch, self.map_name,
            self.map_epoch, self.server_frame,
        )
        if actual != expected:
            raise RustSpatialProviderError(
                f"spatial query identity {actual!r} differs from telemetry {expected!r}"
            )
        if (
            type(self.expected_environment_steps) is not int
            or type(self.environment_steps) is not int
            or self.expected_environment_steps < 0
            or self.environment_steps != self.expected_environment_steps + 1
        ):
            raise RustSpatialProviderError(
                "environment steps must be an exact one-step CAS transition"
            )
        if (
            len(self.survivability) != 3
            or any(not math.isfinite(float(value)) for value in self.survivability)
        ):
            raise RustSpatialProviderError("survivability projection is not finite width 3")
        ids = []
        for objective_id, availability in self.objective_beliefs:
            if type(objective_id) is not int or objective_id < 0:
                raise RustSpatialProviderError("objective ID must be nonnegative integer")
            if not math.isfinite(float(availability)) or not 0.0 <= availability <= 1.0:
                raise RustSpatialProviderError(
                    "objective availability belief must be within [0, 1]"
                )
            ids.append(objective_id)
        if len(ids) != len(set(ids)):
            raise RustSpatialProviderError("objective availability IDs must be unique")
        allowed_kinds = {
            "engagement", "threat", "opportunity", "self_fire", "death"
        }
        event_ids: list[int] = []
        event_kinds: list[str] = []
        for event in self.dyn_events:
            if not isinstance(event, tuple) or len(event) != 3:
                raise RustSpatialProviderError("Dyn event record is malformed")
            event_id, kind, point = event
            if type(event_id) is not int or event_id <= 0:
                raise RustSpatialProviderError("Dyn event ID must be positive integer")
            if kind not in allowed_kinds:
                raise RustSpatialProviderError("Dyn event kind is not admitted")
            if (
                not isinstance(point, tuple)
                or len(point) != 3
                or any(not math.isfinite(float(value)) for value in point)
            ):
                raise RustSpatialProviderError("Dyn event point is not finite xyz")
            expected_id = (self.server_frame << 3) | (
                ("engagement", "threat", "opportunity", "self_fire", "death")
                .index(kind) + 1
            )
            if event_id != expected_id:
                raise RustSpatialProviderError(
                    "Dyn event ID differs from its frame/kind identity"
                )
            event_ids.append(event_id)
            event_kinds.append(kind)
        if event_ids != sorted(event_ids) or len(set(event_ids)) != len(event_ids):
            raise RustSpatialProviderError("Dyn event IDs must be unique ascending")
        if len(set(event_kinds)) != len(event_kinds):
            raise RustSpatialProviderError("at most one Dyn event per kind/frame")


class SpatialQueryInputSource(Protocol):
    def sample(
        self, telemetry: ClientTelemetry, *, map_epoch: int
    ) -> SpatialQueryInputs:
        ...


def _exact_array(value: Any, shape: tuple[int, ...], label: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32)
    if result.shape != shape or not np.isfinite(result).all():
        raise RustSpatialProviderError(
            f"Rust {label} must be finite shape {shape}, received {result.shape}"
        )
    return np.ascontiguousarray(result)


class RustAtlasSpatialProvider:
    """Concrete adapter over admitted ``q2_lattice_rs`` runtime objects."""

    def __init__(
        self,
        *,
        atlas_runtime: Any,
        dyn_runtime: Any,
        input_source: SpatialQueryInputSource,
        expected_atlas_sha256: str,
        runtime_manifest_sha256: str,
        map_name: str,
        map_epoch: int,
        rust_client_id: int,
        map_sha256: str,
        atlas_origin: tuple[int, int, int],
    ):
        if atlas_runtime is None or dyn_runtime is None:
            raise RustSpatialProviderError("Rust AtlasRuntime and DynRuntime are required")
        for name in (
            "recovery_features_with_evidence", "guide_features",
        ):
            if not callable(getattr(atlas_runtime, name, None)):
                raise RustSpatialProviderError(f"AtlasRuntime lacks {name}()")
        for name in ("commit_frame",):
            if not callable(getattr(dyn_runtime, name, None)):
                raise RustSpatialProviderError(f"DynRuntime lacks {name}()")
        if not callable(getattr(input_source, "sample", None)):
            raise RustSpatialProviderError("production provider requires query inputs")
        if (
            getattr(atlas_runtime, "atlas_sha256", None) != expected_atlas_sha256
            or getattr(dyn_runtime, "atlas_sha256", None) != expected_atlas_sha256
        ):
            raise RustSpatialProviderError("Rust Atlas/Dyn digest fence differs")
        if (
            getattr(atlas_runtime, "map_epoch", None) != map_epoch
            or getattr(dyn_runtime, "map_epoch", None) != map_epoch
        ):
            raise RustSpatialProviderError("Rust Atlas/Dyn map epoch fence differs")
        if getattr(atlas_runtime, "map_id", None) != map_name:
            raise RustSpatialProviderError("Atlas map identity differs")
        if getattr(dyn_runtime, "client_id", None) != rust_client_id:
            raise RustSpatialProviderError("Dyn numeric client identity differs")
        if getattr(dyn_runtime, "map_sha256", None) != map_sha256:
            raise RustSpatialProviderError("Dyn map digest fence differs")
        if tuple(getattr(dyn_runtime, "origin", ())) != tuple(atlas_origin):
            raise RustSpatialProviderError("Dyn Atlas origin fence differs")
        self.atlas_runtime = atlas_runtime
        self.dyn_runtime = dyn_runtime
        self.input_source = input_source
        self.expected_atlas_sha256 = expected_atlas_sha256
        self.runtime_manifest_sha256 = runtime_manifest_sha256
        self.map_name = map_name
        self.map_epoch = int(map_epoch)
        self.rust_client_id = int(rust_client_id)
        self.map_sha256 = str(map_sha256)
        self.atlas_origin = tuple(int(value) for value in atlas_origin)
        self._closed = False

    @classmethod
    def from_admitted_bundle(
        cls,
        *,
        extension_module: Any,
        bundle_manifest_path: Path,
        uncompressed_atlas_path: Path,
        dyn_snapshot_path: Path,
        input_source: SpatialQueryInputSource,
        expected_atlas_sha256: str,
        runtime_manifest_sha256: str,
        map_epoch: int,
        rust_client_id: int,
        client_count: int,
        environment_steps: int,
        atlas_origin: tuple[int, int, int],
        dyn_checkpoint_path: Path | None = None,
        dyn_checkpoint_sha256: str = "",
        checkpoint_client_life_epoch: int = 0,
        checkpoint_server_frame: int = 0,
    ) -> "RustAtlasSpatialProvider":
        """Load only exact bundle-v3 members and a Q2LAT002 snapshot."""
        if extension_module is None:
            raise RustSpatialProviderError("q2_lattice_rs extension is unavailable")
        if not all(
            hasattr(extension_module, name) for name in ("AtlasRuntime", "DynRuntime")
        ):
            raise RustSpatialProviderError("q2_lattice_rs lacks AtlasRuntime/DynRuntime")
        if (
            type(client_count) is not int
            or client_count < 1
            or type(rust_client_id) is not int
            or not 0 <= rust_client_id < client_count
        ):
            raise RustSpatialProviderError(
                "Rust client IDs must be within exact 0..client_count-1"
            )
        if (
            not _valid_sha256(expected_atlas_sha256)
            or not _valid_sha256(runtime_manifest_sha256)
        ):
            raise RustSpatialProviderError(
                "provider requires lowercase Atlas/runtime SHA-256 fences"
            )
        if (
            type(map_epoch) is not int
            or map_epoch < 0
            or type(environment_steps) is not int
            or environment_steps < 0
        ):
            raise RustSpatialProviderError(
                "map epoch and environment steps must be nonnegative integers"
            )
        manifest_source = Path(bundle_manifest_path)
        if manifest_source.is_symlink():
            raise RustSpatialProviderError("bundle manifest symlinks are rejected")
        manifest_path = manifest_source.resolve()
        if not manifest_path.is_file():
            raise RustSpatialProviderError("admitted bundle manifest is missing")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RustSpatialProviderError("bundle manifest is invalid JSON") from error
        if (
            not isinstance(manifest, dict)
            or manifest.get("bundle_version") != 3
            or not isinstance(manifest.get("name"), str)
            or not isinstance(manifest.get("files"), dict)
            or not isinstance(manifest.get("analysis_files"), dict)
            or not isinstance(manifest.get("file_sizes"), dict)
            or not isinstance(manifest.get("analysis_file_sizes"), dict)
        ):
            raise RustSpatialProviderError("provider requires an admitted bundle-v3 manifest")
        map_name = manifest["name"]
        if set(manifest["files"]) != {
            f"{map_name}{suffix}" for suffix in _BUNDLE_FILES
        } or set(manifest["analysis_files"]) != {
            f"{map_name}{suffix}" for suffix in _BUNDLE_ANALYSIS_FILES
        }:
            raise RustSpatialProviderError(
                "bundle-v3 manifest does not declare the exact production artifact set"
            )
        if (
            set(manifest["file_sizes"]) != set(manifest["files"])
            or set(manifest["analysis_file_sizes"])
            != set(manifest["analysis_files"])
        ):
            raise RustSpatialProviderError("bundle-v3 size tables differ from payloads")
        directory = manifest_path.parent
        declared = {
            **manifest["files"],
            **manifest["analysis_files"],
        }
        sizes = {
            **manifest["file_sizes"],
            **manifest["analysis_file_sizes"],
        }
        for filename, digest in declared.items():
            path = directory / filename
            if (
                path.parent != directory
                or path.is_symlink()
                or not path.is_file()
                or path.stat().st_size != sizes.get(filename)
                or hashlib.sha256(path.read_bytes()).hexdigest() != digest
            ):
                raise RustSpatialProviderError(
                    f"bundle-v3 member {filename!r} fails digest/size admission"
                )
        bsp_path = directory / f"{map_name}.bsp"
        atlas_manifest_path = directory / f"{map_name}.atlas.manifest.json"
        objectives_path = directory / f"{map_name}.objectives.json"
        atlas_source = Path(uncompressed_atlas_path)
        dyn_source = Path(dyn_snapshot_path)
        if atlas_source.is_symlink() or dyn_source.is_symlink():
            raise RustSpatialProviderError("Atlas/Dyn artifact symlinks are rejected")
        atlas_path = atlas_source.resolve()
        dyn_path = dyn_source.resolve()
        for path, label in (
            (bsp_path, "BSP"), (atlas_manifest_path, "Atlas manifest"),
            (objectives_path, "objectives"), (atlas_path, "uncompressed Atlas"),
            (dyn_path, "Q2LAT002 Dyn snapshot"),
        ):
            if path.is_symlink() or not path.is_file():
                raise RustSpatialProviderError(f"{label} artifact is missing")
        atlas_bytes = atlas_path.read_bytes()
        if hashlib.sha256(atlas_bytes).hexdigest() != expected_atlas_sha256:
            raise RustSpatialProviderError("uncompressed Atlas digest differs")
        bsp_bytes = bsp_path.read_bytes()
        try:
            atlas_manifest = json.loads(
                atlas_manifest_path.read_text(encoding="utf-8")
            )
            artifact_origin = tuple(int(value) for value in atlas_manifest["grid"]["origin"])
            atlas_record = atlas_manifest["artifacts"][f"{map_name}.atlas.bin"]
            recorded_atlas_sha256 = atlas_record["sha256_uncompressed"]
            recorded_atlas_size = atlas_record["uncompressed_size"]
            bsp_record = atlas_manifest["bsp"]
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
            raise RustSpatialProviderError(
                "Atlas manifest does not prove its grid origin"
            ) from error
        if len(artifact_origin) != 3 or artifact_origin != tuple(atlas_origin):
            raise RustSpatialProviderError(
                "caller Atlas origin differs from admitted Atlas artifact"
            )
        if (
            recorded_atlas_sha256 != expected_atlas_sha256
            or recorded_atlas_size != len(atlas_bytes)
            or not isinstance(bsp_record, dict)
            or bsp_record.get("canonical_map_id") != map_name
            or bsp_record.get("sha256") != hashlib.sha256(bsp_bytes).hexdigest()
            or bsp_record.get("size_bytes") != len(bsp_bytes)
        ):
            raise RustSpatialProviderError(
                "Atlas manifest raw-Atlas/BSP identity differs from admitted artifacts"
            )
        atlas_runtime = extension_module.AtlasRuntime(
            atlas_manifest_path.read_bytes(),
            f"{map_name}.atlas.bin",
            atlas_bytes,
            objectives_path.name,
            objectives_path.read_bytes(),
            bsp_bytes,
            map_name,
            int(map_epoch),
        )
        map_sha256 = hashlib.sha256(bsp_bytes).hexdigest()
        if dyn_checkpoint_path is None:
            dyn_runtime = extension_module.DynRuntime(
                dyn_path.read_bytes(), expected_atlas_sha256, map_sha256,
                list(atlas_origin), int(map_epoch), int(rust_client_id),
                int(client_count), int(environment_steps),
            )
        else:
            checkpoint = Path(dyn_checkpoint_path)
            if checkpoint.is_symlink() or not checkpoint.is_file():
                raise RustSpatialProviderError("Dyn checkpoint artifact is missing")
            if not dyn_checkpoint_sha256:
                raise RustSpatialProviderError("Dyn checkpoint digest is required")
            dyn_runtime = extension_module.DynRuntime.from_checkpoint(
                checkpoint.read_bytes(), dyn_checkpoint_sha256,
                expected_atlas_sha256, map_sha256, list(atlas_origin),
                int(map_epoch), int(rust_client_id), int(client_count),
                int(environment_steps), int(checkpoint_client_life_epoch),
                int(checkpoint_server_frame),
            )
        return cls(
            atlas_runtime=atlas_runtime,
            dyn_runtime=dyn_runtime,
            input_source=input_source,
            expected_atlas_sha256=expected_atlas_sha256,
            runtime_manifest_sha256=runtime_manifest_sha256,
            map_name=map_name,
            map_epoch=map_epoch,
            rust_client_id=rust_client_id,
            map_sha256=map_sha256,
            atlas_origin=atlas_origin,
        )

    def _validate_dyn_report(
        self,
        raw: Any,
        query: SpatialQueryInputs,
        telemetry: ClientTelemetry,
    ) -> None:
        try:
            report = dict(raw)
        except (TypeError, ValueError) as error:
            raise RustSpatialProviderError("Rust Dyn ingest report is malformed") from error
        required = {
            "schema", "events_applied", "cells_updated", "decay_intervals",
            "environment_steps", "client_life_epoch", "server_frame",
            "last_event_id", "snapshot_sha256",
        }
        if set(report) != required:
            raise RustSpatialProviderError("Rust Dyn ingest report fields differ")
        expected = {
            "schema": RUST_DYN_EVENT_SCHEMA,
            "events_applied": len(query.dyn_events),
            "environment_steps": query.environment_steps,
            "client_life_epoch": telemetry.causal.client_life_epoch,
            "server_frame": telemetry.server_frame,
            "last_event_id": getattr(self.dyn_runtime, "last_event_id", None),
            "snapshot_sha256": getattr(self.dyn_runtime, "snapshot_sha256", None),
        }
        if any(report[name] != wanted for name, wanted in expected.items()):
            raise RustSpatialProviderError("Rust Dyn ingest report identity differs")
        if getattr(self.dyn_runtime, "environment_steps", None) != query.environment_steps:
            raise RustSpatialProviderError("Dyn environment step did not advance exactly")

    def dyn_checkpoint(self) -> tuple[bytes, str]:
        if not callable(getattr(self.dyn_runtime, "checkpoint_bytes", None)):
            raise RustSpatialProviderError("DynRuntime lacks checkpoint_bytes()")
        payload = bytes(self.dyn_runtime.checkpoint_bytes())
        digest = str(getattr(self.dyn_runtime, "checkpoint_sha256", ""))
        if hashlib.sha256(payload).hexdigest() != digest:
            raise RustSpatialProviderError("Dyn checkpoint wrapper digest differs")
        return payload, digest

    def _private_evidence(
        self,
        raw: Any,
        telemetry: ClientTelemetry,
    ) -> PrivateSpatialRewardEvidence:
        if not isinstance(raw, dict):
            try:
                raw = dict(raw)
            except (TypeError, ValueError) as error:
                raise RustSpatialProviderError(
                    "Rust recovery evidence is not a mapping"
                ) from error
        required = {
            "schema", "atlas_sha256", "map_epoch", "client_id",
            "client_epoch", "server_frame", "l1_index",
            "cost_to_safety_q8", "signed_safe_clearance_q8",
            "hazard_types", "hazard_severity", "atlas_region_id",
            "confidence", "hazard_component_id", "hazard_component_epoch",
        }
        if set(raw) != required:
            raise RustSpatialProviderError(
                "Rust recovery evidence fields differ; "
                f"missing={sorted(required - set(raw))} "
                f"extra={sorted(set(raw) - required)}"
            )
        expected = {
            "schema": RUST_RECOVERY_EVIDENCE_SCHEMA,
            "atlas_sha256": self.expected_atlas_sha256,
            "map_epoch": self.map_epoch,
            "client_id": self.rust_client_id,
            "client_epoch": telemetry.causal.client_life_epoch,
            "server_frame": telemetry.server_frame,
        }
        mismatches = {
            name: (raw[name], wanted)
            for name, wanted in expected.items()
            if raw[name] != wanted
        }
        if mismatches:
            raise RustSpatialProviderError(
                f"Rust recovery evidence identity differs: {mismatches!r}"
            )
        l1 = raw["l1_index"]
        if not isinstance(l1, (tuple, list)) or len(l1) != 3:
            raise RustSpatialProviderError("Rust recovery L1 index is malformed")
        return PrivateSpatialRewardEvidence(
            schema=SPATIAL_REWARD_EVIDENCE_SCHEMA,
            client_id=telemetry.client_id,
            client_slot=telemetry.client_slot,
            map_name=telemetry.map_name,
            map_epoch=self.map_epoch,
            server_frame=telemetry.server_frame,
            client_epoch=telemetry.causal.client_life_epoch,
            l1_index=tuple(int(value) for value in l1),
            cost_to_safety_q8=int(raw["cost_to_safety_q8"]),
            signed_safe_clearance_q8=int(raw["signed_safe_clearance_q8"]),
            hazard_types=int(raw["hazard_types"]),
            hazard_severity=int(raw["hazard_severity"]),
            atlas_region_id=int(raw["atlas_region_id"]),
            confidence=int(raw["confidence"]),
            hazard_component_id=int(raw["hazard_component_id"]),
            hazard_component_epoch=int(raw["hazard_component_epoch"]),
        )

    def sample(
        self,
        telemetry: ClientTelemetry,
        *,
        episode_projection: bool,
    ) -> SpatialProviderFrame:
        if self._closed:
            raise RustSpatialProviderError("Rust spatial provider is closed")
        if telemetry.map_name != self.map_name:
            raise RustSpatialProviderError(
                "map rotation requires an independently attested Rust provider"
            )
        stage_factory = getattr(self.input_source, "stage", None)
        if not callable(stage_factory):
            raise RustSpatialProviderError(
                "production query source lacks reversible stage()"
            )
        stage = stage_factory(
            telemetry,
            map_epoch=self.map_epoch,
            emit_dyn_events=not episode_projection,
        )
        if not all(callable(getattr(stage, name, None)) for name in ("commit", "rollback")):
            raise RustSpatialProviderError("query source returned a malformed transaction")
        try:
            query = stage.inputs
            query.validate(telemetry, expected_map_epoch=self.map_epoch)
            position = tuple(
                float(value) for value in telemetry.observation.self_state[:3]
            )
            yaw = float(telemetry.observation.yaw)

            # Every Atlas-side operation and private evidence check completes
            # before Dyn's clone-and-commit boundary.  Nothing after the Rust
            # call can reject ordinary output without an extension contract
            # violation.
            recovery_raw, raw_evidence = (
                self.atlas_runtime.recovery_features_with_evidence(
                    position,
                    yaw,
                    self.map_epoch,
                    self.rust_client_id,
                    query.client_epoch,
                    telemetry.server_frame,
                    list(query.blocked_nodes),
                    list(query.dynamic_penalties),
                    list(query.enabled_mover_blockers),
                    query.time_to_impact_seconds,
                )
            )
            guides_raw = self.atlas_runtime.guide_features(
                position,
                yaw,
                self.map_epoch,
                list(query.objective_beliefs),
            )
            recovery = _exact_array(recovery_raw, (16,), "Recovery16")
            guides = _exact_array(guides_raw, (60,), "Guide60").reshape(4, 15)
            private = self._private_evidence(raw_evidence, telemetry)
            provisional = SpatialProviderFrame(
                schema=SPATIAL_PROVIDER_SCHEMA,
                feature_schema_sha256=FEATURE_SCHEMA_SHA256,
                atlas_sha256=self.expected_atlas_sha256,
                runtime_manifest_sha256=self.runtime_manifest_sha256,
                client_id=telemetry.client_id,
                client_slot=telemetry.client_slot,
                map_name=telemetry.map_name,
                map_epoch=self.map_epoch,
                server_frame=telemetry.server_frame,
                spatial=SpatialPolicyFeatures(
                    dyn=np.zeros(24, dtype=np.float32),
                    recovery=recovery,
                    objectives=guides,
                ),
                private_reward_evidence=(None if episode_projection else private),
            )
            validate_spatial_provider_frame(
                provisional,
                telemetry,
                expected_atlas_sha256=self.expected_atlas_sha256,
                expected_runtime_manifest_sha256=self.runtime_manifest_sha256,
                expected_map_epoch=self.map_epoch,
                require_private_reward_evidence=not episode_projection,
            )
            if getattr(self.dyn_runtime, "environment_steps", None) != (
                query.expected_environment_steps
            ):
                raise RustSpatialProviderError(
                    "Dyn runtime/query environment-step CAS differs before commit"
                )

            committed = self.dyn_runtime.commit_frame(
                self.expected_atlas_sha256,
                self.map_sha256,
                list(self.atlas_origin),
                self.map_epoch,
                self.rust_client_id,
                telemetry.causal.client_life_epoch,
                telemetry.server_frame,
                query.expected_environment_steps,
                query.environment_steps,
                list(query.dyn_events),
                position,
                yaw,
                query.survivability,
                query.thermal,
            )
            if not isinstance(committed, tuple) or len(committed) != 3:
                raise RustSpatialProviderError("Rust Dyn commit result is malformed")
            report, dyn_raw, nearest_death_score = committed
            self._validate_dyn_report(report, query, telemetry)
            dyn = _exact_array(dyn_raw, (24,), "Dyn24")
            if not math.isfinite(float(nearest_death_score)):
                raise RustSpatialProviderError("Rust nearest-death score is not finite")
            frame = SpatialProviderFrame(
                schema=provisional.schema,
                feature_schema_sha256=provisional.feature_schema_sha256,
                atlas_sha256=provisional.atlas_sha256,
                runtime_manifest_sha256=provisional.runtime_manifest_sha256,
                client_id=provisional.client_id,
                client_slot=provisional.client_slot,
                map_name=provisional.map_name,
                map_epoch=provisional.map_epoch,
                server_frame=provisional.server_frame,
                spatial=SpatialPolicyFeatures(
                    dyn=dyn, recovery=recovery, objectives=guides
                ),
                private_reward_evidence=provisional.private_reward_evidence,
            )
            stage.commit()
            return frame
        except Exception:
            if getattr(stage, "_active", True):
                stage.rollback()
            raise

    def close(self) -> None:
        self._closed = True


class RustAtlasProviderFactory:
    """Strict one-client factory for fresh provider/source pairs per map epoch.

    Artifact selection is an explicit map-name lookup.  There is no default,
    legacy provider, previous-map reuse, or caller-supplied Atlas digest path.
    """

    def __init__(
        self,
        *,
        extension_module: Any,
        artifacts_by_map: Mapping[str, RustMapArtifacts],
        runtime_manifest_sha256: str,
        bound_client_id: str,
        rust_client_id: int,
        client_count: int,
    ):
        if extension_module is None:
            raise RustSpatialProviderError("q2_lattice_rs extension is unavailable")
        if not _valid_sha256(runtime_manifest_sha256):
            raise RustSpatialProviderError(
                "factory runtime manifest digest must be lowercase SHA-256"
            )
        if not bound_client_id:
            raise RustSpatialProviderError("factory requires one bound client identity")
        if (
            type(client_count) is not int
            or client_count < 1
            or type(rust_client_id) is not int
            or not 0 <= rust_client_id < client_count
        ):
            raise RustSpatialProviderError(
                "factory Rust client ID must be in exact 0..client_count-1"
            )
        if not isinstance(artifacts_by_map, Mapping) or not artifacts_by_map:
            raise RustSpatialProviderError("factory requires explicit per-map artifacts")
        artifacts = dict(artifacts_by_map)
        for map_name, item in artifacts.items():
            if not isinstance(map_name, str) or not map_name:
                raise RustSpatialProviderError("factory map names must be nonempty strings")
            if not isinstance(item, RustMapArtifacts):
                raise RustSpatialProviderError(
                    f"factory artifacts for {map_name!r} have unknown type"
                )
            if not _valid_sha256(item.expected_atlas_sha256):
                raise RustSpatialProviderError(
                    f"factory Atlas digest for {map_name!r} is not lowercase SHA-256"
                )
            if (
                type(item.environment_steps_base) is not int
                or item.environment_steps_base < 0
            ):
                raise RustSpatialProviderError(
                    f"factory environment-step base for {map_name!r} is invalid"
                )
            if item.dyn_checkpoint_path is None:
                if (
                    item.dyn_checkpoint_sha256
                    or item.checkpoint_client_life_epoch
                    or item.checkpoint_server_frame
                ):
                    raise RustSpatialProviderError(
                        "checkpoint identity is forbidden without a checkpoint artifact"
                    )
            elif not _valid_sha256(item.dyn_checkpoint_sha256):
                raise RustSpatialProviderError(
                    f"factory Dyn checkpoint for {map_name!r} lacks a SHA-256 fence"
                )
        self.extension_module = extension_module
        self.artifacts_by_map = artifacts
        self.runtime_manifest_sha256 = runtime_manifest_sha256
        self.bound_client_id = bound_client_id
        self.rust_client_id = rust_client_id
        self.client_count = client_count
        self._last_map_epoch: int | None = None

    def reset_session(self) -> None:
        """Allow a closed client process to begin a fresh epoch-zero session."""
        self._last_map_epoch = None

    def create(
        self, telemetry: ClientTelemetry, *, map_epoch: int
    ) -> "SpatialProviderBinding":
        # Local imports avoid the query-source/provider protocol import cycle.
        from .multires_admission import SpatialProviderBinding
        from .multires_query_source import OwnObservationQuerySource

        if telemetry.client_id != self.bound_client_id:
            raise RustSpatialProviderError(
                "provider factory telemetry differs from its bound client"
            )
        if type(map_epoch) is not int or map_epoch < 0:
            raise RustSpatialProviderError("provider factory map epoch is invalid")
        if self._last_map_epoch is not None and map_epoch <= self._last_map_epoch:
            raise RustSpatialProviderError(
                "provider factory refuses repeated or regressed map epochs"
            )
        try:
            artifacts = self.artifacts_by_map[telemetry.map_name]
        except KeyError as error:
            raise RustSpatialProviderError(
                f"map {telemetry.map_name!r} has no admitted Rust artifacts"
            ) from error

        manifest_source = Path(artifacts.bundle_manifest_path)
        if manifest_source.is_symlink() or not manifest_source.is_file():
            raise RustSpatialProviderError("factory bundle manifest is missing or symlinked")
        try:
            manifest = json.loads(manifest_source.read_text(encoding="utf-8"))
            if manifest["bundle_version"] != 3 or manifest["name"] != telemetry.map_name:
                raise RustSpatialProviderError(
                    "factory bundle identity differs from telemetry"
                )
            objectives_name = f"{telemetry.map_name}.objectives.json"
            objectives_sha256 = manifest["analysis_files"][objectives_name]
            atlas_manifest_path = (
                manifest_source.resolve().parent
                / f"{telemetry.map_name}.atlas.manifest.json"
            )
            atlas_manifest = json.loads(atlas_manifest_path.read_text(encoding="utf-8"))
            atlas_origin = tuple(int(value) for value in atlas_manifest["grid"]["origin"])
            atlas_record = atlas_manifest["artifacts"][
                f"{telemetry.map_name}.atlas.bin"
            ]
        except RustSpatialProviderError:
            raise
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise RustSpatialProviderError(
                "factory cannot derive exact bundle/objective/Atlas identity"
            ) from error
        if (
            len(atlas_origin) != 3
            or not _valid_sha256(objectives_sha256)
            or atlas_record.get("sha256_uncompressed")
            != artifacts.expected_atlas_sha256
        ):
            raise RustSpatialProviderError(
                "factory Atlas/objectives identity differs from admitted artifacts"
            )
        atlas_source = Path(artifacts.uncompressed_atlas_path)
        if (
            atlas_source.is_symlink()
            or not atlas_source.is_file()
            or hashlib.sha256(atlas_source.read_bytes()).hexdigest()
            != artifacts.expected_atlas_sha256
        ):
            raise RustSpatialProviderError(
                "factory raw Atlas bytes differ from their admitted digest"
            )

        query_source = OwnObservationQuerySource.from_objectives_artifact(
            bundle_directory=manifest_source.resolve().parent,
            map_name=telemetry.map_name,
            expected_objectives_sha256=objectives_sha256,
            expected_atlas_sha256=artifacts.expected_atlas_sha256,
            atlas_origin=atlas_origin,  # type: ignore[arg-type]
            client_id=self.bound_client_id,
            environment_steps_base=artifacts.environment_steps_base,
        )
        provider = RustAtlasSpatialProvider.from_admitted_bundle(
            extension_module=self.extension_module,
            bundle_manifest_path=manifest_source,
            uncompressed_atlas_path=artifacts.uncompressed_atlas_path,
            dyn_snapshot_path=artifacts.dyn_snapshot_path,
            input_source=query_source,
            expected_atlas_sha256=artifacts.expected_atlas_sha256,
            runtime_manifest_sha256=self.runtime_manifest_sha256,
            map_epoch=map_epoch,
            rust_client_id=self.rust_client_id,
            client_count=self.client_count,
            environment_steps=artifacts.environment_steps_base,
            atlas_origin=atlas_origin,  # type: ignore[arg-type]
            dyn_checkpoint_path=artifacts.dyn_checkpoint_path,
            dyn_checkpoint_sha256=artifacts.dyn_checkpoint_sha256,
            checkpoint_client_life_epoch=artifacts.checkpoint_client_life_epoch,
            checkpoint_server_frame=artifacts.checkpoint_server_frame,
        )
        self._last_map_epoch = map_epoch
        return SpatialProviderBinding(
            provider=provider,
            atlas_sha256=artifacts.expected_atlas_sha256,
        )
