"""Versioned synchronous PPO rollout transport for trusted LAN workers.

The wire format is deterministic and pickle-free. A learner publishes one
policy version, accepts only batches produced by that exact artifact, waits for
a quorum, updates, then publishes the next version. HTTP is used only as a
small dependency-free framing layer; q2ded and lattice queries remain local to
each worker.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence

import numpy as np

from harness.protocol import ML_PROTOCOL_GENERATION, OBS_DIM

from harness.distributed_runtime import (
    AssignmentLease,
    AssignmentLeaseBook,
    AssignmentUnavailableError,
    LatticeArtifact,
    LearnerLatticeStore,
    RolloutAssignment,
    StaleLeaseError,
    build_generation_assignments,
    validate_batch_lease,
)

PROTOCOL_VERSION = 2
POLICY_MAGIC = b"Q2PL0002"
BATCH_MAGIC = b"Q2RB0002"
MAX_POLICY_BYTES = 128 * 1024 * 1024
MAX_BATCH_BYTES = 512 * 1024 * 1024
ALLOWED_DTYPES = {
    "|u1", "|i1", "<i2", "<u2", "<i4", "<u4", "<i8", "<u8", "<f4", "<f8"
}
_SHA256_HEX = frozenset("0123456789abcdef")


def _valid_sha256(value: str) -> bool:
    return len(value) == 64 and all(char in _SHA256_HEX for char in value)

# Telemetry carried beside every real PPO rollout.  The order is part of the
# PPO telemetry schema so workers and learners cannot silently disagree about
# vector positions.
PPO_TELEMETRY_SCHEMA = "ppo-telemetry-multires-v1"
PPO_ACTION_CARDINALITIES = {
    "vertical_intent": 3,
    "fire": 2,
    "hook": 4,
    "weapon": 10,
}
PPO_EPISODE_SUMMARY_COLUMNS = (
    "reward",
    "base_reward",
    "spatial_reward",
    "kills",
    "deaths",
    "length",
)
PPO_BEHAVIOR_METRIC_KEYS = (
    "ammo_depleted",
    "requested_ammo_weapon_unavailable",
    "fire_no_ammo",
    "hook_action",
    "hook_enemy",
    "hook_no_ammo_melee",
    "hook_overspeed",
    "hook_fire_action",
    "hook_noop_action",
    "hook_release_action",
    "hook_release_overspeed",
    "hook_correction_available",
    "hook_correction_needed",
    "hook_correction_active",
    "hook_correction_started",
    "hook_correction_escape",
    "hook_correction_progress",
    "hook_correction_progress_reward",
    "hook_correction_success",
    "hook_correction_timeout",
    "hook_correction_heat",
    "movement_speed",
    "movement_intent",
    "forward_intent",
    "movement_nominal",
    "movement_slow",
    "movement_overspeed",
    "movement_discipline",
    "level_aim_movement_reward",
    "jump_action",
    "jump_slow",
    "entity_count",
    "enemy_count",
    "enemy_visible_any",
    "enemy_visible_count",
    "enemy_visible_nearest",
    "enemy_visible_exposure_sum",
    "enemy_visible_exposure_max",
    "aim_aligned",
    "aim_yaw_error",
    "aim_pitch_error",
    "aim_tracking_quality",
    "target_hit_event",
    "target_hit_streak",
    "target_repeat_hit_event",
    "target_kill_event",
    "fire_aligned",
    "fire_unseen",
    "fire_unaligned",
    "damage_dealt",
    "damage_taken",
    "kills",
    "deaths",
    "session_memory_bonus",
    "session_memory_cells",
    "session_current_engagement",
    "session_current_threat",
    "session_current_opportunity",
    "session_current_self_fire",
    "session_nearest_engagement",
    "session_nearest_threat",
    "session_nearest_opportunity",
    "thermal_target_tracks",
    "thermal_target_heat",
    "thermal_target_age",
    "lattice_prior_loaded",
    "lattice_routes_loaded",
    "lattice_dynamic_cells",
    "lattice_route_active",
    "threat_bonus",
    "threat_in_range",
    "threat_active",
    "threat_ignored",
    "survival_low_health",
    "survival_contact",
    "damage_margin_step",
    "outcome_sample",
    "outcome_bonus",
    "outcome_win",
    "outcome_survival",
    "outcome_loss",
    "outcome_idle",
    "episode_damage_margin",
    "episode_frag_margin",
    "episode_contact_events",
    "offense",
    "survival",
    "damage_prox",
    "exchange_ratio",
    "exchange_dominating",
    "exchange_even",
    "exchange_losing",
    "rune_held",
    "rune_switch",
    "win_margin",
    "target_alignment_bonus",
    "target_acquired",
    "target_aligned",
)


def _canonical_json(value) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _encode_envelope(magic: bytes, manifest: dict, payload: bytes) -> bytes:
    encoded = _canonical_json(manifest)
    return magic + struct.pack("!I", len(encoded)) + encoded + payload


def _decode_envelope(data: bytes, magic: bytes, maximum: int) -> tuple[dict, bytes]:
    if len(data) > maximum:
        raise ValueError(f"message exceeds {maximum} bytes")
    if len(data) < 12 or data[:8] != magic:
        raise ValueError("invalid message header")
    manifest_size = struct.unpack("!I", data[8:12])[0]
    end = 12 + manifest_size
    if end > len(data):
        raise ValueError("truncated message manifest")
    try:
        manifest = json.loads(data[12:end].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("invalid message manifest") from error
    return manifest, data[end:]


@dataclass(frozen=True)
class PolicyArtifact:
    version: int
    payload: bytes
    sha256: str
    config_hash: str = ""
    runtime_manifest_sha256: str = ""

    @classmethod
    def create(
        cls,
        version: int,
        payload: bytes,
        config_hash: str = "",
        runtime_manifest_sha256: str = "",
    ):
        if int(version) < 0:
            raise ValueError("policy version must be nonnegative")
        if len(payload) > MAX_POLICY_BYTES:
            raise ValueError("policy artifact is too large")
        runtime_digest = str(runtime_manifest_sha256)
        if runtime_digest and not _valid_sha256(runtime_digest):
            raise ValueError("runtime manifest digest must be lowercase SHA-256")
        return cls(
            int(version), bytes(payload), _sha256(payload), str(config_hash),
            runtime_digest,
        )

    def encode(self) -> bytes:
        return _encode_envelope(POLICY_MAGIC, {
            "protocol_version": PROTOCOL_VERSION,
            "policy_version": self.version,
            "policy_sha256": self.sha256,
            "config_hash": self.config_hash,
            "runtime_manifest_sha256": self.runtime_manifest_sha256,
            "payload_bytes": len(self.payload),
        }, self.payload)

    @classmethod
    def decode(cls, data: bytes):
        manifest, payload = _decode_envelope(data, POLICY_MAGIC, MAX_POLICY_BYTES + 65536)
        if int(manifest.get("protocol_version", 0)) != PROTOCOL_VERSION:
            raise ValueError("unsupported policy protocol version")
        artifact = cls.create(
            int(manifest["policy_version"]),
            payload,
            manifest.get("config_hash", ""),
            manifest.get("runtime_manifest_sha256", ""),
        )
        if artifact.sha256 != manifest.get("policy_sha256"):
            raise ValueError("policy artifact checksum mismatch")
        if len(payload) != int(manifest.get("payload_bytes", -1)):
            raise ValueError("policy artifact length mismatch")
        return artifact


@dataclass
class RolloutBatch:
    metadata: dict
    arrays: Dict[str, np.ndarray]

    REQUIRED_METADATA = {
        "worker_id", "sequence", "policy_version", "policy_sha256",
        "config_hash", "seed", "game_seed", "rollout_index", "determinism_key",
    }

    def __post_init__(self):
        missing = self.REQUIRED_METADATA - set(self.metadata)
        if missing:
            raise ValueError(f"missing rollout metadata: {sorted(missing)}")
        normalized = {}
        for name, array in self.arrays.items():
            if not name or not name.replace("_", "").isalnum():
                raise ValueError(f"invalid array name: {name!r}")
            value = np.ascontiguousarray(array)
            dtype = value.dtype.str
            if dtype not in ALLOWED_DTYPES:
                raise ValueError(f"unsupported rollout dtype: {dtype}")
            normalized[name] = value
        if not normalized:
            raise ValueError("rollout batch must contain arrays")
        self.arrays = normalized

    def rollout_hash(self) -> str:
        digest = hashlib.sha256()
        stable = {
            key: self.metadata[key]
            for key in (
                "policy_version", "policy_sha256", "config_hash", "seed",
                "game_seed", "rollout_index", "determinism_key",
            )
        }
        stable["runtime_manifest_sha256"] = self.metadata.get(
            "runtime_manifest_sha256", ""
        )
        digest.update(_canonical_json(stable))
        for name in sorted(self.arrays):
            array = self.arrays[name]
            digest.update(_canonical_json({
                "name": name, "dtype": array.dtype.str, "shape": list(array.shape)
            }))
            digest.update(array.tobytes(order="C"))
        return digest.hexdigest()

    def encode(self) -> bytes:
        payload_parts = []
        schemas = []
        offset = 0
        for name in sorted(self.arrays):
            array = self.arrays[name]
            raw = array.tobytes(order="C")
            schemas.append({
                "name": name,
                "dtype": array.dtype.str,
                "shape": list(array.shape),
                "offset": offset,
                "nbytes": len(raw),
            })
            payload_parts.append(raw)
            offset += len(raw)
        manifest = {
            "protocol_version": PROTOCOL_VERSION,
            "metadata": self.metadata,
            "rollout_hash": self.rollout_hash(),
            "arrays": schemas,
            "payload_bytes": offset,
        }
        encoded = _encode_envelope(BATCH_MAGIC, manifest, b"".join(payload_parts))
        if len(encoded) > MAX_BATCH_BYTES:
            raise ValueError("rollout batch is too large")
        return encoded

    @classmethod
    def decode(cls, data: bytes):
        manifest, payload = _decode_envelope(data, BATCH_MAGIC, MAX_BATCH_BYTES)
        if int(manifest.get("protocol_version", 0)) != PROTOCOL_VERSION:
            raise ValueError("unsupported rollout protocol version")
        if len(payload) != int(manifest.get("payload_bytes", -1)):
            raise ValueError("rollout payload length mismatch")
        arrays = {}
        occupied = 0
        for schema in manifest.get("arrays", []):
            name = str(schema["name"])
            dtype = np.dtype(schema["dtype"])
            if dtype.str not in ALLOWED_DTYPES:
                raise ValueError(f"unsupported rollout dtype: {dtype.str}")
            shape = tuple(int(value) for value in schema["shape"])
            if any(value < 0 for value in shape):
                raise ValueError("negative rollout array dimension")
            offset = int(schema["offset"])
            nbytes = int(schema["nbytes"])
            expected = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
            if offset != occupied or nbytes != expected or offset + nbytes > len(payload):
                raise ValueError("invalid rollout array layout")
            arrays[name] = np.frombuffer(
                payload[offset:offset + nbytes], dtype=dtype
            ).reshape(shape).copy()
            occupied += nbytes
        if occupied != len(payload):
            raise ValueError("unclaimed rollout payload bytes")
        batch = cls(dict(manifest["metadata"]), arrays)
        if batch.rollout_hash() != manifest.get("rollout_hash"):
            raise ValueError("rollout hash mismatch")
        return batch

    def validate_ppo_schema(self) -> None:
        required = {
            "obs", "actions", "rewards", "dones", "values", "log_probs",
            "h_states", "c_states", "last_obs", "last_h", "last_c",
            "episode_summaries", "behavior_sums", "behavior_samples",
        }
        missing = required - set(self.arrays)
        if missing:
            raise ValueError(f"missing PPO arrays: {sorted(missing)}")
        obs = self.arrays["obs"]
        if obs.ndim != 3 or obs.shape[0] < 1 or obs.shape[1] < 1:
            raise ValueError("obs must have shape (steps, envs, obs_dim)")
        steps, envs, obs_dim = obs.shape
        if obs_dim != OBS_DIM:
            raise ValueError(
                f"PPO obs width must be the frozen multires width {OBS_DIM}"
            )
        expected = {
            "actions": (steps, envs, 8),
            "rewards": (steps, envs),
            "dones": (steps, envs),
            "values": (steps, envs),
            "log_probs": (steps, envs),
            "h_states": (steps, envs, 256),
            "c_states": (steps, envs, 256),
            "last_obs": (envs, obs_dim),
            "last_h": (envs, 256),
            "last_c": (envs, 256),
        }
        for name, shape in expected.items():
            if self.arrays[name].shape != shape:
                raise ValueError(f"{name} must have shape {shape}")
        float_arrays = required - {
            "dones", "episode_summaries", "behavior_sums", "behavior_samples"
        }
        if any(self.arrays[name].dtype != np.dtype("<f4") for name in float_arrays):
            raise ValueError("PPO floating arrays must be float32")
        if self.arrays["dones"].dtype != np.dtype("uint8"):
            raise ValueError("PPO dones must be uint8")
        episode_summaries = self.arrays["episode_summaries"]
        if episode_summaries.ndim != 2 or episode_summaries.shape[1] != len(
            PPO_EPISODE_SUMMARY_COLUMNS
        ):
            raise ValueError(
                "PPO episode_summaries must have shape "
                f"(episodes, {len(PPO_EPISODE_SUMMARY_COLUMNS)})"
            )
        if episode_summaries.dtype != np.dtype("<f8"):
            raise ValueError("PPO episode_summaries must be float64")
        if self.arrays["behavior_sums"].shape != (len(PPO_BEHAVIOR_METRIC_KEYS),):
            raise ValueError(
                "PPO behavior_sums must match PPO_BEHAVIOR_METRIC_KEYS"
            )
        if self.arrays["behavior_sums"].dtype != np.dtype("<f8"):
            raise ValueError("PPO behavior_sums must be float64")
        if self.arrays["behavior_samples"].shape != (1,):
            raise ValueError("PPO behavior_samples must have shape (1,)")
        if self.arrays["behavior_samples"].dtype != np.dtype("<i8"):
            raise ValueError("PPO behavior_samples must be int64")
        if int(self.arrays["behavior_samples"][0]) < 0:
            raise ValueError("PPO behavior_samples must be nonnegative")
        if int(self.arrays["behavior_samples"][0]) != steps * envs:
            raise ValueError("PPO behavior_samples must equal steps * envs")
        if episode_summaries.shape[0] != int(np.count_nonzero(self.arrays["dones"])):
            raise ValueError("PPO episode_summaries must match completed episodes")
        if not np.isfinite(episode_summaries).all():
            raise ValueError("PPO episode_summaries must be finite")
        if not np.isfinite(self.arrays["behavior_sums"]).all():
            raise ValueError("PPO behavior_sums must be finite")
        if int(self.metadata.get("n_envs", -1)) != envs:
            raise ValueError("PPO n_envs metadata mismatch")
        if self.metadata.get("producer") != "q2":
            raise ValueError("PPO producer must be q2")
        if self.metadata.get("lattice_mode") not in {
            "fresh_worker_session", "persistent", "versioned_snapshot"
        }:
            raise ValueError("invalid PPO lattice_mode")
        if not isinstance(self.metadata.get("deterministic_actions"), bool):
            raise ValueError("PPO deterministic_actions must be boolean")
        if not _valid_sha256(str(self.metadata.get("runtime_manifest_sha256", ""))):
            raise ValueError(
                "PPO runtime_manifest_sha256 must be lowercase SHA-256"
            )
        if self.metadata.get("telemetry_schema") != PPO_TELEMETRY_SCHEMA:
            raise ValueError("invalid PPO telemetry_schema")
        if int(self.metadata.get("protocol_generation", -1)) != int(
            ML_PROTOCOL_GENERATION
        ):
            raise ValueError("invalid PPO protocol_generation")
        if int(self.metadata.get("observation_dim", -1)) != OBS_DIM:
            raise ValueError("invalid PPO observation_dim")
        if self.metadata.get("action_cardinalities") != PPO_ACTION_CARDINALITIES:
            raise ValueError("invalid PPO action_cardinalities")
        if not isinstance(self.metadata.get("map_name"), str) or not self.metadata[
            "map_name"
        ]:
            raise ValueError("PPO map_name must be a non-empty string")


@dataclass(frozen=True)
class BatchDecision:
    status: str
    accepted: bool
    detail: str
    policy_version: int
    quorum_count: int
    assignment_id: str = ""
    lattice_artifact_sha256: str = ""

    def as_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass(frozen=True)
class CoordinatorRecoveryConfig:
    """Opt-in lease and learner-owned lattice settings for one rollout pool."""

    learner_id: str
    steps: int
    n_envs: int
    maps: Sequence[str]
    base_seed: int
    base_game_seed: int
    lease_ttl: float = 45.0
    max_attempts: int = 3
    lattice_store: Optional[LearnerLatticeStore] = None
    require_lattice_snapshot: bool = True
    journal_root: Optional[Path] = None

    def __post_init__(self):
        if not isinstance(self.learner_id, str) or not self.learner_id.strip():
            raise ValueError("recovery learner_id must be nonempty")
        if int(self.steps) < 1 or int(self.n_envs) < 1:
            raise ValueError("recovery steps and n_envs must be positive")
        if not self.maps or any(not isinstance(name, str) or not name for name in self.maps):
            raise ValueError("recovery maps must be nonempty strings")
        if float(self.lease_ttl) <= 0 or int(self.max_attempts) < 1:
            raise ValueError("recovery lease settings must be positive")
        if self.require_lattice_snapshot and self.lattice_store is None:
            raise ValueError("required lattice snapshots need a learner lattice store")
        object.__setattr__(self, "maps", tuple(self.maps))
        journal_root = self.journal_root
        if journal_root is None and self.lattice_store is not None:
            journal_root = self.lattice_store.root / "generation-journal"
        if journal_root is not None:
            object.__setattr__(self, "journal_root", Path(journal_root))


class _GenerationJournal:
    """Atomic write-ahead journal for accepted leased rollout batches."""

    SCHEMA = 1

    def __init__(self, root: Path, learner_id: str, config_hash: str):
        self.root = Path(root)
        self.learner_id = learner_id
        self.config_hash = config_hash

    def _path(self, assignment: RolloutAssignment) -> Path:
        return (
            self.root / f"generation_{assignment.policy_version:020d}" /
            f"lane_{assignment.lane_index:04d}_{assignment.assignment_id}.q2gj"
        )

    @staticmethod
    def _atomic_write(path: Path, payload: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(
            f".{path.name}.tmp-{os.getpid()}-{threading.get_ident()}"
        )
        try:
            with temporary.open("wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass

    def record(
        self,
        assignment: RolloutAssignment,
        encoded_batch: bytes,
        *,
        lease_epoch: int,
        lease_id: str,
    ) -> Path:
        manifest = {
            "schema": self.SCHEMA,
            "learner_id": self.learner_id,
            "config_hash": self.config_hash,
            "assignment": assignment.as_dict(),
            "lease_epoch": int(lease_epoch),
            "lease_id": str(lease_id),
            "batch_sha256": _sha256(encoded_batch),
            "batch_bytes": len(encoded_batch),
        }
        header = _canonical_json(manifest)
        payload = struct.pack("<Q", len(header)) + header + encoded_batch
        path = self._path(assignment)
        if path.exists():
            existing = path.read_bytes()
            if existing != payload:
                raise ValueError("conflicting durable batch journal entry")
            return path
        self._atomic_write(path, payload)
        return path

    def remove(self, assignment: RolloutAssignment) -> None:
        try:
            self._path(assignment).unlink()
        except FileNotFoundError:
            pass

    def load(self, policy_version: int):
        directory = self.root / f"generation_{int(policy_version):020d}"
        result = []
        if not directory.is_dir():
            return result
        for path in sorted(directory.glob("lane_*.q2gj")):
            payload = path.read_bytes()
            if len(payload) < 8:
                raise ValueError("truncated generation journal entry")
            header_bytes = struct.unpack("<Q", payload[:8])[0]
            if header_bytes > len(payload) - 8:
                raise ValueError("invalid generation journal header length")
            manifest = json.loads(payload[8:8 + header_bytes])
            encoded = payload[8 + header_bytes:]
            if (
                manifest.get("schema") != self.SCHEMA
                or manifest.get("learner_id") != self.learner_id
                or manifest.get("config_hash") != self.config_hash
                or manifest.get("batch_bytes") != len(encoded)
                or manifest.get("batch_sha256") != _sha256(encoded)
            ):
                raise ValueError("generation journal identity/checksum mismatch")
            assignment = RolloutAssignment.from_dict(manifest["assignment"])
            if assignment.policy_version != int(policy_version):
                raise ValueError("generation journal policy version mismatch")
            result.append((
                assignment,
                encoded,
                int(manifest["lease_epoch"]),
                str(manifest["lease_id"]),
            ))
        return result


class RolloutCoordinator:
    """Thread-safe synchronous generation barrier for one learner."""

    def __init__(
        self,
        quorum: int,
        schema: str = "any",
        expected_runtime_manifest_sha256: str = "",
        recovery: Optional[CoordinatorRecoveryConfig] = None,
    ):
        if int(quorum) < 1:
            raise ValueError("quorum must be positive")
        self.quorum = int(quorum)
        if schema not in {"any", "ppo"}:
            raise ValueError("schema must be 'any' or 'ppo'")
        self.schema = schema
        self.expected_runtime_manifest_sha256 = str(
            expected_runtime_manifest_sha256
        )
        if (
            self.expected_runtime_manifest_sha256
            and not _valid_sha256(self.expected_runtime_manifest_sha256)
        ):
            raise ValueError("expected runtime manifest must be lowercase SHA-256")
        if self.schema == "ppo" and not self.expected_runtime_manifest_sha256:
            raise ValueError("PPO coordinator requires an expected runtime manifest")
        if recovery is not None and self.schema != "ppo":
            raise ValueError("leased recovery currently requires schema='ppo'")
        if recovery is not None and recovery.n_envs < 1:
            raise ValueError("recovery n_envs must be positive")
        self.recovery = recovery
        self._journal = (
            _GenerationJournal(
                recovery.journal_root,
                recovery.learner_id,
                recovery.lattice_store.config_hash,
            )
            if recovery is not None
            and recovery.journal_root is not None
            and recovery.lattice_store is not None
            else None
        )
        self._lease_book = (
            AssignmentLeaseBook(recovery.lease_ttl, recovery.max_attempts)
            if recovery is not None else None
        )
        self._assignments: Dict[int, Dict[int, RolloutAssignment]] = {}
        self._receipts: Dict[str, BatchDecision] = {}
        self._policy: Optional[PolicyArtifact] = None
        self._batches: Dict[int, Dict[str, RolloutBatch]] = {}
        self._batch_ids: set[str] = set()
        self._determinism: Dict[str, str] = {}
        self._sealed_versions: set[int] = set()
        self._condition = threading.Condition()

    def publish(self, artifact: PolicyArtifact) -> None:
        with self._condition:
            if (
                self.expected_runtime_manifest_sha256
                and artifact.runtime_manifest_sha256
                != self.expected_runtime_manifest_sha256
            ):
                raise ValueError(
                    "policy runtime manifest does not match coordinator runtime"
                )
            if self._policy is not None and artifact.version <= self._policy.version:
                raise ValueError("policy versions must increase monotonically")
            self._policy = artifact
            self._batches.pop(artifact.version, None)
            if self.recovery is not None:
                restored_entries = (
                    self._journal.load(artifact.version)
                    if self._journal is not None else []
                )
                restored_by_lane = {}
                for saved_assignment, encoded, lease_epoch, lease_id in restored_entries:
                    if (
                        saved_assignment.config_hash != artifact.config_hash
                        or saved_assignment.policy_sha256 != artifact.sha256
                    ):
                        raise ValueError(
                            "generation journal does not match published policy"
                        )
                    if saved_assignment.lane_index in restored_by_lane:
                        raise ValueError("generation journal repeats a lane")
                    restored_by_lane[saved_assignment.lane_index] = (
                        saved_assignment, encoded, lease_epoch, lease_id
                    )
                lattice_refs = {}
                if self.recovery.lattice_store is not None:
                    for lane_index in range(self.quorum):
                        latest = self.recovery.lattice_store.latest(lane_index)
                        if latest is not None and latest.policy_version < artifact.version:
                            lattice_refs[lane_index] = latest.artifact_sha256
                        elif (
                            latest is not None
                            and latest.policy_version == artifact.version
                            and lane_index not in restored_by_lane
                        ):
                            raise ValueError(
                                "current-generation lattice has no durable batch journal"
                            )
                assignments = build_generation_assignments(
                    learner_id=self.recovery.learner_id,
                    config_hash=artifact.config_hash,
                    policy_version=artifact.version,
                    policy_sha256=artifact.sha256,
                    rollout_index=artifact.version,
                    lanes=self.quorum,
                    steps=self.recovery.steps,
                    n_envs=self.recovery.n_envs,
                    maps=self.recovery.maps,
                    base_seed=self.recovery.base_seed,
                    base_game_seed=self.recovery.base_game_seed,
                    lattice_artifacts=lattice_refs,
                )
                assignments = tuple(
                    restored_by_lane.get(assignment.lane_index, (assignment,))[0]
                    for assignment in assignments
                )
                self._assignments[artifact.version] = {
                    assignment.lane_index: assignment
                    for assignment in assignments
                }
                assert self._lease_book is not None
                self._lease_book.add(assignments)
                generation = self._batches.setdefault(artifact.version, {})
                for assignment, encoded, lease_epoch, lease_id in restored_entries:
                    batch = RolloutBatch.decode(encoded)
                    if any(
                        batch.metadata.get(name) != value
                        for name, value in assignment.batch_contract().items()
                    ):
                        raise ValueError("journaled batch violates assignment contract")
                    batch.validate_ppo_schema()
                    lattice_array = batch.arrays.get("lattice_payload")
                    if lattice_array is None and self.recovery.require_lattice_snapshot:
                        raise ValueError("journaled batch has no lattice payload")
                    lattice_digest = ""
                    if lattice_array is not None:
                        lattice_artifact = LatticeArtifact.from_assignment(
                            assignment, lattice_array.tobytes()
                        )
                        assert self.recovery.lattice_store is not None
                        self.recovery.lattice_store.publish(
                            lattice_artifact, assignment
                        )
                        lattice_digest = lattice_artifact.artifact_sha256
                    batch_id = _sha256(encoded)
                    rollout_hash = batch.rollout_hash()
                    key = str(batch.metadata["determinism_key"])
                    previous = self._determinism.get(key)
                    if previous is not None and previous != rollout_hash:
                        raise ValueError("journaled batch determinism conflict")
                    self._determinism[key] = rollout_hash
                    generation[assignment.assignment_id] = batch
                    self._batch_ids.add(batch_id)
                    decision = BatchDecision(
                        "accepted", True, "batch restored from journal",
                        artifact.version, len(generation),
                        assignment.assignment_id, lattice_digest,
                    )
                    self._receipts[batch_id] = decision
                    self._lease_book.restore_completed(
                        assignment.assignment_id,
                        attempts=lease_epoch,
                        completed_lease_id=lease_id,
                    )
            self._condition.notify_all()

    def policy(self) -> Optional[PolicyArtifact]:
        with self._condition:
            return self._policy

    @property
    def recovery_enabled(self) -> bool:
        return self.recovery is not None

    def claim_assignment(
        self, worker_id: str, preferred_lane: Optional[int] = None
    ) -> AssignmentLease:
        with self._condition:
            if self.recovery is None or self._lease_book is None:
                raise AssignmentUnavailableError("leased recovery is disabled")
            if self._policy is None:
                raise AssignmentUnavailableError("learner has no policy")
            version = self._policy.version
            if version in self._sealed_versions:
                raise AssignmentUnavailableError("generation is closed")
            assignment_id = None
            if preferred_lane is not None:
                try:
                    assignment_id = self._assignments[version][
                        int(preferred_lane)
                    ].assignment_id
                except (KeyError, TypeError, ValueError) as error:
                    raise AssignmentUnavailableError(
                        "preferred lane is unavailable"
                    ) from error
            lease = self._lease_book.claim(
                worker_id, assignment_id=assignment_id
            )
            if lease is None:
                raise AssignmentUnavailableError(
                    "no assignment is currently leaseable"
                )
            return lease

    def heartbeat_assignment(
        self, lease_id: str, worker_id: str
    ) -> AssignmentLease:
        if self._lease_book is None:
            raise AssignmentUnavailableError("leased recovery is disabled")
        return self._lease_book.heartbeat(lease_id, worker_id)

    def release_assignment(
        self,
        lease_id: str,
        worker_id: str,
        reason: str,
        retryable: bool = True,
    ) -> str:
        if self._lease_book is None:
            raise AssignmentUnavailableError("leased recovery is disabled")
        return self._lease_book.release(
            lease_id,
            worker_id,
            reason,
            retryable=bool(retryable),
        )

    def lattice_artifact(self, lane_index: int, digest: str):
        if self.recovery is None or self.recovery.lattice_store is None:
            raise FileNotFoundError("learner lattice store is disabled")
        return self.recovery.lattice_store.get(int(lane_index), digest)

    def submit(self, encoded: bytes) -> BatchDecision:
        try:
            batch = RolloutBatch.decode(encoded)
        except (KeyError, TypeError, ValueError) as error:
            return BatchDecision("invalid", False, str(error), -1, 0)
        batch_id = _sha256(encoded)
        with self._condition:
            receipt = self._receipts.get(batch_id)
            if receipt is not None:
                return BatchDecision(
                    "duplicate",
                    False,
                    "batch already received",
                    receipt.policy_version,
                    receipt.quorum_count,
                    receipt.assignment_id,
                    receipt.lattice_artifact_sha256,
                )
            if self._policy is None:
                return BatchDecision("no_policy", False, "learner has no policy", -1, 0)
            current = self._policy
            version = int(batch.metadata["policy_version"])
            if version != current.version or batch.metadata["policy_sha256"] != current.sha256:
                label = "stale" if version < current.version else "wrong_policy"
                return BatchDecision(label, False, "batch policy is not current", current.version,
                                     len(self._batches.get(current.version, {})))
            if batch.metadata.get("config_hash", "") != current.config_hash:
                return BatchDecision("wrong_config", False, "config hash mismatch",
                                     current.version, len(self._batches.get(current.version, {})))
            if current.version in self._sealed_versions:
                return BatchDecision("generation_closed", False,
                                     "learner already consumed this generation",
                                     current.version, 0)
            expected_runtime = (
                current.runtime_manifest_sha256
                or self.expected_runtime_manifest_sha256
            )
            if (
                expected_runtime
                and batch.metadata.get("runtime_manifest_sha256")
                != expected_runtime
            ):
                return BatchDecision(
                    "wrong_runtime",
                    False,
                    "runtime manifest digest mismatch",
                    current.version,
                    len(self._batches.get(current.version, {})),
                )
            if self.schema == "ppo":
                try:
                    batch.validate_ppo_schema()
                except ValueError as error:
                    return BatchDecision("invalid_schema", False, str(error),
                                         current.version,
                                         len(self._batches.get(current.version, {})))

            lease = None
            assignment = None
            lease_checked_at = time.monotonic()
            if self.recovery is not None:
                if self._lease_book is None:
                    return BatchDecision(
                        "invalid_lease", False, "lease registry is unavailable",
                        current.version, len(self._batches.get(current.version, {})),
                    )
                try:
                    lease = self._lease_book.active_lease(
                        str(batch.metadata.get("lease_id", "")),
                        now=lease_checked_at,
                    )
                    assignment = lease.assignment
                    validate_batch_lease(
                        batch.metadata,
                        assignment,
                        lease,
                        now=lease_checked_at,
                    )
                except (StaleLeaseError, TypeError, ValueError) as error:
                    return BatchDecision(
                        "invalid_lease", False, str(error), current.version,
                        len(self._batches.get(current.version, {})),
                    )

            key = str(batch.metadata["determinism_key"])
            rollout_hash = batch.rollout_hash()
            previous = self._determinism.get(key)
            if previous is not None and previous != rollout_hash:
                return BatchDecision("determinism_mismatch", False,
                                     "same determinism key produced different rollout",
                                     current.version, len(self._batches.get(current.version, {})))
            worker_key = (
                assignment.assignment_id if assignment is not None
                else f"{batch.metadata['worker_id']}:{int(batch.metadata['sequence'])}"
            )
            generation = self._batches.setdefault(current.version, {})
            if worker_key in generation:
                label = "duplicate_assignment" if assignment is not None else "duplicate_sequence"
                return BatchDecision(
                    label, False, "generation slot already received",
                    current.version, len(generation), worker_key,
                )

            lattice_artifact_sha256 = ""
            lattice_artifact = None
            if assignment is not None and self.recovery is not None:
                lattice_array = batch.arrays.get("lattice_payload")
                if self.recovery.require_lattice_snapshot and lattice_array is None:
                    return BatchDecision(
                        "invalid_lattice", False,
                        "leased batch is missing lattice_payload",
                        current.version, len(generation), assignment.assignment_id,
                    )
                if lattice_array is not None:
                    if (
                        lattice_array.ndim != 1
                        or lattice_array.dtype != np.dtype("uint8")
                    ):
                        return BatchDecision(
                            "invalid_lattice", False,
                            "lattice_payload must be a one-dimensional uint8 array",
                            current.version, len(generation), assignment.assignment_id,
                        )
                    if self.recovery.lattice_store is None:
                        return BatchDecision(
                            "invalid_lattice", False,
                            "learner lattice store is disabled",
                            current.version, len(generation), assignment.assignment_id,
                        )
                    try:
                        lattice_artifact = LatticeArtifact.from_assignment(
                            assignment, lattice_array.tobytes()
                        )
                    except (OSError, TypeError, ValueError) as error:
                        return BatchDecision(
                            "invalid_lattice", False, str(error),
                            current.version, len(generation), assignment.assignment_id,
                        )
                    lattice_artifact_sha256 = lattice_artifact.artifact_sha256

            if lease is not None:
                journal_recorded = False
                try:
                    before_complete = None
                    def _commit_durable_acceptance():
                        nonlocal journal_recorded
                        if self._journal is not None:
                            self._journal.record(
                                assignment,
                                encoded,
                                lease_epoch=lease.epoch,
                                lease_id=lease.lease_id,
                            )
                            journal_recorded = True
                        if lattice_artifact is not None:
                            self.recovery.lattice_store.publish(
                                lattice_artifact, assignment
                            )
                    before_complete = _commit_durable_acceptance
                    self._lease_book.complete(
                        lease.lease_id,
                        lease.worker_id,
                        before_complete=before_complete,
                    )
                except StaleLeaseError as error:
                    return BatchDecision(
                        "invalid_lease", False, str(error), current.version,
                        len(generation), assignment.assignment_id,
                    )
                except (OSError, TypeError, ValueError) as error:
                    if journal_recorded and self._journal is not None:
                        self._journal.remove(assignment)
                    return BatchDecision(
                        "invalid_lattice", False, str(error), current.version,
                        len(generation), assignment.assignment_id,
                    )
            self._determinism[key] = rollout_hash
            generation[worker_key] = batch
            self._batch_ids.add(batch_id)
            self._condition.notify_all()
            decision = BatchDecision(
                "accepted", True, "batch accepted", current.version,
                len(generation),
                assignment.assignment_id if assignment is not None else "",
                lattice_artifact_sha256,
            )
            self._receipts[batch_id] = decision
            return decision

    def wait_for_quorum(self, version: int, timeout: float) -> list[RolloutBatch]:
        deadline = time.monotonic() + max(0.0, float(timeout))
        with self._condition:
            while len(self._batches.get(int(version), {})) < self.quorum:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return []
                self._condition.wait(remaining)
            generation = self._batches[int(version)]
            keys = sorted(generation)[:self.quorum]
            result = [generation.pop(key) for key in keys]
            self._sealed_versions.add(int(version))
            generation.clear()
            return result

    def status(self) -> dict:
        with self._condition:
            version = self._policy.version if self._policy else -1
            lease_states = []
            if self._lease_book is not None:
                lease_states = [
                    self._lease_book.snapshot(assignment.assignment_id)["state"]
                    for assignment in self._assignments.get(version, {}).values()
                ]
            return {
                "protocol_version": PROTOCOL_VERSION,
                "policy_version": version,
                "policy_sha256": self._policy.sha256 if self._policy else "",
                "config_hash": self._policy.config_hash if self._policy else "",
                "quorum": self.quorum,
                "schema": self.schema,
                "generation_closed": version in self._sealed_versions,
                "runtime_manifest_sha256": (
                    self._policy.runtime_manifest_sha256 if self._policy else
                    self.expected_runtime_manifest_sha256
                ),
                "accepted_for_current": len(self._batches.get(version, {})),
                "recovery_enabled": self.recovery_enabled,
                "assignments_current": len(lease_states),
                "assignments_pending": lease_states.count("pending"),
                "assignments_leased": lease_states.count("leased"),
                "assignments_completed": lease_states.count("completed"),
                "assignments_failed": lease_states.count("failed"),
            }


class CoordinatorServer:
    def __init__(self, coordinator: RolloutCoordinator, host: str, port: int,
                 token: str = ""):
        self.coordinator = coordinator
        self.token = token
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def _authorized(self):
                return not owner.token or self.headers.get("Authorization") == f"Bearer {owner.token}"

            def _json(self, status: int, payload: dict):
                encoded = _canonical_json(payload)
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

            def _read_json(self, maximum: int = 65536):
                try:
                    length = int(self.headers.get("Content-Length", "-1"))
                except ValueError as error:
                    raise ValueError("invalid content length") from error
                if length < 0 or length > maximum:
                    raise ValueError("invalid content length")
                try:
                    value = json.loads(self.rfile.read(length))
                except (UnicodeDecodeError, json.JSONDecodeError) as error:
                    raise ValueError("invalid JSON request") from error
                if not isinstance(value, dict):
                    raise ValueError("JSON request must be an object")
                return value

            def do_GET(self):
                if not self._authorized():
                    self._json(401, {"error": "unauthorized"})
                    return
                if self.path == "/v1/status":
                    self._json(200, owner.coordinator.status())
                elif self.path == "/v1/policy":
                    artifact = owner.coordinator.policy()
                    if artifact is None:
                        self._json(404, {"error": "no policy"})
                        return
                    encoded = artifact.encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/octet-stream")
                    self.send_header("Content-Length", str(len(encoded)))
                    self.end_headers()
                    self.wfile.write(encoded)
                elif self.path.startswith("/v1/lattices/"):
                    parts = self.path.split("/")
                    if len(parts) != 5:
                        self._json(404, {"error": "not found"})
                        return
                    try:
                        artifact = owner.coordinator.lattice_artifact(
                            int(parts[3]), parts[4]
                        )
                    except (FileNotFoundError, TypeError, ValueError):
                        self._json(404, {"error": "lattice artifact not found"})
                        return
                    encoded = artifact.encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/octet-stream")
                    self.send_header("Content-Length", str(len(encoded)))
                    self.end_headers()
                    self.wfile.write(encoded)
                else:
                    self._json(404, {"error": "not found"})

            def do_POST(self):
                if not self._authorized():
                    self._json(401, {"error": "unauthorized"})
                    return
                if self.path == "/v1/assignments/claim":
                    try:
                        request = self._read_json()
                        lease = owner.coordinator.claim_assignment(
                            request["worker_id"], request.get("preferred_lane")
                        )
                    except AssignmentUnavailableError as error:
                        self._json(409, {"error": str(error)})
                        return
                    except (KeyError, TypeError, ValueError) as error:
                        self._json(422, {"error": str(error)})
                        return
                    self._json(201, lease.as_dict())
                    return
                if self.path == "/v1/leases/heartbeat":
                    try:
                        request = self._read_json()
                        lease = owner.coordinator.heartbeat_assignment(
                            request["lease_id"], request["worker_id"]
                        )
                    except (AssignmentUnavailableError, StaleLeaseError) as error:
                        self._json(409, {"error": str(error)})
                        return
                    except (KeyError, TypeError, ValueError) as error:
                        self._json(422, {"error": str(error)})
                        return
                    self._json(200, lease.as_dict())
                    return
                if self.path == "/v1/leases/release":
                    try:
                        request = self._read_json()
                        state = owner.coordinator.release_assignment(
                            request["lease_id"],
                            request["worker_id"],
                            request.get("reason", "worker_release"),
                            request.get("retryable", True),
                        )
                    except (AssignmentUnavailableError, StaleLeaseError) as error:
                        self._json(409, {"error": str(error)})
                        return
                    except (KeyError, TypeError, ValueError) as error:
                        self._json(422, {"error": str(error)})
                        return
                    self._json(200, {"state": state})
                    return
                if self.path != "/v1/batches":
                    self._json(404, {"error": "not found"})
                    return
                try:
                    length = int(self.headers.get("Content-Length", "-1"))
                except ValueError:
                    length = -1
                if length < 0 or length > MAX_BATCH_BYTES:
                    self._json(413, {"error": "invalid content length"})
                    return
                decision = owner.coordinator.submit(self.rfile.read(length))
                status = 202 if decision.accepted else (200 if decision.status == "duplicate" else 409)
                if decision.status == "invalid":
                    status = 422
                self._json(status, decision.as_dict())

            def log_message(self, _format, *_args):
                return

        self.httpd = ThreadingHTTPServer((host, int(port)), Handler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    @property
    def address(self):
        return self.httpd.server_address

    def start(self):
        self.thread.start()
        return self

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=5)


class CoordinatorTransportError(RuntimeError):
    """The coordinator could not be reached or the request timed out."""


class CoordinatorRequestError(RuntimeError):
    """The coordinator returned an HTTP response the operation cannot use."""

    def __init__(self, status: int, detail: str):
        super().__init__(detail)
        self.status = int(status)


class CoordinatorClient:
    def __init__(self, base_url: str, token: str = "", timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout = float(timeout)

    def _request(self, path: str, data: Optional[bytes] = None):
        headers = {}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        if data is not None:
            headers["Content-Type"] = "application/octet-stream"
        request = urllib.request.Request(self.base_url + path, data=data, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return response.status, response.read()
        except urllib.error.HTTPError as error:
            return error.code, error.read()
        except (OSError, TimeoutError, urllib.error.URLError) as error:
            raise CoordinatorTransportError(str(error)) from error

    @staticmethod
    def _decode_json_response(status: int, payload: bytes) -> dict:
        try:
            value = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise CoordinatorRequestError(
                status, f"invalid coordinator response ({status})"
            ) from error
        if not isinstance(value, dict):
            raise CoordinatorRequestError(status, "coordinator response is not an object")
        return value

    def fetch_policy(self) -> PolicyArtifact:
        status, payload = self._request("/v1/policy")
        if status != 200:
            raise CoordinatorRequestError(
                status,
                f"policy fetch failed ({status}): {payload.decode(errors='replace')}",
            )
        return PolicyArtifact.decode(payload)

    def submit(self, batch: RolloutBatch) -> BatchDecision:
        status, payload = self._request("/v1/batches", batch.encode())
        decoded = self._decode_json_response(status, payload)
        try:
            return BatchDecision(**decoded)
        except (TypeError, ValueError) as error:
            raise CoordinatorRequestError(
                status, str(decoded.get("error", decoded))
            ) from error

    def status(self) -> dict:
        status, payload = self._request("/v1/status")
        if status != 200:
            raise CoordinatorRequestError(status, f"status fetch failed ({status})")
        return self._decode_json_response(status, payload)

    def claim_assignment(
        self, worker_id: str, preferred_lane: Optional[int] = None
    ) -> AssignmentLease:
        request = {"worker_id": worker_id}
        if preferred_lane is not None:
            request["preferred_lane"] = int(preferred_lane)
        status, payload = self._request(
            "/v1/assignments/claim", _canonical_json(request)
        )
        decoded = self._decode_json_response(status, payload)
        if status != 201:
            raise CoordinatorRequestError(status, str(decoded.get("error", decoded)))
        return AssignmentLease.from_dict(decoded)

    def heartbeat(self, lease: AssignmentLease) -> AssignmentLease:
        status, payload = self._request(
            "/v1/leases/heartbeat",
            _canonical_json({
                "lease_id": lease.lease_id,
                "worker_id": lease.worker_id,
            }),
        )
        decoded = self._decode_json_response(status, payload)
        if status != 200:
            raise CoordinatorRequestError(status, str(decoded.get("error", decoded)))
        return AssignmentLease.from_dict(decoded)

    def release_lease(
        self,
        lease: AssignmentLease,
        reason: str,
        retryable: bool = True,
    ) -> str:
        status, payload = self._request(
            "/v1/leases/release",
            _canonical_json({
                "lease_id": lease.lease_id,
                "worker_id": lease.worker_id,
                "reason": reason,
                "retryable": bool(retryable),
            }),
        )
        decoded = self._decode_json_response(status, payload)
        if status != 200:
            raise CoordinatorRequestError(status, str(decoded.get("error", decoded)))
        return str(decoded["state"])

    def fetch_lattice(self, lane_index: int, digest: str) -> LatticeArtifact:
        status, payload = self._request(
            f"/v1/lattices/{int(lane_index)}/{digest}"
        )
        if status != 200:
            raise CoordinatorRequestError(
                status, f"lattice fetch failed ({status})"
            )
        artifact = LatticeArtifact.decode(payload)
        if artifact.artifact_sha256 != digest or artifact.lane_index != int(lane_index):
            raise ValueError("fetched lattice artifact identity mismatch")
        return artifact


def deterministic_synthetic_batch(
    artifact: PolicyArtifact,
    worker_id: str,
    sequence: int,
    seed: int,
    game_seed: int,
    rollout_index: int,
    steps: int = 32,
    obs_dim: int = 64,
    action_dim: int = 8,
) -> RolloutBatch:
    """Generate a portable deterministic batch for protocol/LAN validation."""
    rng = np.random.default_rng(int(seed) ^ (int(game_seed) << 1) ^ int(rollout_index))
    arrays = {
        "obs": rng.standard_normal((steps, obs_dim), dtype=np.float32),
        "actions": rng.standard_normal((steps, action_dim), dtype=np.float32),
        "rewards": rng.standard_normal(steps, dtype=np.float32),
        "dones": (rng.random(steps) < 0.05).astype(np.uint8),
        "values": rng.standard_normal(steps, dtype=np.float32),
        "log_probs": rng.standard_normal(steps, dtype=np.float32),
    }
    determinism_key = (
        f"v{artifact.version}:{artifact.sha256}:cfg={artifact.config_hash}:"
        f"seed={seed}:game={game_seed}:rollout={rollout_index}"
    )
    return RolloutBatch({
        "worker_id": str(worker_id),
        "sequence": int(sequence),
        "policy_version": artifact.version,
        "policy_sha256": artifact.sha256,
        "config_hash": artifact.config_hash,
        "seed": int(seed),
        "game_seed": int(game_seed),
        "rollout_index": int(rollout_index),
        "determinism_key": determinism_key,
        "runtime_manifest_sha256": artifact.runtime_manifest_sha256,
    }, arrays)


def merge_ppo_batches(batches: Iterable[RolloutBatch]) -> dict[str, object]:
    """Merge one synchronous quorum along the environment axis for PPO."""
    batches = list(batches)
    if not batches:
        raise ValueError("cannot merge an empty rollout quorum")
    for batch in batches:
        batch.validate_ppo_schema()
    reference = batches[0]
    generation = (
        int(reference.metadata["policy_version"]),
        str(reference.metadata["policy_sha256"]),
        str(reference.metadata.get("config_hash", "")),
        str(reference.metadata["runtime_manifest_sha256"]),
        str(reference.metadata["lattice_mode"]),
        bool(reference.metadata["deterministic_actions"]),
    )
    step_shape = reference.arrays["obs"].shape[::2]
    for batch in batches[1:]:
        candidate = (
            int(batch.metadata["policy_version"]),
            str(batch.metadata["policy_sha256"]),
            str(batch.metadata.get("config_hash", "")),
            str(batch.metadata["runtime_manifest_sha256"]),
            str(batch.metadata["lattice_mode"]),
            bool(batch.metadata["deterministic_actions"]),
        )
        if candidate != generation:
            raise ValueError("cannot merge rollout batches from different policies")
        if batch.arrays["obs"].shape[::2] != step_shape:
            raise ValueError("rollout step/observation dimensions do not match")
    time_major = (
        "obs", "actions", "rewards", "dones", "values", "log_probs",
        "h_states", "c_states",
    )
    final_state = ("last_obs", "last_h", "last_c")
    merged = {
        name: np.concatenate([batch.arrays[name] for batch in batches], axis=1)
        for name in time_major
    }
    merged.update({
        name: np.concatenate([batch.arrays[name] for batch in batches], axis=0)
        for name in final_state
    })
    merged["episode_summaries"] = np.concatenate(
        [batch.arrays["episode_summaries"] for batch in batches], axis=0
    )
    merged["episode_map_names"] = tuple(
        str(batch.metadata["map_name"])
        for batch in batches
        for _ in range(batch.arrays["episode_summaries"].shape[0])
    )
    merged["behavior_sums"] = np.stack(
        [batch.arrays["behavior_sums"] for batch in batches], axis=0
    ).sum(axis=0, dtype=np.float64)
    merged["behavior_samples"] = np.array([
        sum(int(batch.arrays["behavior_samples"][0]) for batch in batches)
    ], dtype=np.int64)
    return merged
