"""Trainer-side validation of the atomic B3/B4 runtime evidence.

The exact new wire integers remain owned by B4.  B5 only rejects known legacy
values and requires the runtime to declare the frozen policy contract and
public/teacher separation before constructing a trainer.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping

from .multires_contract import (
    ACTION_DIM,
    FEATURE_SCHEMA_SHA256,
    OBS_DIM,
    POLICY_GENERATION,
    POSTURE_CLASSES,
)
from .multires_lineage import LineageError


LEGACY_OBSERVATION_MAGIC = 0x514D4C50
LEGACY_ACTION_MAGIC = 0x514D4C41
LEGACY_CLIENT_WIRE_VERSION = 4
LEGACY_TEACHER_VERSION = 2
LEGACY_ROLLOUT_SCHEMA = "ppo-telemetry-v8"
B4_OBSERVATION_MAGIC = 0x514D324F
B4_ACTION_MAGIC = 0x514D3241
B4_PROTOCOL_GENERATION = 2
B4_CLIENT_WIRE_VERSION = 8
B4_TEACHER_VERSION = 4
B4_ROLLOUT_SCHEMA = "ppo-telemetry-multires-v1"
B4_CAUSAL_MAGIC = 0x514D3343
B4_CAUSAL_VERSION = 2
B4_CAUSAL_PACKET_BYTES = 80
_SHA256_CHARS = frozenset("0123456789abcdef")


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _valid_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in _SHA256_CHARS for character in value)


@dataclass(frozen=True)
class ValidatedMultiresRuntime:
    atlas_sha256: str
    runtime_manifest_sha256: str
    protocol_generation: int
    observation_magic: int
    action_magic: int
    client_wire_version: int
    teacher_version: int
    rollout_schema: str


def validate_runtime_evidence(
    evidence: Mapping[str, Any],
    *,
    expected_atlas_sha256: str,
) -> ValidatedMultiresRuntime:
    """Reject legacy, incomplete, mixed, or privilege-leaking runtime evidence."""
    required = {
        "policy_generation",
        "feature_schema_sha256",
        "observation_dim",
        "action_dim",
        "posture_classes",
        "protocol_generation",
        "observation_magic",
        "action_magic",
        "client_wire_version",
        "teacher_version",
        "rollout_schema",
        "atlas_sha256",
        "public_teacher_packing_separate",
        "public_teacher_field_violations",
        "recovery_width",
        "guide_width",
        "causal_magic",
        "causal_version",
        "causal_packet_bytes",
    }
    missing = sorted(required - set(evidence))
    if missing:
        raise LineageError(f"runtime evidence is incomplete: missing={missing}")
    expected = {
        "policy_generation": POLICY_GENERATION,
        "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
        "observation_dim": OBS_DIM,
        "action_dim": ACTION_DIM,
        "posture_classes": POSTURE_CLASSES,
        "protocol_generation": B4_PROTOCOL_GENERATION,
        "atlas_sha256": expected_atlas_sha256,
        "public_teacher_packing_separate": True,
        "public_teacher_field_violations": 0,
        "recovery_width": 16,
        "guide_width": 60,
        "causal_magic": B4_CAUSAL_MAGIC,
        "causal_version": B4_CAUSAL_VERSION,
        "causal_packet_bytes": B4_CAUSAL_PACKET_BYTES,
    }
    mismatches = {
        name: (evidence[name], wanted)
        for name, wanted in expected.items()
        if evidence[name] != wanted
    }
    observation_magic = int(evidence["observation_magic"])
    action_magic = int(evidence["action_magic"])
    client_wire_version = int(evidence["client_wire_version"])
    teacher_version = int(evidence["teacher_version"])
    rollout_schema = str(evidence["rollout_schema"])
    if observation_magic != B4_OBSERVATION_MAGIC:
        mismatches["observation_magic"] = (observation_magic, B4_OBSERVATION_MAGIC)
    if action_magic != B4_ACTION_MAGIC:
        mismatches["action_magic"] = (action_magic, B4_ACTION_MAGIC)
    if client_wire_version != B4_CLIENT_WIRE_VERSION:
        mismatches["client_wire_version"] = (
            client_wire_version, B4_CLIENT_WIRE_VERSION
        )
    if teacher_version != B4_TEACHER_VERSION:
        mismatches["teacher_version"] = (teacher_version, B4_TEACHER_VERSION)
    if rollout_schema != B4_ROLLOUT_SCHEMA:
        mismatches["rollout_schema"] = (rollout_schema, B4_ROLLOUT_SCHEMA)
    if not _valid_sha256(str(expected_atlas_sha256)):
        mismatches["expected_atlas_sha256"] = (
            expected_atlas_sha256, "lowercase SHA-256"
        )
    if mismatches:
        details = "; ".join(
            f"{name}={found!r} expected {wanted!r}"
            for name, (found, wanted) in sorted(mismatches.items())
        )
        raise LineageError(f"multires runtime admission failed: {details}")
    digest = str(evidence.get("runtime_manifest_sha256", ""))
    if digest:
        if not _valid_sha256(digest):
            raise LineageError("runtime_manifest_sha256 must be lowercase SHA-256")
    else:
        digest = hashlib.sha256(_canonical_json(dict(evidence))).hexdigest()
    return ValidatedMultiresRuntime(
        atlas_sha256=str(evidence["atlas_sha256"]),
        runtime_manifest_sha256=digest,
        protocol_generation=B4_PROTOCOL_GENERATION,
        observation_magic=observation_magic,
        action_magic=action_magic,
        client_wire_version=client_wire_version,
        teacher_version=teacher_version,
        rollout_schema=rollout_schema,
    )


def adapt_b4_observation_descriptor(
    descriptor: Mapping[str, Any],
    *,
    atlas_sha256: str,
    teacher_field_violations: int = 0,
    runtime_manifest_sha256: str = "",
) -> dict[str, Any]:
    """Translate B4's frozen descriptor into the B5 admission vocabulary."""
    required = {
        "protocol_generation",
        "factual_dim",
        "dyn_dim",
        "recovery_dim",
        "objective_count",
        "objective_dim",
        "total_dim",
        "feature_schema_sha256",
        "observation_magic",
        "action_magic",
        "client_wire_version",
        "teacher_version",
        "rollout_telemetry_schema",
        "logical_action_dim",
        "action_cardinalities",
        "teacher_privileged_packing",
        "causal_magic",
        "causal_version",
        "causal_packet_bytes",
    }
    missing = sorted(required - set(descriptor))
    if missing:
        raise LineageError(f"B4 observation descriptor is incomplete: missing={missing}")
    expected_descriptor = {
        "protocol_generation": B4_PROTOCOL_GENERATION,
        "factual_dim": 198,
        "dyn_dim": 24,
        "recovery_dim": 16,
        "objective_count": 4,
        "objective_dim": 15,
        "total_dim": OBS_DIM,
        "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
        "observation_magic": B4_OBSERVATION_MAGIC,
        "action_magic": B4_ACTION_MAGIC,
        "client_wire_version": B4_CLIENT_WIRE_VERSION,
        "teacher_version": B4_TEACHER_VERSION,
        "rollout_telemetry_schema": B4_ROLLOUT_SCHEMA,
        "logical_action_dim": ACTION_DIM,
        "teacher_privileged_packing": "physically-separate-qm3c-v2",
        "causal_magic": B4_CAUSAL_MAGIC,
        "causal_version": B4_CAUSAL_VERSION,
        "causal_packet_bytes": B4_CAUSAL_PACKET_BYTES,
    }
    mismatches = {
        name: (descriptor[name], wanted)
        for name, wanted in expected_descriptor.items()
        if descriptor[name] != wanted
    }
    expected_cardinalities = {
        "vertical_intent": POSTURE_CLASSES,
        "fire": 2,
        "hook": 4,
        "weapon": 10,
    }
    if descriptor["action_cardinalities"] != expected_cardinalities:
        mismatches["action_cardinalities"] = (
            descriptor["action_cardinalities"], expected_cardinalities
        )
    if mismatches:
        details = "; ".join(
            f"{name}={found!r} expected {wanted!r}"
            for name, (found, wanted) in sorted(mismatches.items())
        )
        raise LineageError(f"B4 descriptor/B5 contract mismatch: {details}")
    evidence = {
        "policy_generation": POLICY_GENERATION,
        "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
        "observation_dim": OBS_DIM,
        "action_dim": ACTION_DIM,
        "posture_classes": POSTURE_CLASSES,
        "protocol_generation": B4_PROTOCOL_GENERATION,
        "observation_magic": B4_OBSERVATION_MAGIC,
        "action_magic": B4_ACTION_MAGIC,
        "client_wire_version": B4_CLIENT_WIRE_VERSION,
        "teacher_version": B4_TEACHER_VERSION,
        "rollout_schema": B4_ROLLOUT_SCHEMA,
        "atlas_sha256": atlas_sha256,
        "public_teacher_packing_separate": True,
        "public_teacher_field_violations": int(teacher_field_violations),
        "recovery_width": 16,
        "guide_width": 60,
        "causal_magic": B4_CAUSAL_MAGIC,
        "causal_version": B4_CAUSAL_VERSION,
        "causal_packet_bytes": B4_CAUSAL_PACKET_BYTES,
    }
    if runtime_manifest_sha256:
        evidence["runtime_manifest_sha256"] = runtime_manifest_sha256
    return evidence
