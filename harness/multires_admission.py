"""Fail-closed admission for multires spatial and private causal telemetry.

The public 298-vector and private reward facts share identity fences but never
share storage.  Provider frames may be attached to collector ``info`` objects
temporarily; only ``SpatialPolicyFeatures`` reaches the policy and only the
strictly admitted ``CausalRewardFrame`` reaches the reward reducer.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
import math
from typing import Any, Mapping, Optional, Protocol, Sequence, runtime_checkable

from .causal_protocol import CausalFlags, CausalTelemetry
from .client_protocol import ClientTelemetry
from .multires_contract import FEATURE_SCHEMA_SHA256
from .multires_reward import (
    FIRE_SUPPRESSION_NONE,
    FIRE_SUPPRESSION_PROTECTED,
    FIRE_SUPPRESSION_UNALIGNED,
    CausalRewardFrame,
    RewardAdmissionError,
)
from .protocol import SpatialPolicyFeatures


SPATIAL_PROVIDER_SCHEMA = "q2-multires-spatial-provider-v1"
PRIVATE_CAUSAL_SCHEMA = "q2-multires-private-causal-v1"
PRIVATE_CAUSAL_INFO_KEY = "_multires_private_causal"
PRIVATE_SPATIAL_REWARD_INFO_KEY = "_multires_private_spatial_reward"
SPATIAL_ATTESTATION_INFO_KEY = "_multires_spatial_attestation"
SPATIAL_REWARD_EVIDENCE_SCHEMA = "q2-multires-spatial-reward-evidence-v1"
_SHA256_CHARS = frozenset("0123456789abcdef")
SIGNED_CLEARANCE_UNREACHABLE_SAFE = (1 << 31) - 1
SIGNED_CLEARANCE_UNREACHABLE_HAZARD = -(1 << 31)


class MultiresAdmissionError(RewardAdmissionError):
    """Raised when provider or causal identity cannot be proven exactly."""


def _valid_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in _SHA256_CHARS for character in value)


@dataclass(frozen=True)
class SpatialProviderFrame:
    schema: str
    feature_schema_sha256: str
    atlas_sha256: str
    runtime_manifest_sha256: str
    client_id: str
    client_slot: int
    map_name: str
    map_epoch: int
    server_frame: int
    spatial: SpatialPolicyFeatures
    private_reward_evidence: Optional["PrivateSpatialRewardEvidence"] = None


@dataclass(frozen=True)
class PrivateSpatialRewardEvidence:
    """Raw, non-policy B3 facts; never decoded from normalized recovery input."""

    schema: str
    client_id: str
    client_slot: int
    map_name: str
    map_epoch: int
    server_frame: int
    client_epoch: int
    l1_index: tuple[int, int, int]
    cost_to_safety_q8: int
    signed_safe_clearance_q8: int
    hazard_types: int
    hazard_severity: int
    atlas_region_id: int
    confidence: int
    hazard_component_id: int
    hazard_component_epoch: int


@runtime_checkable
class SpatialFeatureProvider(Protocol):
    """Injected B3/B4 adapter; no legacy implementation satisfies this API."""

    def sample(
        self,
        telemetry: ClientTelemetry,
        *,
        episode_projection: bool,
    ) -> SpatialProviderFrame:
        ...

    def close(self) -> None:
        ...


@dataclass(frozen=True)
class SpatialProviderBinding:
    provider: SpatialFeatureProvider
    atlas_sha256: str


@runtime_checkable
class SpatialFeatureProviderFactory(Protocol):
    """Create a fresh map/epoch-bound provider; rotation has no reuse path."""

    def create(
        self, telemetry: ClientTelemetry, *, map_epoch: int
    ) -> SpatialProviderBinding:
        ...


def validate_spatial_provider_frame(
    frame: SpatialProviderFrame,
    telemetry: ClientTelemetry,
    *,
    expected_atlas_sha256: str,
    expected_runtime_manifest_sha256: str,
    expected_map_epoch: int,
    require_private_reward_evidence: bool,
) -> SpatialPolicyFeatures:
    """Validate exact provider identity before assembling any policy vector."""
    if not isinstance(frame, SpatialProviderFrame):
        raise MultiresAdmissionError("spatial provider returned an unknown frame type")
    expected = {
        "schema": SPATIAL_PROVIDER_SCHEMA,
        "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
        "atlas_sha256": expected_atlas_sha256,
        "runtime_manifest_sha256": expected_runtime_manifest_sha256,
        "client_id": telemetry.client_id,
        "client_slot": telemetry.client_slot,
        "map_name": telemetry.map_name,
        "server_frame": telemetry.server_frame,
        "map_epoch": expected_map_epoch,
    }
    mismatches = {
        name: (getattr(frame, name), wanted)
        for name, wanted in expected.items()
        if getattr(frame, name) != wanted
    }
    if expected_map_epoch < 0:
        mismatches["expected_map_epoch"] = (expected_map_epoch, ">= 0")
    if not _valid_sha256(str(frame.atlas_sha256)):
        mismatches["atlas_sha256"] = (frame.atlas_sha256, "lowercase SHA-256")
    if not _valid_sha256(str(frame.runtime_manifest_sha256)):
        mismatches["runtime_manifest_sha256"] = (
            frame.runtime_manifest_sha256, "lowercase SHA-256"
        )
    if mismatches:
        details = "; ".join(
            f"{name}={found!r} expected {wanted!r}"
            for name, (found, wanted) in sorted(mismatches.items())
        )
        raise MultiresAdmissionError(f"spatial provider attestation failed: {details}")
    if not isinstance(frame.spatial, SpatialPolicyFeatures):
        raise MultiresAdmissionError(
            "provider must return exact SpatialPolicyFeatures, not a legacy vector"
        )
    try:
        vector = frame.spatial.to_vector()
    except (TypeError, ValueError) as error:
        raise MultiresAdmissionError("provider spatial features are invalid") from error
    if vector.shape != (100,) or not math.isfinite(float(vector.sum())):
        raise MultiresAdmissionError("provider spatial block is not finite 100-float data")
    evidence = frame.private_reward_evidence
    if require_private_reward_evidence:
        if not isinstance(evidence, PrivateSpatialRewardEvidence):
            raise MultiresAdmissionError(
                "accepted transition lacks private raw spatial reward evidence"
            )
        expected_evidence = {
            "schema": SPATIAL_REWARD_EVIDENCE_SCHEMA,
            "client_id": telemetry.client_id,
            "client_slot": telemetry.client_slot,
            "map_name": telemetry.map_name,
            "map_epoch": expected_map_epoch,
            "server_frame": telemetry.server_frame,
            "client_epoch": telemetry.causal.client_life_epoch,
        }
        evidence_mismatches = {
            name: (getattr(evidence, name), wanted)
            for name, wanted in expected_evidence.items()
            if getattr(evidence, name) != wanted
        }
        if (
            not isinstance(evidence.l1_index, tuple)
            or len(evidence.l1_index) != 3
            or any(type(value) is not int for value in evidence.l1_index)
        ):
            evidence_mismatches["l1_index"] = (
                evidence.l1_index, "exact integer (x, y, z) tuple"
            )
        for name in ("client_epoch", "cost_to_safety_q8", "hazard_types",
                     "hazard_severity", "atlas_region_id", "confidence",
                     "hazard_component_id", "hazard_component_epoch"):
            value = getattr(evidence, name)
            if type(value) is not int or value < 0:
                evidence_mismatches[name] = (value, "nonnegative integer")
        if evidence.cost_to_safety_q8 > 0xFFFFFFFF:
            evidence_mismatches["cost_to_safety_q8"] = (
                evidence.cost_to_safety_q8, "u32 Q8 or u32::MAX unreachable"
            )
        if type(evidence.signed_safe_clearance_q8) is not int:
            evidence_mismatches["signed_safe_clearance_q8"] = (
                evidence.signed_safe_clearance_q8, "signed integer Q8"
            )
        elif not -(1 << 31) <= evidence.signed_safe_clearance_q8 < (1 << 31):
            evidence_mismatches["signed_safe_clearance_q8"] = (
                evidence.signed_safe_clearance_q8, "signed i32 Q8 or sentinel"
            )
        if evidence.confidence > 0xFFFF:
            evidence_mismatches["confidence"] = (
                evidence.confidence, "u16 fixed-point confidence"
            )
        if bool(evidence.hazard_component_id) != bool(
            evidence.hazard_component_epoch
        ):
            evidence_mismatches["hazard_component_identity"] = (
                (evidence.hazard_component_id, evidence.hazard_component_epoch),
                "both zero or both positive",
            )
        if (
            evidence.hazard_component_id
            and evidence.hazard_component_epoch != expected_map_epoch
        ):
            evidence_mismatches["hazard_component_epoch"] = (
                evidence.hazard_component_epoch,
                "exact attested map_epoch",
            )
        if (
            evidence.hazard_types
            or evidence.hazard_severity
            or evidence.cost_to_safety_q8 not in (0, 0xFFFFFFFF)
        ) and not evidence.hazard_component_id:
            evidence_mismatches["hazard_component_id"] = (
                evidence.hazard_component_id,
                "positive for hazard or finite recovery-cost evidence",
            )
        if evidence_mismatches:
            details = "; ".join(
                f"{name}={found!r} expected {wanted!r}"
                for name, (found, wanted) in sorted(evidence_mismatches.items())
            )
            raise MultiresAdmissionError(
                f"private spatial reward attestation failed: {details}"
            )
    elif evidence is not None:
        raise MultiresAdmissionError(
            "episode projection must not carry reward-bearing spatial evidence"
        )
    return frame.spatial


@dataclass(frozen=True)
class _PendingHookAdmission:
    client_life_epoch: int
    map_epoch: int
    attempt_tick: int
    action_generation: int
    hazard_component_id: int
    hazard_component_epoch: int
    hook_zone_id: int = 0


class CausalRewardAdmission:
    """Stateful per-client admission for delayed authoritative hook outcomes."""

    def __init__(self) -> None:
        self._pending: dict[str, _PendingHookAdmission] = {}

    def reset(self) -> None:
        self._pending.clear()

    def reset_client(self, client_id: str) -> None:
        if not client_id:
            raise MultiresAdmissionError(
                "causal admission reset requires a client identity"
            )
        self._pending.pop(client_id, None)

    def admit(
        self,
        private: CausalTelemetry,
        spatial_reward: PrivateSpatialRewardEvidence,
        info: Mapping[str, Any],
    ) -> CausalRewardFrame:
        return _admit_causal_reward_frame(private, spatial_reward, info, self)


def _admit_causal_reward_frame(
    private: CausalTelemetry,
    spatial_reward: PrivateSpatialRewardEvidence,
    info: Mapping[str, Any],
    admission: CausalRewardAdmission,
) -> CausalRewardFrame:
    """Convert one frozen QM3C block after exact synchronized echo agreement."""
    if not isinstance(private, CausalTelemetry):
        raise MultiresAdmissionError("private QM3C causal telemetry is missing")
    if not isinstance(spatial_reward, PrivateSpatialRewardEvidence):
        raise MultiresAdmissionError("private raw spatial reward evidence is missing")
    identity = {
        "tick": info.get("server_frame"),
        "echo_tick": info.get("authoritative_echo_tick"),
        "action_generation": info.get("authoritative_action_generation"),
    }
    for name, wanted in identity.items():
        if getattr(private, name) != wanted:
            raise MultiresAdmissionError(
                f"private causal {name}={getattr(private, name)!r} expected {wanted!r}"
            )
    attestation = info.get(SPATIAL_ATTESTATION_INFO_KEY)
    if not isinstance(attestation, Mapping):
        raise MultiresAdmissionError("spatial attestation is missing")
    spatial_identity = {
        "client_id": info.get("client_id"),
        "client_slot": info.get("client_slot"),
        "map_name": info.get("map"),
        "server_frame": info.get("server_frame"),
        "map_epoch": info.get("authoritative_map_epoch"),
        "client_epoch": private.client_life_epoch,
    }
    for name, wanted in spatial_identity.items():
        if getattr(spatial_reward, name) != wanted:
            raise MultiresAdmissionError(
                f"private spatial reward {name} differs from batch authority"
            )
    if (
        spatial_reward.hazard_component_id
        and spatial_reward.hazard_component_epoch
        != info.get("authoritative_map_epoch")
    ):
        raise MultiresAdmissionError(
            "hazard component epoch differs from attested map epoch"
        )
    if attestation.get("map_epoch") != info.get("authoritative_map_epoch"):
        raise MultiresAdmissionError("provider and synchronized batch map epochs differ")
    if info.get("authoritative_echo_valid") is not True:
        raise MultiresAdmissionError("causal reward requires an authoritative echo")
    if info.get("trainable_transition") is not True:
        raise MultiresAdmissionError("causal reward requires a trainable transition")
    if not (
        private.echo_valid
        and private.facts_complete
        and private.transition_trainable
        and info.get("causal_echo_valid") is True
        and info.get("causal_facts_complete") is True
        and info.get("causal_transition_trainable") is True
    ):
        raise MultiresAdmissionError(
            "QM3C echo-valid, facts-complete, and transition-trainable are required"
        )
    unreachable_clearance = spatial_reward.signed_safe_clearance_q8 in (
        SIGNED_CLEARANCE_UNREACHABLE_SAFE,
        SIGNED_CLEARANCE_UNREACHABLE_HAZARD,
    )
    penalty_only_terminal = bool(
        private.has(CausalFlags.ENV_DEATH)
        or info.get("terminated") is True
    )
    if unreachable_clearance and not penalty_only_terminal:
        raise MultiresAdmissionError(
            "unreachable signed-clearance sentinel makes reward transition nontrainable"
        )
    if bool(info.get("action_state_resync") or info.get("map_epoch_resync")):
        raise MultiresAdmissionError("resynchronization frames cannot carry reward")
    flags = private.flags
    requested_hook_fire = int(info.get("action_debug_hook", 0)) == 1
    hook_active = private.has(CausalFlags.HOOK_ATTEMPTED)
    hook_attached = private.has(CausalFlags.HOOK_ATTACHED)
    hook_valid = private.has(CausalFlags.HOOK_VALID)
    hook_invalid = private.has(CausalFlags.HOOK_INVALID)
    hook_necessity_known = private.has(CausalFlags.HOOK_NECESSITY_KNOWN)
    hook_was_necessary = private.has(CausalFlags.HOOK_WAS_NECESSARY)
    hook_outcome = bool(
        hook_attached
        or hook_valid
        or hook_invalid
        or hook_necessity_known
        or hook_was_necessary
    )
    new_hook_attempt = bool(
        hook_active
        and private.hook_attempt_tick == info.get("authoritative_echo_tick")
    )
    pending_continuation = bool(
        hook_active and not new_hook_attempt and not hook_outcome
    )
    delayed_outcome = bool(
        hook_active and not new_hook_attempt and hook_outcome
    )
    client_id = str(info.get("client_id", ""))
    map_epoch = int(info.get("authoritative_map_epoch", -1))
    pending = admission._pending.get(client_id)
    pending_replacement = pending
    if pending is not None and (
        pending.client_life_epoch != private.client_life_epoch
        or pending.map_epoch != map_epoch
    ):
        if delayed_outcome or pending_continuation:
            raise MultiresAdmissionError(
                "delayed hook outcome crossed a life or map boundary"
            )
        pending = None
        pending_replacement = None

    if new_hook_attempt:
        if not requested_hook_fire:
            raise MultiresAdmissionError(
                "QM3C accepted hook attempt lacks a hook-fire action echo"
            )
        if (
            private.hook_action_generation
            != info.get("authoritative_action_generation")
        ):
            raise MultiresAdmissionError(
                "hook attempt origin differs from the current echoed request"
            )
        if pending is not None:
            raise MultiresAdmissionError(
                "new hook attempt overlaps an unresolved pending attempt"
            )
        pending = _PendingHookAdmission(
            client_life_epoch=private.client_life_epoch,
            map_epoch=map_epoch,
            attempt_tick=private.hook_attempt_tick,
            action_generation=private.hook_action_generation,
            hazard_component_id=spatial_reward.hazard_component_id,
            hazard_component_epoch=spatial_reward.hazard_component_epoch,
            hook_zone_id=private.hook_zone_id,
        )
    elif pending_continuation or delayed_outcome:
        if pending is None:
            raise MultiresAdmissionError("orphan delayed hook outcome")
        if (
            private.hook_attempt_tick != pending.attempt_tick
            or private.hook_action_generation != pending.action_generation
        ):
            raise MultiresAdmissionError("stale or mixed delayed hook origin")
        safe_component_transition = bool(
            spatial_reward.hazard_component_id == 0
            and spatial_reward.hazard_component_epoch == 0
            and spatial_reward.cost_to_safety_q8 == 0
            and spatial_reward.signed_safe_clearance_q8 >= 0
        )
        if (
            spatial_reward.hazard_component_id != pending.hazard_component_id
            or spatial_reward.hazard_component_epoch
            != pending.hazard_component_epoch
        ) and not safe_component_transition:
            raise MultiresAdmissionError(
                "delayed hook outcome crossed its Atlas hazard component"
            )
        if (
            pending.hook_zone_id
            and private.hook_zone_id
            and pending.hook_zone_id != private.hook_zone_id
        ):
            raise MultiresAdmissionError("delayed hook outcome changed hook zone")

    elif hook_outcome or hook_active:
        raise MultiresAdmissionError("incoherent hook pending/outcome state")

    if new_hook_attempt or pending_continuation or delayed_outcome:
        assert pending is not None
        zone_id = private.hook_zone_id or pending.hook_zone_id
        pending = _PendingHookAdmission(
            client_life_epoch=pending.client_life_epoch,
            map_epoch=pending.map_epoch,
            attempt_tick=pending.attempt_tick,
            action_generation=pending.action_generation,
            hazard_component_id=pending.hazard_component_id,
            hazard_component_epoch=pending.hazard_component_epoch,
            hook_zone_id=zone_id,
        )
        pending_replacement = (
            None if hook_attached or hook_invalid or hook_necessity_known else pending
        )

    if type(info.get("requested_action_fire")) is not bool:
        raise MultiresAdmissionError(
            "causal fire reward requires the exact echoed policy fire request"
        )
    fire_requested = bool(info["requested_action_fire"])
    fire_suppressed = info.get("fire_gate_suppressed") is True
    fire_executed = info.get("effective_action_fire")
    echoed_fire = info.get("action_debug_fire")
    if type(fire_executed) is not bool or type(echoed_fire) is not bool:
        raise MultiresAdmissionError("effective fire echo must be boolean")
    if fire_executed != echoed_fire:
        raise MultiresAdmissionError(
            "effective fire differs from the authoritative action echo"
        )
    target_hit = private.has(CausalFlags.TARGET_HIT)
    observed_damage = float(info.get("damage_dealt", 0.0))
    if target_hit != (observed_damage > 0.0):
        raise MultiresAdmissionError("QM3C target-hit differs from damage event")
    target_aligned = bool(
        info.get("fire_gate_target") and not info.get("fire_gate_protected")
    )
    if fire_suppressed:
        if not fire_requested or fire_executed or target_aligned:
            raise MultiresAdmissionError(
                "fire suppression disagrees with request/echo/gate authority"
            )
        fire_suppression_reason = (
            FIRE_SUPPRESSION_PROTECTED
            if info.get("fire_gate_protected") is True
            else FIRE_SUPPRESSION_UNALIGNED
        )
    else:
        if fire_requested != fire_executed:
            raise MultiresAdmissionError(
                "unsuppressed fire request differs from its action echo"
            )
        fire_suppression_reason = FIRE_SUPPRESSION_NONE
    frame = CausalRewardFrame(
        tick=private.tick,
        client_life_epoch=private.client_life_epoch,
        authoritative_echo_valid=True,
        trainable_transition=True,
        state_resync=False,
        teacher_field_violations=0,
        target_id=private.target_id,
        target_epoch=private.target_epoch,
        actionable_exposure=target_aligned,
        post_command_aligned=target_aligned,
        fire_permitted=target_aligned,
        fire_requested=fire_requested,
        fire_suppressed=fire_suppressed,
        fire_suppression_reason=fire_suppression_reason,
        fire_executed=fire_executed,
        damage_dealt=observed_damage,
        target_killed=private.has(CausalFlags.TARGET_KILLED),
        # Atlas component identity owns episode continuity. QM3C's server
        # event-source identity is preserved separately below.
        hazard_component_id=spatial_reward.hazard_component_id,
        hazard_component_epoch=spatial_reward.hazard_component_epoch,
        environmental_source_id=private.environmental_source_id,
        environmental_source_epoch=private.environmental_source_epoch,
        environmental_mod=private.environmental_mod,
        environmental_hazard_evidence=bool(
            spatial_reward.hazard_types or spatial_reward.hazard_severity
        ),
        cost_to_safety=(
            None if spatial_reward.cost_to_safety_q8 == 0xFFFFFFFF
            else float(spatial_reward.cost_to_safety_q8) / 256.0
        ),
        safe_clearance=(
            0.0 if unreachable_clearance
            else float(spatial_reward.signed_safe_clearance_q8) / 256.0
        ),
        environmental_source_cleared=private.has(
            CausalFlags.ENV_SOURCE_CLEARED
        ),
        environmental_damage=float(private.environmental_damage),
        environmental_death=private.has(CausalFlags.ENV_DEATH),
        hook_attempted=new_hook_attempt,
        hook_pending=bool(
            (new_hook_attempt or pending_continuation)
            and not hook_attached
            and not hook_invalid
            and not hook_necessity_known
        ),
        hook_attached=hook_attached,
        hook_valid=hook_valid,
        hook_invalid=hook_invalid,
        hook_was_necessary=hook_was_necessary,
        hook_necessity_known=hook_necessity_known,
        hook_zone_id=private.hook_zone_id,
        hook_attempt_tick=private.hook_attempt_tick,
        hook_action_generation=private.hook_action_generation,
        action_generation=private.action_generation,
        requested_vertical=int(info["action_debug_vertical_intent"]),
        actual_ducked=bool(info["action_debug_actual_ducked"]),
        standing_blocked=bool(info["standing_blocked"]),
        water_vertical_mode=bool(info["action_debug_water_vertical_mode"]),
        crouch_edge_id=private.crouch_edge_id,
        crouch_edge_epoch=private.crouch_edge_epoch,
        crouch_edge_active=private.has(CausalFlags.CROUCH_EDGE_ACTIVE),
        crouch_edge_entered=private.has(CausalFlags.CROUCH_EDGE_ENTERED),
        crouch_edge_completed=private.has(CausalFlags.CROUCH_EDGE_COMPLETED),
    )
    frame.validate()
    if pending_replacement is None:
        admission._pending.pop(client_id, None)
    elif (
        pending_replacement is not pending
        or new_hook_attempt
        or pending_continuation
        or delayed_outcome
    ):
        admission._pending[client_id] = pending_replacement
    return frame


def admit_causal_reward_frame(
    private: CausalTelemetry,
    spatial_reward: PrivateSpatialRewardEvidence,
    info: Mapping[str, Any],
) -> CausalRewardFrame:
    """One-frame convenience admission; delayed outcomes require explicit state."""
    return CausalRewardAdmission().admit(private, spatial_reward, info)


def detach_private_causal(info: dict[str, Any]) -> CausalTelemetry:
    """Pop private telemetry so it cannot leak to policy/metrics serializers."""
    try:
        value = info.pop(PRIVATE_CAUSAL_INFO_KEY)
    except KeyError as error:
        raise MultiresAdmissionError(
            "accepted multires transition lacks private causal telemetry"
        ) from error
    if not isinstance(value, CausalTelemetry):
        raise MultiresAdmissionError("private causal telemetry is malformed")
    return value


def detach_private_spatial_reward(
    info: dict[str, Any],
) -> PrivateSpatialRewardEvidence:
    try:
        value = info.pop(PRIVATE_SPATIAL_REWARD_INFO_KEY)
    except KeyError as error:
        raise MultiresAdmissionError(
            "accepted transition lacks private spatial reward evidence"
        ) from error
    if not isinstance(value, PrivateSpatialRewardEvidence):
        raise MultiresAdmissionError("private spatial reward evidence is malformed")
    return value


def validate_safe_clearance_rearm_capability(
    frames: Sequence[PrivateSpatialRewardEvidence],
    *,
    threshold: float,
    consecutive_ticks: int,
) -> float:
    """Conformance gate proving real raw clearance can rearm hazard rewards."""
    if not math.isfinite(float(threshold)) or threshold <= 0.0:
        raise ValueError("safe-clearance threshold must be finite and positive")
    if consecutive_ticks < 1:
        raise ValueError("consecutive_ticks must be positive")
    if not frames:
        raise MultiresAdmissionError("safe-clearance conformance has no frames")
    identity = None
    run = 0
    maximum = -math.inf
    previous_tick = None
    for frame in frames:
        if not isinstance(frame, PrivateSpatialRewardEvidence):
            raise MultiresAdmissionError("safe-clearance fixture is not private evidence")
        current_identity = (
            frame.client_id, frame.client_slot, frame.map_name,
            frame.map_epoch, frame.client_epoch,
        )
        if identity is None:
            identity = current_identity
        elif current_identity != identity:
            raise MultiresAdmissionError(
                "safe-clearance conformance crossed provider identity fences"
            )
        clearance = float(frame.signed_safe_clearance_q8) / 256.0
        if frame.signed_safe_clearance_q8 in (
            SIGNED_CLEARANCE_UNREACHABLE_SAFE,
            SIGNED_CLEARANCE_UNREACHABLE_HAZARD,
        ):
            raise MultiresAdmissionError(
                "safe-clearance conformance contains an unreachable sentinel"
            )
        maximum = max(maximum, clearance)
        consecutive_frame = (
            previous_tick is None or frame.server_frame == previous_tick + 1
        )
        run = run + 1 if consecutive_frame and clearance >= threshold else (
            1 if clearance >= threshold else 0
        )
        previous_tick = frame.server_frame
        if run >= consecutive_ticks:
            return maximum
    raise MultiresAdmissionError(
        "raw signed safe clearance never crossed the configured rearm threshold "
        f"for {consecutive_ticks} consecutive ticks (maximum={maximum})"
    )
