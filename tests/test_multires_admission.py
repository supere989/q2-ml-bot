from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from harness.causal_protocol import CausalFlags, CausalTelemetry
from harness.multires_admission import (
    SPATIAL_ATTESTATION_INFO_KEY,
    SPATIAL_PROVIDER_SCHEMA,
    SPATIAL_REWARD_EVIDENCE_SCHEMA,
    MultiresAdmissionError,
    CausalRewardAdmission,
    PrivateSpatialRewardEvidence,
    SpatialProviderFrame,
    admit_causal_reward_frame,
    validate_spatial_provider_frame,
    validate_safe_clearance_rearm_capability,
)
from harness.multires_contract import FEATURE_SCHEMA_SHA256
from harness.multires_reward import CausalRewardConfig, CausalRewardReducer
from harness.protocol import SpatialPolicyFeatures


ATLAS = "a" * 64
RUNTIME = "b" * 64


def causal(flags=None):
    return CausalTelemetry(
        tick=10,
        client_life_epoch=3,
        target_id=0,
        target_epoch=0,
        environmental_source_id=0,
        environmental_source_epoch=0,
        environmental_mod=0,
        environmental_damage=0,
        crouch_edge_id=0,
        crouch_edge_epoch=0,
        echo_tick=9,
        action_generation=2,
        hook_zone_id=0,
        hook_attempt_tick=0,
        hook_action_generation=0,
        flags=flags if flags is not None else (
            CausalFlags.ECHO_VALID
            | CausalFlags.FACTS_COMPLETE
            | CausalFlags.TRANSITION_TRAINABLE
        ),
    )


def evidence(**changes):
    values = dict(
        schema=SPATIAL_REWARD_EVIDENCE_SCHEMA,
        client_id="c0",
        client_slot=1,
        map_name="q2dm1",
        map_epoch=4,
        server_frame=10,
        client_epoch=3,
        l1_index=(7, 8, 9),
        cost_to_safety_q8=0xFFFFFFFF,
        signed_safe_clearance_q8=256,
        hazard_types=0,
        hazard_severity=0,
        atlas_region_id=91,
        confidence=0xFFFF,
        hazard_component_id=0,
        hazard_component_epoch=0,
    )
    values.update(changes)
    return PrivateSpatialRewardEvidence(**values)


def telemetry():
    return SimpleNamespace(
        client_id="c0", client_slot=1, map_name="q2dm1", server_frame=10,
        causal=causal(),
    )


def spatial_frame(**changes):
    values = dict(
        schema=SPATIAL_PROVIDER_SCHEMA,
        feature_schema_sha256=FEATURE_SCHEMA_SHA256,
        atlas_sha256=ATLAS,
        runtime_manifest_sha256=RUNTIME,
        client_id="c0",
        client_slot=1,
        map_name="q2dm1",
        map_epoch=4,
        server_frame=10,
        spatial=SpatialPolicyFeatures(
            dyn=np.zeros(24, dtype=np.float32),
            recovery=np.zeros(16, dtype=np.float32),
            objectives=np.zeros((4, 15), dtype=np.float32),
        ),
        private_reward_evidence=evidence(),
    )
    values.update(changes)
    return SpatialProviderFrame(**values)


def info():
    return {
        "client_id": "c0",
        "client_slot": 1,
        "map": "q2dm1",
        "server_frame": 10,
        "action_tick": 9,
        "authoritative_echo_tick": 9,
        "authoritative_action_generation": 2,
        "authoritative_map_epoch": 4,
        "authoritative_echo_valid": True,
        "trainable_transition": True,
        "causal_echo_valid": True,
        "causal_facts_complete": True,
        "causal_transition_trainable": True,
        "fire_gate_target": False,
        "fire_gate_protected": False,
        "effective_action_fire": False,
        "action_debug_fire": False,
        "requested_action_fire": False,
        "fire_gate_suppressed": False,
        "action_debug_hook": 0,
        "damage_dealt": 0.0,
        "action_debug_vertical_intent": 1,
        "action_debug_actual_ducked": False,
        "standing_blocked": False,
        "action_debug_water_vertical_mode": False,
        SPATIAL_ATTESTATION_INFO_KEY: {"map_epoch": 4},
    }


def test_provider_requires_exact_map_epoch_and_raw_recovery_evidence():
    result = validate_spatial_provider_frame(
        spatial_frame(), telemetry(),
        expected_atlas_sha256=ATLAS,
        expected_runtime_manifest_sha256=RUNTIME,
        expected_map_epoch=4,
        require_private_reward_evidence=True,
    )
    assert result.to_vector().shape == (100,)
    with pytest.raises(MultiresAdmissionError, match="map_epoch"):
        validate_spatial_provider_frame(
            spatial_frame(map_epoch=3), telemetry(),
            expected_atlas_sha256=ATLAS,
            expected_runtime_manifest_sha256=RUNTIME,
            expected_map_epoch=4,
            require_private_reward_evidence=True,
        )
    with pytest.raises(MultiresAdmissionError, match="raw spatial reward"):
        validate_spatial_provider_frame(
            spatial_frame(private_reward_evidence=None), telemetry(),
            expected_atlas_sha256=ATLAS,
            expected_runtime_manifest_sha256=RUNTIME,
            expected_map_epoch=4,
            require_private_reward_evidence=True,
        )


def test_qm3c_and_b3_private_evidence_merge_without_normalized_placeholders():
    admitted = admit_causal_reward_frame(causal(), evidence(), info())
    assert admitted.action_generation == 2
    assert admitted.cost_to_safety is None  # explicit u32::MAX unreachable
    assert admitted.safe_clearance == 1.0  # signed raw Q8, not Recovery16
    # Atlas SCC identity and server event-source identity are separate domains.
    assert evidence().atlas_region_id == 91


def test_missing_causal_facts_or_provider_epoch_mismatch_fails_closed():
    incomplete = causal(CausalFlags.ECHO_VALID)
    with pytest.raises(MultiresAdmissionError, match="facts-complete"):
        admit_causal_reward_frame(incomplete, evidence(), info())
    wrong_epoch = evidence(client_epoch=2)
    with pytest.raises(MultiresAdmissionError, match="client_epoch"):
        admit_causal_reward_frame(causal(), wrong_epoch, info())


def test_hook_hold_is_not_reinterpreted_as_a_new_hook_attempt():
    values = info()
    values["action_debug_hook"] = 2
    admitted = admit_causal_reward_frame(causal(), evidence(), values)
    assert not admitted.hook_attempted


def test_hook_fire_denied_by_engine_is_admitted_without_hook_event():
    values = info()
    values["action_debug_hook"] = 1
    admitted = admit_causal_reward_frame(causal(), evidence(), values)
    assert not admitted.hook_attempted


def test_hook_same_frame_and_delayed_outcomes_bind_exact_pending_origin():
    base_flags = (
        CausalFlags.ECHO_VALID
        | CausalFlags.FACTS_COMPLETE
        | CausalFlags.TRANSITION_TRAINABLE
    )
    spatial = evidence(
        cost_to_safety_q8=10 * 256,
        signed_safe_clearance_q8=-256,
        hazard_types=1,
        hazard_severity=1,
        hazard_component_id=44,
        hazard_component_epoch=4,
    )

    same_frame = CausalRewardAdmission()
    same_private = replace(
        causal(base_flags | CausalFlags.HOOK_ATTEMPTED
               | CausalFlags.HOOK_ATTACHED | CausalFlags.HOOK_VALID
               | CausalFlags.HOOK_NECESSITY_KNOWN
               | CausalFlags.HOOK_WAS_NECESSARY),
        hook_zone_id=7,
        hook_attempt_tick=9,
        hook_action_generation=2,
    )
    same_info = info()
    same_info["action_debug_hook"] = 1
    admitted = same_frame.admit(same_private, spatial, same_info)
    assert admitted.hook_attempted and admitted.hook_attached
    assert admitted.hook_was_necessary and not admitted.hook_pending

    delayed = CausalRewardAdmission()
    attempt_info = info()
    attempt_info["action_debug_hook"] = 1
    attempt = replace(
        causal(base_flags | CausalFlags.HOOK_ATTEMPTED),
        hook_attempt_tick=9,
        hook_action_generation=2,
    )
    opened = delayed.admit(attempt, spatial, attempt_info)
    assert opened.hook_attempted and opened.hook_pending

    continuation_info = info()
    continuation_info.update({
        "server_frame": 11,
        "action_tick": 10,
        "authoritative_echo_tick": 10,
        "authoritative_action_generation": 3,
        "action_debug_hook": 0,
    })
    continuation = replace(
        attempt,
        tick=11,
        echo_tick=10,
        action_generation=3,
    )
    continued = delayed.admit(
        continuation, replace(spatial, server_frame=11), continuation_info
    )
    assert continued.hook_pending and not continued.hook_attempted

    outcome_info = dict(continuation_info)
    outcome_info.update({
        "server_frame": 12,
        "action_tick": 11,
        "authoritative_echo_tick": 11,
        "authoritative_action_generation": 4,
    })
    outcome = replace(
        continuation,
        tick=12,
        echo_tick=11,
        action_generation=4,
        hook_zone_id=7,
        flags=(base_flags | CausalFlags.HOOK_ATTEMPTED
               | CausalFlags.HOOK_ATTACHED | CausalFlags.HOOK_VALID
               | CausalFlags.HOOK_NECESSITY_KNOWN
               | CausalFlags.HOOK_WAS_NECESSARY),
    )
    resolved = delayed.admit(
        outcome, replace(spatial, server_frame=12), outcome_info
    )
    assert not resolved.hook_attempted and not resolved.hook_pending
    assert resolved.hook_attached and resolved.hook_was_necessary

    replay_info = dict(outcome_info)
    replay_info.update({
        "server_frame": 13,
        "action_tick": 12,
        "authoritative_echo_tick": 12,
        "authoritative_action_generation": 5,
    })
    replay = replace(
        outcome, tick=13, echo_tick=12, action_generation=5
    )
    with pytest.raises(MultiresAdmissionError, match="orphan"):
        delayed.admit(replay, replace(spatial, server_frame=13), replay_info)


def test_delayed_hook_rejects_stale_origin_and_component_mixing():
    flags = (
        CausalFlags.ECHO_VALID | CausalFlags.FACTS_COMPLETE
        | CausalFlags.TRANSITION_TRAINABLE | CausalFlags.HOOK_ATTEMPTED
    )
    spatial = evidence(
        cost_to_safety_q8=256,
        hazard_types=1,
        hazard_component_id=8,
        hazard_component_epoch=4,
    )
    admission = CausalRewardAdmission()
    start_info = info()
    start_info["action_debug_hook"] = 1
    start = replace(
        causal(flags), hook_attempt_tick=9, hook_action_generation=2
    )
    admission.admit(start, spatial, start_info)

    later_info = info()
    later_info.update({
        "server_frame": 11,
        "authoritative_echo_tick": 10,
        "authoritative_action_generation": 3,
    })
    stale = replace(
        start, tick=11, echo_tick=10, action_generation=3,
        hook_attempt_tick=8,
        flags=flags | CausalFlags.HOOK_INVALID,
    )
    with pytest.raises(MultiresAdmissionError, match="stale or mixed"):
        admission.admit(stale, replace(spatial, server_frame=11), later_info)

    mixed = replace(stale, hook_attempt_tick=9)
    mixed_spatial = replace(
        spatial, server_frame=11, hazard_component_id=9
    )
    with pytest.raises(MultiresAdmissionError, match="Atlas hazard component"):
        admission.admit(mixed, mixed_spatial, later_info)


def test_fire_request_suppression_is_admitted_and_rewarded_causally():
    values = info()
    values.update({
        "requested_action_fire": True,
        "fire_gate_suppressed": True,
        "fire_gate_target": False,
        "effective_action_fire": False,
        "action_debug_fire": False,
    })
    frame = admit_causal_reward_frame(causal(), evidence(), values)
    result = CausalRewardReducer().step(frame)
    assert frame.fire_requested and frame.fire_suppressed
    assert result.reward == pytest.approx(-0.025)
    assert result.metrics["combat/hidden_fire"] == 1.0


def test_aligned_executed_fire_and_no_request_receive_no_hidden_fire_penalty():
    target = replace(
        causal(causal().flags | CausalFlags.TARGET_VALID),
        target_id=11,
        target_epoch=2,
    )
    aligned_info = info()
    aligned_info.update({
        "requested_action_fire": True,
        "fire_gate_target": True,
        "effective_action_fire": True,
        "action_debug_fire": True,
    })
    aligned = admit_causal_reward_frame(target, evidence(), aligned_info)
    assert CausalRewardReducer().step(aligned).metrics["combat/hidden_fire"] == 0.0
    idle = admit_causal_reward_frame(causal(), evidence(), info())
    assert CausalRewardReducer().step(idle).metrics["combat/hidden_fire"] == 0.0


def test_environmental_death_with_unreachable_clearance_is_penalty_only():
    death = replace(
        causal(causal().flags | CausalFlags.ENV_SOURCE_EVIDENCE
               | CausalFlags.ENV_DEATH),
        environmental_source_id=91,
        environmental_source_epoch=4,
        environmental_mod=22,
    )
    terminal = evidence(
        signed_safe_clearance_q8=(1 << 31) - 1,
    )
    admitted = admit_causal_reward_frame(death, terminal, info())
    result = CausalRewardReducer().step(admitted)
    assert admitted.cost_to_safety is None
    assert result.reward == pytest.approx(-1.0)
    assert result.metrics["hazard/safe_arrival_credit"] == 0.0


def test_real_provider_must_prove_positive_thirty_tick_safe_rearm_clearance():
    frames = [
        replace(evidence(), server_frame=20 + index)
        for index in range(30)
    ]
    assert validate_safe_clearance_rearm_capability(
        frames, threshold=0.5, consecutive_ticks=30
    ) == 1.0
    zero_clearance = [
        replace(frame, signed_safe_clearance_q8=0) for frame in frames
    ]
    with pytest.raises(MultiresAdmissionError, match="never crossed"):
        validate_safe_clearance_rearm_capability(
            zero_clearance, threshold=0.5, consecutive_ticks=30
        )


def test_admitted_raw_clearance_drives_exact_thirty_tick_rearm_state_machine():
    config = CausalRewardConfig(
        safe_clearance_threshold=0.5, safe_rearm_ticks=30
    )
    reducer = CausalRewardReducer(config)

    def qm3c(tick, event_flags):
        return CausalTelemetry(
            tick=tick,
            client_life_epoch=3,
            target_id=0,
            target_epoch=0,
            environmental_source_id=0,
            environmental_source_epoch=0,
            environmental_mod=0,
            environmental_damage=0,
            crouch_edge_id=0,
            crouch_edge_epoch=0,
            echo_tick=tick - 1,
            action_generation=2,
            hook_zone_id=0,
            hook_attempt_tick=0,
            hook_action_generation=0,
            flags=(CausalFlags.ECHO_VALID | CausalFlags.FACTS_COMPLETE
                   | CausalFlags.TRANSITION_TRAINABLE | event_flags),
        )

    def private(
        tick, clearance_q8, cost_q8=0xFFFFFFFF, *, hazard_types=0
    ):
        component_id = int(bool(hazard_types or cost_q8 not in (0, 0xFFFFFFFF))) * 44
        return replace(
            evidence(),
            server_frame=tick,
            cost_to_safety_q8=cost_q8,
            signed_safe_clearance_q8=clearance_q8,
            hazard_types=hazard_types,
            hazard_severity=int(bool(hazard_types)),
            # The fixed exporter reports the all-zero component identity on
            # safe cells; the reducer must retain the previously active
            # episode while consuming those clearance samples.
            hazard_component_id=component_id,
            hazard_component_epoch=4 if component_id else 0,
        )

    def batch_info(tick):
        values = info()
        values.update({
            "server_frame": tick,
            "action_tick": tick - 1,
            "authoritative_echo_tick": tick - 1,
        })
        return values

    opened = admit_causal_reward_frame(
        qm3c(100, CausalFlags(0)),
        private(100, -256, 10 * 256, hazard_types=1),
        batch_info(100),
    )
    reducer.step(opened)
    arrived = admit_causal_reward_frame(
        qm3c(101, CausalFlags(0)),
        private(101, 256, 0),
        batch_info(101),
    )
    assert reducer.step(arrived).metrics["hazard/safe_arrival_credit"] == 1.0

    for offset in range(29):
        tick = 102 + offset
        frame = admit_causal_reward_frame(
            qm3c(tick, CausalFlags(0)), private(tick, 256), batch_info(tick)
        )
        result = reducer.step(frame)
        assert result.metrics["hazard/rearmed"] == 0.0
    tick = 131
    frame = admit_causal_reward_frame(
        qm3c(tick, CausalFlags(0)), private(tick, 256), batch_info(tick)
    )
    assert reducer.step(frame).metrics["hazard/rearmed"] == 1.0

    with pytest.raises(MultiresAdmissionError, match="sentinel"):
        admit_causal_reward_frame(
            qm3c(132, CausalFlags(0)),
            private(132, (1 << 31) - 1),
            batch_info(132),
        )


def test_qm3c_source_identity_attributes_damage_but_does_not_open_atlas_episode():
    source = replace(
        causal(
            CausalFlags.ECHO_VALID
            | CausalFlags.FACTS_COMPLETE
            | CausalFlags.TRANSITION_TRAINABLE
            | CausalFlags.ENV_SOURCE_EVIDENCE
            | CausalFlags.ENV_DAMAGE
        ),
        environmental_source_id=700,
        environmental_source_epoch=4,
        environmental_mod=22,
        environmental_damage=5,
    )
    frame = admit_causal_reward_frame(source, evidence(), info())
    assert frame.environmental_source_id == 700
    assert frame.environmental_source_epoch == 4
    assert frame.environmental_damage == 5.0
    assert not frame.environmental_hazard_evidence
    result = CausalRewardReducer().step(frame)
    assert result.metrics["hazard/environmental_damage"] == 5.0
    assert result.metrics["hazard/evidence"] == 0.0
