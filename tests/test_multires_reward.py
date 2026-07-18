import pytest

from harness.multires_contract import POSTURE_CROUCH_OR_DOWN, POSTURE_NEUTRAL
from harness.multires_reward import (
    FIRE_SUPPRESSION_UNALIGNED,
    CausalRewardConfig,
    CausalRewardFrame,
    CausalRewardReducer,
    RewardAdmissionError,
)


def frame(tick, **changes):
    values = {
        "tick": tick,
        "client_life_epoch": 7,
        "authoritative_echo_valid": True,
        "trainable_transition": True,
        "action_generation": 1,
    }
    values.update(changes)
    if values.get("hazard_component_id", 0):
        values.setdefault("hazard_component_epoch", 1)
    if any(values.get(name) for name in (
        "environmental_source_cleared", "environmental_damage",
        "environmental_death",
    )):
        values.setdefault("environmental_source_id", 1)
        values.setdefault("environmental_source_epoch", 1)
        values.setdefault("environmental_mod", 1)
    if values.get("crouch_edge_id", 0):
        values.setdefault("crouch_edge_epoch", 1)
    if values.get("hook_was_necessary"):
        values.setdefault("hook_necessity_known", True)
        values.setdefault("hook_valid", True)
        values.setdefault("hook_zone_id", 1)
    if values.get("hook_attached"):
        values.setdefault("hook_valid", True)
    if any(values.get(name) for name in (
        "hook_attempted", "hook_pending", "hook_attached", "hook_valid",
        "hook_invalid", "hook_necessity_known", "hook_was_necessary",
    )):
        values.setdefault("hook_attempt_tick", tick)
        values.setdefault("hook_action_generation", values["action_generation"])
    if values.get("hook_attempted") and not (
        values.get("hook_invalid") or values.get("hook_necessity_known")
    ):
        values.setdefault("hook_pending", True)
    return CausalRewardFrame(**values)


def test_alignment_is_once_per_persistent_target_not_a_dense_tick_reward():
    reducer = CausalRewardReducer()
    first = reducer.step(frame(
        1, target_id=2, target_epoch=9,
        actionable_exposure=True, post_command_aligned=True,
    ))
    held = reducer.step(frame(
        2, target_id=2, target_epoch=9,
        actionable_exposure=True, post_command_aligned=True,
    ))
    switched = reducer.step(frame(
        3, target_id=3, target_epoch=1,
        actionable_exposure=True, post_command_aligned=True,
    ))
    hidden = reducer.step(frame(
        4, fire_requested=True, fire_suppressed=True,
        fire_suppression_reason=FIRE_SUPPRESSION_UNALIGNED,
    ))

    assert first.reward == pytest.approx(0.02)
    assert held.reward == 0.0
    assert switched.reward == pytest.approx(0.02)
    assert hidden.reward == pytest.approx(-0.025)
    assert held.metrics["combat/alignment_acquisition"] == 0.0
    assert CausalRewardReducer.positive_actuator_rate_rewards == ()


def test_persistent_hits_increase_until_switch_or_kill():
    reducer = CausalRewardReducer()
    first = reducer.step(frame(
        1, target_id=2, target_epoch=9, damage_dealt=10.0
    ))
    repeated = reducer.step(frame(
        2, target_id=2, target_epoch=9, damage_dealt=10.0
    ))
    switched = reducer.step(frame(
        3, target_id=3, target_epoch=1, damage_dealt=10.0
    ))
    killed = reducer.step(frame(
        4, target_id=3, target_epoch=1, damage_dealt=10.0, target_killed=True
    ))

    assert first.reward == pytest.approx(0.03)
    assert repeated.reward == pytest.approx(0.04)
    assert repeated.metrics["combat/repeated_hit"] == 1.0
    assert repeated.metrics["combat/hit_streak"] == 2.0
    assert switched.reward == pytest.approx(first.reward)
    assert killed.reward == pytest.approx(1.04)


def test_hit_without_persistent_identity_and_duplicate_tick_fail_closed():
    reducer = CausalRewardReducer()
    with pytest.raises(RewardAdmissionError, match="persistent target"):
        reducer.step(frame(1, damage_dealt=1.0))

    reducer = CausalRewardReducer()
    reducer.step(frame(1))
    with pytest.raises(RewardAdmissionError, match="does not advance"):
        reducer.step(frame(1))


def test_noncausal_transition_and_life_epoch_rollback_fail_closed():
    reducer = CausalRewardReducer()
    with pytest.raises(RewardAdmissionError, match="authoritative-echo"):
        reducer.step(frame(1, trainable_transition=False))
    reducer.step(frame(2, client_life_epoch=8))
    with pytest.raises(RewardAdmissionError, match="cannot roll back"):
        reducer.step(frame(3, client_life_epoch=7))


def test_target_geometry_and_recovery_cost_require_causal_episode_identity():
    reducer = CausalRewardReducer()
    with pytest.raises(RewardAdmissionError, match="target geometry"):
        reducer.step(frame(1, actionable_exposure=True))
    reducer = CausalRewardReducer()
    with pytest.raises(RewardAdmissionError, match="hazard_component_id"):
        reducer.step(frame(1, cost_to_safety=1.0))
    assert CausalRewardReducer().step(
        frame(1, cost_to_safety=0.0, safe_clearance=1.0)
    ).reward == 0.0


def test_hazard_progress_arrival_and_necessary_hook_are_bounded_events():
    reducer = CausalRewardReducer()
    opened = reducer.step(frame(
        1, hazard_component_id=4, environmental_hazard_evidence=True,
        cost_to_safety=100.0,
    ))
    quarter = reducer.step(frame(
        2, hazard_component_id=4, environmental_hazard_evidence=True,
        cost_to_safety=74.0,
    ))
    hooked = reducer.step(frame(
        3, hazard_component_id=4, environmental_hazard_evidence=True,
        cost_to_safety=40.0, hook_attempted=True, hook_valid=True,
        hook_necessity_known=True, hook_was_necessary=True,
    ))
    arrived = reducer.step(frame(
        4, hazard_component_id=4, cost_to_safety=0.0,
        safe_clearance=1.0,
    ))
    replay = reducer.step(frame(
        5, hazard_component_id=4, environmental_hazard_evidence=True,
        cost_to_safety=100.0,
    ))

    assert opened.reward == 0.0
    assert quarter.reward == pytest.approx(0.04)
    assert hooked.reward == pytest.approx(0.04)
    assert arrived.reward == pytest.approx(0.31)
    assert arrived.metrics["hook/necessary_safe_arrival_credit"] == 1.0
    assert arrived.metrics["hook/safe_arrival_with_hook"] == 1.0
    assert arrived.metrics["hazard/recovery_ticks"] == 3.0
    assert replay.reward == 0.0


def test_timeout_retains_highwater_until_thirty_safe_ticks_rearm_region():
    config = CausalRewardConfig(hazard_timeout_ticks=3, safe_rearm_ticks=30)
    reducer = CausalRewardReducer(config)
    reducer.step(frame(
        1, hazard_component_id=4, environmental_hazard_evidence=True,
        cost_to_safety=100.0,
    ))
    progress = reducer.step(frame(
        2, hazard_component_id=4, environmental_hazard_evidence=True,
        cost_to_safety=70.0,
    ))
    timeout = reducer.step(frame(4, hazard_component_id=4, cost_to_safety=70.0))
    blocked = reducer.step(frame(
        5, hazard_component_id=4, environmental_hazard_evidence=True,
        cost_to_safety=100.0,
    ))
    assert progress.reward == pytest.approx(0.04)
    assert timeout.metrics["hazard/timeout"] == 1.0
    assert blocked.reward == 0.0

    for tick in range(6, 36):
        result = reducer.step(frame(
            tick, hazard_component_id=4, safe_clearance=1.0
        ))
    assert result.metrics["hazard/rearmed"] == 1.0
    reopened = reducer.step(frame(
        36, hazard_component_id=4, environmental_hazard_evidence=True,
        cost_to_safety=100.0,
    ))
    assert reopened.reward == 0.0
    fresh_progress = reducer.step(frame(
        37, hazard_component_id=4, environmental_hazard_evidence=True,
        cost_to_safety=70.0,
    ))
    assert fresh_progress.reward == pytest.approx(0.04)


def test_hook_attempt_has_no_positive_rate_reward_and_invalid_hook_is_penalized():
    reducer = CausalRewardReducer()
    attempt = reducer.step(frame(1, hook_attempted=True))
    invalid = reducer.step(frame(
        2, hook_invalid=True, hook_attempt_tick=1,
        hook_action_generation=1,
    ))
    assert attempt.reward == 0.0
    assert invalid.reward == pytest.approx(-0.02)


def test_delayed_hook_necessity_label_only_pays_after_valid_hook_and_safe_arrival():
    reducer = CausalRewardReducer()
    reducer.step(frame(
        1, hazard_component_id=5, environmental_hazard_evidence=True,
        cost_to_safety=10.0,
    ))
    reducer.step(frame(
        2, hazard_component_id=5, environmental_hazard_evidence=True,
        cost_to_safety=10.0, hook_attempted=True,
    ))
    arrived = reducer.step(frame(
        3, hazard_component_id=5, cost_to_safety=0.0,
        hook_attached=True, hook_valid=True, hook_necessity_known=True,
        hook_was_necessary=True,
        hook_attempt_tick=2, hook_action_generation=1,
    ))
    assert arrived.metrics["hook/necessary_safe_arrival_credit"] == 1.0


def test_crouch_pays_once_for_validated_edge_outcomes_only():
    reducer = CausalRewardReducer()
    unnecessary = reducer.step(frame(
        1, requested_vertical=POSTURE_CROUCH_OR_DOWN, actual_ducked=True
    ))
    held = reducer.step(frame(
        2, requested_vertical=POSTURE_CROUCH_OR_DOWN, actual_ducked=True
    ))
    reducer.step(frame(3, requested_vertical=POSTURE_NEUTRAL, actual_ducked=False))
    entered = reducer.step(frame(
        4, requested_vertical=POSTURE_CROUCH_OR_DOWN, actual_ducked=True,
        crouch_edge_id=12, crouch_edge_active=True, crouch_edge_entered=True,
    ))
    completed = reducer.step(frame(
        5, requested_vertical=POSTURE_CROUCH_OR_DOWN, actual_ducked=True,
        crouch_edge_id=12, crouch_edge_completed=True,
    ))
    replayed = reducer.step(frame(
        6, requested_vertical=POSTURE_CROUCH_OR_DOWN, actual_ducked=True,
        crouch_edge_id=12, crouch_edge_completed=True,
    ))

    assert unnecessary.reward == pytest.approx(-0.01)
    assert held.reward == 0.0
    assert entered.reward == pytest.approx(0.03)
    assert completed.reward == pytest.approx(0.06)
    assert replayed.reward == 0.0


def test_persistent_crouch_edge_id_does_not_imply_edge_is_active():
    reducer = CausalRewardReducer()
    reducer.step(frame(1, crouch_edge_id=12, actual_ducked=False))
    entry = reducer.step(frame(
        2, crouch_edge_id=12, crouch_edge_active=False, actual_ducked=True,
        requested_vertical=POSTURE_CROUCH_OR_DOWN,
    ))
    assert entry.reward == pytest.approx(-0.01)
    assert entry.metrics["posture/unnecessary_crouch_entry"] == 1.0
