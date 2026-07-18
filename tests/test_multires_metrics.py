import pytest

from harness.multires_lineage import LineageError
from harness.multires_metrics import MultiresSeasonMetrics, SEASON_SCHEMA
from harness.multires_reward import CausalRewardFrame, CausalRewardReducer


def test_season_report_contains_causal_combat_posture_hook_and_privilege_fields():
    reducer = CausalRewardReducer()
    frame = CausalRewardFrame(
        tick=1,
        client_life_epoch=1,
        authoritative_echo_valid=True,
        trainable_transition=True,
        action_generation=1,
        target_id=2,
        target_epoch=3,
        actionable_exposure=True,
        post_command_aligned=True,
        fire_permitted=True,
        fire_requested=True,
        fire_executed=True,
        damage_dealt=10.0,
        aim_yaw_error_deg=4.0,
        aim_pitch_error_deg=2.0,
    )
    result = reducer.step(frame)
    season = MultiresSeasonMetrics("season-a", "a" * 64, 10)
    season.observe(
        frame,
        result,
        command_echo_match=True,
        guide_dropped=(True, False, False, False),
        guide_classes=(0, 1, None, None),
        global_guide_drop=False,
        movement_speed=240.0,
        forward_command=1.0,
        true_view_pitch_deg=20.0,
    )
    season.observe_runtime_snapshot(
        atlas_loaded=True,
        atlas_hash_match=True,
        atlas_resident_bytes=1024,
        atlas_build_peak_rss_bytes=4096,
        atlas_cell_count=100,
        atlas_chunk_count=4,
        atlas_deserialize_ms=2.0,
        query_timings_us={
            "dyn_query_us": 1.0,
            "atlas_lookup_us": 2.0,
            "recovery_query_us": 3.0,
            "guide_query_us": 4.0,
        },
        dyn_cell_count=10,
        live_thermal_tracks=2,
        expired_thermal_tracks=1,
        dyn_snapshot_bytes=128,
    )
    report = season.report(policy_end_version=11)

    assert report["schema"] == SEASON_SCHEMA
    assert report["transport"]["command_echo_match_rate"] == 1.0
    assert report["privilege"]["teacher_field_violations"] == 0
    assert report["combat"]["actionable_exposure"] == 1
    assert report["combat"]["post_command_alignment"] == 1
    assert report["combat"]["fire_permission"] == 1
    assert report["combat"]["executed_fire"] == 1
    assert report["combat"]["hits"] == 1
    assert report["combat"]["visible_contact_yaw_mae_deg"] == 4.0
    assert report["guides"]["per_class_drop_rate"]["weapon"] == 1.0
    assert report["movement"]["speed_mean"] == 240.0
    assert report["movement"]["downlook_over_15deg_rate"] == 1.0
    assert report["atlas"]["hash_failures"] == 0
    assert report["atlas"]["query_p99_us"]["guide_query_us"] == 4.0
    assert report["dyn"]["thermal_checkpoint_fields"] == 0


def test_noncausal_or_teacher_leaking_transition_cannot_enter_season():
    reducer = CausalRewardReducer()
    frame = CausalRewardFrame(
        tick=1, client_life_epoch=1,
        authoritative_echo_valid=True, trainable_transition=True,
        action_generation=1,
    )
    result = reducer.step(frame)
    season = MultiresSeasonMetrics("season", "a" * 64, 0)
    with pytest.raises(LineageError, match="teacher-only"):
        season.observe(
            frame, result, command_echo_match=True, teacher_field_violations=1
        )
    with pytest.raises(LineageError, match="noncausal"):
        season.observe(frame, result, command_echo_match=False)
