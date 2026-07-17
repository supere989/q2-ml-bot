import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from harness.causal_protocol import CausalFlags, CausalTelemetry
from harness.client_protocol import ClientTelemetry
from harness.multires_query_source import (
    OBJECTIVES_SCHEMA,
    OwnObservationQuerySource,
    QuerySourceError,
    THERMAL_TTL_TICKS,
)
from harness.protocol import ActionDebugIndex


ATLAS = "a" * 64
MAP = "q2dm1"
CLIENT = "client-17"


def write_objectives(
    tmp_path, records=None, *, atlas=ATLAS, map_id=MAP, origin=(0, 0, 0)
):
    if records is None:
        records = [
            {
                "class": "health", "classname": "item_health", "confidence": 65535,
                "l1_index": [0, 0, 0], "objective_id": 5, "risk": 0,
                "world_milliunits": [0, 0, 0],
            },
            {
                "class": "armor", "classname": "item_armor_combat",
                "confidence": 65535, "l1_index": [4, 0, 0], "objective_id": 9,
                "risk": 0, "world_milliunits": [512000, 0, 0],
            },
            {
                "class": "spawn_egress", "classname": "info_player_deathmatch",
                "confidence": 65535, "l1_index": [8, 0, 0], "objective_id": 12,
                "risk": 0, "world_milliunits": [1024000, 0, 0],
            },
        ]
    document = {
        "atlas_sha256": atlas, "bsp_sha256": "c" * 64,
        "canonical_map_id": map_id, "objectives": records,
        "origin": list(origin), "schema": OBJECTIVES_SCHEMA,
    }
    raw = json.dumps(document, sort_keys=True).encode() + b"\n"
    path = tmp_path / f"{MAP}.objectives.json"
    path.write_bytes(raw)
    return path, hashlib.sha256(raw).hexdigest()


def source(tmp_path, records=None, *, atlas_origin=(0, 0, 0)):
    path, digest = write_objectives(tmp_path, records, origin=atlas_origin)
    return OwnObservationQuerySource(
        objectives_path=path,
        expected_objectives_sha256=digest,
        expected_atlas_sha256=ATLAS,
        atlas_origin=atlas_origin,
        map_name=MAP,
        client_id=CLIENT,
    )


def telemetry(
    *,
    frame=10,
    life_epoch=3,
    position=(300.0, 300.0, 0.0),
    health=100.0,
    armor=0.0,
    ammo=10.0,
    pickup=0.0,
    damage_dealt=0.0,
    damage_taken=0.0,
    kill=0.0,
    death=0.0,
    fire=False,
    fire_accepted=False,
    enemy=None,
    yaw=0.0,
    map_name=MAP,
    client_id=CLIENT,
    causal_flags=(CausalFlags.ECHO_VALID | CausalFlags.FACTS_COMPLETE
                  | CausalFlags.TRANSITION_TRAINABLE),
):
    entities = np.zeros((8, 9), dtype=np.float32)
    entity_debug = np.zeros((8, 4), dtype=np.float32)
    entity_count = 0
    if enemy is not None:
        entities[0, :3] = enemy.get("local", (100.0, 0.0, 0.0))
        entities[0, 6] = enemy.get("health", 80.0)
        entities[0, 7] = 1.0
        entities[0, 8] = enemy.get("exposure", 0.75)
        entity_debug[0, 0] = enemy.get("edict", 7)
        entity_debug[0, 3] = enemy.get("epoch_bits", 0)
        entity_count = 1
    action_debug = np.zeros(15, dtype=np.float32)
    action_debug[ActionDebugIndex.ACCEPTED] = float(fire_accepted)
    action_debug[ActionDebugIndex.FIRE] = float(fire)
    observation = SimpleNamespace(
        self_state=np.array(
            [*position, 0.0, 0.0, 0.0, health, armor, 1.0, ammo],
            dtype=np.float32,
        ),
        yaw=yaw,
        pitch=0.0,
        entities=entities,
        entity_count=entity_count,
        entity_debug=entity_debug,
        self_debug=np.zeros(4, dtype=np.float32),
        rune_flags=np.zeros(5, dtype=np.float32),
        reward_item_pickup=pickup,
        reward_damage_dealt=damage_dealt,
        reward_damage_taken=damage_taken,
        reward_kill=kill,
        reward_death=death,
        action_debug=action_debug,
    )
    causal = CausalTelemetry(
        tick=frame, client_life_epoch=life_epoch, target_id=0, target_epoch=0,
        environmental_source_id=0, environmental_source_epoch=0,
        environmental_mod=0, environmental_damage=0, crouch_edge_id=0,
        crouch_edge_epoch=0, echo_tick=frame - 1, action_generation=2,
        hook_zone_id=0, hook_attempt_tick=0, hook_action_generation=0,
        flags=causal_flags,
    )
    return ClientTelemetry(
        sequence=1, client_slot=2, server_frame=frame, client_id=client_id,
        map_name=map_name, observation=observation, causal=causal,
    )


def belief_map(inputs):
    return dict(inputs.objective_beliefs)


def test_map_start_priors_report_every_objective_available(tmp_path):
    inputs = source(tmp_path).sample(telemetry(frame=10), map_epoch=4)
    assert belief_map(inputs) == {5: 1.0, 9: 1.0, 12: 1.0}
    assert inputs.thermal is None
    assert inputs.blocked_nodes == ()
    assert inputs.dynamic_penalties == ()
    assert inputs.enabled_mover_blockers == ()
    assert inputs.time_to_impact_seconds is None


def test_own_pickup_zeroes_then_ramps_along_predicted_respawn(tmp_path):
    src = source(tmp_path)
    # Standing on the health objective (id 5) at the origin, pickup fires.
    consumed = src.sample(
        telemetry(frame=100, position=(0.0, 0.0, 0.0), pickup=1.0), map_epoch=4
    )
    assert belief_map(consumed)[5] == 0.0
    assert belief_map(consumed)[9] == 1.0
    # Health respawn is 30 s at 10 Hz: half period after 150 ticks.
    halfway = src.sample(telemetry(frame=250), map_epoch=4)
    assert belief_map(halfway)[5] == pytest.approx(0.5)
    respawned = src.sample(telemetry(frame=400), map_epoch=4)
    assert belief_map(respawned)[5] == 1.0


def test_unseen_opponent_pickup_never_becomes_exact_knowledge(tmp_path):
    src = source(tmp_path)
    # An opponent takes the armor somewhere else; this client saw nothing and
    # received no pickup event, so the belief must stay at the prior — the
    # source must not consult any global timer.
    for frame in (100, 101, 102):
        inputs = src.sample(telemetry(frame=frame), map_epoch=4)
        assert belief_map(inputs)[9] == 1.0


def test_ambiguous_pickup_resolution_fails_closed(tmp_path):
    records = [
        {
            "class": "health", "classname": "item_health", "confidence": 65535,
            "l1_index": [0, 0, 0], "objective_id": 5, "risk": 0,
            "world_milliunits": [0, 0, 0],
        },
        {
            "class": "ammunition", "classname": "ammo_bullets",
            "confidence": 65535, "l1_index": [1, 0, 0], "objective_id": 6,
            "risk": 0, "world_milliunits": [40000, 0, 0],
        },
    ]
    src = source(tmp_path, records)
    with pytest.raises(QuerySourceError, match="exactly one objective"):
        src.sample(
            telemetry(frame=100, position=(20.0, 0.0, 0.0), pickup=1.0),
            map_epoch=4,
        )


def test_pickup_far_from_any_objective_fails_closed(tmp_path):
    with pytest.raises(QuerySourceError, match="exactly one objective"):
        source(tmp_path).sample(
            telemetry(frame=100, position=(5000.0, 5000.0, 0.0), pickup=1.0),
            map_epoch=4,
        )


def test_survivability_is_safe_without_incoming_damage(tmp_path):
    inputs = source(tmp_path).sample(
        telemetry(frame=10, health=150.0, armor=50.0), map_epoch=4
    )
    margin, ehp_norm, dps_share = inputs.survivability
    assert margin == 1.0
    assert ehp_norm == 1.0
    assert dps_share == pytest.approx(1.0, abs=1e-3)


def test_survivability_projects_losing_exchange(tmp_path):
    src = source(tmp_path)
    enemy = {"health": 100.0, "exposure": 0.9}
    src.sample(
        telemetry(frame=10, health=30.0, damage_taken=20.0, enemy=enemy),
        map_epoch=4,
    )
    inputs = src.sample(
        telemetry(frame=11, health=10.0, damage_taken=20.0, enemy=enemy),
        map_epoch=4,
    )
    margin, ehp_norm, dps_share = inputs.survivability
    assert margin < 0.0
    assert ehp_norm == pytest.approx(10.0 / 200.0)
    assert 0.0 < dps_share < 0.5
    assert all(np.isfinite(inputs.survivability))


def test_out_of_ammo_removes_own_dps(tmp_path):
    inputs = source(tmp_path).sample(
        telemetry(frame=10, ammo=0.0, damage_taken=20.0,
                  enemy={"health": 100.0}),
        map_epoch=4,
    )
    margin, _, dps_share = inputs.survivability
    assert dps_share == pytest.approx(0.0, abs=1e-3)
    assert margin < 0.0


def test_thermal_evidence_uses_stable_identity_and_world_geometry(tmp_path):
    src = source(tmp_path)
    inputs = src.sample(
        telemetry(
            frame=10, position=(0.0, 0.0, 0.0), yaw=0.0,
            enemy={"local": (100.0, 0.0, 0.0), "exposure": 0.75, "edict": 7},
        ),
        map_epoch=4,
    )
    target_id, world_point, heat, observed_tick = inputs.thermal
    assert target_id == 7 << 14
    assert world_point == pytest.approx((100.0, 0.0, 22.0))
    assert heat == pytest.approx(0.75)
    assert observed_tick == 10


def test_thermal_expires_after_five_ticks(tmp_path):
    src = source(tmp_path)
    src.sample(telemetry(frame=10, enemy={"edict": 7}), map_epoch=4)
    for offset in range(1, THERMAL_TTL_TICKS + 1):
        inputs = src.sample(telemetry(frame=10 + offset), map_epoch=4)
        assert inputs.thermal is not None
        assert inputs.thermal[3] == 10
    expired = src.sample(
        telemetry(frame=10 + THERMAL_TTL_TICKS + 1), map_epoch=4
    )
    assert expired.thermal is None


def test_thermal_without_stable_debug_identity_is_not_evidence(tmp_path):
    inputs = source(tmp_path).sample(
        telemetry(frame=10, enemy={"edict": 0}), map_epoch=4
    )
    assert inputs.thermal is None


def test_life_epoch_reset_clears_thermal_and_exchange_but_keeps_beliefs(tmp_path):
    src = source(tmp_path)
    src.sample(
        telemetry(frame=100, position=(0.0, 0.0, 0.0), pickup=1.0,
                  damage_taken=20.0, enemy={"edict": 7}),
        map_epoch=4,
    )
    reborn = src.sample(telemetry(frame=101, life_epoch=4), map_epoch=4)
    assert reborn.thermal is None
    assert reborn.survivability[0] == 1.0  # decayed enemy DPS was cleared
    assert belief_map(reborn)[5] < 1.0  # item timing survives death
    with pytest.raises(QuerySourceError, match="life epoch regressed"):
        src.sample(telemetry(frame=102, life_epoch=3), map_epoch=4)


def test_map_epoch_advance_resets_beliefs_and_frames(tmp_path):
    src = source(tmp_path)
    src.sample(
        telemetry(frame=100, position=(0.0, 0.0, 0.0), pickup=1.0), map_epoch=4
    )
    fresh = src.sample(telemetry(frame=5), map_epoch=5)
    assert belief_map(fresh)[5] == 1.0
    with pytest.raises(QuerySourceError, match="map epoch regressed"):
        src.sample(telemetry(frame=6), map_epoch=4)


def test_stale_or_repeated_server_frame_fails_closed(tmp_path):
    src = source(tmp_path)
    src.sample(telemetry(frame=10), map_epoch=4)
    with pytest.raises(QuerySourceError, match="stale"):
        src.sample(telemetry(frame=10), map_epoch=4)
    with pytest.raises(QuerySourceError, match="stale"):
        src.sample(telemetry(frame=9), map_epoch=4)


def test_mixed_client_or_map_identity_fails_closed(tmp_path):
    src = source(tmp_path)
    with pytest.raises(QuerySourceError, match="differs from bound client"):
        src.sample(telemetry(client_id="client-99"), map_epoch=4)
    with pytest.raises(QuerySourceError, match="differs from bound map"):
        src.sample(telemetry(map_name="q2dm2"), map_epoch=4)


def test_environment_steps_are_monotonic(tmp_path):
    src = source(tmp_path)
    steps = [
        src.sample(telemetry(frame=frame), map_epoch=4).environment_steps
        for frame in (10, 11, 12)
    ]
    assert steps == [1, 2, 3]


def test_dyn_engagement_is_exactly_attributed_or_omitted(tmp_path):
    src = source(tmp_path)
    first = src.sample(
        telemetry(frame=10, enemy={"edict": 7, "local": (100.0, 0.0, 0.0)}),
        map_epoch=4,
    )
    assert first.dyn_events == (
        ((10 << 3) | 3, "opportunity", pytest.approx((400.0, 300.0, 22.0))),
    )
    hit = src.sample(
        telemetry(frame=11, damage_dealt=8.0,
                  enemy={"edict": 7, "local": (100.0, 0.0, 0.0)}),
        map_epoch=4,
    )
    assert hit.dyn_events == (
        ((11 << 3) | 1, "engagement", pytest.approx((400.0, 300.0, 22.0))),
    )

    ambiguous = source(tmp_path)
    ambiguous.sample(telemetry(frame=20, enemy={"edict": 7}), map_epoch=4)
    ambiguous.sample(telemetry(frame=21, enemy={"edict": 8}), map_epoch=4)
    events = ambiguous.sample(
        telemetry(frame=22, damage_dealt=4.0), map_epoch=4
    ).dyn_events
    assert all(kind != "engagement" for _, kind, _ in events)

    missing = source(tmp_path).sample(
        telemetry(frame=30, damage_dealt=4.0), map_epoch=4
    )
    assert missing.dyn_events == ()


def test_dyn_threat_uses_own_position_and_exact_event_identity(tmp_path):
    inputs = source(tmp_path).sample(
        telemetry(frame=17, position=(12.0, 34.0, 56.0), damage_taken=3.0),
        map_epoch=4,
    )
    assert inputs.dyn_events == (
        ((17 << 3) | 2, "threat", (12.0, 34.0, 56.0)),
    )


def test_dyn_opportunity_emits_only_for_new_target_or_l2_cell(tmp_path):
    src = source(tmp_path)
    new = src.sample(
        telemetry(frame=10, position=(0.0, 0.0, 0.0),
                  enemy={"edict": 7, "local": (10.0, 0.0, 0.0)}),
        map_epoch=4,
    )
    assert [kind for _, kind, _ in new.dyn_events] == ["opportunity"]
    unchanged = src.sample(
        telemetry(frame=11, position=(0.0, 0.0, 0.0),
                  enemy={"edict": 7, "local": (20.0, 0.0, 0.0)}),
        map_epoch=4,
    )
    assert unchanged.dyn_events == ()
    moved = src.sample(
        telemetry(frame=12, position=(0.0, 0.0, 0.0),
                  enemy={"edict": 7, "local": (80.0, 0.0, 0.0)}),
        map_epoch=4,
    )
    assert moved.dyn_events[0][:2] == ((12 << 3) | 3, "opportunity")


def test_dyn_opportunity_l2_cells_are_relative_to_negative_nonaligned_origin(
    tmp_path,
):
    # With origin x=-33, world x=30 and x=31 straddle the authoritative L2
    # boundary.  An implicit world-zero floor(x/64) would put both in cell 0
    # and miss the second opportunity edge.
    src = source(tmp_path, atlas_origin=(-33, 17, -129))
    first = src.sample(
        telemetry(frame=10, position=(0.0, 0.0, 0.0),
                  enemy={"edict": 7, "local": (30.0, 0.0, 0.0)}),
        map_epoch=4,
    )
    assert [kind for _, kind, _ in first.dyn_events] == ["opportunity"]
    crossed = src.sample(
        telemetry(frame=11, position=(0.0, 0.0, 0.0),
                  enemy={"edict": 7, "local": (31.0, 0.0, 0.0)}),
        map_epoch=4,
    )
    assert crossed.dyn_events[0][:2] == ((11 << 3) | 3, "opportunity")


def test_dyn_self_fire_is_accepted_echo_rising_edge(tmp_path):
    src = source(tmp_path)
    suppressed = src.sample(
        telemetry(frame=10, fire=True, fire_accepted=False), map_epoch=4
    )
    assert suppressed.dyn_events == ()
    pressed = src.sample(
        telemetry(frame=11, fire=True, fire_accepted=True), map_epoch=4
    )
    assert [event[:2] for event in pressed.dyn_events] == [
        ((11 << 3) | 4, "self_fire")
    ]
    held = src.sample(
        telemetry(frame=12, fire=True, fire_accepted=True), map_epoch=4
    )
    assert held.dyn_events == ()
    src.sample(telemetry(frame=13), map_epoch=4)
    repressed = src.sample(
        telemetry(frame=14, fire=True, fire_accepted=True), map_epoch=4
    )
    assert [event[:2] for event in repressed.dyn_events] == [
        ((14 << 3) | 4, "self_fire")
    ]


def test_dyn_death_is_one_shot_public_event(tmp_path):
    src = source(tmp_path)
    died = src.sample(
        telemetry(frame=10, health=0.0, death=1.0), map_epoch=4
    )
    assert died.dyn_events == (
        ((10 << 3) | 5, "death", (300.0, 300.0, 0.0)),
    )
    repeated = src.sample(
        telemetry(frame=11, health=0.0, death=1.0), map_epoch=4
    )
    assert repeated.dyn_events == ()

    health_edge = source(tmp_path).sample(
        telemetry(frame=20, health=0.0, death=0.0), map_epoch=4
    )
    assert health_edge.dyn_events == (
        ((20 << 3) | 5, "death", (300.0, 300.0, 0.0)),
    )


def test_dyn_derivation_ignores_private_causal_completeness(tmp_path):
    src = source(tmp_path)
    inputs = src.sample(
        telemetry(
            frame=10,
            causal_flags=CausalFlags(0),
            damage_taken=2.0,
            fire=True,
            fire_accepted=True,
        ),
        map_epoch=4,
    )
    assert [event[:2] for event in inputs.dyn_events] == [
        ((10 << 3) | 2, "threat"),
        ((10 << 3) | 4, "self_fire"),
    ]
    assert inputs.expected_environment_steps == 0
    assert inputs.environment_steps == 1


def test_staged_query_rollback_makes_same_frame_retry_identical(tmp_path):
    src = source(tmp_path)
    pending = src.stage(
        telemetry(frame=10, damage_taken=2.0, fire=True, fire_accepted=True),
        map_epoch=4,
        emit_dyn_events=True,
    )
    expected = pending.inputs
    pending.rollback()
    retried = src.sample(
        telemetry(frame=10, damage_taken=2.0, fire=True, fire_accepted=True),
        map_epoch=4,
    )
    assert retried == expected
    assert retried.expected_environment_steps == 0


def test_projection_barrier_advances_cas_but_persists_no_dyn_event(tmp_path):
    src = source(tmp_path)
    barrier = src.stage(
        telemetry(
            frame=10,
            damage_taken=2.0,
            fire=True,
            fire_accepted=True,
            enemy={"edict": 7},
        ),
        map_epoch=4,
        emit_dyn_events=False,
    )
    assert barrier.inputs.dyn_events == ()
    barrier.commit()
    # Held fire and visible opportunity edges are consumed by the barrier but
    # not persisted to Dyn, so neither can replay on the next admitted frame.
    admitted = src.sample(
        telemetry(frame=11, fire=True, fire_accepted=True, enemy={"edict": 7}),
        map_epoch=4,
    )
    assert admitted.dyn_events == ()
    assert admitted.expected_environment_steps == 1
    assert admitted.environment_steps == 2


def test_nonfinite_facts_fail_closed(tmp_path):
    src = source(tmp_path)
    with pytest.raises(QuerySourceError, match="not finite"):
        src.sample(telemetry(frame=10, health=float("nan")), map_epoch=4)


def test_artifact_digest_and_identity_admission(tmp_path):
    path, digest = write_objectives(tmp_path)
    with pytest.raises(QuerySourceError, match="digest differs"):
        OwnObservationQuerySource(
            objectives_path=path, expected_objectives_sha256="0" * 64,
            expected_atlas_sha256=ATLAS, atlas_origin=(0, 0, 0),
            map_name=MAP, client_id=CLIENT,
        )
    with pytest.raises(QuerySourceError, match="Atlas digest fence"):
        OwnObservationQuerySource(
            objectives_path=path, expected_objectives_sha256=digest,
            expected_atlas_sha256="b" * 64, atlas_origin=(0, 0, 0),
            map_name=MAP, client_id=CLIENT,
        )
    with pytest.raises(QuerySourceError, match="map identity"):
        OwnObservationQuerySource(
            objectives_path=path, expected_objectives_sha256=digest,
            expected_atlas_sha256=ATLAS, atlas_origin=(0, 0, 0),
            map_name="q2dm2", client_id=CLIENT,
        )
    with pytest.raises(QuerySourceError, match="origin fence"):
        OwnObservationQuerySource(
            objectives_path=path, expected_objectives_sha256=digest,
            expected_atlas_sha256=ATLAS, atlas_origin=(-1, 0, 0),
            map_name=MAP, client_id=CLIENT,
        )
    with pytest.raises(QuerySourceError, match="missing"):
        OwnObservationQuerySource(
            objectives_path=tmp_path / "absent.objectives.json",
            expected_objectives_sha256=digest,
            expected_atlas_sha256=ATLAS, atlas_origin=(0, 0, 0),
            map_name=MAP, client_id=CLIENT,
        )


def test_same_inputs_produce_canonical_identical_results(tmp_path):
    def run():
        src = source(tmp_path)
        outputs = []
        frames = [
            telemetry(frame=100, position=(0.0, 0.0, 0.0), pickup=1.0,
                      damage_taken=12.0, enemy={"edict": 7, "exposure": 0.6}),
            telemetry(frame=101, damage_dealt=8.0,
                      enemy={"edict": 7, "exposure": 0.4}),
            telemetry(frame=250),
        ]
        for frame in frames:
            outputs.append(src.sample(frame, map_epoch=4))
        return outputs

    first, second = run(), run()
    assert first == second
