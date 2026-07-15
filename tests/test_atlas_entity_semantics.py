from __future__ import annotations

from dataclasses import dataclass
import math

import pytest

from harness.atlas_entity_semantics import (
    Authority,
    GeometryClassification,
    L0_BITPLANE_BYTES,
    L0_CHUNK_HEADER_BYTES,
    L0_SCALAR_PLANE_BYTES,
    L0BudgetState,
    L0PlaneKind,
    RotationAxis,
    entity_angles,
    platform_mover_semantics,
    rotating_mover_semantics,
    set_movedir,
    sliding_mover_semantics,
    train_topology,
    trigger_hurt_semantics,
)


@dataclass(frozen=True)
class Entity:
    index: int
    classname: str
    properties: tuple[tuple[str, str], ...]


def entity(index: int, classname: str, *properties: tuple[str, str]) -> Entity:
    return Entity(index, classname, (("classname", classname), *properties))


def exact(result):
    assert result.authority is Authority.EXACT, result.reason
    assert result.value is not None
    return result.value


def test_angle_and_angles_share_one_ordered_destination() -> None:
    properties = (
        ("angle", "90"),
        ("angles", "10 20 30"),
        ("ANGLE", "-1"),
    )
    assert exact(entity_angles(properties)) == (0.0, -1.0, 0.0)
    assert exact(set_movedir(properties)).movedir == (0.0, 0.0, 1.0)

    reversed_properties = (("angle", "-1"), ("angles", "0 90 0"))
    movedir = exact(set_movedir(reversed_properties))
    assert movedir.parsed_angles == (0.0, 90.0, 0.0)
    assert movedir.movedir == pytest.approx((0.0, 1.0, 0.0), abs=1e-12)
    assert movedir.runtime_angles == (0.0, 0.0, 0.0)


def test_movedir_supports_full_pitch_and_rejects_malformed_last_angles() -> None:
    movedir = exact(set_movedir((("angles", "-90 0 0"),)))
    assert movedir.movedir == pytest.approx((0.0, 0.0, 1.0), abs=1e-12)

    result = entity_angles((("angle", "90"), ("angles", "0 90")))
    assert result.authority is Authority.UNKNOWN
    assert result.value is None


def test_sliding_door_uses_abs_component_extent_and_keeps_negative_travel() -> None:
    door = entity(
        1,
        "func_door",
        ("origin", "100 200 300"),
        ("angle", "45"),
        ("lip", "8"),
    )
    semantics = exact(sliding_mover_semantics(door, (0, 0, 0), (64, 32, 16)))
    expected = 64 / math.sqrt(2) + 32 / math.sqrt(2) - 8
    assert semantics.distance == pytest.approx(expected)
    assert semantics.pos1 == (100.0, 200.0, 300.0)
    assert semantics.pos2 == pytest.approx(
        (100 + expected / math.sqrt(2), 200 + expected / math.sqrt(2), 300)
    )
    assert semantics.reference_pose.collision_authority is Authority.UNKNOWN
    assert semantics.potential_envelope.classification is GeometryClassification.POTENTIAL_SWEPT_ENVELOPE

    over_lipped = entity(2, "func_door", ("angle", "0"), ("lip", "100"))
    negative = exact(sliding_mover_semantics(over_lipped, (0, 0, 0), (32, 16, 16)))
    assert negative.distance == -68
    assert negative.pos2 == (-68.0, 0.0, 0.0)


def test_sliding_start_open_swaps_declared_poses_and_zero_lip_uses_default() -> None:
    door = entity(
        1,
        "func_door",
        ("origin", "10 20 30"),
        ("angle", "-1"),
        ("lip", "0"),
        ("spawnflags", "1"),
    )
    semantics = exact(sliding_mover_semantics(door, (-4, -4, -4), (4, 4, 20)))
    assert semantics.lip == 8
    assert semantics.distance == 16
    assert semantics.start_open
    assert semantics.pos1 == (10.0, 20.0, 46.0)
    assert semantics.pos2 == (10.0, 20.0, 30.0)
    assert semantics.current_origin == semantics.pos1

    water = entity(2, "func_water", ("angle", "-2"), ("lip", "0"))
    water_semantics = exact(sliding_mover_semantics(water, (0, 0, 0), (8, 8, 24)))
    assert water_semantics.lip == 0
    assert water_semantics.distance == 24


def test_platform_uses_exact_vertical_endpoint_law_and_initial_pose() -> None:
    platform = entity(
        1, "func_plat", ("origin", "100 200 300"), ("lip", "0"),
    )
    semantics = exact(platform_mover_semantics(platform, (-32, -16, -8), (32, 16, 56)))
    assert semantics.lip == 8
    assert semantics.height is None
    assert semantics.pos1 == (100.0, 200.0, 300.0)
    assert semantics.pos2 == (100.0, 200.0, 244.0)
    assert semantics.current_origin == semantics.pos2
    assert not semantics.target_disabled
    assert semantics.potential_envelope.bounds.mins == (68.0, 184.0, 236.0)
    assert semantics.potential_envelope.bounds.maxs == (132.0, 216.0, 356.0)
    assert semantics.potential_envelope.collision_authority is Authority.UNKNOWN

    disabled = entity(
        2, "func_plat", ("origin", "0 0 64"), ("height", "40.9"),
        ("lip", "2"), ("targetname", "manual"),
    )
    explicit = exact(platform_mover_semantics(disabled, (0, 0, 0), (64, 64, 16)))
    assert explicit.height == 40
    assert explicit.pos2 == (0.0, 0.0, 24.0)
    assert explicit.current_origin == explicit.pos1
    assert explicit.target_disabled


def test_platform_rejects_other_classes_and_malformed_origin() -> None:
    assert platform_mover_semantics(
        entity(1, "func_door"), (0, 0, 0), (8, 8, 8),
    ).authority is Authority.UNKNOWN
    malformed = entity(2, "func_plat", ("origin", "0 0"))
    assert platform_mover_semantics(
        malformed, (0, 0, 0), (8, 8, 8),
    ).authority is Authority.UNKNOWN


def test_train_unique_open_chain_is_exact_and_aligns_model_minimum() -> None:
    train = entity(10, "func_train", ("target", "A"))
    entities = (
        entity(20, "path_corner", ("targetname", "A"), ("target", "B"), ("origin", "100 0 24")),
        entity(21, "path_corner", ("targetname", "B"), ("origin", "200 0 24")),
    )
    topology = exact(train_topology(train, entities, (-16, -8, -4)))
    assert [group.lookup for group in topology.groups] == ["A", "B"]
    first = topology.group("a").eligible[0]
    assert exact(first.train_origin) == (116.0, 8.0, 28.0)
    assert topology.open_chain_entity_indices == (21,)
    assert topology.unresolved_lookups == ()


def test_train_duplicates_preserve_first_eight_and_never_admit_one_route() -> None:
    train = entity(1, "func_train", ("target", "choice"))
    entities = tuple(
        entity(
            index,
            "path_corner",
            ("targetname", "CHOICE"),
            ("origin", f"{index} 0 0"),
        )
        for index in range(10, 19)
    )
    result = train_topology(train, entities, (0, 0, 0))
    assert result.authority is Authority.UNKNOWN
    assert "G_PickTarget" in result.reason
    topology = result.value
    assert topology is not None
    group = topology.group("choice")
    assert group is not None
    assert [candidate.entity_index for candidate in group.eligible] == list(range(10, 18))
    assert group.ignored_matching_entity_indices == (18,)
    assert group.selection_authority is Authority.UNKNOWN


def test_train_teleport_and_unresolved_stop_topology_are_explicit() -> None:
    train = entity(1, "func_train", ("target", "A"))
    entities = (
        entity(10, "path_corner", ("targetname", "A"), ("target", "T1"), ("origin", "0 0 0")),
        entity(
            11, "path_corner", ("targetname", "T1"), ("target", "T2"),
            ("origin", "64 0 0"), ("spawnflags", "1"),
        ),
        entity(
            12, "path_corner", ("targetname", "T2"), ("target", "missing"),
            ("origin", "128 0 0"), ("spawnflags", "1"),
        ),
    )
    topology = exact(train_topology(train, entities, (0, 0, 0)))
    assert topology.consecutive_teleport_pairs == ((11, 12),)
    assert topology.unresolved_lookups == ("missing",)
    assert topology.open_chain_entity_indices == ()


def test_train_requires_edict_order_and_flags_unexpected_target_class() -> None:
    train = entity(1, "func_train", ("target", "A"))
    reversed_entities = (
        entity(3, "path_corner", ("targetname", "A"), ("origin", "0 0 0")),
        entity(2, "path_corner", ("targetname", "B"), ("origin", "0 0 0")),
    )
    assert train_topology(train, reversed_entities, (0, 0, 0)).authority is Authority.UNKNOWN

    noncorner = (entity(2, "info_notnull", ("targetname", "A"), ("origin", "0 0 0")),)
    result = train_topology(train, noncorner, (0, 0, 0))
    assert result.authority is Authority.UNKNOWN
    assert result.value.unexpected_target_entity_indices == (2,)


def test_func_rotating_axis_precedence_reverse_and_claim_boundary() -> None:
    rotating = entity(
        1,
        "func_rotating",
        ("origin", "100 200 300"),
        ("angles", "0 30 0"),
        ("spawnflags", str(4 | 8 | 2)),
    )
    semantics = exact(rotating_mover_semantics(rotating, (-8, -16, -4), (8, 16, 4)))
    assert semantics.axis is RotationAxis.X
    assert semantics.movedir == (0.0, 0.0, -1.0)
    assert semantics.reverse
    assert semantics.current_angles == (0.0, 30.0, 0.0)
    assert semantics.endpoint_pose is None
    assert semantics.reference_pose.transform_authority is Authority.EXACT
    assert semantics.reference_pose.collision_authority is Authority.UNKNOWN
    assert semantics.potential_envelope.classification is GeometryClassification.POTENTIAL_SWEPT_ENVELOPE
    assert semantics.potential_envelope.bounds.mins[0] < 100 < semantics.potential_envelope.bounds.maxs[0]


def test_rotating_door_clears_angles_parses_integer_distance_and_start_open() -> None:
    door = entity(
        1,
        "func_door_rotating",
        ("origin", "10 20 30"),
        ("angles", "20 30 40"),
        ("distance", "45.9"),
        ("spawnflags", str(1 | 2 | 64)),
    )
    semantics = exact(rotating_mover_semantics(door, (-4, -8, -2), (4, 8, 2)))
    assert semantics.axis is RotationAxis.X
    assert semantics.distance_degrees == 45
    assert semantics.start_open
    # X-axis is ROLL. REVERSE makes the unopened endpoint -45, then
    # START_OPEN swaps endpoints and reverses the future movedir.
    assert semantics.current_angles == (0.0, 0.0, -45.0)
    assert semantics.start_angles == (0.0, 0.0, -45.0)
    assert semantics.end_angles == (0.0, 0.0, 0.0)
    assert semantics.movedir == (0.0, 0.0, 1.0)
    assert semantics.endpoint_pose is not None


def test_trigger_hurt_uses_exact_linked_aabb_dilation() -> None:
    trigger = entity(1, "trigger_hurt", ("origin", "100 0 0"), ("angles", "0 90 0"))
    semantics = exact(trigger_hurt_semantics(trigger, (0, 0, 0), (64, 64, 16)))
    assert semantics.linked_touch_bounds.mins == (98.0, -2.0, -2.0)
    assert semantics.linked_touch_bounds.maxs == (166.0, 66.0, 18.0)
    assert semantics.standing_forbidden_origins.mins == (81.0, -19.0, -35.0)
    assert semantics.standing_forbidden_origins.maxs == (183.0, 83.0, 43.0)
    assert semantics.crouched_forbidden_origins.mins == (81.0, -19.0, -7.0)
    assert semantics.crouched_forbidden_origins.maxs == (183.0, 83.0, 43.0)
    assert semantics.initially_active
    assert semantics.runtime_standing_forbidden_origins.authority is Authority.EXACT
    assert semantics.runtime_crouched_forbidden_origins.authority is Authority.EXACT
    assert semantics.active_state_confidence_u16 == 0xFFFF


def test_trigger_hurt_active_state_is_exact_without_toggle_and_unknown_with_it() -> None:
    permanently_off = entity(1, "trigger_hurt", ("spawnflags", "1"))
    off = exact(trigger_hurt_semantics(permanently_off, (0, 0, 0), (8, 8, 8)))
    assert not off.initially_active
    assert off.initial_standing_forbidden_origins is None
    assert off.initial_crouched_forbidden_origins is None
    assert off.runtime_standing_forbidden_origins.authority is Authority.EXACT
    assert off.runtime_standing_forbidden_origins.value is None
    assert off.runtime_crouched_forbidden_origins.value is None

    toggle = entity(2, "trigger_hurt", ("spawnflags", str(1 | 2)))
    dynamic = exact(trigger_hurt_semantics(toggle, (0, 0, 0), (8, 8, 8)))
    assert dynamic.runtime_standing_forbidden_origins.authority is Authority.UNKNOWN
    assert dynamic.runtime_standing_forbidden_origins.value == dynamic.standing_forbidden_origins
    assert dynamic.runtime_crouched_forbidden_origins.authority is Authority.UNKNOWN
    assert dynamic.runtime_crouched_forbidden_origins.value == dynamic.crouched_forbidden_origins
    assert dynamic.active_state_confidence_u16 == 0


def test_l0_accounting_charges_only_first_plane_materialization() -> None:
    state = L0BudgetState(max_chunks=4, max_bytes=100_000)
    first = exact(state.reserve((0, 0, 0), L0PlaneKind.BIT, "solid"))
    assert first.accepted
    assert first.added_chunks == 1
    assert first.added_bytes == L0_CHUNK_HEADER_BYTES + L0_BITPLANE_BYTES

    duplicate = exact(first.state.reserve((0, 0, 0), L0PlaneKind.BIT, "solid"))
    assert duplicate.accepted
    assert duplicate.added_chunks == 0
    assert duplicate.added_bytes == 0

    scalar = exact(duplicate.state.reserve((0, 0, 0), L0PlaneKind.SCALAR, "confidence"))
    assert scalar.added_bytes == L0_SCALAR_PLANE_BYTES
    assert scalar.prospective_bytes == (
        L0_CHUNK_HEADER_BYTES + L0_BITPLANE_BYTES + L0_SCALAR_PLANE_BYTES
    )


def test_l0_accounting_rejects_prospectively_without_mutating_state() -> None:
    state = L0BudgetState(max_chunks=1, max_bytes=L0_CHUNK_HEADER_BYTES + L0_BITPLANE_BYTES)
    accepted = exact(state.reserve((0, 0, 0), L0PlaneKind.BIT, "solid"))
    byte_reject = exact(accepted.state.reserve((0, 0, 0), L0PlaneKind.SCALAR, "confidence"))
    assert not byte_reject.accepted
    assert byte_reject.state is accepted.state
    assert "L0 bytes" in byte_reject.rejection

    chunk_reject = exact(accepted.state.reserve((1, 0, 0), L0PlaneKind.BIT, "solid"))
    assert not chunk_reject.accepted
    assert chunk_reject.state is accepted.state
    assert "chunk count" in chunk_reject.rejection


def test_l0_1200_chunk_boundary_and_canonical_key_order() -> None:
    state = L0BudgetState()
    for ordinal in range(1_200):
        reservation = exact(state.reserve((ordinal, -ordinal, ordinal % 3), L0PlaneKind.BIT, "solid"))
        assert reservation.accepted
        state = reservation.state
    assert len(state.chunks) == 1_200
    assert state.encoded_bytes == 1_200 * (L0_CHUNK_HEADER_BYTES + L0_BITPLANE_BYTES)
    assert [chunk.key for chunk in state.chunks] == sorted(
        (chunk.key for chunk in state.chunks), key=lambda key: (key[2], key[1], key[0])
    )

    rejected = exact(state.reserve((1_201, 0, 0), L0PlaneKind.BIT, "solid"))
    assert not rejected.accepted
    assert rejected.prospective_chunks == 1_201


def test_l0_invalid_reservation_is_unknown() -> None:
    state = L0BudgetState()
    assert state.reserve((0, 0, 0), L0PlaneKind.BIT, "").authority is Authority.UNKNOWN
    assert state.reserve((0, True, 0), L0PlaneKind.BIT, "solid").authority is Authority.UNKNOWN
