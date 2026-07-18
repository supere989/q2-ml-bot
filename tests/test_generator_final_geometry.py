from __future__ import annotations

import ast
import math
import os
from pathlib import Path
import subprocess
import sys

import pytest

from maps.generator import (
    DM_SPAWN_COUNT,
    LAVA_DEPTH,
    MIN_SAFE_HEADROOM,
    MIN_SPAWN_SEPARATION,
    MapGenerator,
    Room,
    SPAWN_COMPILED_COLUMN_HEIGHT,
    SPAWN_COLUMN_SWEEP,
    SPAWN_LINK_LIFT,
    SolidBox,
    generate_map,
)
from maps.routes import source_endpoint_components
from tools.source_route_contract import load_source_route_contract
from tools.validate_maps import (
    _player_column_is_clear as static_player_column_is_clear,
    deathmatch_spawn_origins,
)


@pytest.mark.parametrize(
    ("style", "seed"),
    (
        ("open", 71435001),
        ("canyon", 71435200),
        ("canyon", 71435202),
        ("canyon", 71435203),
        ("pits", 71435300),
        ("pits", 71435301),
        ("arena_open", 71435403),
        ("arena_lanes", 71435600),
        ("arena_lanes", 71435603),
    ),
)
def test_failed_71435_spawns_share_one_source_standing_component(
    style: str, seed: int,
) -> None:
    generator = MapGenerator(seed=seed, style=style)
    generator.generate(5)

    assert len(generator.spawn_points) == DM_SPAWN_COUNT
    components = source_endpoint_components(
        generator.rooms,
        generator.spawn_points,
        generator.spawn_blockers,
        generator.lava_pools,
    )
    assert len(components) == DM_SPAWN_COUNT
    assert len(set(components.values())) == 1
    assert generator._spawn_span_ok(generator.spawn_points)
    assert min(
        math.hypot(left[0] - right[0], left[1] - right[1])
        for index, left in enumerate(generator.spawn_points)
        for right in generator.spawn_points[index + 1:]
    ) >= MIN_SPAWN_SEPARATION


def test_spawn_placement_fails_when_no_component_can_span_the_map() -> None:
    generator = MapGenerator(seed=17, style="open")
    generator.rooms = [
        Room(0, 0, 0, 0, 1024, 1024, 0, 384, "room"),
        Room(4, 0, 2048, 0, 1024, 1024, 0, 384, "room"),
    ]

    with pytest.raises(
        RuntimeError, match="one source standing component"
    ) as failure:
        generator._place_combat_spawns()

    diagnostic = str(failure.value)
    assert "legal_candidates=" in diagnostic
    assert "component_bounds=[" in diagnostic
    assert "span=" in diagnostic


def test_zero_arena_promotion_is_deterministic_and_rng_neutral() -> None:
    generator = MapGenerator(seed=17, style="pits")
    generator.rooms = [
        Room(4, 2, 2048, 1024, 1024, 512, 0, 320, "room"),
        Room(3, 3, 1536, 1536, 1024, 1024, 0, 379, "room"),
        Room(3, 1, 1536, 512, 512, 1024, 96, 267, "corridor"),
    ]
    rng_state = generator.rng.getstate()

    generator._ensure_spawn_arena(5)

    assert generator.rng.getstate() == rng_state
    promoted = generator.rooms[1]
    assert (
        promoted.kind,
        promoted.w,
        promoted.d,
        promoted.floor_z,
        promoted.ceil_z,
    ) == ("arena", 1536, 1536, 0, 384)
    assert generator.rooms[0].kind == "room"
    assert generator.rooms[2].kind == "corridor"


def test_failed_71813_pits_member_gets_protected_spanning_spawn_component() -> None:
    generator = MapGenerator(seed=71_813_302, style="pits")
    generator.generate(5)

    anchor = generator._spawn_protected_anchor
    assert anchor is not None
    assert (anchor.gx, anchor.gy, anchor.kind, anchor.w, anchor.d) == (
        3, 3, "arena", 1536, 1536,
    )
    assert generator.spawn_points == generator._spawn_protected_witnesses
    assert len(generator.spawn_points) == DM_SPAWN_COUNT
    assert generator._spawn_span_ok(generator.spawn_points)
    assert generator._shared_spawn_source_component(
        generator.spawn_points
    ) is not None
    assert min(
        math.hypot(left[0] - right[0], left[1] - right[1])
        for index, left in enumerate(generator.spawn_points)
        for right in generator.spawn_points[index + 1:]
    ) >= MIN_SPAWN_SEPARATION


def test_spawn_arena_floor_normalization_preserves_height_and_nonoverlap() -> None:
    generator = MapGenerator(seed=23, style="pits")
    anchor = Room(1, 1, 512, 512, 1536, 1536, 0, 384, "arena")
    overlapping = Room(2, 1, 1024, 512, 1024, 1024, 96, 256, "room")
    nonoverlapping = Room(4, 1, 2560, 512, 1024, 1024, 96, 256, "room")
    generator.rooms = [anchor, overlapping, nonoverlapping]
    authored_height = overlapping.ceil_z - overlapping.floor_z

    generator._normalize_spawn_arena_floor()

    assert overlapping.floor_z == anchor.floor_z
    assert overlapping.ceil_z - overlapping.floor_z == authored_height
    assert (
        overlapping.ceil_z
        >= anchor.floor_z + SPAWN_COMPILED_COLUMN_HEIGHT + 8
    )
    assert (nonoverlapping.floor_z, nonoverlapping.ceil_z) == (96, 256)
    assert len(generator._spawn_protected_domains) == 4
    assert all(
        domain.z0 == anchor.floor_z + 1
        and domain.z1 == anchor.floor_z + SPAWN_COMPILED_COLUMN_HEIGHT
        for domain in generator._spawn_protected_domains
    )
    assert len(generator._spawn_protected_witnesses) == DM_SPAWN_COUNT
    assert generator._spawn_span_ok(generator._spawn_protected_witnesses)
    assert min(
        math.hypot(left[0] - right[0], left[1] - right[1])
        for index, left in enumerate(generator._spawn_protected_witnesses)
        for right in generator._spawn_protected_witnesses[index + 1:]
    ) >= MIN_SPAWN_SEPARATION
    generator._assert_spawn_protected_capacity()


@pytest.mark.parametrize(
    ("overhead_height", "expected_clear"),
    ((104, False), (105, False), (106, True)),
)
def test_source_spawn_column_uses_conservative_106_unit_boundary(
    overhead_height: int, expected_clear: bool,
) -> None:
    """Exercise the source policy mirror, not compiled-world authority."""
    assert SPAWN_LINK_LIFT == 9
    assert SPAWN_COLUMN_SWEEP == MIN_SAFE_HEADROOM - 56 == 40
    assert SPAWN_COMPILED_COLUMN_HEIGHT == 106

    blocker = SolidBox(-64, -64, overhead_height, 64, 64, 128)
    generator = MapGenerator(seed=1, style="open")
    generator.spawn_blockers = [blocker]
    parsed_blocker = [
        (blocker.x0, blocker.y0, blocker.z0,
         blocker.x1, blocker.y1, blocker.z1)
    ]

    assert generator._player_column_is_clear(0, 0, 0) is expected_clear
    assert static_player_column_is_clear(
        0, 0, 0, parsed_blocker
    ) is expected_clear


@pytest.mark.parametrize(
    ("style", "seed", "formerly_blocked_origins"),
    (
        ("towers", 71442103, {(1120, 608, 120)}),
        (
            "arena_vertical", 71442500,
            {(2464, 2464, 216), (1120, 2464, 216)},
        ),
        ("arena_vertical", 71442502, {(2464, 1760, 120)}),
    ),
)
def test_retired_71442_compiled_column_failures_are_source_clear(
    style: str,
    seed: int,
    formerly_blocked_origins: set[tuple[int, int, int]],
) -> None:
    """Retired members are regression fixtures only, never publication rows."""
    generator = MapGenerator(seed=seed, style=style)
    generator.generate(5)

    assert formerly_blocked_origins <= set(generator.spawn_points)
    for x, y, z in generator.spawn_points:
        floor_z = z - 24
        assert generator._player_column_is_clear(x, y, floor_z)
        overlapping_overheads = [
            blocker.z0
            for blocker in generator.spawn_blockers
            if blocker.x1 > x - 16 and blocker.x0 < x + 16
            and blocker.y1 > y - 16 and blocker.y0 < y + 16
            and blocker.z0 >= floor_z + 1
        ]
        assert not overlapping_overheads or min(overlapping_overheads) >= (
            floor_z + SPAWN_COMPILED_COLUMN_HEIGHT
        )


def test_spawn_arena_floor_normalization_is_noop_without_an_arena() -> None:
    generator = MapGenerator(seed=29, style="pits")
    generator.rooms = [
        Room(0, 0, 0, 0, 1024, 1024, 0, 384, "room"),
        Room(1, 0, 1024, 0, 1024, 1024, 96, 320, "corridor"),
    ]
    before = [
        (room.floor_z, room.ceil_z, room.kind) for room in generator.rooms
    ]

    generator._normalize_spawn_arena_floor()

    assert [
        (room.floor_z, room.ceil_z, room.kind) for room in generator.rooms
    ] == before
    assert generator._spawn_protected_domains == []


@pytest.mark.parametrize(
    ("retired_seed", "style"),
    (
        (71436301, "pits"),
        (71437202, "canyon"),
        (71437203, "canyon"),
    ),
)
def test_retired_spawn_regressions_are_temp_only_and_fresh_process_deterministic(
    retired_seed: int,
    style: str,
    tmp_path: Path,
) -> None:
    """The retired cohort seed is a fixture only, never a retry/publication."""
    map_id = f"retired_{retired_seed}_spawn_regression"
    primary = tmp_path / "primary"
    primary.mkdir()

    generate_map(map_id, retired_seed, primary, grid_n=5, style=style)
    cold_runs = []
    generate_script = (
        "from pathlib import Path; import sys; "
        "from maps.generator import generate_map; "
        "generate_map(sys.argv[2], int(sys.argv[3]), Path(sys.argv[1]), "
        "grid_n=5, style=sys.argv[4])"
    )
    for hash_seed in (1, 31337):
        cold = tmp_path / f"cold-{hash_seed}"
        cold.mkdir()
        subprocess.run(
            [
                sys.executable, "-c", generate_script,
                str(cold), map_id, str(retired_seed), style,
            ],
            cwd=Path(__file__).resolve().parents[1],
            env={**os.environ, "PYTHONHASHSEED": str(hash_seed)},
            check=True,
            capture_output=True,
            text=True,
        )
        cold_runs.append(cold)

    route_contract = load_source_route_contract(
        primary / f"{map_id}.routes.json", map_id
    )
    map_origins = deathmatch_spawn_origins(primary / f"{map_id}.map")
    route_origins = tuple(
        tuple(origin) for origin in route_contract["spawn_origins"]
    )
    certificate = MapGenerator(seed=retired_seed, style=style)
    certificate.generate(5)

    assert route_origins == map_origins
    assert map_origins == tuple(sorted(certificate._spawn_protected_witnesses))
    assert route_contract["spawn_count"] == DM_SPAWN_COUNT
    assert route_contract["all_spawn_origins_unique"] is True
    assert route_contract[
        "all_spawns_share_source_standing_component"
    ] is True
    assert max(origin[0] for origin in map_origins) - min(
        origin[0] for origin in map_origins
    ) >= 1024
    assert max(origin[1] for origin in map_origins) - min(
        origin[1] for origin in map_origins
    ) >= 1024
    assert min(
        math.hypot(left[0] - right[0], left[1] - right[1])
        for index, left in enumerate(map_origins)
        for right in map_origins[index + 1:]
    ) >= MIN_SPAWN_SEPARATION
    for suffix in (
        ".map", ".json", ".meta.json", ".lattice.json", ".routes.json",
    ):
        primary_bytes = (primary / f"{map_id}{suffix}").read_bytes()
        for cold in cold_runs:
            assert primary_bytes == (cold / f"{map_id}{suffix}").read_bytes()


@pytest.mark.parametrize(
    ("collection_name", "message"),
    (
        ("spawn_blockers", "spawn blocker 0"),
        ("lava_pools", "lava pool 0"),
    ),
)
def test_spawn_protected_domain_violation_fails_closed(
    collection_name: str, message: str,
) -> None:
    generator = MapGenerator(seed=31, style="canyon")
    generator.rooms = [
        Room(1, 1, 512, 512, 1536, 1536, 0, 384, "arena"),
    ]
    generator._normalize_spawn_arena_floor()
    protected = generator._spawn_protected_domains[0]
    intruder = SolidBox(
        protected.x0, protected.y0, protected.z0,
        protected.x0 + 32, protected.y0 + 32, protected.z0 + 32,
    )
    getattr(generator, collection_name).append(intruder)

    with pytest.raises(RuntimeError, match=message):
        generator._place_combat_spawns()


def test_authored_neighbor_hallway_suppresses_pair_on_late_member_spill() -> None:
    generator = MapGenerator(seed=37, style="canyon")
    generator.rooms = [
        Room(1, 1, 512, 512, 1536, 1536, 0, 384, "arena"),
        # The first optional side wall is clear; the second, later member spills
        # from this west-neighbor room into the anchor's south ring.
        Room(0, 1, 0, 256, 1024, 512, 0, 192, "corridor"),
    ]
    generator._normalize_spawn_arena_floor()
    first_wall = SolidBox(32, 360, 0, 992, 384, 192)
    second_wall = SolidBox(32, 640, 0, 992, 664, 192)
    assert generator._admit_spawn_blocker(first_wall, register=False)
    assert not generator._admit_spawn_blocker(second_wall, register=False)

    generator._emit_hallways()
    # hallway_count is authored low-ceiling corridor rooms, not optional wall
    # assemblies; the all-or-none pair emits/registers no standing solid.
    assert generator.hallway_count == 1
    assert generator.spawn_blockers == []
    generator._assert_spawn_protected_domain_clear()


def test_protected_lane_four_segments_emit_inside_spawn_ring() -> None:
    generator = MapGenerator(seed=37, style="canyon")
    generator.rooms = [
        Room(1, 1, 512, 512, 1536, 1536, 0, 384, "arena"),
    ]
    generator._normalize_spawn_arena_floor()
    generator.lane_prob = 1.0

    generator._emit_lane_walls()

    assert generator.lane_wall_count == 2
    assert len(generator.spawn_blockers) == 4
    assert generator.writer.solid_boxes == generator.spawn_blockers
    assert all(
        not generator._overlaps_spawn_protected_domain(segment)
        for segment in generator.spawn_blockers
    )
    generator._assert_spawn_protected_domain_clear()


def test_lane_four_segment_assembly_rejects_late_member_atomically(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generator = MapGenerator(seed=39, style="arena_lanes")
    generator.rooms = [
        Room(1, 1, 512, 512, 1536, 1536, 0, 384, "arena"),
    ]
    # Force horizontal walls, then make only the fourth segment enter the
    # protected volume.  Admission must retain neither wall nor any segment.
    monkeypatch.setattr(generator.rng, "random", lambda: 0.0)
    generator._spawn_protected_domains = [
        SolidBox(1800, 1650, 1, 1900, 1680, MIN_SAFE_HEADROOM),
    ]

    generator._emit_lane_walls()

    assert generator.lane_wall_count == 0
    assert generator.spawn_blockers == []
    assert generator.writer.solid_boxes == []


@pytest.mark.parametrize("seed", (71425600, 71425603))
def test_arena_lanes_retains_defining_walls_and_certified_spawns(
    seed: int,
) -> None:
    generator = MapGenerator(seed=seed, style="arena_lanes")
    generator.generate(5)

    assert generator.lane_wall_count >= 2
    assert generator.spawn_points == generator._spawn_protected_witnesses
    assert all(
        not generator._overlaps_spawn_protected_domain(blocker)
        for blocker in generator.spawn_blockers
    )
    components = source_endpoint_components(
        generator.rooms,
        generator.spawn_points,
        generator.spawn_blockers,
        generator.lava_pools,
    )
    assert len(components) == DM_SPAWN_COUNT
    assert len(set(components.values())) == 1
    generator._assert_spawn_protected_capacity()


def test_protected_building_shell_is_rejected_before_any_member_emits() -> None:
    generator = MapGenerator(seed=47, style="arena_open")
    generator.rooms = [
        Room(1, 1, 512, 512, 1536, 1536, 0, 512, "arena"),
    ]
    generator._normalize_spawn_arena_floor()

    generator._emit_large_buildings()

    assert generator.large_building_count == 0
    assert generator.spawn_blockers == []


def test_corner_pair_preflights_both_members_before_registration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generator = MapGenerator(seed=53, style="arena_open")
    generator.rooms = [
        Room(1, 1, 512, 512, 1536, 1536, 0, 512, "arena"),
    ]
    generator._normalize_spawn_arena_floor()
    decisions = iter((True, False) * 4)
    monkeypatch.setattr(
        generator, "_structure_is_clear", lambda _box: next(decisions)
    )

    generator._emit_corner_pockets()

    assert generator.corner_count == 0
    assert generator.spawn_blockers == []


def test_protected_stair_run_is_all_or_none_with_truthful_count() -> None:
    def staircase_generator(protected: bool) -> MapGenerator:
        generator = MapGenerator(seed=41, style="pits")
        anchor = Room(1, 1, 512, 512, 1536, 1536, 0, 384, "arena")
        generator.rooms = [anchor]
        if protected:
            generator._normalize_spawn_arena_floor()
        generator.rooms.append(
            Room(0, 1, 0, 512, 512, 1536, 96, 320, "room")
        )
        generator._adjacent = [(0, 1)]
        return generator

    protected = staircase_generator(True)
    protected._emit_stairs()
    assert protected.stair_count == 0
    assert protected.spawn_blockers == []

    unprotected = staircase_generator(False)
    unprotected._emit_stairs()
    assert unprotected.stair_count == 1
    assert len(unprotected.spawn_blockers) == 6


def test_spawn_blocker_append_is_centralized_in_admission_api() -> None:
    source = Path(__file__).resolve().parents[1] / "maps" / "generator.py"
    tree = ast.parse(source.read_text())
    append_owners = []
    pop_owners = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for child in ast.walk(node):
            if not isinstance(child, ast.Call) or not isinstance(
                child.func, ast.Attribute
            ):
                continue
            target = child.func.value
            if (
                child.func.attr == "append"
                and isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == "self"
                and target.attr == "spawn_blockers"
            ):
                append_owners.append(node.name)
            if (
                child.func.attr == "pop"
                and isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == "self"
                and target.attr == "spawn_blockers"
            ):
                pop_owners.append(node.name)

    assert append_owners == ["_admit_spawn_blocker"]
    assert pop_owners == ["_rollback_spawn_blocker"]


@pytest.mark.parametrize("seed", (71437202, 71437203))
def test_final_spawn_selector_consumes_protected_capacity_witnesses(
    seed: int,
) -> None:
    generator = MapGenerator(seed=seed, style="canyon")
    generator.generate(5)

    assert generator.spawn_points == generator._spawn_protected_witnesses
    generator._assert_spawn_protected_capacity()


def test_objective_temporary_blocker_probe_restores_identical_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generator = MapGenerator(seed=43, style="canyon")
    generator.rooms = [
        Room(1, 1, 512, 512, 1536, 1536, 0, 512, "arena"),
    ]
    generator._normalize_spawn_arena_floor()
    sentinels = (
        SolidBox(-512, -512, 0, -448, -448, 64),
        SolidBox(-384, -384, 0, -320, -320, 64),
    )
    generator.spawn_blockers.extend(sentinels)
    before = tuple(generator.spawn_blockers)
    monkeypatch.setattr(generator, "_floor_item_spot", lambda *_a, **_k: None)

    generator._place_objectives()

    assert tuple(generator.spawn_blockers) == before
    assert all(
        actual is expected
        for actual, expected in zip(generator.spawn_blockers, sentinels)
    )
    assert generator.objectives == []


@pytest.mark.parametrize(
    ("seed", "style"),
    (
        (71437202, "canyon"),
        (71436301, "pits"),
    ),
)
def test_emitted_standing_solids_have_registered_blocker_provenance(
    seed: int, style: str,
) -> None:
    generator = MapGenerator(seed=seed, style=style)
    generator.generate(5)

    def contains(outer: SolidBox, inner: SolidBox) -> bool:
        return (
            outer.x0 <= inner.x0 and outer.y0 <= inner.y0
            and outer.z0 <= inner.z0 and outer.x1 >= inner.x1
            and outer.y1 >= inner.y1 and outer.z1 >= inner.z1
        )

    uncovered = [
        solid for solid in generator.writer.solid_boxes
        if not any(
            contains(blocker, solid) for blocker in generator.spawn_blockers
        )
    ]
    # Six skybox shells and the kill-plane catch ground are intentionally not
    # standing blockers. The trigger_hurt brush is an entity brush and is not
    # recorded in MapWriter.solid_boxes; ceiling lights are contained by their
    # registered ceiling blocker.
    assert uncovered == generator.writer.solid_boxes[:7]
    assert len(uncovered) == 7
    assert all(
        any(contains(blocker, lava) for blocker in generator.spawn_blockers)
        for lava in generator.lava_pools
    )


@pytest.mark.parametrize(
    ("style", "seed"),
    (
        ("arena_lanes", 71428600),
        ("arena_vertical", 71428502),
        ("arena_vertical", 71428503),
        ("pits", 71428300),
        ("arena_open", 71428400),
    ),
)
def test_failed_71428_endpoint_seeds_are_final_standing_clear(
    style: str, seed: int,
) -> None:
    generator = MapGenerator(seed=seed, style=style)
    generator.generate(5)

    endpoints = [
        *((item, x, y, z) for item, x, y, z in generator._placed_loot),
        *((item, x, y, z) for item, x, y, z in generator._heat_placed),
        *((
            objective["item"], objective["x"], objective["y"],
            objective["z"],
        ) for objective in generator.objectives),
    ]
    assert endpoints
    assert all(
        generator._origin_has_final_standing_floor((x, y, z))
        for _item, x, y, z in endpoints
    )
    assert all(
        generator._origin_has_final_standing_floor(origin)
        for origin in generator.spawn_points
    )


def test_lava_prior_uses_exact_emitted_liquid_not_padded_clearance() -> None:
    generator = MapGenerator(seed=71428300, style="pits")
    generator.generate(5)

    assert generator.lava_pools
    for hazard in generator.lava_pools:
        assert hazard.z1 - hazard.z0 == LAVA_DEPTH
        assert any(
            blocker.x0 == hazard.x0 - 16
            and blocker.y0 == hazard.y0 - 16
            and blocker.x1 == hazard.x1 + 16
            and blocker.y1 == hazard.y1 + 16
            and blocker.z0 == hazard.z0 - 12
            and blocker.z1 == hazard.z0 + 52
            for blocker in generator.spawn_blockers
        )


def test_lava_candidate_rejects_existing_static_structure() -> None:
    generator = MapGenerator(seed=91, style="pits")
    spawn_room = Room(0, 0, 0, 0, 1536, 1536, 0, 384, "arena")
    hazard_room = Room(4, 0, 2048, 0, 1024, 1024, 0, 384, "room")
    generator.rooms = [spawn_room, hazard_room]
    generator.lava_prob = 1.0
    generator.spawn_blockers.append(SolidBox(
        hazard_room.wx, hazard_room.wy, hazard_room.floor_z,
        hazard_room.wx + hazard_room.w,
        hazard_room.wy + hazard_room.d,
        hazard_room.floor_z + 128,
    ))

    generator._emit_lava_pools()

    assert generator.lava_pools == []


def test_lava_candidate_without_reward_rolls_back_optional_geometry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generator = MapGenerator(seed=92, style="pits")
    spawn_room = Room(0, 0, 0, 0, 1536, 1536, 0, 384, "arena")
    hazard_room = Room(4, 0, 2048, 0, 1024, 1024, 0, 384, "room")
    generator.rooms = [spawn_room, hazard_room]
    generator.lava_prob = 1.0
    monkeypatch.setattr(generator, "_floor_item_spot", lambda *_args, **_kwargs: None)

    generator._emit_lava_pools()

    assert generator.spawn_blockers == []
    assert generator.lava_pools == []
    assert generator._placed_loot == []
