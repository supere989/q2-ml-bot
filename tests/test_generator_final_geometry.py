from __future__ import annotations

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
    SolidBox,
    generate_map,
)
from maps.routes import source_endpoint_components
from tools.source_route_contract import load_source_route_contract
from tools.validate_maps import deathmatch_spawn_origins


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
    assert overlapping.ceil_z >= anchor.floor_z + MIN_SAFE_HEADROOM + 8
    assert (nonoverlapping.floor_z, nonoverlapping.ceil_z) == (96, 256)


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


def test_retired_71436301_pits_spawn_regression_is_temp_only_and_deterministic(
    tmp_path: Path,
) -> None:
    """The retired cohort seed is a fixture only, never a retry/publication."""
    map_id = "retired_71436301_spawn_regression"
    primary = tmp_path / "primary"
    primary.mkdir()

    generate_map(map_id, 71436301, primary, grid_n=5, style="pits")
    cold_runs = []
    generate_script = (
        "from pathlib import Path; import sys; "
        "from maps.generator import generate_map; "
        "generate_map(sys.argv[2], int(sys.argv[3]), Path(sys.argv[1]), "
        "grid_n=5, style='pits')"
    )
    for hash_seed in (1, 31337):
        cold = tmp_path / f"cold-{hash_seed}"
        cold.mkdir()
        subprocess.run(
            [
                sys.executable, "-c", generate_script,
                str(cold), map_id, "71436301",
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

    assert route_origins == map_origins
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
