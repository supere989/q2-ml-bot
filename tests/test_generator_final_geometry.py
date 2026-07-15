from __future__ import annotations

import pytest

from maps.generator import LAVA_DEPTH, MapGenerator, Room, SolidBox


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

